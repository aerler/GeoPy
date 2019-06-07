'''
Created on 2013-09-28

This module contains common meta data and access functions for WRF model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import collections as col
import os
try: import pickle as pickle
except: import pickle
import osr # from GDAL
# from atmdyn.properties import variablePlotatts
from geodata.base import concatDatasets
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition, GDALError, loadPickledGridDef, pickleGridDef
from geodata.misc import DatasetError, AxisError, DateError, ArgumentError, isNumber, isInt, EmptyDatasetError
from datasets.common import grid_folder, selectElements, stn_params, shp_params, nullNaN, getRootFolder
#from projects.WRF_experiments import Exp, exps, ensembles 
from warnings import warn
from collections import OrderedDict
from utils.constants import precip_thresholds

dataset_name = 'WRF' # dataset name
root_folder = getRootFolder(dataset_name=dataset_name)

## class that defines experiments
class Exp(object):
  ''' class of objects that contain meta data for WRF experiments '''
  # experiment parameter definition (class property)
  parameters = OrderedDict() # order matters, because parameters can depend on one another for defaults
  parameters['name'] = dict(type=str,req=True) # name
  parameters['shortname'] = dict(type=str,req=False) # short name
  parameters['title'] = dict(type=str,req=False) # title used in plots
  parameters['grid'] = dict(type=str,req=True) # name of the grid layout
  parameters['domains'] = dict(type=int,req=True) # number of domains
  parameters['parent'] = dict(type=str,req=True) # driving dataset
  parameters['project'] = dict(type=str,req=True) # project name dataset
  parameters['ensemble'] = dict(type=str,req=False) # ensemble this run is a member of
  parameters['begindate'] = dict(type=str,req=True) # simulation start date
  parameters['beginyear'] = dict(type=int,req=True) # simulation start year
  parameters['enddate'] = dict(type=str,req=False) # simulation end date (if it already finished)
  parameters['endyear'] = dict(type=int,req=False) # simulation end year
  parameters['avgfolder'] = dict(type=str,req=True) # folder for monthly averages
  parameters['outfolder'] = dict(type=str,req=False) # folder for direct WRF averages
  # default values (functions)
  defaults = dict()
  defaults['shortname'] = lambda atts: atts['name']
  defaults['title'] = lambda atts: atts['name'] # need lambda, because parameters are not set yet
  defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/{2:s}/'.format(avgfolder,atts['project'],atts['name'])
  defaults['beginyear'] = lambda atts: int(atts['begindate'].split('-')[0])  if atts['begindate'] else None # first field
  defaults['endyear'] = lambda atts: int(atts['enddate'].split('-')[0]) if atts['enddate'] else None # first field
  
  def __init__(self, **kwargs):
    ''' initialize values from arguments '''
    # loop over simulation parameters
    for argname,argatt in list(self.parameters.items()):
      if argname in kwargs:
        # assign argument based on keyword
        arg = kwargs[argname]
        if not isinstance(arg,argatt['type']): 
          raise TypeError("Argument '{0:s}' must be of type '{1:s}'.".format(argname,argatt['type'].__name__))
        self.__dict__[argname] = arg
      elif argname in self.defaults:
        # assign some default values, if necessary
        if callable(self.defaults[argname]): 
          self.__dict__[argname] = self.defaults[argname](self.__dict__)
        else: self.__dict__[argname] = self.defaults[argname]
      elif argatt['req']:
        # if the argument is required and there is no default, raise error
        raise ValueError("Argument '{0:s}' for experiment '{1:s}' required.".format(argname,self.name))
      else:
        # if the argument is not required, just assign None 
        self.__dict__[argname] = None    

def mask_array(data, var=None, slc=None):
  ''' mask the value missing_value found in var.atts in the data array '''
  if not var.masked and 'missing_value' in var.atts:
    missing_value = var.atts['missing_value']
    data = ma.masked_equal(data, missing_value, copy=False)
  # return array, masked or not...
  return data
  
## variable attributes and name
# convert water mass mixing ratio to water vapor partial pressure ( kg/kg -> Pa ) 
Q = 96000.*28./18. # surface pressure * molecular weight ratio ( air / water )
# a generic class for WRF file types
class FileType(object): 
  ''' A generic class that describes WRF file types. '''
  def __init__(self, *args, **kwargs):
    ''' generate generic attributes using a name argument '''
    if len(args) == 1: 
      name = args[0]
      self.name = name
      self.atts = dict() # should be properly formatted already
      #self.atts = dict(netrad = dict(name='netrad', units='W/m^2'))
      self.vars = [] 
      self.ignore_list = []
      self.climfile = 'wrf{:s}_d{{0:0=2d}}{{1:s}}_clim{{2:s}}.nc'.format(name) # generic climatology name 
      # final filename needs to be extended by (domain,'_'+grid,'_'+period)
      self.tsfile = 'wrf{:s}_d{{0:0=2d}}{{1:s}}_monthly.nc'.format(name) # generic time-series name
      # final filename needs to be extended by (domain, grid)
    else: raise ArgumentError  
# constants
class Const(FileType):
  ''' Variables and attributes of the constants files. '''
  def __init__(self):
    self.name = 'const' 
    self.atts = dict(HGT      = dict(name='zs', units='m'), # surface elevation
                     LU_INDEX = dict(name='landuse', units='', dtype=np.int8), # land-use type
                     LU_MASK  = dict(name='landmask', units='', dtype=np.int8), # land mask
                     SOILCAT  = dict(name='soilcat', units='', dtype=np.int8), # soil category
                     VEGCAT   = dict(name='vegcat', units='', dtype=np.int8), # vegetation category
                     SLOPECAT = dict(name='slopecat', units='', dtype=np.int8), # slope category (not used...?)
                     XLONG    = dict(name='lon2D', units='deg E'), # geographic longitude field
                     XLAT     = dict(name='lat2D', units='deg N'), # geographic latitude field
                     SINALPHA = dict(name='sina', units=''), # sine of map rotation
                     COSALPHA = dict(name='cosa', units='')) # cosine of map rotation                   
    self.vars = list(self.atts.keys())    
    self.ignore_list = ['CON']
    self.climfile = 'wrfconst_d{0:0=2d}{1:s}.nc' # the filename needs to be extended by (domain,'_'+grid)
    self.tsfile = 'wrfconst_d{0:0=2d}{1:s}.nc' # the filename needs to be extended by (domain,grid)
# surface variables
class Srfc(FileType):
  ''' Variables and attributes of the surface files. '''
  def __init__(self):
    self.name = 'srfc'
    self.atts = dict(T2           = dict(name='T2', units='K'), # 2m Temperature
                     TSK          = dict(name='Ts', units='K'), # Skin Temperature (SST)
                     SummerDays   = dict(name='sumfrq', units='', atts=dict(long_name='Fraction of Summer Days (>25C)')),
                     FrostDays    = dict(name='frzfrq', units='', atts=dict(long_name='Fraction of Frost Days (< 0C)')),
                     Q2           = dict(name='q2', units='kg/kg'), # 2m water vapor mass mixing ratio
                     RAIN         = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
                     RAINC        = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate (kg/m^2/s)
                     RAINNC       = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate (kg/m^2/s)
                     SNOW         = dict(name='snow', units='kg/m^2'), # snow water equivalent
                     SNOWH        = dict(name='snowh', units='m'), # snow depth
                     PSFC         = dict(name='ps', units='Pa'), # surface pressure
                     HFX          = dict(name='hfx', units='W/m^2'), # surface sensible heat flux
                     LH           = dict(name='lhfx', units='W/m^2'), # surface latent heat flux
                     QFX          = dict(name='evap', units='kg/m^2/s'), # surface evaporation
#                      OLR          = dict(name='OLR', units='W/m^2'), # Outgoing Longwave Radiation
#                      GLW          = dict(name='GLW', units='W/m^2'), # Downwelling Longwave Radiation at Surface
#                      SWDOWN       = dict(name='SWD', units='W/m^2'), # Downwelling Shortwave Radiation at Surface
                     OLR          = dict(name='LWUPT', units='W/m^2'), # Outgoing Longwave Radiation
                     GLW          = dict(name='LWDNB', units='W/m^2'), # Downwelling Longwave Radiation at Surface
                     SWD          = dict(name='SWDNB', units='W/m^2'), # Downwelling Shortwave Radiation at Surface
                     SWNORM       = dict(name='SWN', units='W/m^2'), # Downwelling Normal Shortwave Radiation at Surface
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip_SR = dict(name='liqprec_sr', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip_SR  = dict(name='solprec_sr', units='kg/m^2/s'), # solid precipitation rate
                     WaterVapor   = dict(name='Q2', units='Pa'), # water vapor partial pressure
                     U10          = dict(name='u10', units='m/s'), # Westerly Wind (at 10m)
                     V10          = dict(name='v10', units='m/s'), # Southerly Wind (at 10m)
                     MaxACSNOW    = dict(name='MaxSolprec_6h', units='kg/m^2/s'), # maximum 6-hourly solid precip
                     MaxACSNOW_1d = dict(name='MaxSolprec_1d', units='kg/m^2/s'), # maximum daily solid precip                             
                     MaxACSNOW_5d = dict(name='MaxSolprec_5d', units='kg/m^2/s'), # maximum pendat (5 day) solid precip
                     MaxRAIN      = dict(name='MaxPrecip_6h', units='kg/m^2/s'), # maximum 6-hourly precip                    
                     MaxRAIN_1d   = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAIN_5d   = dict(name='MaxPrecip_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
                     MaxRAINC     = dict(name='MaxPreccu_6h', units='kg/m^2/s'), # maximum 6-hourly convective precip
                     MaxRAINC_1d  = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxRAINC_5d  = dict(name='MaxPreccu_5d', units='kg/m^2/s'), # maximum pendat (5 day) conv. precip
                     MaxRAINNC    = dict(name='MaxPrecnc_6h', units='kg/m^2/s'), # maximum 6-hourly non-convective precip
                     MaxRAINNC_1d = dict(name='MaxPrecnc_1d', units='kg/m^2/s'), # maximum daily non-convective precip
                     MaxRAINNC_5d = dict(name='MaxPrecnc_5d', units='kg/m^2/s'), # maximum pendat (5 day) n-c precip
                     OrographicIndex = dict(name='OI', units='', atts=dict(long_name='Orographic Index')), # projection of wind onto slope
                     CovOIP       = dict(name='COIP', units='', atts=dict(long_name='Cov(OI,p)')), # covariance of OI and precip
                     # lake variables (some need to be masked explicitly)
                     SSTSK        = dict(name='SSTs', units='K', transform=mask_array, # Sea Surface Skin Temperature (WRF)
                                         atts=dict(missing_value=0, long_name='Sea Surface Skin Temperature')), 
                     SEAICE       = dict(name='seaice', units='', atts=dict(long_name='Sea/Lake Ice Cover')),# Sea/Lake Ice Cover (WRF)                                          
                     T_SFC_LAKE   = dict(name='Tlake', units='K', transform=mask_array,  # lake surface temperature (FLake)
                                         atts=dict(missing_value=0, long_name='Lake Surface Temperature')),
                     T_ICE_LAKE   = dict(name='Tice', units='K', transform=mask_array, # lake ice temperature (FLake)
                                         atts=dict(missing_value=0, long_name='Lake Ice Temperature')), 
                     T_SNOW_LAKE  = dict(name='Tsnow', units='K', transform=mask_array, # lake snow temperature (FLake)
                                         atts=dict(missing_value=0, long_name='Lake Snow Temperature')), 
                     T_WML_LAKE   = dict(name='Tmix', units='K', transform=mask_array, # mixed layer temperature (FLake)
                                         atts=dict(missing_value=0, long_name='Mixed Layer Temperature')), 
                     H_ICE_LAKE   = dict(name='Hice', units='m', atts=dict(long_name='Lake Ice Height')), # FLake
                     H_SNOW_LAKE  = dict(name='Hsnow', units='m', atts=dict(long_name='Lake Snow Height')), # FLake
                     H_ML_LAKE    = dict(name='Hmix', units='m', atts=dict(long_name='Mixed Layer Height')), # FLake
                     )
    for threshold in precip_thresholds: # add variables with different wet-day thresholds
        suffix = '_{:03d}'.format(int(10*threshold))
        self.atts['WetDays'+suffix]      = dict(name='wetfrq'+suffix, units='') # fraction of wet/rainy days                    
        self.atts['WetDayRain'+suffix]   = dict(name='dryprec'+suffix, units='kg/m^2/s') # precipitation rate above dry-day thre
        self.atts['WetDayPrecip'+suffix] = dict(name='wetprec'+suffix, units='kg/m^2/s', 
                                                atts=dict(fillValue=0), transform=nullNaN) # wet-day precipitation rate (kg/m^2/s)
    self.vars = list(self.atts.keys())  
    self.ignore_list = []  
    self.climfile = 'wrfsrfc_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfsrfc_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# hydro variables (mostly for HGS)
class Hydro(FileType):
  ''' Variables and attributes of the hydrological files. '''
  def __init__(self):
    self.name = 'hydro'
    self.atts = dict(T2MEAN       = dict(name='Tmean', units='K'), # daily mean 2m Temperature
                     RAIN         = dict(name='precip', units='kg/m^2/s'), # total precipitation rate
                     RAINC        = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate
                     RAINNC       = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate
                     SFCEVP       = dict(name='evap', units='kg/m^2/s'), # actual surface evaporation/ET rate
                     ACSNOM       = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate 
                     POTEVP       = dict(name='pet_wrf', units='kg/m^2/s'), # potential evapo-transpiration rate
                     #pet          = dict(name='pet_wrf', units='kg/m^2/s',), # just renaming of old variable
                     #pet          = dict(name='pet_wrf', units='kg/m^2/s', scalefactor=999.70,), # just renaming of old variable
                     #pet_wrf      = dict(name='pet_wrf', units='kg/m^2/s', scalefactor=999.70,), # just rescaling of old variable
                     # N.B.: for some strange reason WRF outputs PET in m/s, rather than kg/m^2/s
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation rate
                     NetWaterFlux = dict(name='waterflx', units='kg/m^2/s'), # total water downward flux
                     #WetDays      = dict(name='wetfrq', units=''), # fraction of wet/rainy days 
                     #WetDayRain   = dict(name='dryprec', units='kg/m^2/s'), # precipitation rate above dry-day threshold (kg/m^2/s)
                     #WetDayPrecip = dict(name='wetprec', units='kg/m^2/s'), # wet-day precipitation rate (kg/m^2/s)
                     MaxNetWaterFlux = dict(name='MaxWaterFlx_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxRAIN      = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAIN_5d   = dict(name='MaxPrecip_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
                     MaxACSNOW    = dict(name='MaxSolprec_1d', units='kg/m^2/s'), # maximum daily precip
                     MaxACSNOW_5d = dict(name='MaxSolprec_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
                     MaxRAINC     = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxRAINC_5d  = dict(name='MaxPreccu_5d', units='kg/m^2/s'), # maximum pendat convective precip
                     MaxRAINNC    = dict(name='MaxPrecnc_1d', units='kg/m^2/s'), # maximum daily non-convective precip
                     MaxRAINNC_5d = dict(name='MaxPrecnc_5d', units='kg/m^2/s'),) # maximum pendat non-convective precip
    for threshold in precip_thresholds: # add variables with different wet-day thresholds
        suffix = '_{:03d}'.format(int(10*threshold))
        self.atts['WetDays'+suffix]      = dict(name='wetfrq'+suffix, units='') # fraction of wet/rainy days                    
        self.atts['WetDayRain'+suffix]   = dict(name='dryprec'+suffix, units='kg/m^2/s') # precipitation rate above dry-day threshold
        self.atts['WetDayPrecip'+suffix] = dict(name='wetprec'+suffix, units='kg/m^2/s', 
                                                atts=dict(fillValue=0), transform=nullNaN) # wet-day precipitation rate (kg/m^2/s)
    self.vars = list(self.atts.keys())
    self.ignore_list = []
    self.climfile = 'wrfhydro_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfhydro_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# land surface model variables
class LSM(FileType):
  ''' Variables and attributes of the land surface files. '''
  def __init__(self):
    self.name = 'lsm'
    self.atts = dict(ALBEDO   = dict(name='A', units=''), # Albedo
                     EMISS    = dict(name='e', units=''), # surface emissivity
                     ACGRDFLX = dict(name='grdflx', units='W/m^2'), # heat released from the soil (upward)
                     ACLHF    = dict(name='lhflx', units='W/m^2'), # latent heat flux from surface (upward)
                     ACHFX    = dict(name='hflx', units='W/m^2'), # sensible heat flux from surface (upward)
                     SNOWC    = dict(name='snwcvr', units=''), # snow cover (binary)
                     ACSNOM   = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate 
                     ACSNOW   = dict(name='snwacc', units='kg/m^2/s'), # snow accumulation rate
                     SFCEVP   = dict(name='evap', units='kg/m^2/s'), # actual surface evaporation/ET rate
                     POTEVP   = dict(name='pet_wrf', units='kg/m^2/s'), # potential evapo-transpiration rate
                     #pet      = dict(name='pet_wrf', units='kg/m^2/s', scalefactor=999.70,), # just renaming of old variable
                     #pet_wrf  = dict(name='pet_wrf', units='kg/m^2/s', scalefactor=999.70,), # just rescaling of old variable
                     # N.B.: for some strange reason WRF outputs PET in m/s, rather than kg/m^2/s
                     SFROFF   = dict(name='sfroff', units='kg/m^2/s'), # surface run-off
                     UDROFF   = dict(name='ugroff', units='kg/m^2/s'), # sub-surface/underground run-off
                     Runoff   = dict(name='runoff', units='kg/m^2/s'), # total surface and sub-surface run-off
                     TSLB     = dict(name='Tslb', units='K'), # soil temperature
                     SMOIS    = dict(name='aSM', units='m^3/m^3'), # absolute soil moisture
                     SMCREL   = dict(name='rSM', units=''), # relative soil moisture
                     # 3D lake variables
                     T_LAKE3D       = dict(name='T_LAKE3D', units='K', atts=dict(missing_value=-999., long_name='Lake Temperature')),
                     LAKE_ICEFRAC3D = dict(name='LAKE_ICEFRAC3D', units='', atts=dict(missing_value=-999., long_name='Lake Ice Fraction')),
                     Z_LAKE3D       = dict(name='Z_LAKE3D', units='m', atts=dict(missing_value=-999., long_name='Lake Layer Depth')),
                     DZ_LAKE3D      = dict(name='DZ_LAKE3D', units='m', atts=dict(missing_value=-999., long_name='Lake Layer Thickness')),
                     )
    self.vars = list(self.atts.keys()) 
    self.ignore_list = []   
    self.climfile = 'wrflsm_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrflsm_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# radiation variables
class Rad(FileType):
  ''' Variables and attributes of the radiation files. '''
  def __init__(self):
    self.name = 'rad'
    self.atts = dict(# radiation at surface (=*B)
                     ACSWUPB = dict(name='SWUPB', units='W/m^2'), # SW = short-wave
                     ACSWDNB = dict(name='SWDNB', units='W/m^2'), # LW = long-wave
                     ACLWUPB = dict(name='LWUPB', units='W/m^2'), # UP = up-welling
                     ACLWDNB = dict(name='LWDNB', units='W/m^2'), # DN = down-welling
                     # radiation at ToA (=*T)
                     ACSWUPT = dict(name='SWUPT', units='W/m^2'), 
                     ACSWDNT = dict(name='SWDNT', units='W/m^2'),
                     ACLWUPT = dict(name='LWUPT', units='W/m^2'),
                     ACLWDNT = dict(name='LWDNT', units='W/m^2'),
                     # clear sky (=*C)
                     ACSWUPBC = dict(name='SWUPBC', units='W/m^2'),
                     ACSWDNBC = dict(name='SWDNBC', units='W/m^2'),
                     ACLWUPBC = dict(name='LWUPBC', units='W/m^2'),
                     ACLWDNBC = dict(name='LWDNBC', units='W/m^2'),
                     ACSWUPTC = dict(name='SWUPTC', units='W/m^2'),
                     ACSWDNTC = dict(name='SWDNTC', units='W/m^2'),
                     ACLWUPTC = dict(name='LWUPTC', units='W/m^2'),
                     ACLWDNTC = dict(name='LWDNTC', units='W/m^2'),                     )
    self.vars = list(self.atts.keys())
    self.ignore_list = []    
    self.climfile = 'wrfrad_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfrad_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# extreme value variables
class Xtrm(FileType):
  ''' Variables and attributes of the extreme value files. '''
  def __init__(self):
    self.name = 'xtrm'
    self.atts = dict(#T2MEAN        = dict(name='Tmean', units='K'),  # daily mean Temperature (at 2m)
                     T2MEAN        = dict(name='Tmean', units='K'),  # daily mean Temperature (at 2m)
                     T2MIN         = dict(name='Tmin', units='K'),   # daily minimum Temperature (at 2m)
                     T2MAX         = dict(name='Tmax', units='K'),   # daily maximum Temperature (at 2m)
                     T2STD         = dict(name='Tstd', units='K'),   # daily Temperature standard deviation (at 2m)
                     SummerDays = dict(name='sumfrq', units='', atts=dict(long_name='Fraction of Summer Days (>25C)')),
                     FrostDays  = dict(name='frzfrq', units='', atts=dict(long_name='Fraction of Frost Days (< 0C)')),
                     SKINTEMPMEAN  = dict(name='TSmean', units='K'),  # daily mean Skin Temperature
                     Ts            = dict(name='TSmean', units='K'),  # daily mean Skin Temperature
                     #SKINTEMPMEAN  = dict(name='Ts', units='K'),  # daily mean Skin Temperature
                     SKINTEMPMIN   = dict(name='TSmin', units='K'),   # daily minimum Skin Temperature
                     SKINTEMPMAX   = dict(name='TSmax', units='K'),   # daily maximum Skin Temperature
                     SKINTEMPSTD   = dict(name='TSstd', units='K'),   # daily Skin Temperature standard deviation                     
                     Q2MEAN        = dict(name='qmean', units='kg/kg', scalefactor=1), # daily mean Water Vapor Mixing Ratio (at 2m)
                     Q2            = dict(name='qmean', units='kg/kg', scalefactor=1./Q), # daily mean Water Vapor Mixing Ratio (at 2m)
                     Q2MIN         = dict(name='qmin', units='kg/kg', scalefactor=1),  # daily minimum Water Vapor Mixing Ratio (at 2m)
                     Q2MAX         = dict(name='qmax', units='kg/kg', scalefactor=1),  # daily maximum Water Vapor Mixing Ratio (at 2m)
                     Q2STD         = dict(name='qstd', units='kg/kg', scalefactor=1),  # daily Water Vapor Mixing Ratio standard deviation (at 2m)
                     #SPDUV10MEAN   = dict(name='U10mean', units='m/s'), # daily mean Wind Speed (at 10m)
                     SPDUV10MEAN   = dict(name='U10', units='m/s'), # daily mean Wind Speed (at 10m)
                     SPDUV10MAX    = dict(name='Umax', units='m/s'),  # daily maximum Wind Speed (at 10m)
                     SPDUV10STD    = dict(name='Ustd', units='m/s'),  # daily Wind Speed standard deviation (at 10m)
                     #U10MEAN       = dict(name='u10mean', units='m/s'), # daily mean Westerly Wind (at 10m)
                     #V10MEAN       = dict(name='v10mean', units='m/s'), # daily mean Southerly Wind (at 10m)
                     U10MEAN       = dict(name='u10', units='m/s'), # daily mean Westerly Wind (at 10m)
                     V10MEAN       = dict(name='v10', units='m/s'), # daily mean Southerly Wind (at 10m)
                     #RAINMEAN      = dict(name='precipmean', units='kg/m^2/s'), # daily mean precipitation rate
                     RAINMEAN      = dict(name='precip', units='kg/m^2/s'), # daily mean precipitation rate
                     RAINMAX       = dict(name='precipmax', units='kg/m^2/s'), # daily maximum precipitation rate
                     RAINSTD       = dict(name='precipstd', units='kg/m^2/s'), # daily precip standard deviation                     
                     #RAINCVMEAN    = dict(name='preccumean', units='kg/m^2/s'), # daily mean convective precipitation rate
                     RAINCVMEAN    = dict(name='preccu', units='kg/m^2/s'), # daily mean convective precipitation rate
                     RAINCVMAX     = dict(name='preccumax', units='kg/m^2/s'), # daily maximum convective precipitation rate
                     RAINCVSTD     = dict(name='preccustd', units='kg/m^2/s'), # daily convective precip standard deviation
                     #RAINNCVMEAN   = dict(name='precncmean', units='kg/m^2/s'), # daily mean grid-scale precipitation rate
                     RAINNCVMEAN   = dict(name='precnc', units='kg/m^2/s'), # daily mean grid-scale precipitation rate
                     RAINNCVMAX    = dict(name='precncmax', units='kg/m^2/s'), # daily maximum grid-scale precipitation rate
                     RAINNCVSTD    = dict(name='precncstd', units='kg/m^2/s'), # daily grid-scale precip standard deviation
                     WetDays       = dict(name='wetfrq', units=''), # fraction of wet/rainy days 
                     MaxRAINMEAN   = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAINCVMEAN = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxPrecip     = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu     = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # for short-term consistency                     
                     MaxPrecnc     = dict(name='MaxPrecnc_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxRAINCVMAX  = dict(name='MaxPreccu_1h', units='kg/m^2/s'), # maximum hourly convective precip                    
                     MaxRAINNCVMAX = dict(name='MaxPrecnc_1h', units='kg/m^2/s'), # maximum hourly non-convective precip
                     MaxPreccumax  = dict(name='MaxPreccu_1h', units='kg/m^2/s'), # maximum hourly convective precip                    
                     MaxPrecncmax  = dict(name='MaxPrecnc_1h', units='kg/m^2/s'),) # maximum hourly non-convective precip
    self.vars = list(self.atts.keys()) 
    self.ignore_list = []   
    self.climfile = 'wrfxtrm_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfxtrm_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# variables on selected pressure levels: 850 hPa, 700 hPa, 500 hPa, 250 hPa, 100 hPa
class Plev3D(FileType):
  ''' Variables and attributes of the pressure level files. '''
  def __init__(self):
    self.name = 'plev3d'
    self.atts = dict(T_PL      = dict(name='T',  units='K',   fillValue=-999, atts=dict(long_name='Temperature')), # Temperature
                     TD_PL     = dict(name='Td', units='K',   fillValue=-999, atts=dict(long_name='Dew-point Temperature')), # Dew-point Temperature
                     RH_PL     = dict(name='RH', units='\%',  fillValue=-999, atts=dict(long_name='Relative Humidity')), # Relative Humidity
                     GHT_PL    = dict(name='Z',  units='m',   fillValue=-999, atts=dict(long_name='Geopotential Height ')), # Geopotential Height 
                     S_PL      = dict(name='U',  units='m/s', fillValue=-999, atts=dict(long_name='Absolute Wind Speed')), # Wind Speed
                     U_PL      = dict(name='u',  units='m/s', fillValue=-999, atts=dict(long_name='Zonal Wind Speed')), # Zonal Wind Speed
                     V_PL      = dict(name='v',  units='m/s', fillValue=-999, atts=dict(long_name='Meridional Wind Speed')), # Meridional Wind Speed
                     WaterFlux_U      = dict(name='qwu',  units='kg/m^2/s', fillValue=-999, atts=dict(ong_name='Zonal Water Vapor Flux')), # zonal water (vapor) flux
                     WaterFlux_V      = dict(name='qwv',  units='kg/m^2/s', fillValue=-999, atts=dict(ong_name='Meridional Water Vapor Flux')), # meridional water (vapor) flux
                     WaterTransport_U = dict(name='cqwu', units='kg/m/s',   fillValue=-999, atts=dict(ong_name='column-integrated Zonal Water Vapor Transport')), # column-integrated zonal water (vapor) transport
                     WaterTransport_V = dict(name='cqwv', units='kg/m/s',   fillValue=-999, atts=dict(ong_name='column-integrated Meridional Water Vapor Transport')), # column-integrated meridional water (vapor) transport
                     ColumnWater      = dict(name='cqw',  units='kg/m^2',   fillValue=-999, atts=dict(ong_name='Column-integrated Water Vapor Content')), # column-integrated water (vapor) content
                     HeatFlux_U       = dict(name='qhu',  units='J/m^2/s',  fillValue=-999, atts=dict(ong_name='Zonal Heat Flux')), # zonal heat flux
                     HeatFlux_V       = dict(name='qhv',  units='J/m^2/s',  fillValue=-999, atts=dict(ong_name='Meridional Heat Flux')), # meridional heat flux
                     HeatTransport_U  = dict(name='cqhu', units='J/m/s',    fillValue=-999, atts=dict(ong_name='Column-integrated Zonal Heat Transport')), # column-integrated zonal heat transport
                     HeatTransport_V  = dict(name='cqhv', units='J/m/s',    fillValue=-999, atts=dict(ong_name='Column-integrated Meridional Heat Transport')), # column-integrated meridional heat transport
                     ColumnHeat       = dict(name='cqh',  units='J/m^2',    fillValue=-999, atts=dict(ong_name='Column-integrated Heat Content')), # column-integrated heat content
                     Vorticity        = dict(name='zeta', units='1/s',      fillValue=-999, atts=dict(ong_name='Relative Vorticity')), # (relative) Vorticity
                     OrographicIndex  = dict(name='OI', units='', atts=dict(long_name='Orographic Index')), # projection of wind onto slope
                     P_PL      = dict(name='p', units='Pa', atts=dict(long_name='Pressure')))  # Pressure
    self.vars = list(self.atts.keys())  
    self.ignore_list = []  
    self.climfile = 'wrfplev3d_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfplev3d_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)

# axes (don't have their own file)
class Axes(FileType):
  ''' A mock-filetype for axes. '''
  def __init__(self):
    self.name = 'axes'
    self.atts = dict(time        = dict(name='time', units='month'), # time coordinate
                     #Time        = dict(name='Time', units='day'), # original WRF time coordinate
                     # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                     #       the time offset is chose such that 1979 begins with the origin (time=0)
                     west_east   = dict(name='x', units='m'), # projected west-east coordinate
                     south_north = dict(name='y', units='m'), # projected south-north coordinate
                     x           = dict(name='x', units='m'), # projected west-east coordinate
                     y           = dict(name='y', units='m'), # projected south-north coordinate
                     soil_layers_stag = dict(name='i_s', units=''), # soil layer coordinate
                     num_press_levels_stag = dict(name='i_p', units=''), # pressure coordinate
                     station     = dict(name='station', units='#') ) # station axis for station data
    self.vars = list(self.atts.keys())
    self.ignore_list = []
    self.climfile = None
    self.tsfile = None

# data source/location
fileclasses = dict(aux=FileType('aux'), const=Const(), srfc=Srfc(), hydro=Hydro(), lsm=LSM(), rad=Rad(), xtrm=Xtrm(), plev3d=Plev3D(), axes=Axes())
outfolder = root_folder + 'wrfout/' # WRF output folder
avgfolder = root_folder + 'wrfavg/' # long-term mean folder

# add generic extremes to varatts dicts
for fileclass in fileclasses.values():
  atts = dict()
  for key,val in fileclass.atts.items():
#     if val['units'].lower() == 'k': 
#     else: extrema = ('Max',) # precip et al. only has maxima
    extrema = ('Max','Min') # only temperature has extreme minima
    for x in extrema:
      if key[:3] not in ('Max','Min'):
        att = val.copy()
        att['name'] = x+att['name'].title()
        atts[x+key] = att
        if fileclass.name in ('hydro','lsm'):
          att = att.copy()
          att['name'] = att['name']+'_5d'
          atts[x+key+'_5d'] = att
        if fileclass.name in ('xtrm',):
          att = att.copy()
          att['name'] = 'Tof'+att['name'][0].upper()+att['name'][1:]
          #del att['units'] # just leave units as they are
          att['units'] = 'min' # minutes since simulation start
          atts['T'+key] = att
  atts.update(fileclass.atts) # don't overwrite existing definitions
  fileclass.atts = atts


## get WRF projection and grid definition 
# N.B.: Unlike with observational datasets, model Meta-data depends on the experiment and has to be 
#       loaded from the NetCFD-file; a few conventions have to be defined, however.

# get projection NetCDF attributes
def getWRFproj(dataset, name=''):
  ''' Method to infer projection parameters from a WRF output file and return a GDAL SpatialReference object. '''
  if not isinstance(dataset,nc.Dataset): raise TypeError
  if not isinstance(name,str): raise TypeError
  if dataset.MAP_PROJ == 1: 
    # Lambert Conformal Conic projection parameters
    proj = 'lcc' # Lambert Conformal Conic  
    lat_1 = dataset.TRUELAT1 # Latitude of first standard parallel
    lat_2 = dataset.TRUELAT2 # Latitude of second standard parallel
    lat_0 = dataset.CEN_LAT # Latitude of natural origin
    lon_0 = dataset.STAND_LON # Longitude of natural origin
    lon_1 = dataset.CEN_LON # actual center of map
  else:
    raise NotImplementedError("Can only infer projection parameters for Lambert Conformal Conic projection (#1).")
  projdict = dict(proj=proj,lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0,lon_1=lon_1)
  # pass results to GDAL module to get projection object
  return getProjFromDict(projdict, name=name, GeoCS='WGS84', convention='Proj4')  

# infer grid (projection and axes) from constants file
def getWRFgrid(name=None, experiment=None, domains=None, folder=None, filename='wrfconst_d{0:0=2d}.nc', 
               ncformat='NETCDF4', exps=None):
  ''' Infer the WRF grid configuration from an output file and return a GridDefinition object. '''
  # check input
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, 
                                                        folder=folder, exps=exps)
  if isinstance(filename,str): filepath = '{}/{}'.format(folder,filename) # still contains formaters
  else: raise TypeError(filename)
  # figure out experiment
  if experiment is None:
    if not isinstance(name,str): raise TypeError
    if isinstance(exps,dict):
      if name in exps: experiment = exps[name]
      elif len(names) > 0: 
        tmp = names[0].split('_')[0]
        if tmp in exps: experiment = exps[tmp]
  elif not isinstance(experiment,Exp): raise TypeError  
  maxdom = max(domains) # max domain
  # files to work with
  for n in range(1,maxdom+1):
    dnfile = filepath.format(n,'') # expects grid name as well, but we are only looking for the native grid
    if not os.path.exists(dnfile):
      if n in domains: raise IOError('File {} for domain {:d} not found!'.format(dnfile,n))
      else: raise IOError('File {} for domain {:d} not found; this file is necessary to infer the geotransform for other domains.'.format(dnfile,n))
  # open first domain file (special treatment)
  dn = nc.Dataset(filepath.format(1,''), mode='r', format=ncformat)
  gridname = experiment.grid if isinstance(experiment,Exp) else name # use experiment name as default
  projection = getWRFproj(dn, name=gridname) # same for all
  # get coordinates of center point  
  clon = dn.CEN_LON; clat = dn.CEN_LAT
  wgs84 = osr.SpatialReference() # create coordinate/spatial reference system
  wgs84.SetWellKnownGeogCS("WGS84") # set to regular lat/lon
#   wgs84.ImportFromEPSG(4326) # for some reason this causes an OGR Error...
  tx = osr.CoordinateTransformation(wgs84, projection) # transformation object
  cx, cy, cz = tx.TransformPoint(float(clon),float(clat)); del cz # center point in projected (WRF) coordinates
  #print ' (CX,CY,CZ) = ', cx, cy, cz 
  # infer size and geotransform
  def getXYlen(ds):
    ''' a short function to infer the length of horizontal axes from a dataset with unknown naming conventions ''' 
    if 'west_east' in ds.dimensions and 'south_north' in ds.dimensions:
      nx = len(ds.dimensions['west_east']); ny = len(ds.dimensions['south_north'])
    elif 'x' in ds.dimensions and 'y' in ds.dimensions:
      nx = len(ds.dimensions['x']); ny = len(ds.dimensions['y'])
    else: raise AxisError('No horizontal axis found, necessary to infer projection/grid configuration.')
    return nx,ny
  dx = float(dn.DX); dy = float(dn.DY)
  nx,ny = getXYlen(dn)
  x0 = -float(nx)*dx/2.; y0 = -float(ny)*dy/2.
  x0 += cx; y0 += cy # shift center, if necessary 
  size = (nx, ny); geotransform = (x0,dx,0.,y0,0.,dy)
  name = names[0] if 1 in domains else 'tmp'  # update name, if first domain has a name...
  griddef = GridDefinition(name=name, projection=projection, geotransform=geotransform, size=size)
  dn.close()
  if 1 in domains: griddefs = [griddef]
  else: griddefs = []
  if maxdom > 1:
    # now infer grid of domain of interest
    geotransforms = [geotransform]
    # loop over grids
    for n in range(2,maxdom+1):
      # open file
      dn = nc.Dataset(filepath.format(n,''), mode='r', format=ncformat)
      if not n == dn.GRID_ID: raise DatasetError # just a check
      pid = dn.PARENT_ID-1 # parent grid ID
      # infer size and geotransform      
      px0,pdx,s,py0,t,pdy = geotransforms[pid]; del s,t
      dx = float(dn.DX); dy = float(dn.DY)
      x0 = px0+float(dn.I_PARENT_START-0.5)*pdx - 0.5*dx  
      y0 = py0+float(dn.J_PARENT_START-0.5)*pdy - 0.5*dy
      size = getXYlen(dn) 
      geotransform = (x0,dx,0.,y0,0.,dy)
      dn.close()
      geotransforms.append(geotransform) # we need that to construct the next nested domain
      if n in domains:
        name = '{0:s}_d{1:02d}'.format(gridname,n) 
        griddefs.append(GridDefinition(name=name, projection=projection, geotransform=geotransform, size=size))
  # return a GridDefinition object
  return tuple(griddefs)  

# return name and folder
def getFolderNameDomain(name=None, experiment=None, domains=None, folder=None, lexp=False, exps=None):
  ''' Convenience function to infer and type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  if name is None and experiment is None: raise ArgumentError
  if folder is None and experiment is None: raise ArgumentError 
  # handle experiment
  if experiment is not None and name is None:
    if isinstance(experiment,Exp):
      name = experiment.name
    elif isinstance(experiment,str) or isinstance(name,(list,tuple)): 
      name = experiment # use name-logic to procede
      experiment = None # eventually reset below
    else: TypeError
  ## figure out experiment name(s) and domain(s) (potentially based on name)
  # name check
  if isinstance(name,(list,tuple)) and all([isinstance(n,str) for n in name]): names = name
  elif isinstance(name,str): names = [name]
  else: raise TypeError(name)
  # domain check
  if not isinstance(domains,(list,tuple)): domains = [domains]*len(names)
  elif isinstance(domains,tuple): domains = list(domains)
  if not all(dom is None or isinstance(dom,(int,np.integer)) for dom in domains): raise TypeError
  if len(domains) == 1: domains = domains*len(names)
  if len(names) == 1: names = names*len(domains)
  if len(domains) != len(names): raise ArgumentError
  # parse for domain string
  basenames = []
  for i,name in enumerate(names):
    # infer domain from suffix...
    if name[-4:-2] == '_d' and int(name[-2:]) > 0:
      domains[i] = int(name[-2:]) # overwrite domain list entry
      basenames.append(name[:-4]) # ... and truncate basename
    else: basenames.append(name)
  if len(set(basenames)) != 1: raise DatasetError("Dataset base names are inconsistent.")
  # ensure uniqueness of names
  if len(set(names)) != len(names):
    names = ['{0:s}_d{1:0=2d}'.format(name,domain) for name,domain in zip(names,domains)]
  name = basenames[0]
  # evaluate experiment
  if experiment is None: 
    if exps is None:
      if lexp: raise DatasetError('No dictionary of Exp instances specified.')
    else:
      if name in exps: experiment = exps[name] # load experiment meta data
      elif lexp: raise DatasetError('Dataset of name \'{0:s}\' not found!'.format(names[0]))
  if not ( experiment or folder ): raise DatasetError("Need to specify either a valid experiment name or a full path.")
  # patch up folder
  if experiment: # should already have checked that either folder or experiment are specified
    folder = experiment.avgfolder
    # assign unassigned domains
    domains = [experiment.domains if dom is None else dom for dom in domains]
  elif isinstance(folder,str): 
    if not folder.endswith((name,name+'/')): folder = '{:s}/{:s}/'.format(folder,name)
  else: raise TypeError(folder)
  # check types
  if not isinstance(domains,(tuple,list)): raise TypeError    
  if not all(isInt(domains)): raise TypeError
  if not domains == sorted(domains): raise IndexError('Domains have to be sorted in ascending order.')
  if not isinstance(names,(tuple,list)): raise TypeError
  if not all(isinstance(nm,str) for nm in names): raise TypeError
  if len(domains) != len(names): raise ArgumentError  
  # check if folder exists
  if not os.path.exists(folder): raise IOError(folder)
  # return name and folder
  return folder, experiment, tuple(names), tuple(domains)


## Functions to load different types of WRF datasets

# Station Time-series (monthly, with extremes)
def loadWRF_StnTS(experiment=None, name=None, domains=None, station=None, grid=None, filetypes=None, 
                  varlist=None, varatts=None, lctrT=True, lfixPET=True, lwrite=False, ltrimT=True, 
                  exps=None, bias_correction=None, lconst=True):
  ''' Get a properly formatted WRF dataset with monthly time-series at station locations. '''  
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=station, 
                     period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                     lconst=lconst, lautoregrid=False, lctrT=lctrT, lfixPET=lfixPET, mode='time-series', 
                     lwrite=lwrite, ltrimT=ltrimT, check_vars='station_name', exps=exps,
                     bias_correction=bias_correction)  

# Regiona/Shape Time-series (monthly, with extremes)
def loadWRF_ShpTS(experiment=None, name=None, domains=None, shape=None, grid=None, filetypes=None, varlist=None, 
                  varatts=None, lctrT=True, lfixPET=True, lencl=False, lwrite=False, ltrimT=True, exps=None,
                  bias_correction=None, lconst=True):
  ''' Get a properly formatted WRF dataset with monthly time-series averaged over regions. '''  
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, shape=shape, lencl=lencl, 
                     station=None, period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                     lconst=lconst, lautoregrid=False, lctrT=lctrT, lfixPET=lfixPET, mode='time-series', lwrite=lwrite, 
                     ltrimT=ltrimT, check_vars='shape_name', exps=exps, bias_correction=bias_correction)  

def loadWRF_TS(experiment=None, name=None, domains=None, grid=None, filetypes=None, varlist=None, 
               varatts=None, lconst=True, lautoregrid=True, lctrT=True, lfixPET=True, lwrite=False, ltrimT=True, 
               exps=None, bias_correction=None):
  ''' Get a properly formatted WRF dataset with monthly time-series. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=None, exps=exps, 
                     period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=lautoregrid, lctrT=lctrT, lfixPET=lfixPET, mode='time-series', lwrite=lwrite, 
                     ltrimT=ltrimT, bias_correction=bias_correction)  

def loadWRF_Stn(experiment=None, name=None, domains=None, station=None, grid=None, period=None, filetypes=None, 
                varlist=None, varatts=None, lctrT=True, lfixPET=True, lwrite=False, ltrimT=False, exps=None,
                bias_correction=None, lconst=True):
  ''' Get a properly formatted station dataset from a monthly WRF climatology at station locations. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=station, 
                     period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=False, lctrT=lctrT, lfixPET=lfixPET, mode='climatology', lwrite=lwrite, 
                     ltrimT=ltrimT, check_vars='station_name', exps=exps, bias_correction=bias_correction)  

def loadWRF_Shp(experiment=None, name=None, domains=None, shape=None, grid=None, period=None, filetypes=None, 
                varlist=None, varatts=None, lctrT=True, lfixPET=True, lencl=False, lwrite=False, ltrimT=False, 
                exps=None, bias_correction=None, lconst=True):
  ''' Get a properly formatted station dataset from a monthly WRF climatology averaged over regions. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, shape=shape, lencl=lencl,
                     station=None, period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                     lconst=lconst, lautoregrid=False, lctrT=lctrT, lfixPET=lfixPET, mode='climatology', lwrite=lwrite, 
                     ltrimT=ltrimT, check_vars='shape_name', exps=exps, bias_correction=bias_correction)  

def loadWRF(experiment=None, name=None, domains=None, grid=None, period=None, filetypes=None, varlist=None, 
            varatts=None, lconst=True, lautoregrid=True, lctrT=True, lfixPET=True, lwrite=False, ltrimT=False, 
            exps=None, bias_correction=None):
  ''' Get a properly formatted monthly WRF climatology as NetCDFDataset. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=None, exps=exps, 
                     period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=lautoregrid, lctrT=lctrT, lfixPET=lfixPET, mode='climatology', lwrite=lwrite, 
                     ltrimT=ltrimT, bias_correction=bias_correction)  

# pre-processed climatology files (varatts etc. should not be necessary) 
def loadWRF_All(experiment=None, name=None, domains=None, grid=None, station=None, shape=None, period=None, 
                filetypes=None, varlist=None, varatts=None, lfilevaratts=False, lconst=True, lautoregrid=True, 
                lencl=False, lctrT=False, lfixPET=True, folder=None, lpickleGrid=True, mode='climatology', 
                lwrite=False, ltrimT=False, check_vars=None, exps=None, bias_correction=None):
  ''' Get any WRF data files as a properly formatted NetCDFDataset. '''
  # prepare input  
  ltuple = isinstance(domains,col.Iterable)  
  # prepare input  
  if experiment is None and name is not None: 
    experiment = name; name=None # allow 'name' to define an experiment  
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, 
                                                        folder=folder, exps=exps)
  if lctrT and experiment is None: 
    raise DatasetError("Experiment '{0:s}' not found in database; need time information to center time axis.".format(names[0]))    
  # figure out period
  if isinstance(period,(tuple,list)):
    if not all(isNumber(period)): raise ValueError
  elif isinstance(period,str): period = [int(prd) for prd in period.split('-')]
  elif isinstance(period,(int,np.integer)): 
    beginyear = int(experiment.begindate[0:4])
    period = (beginyear, beginyear+period)
  elif period is None: pass # handled later
  else: raise DateError("Illegal period definition: {:s}".format(str(period)))
  lclim = False; lts = False # mode switches
  if mode.lower() == 'climatology': # post-processed climatology files
    lclim = True
    periodstr = '_{0:4d}-{1:4d}'.format(*period)
    if period is None: raise DateError('Currently WRF Climatologies have to be loaded with the period explicitly specified.')
  elif mode.lower() in ('time-series','timeseries'): # concatenated time-series files
    lts = True; lclim = False; period = None; periodstr = None # to indicate time-series (but for safety, the input must be more explicit)
    if lautoregrid is None: lautoregrid = False # this can take very long!
  # cast/copy varlist
  if isinstance(varlist,str): varlist = [varlist] # cast as list
  elif varlist is not None: varlist = list(varlist) # make copy to avoid interference
  # figure out station and shape options
  if station and shape: raise ArgumentError
  elif station or shape: 
    if lautoregrid: raise GDALError('Station data can not be regridded, since it is not map data.')   
    lstation = bool(station); lshape = bool(shape)
    # add station/shape parameters
    if varlist:
      params = stn_params if lstation else shp_params
      for param in params:
        if param not in varlist: varlist.append(param)
  else:
    lstation = False; lshape = False
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = ('hydro','xtrm','srfc','plev3d','lsm')
  if isinstance(filetypes,(tuple,set)): filetypes = list(filetypes)
  elif isinstance(filetypes,str): filetypes = [filetypes,]
  elif isinstance(filetypes,list): filetypes = list(filetypes) # also make copy for modification
  else: raise TypeError
  if 'axes' in filetypes: del filetypes[filetypes.index('axes')] # remove axes - not a real filetype
  #if 'const' not in filetypes and grid is None: filetypes.append('const')
  if bias_correction: # optional filetype for bias-corrected data
      filetypes = [bias_correction.lower(),]+filetypes # add before others, so that bias-corrected variables are loaded
  atts = []; filelist = []; typelist = []; ignore_lists = []
  for filetype in filetypes: # last filetype in list has precedence
    fileclass = fileclasses[filetype] if filetype in fileclasses else FileType(filetype)
    if lclim and fileclass.climfile is not None:
      filelist.append(fileclass.climfile)
      typelist.append(filetype) # this eliminates const files
    elif lts: 
      if fileclass.tsfile is not None: 
        filelist.append(fileclass.tsfile)
        typelist.append(filetype) # this eliminates const files
    ignore_lists.append(fileclass.ignore_list) # list of ignore lists
    # get varatts
#     if filetype == 'const': lconst = True
#     else:
    att = fileclasses['axes'].atts.copy()
    att.update(fileclass.atts) # use axes atts as basis and override with filetype-specific atts
    atts.append(att) # list of atts for each filetype    
  # resolve varatts argument and update default atts
  if varatts is None: pass
  elif isinstance(varatts,dict):
    if lfilevaratts:
      for filetype,att in list(varatts.items()):
        if not isinstance(att, dict): raise TypeError(filetype,att)
        atts[typelist.index(filetype, )].update(att) # happens in-place
    else:
      for att in atts: att.update(varatts) # happens in-place
  elif isinstance(varatts,(list,tuple)):
    for att,varatt in zip(atts,varatts): att.update(varatt) # happens in-place
  else:
    raise TypeError(varatts)
  # NetCDF file mode
  ncmode = 'rw' if lwrite else 'r' 
  # center time axis to 1979
  if lctrT and experiment is not None:
    for att in atts: # loop over all filetypes and change time axis atts based on experiment meta data
      if 'time' in att: tatt = att['time']
      else: tatt = dict()
      ys,ms,ds = [int(t) for t in experiment.begindate.split('-')]; assert ds == 1   
      tatt['offset'] = (ys-1979)*12 + (ms-1)
      tatt['atts'] = dict(long_name='Month since 1979-01')
      att['time'] = tatt # this should happen in-place  
  # translate varlist
  #if varlist is not None: varlist = translateVarNames(varlist, atts) # default_varatts
  # N.B.: renaming of variables in the varlist is now handled in theDatasetNetCDF initialization routine
  # infer projection and grid and generate horizontal map axes
  # N.B.: unlike with other datasets, the projection has to be inferred from the netcdf files  
  if not lstation and not lshape:
    if grid is None:
      # load pickled griddefs from disk (much faster than recomputing!)
      if not lpickleGrid or experiment is None: raise IOError # don't load pickle!
      griddefs = []
      for domain in domains:
        # different "native" grid for each domain
        griddefs.append( loadPickledGridDef(grid=experiment.grid, res='d0{:d}'.format(domain), 
                                            filename=None, folder=grid_folder, check=True) )
      # N.B.: pickles are mainly used to speed up loading datasets; if pickles are not available, infer grid from files
      if not all(griddefs):
        # print warning to alert user that this takes a bit longer
        if experiment is not None:
          name = "'{:s}' ('{:s}')".format(experiment.name,experiment.grid) 
        else: name = "'{:s}'".format(names[0])        
        warn("Recomputing Grid Definition for Experiment {:s}".format(name))
        # compute grid definition from wrfconst files (requires all parent domains) 
        griddefs = None; c = 0
        filename = list(fileclasses.values())[c].tsfile # just use the first filetype
        while griddefs is None:
          # some experiments do not have all files... try, until one works...
          try:
            if filename is None: raise IOError # skip and try next one
            griddefs = getWRFgrid(name=names, experiment=experiment, domains=domains, folder=folder, filename=filename, exps=exps)
          except IOError:
            c += 1
            if c >= len(list(fileclasses.values())): 
              raise GDALError("Unable to infer grid definition for experiment '{:s}'.\n Not enough information in source files or required source file not available.".format(name))
            filename = list(fileclasses.values())[c].tsfile
    else:
      griddefs = [loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder, check=True)]*len(domains)
    assert len(griddefs) == len(domains)
  else:
    griddefs = [None]*len(domains) # not actually needed
  # grid
  datasets = []
  for name,domain,griddef in zip(names,domains,griddefs):
#     if grid is None or grid.split('_')[0] == experiment.grid: gridstr = ''
    native_gridstr = '{0:s}_d{1:02d}'.format(experiment.grid,domain)
    if lstation or lshape:
      # the station or shape name can be inserted as the grid name
      if lstation: gridstr = '_'+station.lower(); # only use lower case for filenames
      elif lshape: gridstr = '_'+shape.lower(); # only use lower case for filenames
      llconst = False # don't load constants (some constants are already in the file anyway)
      axes = None
      if grid and grid != native_gridstr: gridstr += '_'+grid.lower(); # only use lower case for filenames
    else:
      if grid is None or grid == native_gridstr: 
          gridstr = ''; llconst = lconst
      else: 
        gridstr = '_'+grid.lower(); # only use lower case for filenames
        llconst = False # don't load constants     
      # domain-sensitive parameters
      axes = dict(west_east=griddef.xlon, south_north=griddef.ylat, x=griddef.xlon, y=griddef.ylat) # map axes
    # load constants
    if llconst:              
      constfile = fileclasses['const']    
      catts = fileclasses['axes'].atts.copy()
      catts.update(constfile.atts) # use axes atts as basis and override with filetype-specific atts
      filename = constfile.tsfile.format(domain,gridstr)  
      # check file path
      constfolder = folder
      if not os.path.exists('{:s}/{:s}'.format(constfolder,filename)) and experiment and experiment.grid:
          constfolder = folder[:folder.find(experiment.name)] + '{:s}/'.format(experiment.grid)
      if not os.path.exists('{:s}/{:s}'.format(constfolder,filename)):
        raise IOError("Constant file for experiment '{:s}' not found.\n('{:s}')".format(name,'{:s}/{:s}'.format(constfolder,filename)))
      # i.e. if there is no constant file in the experiment folder, use the one in the grid definition folder (if available)
      # load dataset
      const = DatasetNetCDF(name=name, folder=constfolder, filelist=[filename], varatts=catts, 
                            axes=axes, varlist=constfile.vars, multifile=False, ncformat='NETCDF4', 
                            mode=ncmode, squeeze=True, ignore_list=[constfile.ignore_list])      
      lenc = len(const) # length of const dataset
    else: lenc = 0 # empty
    ## load regular variables
    filenames = []
    for filetype,fileformat in zip(typelist,filelist):
      if lclim: filename = fileformat.format(domain,gridstr,periodstr) # insert domain number, grid, and period
      elif lts: filename = fileformat.format(domain,gridstr) # insert domain number, and grid
      filenames.append(filename) # file list to be passed on to DatasetNetCDF
      # check existence
      filepath = '{:s}/{:s}'.format(folder,filename)
      if not os.path.exists(filepath):
        if lclim: nativename = fileformat.format(domain,'',periodstr) # original filename (before regridding)
        elif lts: nativename = fileformat.format(domain,'') # original filename (before regridding)
        nativepath = '{:s}/{:s}'.format(folder,nativename)
        if os.path.exists(nativepath):
          if lautoregrid: # already set to False for stations
            from processing.regrid import performRegridding # causes circular reference if imported earlier
            #griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder) # already done above
            dataargs = dict(experiment=experiment, filetypes=[filetype], domain=domain, period=period)
            print(("The '{:s}' (WRF) dataset for the grid ('{:s}') is not available:\n Attempting regridding on-the-fly.".format(name,filename,grid)))
            if performRegridding('WRF', 'climatology', griddef, dataargs): # True if exitcode 1
              raise IOError("Automatic regridding failed!")
            print(("Output: '{:s}'".format(name,filename,grid,filepath)))            
          else: raise IOError("The  '{:s}' (WRF) dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.\n('{:s}')".format(name,filename,grid,filepath)) 
        else: raise IOError("The file '{:s}' in WRF dataset '{:s}' does not exits!\n('{:s}')".format(filename,name,filepath))   
       
    # load dataset
    check_override = ['time'] if lctrT else None
    try:
      dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=axes, 
                              varatts=atts, multifile=False, ncformat='NETCDF4', ignore_list=ignore_lists, 
                              mode=ncmode, squeeze=True, check_override=check_override, check_vars=check_vars)
    except EmptyDatasetError:
      if lenc == 0: raise # allow loading of cosntants without other variables
    if ltrimT and dataset.hasAxis('time') and len(dataset.time) > 180:
      if lwrite: raise ArgumentError("Cannot trim time-axis when NetCDF write mode is enabled!")
      dataset = dataset(time=slice(0,180), lidx=True)
      assert len(dataset.time) == 180, len(dataset.time) 
    # check
    if (len(dataset)+lenc) == 0: 
        raise DatasetError('Dataset is empty - check source file or variable list!')
    # check time axis and center at 1979-01 (zero-based)
    if lctrT and dataset.hasAxis('time'):
      # N.B.: we can directly change axis vectors, since they reside in memory;
      #       the source file is not changed, unless sync() is called
      if lts:
        dataset.time.coord = np.arange(len(dataset.time), dtype=dataset.time.dtype) + (ys-1979)*12 + (ms-1)
        # N.B.: shifting is dangerous, because of potential repeated application
#         t0 = dataset.time.coord[0]
#         tm = t0%12; ty = int(np.floor(t0/12))  
#         if tm != ms - 1: 
#           dataset.time.coord -= ( tm+1 - ms )
#           dataset.time.offset -= ( tm+1 - ms )
#         if ty != ys - 1979: 
#           dataset.time.coord -= ( ty+1979 - ys )*12
#           dataset.time.offset -= ( ty+1979 - ys )*12 
      elif lclim:
        dataset.time.coord = np.arange(len(dataset.time), dtype=dataset.time.dtype) + 1
        # N.B.: shifting is dangerous, because of potential repeated application
#         t0 = dataset.time.coord[0]
#         # there is no "year" - just start with "1" for january 
#         if t0 != 1: 
#           dataset.time.coord -= ( t0 - 1 )
#           dataset.time.offset -= ( t0 - 1 )
      # correct ordinal number of shape (should start at 1, not 0)
    if lfixPET and 'pet_wrf' in dataset:
        pet_wrf = dataset['pet_wrf'].load()
        assert pet_wrf.units == 'kg/m^2/s', pet_wrf
        if pet_wrf.mean() < 1e-7: 
            warn("WARNING: WRF PET values too low; multiplying by 999.7!")
            pet_wrf *= 999.70
        elif pet_wrf.mean() > 1e-4: 
            pet_wrf /= 999.70
            warn("WARNING: WRF PET values too high; divinding by 999.7!")
        # N.B.: this is quite ugly, but seems to be necessayr to clear up the mess with PET; 
        #       ideally it should not stay here for long...
    if lshape:
      # mask all shapes that are incomplete in dataset
      if lencl and 'shp_encl' in dataset: dataset.mask(mask='shp_encl', invert=True)
      if dataset.hasAxis('shapes'): raise AxisError("Axis 'shapes' should be renamed to 'shape'!")
      if not dataset.hasAxis('shape'): raise AxisError
      if dataset.shape.coord[0] == 0: dataset.shape.coord += 1
    # add constants to dataset
    if llconst:
      for var in const:
        if var.name not in dataset:
          dataset.addVariable(var, asNC=False, copy=False, loverwrite=False, deepcopy=False)
    if not lstation and not lshape:
      # add projection
      dataset = addGDALtoDataset(dataset, griddef=griddef, gridfolder=grid_folder, geolocator=True)      
    # add resolution string
    if 'DX' in dataset.atts and 'DY' in dataset.atts:
        dataset.atts['resstr'] = "{:d}km".format(int((dataset.atts['DX']+dataset.atts['DY'])/2000.))
    else: dataset.atts['resstr'] = None
    # N.B.: all WRF datasets should inherit these attributed from the original netcdf files!
    # append to list
    datasets.append(dataset) 
  # return formatted dataset
  if not ltuple: datasets = datasets[0]
  else: tuple(datasets)
  return datasets


# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_StnEns(ensemble=None, name=None, station=None, grid=None, filetypes=None, years=None, domains=None, 
                   varlist=None, title=None, varatts=None, translateVars=None, lcheckVars=None, 
                   lcheckAxis=True, lwrite=False, axis=None, lensembleAxis=False, exps=None, enses=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadWRF_Ensemble(ensemble=ensemble, grid=grid, station=station, domains=domains, 
                          filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, 
                          translateVars=translateVars, lautoregrid=False, lctrT=True, lconst=False,
                          lcheckVars=lcheckVars, lcheckAxis=lcheckAxis, name=name, title=title, 
                          lwrite=lwrite, lensembleAxis=lensembleAxis, check_vars='station_name', 
                          exps=exps, enses=enses)
  
# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_ShpEns(ensemble=None, name=None, shape=None, grid=None, filetypes=None, years=None, domains=None, 
                   varlist=None, title=None, varatts=None, translateVars=None, lcheckVars=None, 
                   lcheckAxis=True, lencl=False, lwrite=False, axis=None, lensembleAxis=False, 
                   exps=None, enses=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadWRF_Ensemble(ensemble=ensemble, grid=grid, station=None, shape=shape, domains=domains, 
                          filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, lencl=lencl, 
                          translateVars=translateVars, lautoregrid=False, lctrT=True, lconst=False,
                          lcheckVars=lcheckVars, lcheckAxis=lcheckAxis, name=name, title=title, 
                          lwrite=lwrite, axis=axis, lensembleAxis=lensembleAxis, check_vars='shape_name', 
                          exps=exps, enses=enses)
  
# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_Ensemble(ensemble=None, name=None, grid=None, station=None, shape=None, domains=None, 
                     filetypes=None, years=None, varlist=None, varatts=None, translateVars=None, 
                     lautoregrid=None, title=None, lctrT=True, lconst=True, lcheckVars=None, 
                     lcheckAxis=True, lencl=False, lwrite=False, axis=None, lensembleAxis=False,
                     check_vars=None, exps=None, enses=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  # obviously this only works for datasets that have a time-axis
  # figure out ensemble
#   from projects.WRF_experiments import ensembles, exps, Exp # need to leave this here, to avoid circular reference...
  if ensemble is None and name is not None: 
    ensemble = name; name = None # just switch
  if isinstance(ensemble,(list,tuple)):
    for ens in ensemble:
      if isinstance(ens,Exp) or ( isinstance(ens,str) and ens in exps ) : pass
      else: raise TypeError
    ensemble = [ens if isinstance(ens,Exp) else exps[ens] for ens in ensemble] # convert to Exp's    
    # annotation
    if name is None: name = ensemble[0].shortname
    if title is None: title = ensemble[0].title
  else:
    if isinstance(ensemble,Exp): ensname = ensemble.shortname
    elif isinstance(ensemble,str): 
      # infer domain from suffix...
      if ensemble[-4:-2] == '_d' and int(ensemble[-2:]) > 0:
        domains = int(ensemble[-2:]) # overwrite domains argument
        ensname = ensemble[:-4] 
        name = ensemble if name is None else name # save original name
      else: ensname = ensemble 
      # convert name to experiment object
      if not isinstance(exps,dict): raise DatasetError('No dictionary of Exp instances specified.')
      if ensname in exps: ensemble = exps[ensname]
      else: raise KeyError("Experiment name '{:s}' not found in experiment list.".format(ensname))
    else: raise TypeError
    # annotation (while ensemble is an Exp instance)
    if name is None: name = ensemble.shortname
    if title is None: title = ensemble.title
    # convert name to actual ensemble object
    if not isinstance(enses,dict): raise DatasetError('No dictionary of ensemble tuples specified.')
    if ensname in enses: ensemble = enses[ensname]
    else: raise KeyError("Ensemble name '{:s}' not found in ensemble list.".format(ensname))
  # figure out time period
  if years is None: 
    montpl = (0,180-1)
    warn('Trimming ensemble members to 15 years (time_idx=[0,179]).')
  elif isinstance(years,(list,tuple)) and len(years)==2: 
    if not all([isInt(yr) for yr in years]): raise TypeError(years)
    montpl = (years[0]*12,years[1]*12-1)
  elif isInt(years): montpl = (0,years*12-1)
  else: raise TypeError(years)
  # special treatment for single experiments (i.e. not an ensemble...)
  if not isinstance(ensemble,(tuple,list)):
    if lensembleAxis: raise DatasetError("Wont add singleton ensemble axis to single Dataset!")
    dataset = loadWRF_All(experiment=None, name=ensemble, grid=grid, station=station, shape=shape, 
                          period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                          mode='time-series', lencl=lencl, lautoregrid=lautoregrid, lctrT=lctrT, 
                          lconst=lconst, domains=domains, lwrite=lwrite, check_vars=check_vars)
    # N.B.: passing exps or enses should not be necessary here
  else:
    # load datasets (and load!)
    datasets = []; res = None
    for exp in ensemble:
      #print exp.name
      ds = loadWRF_All(experiment=None, name=exp, grid=grid, station=station, shape=shape, 
                       period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                       mode='time-series', lencl=lencl, lautoregrid=lautoregrid, lctrT=lctrT, 
                       lconst=lconst, domains=domains, lwrite=lwrite, check_vars=check_vars).load()
      if montpl: ds = ds(time=montpl, lidx=True) # slice the time dimension to make things consistent
      if res is None: res = ds.atts['resstr']
      elif res != ds.atts['resstr']: 
        raise DatasetError("Resolution of ensemble members has to be the same: {} != {}".format(res,ds.atts['resstr']))
      datasets.append(ds)
    # harmonize axes
    for axname,ax in ds.axes.items():
      if not all([dataset.hasAxis(axname) for dataset in datasets]): 
        raise AxisError("Not all datasets have Axis '{:s}'.".format(axname))
      if not all([len(dataset.axes[axname]) == len(ax) for dataset in datasets]):
        datasets = selectElements(datasets, axis=axname, testFct=None, master=None, linplace=False, lall=True)
    # concatenate datasets (along 'time' axis, WRF doesn't have 'year')  
    if axis is None: axis = 'ensemble' if lensembleAxis else 'time'
    if lcheckVars is None: lcheckVars = bool(varlist)
    dataset = concatDatasets(datasets, axis=axis, coordlim=None, idxlim=None, offset=None, axatts=None, 
                             lcpOther=True, lcpAny=False, lcheckVars=lcheckVars, lcheckAxis=lcheckAxis,
                             name=name, title=title, lensembleAxis=lensembleAxis, check_vars=check_vars)
  # return concatenated dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
avgfolder # root folder for monthly averages
outfolder # root folder for direct WRF output
ts_file_pattern = 'wrf{0:s}_d{1:02d}{2:s}_monthly.nc' # filename pattern: filetype, domain, grid
clim_file_pattern = 'wrf{0:s}_d{1:02d}{2:s}_clim{3:s}.nc' # filename pattern: filetype, domain, grid, period
data_folder = root_folder # folder for user data
grid_def = {'d02':None,'d01':None} # there are too many... 
grid_res = {'d02':0.13,'d01':3.82} # approximate grid resolution at 45 degrees latitude
default_grid = None 
# functions to access specific datasets
loadLongTermMean = None # WRF doesn't have that...
loadClimatology = loadWRF # pre-processed, standardized climatology
loadTimeSeries = loadWRF_TS # time-series data
loadStationClimatology = loadWRF_Stn # pre-processed, standardized climatology at stations
loadStationTimeSeries = loadWRF_StnTS # time-series data at stations
loadShapeClimatology = loadWRF_Shp # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries = loadWRF_ShpTS # time-series without associated grid (e.g. provinces or basins)


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
  mode = 'test_climatology'
#   mode = 'test_timeseries'
#   mode = 'test_ensemble'
#   mode = 'test_point_climatology'
#   mode = 'test_point_timeseries'
#   mode = 'test_point_ensemble'
  mode = 'pickle_grid' 
#   pntset = 'wcshp'
  pntset = 'glbshp'
#   pntset = 'ecprecip'
#   filetypes = ['srfc','xtrm','plev3d','hydro','lsm','rad']
#   grids = ['glb1-90km','glb1','arb1', 'arb2', 'arb2-120km', 'arb3']
#   domains = [(1,)]+[(1,2)]*3+[(1,),(1,2,3)]; regions = ['GreatLakes']*2+['WesternCanada']*4
  grids = ['arb1', 'arb2', 'arb2-120km', 'arb3']
  domains = [(1,)]+[(1,2)]*2+[(1,),(1,2,3)]; regions = ['WesternCanada']*4
#   grids = ['wc2']; domains = [(1,2)]; regions = ['Columbia']
#   grids = ['arb2-120km']; experiments = ['max-lowres']; domains = [1,]   
    
#   from projects.WesternCanada.WRF_experiments import Exp, WRF_exps, ensembles
  from projects.GreatLakes.WRF_experiments import Exp, WRF_exps, ensembles
  # N.B.: importing Exp through WRF_experiments is necessary, otherwise some isinstance() calls fail
    
  # pickle grid definition
  if mode == 'pickle_grid':
    
    for region,grid,doms in zip(regions,grids,domains):
      
      for domain in doms:
        
        print('')
        res = 'd{0:02d}'.format(domain) # for compatibility with dataset.common
        folder = '{0:s}/{1:s}/'.format(avgfolder,region)
        gridstr = '{0:s}_{1:s}'.format(grid,res) 
        print(('   ***   Pickling Grid Definition for {0:s} Domain {1:d}   ***   '.format(grid,domain)))
        print('')
        
        # load GridDefinition
        
        griddef = getWRFgrid(name=grid, folder=folder, domains=domain, exps=None)[0] # filename='wrfconst_d{0:0=2d}.nc', experiment=experiment
        griddef.name = gridstr
        print(('   Loading Definition from \'{0:s}\''.format(folder)))
#         print(griddef)
        # save pickle
        filepath = pickleGridDef(griddef, lfeedback=True, loverwrite=True, lgzip=True)
        
        print(('   Saving Pickle to \'{0:s}\''.format(filepath)))
        print('')
        
        # load pickle to make sure it is right
        del griddef
        griddef = loadPickledGridDef(grid, res=res, folder=grid_folder)
        print(griddef)
        print('')
    
  # load averaged climatology file
  elif mode == 'test_climatology':
    
    print('')
    dataset = loadWRF(experiment='g-ctrl', domains=2, grid=None, filetypes=['hydro'], 
                      period=(1979,1994), exps=WRF_exps)
#     dataset = loadWRF(experiment='max-ensemble', domains=None, filetypes=['plev3d'], period=(1979,1994),
#                       varlist=['u','qhv','cqwu','cqw','RH'], lconst=True, exps=WRF_exps)
    print(dataset)
#     dataset.lon2D.load()
#     print('')
#     print(dataset.geotransform)
    print('')
    print((dataset.zs))
    var = dataset.zs.getArray()
    print((var.min(),var.mean(),var.std(),var.max()))
  
  # load monthly time-series file
  elif mode == 'test_timeseries':
    
#     dataset = loadWRF_TS(experiment='new-ctrl', domains=2, grid='arb2_d02', filetypes=['srfc'], exps=WRF_exps)
    dataset = loadWRF_TS(experiment='g-ctrl', domains=None, varlist=None, lconst=True,
                         filetypes=['hydro'], exps=WRF_exps)
#     dataset = loadWRF_All(name='new-ctrl-2050', folder='/data/WRF/wrfavg/', domains=2, filetypes=['hydro'], 
#                           lctrT=True, mode='time-series', exps=WRF_exps)
#     for dataset in datasets:
    print('')
    print(dataset)
    print((dataset.name))
    print((dataset.title))
    print((dataset.filelist))
    print('')
    var = dataset.landuse
    print(var)
    print('')
    print((var.dtype, var[:].dtype))
    print((var.min(),var.mean(),var.std(),var.max()))
    print('')
#     print(dataset.time)
#     print(dataset.time.offset)
#     print(dataset.time.coord)

  # load ensemble "time-series"
  elif mode == 'test_ensemble':
    
    print('')
    dataset = loadWRF_Ensemble(ensemble='g-ens', varlist=['precip','MaxPrecip_1d'], filetypes=['hydro'], domains=2, exps=WRF_exps, enses=ensembles)
#     dataset = loadWRF_Ensemble(ensemble=['max-ctrl'], varlist=['precip','MaxPrecip_1d'], filetypes=['xtrm'])
#     dataset = loadWRF_Ensemble(ensemble=['max-ctrl','max-ctrl'], varlist=['precip','MaxPrecip_1d'], filetypes=['xtrm'])
    # 2.03178e-05 0.00013171
    print('')
    print(dataset)
    print((dataset.name))
    print((dataset.title))
#     print('')
#     print(dataset.precip.mean())
#     print(dataset.MaxPrecip_1d.mean())
#   print('')
#     print(dataset.time)
#     print(dataset.time.coord)
  

  # load station climatology file
  elif mode == 'test_point_climatology':
    
    print('')
    if pntset in ('shpavg','wcshp','glbshp','glakes'):
      dataset = loadWRF_Shp(experiment='erai-g3', domains=1, shape=pntset, grid='grw2', period=(1979,1994), 
                            filetypes=['aux'],exps=WRF_exps)
      print('')
      print((dataset.shape))
      print((dataset.shape.coord))
    else:
      dataset = loadWRF_Stn(experiment='erai-g', domains=None, station=pntset, filetypes=['hydro'], period=(1979,1984), exps=WRF_exps)
      zs_err = dataset.zs.getArray() - dataset.stn_zs.getArray()
      print((zs_err.min(),zs_err.mean(),zs_err.std(),zs_err.max()))
#       print('')
#       print(dataset.station)
#       print(dataset.station.coord)
    dataset.load()
    print('')
    print(dataset)
    print('')
    print('')
    print((dataset.pet_wrf))
    print('')
    print((dataset.pet_wrf.mean()))
    
  # load station time-series file
  elif mode == 'test_point_timeseries':
    
    print('')
    if pntset in ('shpavg','wcshp','glbshp','glakes'):
      dataset = loadWRF_ShpTS(experiment='g-ctrl-2100', domains=1, varlist=None, #['zs','stn_zs','precip','MaxPrecip_1d','wetfrq_010'], 
                              shape=pntset, filetypes=['aux','hydro'], exps=WRF_exps,)
    else:
      dataset = loadWRF_StnTS(experiment='erai-g', domains=None, varlist=['zs','stn_zs','MaxPrecip_6h'],
#                               varlist=['zs','stn_zs','precip','MaxPrecip_6h','MaxPreccu_1h','MaxPrecip_1d'], 
                              station=pntset, filetypes=['srfc'], exps=WRF_exps)
      zs_err = dataset.zs.getArray() - dataset.stn_zs.getArray()
      print((zs_err.min(),zs_err.mean(),zs_err.std(),zs_err.max()))
    print('')
    print(dataset)
    print('')
    print((dataset.time))
    print((dataset.time.offset))
    print((dataset.time.coord))
    print('')
    for name in dataset.shape_name[:]: print(name)
    print('')
    print((dataset.pet.mean()))
    print('')
    print((dataset.pet_wrf.mean()))    
  
  # load station ensemble "time-series"
  elif mode == 'test_point_ensemble':
    lensembleAxis = False
    print('')
    if pntset in ('shpavg','wcshp','glbshp','glakes'):
#       dataset = loadWRF_ShpEns(ensemble=['max-ctrl','max-ens-A'], shape=pntset, domains=None, filetypes=['hydro','srfc'])
      dataset = loadWRF_ShpEns(ensemble='g-ens', shape=pntset, varlist=['precip','runoff'], domains=2, 
                               filetypes=['srfc','lsm',], lensembleAxis=lensembleAxis, exps=WRF_exps, enses=ensembles)
    else:
      dataset = loadWRF_StnEns(ensemble='max-ens-2100', station=pntset, lensembleAxis=lensembleAxis,  
                               varlist=['MaxPrecip_6h'], filetypes=['srfc'], exps=WRF_exps, enses=ensembles)
    assert not lensembleAxis or dataset.hasAxis('ensemble')
    dataset.load()
    print('')
    print(dataset)
#     print('')
#     print(dataset.precip.mean())
#     print(dataset.MaxPrecip_1d.mean())
#     print('')
#     print('')
#     print(dataset.time)
#     print(dataset.time.coord)
  
