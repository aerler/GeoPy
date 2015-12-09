'''
Created on 2013-09-28

This module contains common meta data and access functions for WRF model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc
import collections as col
import os, pickle
import osr
# from atmdyn.properties import variablePlotatts
from datasets.EC import nullNaN
from wrfavg.derived_variables import precip_thresholds
from geodata.base import concatDatasets
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition, GDALError
from geodata.misc import DatasetError, AxisError, DateError, ArgumentError, isNumber, isInt,\
  EmptyDatasetError
from datasets.common import translateVarNames, data_root, grid_folder, selectElements,\
  stn_params, shp_params
from geodata.gdal import loadPickledGridDef, griddef_pickle
from projects.WRF_experiments import Exp, exps, ensembles 
from warnings import warn


## get WRF projection and grid definition 
# N.B.: Unlike with observational datasets, model Meta-data depends on the experiment and has to be 
#       loaded from the NetCFD-file; a few conventions have to be defined, however.

# get projection NetCDF attributes
def getWRFproj(dataset, name=''):
  ''' Method to infer projection parameters from a WRF output file and return a GDAL SpatialReference object. '''
  if not isinstance(dataset,nc.Dataset): raise TypeError
  if not isinstance(name,basestring): raise TypeError
  if dataset.MAP_PROJ == 1: 
    # Lambert Conformal Conic projection parameters
    proj = 'lcc' # Lambert Conformal Conic  
    lat_1 = dataset.TRUELAT1 # Latitude of first standard parallel
    lat_2 = dataset.TRUELAT2 # Latitude of second standard parallel
    lat_0 = dataset.CEN_LAT # Latitude of natural origin
    lon_0 = dataset.STAND_LON # Longitude of natural origin
    lon_1 = dataset.CEN_LON # actual center of map
  else:
    raise NotImplementedError, "Can only infer projection parameters for Lambert Conformal Conic projection (#1)."
  projdict = dict(proj=proj,lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0,lon_1=lon_1)
  # pass results to GDAL module to get projection object
  return getProjFromDict(projdict, name=name, GeoCS='WGS84', convention='Proj4')  

# infer grid (projection and axes) from constants file
def getWRFgrid(name=None, experiment=None, domains=None, folder=None, filename='wrfconst_d{0:0=2d}.nc', ncformat='NETCDF4'):
  ''' Infer the WRF grid configuration from an output file and return a GridDefinition object. '''
  # check input
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, folder=folder)
  if isinstance(filename,basestring): filepath = '{}/{}'.format(folder,filename) # still contains formaters
  else: raise TypeError, filename
  # figure out experiment
  if experiment is None:
    if isinstance(name,basestring) and name in exps: experiment = exps[name]
    elif len(names) > 0: 
      tmp = names[0].split('_')[0]
      if tmp in exps: experiment = exps[tmp]
  elif not isinstance(experiment,Exp): raise TypeError  
  maxdom = max(domains) # max domain
  # files to work with
  for n in xrange(1,maxdom+1):
    dnfile = filepath.format(n,'') # expects grid name as well, but we are only looking for the native grid
    if not os.path.exists(dnfile):
      if n in domains: raise IOError, 'File {} for domain {:d} not found!'.format(dnfile,n)
      else: raise IOError, 'File {} for domain {:d} not found; this file is necessary to infer the geotransform for other domains.'.format(dnfile,n)
  # open first domain file (special treatment)
  dn = nc.Dataset(filepath.format(1,''), mode='r', format=ncformat)
  gridname = experiment.grid if isinstance(experiment,Exp) else name # use experiment name as default
  projection = getWRFproj(dn, name=gridname) # same for all
  # get coordinates of center point  
  clon = dn.CEN_LON; clat = dn.CEN_LAT
  wgs84 = osr.SpatialReference (); wgs84.ImportFromEPSG (4326) # regular lat/lon geographic grid
  tx = osr.CoordinateTransformation (wgs84, projection) # transformation object
  cx, cy, cz = tx.TransformPoint(float(clon),float(clat)); del cz # center point in projected (WRF) coordinates
  #print ' (CX,CY,CZ) = ', cx, cy, cz 
  # infer size and geotransform
  def getXYlen(ds):
    ''' a short function to infer the length of horizontal axes from a dataset with unknown naming conventions ''' 
    if 'west_east' in ds.dimensions and 'south_north' in ds.dimensions:
      nx = len(ds.dimensions['west_east']); ny = len(ds.dimensions['south_north'])
    elif 'x' in ds.dimensions and 'y' in ds.dimensions:
      nx = len(ds.dimensions['x']); ny = len(ds.dimensions['y'])
    else: raise AxisError, 'No horizontal axis found, necessary to infer projection/grid configuration.'
    return nx,ny
  dx = float(dn.DX); dy = float(dn.DY)
  nx,ny = getXYlen(dn)
  x0 = -float(nx+1)*dx/2.; y0 = -float(ny+1)*dy/2.
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
    for n in xrange(2,maxdom+1):
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
def getFolderNameDomain(name=None, experiment=None, domains=None, folder=None, lexp=False):
  ''' Convenience function to infer and type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  if name is None and experiment is None: raise ArgumentError
  if folder is None and experiment is None: raise ArgumentError 
  # handle experiment
  if experiment is not None and name is None:
    if isinstance(experiment,Exp):
      name = experiment.name
    elif isinstance(experiment,basestring) or isinstance(name,(list,tuple)): 
      name = experiment # use name-logic to procede
      experiment = None # eventually reset below
    else: TypeError
  ## figure out experiment name(s) and domain(s) (potentially based on name)
  # name check
  if isinstance(name,(list,tuple)) and all([isinstance(n,basestring) for n in name]): names = name
  elif isinstance(name,basestring): names = [name]
  else: raise TypeError
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
  if len(set(basenames)) != 1: raise DatasetError, "Dataset base names are inconsistent."
  name = basenames[0]
  # ensure uniqueness of names
  if len(set(names)) != len(names):
    names = ['{0:s}_d{1:0=2d}'.format(name,domain) for name,domain in zip(names,domains)]
  # evaluate experiment
  if experiment is None: 
    if name in exps: experiment = exps[name] # load experiment meta data
    elif lexp: raise DatasetError, 'Dataset of name \'{0:s}\' not found!'.format(names[0])
  # assign unassigned domains
  domains = [experiment.domains if dom is None else dom for dom in domains]
  # patch up folder
  if folder is None: # should already have checked that either folder or experiment are specified
    folder = experiment.avgfolder
  elif isinstance(folder,basestring): 
    if not folder.endswith((name,name+'/')): folder = '{:s}/{:s}/'.format(folder,name)
  else: raise TypeError
  # check types
  if not isinstance(domains,(tuple,list)): raise TypeError    
  if not all(isInt(domains)): raise TypeError
  if not domains == sorted(domains): raise IndexError, 'Domains have to be sorted in ascending order.'
  if not isinstance(names,(tuple,list)): raise TypeError
  if not all(isinstance(nm,basestring) for nm in names): raise TypeError
  if len(domains) != len(names): raise ArgumentError  
  # check if folder exists
  if not os.path.exists(folder): raise IOError, folder
  # return name and folder
  return folder, experiment, tuple(names), tuple(domains)
  
## variable attributes and name
# convert water mass mixing ratio to water vapor partial pressure ( kg/kg -> Pa ) 
Q = 96000.*28./18. # surface pressure * molecular weight ratio ( air / water )
class FileType(object): pass # ''' Container class for all attributes of of the constants files. '''
# constants
class Const(FileType):
  ''' Variables and attributes of the constants files. '''
  def __init__(self):
    self.name = 'const' 
    self.atts = dict(HGT    = dict(name='zs', units='m'), # surface elevation
                     XLONG  = dict(name='lon2D', units='deg E'), # geographic longitude field
                     XLAT   = dict(name='lat2D', units='deg N'), # geographic latitude field
                     SINALPHA = dict(name='sina', units=''), # sine of map rotation
                     COSALPHA = dict(name='cosa', units='')) # cosine of map rotation                   
    self.vars = self.atts.keys()    
    self.climfile = None #'wrfconst_d{0:0=2d}{1:s}.nc' # the filename needs to be extended by (domain,'_'+grid)
    self.tsfile = 'wrfconst_d{0:0=2d}.nc' # the filename needs to be extended by (domain,)
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
                     OLR          = dict(name='OLR', units='W/m^2'), # Outgoing Longwave Radiation
                     GLW          = dict(name='GLW', units='W/m^2'), # Ground Longwave Radiation
                     SWDOWN       = dict(name='SWD', units='W/m^2'), # Downwelling Shortwave Radiation
                     SWNORM       = dict(name='SWN', units='W/m^2'), # Downwelling Normal Shortwave Radiation
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip = dict(name='liqprec_sr', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip  = dict(name='solprec_sr', units='kg/m^2/s'), # solid precipitation rate
                     WaterVapor   = dict(name='Q2', units='Pa'), # water vapor partial pressure
                     #WetDays      = dict(name='wetfrq', units=''), # fraction of wet/rainy days 
                     #WetDayRain   = dict(name='dryprec', units='kg/m^2/s'), # precipitation rate above dry-day threshold (kg/m^2/s)
                     #WetDayPrecip = dict(name='wetprec', units='kg/m^2/s'), # wet-day precipitation rate (kg/m^2/s)                     MaxRAIN      = dict(name='MaxPrecip_6h', units='kg/m^2/s'), # maximum 6-hourly precip                    
                     MaxACSNOW    = dict(name='MaxSolprec_1d', units='kg/m^2/s'), # maximum daily precip
                     MaxACSNOW_1d = dict(name='MaxSolprec_1d', units='kg/m^2/s'), # maximum daily precip                             
                     MaxACSNOW_5d = dict(name='MaxSolprec_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
                     MaxRAIN      = dict(name='MaxPrecip_6h', units='kg/m^2/s'), # maximum 6-hourly precip                    
                     MaxRAINC     = dict(name='MaxPreccu_6h', units='kg/m^2/s'), # maximum 6-hourly convective precip
                     MaxRAINNC    = dict(name='MaxPrecnc_6h', units='kg/m^2/s'), # maximum 6-hourly non-convective precip
                     MaxPrecip    = dict(name='MaxPrecip_6h', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu    = dict(name='MaxPreccu_6h', units='kg/m^2/s'), # for short-term consistency
                     MaxPrecnc    = dict(name='MaxPrecnc_6h', units='kg/m^2/s'), # for short-term consistency
                     MaxRAIN_1d   = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAINC_1d  = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxRAINNC_1d = dict(name='MaxPrecnc_1d', units='kg/m^2/s'), # maximum daily non-convective precip
                     MaxPrecip_1d = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu_1d = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxPrecnc_1d = dict(name='MaxPrecnc_1d', units='kg/m^2/s')) # for short-term consistency
    for threshold in precip_thresholds: # add variables with different wet-day thresholds
        suffix = '_{:03d}'.format(int(10*threshold))
        self.atts['WetDays'+suffix]      = dict(name='wetfrq'+suffix, units='') # fraction of wet/rainy days                    
        self.atts['WetDayRain'+suffix]   = dict(name='dryprec'+suffix, units='kg/m^2/s') # precipitation rate above dry-day thre
        self.atts['WetDayPrecip'+suffix] = dict(name='wetprec'+suffix, units='kg/m^2/s', 
                                                atts=dict(fillValue=0), transform=nullNaN) # wet-day precipitation rate (kg/m^2/s)
    self.vars = self.atts.keys()    
    self.climfile = 'wrfsrfc_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfsrfc_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# hydro variables
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
                     POTEVP       = dict(name='pet', units='kg/m^2/s', scalefactor=999.70), # potential evapo-transpiration rate
                     pet          = dict(name='pet', units='kg/m^2/s', scalefactor=2./3.), # correction for pre-processed PET
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation rate
                     NetWaterFlux = dict(name='waterflx', units='kg/m^2/s'), # total water downward flux
                     #WetDays      = dict(name='wetfrq', units=''), # fraction of wet/rainy days 
                     #WetDayRain   = dict(name='dryprec', units='kg/m^2/s'), # precipitation rate above dry-day threshold (kg/m^2/s)
                     #WetDayPrecip = dict(name='wetprec', units='kg/m^2/s'), # wet-day precipitation rate (kg/m^2/s)
                     MaxNetWaterFlux = dict(name='MaxWaterFlx_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxPrecip    = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPrecnc    = dict(name='MaxPrecnc_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxPreccu    = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxRAIN      = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAIN_5d   = dict(name='MaxPrecip_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
                     MaxACSNOW    = dict(name='MaxSolprec_1d', units='kg/m^2/s'), # maximum daily precip
                     MaxACSNOW_5d = dict(name='MaxSolprec_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
                     MaxRAINC     = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxRAINNC    = dict(name='MaxPrecnc_1d', units='kg/m^2/s'), # maximum daily non-convective precip
                     #MaxACSNOW    = dict(name='MaxSnow_1d', units='kg/m^2/s'), # maximum daily snow fall
                     #MaxACSNOW_5d = dict(name='MaxSnow_5d', units='kg/m^2/s'), # maximum pendat (5 day) snow
                     MaxPrecip_1d = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu_1d = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # for short-term consistency
                     MaxPrecnc_1d = dict(name='MaxPrecnc_1d', units='kg/m^2/s'),) # for short-term consistency                     
    for threshold in precip_thresholds: # add variables with different wet-day thresholds
        suffix = '_{:03d}'.format(int(10*threshold))
        self.atts['WetDays'+suffix]      = dict(name='wetfrq'+suffix, units='') # fraction of wet/rainy days                    
        self.atts['WetDayRain'+suffix]   = dict(name='dryprec'+suffix, units='kg/m^2/s') # precipitation rate above dry-day thre
        self.atts['WetDayPrecip'+suffix] = dict(name='wetprec'+suffix, units='kg/m^2/s', 
                                                atts=dict(fillValue=0), transform=nullNaN) # wet-day precipitation rate (kg/m^2/s)
    self.vars = self.atts.keys()
    self.climfile = 'wrfhydro_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfhydro_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# lsm variables
class LSM(FileType):
  ''' Variables and attributes of the land surface files. '''
  def __init__(self):
    self.name = 'lsm'
    self.atts = dict(ALBEDO = dict(name='A', units=''), # Albedo
                     SNOWC  = dict(name='snwcvr', units=''), # snow cover (binary)
                     ACSNOM = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate 
                     ACSNOW = dict(name='snwacc', units='kg/m^2/s'), # snow accumulation rate
                     SFCEVP = dict(name='evap', units='kg/m^2/s'), # actual surface evaporation/ET rate
                     POTEVP = dict(name='pet', units='kg/m^2/s', scalefactor=999.70), # potential evapo-transpiration rate
                     pet    = dict(name='pet', units='kg/m^2/s', scalefactor=2./3.), # correction for pre-processed PET
                     SFROFF = dict(name='sfroff', units='kg/m^2/s'), # surface run-off
                     UDROFF = dict(name='ugroff', units='kg/m^2/s'), # sub-surface/underground run-off
                     Runoff = dict(name='runoff', units='kg/m^2/s'), # total surface and sub-surface run-off
                     SMOIS  = dict(name='aSM', units='m^3/m^3'), # absolute soil moisture
                     SMCREL = dict(name='rSM', units='')) # relative soil moisture
    self.vars = self.atts.keys()    
    self.climfile = 'wrflsm_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrflsm_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# lsm variables
class Rad(FileType):
  ''' Variables and attributes of the radiation files. '''
  def __init__(self):
    self.name = 'rad'
    self.atts = dict() # currently empty
    self.vars = self.atts.keys()    
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
                     #SKINTEMPMEAN  = dict(name='TSmean', units='K'),  # daily mean Skin Temperature
                     SKINTEMPMEAN  = dict(name='Ts', units='K'),  # daily mean Skin Temperature
                     SKINTEMPMIN   = dict(name='TSmin', units='K'),   # daily minimum Skin Temperature
                     SKINTEMPMAX   = dict(name='TSmax', units='K'),   # daily maximum Skin Temperature
                     SKINTEMPSTD   = dict(name='TSstd', units='K'),   # daily Skin Temperature standard deviation                     
                     #Q2MEAN        = dict(name='Qmean', units='Pa', scalefactor=Q), # daily mean Water Vapor Pressure (at 2m)
                     Q2MEAN        = dict(name='Q2', units='Pa', scalefactor=Q), # daily mean Water Vapor Pressure (at 2m)
                     Q2MIN         = dict(name='Qmin', units='Pa', scalefactor=Q),  # daily minimum Water Vapor Pressure (at 2m)
                     Q2MAX         = dict(name='Qmax', units='Pa', scalefactor=Q),  # daily maximum Water Vapor Pressure (at 2m)
                     Q2STD         = dict(name='Qstd', units='Pa', scalefactor=Q),  # daily Water Vapor Pressure standard deviation (at 2m)
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
    self.vars = self.atts.keys()    
    self.climfile = 'wrfxtrm_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfxtrm_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# variables on selected pressure levels: 850 hPa, 700 hPa, 500 hPa, 250 hPa, 100 hPa
class Plev3D(FileType):
  ''' Variables and attributes of the pressure level files. '''
  def __init__(self):
    self.name = 'plev3d'
    self.atts = dict(T_PL      = dict(name='T', units='K', fillValue=-999),      # Temperature
                     TD_PL     = dict(name='Td', units='K', fillValue=-999),     # Dew-point Temperature
                     RH_PL     = dict(name='RH', units='\%', fillValue=-999),     # Relative Humidity
                     GHT_PL    = dict(name='Z', units='m', fillValue=-999),      # Geopotential Height 
                     S_PL      = dict(name='U', units='m/s', fillValue=-999),    # Wind Speed
                     U_PL      = dict(name='u', units='m/s', fillValue=-999),    # Zonal Wind Speed
                     V_PL      = dict(name='v', units='m/s', fillValue=-999),    # Meridional Wind Speed
                     WaterFlux_U = dict(name='qwu', units='kg/m^2/s', fillValue=-999), # zonal water (vapor) flux
                     WaterFlux_V = dict(name='qwv', units='kg/m^2/s', fillValue=-999), # meridional water (vapor) flux
                     WaterTransport_U = dict(name='cqwu', units='kg/m/s', fillValue=-999), # column-integrated zonal water (vapor) transport
                     WaterTransport_V = dict(name='cqwv', units='kg/m/s', fillValue=-999), # column-integrated meridional water (vapor) transport
                     ColumnWater = dict(name='cqw', units='kg/m^2', fillValue=-999), # column-integrated water (vapor) content
                     HeatFlux_U = dict(name='qhu', units='J/m^2/s', fillValue=-999), # zonal heat flux
                     HeatFlux_V = dict(name='qhv', units='J/m^2/s', fillValue=-999), # meridional heat flux
                     Heatransport_U = dict(name='cqhu', units='J/m/s', fillValue=-999), # column-integrated zonal heat transport
                     Heatransport_V = dict(name='cqhv', units='J/m/s', fillValue=-999), # column-integrated meridional heat transport
                     ColumnHeat = dict(name='cqh', units='J/m^2', fillValue=-999), # column-integrated heat content
                     Vorticity = dict(name='zeta', units='1/s', fillValue=-999)) # (relative) Vorticity
#                      P_PL     = dict(name='p', units='Pa'))  # Pressure
    self.vars = self.atts.keys()    
    self.climfile = 'wrfplev3d_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfplev3d_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)

# axes (don't have their own file)
class Axes(FileType):
  ''' A mock-filetype for axes. '''
  def __init__(self):
    self.name = 'axes'
    self.atts = dict(Time        = dict(name='time', units='month'), # time coordinate
                     time        = dict(name='time', units='month'), # time coordinate
                     # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                     #       the time offset is chose such that 1979 begins with the origin (time=0)
                     west_east   = dict(name='x', units='m'), # projected west-east coordinate
                     south_north = dict(name='y', units='m'), # projected south-north coordinate
                     x           = dict(name='x', units='m'), # projected west-east coordinate
                     y           = dict(name='y', units='m'), # projected south-north coordinate
                     soil_layers_stag = dict(name='s', units=''), # soil layer coordinate
                     num_press_levels_stag = dict(name='p', units='Pa'), # pressure coordinate
                     station     = dict(name='station', units='#') ) # station axis for station data
    self.vars = self.atts.keys()
    self.climfile = None
    self.tsfile = None

# data source/location
fileclasses = dict(const=Const(), srfc=Srfc(), hydro=Hydro(), lsm=LSM(), rad=Rad(), xtrm=Xtrm(), plev3d=Plev3D(), axes=Axes())
root_folder = data_root + 'WRF/' # long-term mean folder
outfolder = root_folder + 'wrfout/' # WRF output folder
avgfolder = root_folder + 'wrfavg/' # long-term mean folder

# add generic extremes to varatts dicts
for fileclass in fileclasses.itervalues():
  atts = dict()
  for key,val in fileclass.atts.iteritems():
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


## Functions to load different types of WRF datasets

# Station Time-series (monthly, with extremes)
def loadWRF_StnTS(experiment=None, name=None, domains=None, station=None, filetypes=None, 
                  varlist=None, varatts=None, lctrT=True, lwrite=False, ltrimT=True):
  ''' Get a properly formatted WRF dataset with monthly time-series at station locations. '''  
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=None, station=station, 
                     period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                     lconst=False, lautoregrid=False, lctrT=lctrT, mode='time-series', 
                     lwrite=lwrite, ltrimT=ltrimT, check_vars='station_name')  

# Regiona/Shape Time-series (monthly, with extremes)
def loadWRF_ShpTS(experiment=None, name=None, domains=None, shape=None, filetypes=None, 
                  varlist=None, varatts=None, lctrT=True, lencl=False, lwrite=False, ltrimT=True):
  ''' Get a properly formatted WRF dataset with monthly time-series averaged over regions. '''  
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=None, shape=shape, lencl=lencl, 
                     station=None, period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                     lconst=False, lautoregrid=False, lctrT=lctrT, mode='time-series', lwrite=lwrite, 
                     ltrimT=ltrimT, check_vars='shape_name')  

def loadWRF_TS(experiment=None, name=None, domains=None, grid=None, filetypes=None, varlist=None, 
               varatts=None, lconst=True, lautoregrid=True, lctrT=True, lwrite=False, ltrimT=True):
  ''' Get a properly formatted WRF dataset with monthly time-series. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=None, 
                     period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=lautoregrid, lctrT=lctrT, mode='time-series', lwrite=lwrite, ltrimT=ltrimT)  

def loadWRF_Stn(experiment=None, name=None, domains=None, station=None, period=None, filetypes=None, 
                varlist=None, varatts=None, lctrT=True, lwrite=False, ltrimT=False):
  ''' Get a properly formatted station dataset from a monthly WRF climatology at station locations. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=None, station=station, 
                     period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=False, 
                     lautoregrid=False, lctrT=lctrT, mode='climatology', lwrite=lwrite, ltrimT=ltrimT,
                     check_vars='station_name')  

def loadWRF_Shp(experiment=None, name=None, domains=None, shape=None, period=None, filetypes=None, 
                varlist=None, varatts=None, lctrT=True, lencl=False, lwrite=False, ltrimT=False):
  ''' Get a properly formatted station dataset from a monthly WRF climatology averaged over regions. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=None, shape=shape, lencl=lencl,
                     station=None, period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                     lconst=False, lautoregrid=False, lctrT=lctrT, mode='climatology', lwrite=lwrite, 
                     ltrimT=ltrimT, check_vars='shape_name')  

def loadWRF(experiment=None, name=None, domains=None, grid=None, period=None, filetypes=None, varlist=None, 
            varatts=None, lconst=True, lautoregrid=True, lctrT=True, lwrite=False, ltrimT=False):
  ''' Get a properly formatted monthly WRF climatology as NetCDFDataset. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=None, 
                     period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=lautoregrid, lctrT=lctrT, mode='climatology', lwrite=lwrite, ltrimT=ltrimT)  

# pre-processed climatology files (varatts etc. should not be necessary) 
def loadWRF_All(experiment=None, name=None, domains=None, grid=None, station=None, shape=None, 
                period=None, filetypes=None, varlist=None, varatts=None, lconst=True, lautoregrid=True, 
                lencl=False, lctrT=False, folder=None, lpickleGrid=True, mode='climatology', 
                lwrite=False, ltrimT=False, check_vars=None):
  ''' Get any WRF data files as a properly formatted NetCDFDataset. '''
  # prepare input  
  ltuple = isinstance(domains,col.Iterable)  
  # prepare input  
  if experiment is None and name is not None: 
    experiment = name; name=None # allow 'name' to define an experiment  
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, folder=folder)
  if lctrT and experiment is None: 
    raise DatasetError, "Experiment '{0:s}' not found in database; need time information to center time axis.".format(names[0])    
  # figure out period
  if isinstance(period,(tuple,list)):
    if not all(isNumber(period)): raise ValueError
  elif isinstance(period,basestring): period = [int(prd) for prd in period.split('-')]
  elif isinstance(period,(int,np.integer)): 
    beginyear = int(experiment.begindate[0:4])
    period = (beginyear, beginyear+period)
  elif period is None: pass # handled later
  else: raise DateError, "Illegal period definition: {:s}".format(str(period))
  lclim = False; lts = False # mode switches
  if mode.lower() == 'climatology': # post-processed climatology files
    lclim = True
    periodstr = '_{0:4d}-{1:4d}'.format(*period)
    if period is None: raise DateError, 'Currently WRF Climatologies have to be loaded with the period explicitly specified.'
  elif mode.lower() in ('time-series','timeseries'): # concatenated time-series files
    lts = True; lclim = False; period = None; periodstr = None # to indicate time-series (but for safety, the input must be more explicit)
    if lautoregrid is None: lautoregrid = False # this can take very long!
  # cast/copy varlist
  if isinstance(varlist,basestring): varlist = [varlist] # cast as list
  elif varlist is not None: varlist = list(varlist) # make copy to avoid interference
  # figure out station and shape options
  if station and shape: raise ArgumentError
  elif station or shape: 
    if grid is not None: raise NotImplementedError, 'Currently WRF station data can only be loaded from the native grid.'
    if lautoregrid: raise GDALError, 'Station data can not be regridded, since it is not map data.'   
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
  elif isinstance(filetypes,basestring): filetypes = [filetypes,]
  elif not isinstance(filetypes,list): raise TypeError  
  if 'axes' not in filetypes: filetypes.append('axes')
  #if 'const' not in filetypes and grid is None: filetypes.append('const')
  atts = dict(); filelist = []; typelist = []
  for filetype in filetypes: # last filetype in list has precedence
    fileclass = fileclasses[filetype]    
    if lclim and fileclass.climfile is not None:
      filelist.append(fileclass.climfile)
      typelist.append(filetype) # this eliminates const files
    elif lts: 
      if fileclass.tsfile is not None: 
        filelist.append(fileclass.tsfile)
        typelist.append(filetype) # this eliminates const files
#       if not lstation and grid is None: 
    atts.update(fileclass.atts) # only for original time-series      
  if varatts is not None: atts.update(varatts)
  # NetCDF file mode
  ncmode = 'rw' if lwrite else 'r' 
  # center time axis to 1979
  if lctrT and experiment is not None:
    if 'time' in atts: tatts = atts['time']
    else: tatts = dict()
    ys,ms,ds = [int(t) for t in experiment.begindate.split('-')]; assert ds == 1   
    tatts['offset'] = (ys-1979)*12 + (ms-1)
    tatts['atts'] = dict(long_name='Month since 1979-01')
    atts['time'] = tatts   
  # translate varlist
  if varlist is not None: varlist = translateVarNames(varlist, atts) # default_varatts
  # infer projection and grid and generate horizontal map axes
  # N.B.: unlike with other datasets, the projection has to be inferred from the netcdf files  
  if not lstation and not lshape:
    if grid is None:
      try:
        # load pickled griddefs from disk (much faster than recomputing!)
        if not lpickleGrid or experiment is None: raise IOError # don't load pickle!
        griddefs = []
        for domain in domains:
          # different "native" grid for each domain
          griddefs.append( loadPickledGridDef(grid=experiment.grid, res='d0{:d}'.format(domain), 
                                              filename=None, folder=grid_folder, check=True) )
        # this is mainly to speed up loading datasets
      except:
        # print warning to alert user that this takes a bit longer
        if experiment is not None:
          name = "'{:s}' ('{:s}')".format(experiment.name,experiment.grid) 
        else: name = "'{:s}'".format(names[0])        
        warn("Recomputing Grid Definition for Experiment {:s}".format(name))
        # compute grid definition from wrfconst files (requires all parent domains) 
        griddefs = None; c = 0
        filename = fileclasses.values()[c].tsfile # just use the first filetype
        while griddefs is None:
          # some experiments do not have all files... try, until one works...
          try:
            if filename is None: raise IOError # skip and try next one
            griddefs = getWRFgrid(name=names, experiment=experiment, domains=domains, folder=folder, filename=filename)
          except IOError:
            c += 1
            filename = fileclasses.values()[c].tsfile
    else:
      griddefs = [loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder, check=True)]*len(domains)
    assert len(griddefs) == len(domains)
  else:
    griddefs = [None]*len(domains) # not actually needed
  # grid
  datasets = []
  for name,domain,griddef in zip(names,domains,griddefs):
#     if grid is None or grid.split('_')[0] == experiment.grid: gridstr = ''
    if lstation:
      # the station name can be inserted as the grid name
      gridstr = '_'+station.lower(); # only use lower case for filenames
      llconst = False # don't load constants (some constants are already in the file anyway)
      axes = None
    elif lshape:
      # the shape collection name can be inserted as the grid name
      gridstr = '_'+shape.lower(); # only use lower case for filenames
      llconst = False # don't load constants (some constants are already in the file anyway)
      axes = None
    else:
      if grid is None or grid == '{0:s}_d{1:02d}'.format(experiment.grid,domain): 
          gridstr = ''; llconst = lconst
      else: 
        gridstr = '_'+grid.lower(); # only use lower case for filenames
        llconst = False # don't load constants     
      # domain-sensitive parameters
      axes = dict(west_east=griddef.xlon, south_north=griddef.ylat, x=griddef.xlon, y=griddef.ylat) # map axes
    # load constants
    if llconst:              
      constfile = fileclasses['const']    
      filename = constfile.tsfile.format(domain,gridstr)  
      # check file path
      constfolder = folder
      if not os.path.exists('{:s}/{:s}'.format(constfolder,filename)) and experiment and experiment.grid:
          constfolder = folder[:folder.find(experiment.name)] + '{:s}/'.format(experiment.grid)
      if not os.path.exists('{:s}/{:s}'.format(constfolder,filename)):
        raise IOError, "Constant file for experiment '{:s}' not found.\n('{:s}')".format(name,'{:s}/{:s}'.format(constfolder,filename))
      # i.e. if there is no constant file in the experiment folder, use the one in the grid definition folder (if available)
      # load dataset
      const = DatasetNetCDF(name=name, folder=constfolder, filelist=[filename], varatts=constfile.atts, 
                            axes=axes, varlist=constfile.vars, multifile=False, ncformat='NETCDF4', 
                            mode=ncmode, squeeze=True)      
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
            print("The '{:s}' (WRF) dataset for the grid ('{:s}') is not available:\n Attempting regridding on-the-fly.".format(name,filename,grid))
            if performRegridding('WRF', 'climatology', griddef, dataargs): # True if exitcode 1
              raise IOError, "Automatic regridding failed!"
            print("Output: '{:s}'".format(name,filename,grid,filepath))            
          else: raise IOError, "The  '{:s}' (WRF) dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.\n('{:s}')".format(name,filename,grid,filepath) 
        else: raise IOError, "The file '{:s}' in WRF dataset '{:s}' does not exits!\n('{:s}')".format(filename,name,filepath)   
       
    # load dataset
    check_override = ['time'] if lctrT else None
    try:
      dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=axes, 
                              varatts=atts, multifile=False, ncformat='NETCDF4', mode=ncmode, 
                              squeeze=True, check_override=check_override, check_vars=check_vars)
    except EmptyDatasetError:
      if lenc == 0: raise # allow loading of cosntants without other variables
    if ltrimT and dataset.hasAxis('time') and len(dataset.time) > 180:
      if lwrite: raise ArgumentError, "Cannot trim time-axis when NetCDF write mode is enabled!"
      dataset = dataset(time=slice(0,180), lidx=True)
      assert len(dataset.time) == 180, len(dataset.time) 
    # check
    if (len(dataset)+lenc) == 0: raise DatasetError, 'Dataset is empty - check source file or variable list!'
    # check time axis and center at 1979-01 (zero-based)
    if lctrT and dataset.hasAxis('time'):
      # N.B.: we can directly change axis vectors, since they reside in memory;
      #       the source file is not changed, unless sync() is called
      if lts:
        dataset.time.coord = np.arange(len(dataset.time)) + (ys-1979)*12 + (ms-1)
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
        dataset.time.coord = np.arange(len(dataset.time)) + 1
        # N.B.: shifting is dangerous, because of potential repeated application
#         t0 = dataset.time.coord[0]
#         # there is no "year" - just start with "1" for january 
#         if t0 != 1: 
#           dataset.time.coord -= ( t0 - 1 )
#           dataset.time.offset -= ( t0 - 1 )
      # correct ordinal number of shape (should start at 1, not 0)
    if lshape:
      # mask all shapes that are incomplete in dataset
      if lencl and 'shp_encl' in dataset: dataset.mask(mask='shp_encl', invert=True)
      if dataset.hasAxis('shapes'): raise AxisError, "Axis 'shapes' should be renamed to 'shape'!"
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
    # append to list
    datasets.append(dataset) 
  # return formatted dataset
  if not ltuple: datasets = datasets[0]
  else: tuple(datasets)
  return datasets


# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_StnEns(ensemble=None, name=None, station=None, filetypes=None, years=None, domains=None, 
                   varlist=None, title=None, varatts=None, translateVars=None, lcheckVars=None, 
                   lcheckAxis=True, lwrite=False, axis=None, lensembleAxis=False):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadWRF_Ensemble(ensemble=ensemble, grid=None, station=station, domains=domains, 
                          filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, 
                          translateVars=translateVars, lautoregrid=False, lctrT=True, lconst=False,
                          lcheckVars=lcheckVars, lcheckAxis=lcheckAxis, name=name, title=title, 
                          lwrite=lwrite, lensembleAxis=lensembleAxis, check_vars='station_name')
  
# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_ShpEns(ensemble=None, name=None, shape=None, filetypes=None, years=None, domains=None, 
                   varlist=None, title=None, varatts=None, translateVars=None, lcheckVars=None, 
                   lcheckAxis=True, lencl=False, lwrite=False, axis=None, lensembleAxis=False):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadWRF_Ensemble(ensemble=ensemble, grid=None, station=None, shape=shape, domains=domains, 
                          filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, lencl=lencl, 
                          translateVars=translateVars, lautoregrid=False, lctrT=True, lconst=False,
                          lcheckVars=lcheckVars, lcheckAxis=lcheckAxis, name=name, title=title, 
                          lwrite=lwrite, axis=axis, lensembleAxis=lensembleAxis, check_vars='shape_name')
  
# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_Ensemble(ensemble=None, name=None, grid=None, station=None, shape=None, domains=None, 
                     filetypes=None, years=None, varlist=None, varatts=None, translateVars=None, 
                     lautoregrid=None, title=None, lctrT=True, lconst=True, lcheckVars=None, 
                     lcheckAxis=True, lencl=False, lwrite=False, axis=None, lensembleAxis=False,
                     check_vars=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  # obviously this only works for datasets that have a time-axis
  # figure out ensemble
#   from projects.WRF_experiments import ensembles, exps, Exp # need to leave this here, to avoid circular reference...
  if ensemble is None and name is not None: 
    ensemble = name; name = None # just switch
  if isinstance(ensemble,(list,tuple)):
    for ens in ensemble:
      if isinstance(ens,Exp) or ( isinstance(ens,basestring) and ens in exps ) : pass
      else: raise TypeError
    ensemble = [ens if isinstance(ens,Exp) else exps[ens] for ens in ensemble] # convert to Exp's    
    # annotation
    if name is None: name = ensemble[0].shortname
    if title is None: title = ensemble[0].title
  else:
    if isinstance(ensemble,Exp): ensname = ensemble.shortname
    elif isinstance(ensemble,basestring): 
      # infer domain from suffix...
      if ensemble[-4:-2] == '_d' and int(ensemble[-2:]) > 0:
        domains = int(ensemble[-2:]) # overwrite domains argument
        ensname = ensemble[:-4] 
        name = ensemble if name is None else name # save original name
      else: ensname = ensemble 
      ensemble = exps[ensname] # treat ensemble like experiment until later
    else: raise TypeError
    # annotation (while ensemble is an Exp instance)
    if name is None: name = ensemble.shortname
    if title is None: title = ensemble.title
    # convert actual ensemble to list
    if ensname in ensembles: ensemble = ensembles[ensname]
    else: raise TypeError
  # figure out time period
  if years is None: montpl = (0,180)
  elif isinstance(years,(list,tuple)) and len(years)==2: 
    if not all([isInt(yr) for yr in years]): raise TypeError, years
    montpl = (years[0]*12,years[1]*12)
  elif isInt(years): montpl = (0,years*12)
  else: raise TypeError, years
  # special treatment for single experiments (i.e. not an ensemble...)
  if not isinstance(ensemble,(tuple,list)):
    if lensembleAxis: raise DatasetError, "Wont add singleton ensemble axis to single Dataset!"
    dataset = loadWRF_All(experiment=None, name=ensemble, grid=grid, station=station, shape=shape, 
                          period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                          mode='time-series', lencl=lencl, lautoregrid=lautoregrid, lctrT=lctrT, 
                          lconst=lconst, domains=domains, lwrite=lwrite, check_vars=check_vars)
  else:
    # load datasets (and load!)
    datasets = []
    for exp in ensemble:
      ds = loadWRF_All(experiment=None, name=exp, grid=grid, station=station, shape=shape, 
                       period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                       mode='time-series', lencl=lencl, lautoregrid=lautoregrid, lctrT=lctrT, 
                       lconst=lconst, domains=domains, lwrite=lwrite, check_vars=check_vars).load()
      if montpl: ds = ds(time=montpl, lidx=True) # slice the time dimension to make things consistent
      datasets.append(ds)
    # harmonize axes
    for axname,ax in ds.axes.iteritems():
      if not all([dataset.hasAxis(axname) for dataset in datasets]): 
        raise AxisError, "Not all datasets have Axis '{:s}'.".format(axname)
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

dataset_name = 'WRF' # dataset name
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
#   mode = 'pickle_grid'  
  pntset = 'shpavg'
#   pntset = 'ecprecip'
#   filetypes = ['srfc','xtrm','plev3d','hydro','lsm','rad']
  grids = ['arb1', 'arb2', 'arb3']; domains = [1,2]
  experiments = ['rrtmg', 'ctrl', 'new']
#   grids = ['col1','col2','coast1']; experiments = ['columbia','max-3km','coast']; domains = [1,2,3]   
#   grids = ['grb1']; experiments = ['']; domains = [1,2]
#   grids = ['wc2']; experiments = ['erai-wc2-2013']; domains = [1,2]
#   grids = ['arb2-120km']; experiments = ['max-lowres']; domains = [1,]   
    
  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,experiment in zip(grids,experiments):
      
      for domain in domains:
        
        print('')
        res = 'd{0:02d}'.format(domain) # for compatibility with dataset.common
        folder = '{0:s}/'.format(avgfolder)
        gridstr = '{0:s}_{1:s}'.format(grid,res) 
        print('   ***   Pickling Grid Definition for {0:s} Domain {1:d}   ***   '.format(grid,domain))
        print('')
        
        # load GridDefinition
        
        griddef, = getWRFgrid(name=grid, folder=folder, domains=domain) # filename='wrfconst_d{0:0=2d}.nc', experiment=experiment
        griddef.name = gridstr
        print('   Loading Definition from \'{0:s}\''.format(folder))
#         print(griddef)
        # save pickle
        filename = '{0:s}/{1:s}'.format(grid_folder,griddef_pickle.format(gridstr))
        if os.path.exists(filename): os.remove(filename) # overwrite 
        filehandle = open(filename, 'w')
        pickle.dump(griddef, filehandle)
        filehandle.close()
        
        print('   Saving Pickle to \'{0:s}\''.format(filename))
        print('')
        
        # load pickle to make sure it is right
        del griddef
        griddef = loadPickledGridDef(grid, res=res, folder=grid_folder)
        print(griddef)
        print('')
    
  # load averaged climatology file
  elif mode == 'test_climatology':
    
    print('')
#     dataset = loadWRF(experiment='max-1deg', domains=2, grid='arb2_d02', filetypes=['srfc'], period=(1979,1994))
    dataset = loadWRF(experiment='max-ensemble', domains=None, filetypes=['plev3d'], period=(1979,1994),
                      varlist=['u','qhv','cqwu','cqw','RH'], lconst=True)
    print(dataset)
#     dataset.lon2D.load()
#     print('')
#     print(dataset.geotransform)
    print('')
    print(dataset.zs)
    var = dataset.zs.getArray()
    print(var.min(),var.mean(),var.std(),var.max())
  
  # load monthly time-series file
  elif mode == 'test_timeseries':
    
#     dataset = loadWRF_TS(experiment='new-ctrl', domains=2, grid='arb2_d02', filetypes=['srfc'])
    dataset = loadWRF_TS(experiment='max-ctrl', domains=None, varlist=None, filetypes=['srfc'])
#     dataset = loadWRF_All(name='new-ctrl-2050', folder='/data/WRF/wrfavg/', domains=2, filetypes=['hydro'], 
#                           lctrT=True, mode='time-series')
#     for dataset in datasets:
    print('')
    print(dataset)
    print(dataset.name)
    print(dataset.title)
    print('')
    var = dataset.zs
    print(var)
    print(var.min(),var.mean(),var.std(),var.max())
    print('')
    print(dataset.time)
    print(dataset.time.offset)
    print(dataset.time.coord)

  # load ensemble "time-series"
  elif mode == 'test_ensemble':
    
    print('')
    dataset = loadWRF_Ensemble(ensemble='max-ens', varlist=['precip','MaxPrecip_1d'], filetypes=['hydro'])
#     dataset = loadWRF_Ensemble(ensemble=['max-ctrl'], varlist=['precip','MaxPrecip_1d'], filetypes=['xtrm'])
#     dataset = loadWRF_Ensemble(ensemble=['max-ctrl','max-ctrl'], varlist=['precip','MaxPrecip_1d'], filetypes=['xtrm'])
    # 2.03178e-05 0.00013171
    print('')
    print(dataset)
    print(dataset.name)
    print(dataset.title)
    print('')
    print(dataset.precip.mean())
    print(dataset.MaxPrecip_1d.mean())
#   print('')
#     print(dataset.time)
#     print(dataset.time.coord)
  

  # load station climatology file
  elif mode == 'test_point_climatology':
    
    print('')
    if pntset in ('shpavg',):
      dataset = loadWRF_Shp(experiment='max', domains=None, shape=pntset, filetypes=['hydro'], period=(1979,1994))
      print('')
      print(dataset.shape)
      print(dataset.shape.coord)
    else:
      dataset = loadWRF_Stn(experiment='erai', domains=None, station=pntset, filetypes=['hydro'], period=(1979,1984))
      zs_err = dataset.zs.getArray() - dataset.stn_zs.getArray()
      print(zs_err.min(),zs_err.mean(),zs_err.std(),zs_err.max())
#       print('')
#       print(dataset.station)
#       print(dataset.station.coord)
    dataset.load()
    print('')
    print(dataset)
    print('')
  
  # load station time-series file
  elif mode == 'test_point_timeseries':
    
    print('')
    if pntset in ('shpavg',):
      dataset = loadWRF_ShpTS(experiment='max-ctrl', domains=None, varlist=None, #['zs','stn_zs','precip','MaxPrecip_1d','wetfrq_010'], 
                              shape=pntset, filetypes=['srfc','lsm'])
    else:
      dataset = loadWRF_StnTS(experiment='max-ens-A', domains=None, varlist=['zs','stn_zs','MaxPrecip_6h'],
#                               varlist=['zs','stn_zs','precip','MaxPrecip_6h','MaxPreccu_1h','MaxPrecip_1d'], 
                              station=pntset, filetypes=['srfc'])
      zs_err = dataset.zs.getArray() - dataset.stn_zs.getArray()
      print(zs_err.min(),zs_err.mean(),zs_err.std(),zs_err.max())
    print('')
    print(dataset)
    print('')
    print(dataset.time)
    print(dataset.time.offset)
    print(dataset.time.coord)
  
  # load station ensemble "time-series"
  elif mode == 'test_point_ensemble':
    lensembleAxis = False
    print('')
    if pntset in ('shpavg',):
#       dataset = loadWRF_ShpEns(ensemble=['max-ctrl','max-ens-A'], shape=pntset, domains=None, filetypes=['hydro','srfc'])
      dataset = loadWRF_ShpEns(ensemble='max-ens', shape=pntset, varlist=['precip','runoff'], domains=2, 
                               filetypes=['srfc','lsm',], lensembleAxis=lensembleAxis)
    else:
      dataset = loadWRF_StnEns(ensemble='max-ens-2100', station=pntset, lensembleAxis=lensembleAxis,  
                               varlist=['MaxPrecip_6h'], filetypes=['srfc'])
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
  