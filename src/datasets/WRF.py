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
from geodata.base import concatDatasets
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition, addGeoLocator, GDALError
from geodata.misc import DatasetError, AxisError, DateError, ArgumentError, isNumber, isInt
from datasets.common import translateVarNames, data_root, grid_folder, default_varatts 
from geodata.gdal import loadPickledGridDef, griddef_pickle
from projects.WRF_experiments import Exp, exps 


## get WRF projection and grid definition 
# N.B.: Unlike with observational datasets, model Meta-data depends on the experiment and has to be 
#       loaded from the NetCFD-file; a few conventions have to be defied, however.

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
  else: raise TypeError
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
  cx, cy, cz = tx.TransformPoint(float(clon),float(clat)) # center point in projected (WRF) coordinates
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
def getFolderNameDomain(name=None, experiment=None, domains=None, folder=None):
  ''' Convenience function to infer and type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  #from projects.WRF_experiments import exps, Exp # need to leave this here, to avoid circular reference...   
  # check domains
  if isinstance(domains,col.Iterable):
    if not all(isInt(domains)): raise TypeError
    if not isinstance(domains,list): domains = list(domains)
    if not domains == sorted(domains): raise IndexError, 'Domains have to sorted in ascending order.'
  elif isInt(domains): domains = [domains]
  else: raise TypeError  
  # figure out experiment name
  if experiment is None:
    if not isinstance(folder,basestring): 
      raise IOError, "Need to specify an experiment folder in order to load data."
    if isinstance(name,(list,tuple)) and all([isinstance(n,basestring) for n in name]): 
      names = name
      if names[0] in exps: experiment = exps[names[0]]
      else: name = names[0].split('_')[0]
    elif isinstance(name,basestring): 
      names = [name]
      folder = folder + '/' + name
    # load experiment meta data
    if name in exps: experiment = exps[name]
#     else: raise DatasetError, 'Dataset of name \'{0:s}\' not found!'.format(names[0])
  else:
    if isinstance(experiment,(Exp,basestring)):
      if isinstance(experiment,basestring): experiment = exps[experiment] 
      # root folder
      if folder is None: folder = experiment.avgfolder
      elif not isinstance(folder,basestring): raise TypeError
      # name
      if name is None: name = experiment.name
    if isinstance(name,basestring): 
      names = ['{0:s}_d{1:0=2d}'.format(name,domain) for domain in domains]
    elif isinstance(name,col.Iterable):
      if len(domains) != len(name): raise DatasetError  
      names = name 
    else: raise TypeError      
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
                     XLAT   = dict(name='lat2D', units='deg N')) # geographic latitude field
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
                     WetDays       = dict(name='wetfrq', units=''), # fraction of wet/rainy days 
                     MaxRAIN      = dict(name='MaxPrecip_6h', units='kg/m^2/s'), # maximum 6-hourly precip                    
                     MaxRAINC     = dict(name='MaxPreccu_6h', units='kg/m^2/s'), # maximum 6-hourly convective precip
                     MaxPrecip    = dict(name='MaxPrecip_6h', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu    = dict(name='MaxPreccu_6h', units='kg/m^2/s')) # for short-term consistency
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
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation rate
                     NetWaterFlux = dict(name='waterflx', units='kg/m^2/s'), # total water downward flux
                     MaxRAIN      = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAINC     = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxPrecip    = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu    = dict(name='MaxPreccu_1d', units='kg/m^2/s')) # for short-term consistency                     
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
                     SFROFF = dict(name='sfroff', units='kg/m^2/s'), # surface run-off
                     UDROFF = dict(name='ugroff', units='kg/m^2/s'), # sub-surface/underground run-off
                     Runoff = dict(name='runoff', units='kg/m^2/s')) # total surface and sub-surface run-off
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
                     T2MEAN        = dict(name='T2', units='K'),  # daily mean Temperature (at 2m)
                     T2MIN         = dict(name='Tmin', units='K'),   # daily minimum Temperature (at 2m)
                     T2MAX         = dict(name='Tmax', units='K'),   # daily maximum Temperature (at 2m)
                     T2STD         = dict(name='Tstd', units='K'),   # daily Temperature standard deviation (at 2m)
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
                     MaxRAINMEAN   = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
                     MaxRAINCVMEAN = dict(name='MaxPreccu_1d', units='kg/m^2/s'), # maximum daily convective precip
                     MaxPrecip     = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # for short-term consistency                    
                     MaxPreccu     = dict(name='MaxPreccu_1d', units='kg/m^2/s')) # for short-term consistency                     
    self.vars = self.atts.keys()    
    self.climfile = 'wrfxtrm_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfxtrm_d{0:0=2d}{1:s}_monthly.nc' # the filename needs to be extended by (domain, grid)
# variables on selected pressure levels: 850 hPa, 700 hPa, 500 hPa, 250 hPa, 100 hPa
class Plev3D(FileType):
  ''' Variables and attributes of the pressure level files. '''
  def __init__(self):
    self.name = 'plev3d'
    self.atts = dict(T_PL     = dict(name='T', units='K', fillValue=-999),   # Temperature
                     TD_PL    = dict(name='Td', units='K', fillValue=-999),  # Dew-point Temperature
                     RH_PL    = dict(name='RH', units='', fillValue=-999),   # Relative Humidity
                     GHT_PL   = dict(name='Z', units='m', fillValue=-999),   # Geopotential Height 
                     S_PL     = dict(name='U', units='m/s', fillValue=-999), # Wind Speed
                     U_PL     = dict(name='u', units='m/s', fillValue=-999), # Zonal Wind Speed
                     V_PL     = dict(name='v', units='m/s', fillValue=-999)) # Meridional Wind Speed
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
                     station     = dict(name='station', units='') ) # station axis for station data
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
          att['name'] = att['name']+'_7d'
          atts[x+key+'_7d'] = att
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
def loadWRF_StnTS(experiment=None, name=None, domains=2, station=None, filetypes=None, 
                varlist=None, varatts=None, lctrT=True):
  ''' Get a properly formatted WRF dataset with monthly time-series at station locations. '''  
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=None, station=station, 
                     period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=False, 
                     lautoregrid=False, lctrT=lctrT, mode='time-series')  

def loadWRF_TS(experiment=None, name=None, domains=2, grid=None, filetypes=None, varlist=None, 
               varatts=None, lconst=True, lautoregrid=True, lctrT=True):
  ''' Get a properly formatted WRF dataset with monthly time-series. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=None, 
                     period=None, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=lautoregrid, lctrT=lctrT, mode='time-series')  

def loadWRF_Stn(experiment=None, name=None, domains=2, station=None, period=None, filetypes=None, 
                varlist=None, varatts=None, lctrT=True):
  ''' Get a properly formatted station dataset from a monthly WRF climatology. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=None, station=station, 
                     period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=False, 
                     lautoregrid=False, lctrT=lctrT, mode='climatology')  

def loadWRF(experiment=None, name=None, domains=2, grid=None, period=None, filetypes=None, varlist=None, 
            varatts=None, lconst=True, lautoregrid=True, lctrT=True):
  ''' Get a properly formatted monthly WRF climatology as NetCDFDataset. '''
  return loadWRF_All(experiment=experiment, name=name, domains=domains, grid=grid, station=None, 
                     period=period, filetypes=filetypes, varlist=varlist, varatts=varatts, lconst=lconst, 
                     lautoregrid=lautoregrid, lctrT=lctrT, mode='climatology')  

# pre-processed climatology files (varatts etc. should not be necessary) 
def loadWRF_All(experiment=None, name=None, domains=2, grid=None, station=None, period=None, 
                filetypes=None, varlist=None, varatts=None, lconst=True, lautoregrid=True, 
                lctrT=False, folder=None, mode='climatology'):
  ''' Get any WRF data files as a properly formatted NetCDFDataset. '''
  # prepare input  
  ltuple = isinstance(domains,col.Iterable)  
  # prepare input  
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
  if station is None: 
    lstation = False
  else: 
    lstation = True
    if grid is not None: raise NotImplementedError, 'Currently WRF station data can only be loaded from the native grid.'
    if lconst: raise NotImplementedError, 'Currently WRF constants are not available as station data.'
    if lautoregrid: raise GDALError, 'Station data can not be regridded, since it is not map data.' 
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = fileclasses.keys()
  elif isinstance(filetypes,(tuple,set)): filetypes = list(filetypes)
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
  if station is None:
    if grid is None:
      griddefs = None; c = -1; filename = None
      while filename is None:
        c += 1 # this is necessary, because not all filetypes have time-series files
        filename = fileclasses.values()[c].tsfile # just use the first filetype
      while griddefs is None:
        # some experiments do not have all files... try, until one works...
        try:
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
      # load dataset
      const = DatasetNetCDF(name=name, folder=folder, filelist=[filename], varatts=constfile.atts, axes=axes,  
                            varlist=constfile.vars, multifile=False, ncformat='NETCDF4', squeeze=True)      
    ## load regular variables
    filenames = []
    for filetype,fileformat in zip(typelist,filelist):
      if lclim: filename = fileformat.format(domain,gridstr,periodstr) # insert domain number, grid, and period
      elif lts: filename = fileformat.format(domain,gridstr) # insert domain number, and grid
      filenames.append(filename) # file list to be passed on to DatasetNetCDF
      # check existance
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
            if performRegridding('WRF', 'climatology', griddef, dataargs): # default kwargs
              raise IOError, "Automatic regridding failed!"
          else: raise IOError, "The  '{:s}' (WRF) dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.\n('{:s}')".format(name,filename,grid,filepath) 
        else: raise IOError, "The file '{:s}' in WRF dataset '{:s}' does not exits!\n('{:s}')".format(filename,name,filepath)   
       
    # load dataset
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=axes, 
                            varatts=atts, multifile=False, ncformat='NETCDF4', squeeze=True)
    # check
    if llconst: lenc = len(const)
    else: lenc = 0 
    if (len(dataset)+lenc) == 0: raise DatasetError, 'Dataset is empty - check source file or variable list!'
    # check time axis and center at 1979-01 (zero-based)
    if lctrT and dataset.hasAxis('time'):
      # N.B.: we can directly change axis vectors, since they reside in memory;
      #       the source file is not changed, unless sync() is called
      if lts:
        t0 = dataset.time.coord[0]
        tm = t0%12; ty = int(np.floor(t0/12))  
        if tm != ms - 1: 
          dataset.time.coord -= ( tm+1 - ms )
          dataset.time.offset -= ( tm+1 - ms )
        if ty != ys - 1979: 
          dataset.time.coord -= ( ty+1979 - ys )*12
          dataset.time.offset -= ( ty+1979 - ys )*12 
      elif lclim:
        t0 = dataset.time.coord[0]
        # there is no "year" - just start with "1" for january 
        if t0 != 1: 
          dataset.time.coord -= ( t0 - 1 )
          dataset.time.offset -= ( t0 - 1 )
    # add constants to dataset
    if llconst:
      for var in const: 
        if var.name not in dataset:
          dataset.addVariable(var, asNC=False, copy=False, overwrite=False, deepcopy=False)
    if not lstation:
      # add projection
      dataset = addGDALtoDataset(dataset, griddef=griddef, gridfolder=grid_folder, geolocator=True)
      #print dataset
      # safety checks
      if dataset.isProjected:
        assert dataset.axes['x'] == griddef.xlon
        assert dataset.axes['y'] == griddef.ylat   
        assert all([dataset.axes['x'] == var.getAxis('x') for var in dataset.variables.values() if var.hasAxis('x')])
        assert all([dataset.axes['y'] == var.getAxis('y') for var in dataset.variables.values() if var.hasAxis('y')])
    # append to list
    datasets.append(dataset) 
  # return formatted dataset
  if not ltuple: datasets = datasets[0]
  return datasets


# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_StnEns(ensemble=None, station=None, filetypes=None, years=None, domains=2, varlist=None, 
                   varatts=None, translateVars=None, lcheckVars=True, lcheckAxis=True):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadWRF_Ensemble(ensemble=ensemble, grid=None, station=station, domains=domains, 
                          filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, 
                          translateVars=translateVars, lautoregrid=False, lctrT=True, lconst=False,
                          lcheckVars=lcheckVars, lcheckAxis=lcheckAxis)
  
# load a pre-processed WRF ensemble and concatenate time-series 
def loadWRF_Ensemble(ensemble=None, grid=None, station=None, domains=2, filetypes=None, 
                     years=None, varlist=None, varatts=None, translateVars=None, lautoregrid=None, 
                     lctrT=True, lconst=True, lcheckVars=True, lcheckAxis=True):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  # obviously this only works for modes that produce a time-axis
  # figure out ensemble
  from projects.WRF_experiments import ensembles, Exp # need to leave this here, to avoid circular reference...
  if isinstance(ensemble,Exp): ensemble = ensembles[ensemble.shortname]
  elif isinstance(ensemble,basestring): ensemble = ensembles[ensemble]
  else: raise TypeError
  if not isinstance(ensemble,(tuple,list)): raise TypeError
#   if not ( isinstance(ensemble,(tuple,list)) and
#            all([isinstance(exp,(basestring,Exp)) for exp in ensemble]) ): raise TypeError
  # figure out time period
  if years is None: years =15
  elif isinstance(years,(list,tuple)) and len(years)==2: raise NotImplementedError 
  elif not isInt(years): raise TypeError  
  montpl = (0,years*12)
  # load datasets (and load!)
  datasets = []
  for exp in ensemble:
    ds = loadWRF_All(experiment=exp, name=None, grid=grid, station=station, filetypes=filetypes, 
                     varlist=varlist, varatts=varatts, period=None, mode='time-series', 
                     lautoregrid=lautoregrid, lctrT=lctrT, lconst=lconst, domains=domains)
    datasets.append(ds.load())
  # concatenate datasets (along 'time' axis, WRF doesn't have 'year')  
  dataset = concatDatasets(datasets, axis='time', coordlim=None, idxlim=montpl, offset=None, axatts=None, 
                           lcpOther=True, lcpAny=False, lcheckVars=lcheckVars, lcheckAxis=lcheckAxis)
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
loadTimeSeries = loadWRF_TS # time-series data
loadStationTimeSeries = loadWRF_StnTS # time-series data at stations
loadClimatology = loadWRF # pre-processed, standardized climatology
loadStationClimatology = loadWRF_Stn # pre-processed, standardized climatology at stations


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
#   mode = 'test_climatology'
#   mode = 'test_station_climatology'
#   mode = 'test_timeseries'
  mode = 'test_station_timeseries'
#   mode = 'test_ensemble'
#   mode = 'test_station_ensemble'
#   mode = 'pickle_grid'  
  filetypes = ['srfc','xtrm','plev3d','hydro','lsm','rad']
  grids = ['arb1', 'arb2', 'arb3']; domains = [1,2]
  experiments = ['rrtmg', 'ctrl', 'new']
#   grids = ['col1','col2','coast1']; experiments = ['columbia','max-3km','coast']; domains = [1,2,3]   
#   grids = ['grb1']; experiments = ['']; domains = [1,2]
    
  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,experiment in zip(grids,experiments):
      
      for domain in domains:
        
        print('')
        res = 'd{0:02d}'.format(domain) # for compatibility with dataset.common
        folder = '{0:s}/{1:s}/'.format(avgfolder,grid)
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
    dataset = loadWRF(experiment='max-1deg', domains=2, filetypes=['srfc'], period=(1979,1984))
    print(dataset)
#     dataset.lon2D.load()
#     print('')
#     print(dataset.geotransform)
    print('')
    print(dataset.zs)
    var = dataset.zs.getArray()
    print(var.min(),var.mean(),var.std(),var.max())
  
  # load station climatology file
  elif mode == 'test_station_climatology':
    
    print('')
    dataset = loadWRF_Stn(experiment='erai', domains=2, station='ecprecip', filetypes=['hydro'], period=(1979,1984))
    print('')
    print(dataset)
    print('')
    zs_err = dataset.zs.getArray() - dataset.stn_zs.getArray()
    print(zs_err.min(),zs_err.mean(),zs_err.std(),zs_err.max())
#     print('')
#     print(dataset.station)
#     print(dataset.station.coord)

  
  # load monthly time-series file
  elif mode == 'test_timeseries':
    
#     dataset = loadWRF_TS(experiment='new-ctrl', domains=2, grid='arb2_d02', filetypes=['srfc'])
    dataset = loadWRF_TS(experiment='max-ctrl', domains=2, varlist=None, filetypes=['srfc'])
#     dataset = loadWRF_All(name='new-ctrl-2050', folder='/data/WRF/wrfavg/', domains=2, filetypes=['hydro'], 
#                           lctrT=True, mode='time-series')
#     for dataset in datasets:
    print('')
    print(dataset)
    print('')
    var = dataset.zs
    print(var)
    print(var.min(),var.mean(),var.std(),var.max())
#     print('')
#     print(dataset.time)
#     print(dataset.time.offset)
#     print(dataset.time.coord)

  # load station time-series file
  elif mode == 'test_station_timeseries':
    
    print('')
    dataset = loadWRF_StnTS(experiment='max-ctrl', domains=2, varlist=['zs','stn_zs','precip','MaxPrecip_1d','MaxPrecip_7d'], station='ecprecip', filetypes=['hydro'])
    print('')
    print(dataset)
    print('')
    zs_err = dataset.zs.getArray() - dataset.stn_zs.getArray()
#     print(zs_err)
    print(zs_err.min(),zs_err.mean(),zs_err.std(),zs_err.max())
#     print('')
#     print(dataset.time)
#     print(dataset.time.offset)
#     print(dataset.time.coord)
  
  # load ensemble "time-series"
  elif mode == 'test_ensemble':
    
    print('')
    dataset = loadWRF_Ensemble(ensemble='max-ens-2100', varlist=['precip','MaxPrecip'], filetypes=['xtrm'])
    print('')
    print(dataset)
    print('')
    print(dataset.precip.mean())
    print(dataset.MaxPrecip.mean())
#   print('')
#     print(dataset.time)
#     print(dataset.time.coord)
  
  # load station ensemble "time-series"
  elif mode == 'test_station_ensemble':
    
    print('')
    dataset = loadWRF_StnEns(ensemble='max-ens', station='ecprecip', domains=2, filetypes=['xtrm']).load()
    print('')
    print(dataset)
    print('')
    print(dataset.precip.mean())
    print(dataset.MaxPrecip.mean())
#     print(dataset.precip.atts)
#     print(dataset.precip.plot)
    print('')
#     hp = dataset.MaxPrecip.histogram(bins=30, axis='time')
#     print(hp.atts)
#     print(hp.plot)
#     print(hp.axes[0].atts)
#     print(hp.axes[0].plot)
#   
#     print('')
#     print(dataset.time)
#     print(dataset.time.coord)
  