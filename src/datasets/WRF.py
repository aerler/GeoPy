'''
Created on 2013-09-28

This module contains common meta data and access functions for WRF model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc
import collections as col
import os
import pickle
# from atmdyn.properties import variablePlotatts
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition
from geodata.misc import DatasetError, isInt, AxisError, DateError, isNumber
from datasets.common import translateVarNames, data_root, grid_folder, default_varatts 
from geodata.gdal import loadPickledGridDef, griddef_pickle

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
    lon_0 = dataset.CEN_LON # Longitude of natural origin
    #if dataset.CEN_LON != dataset.STAND_LON: raise GDALError  
  else:
    raise NotImplementedError, "Can only infer projection parameters for Lambert Conformal Conic projection (#1)."
  projdict = dict(proj=proj,lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0)
  # pass results to GDAL module to get projection object
  return getProjFromDict(projdict, name=name, GeoCS='WGS84', convention='Proj4')  

# infer grid (projection and axes) from constants file
def getWRFgrid(name=None, experiment=None, domains=None, folder=None, filename='wrfconst_d{0:0=2d}.nc', ncformat='NETCDF4'):
  ''' Infer the WRF grid configuration from an output file and return a GridDefinition object. '''
  from datasets.WRF_experiments import Exp, exps 
  # check input
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, folder=folder)
  if isinstance(filename,basestring): filepath = '{}/{}'.format(folder,filename) # still contains formaters
  else: raise TypeError
  # figure out experiment
  if experiment is None:
    if isinstance(name,basestring): experiment = exps[name]
    elif len(names) > 0: exps[names[0].split('_')[0]]
  elif isinstance(experiment,basestring): experiment = exps[experiment]
  elif not isinstance(experiment,Exp): raise TypeError  
  maxdom = max(domains) # max domain
  # files to work with
  for n in xrange(1,maxdom+1):
    dnfile = filepath.format(n)
    if not os.path.exists(dnfile):
      if n in domains: raise IOError, 'File {} for domain {:d} not found!'.format(dnfile,n)
      else: raise IOError, 'File {} for domain {:d} not found; this file is necessary to infer the geotransform for other domains.'.format(dnfile,n)
  # open first domain file (special treatment)
  dn = nc.Dataset(filepath.format(1), mode='r', format=ncformat)
  #name = experiment if isinstance(experiment,basestring) else names[0] # omit domain information, which is irrelevant
  projection = getWRFproj(dn, name=experiment.grid) # same for all
  # infer size and geotransform
  def getXYlen(ds):
    ''' a short function to infer the length of horizontal axes from a dataset with unknown naming conventions ''' 
    if 'west_east' in ds.dimensions and 'south_north' in ds.dimensions:
      nx = len(ds.dimensions['west_east']); ny = len(ds.dimensions['south_north'])
    elif 'x' in ds.dimensions and 'y' in ds.dimensions:
      nx = len(ds.dimensions['x']); ny = len(ds.dimensions['y'])
    else: raise AxisError, 'No horizontal axis found, necessary to infer projection/grid configuration.'
    return nx,ny
  dx = dn.DX; dy = dn.DY
  nx,ny = getXYlen(dn)
  x0 = -nx*dx/2; y0 = -ny*dy/2 
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
      dn = nc.Dataset(filepath.format(n), mode='r', format=ncformat)
      if not n == dn.GRID_ID: raise DatasetError # just a check
      pid = dn.PARENT_ID-1 # parent grid ID
      # infer size and geotransform      
      px0,pdx,s,py0,t,pdy = geotransforms[pid]      
      x0 = px0+dn.I_PARENT_START*pdx; y0 = py0+dn.J_PARENT_START*pdy
      size = getXYlen(dn) 
      geotransform = (x0,dn.DX,0.,y0,0.,dn.DY)
      dn.close()
      geotransforms.append(geotransform) # we need that to construct the next nested domain
      if n in domains:
        name = '{0:s}_d{1:02d}'.format(experiment.grid,n) 
        griddefs.append(GridDefinition(name=name, projection=projection, geotransform=geotransform, size=size))
  # return a GridDefinition object
  return tuple(griddefs)  

# return name and folder
def getFolderNameDomain(name=None, experiment=None, domains=None, folder=None):
  ''' Convenience function to infer type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  from datasets.WRF_experiments import exps, Exp # need to leave this here, to avoid circular reference...   
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
    if isinstance(name,col.Iterable) and all([isinstance(n,basestring) for n in name]): 
      names = name
      if names[0] in exps: experiment = exps[names[0]]
      else: name = names[0].split('_')[0]
    elif isinstance(name,basestring): names = [name]
    # load experiment meta data
    if name in exps: experiment = exps[name]
    else: raise DatasetError, 'Dataset of name \'{0:s}\' not found!'.format(names[0])
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
    self.atts = dict(HGT    = dict(name='zs', units='m'), # surface elevation
                     XLONG  = dict(name='lon2D', units='deg E'), # geographic longitude field
                     XLAT   = dict(name='lat2D', units='deg N')) # geographic latitude field
    self.vars = self.atts.keys()    
    self.climfile = 'wrfconst_d{0:0=2d}(1:s}.nc' # the filename needs to be extended by (domain,'_'+grid)
    self.tsfile = 'wrfconst_d{0:0=2d}.nc' # the filename needs to be extended by (domain,)
# surface variables
class Srfc(FileType):
  ''' Variables and attributes of the surface files. '''
  def __init__(self):
    self.atts = dict(T2     = dict(name='T2', units='K'), # 2m Temperature
                     Q2     = dict(name='q2', units='kg/kg'), # 2m water vapor mass mixing ratio
                     RAIN   = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
                     RAINC  = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate (kg/m^2/s)
                     RAINNC = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate (kg/m^2/s)
                     SNOW   = dict(name='snow', units='kg/m^2'), # snow water equivalent
                     SNOWH  = dict(name='snowh', units='m'), # snow depth
                     PSFC   = dict(name='ps', units='Pa'), # surface pressure
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation rate
                     WaterVapor  = dict(name='Q2', units='Pa')) # water vapor partial pressure                     
    self.vars = self.atts.keys()    
    self.climfile = 'wrfsrfc_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfsrfc_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
# hydro variables
class Hydro(FileType):
  ''' Variables and attributes of the hydrological files. '''
  def __init__(self):
    self.atts = dict(T2MEAN = dict(name='T2', units='K'), # daily mean 2m Temperature
                     RAIN   = dict(name='precip', units='kg/m^2/s'), # total precipitation rate
                     RAINC  = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate
                     RAINNC = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate
                     SFCEVP = dict(name='evap', units='kg/m^2/s'), # actual surface evaporation/ET rate
                     ACSNOM = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate 
                     POTEVP = dict(name='pet', units='kg/m^2/s', scalefactor=999.70), # potential evapo-transpiration rate
                     NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     LiquidPrecip = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation rate
                     SolidPrecip  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation rate
                     NetWaterFlux = dict(name='waterflx', units='kg/m^2/s')) # total water downward flux                     
    self.vars = self.atts.keys()
    self.climfile = 'wrfhydro_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfhydro_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
# lsm variables
class LSM(FileType):
  ''' Variables and attributes of the land surface files. '''
  def __init__(self):
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
    self.tsfile = 'wrflsm_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
# lsm variables
class Rad(FileType):
  ''' Variables and attributes of the radiation files. '''
  def __init__(self):
    self.atts = dict() # currently empty
    self.vars = self.atts.keys()    
    self.climfile = 'wrfrad_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfrad_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
# extreme value variables
class Xtrm(FileType):
  ''' Variables and attributes of the extreme value files. '''
  def __init__(self):
    self.atts = dict(T2MEAN = dict(name='Tmean', units='K'),  # daily mean Temperature (at 2m)
                     T2MIN  = dict(name='Tmin', units='K'),   # daily minimum Temperature (at 2m)
                     T2MAX  = dict(name='Tmax', units='K'),   # daily maximum Temperature (at 2m)
                     T2STD  = dict(name='Tstd', units='K'),   # daily Temperature standard deviation (at 2m)
                     SKINTEMPMEAN = dict(name='TSmean', units='K'),  # daily mean Skin Temperature
                     SKINTEMPMIN  = dict(name='TSmin', units='K'),   # daily minimum Skin Temperature
                     SKINTEMPMAX  = dict(name='TSmax', units='K'),   # daily maximum Skin Temperature
                     SKINTEMPSTD  = dict(name='TSstd', units='K'),   # daily Skin Temperature standard deviation                     
                     Q2MEAN = dict(name='Qmean', units='Pa', scalefactor=Q), # daily mean Water Vapor Pressure (at 2m)
                     Q2MIN  = dict(name='Qmin', units='Pa', scalefactor=Q),  # daily minimum Water Vapor Pressure (at 2m)
                     Q2MAX  = dict(name='Qmax', units='Pa', scalefactor=Q),  # daily maximum Water Vapor Pressure (at 2m)
                     Q2STD  = dict(name='Qstd', units='Pa', scalefactor=Q),  # daily Water Vapor Pressure standard deviation (at 2m)
                     SPDUV10MEAN = dict(name='U10mean', units='m/s'), # daily mean Wind Speed (at 10m)
                     SPDUV10MAX  = dict(name='U10max', units='m/s'),  # daily maximum Wind Speed (at 10m)
                     SPDUV10STD  = dict(name='U10std', units='m/s'),  # daily Wind Speed standard deviation (at 10m)
                     U10MEAN = dict(name='u10mean', units='m/s'), # daily mean Westerly Wind (at 10m)
                     V10MEAN = dict(name='v10mean', units='m/s'), # daily mean Southerly Wind (at 10m)                     
                     RAINCVMEAN  = dict(name='preccumean', units='kg/m^2/s'), # daily mean convective precipitation rate
                     RAINCVMAX  = dict(name='preccumax', units='kg/m^2/s'), # daily maximum convective precipitation rate
                     RAINCVSTD  = dict(name='preccustd', units='kg/m^2/s'), # daily convective precip standard deviation
                     RAINNCVMEAN = dict(name='precncmean', units='kg/m^2/s'), # daily mean grid-scale precipitation rate
                     RAINNCVMAX  = dict(name='precncmax', units='kg/m^2/s'), # daily maximum grid-scale precipitation rate
                     RAINNCVSTD  = dict(name='precncstd', units='kg/m^2/s')) # daily grid-scale precip standard deviation                     
    self.vars = self.atts.keys()    
    self.climfile = 'wrfxtrm_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfxtrm_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
# variables on selected pressure levels: 850 hPa, 700 hPa, 500 hPa, 250 hPa, 100 hPa
class Plev3D(FileType):
  ''' Variables and attributes of the pressure level files. '''
  def __init__(self):
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
    self.tsfile = 'wrfplev3d_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)

# axes (don't have their own file)
class Axes(FileType):
  ''' A mock-filetype for axes. '''
  def __init__(self):
    self.atts = dict(Time        = dict(name='time', units='month'), # time coordinate
                     time        = dict(name='time', units='month'), # time coordinate
                     # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                     #       the time offset is chose such that 1979 begins with the origin (time=0)
                     west_east   = dict(name='x', units='m'), # projected west-east coordinate
                     south_north = dict(name='y', units='m'), # projected south-north coordinate
                     x           = dict(name='x', units='m'), # projected west-east coordinate
                     y           = dict(name='y', units='m'), # projected south-north coordinate
                     soil_layers_stag = dict(name='s', units=''), # soil layer coordinate
                     num_press_levels_stag = dict(name='p', units='Pa')) # pressure coordinate
    self.vars = self.atts.keys()
    self.climfile = None
    self.tsfile = None

# data source/location
fileclasses = dict(const=Const(), srfc=Srfc(), hydro=Hydro(), lsm=LSM(), rad=Rad(), xtrm=Xtrm(), plev3d=Plev3D(), axes=Axes())
root_folder = data_root + 'WRF/' # long-term mean folder
outfolder = root_folder + 'wrfout/' # WRF output folder
avgfolder = root_folder + 'wrfavg/' # long-term mean folder


## Functions to load different types of WRF datasets

# Time-Series (monthly)
def loadWRF_TS(experiment=None, name=None, domains=2, filetypes=None, varlist=None, varatts=None):
  ''' Get a properly formatted WRF dataset with monthly time-series. '''
  # prepare input  
  ltuple = isinstance(domains,col.Iterable)
  # N.B.: 'experiment' can be a string name or an Exp instance
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, folder=None)
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = fileclasses.keys()
  elif isinstance(filetypes,list):
    if 'axes' not in filetypes: filetypes.append('axes')
  elif isinstance(filetypes,tuple):
    if 'axes' not in filetypes: filetypes = filetypes + ('axes',)
  else: raise TypeError  
  atts = dict(); filelist = [] 
  for filetype in filetypes:
    fileclass = fileclasses[filetype]
    if fileclass.tsfile is not None: filelist.append(fileclass.tsfile) 
    atts.update(fileclass.atts)  
  if varatts is not None: atts.update(varatts)
  # translate varlist
  #if varlist is None: varlist = atts.keys()
  if varatts: varlist = translateVarNames(varlist, varatts)
  # infer projection and grid and generate horizontal map axes
  # N.B.: unlike with other datasets, the projection has to be inferred from the netcdf files  
  if 'const' in filetypes: filename = fileclasses['const'].tsfile # constants files preferred...
  else: filename = fileclasses[filetypes[0]].tsfile # just use the first filetype
  griddefs = getWRFgrid(name=names, experiment=None, domains=domains, folder=folder, filename=filename)
  assert len(griddefs) == len(domains)
  datasets = []
  for name,domain,griddef in zip(names,domains,griddefs):
    # domain-sensitive parameters
    axes = dict(west_east=griddef.xlon, south_north=griddef.ylat, x=griddef.xlon, y=griddef.ylat) # map axes
    filenames = [filename.format(domain) for filename in filelist] # insert domain number
    # load dataset
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, varatts=atts, 
                            axes=axes, multifile=False, ncformat='NETCDF4', squeeze=True)
    # load pressure levels (doesn't work automatically, because variable and dimension have different names and dimensions)
    if dataset.hasAxis('p'): 
      dataset.axes['p'].updateCoord(dataset.dataset.variables['P_PL'][0,:])
    # add projection
    dataset = addGDALtoDataset(dataset, griddef=griddef, folder=grid_folder)
    # safety checks
    assert dataset.axes['x'] == griddef.xlon
    assert dataset.axes['y'] == griddef.ylat   
    assert all([dataset.axes['x'] == var.getAxis('x') for var in dataset.variables.values() if var.hasAxis('x')])
    assert all([dataset.axes['y'] == var.getAxis('y') for var in dataset.variables.values() if var.hasAxis('y')])
    # append to list
    datasets.append(dataset) 
  # return formatted dataset
  if not ltuple: datasets = datasets[0]
  return datasets
  

# pre-processed climatology files (varatts etc. should not be necessary) 
def loadWRF(experiment=None, name=None, domains=2, grid=None, period=None, filetypes=None, varlist=None, varatts=None):
  ''' Get a properly formatted monthly WRF climatology as NetCDFDataset. '''
  # prepare input  
  ltuple = isinstance(domains,col.Iterable)
  # N.B.: 'experiment' can be a string name or an Exp instance
  folder,experiment,names,domains = getFolderNameDomain(name=name, experiment=experiment, domains=domains, folder=None)
  # period  
  from WRF_experiments import Exp
  if isinstance(period,(tuple,list)): pass
  elif isinstance(period,basestring): pass
  elif period is None: pass
  elif isinstance(period,(int,np.integer)) and isinstance(experiment,Exp):
    period = (experiment.beginyear, experiment.beginyear+period)
  else: raise DateError   
  if period is None or period == '': periodstr = ''
  elif isinstance(period,basestring): periodstr = '_{0:s}'.format(period)
  else: periodstr = '_{0:4d}-{1:4d}'.format(*period)  
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = fileclasses.keys()
  elif isinstance(filetypes,list):  
    if 'axes' not in filetypes: filetypes.append('axes')
    if 'const' not in filetypes and grid is None: filetypes.append('const')
  else: raise TypeError  
  atts = dict(); filelist = []; constfile = None
  for filetype in filetypes:
    fileclass = fileclasses[filetype]
    if filetype == 'const': 
      constfile = fileclass.tsfile
      atts.update(fileclass.atts)  
    elif fileclass.tsfile is not None: 
      filelist.append(fileclass.climfile) 
  if varatts is not None: atts.update(varatts)
  lconst = constfile is not None
  # translate varlist
  #if varlist is None: varlist = default_varatts.keys() + atts.keys()
  if varatts: varlist = translateVarNames(varlist, varatts) # default_varatts
  # infer projection and grid and generate horizontal map axes
  # N.B.: unlike with other datasets, the projection has to be inferred from the netcdf files  
  if constfile is not None: filename = constfile # constants files preferred...
  else: filename = fileclasses.values()[0].tsfile # just use the first filetype
  if grid is None:
    griddefs = getWRFgrid(name=names, experiment=experiment, domains=domains, folder=folder, filename=filename)
  else:
    griddefs = [loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder, check=True)]*len(domains)
  assert len(griddefs) == len(domains)
  # grid
  datasets = []
  for name,domain,griddef in zip(names,domains,griddefs):
    if grid is None or grid.split('_')[0] == experiment.grid: gridstr = ''
    else: gridstr = '_%s'%grid.lower() # only use lower case for filenames     
    # domain-sensitive parameters
    axes = dict(west_east=griddef.xlon, south_north=griddef.ylat, x=griddef.xlon, y=griddef.ylat) # map axes
    # load constants
    if lconst:
      # load dataset
      const = DatasetNetCDF(name=name, folder=folder, filelist=[constfile.format(domain)], varatts=atts, axes=axes,  
                            varlist=fileclasses['const'].vars, multifile=False, ncformat='NETCDF4', squeeze=True)      
    # load regular variables
    filenames = [filename.format(domain,gridstr,periodstr) for filename in filelist] # insert domain number
    # load dataset
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=axes, 
                            varatts=atts, multifile=False, ncformat='NETCDF4', squeeze=True)
    # check
    if len(dataset) == 0: raise DatasetError, 'Dataset is empty - check source file or variable list!'
    # add constants to dataset
    if lconst:
      for var in const: 
        if not dataset.hasVariable(var): # 
          dataset.addVariable(var, asNC=False, copy=False, overwrite=False, deepcopy=False)
    # add projection
    dataset = addGDALtoDataset(dataset, griddef=griddef, gridfolder=grid_folder)
    # safety checks
    assert dataset.axes['x'] == griddef.xlon
    assert dataset.axes['y'] == griddef.ylat   
    assert all([dataset.axes['x'] == var.getAxis('x') for var in dataset.variables.values() if var.hasAxis('x')])
    assert all([dataset.axes['y'] == var.getAxis('y') for var in dataset.variables.values() if var.hasAxis('y')])
    # append to list
    datasets.append(dataset) 
  # return formatted dataset
  if not ltuple: datasets = datasets[0]
  return datasets

## Dataset API

dataset_name = 'WRF' # dataset name
root_folder # root folder of the dataset
avgfolder # root folder for monthly averages
outfolder # root folder for direct WRF output
file_pattern = 'wrf{0:s}_d{1:02d}{2:s}_clim{3:s}.nc' # filename pattern: filetype, domain, grid, period
data_folder = root_folder # folder for user data
grid_def = {'d02':None,'d01':None} # there are too many... 
grid_res = {'d02':0.13,'d01':3.82} # approximate grid resolution at 45 degrees latitude 
# grid_def = {0.13:None,3.82:None} # approximate grid resolution at 45 degrees latitude
# grid_tag = {0.13:'d02',3.82:'d01'} 
# functions to access specific datasets
loadLongTermMean = None # WRF doesn't have that...
loadTimeSeries = loadWRF_TS # time-series data
loadClimatology = loadWRF # pre-processed, standardized climatology


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
  mode = 'test_climatology'
#   mode = 'test_timeseries'
#   mode = 'pickle_grid'
  experiment = 'new-ctrl'
  domains = [1,2]
  filetypes = ['srfc','xtrm','plev3d','hydro','lsm','rad']
  grids = ['arb1', 'arb2', 'arb3']   
    
  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid in grids:
      
      for domain in domains:
        
        print('')
        res = 'd{0:02d}'.format(domain) # for compatibility with dataset.common
        folder = '{0:s}/{1:s}/'.format(avgfolder,grid)
        gridstr = '{0:s}_{1:s}'.format(grid,res) 
        print('   ***   Pickling Grid Definition for {0:s} Domain {1:d}   ***   '.format(grid,domain))
        print('')
        
        # load GridDefinition
        
        griddef, = getWRFgrid(name=(gridstr,), domains=(domain,), folder=folder, filename='wrfconst_d{0:0=2d}.nc')
        griddef.name = gridstr
        print('   Loading Definition from \'{0:s}\''.format(folder))
        # save pickle
        filename = '{0:s}/{1:s}'.format(grid_folder,griddef_pickle.format(gridstr))
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
    dataset = loadWRF(experiment=experiment, domains=2, grid='arb2_d02', filetypes=['srfc'], period=(1979,1989))
    print(dataset)
    dataset.T2.load()
    print('')
    print(dataset.geotransform)
  
  
  # load monthly time-series file
  elif mode == 'test_timeseries':
    
    dataset = loadWRF_TS(experiment='max-ctrl', domains=1, filetypes=['srfc'])
#     for dataset in datasets:
    print('')
    print(dataset)
    print('')
    print(dataset.geotransform)
