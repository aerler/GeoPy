'''
Created on 2013-09-28

This module contains meta data and access functions for WRF model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc
import os
# from atmdyn.properties import variablePlotatts
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition
from geodata.misc import DatasetError, GDALError
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root
from geodata.process import CentralProcessingUnit


## get projection 
# N.B.: Unlike with observational datasets, model Meta-data depends on the experiment and has to be 
#       loaded from the NetCFD-file; a few conventions have to be defied, however.
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
#     print dataset.CEN_LON,dataset.STAND_LON
    if dataset.CEN_LON != dataset.STAND_LON: raise GDALError  
  else:
    raise NotImplementedError, "Can only infer projection parameters for Lambert Conformal Conic projection (#1)."
  projdict = dict(proj=proj,lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0)
  # pass results to GDAL module to get projection object
  return getProjFromDict(projdict, name=name, GeoCS='WGS84', convention='Proj4')  


## variable attributes and name
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
                     Q2     = dict(name='Q2', units='Pa'), # 2m water vapor pressure
                     RAIN   = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
                     RAINC  = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate (kg/m^2/s)
                     RAINNC = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate (kg/m^2/s)
                     SNOW   = dict(name='snow', units='kg/m^2'), # snow water equivalent
                     SNOWH  = dict(name='snowh', units='m'), # snow depth
                     PSFC   = dict(name='ps', units='Pa')) # surface pressure
    self.vars = self.atts.keys()    
    self.climfile = 'wrfsrfc_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfsrfc_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
# surface variables
class Hydro(FileType):
  ''' Variables and attributes of the hydrological files. '''
  def __init__(self):
    self.atts = dict(T2     = dict(name='T2', units='K'), # daily mean 2m Temperature
                     RAIN   = dict(name='precip', units='kg/m^2/s'), # total precipitation rate
                     RAINC  = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate
                     RAINNC = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate
                     SFCEVP = dict(name='evap', units='kg/m^2/s'), # actual surface evaporation/ET rate
                     ACSNOM = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate 
                     POTEVP = dict(name='pet', units='kg/m^2/s')) # potential evapo-transpiration rate
    self.vars = self.atts.keys()    
    self.climfile = 'wrfhydro_d{0:0=2d}{1:s}_clim{2:s}.nc' # the filename needs to be extended by (domain,'_'+grid,'_'+period)
    self.tsfile = 'wrfhydro_d{0:0=2d}_monthly.nc' # the filename needs to be extended by (domain,)
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
                     y           = dict(name='y', units='m')) # projected south-north coordinate                 
    self.vars = self.atts.keys()
    self.climfile = None
    self.tsfile = None

# # include these variables in monthly means 
# varlist = ['ps','T2','Ts','snow','snowh','rainnc','rainc','snownc','graupelnc',
#            'Q2','evap','hfx','lhfx','OLR','GLW','SWDOWN'] # 'SWNORM'
# varmap = dict(ps='PSFC',Q2='Q2',T2='T2',Ts='TSK',snow='SNOW',snowh='SNOWH', # original (WRF) names of variables
#               rainnc='RAINNC',rainc='RAINC',rainsh='RAINSH',snownc='SNOWNC',graupelnc='GRAUPELNC',
#               hfx='HFX',lhfx='LH',evap='QFX',OLR='OLR',GLW='GLW',SWD='SWDOWN',SWN='SWNORM') 

# the projection and grid configuration will be inferred from the source file upon loading;
# the axes variables are created on-the-fly and coordinate values are inferred from the source dimensions     


# data source/location
fileclasses = dict(const=Const(), srfc=Srfc(), hydro=Hydro(), axes=Axes())
root_folder = data_root + 'WRF/Downscaling/' # long-term mean folder


## Functions to load different types of WRF datasets

# Time-Series (monthly)
def loadWRF_TS(experiment=None, name=None, domain=2, filetypes=['hydro','const'], varlist=None, varatts=None):
  ''' Get a properly formatted WRF dataset with monthly time-series. '''
  # figure out experiment name
  if experiment is None:
    if name is not None: experiment = name
    else: raise DatasetError, "Need to specify an experiment name in order to load data."
  if not isinstance(experiment,basestring): raise TypeError
  if name is None: name = '{0:s}_d{1:0=2d}'.format(experiment,domain)
  elif not isinstance(name,basestring): raise TypeError 
  # prepare input  
  folder = root_folder + '{0:s}/'.format(experiment)
  # generate filelist and attributes based on filetypes and domain
  atts = dict(); filelist = [] 
  for filetype in filetypes + ['axes']:
    fileclass = fileclasses[filetype]
    if fileclass.tsfile is not None: filelist.append(fileclass.tsfile.format(domain)) 
    atts.update(fileclass.atts)  
  if varatts is not None: atts.update(varatts)
  # translate varlist
  if varlist is None: varlist = atts.keys()
  elif varatts: varlist = translateVarNames(varlist, varatts)
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=atts, 
                          multifile=False, ncformat='NETCDF4', squeeze=True)
  # figure out horizontal coordinates (from number of grid points and resolution)
  dx = dataset.atts.DX; nx = len(dataset.x); dy = dataset.atts.DY; ny = len(dataset.y)
  x = np.arange((1-nx)*dx/2, (nx+1)*dx/2, dx)
  assert x.mean()==0 and x[0]==-1*x[-1] # print x.mean(), x[0], x[-1] # centered
  y = np.arange((1-ny)*dy/2, (ny+1)*dy/2, dy)
  assert y.mean()==0 and y[0]==-1*y[-1] # print y.mean(), y[0], y[-1] # centered
  #xax = Axis(name='x', units='m', coord=x); dataset.replaceAxis('x', xax)
  #yax = Axis(name='y', units='m', coord=y); dataset.replaceAxis('y', yax)
  dataset.x.updateCoord(coord=x); dataset.y.updateCoord(coord=y)
  assert all([dataset.axes['x'] == var.getAxis('x') for var in dataset.variables.values()])
  assert all([dataset.axes['y'] == var.getAxis('y') for var in dataset.variables.values()]) 
  # N.B.: unlike with other datasets, the projection has to be inferred from the dataset  
  # add projection
  projection = getWRFproj(dataset.dataset, name='{0:s} Coordinate System'.format(name))
  print dataset
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None)
  # return formatted dataset
  return dataset
  

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'narravg/' 
# function to load these files...
def loadWRF(name=None, domain=None, period=None, grid=None, varlist=None):
  ''' Get the pre-processed monthly NARR climatology as a DatasetNetCDF. '''
  avgfolder = data_root + name + '/' # long-term mean folder
  # prepare input
  if domain not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # load variables separately
  if 'p' in varlist:
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=['normals_v2011_%s.nc'%resolution], varlist=['p'], 
                            varatts=varatts, ncformat='NETCDF4_CLASSIC')
  if 's' in varlist: 
    gauges = nc.Dataset(folder+'normals_gauges_v2011_%s.nc'%resolution, mode='r', format='NETCDF4_CLASSIC')
    stations = Variable(data=gauges.variables['p'][0,:,:], axes=(dataset.lat,dataset.lon), **varatts['s'])
    # consolidate dataset
    dataset.addVariable(stations, asNC=False, copy=True)  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # return formatted dataset
  return dataset


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
#   mode = 'test_climatology'
  mode = 'test_timeseries'
#   mode = 'average_timeseries'
  grid = '25'
  period = (1979,1981)

  
  # load averaged climatology file
  if mode == 'test_climatology':
    
    print('')
    dataset = loadWRF(period=period)
    print(dataset)
    print('')
    print(dataset.geotransform)
  
  
  # load monthly time-series file
  elif mode == 'test_timeseries':
    
    print('')
    dataset = loadWRF_TS(experiment='max-ctrl', domain=1)
    print(dataset)
    print('')
    print(dataset.geotransform)
    
                        
  # generate averaged climatology
  elif mode == 'average_timeseries':
    
    # load source
    periodstr = '%4i-%4i'%period
    print('\n')
    print('   ***   Processing Grid %s from %s   ***   '%(grid,periodstr))
    print('\n')
    source = loadWRF_TS()
    print(source)
    print('\n')
    # prepare sink
    gridstr = '' if grid is 'NARR' else '_'+grid
    filename = avgfile%(gridstr,'_'+periodstr)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
    sink = DatasetNetCDF(name='NARR Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
    sink.atts.period = periodstr 
    
    # determine averaging interval
    offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, varlist=['precip', 'T2'], tmp=True) # no need for lat/lon
    
    # start processing climatology
    print('')
    print('   +++   processing climatology   +++   ') 
    CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
    print('\n')      


    if grid != 'NARR':    
      # find new coordinate arrays
      if grid == '025': dlon = dlat = 0.25 # resolution
      elif grid == '05': dlon = dlat = 0.5
      elif grid == '10': dlon = dlat = 1.0
      elif grid == '25': dlon = dlat = 2.5 
      slon, slat, elon, elat = -179.75, 3.25, -69.75, 85.75
      assert (elon-slon) % dlon == 0 
      lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
      assert (elat-slat) % dlat == 0
      lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
      # add new geographic coordinate axes for projected map
      xlon = Axis(coord=lon, atts=dict(name='lon', long_name='longitude', units='deg E'))
      ylat = Axis(coord=lat, atts=dict(name='lat', long_name='latitude', units='deg N'))
      # reproject and resample (regrid) dataset
      print('')
      print('   +++   processing regidding   +++   ') 
      print('    ---   (%3.2f,  %3i x %3i)   ---   '%(dlon, len(lon), len(lat)))
      CPU.Regrid(xlon=xlon, ylat=ylat, flush=False)
      print('\n')
    
    
    # sync temporary storage with output
    CPU.sync(flush=True)

#     # make new masks
#     sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=True, merge=False)

    # add names and length of months
    sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                        atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
    #print '   ===   month   ===   '
    sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                  atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
    
    # close...
    sink.sync()
    sink.close()
    # print dataset
    print('')
    print(sink)     
    