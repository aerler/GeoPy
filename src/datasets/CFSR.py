'''
Created on 2013-09-12

This module contains meta data and access functions for the monthly CFSR time-series data.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os
# internal imports
from geodata.netcdf import DatasetNetCDF, Axis
from geodata.misc import DatasetError
from geodata.gdal import addGDALtoDataset, GridDefinition
from datasets.common import translateVarNames, name_of_month, data_root, loadObservations, grid_folder
from processing.process import CentralProcessingUnit


## CRU Meta-data

dataset_name = 'CFSR'
root_folder = '{:s}/{:s}/'.format(data_root,dataset_name) # long-term mean folder

# CFSR grid definition           
# geotransform_031 = (-180.15625, 0.3125, 0.0, 89.915802001953125, 0.0, -0.30960083)
geotransform_031 = (-0.15625, 0.3125, 0.0, 89.915802001953125, 0.0, -0.30960083)
size_031 = (1152,576) # (x,y) map size
# geotransform_05 = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
geotransform_05 = (-0.25, 0.5, 0.0, 90.25, 0.0, -0.5) # this grid actually has a grid point at the poles!
size_05 = (720,361) # (x,y) map size

# make GridDefinition instance
CFSR_031_grid = GridDefinition(name='CFSR_031', projection=None, geotransform=geotransform_031, size=size_031, lwrap360=True)
CFSR_05_grid = GridDefinition(name='CFSR_05', projection=None, geotransform=geotransform_05, size=size_05, lwrap360=True)
CFSR_grid = CFSR_031_grid # default

# variable attributes and name
varatts = dict(TMP_L103_Avg = dict(name='T2', units='K'), # 2m average temperature
               TMP_L1 = dict(name='Ts', units='K'), # average skin temperature
               PRATE_L1 = dict(name='precip', units='kg/m^2/s'), # total precipitation
               PRES_L1 = dict(name='ps', units='Pa'), # surface pressure
               WEASD_L1 = dict(name='snow', units='kg/m^2'), # snow water equivalent
               SNO_D_L1 = dict(name='snowh', units='m'), # snow depth
               # lower resolution
               PRMSL_L101 = dict(name='pmsl', units='Pa'), # sea-level pressure
               # static fields (full resolution)
               LAND_L1 = dict(name='landmask', units=''), # land mask
               HGT_L1 = dict(name='zs', units='m'), # surface elevation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', scalefactor=1/24., offset=-6.), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
# N.B.: the time-series begins in 1979 (time=0), so no offset is necessary
tsvaratts = varatts

nofile = ('lat','lon','time') # variables that don't have their own files
# file names by variable
hiresfiles = dict(TMP_L103_Avg='TMP.2m', TMP_L1='TMP.SFC', PRATE_L1='PRATE.SFC', PRES_L1='PRES.SFC',
                 WEASD_L1='WEASD.SFC', SNO_D_L1='SNO_D.SFC')
hiresstatic = dict(LAND_L1='LAND.SFC', HGT_L1='HGT.SFC')
lowresfiles = dict(PRMSL_L101='PRMSL.MSL')
lowresstatic = dict() # currently none available
hiresfiles = {key:'flxf06.gdas.{0:s}.grb2.nc'.format(value) for key,value in hiresfiles.iteritems()}
hiresstatic = {key:'flxf06.gdas.{0:s}.grb2.nc'.format(value) for key,value in hiresstatic.iteritems()}
lowresfiles = {key:'pgbh06.gdas.{0:s}.grb2.nc'.format(value) for key,value in lowresfiles.iteritems()}
lowresstatic = {key:'pgbh06.gdas.{0:s}.grb2.nc'.format(value) for key,value in lowresstatic.iteritems()}
# N.B.: to trim time dimension to the common length, use: 
#       ncks -a -O -d time,0,371 orig_flxf06.gdas.TMP.2m.grb2.nc flxf06.gdas.TMP.2m.grb2.nc
# list of variables to load
# varlist = ['precip','snowh'] + hiresstatic.keys() + list(nofile) # hires + coordinates
varlist_hires = hiresfiles.keys() + hiresstatic.keys() + list(nofile) # hires + coordinates    
varlist_lowres = lowresfiles.keys() + lowresstatic.keys() + list(nofile) # hires + coordinates



## Functions to load different types of CFSR datasets 

def checkGridRes(grid, resolution):
  ''' helper function to verify grid/resoluton selection ''' 
  # prepare input
  if grid is not None and grid[0:5].lower() == 'cfsr_': 
    resolution = grid[5:]
    grid = None
  elif resolution is None: resolution = '031'
  # check for valid resolution
  if resolution == 'hires' or resolution == '03': resolution = '031' 
  elif resolution == 'lowres': resolution = '05' 
  elif resolution not in ('031','05'): 
    raise DatasetError, "Selected resolution '{0:s}' is not available!".format(resolution)  
  # return
  return grid, resolution

# time-series
orig_ts_folder = root_folder + 'Monthly/'
tsfile = 'cfsr{0:s}_monthly.nc' # extend with grid type only
def loadCFSR_TS(name=dataset_name, grid=None, varlist=None, varatts=None, resolution='hires', 
                filelist=None, folder=None, lautoregrid=None):
  ''' Get a properly formatted CFSR dataset with monthly mean time-series. '''
  if grid is None:
    # load from original time-series files 
    if folder is None: folder = orig_ts_folder
    # translate varlist
    if varatts is None: varatts = tsvaratts.copy()
    if varlist is None:
      if resolution == 'hires' or resolution == '03' or resolution == '031': varlist = varlist_hires
      elif resolution == 'lowres' or resolution == '05': varlist = varlist_lowres     
    if varlist and varatts: varlist = translateVarNames(varlist, varatts)
    if filelist is None: # generate default filelist
      if resolution == 'hires' or resolution == '03' or resolution == '031': 
        files = [hiresfiles[var] for var in varlist if var in hiresfiles]
      elif resolution == 'lowres' or resolution == '05': 
        files = [lowresfiles[var] for var in varlist if var in lowresfiles]
    # load dataset
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=files, varlist=varlist, varatts=varatts, 
                            check_override=['time'], multifile=False, ncformat='NETCDF4_CLASSIC')
    # load static data
    if filelist is None: # generate default filelist
      if resolution == 'hires' or resolution == '03' or resolution == '031': 
        files = [hiresstatic[var] for var in varlist if var in hiresstatic]
      elif resolution == 'lowres' or resolution == '05': 
        files = [lowresstatic[var] for var in varlist if var in lowresstatic]
      # load constants, if any (and with singleton time axis)
      if len(files) > 0:
        staticdata = DatasetNetCDF(name=name, folder=folder, filelist=files, varlist=varlist, varatts=varatts, 
                                   axes=dict(lon=dataset.lon, lat=dataset.lat), multifile=False, 
                                   check_override=['time'], ncformat='NETCDF4_CLASSIC')
        # N.B.: need to override the axes, so that the datasets are consistent
        if len(staticdata.variables) > 0:
          for var in staticdata.variables.values(): 
            if not dataset.hasVariable(var.name):
              var.squeeze() # remove time dimension
              dataset.addVariable(var, copy=False) # no need to copy... but we can't write to the netcdf file!
    # replace time axis with number of month since Jan 1979 
    data = np.arange(0,len(dataset.time),1, dtype='int16') # month since 1979 (Jan 1979 = 0)
    timeAxis = Axis(name='time', units='month', coord=data, atts=dict(long_name='Month since 1979-01'))
    dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
    # add projection  
    dataset = addGDALtoDataset(dataset, projection=None, geotransform=None, gridfolder=grid_folder)
    # N.B.: projection should be auto-detected as geographic
  else:
    # load from neatly formatted and regridded time-series files
    if folder is None: folder = avgfolder
    grid, resolution = checkGridRes(grid, resolution)
    dataset = loadObservations(name=name, folder=folder, projection=None, resolution=resolution, grid=grid, 
                               period=None, varlist=varlist, varatts=varatts, filepattern=tsfile, 
                               filelist=filelist, lautoregrid=lautoregrid, mode='time-series')
  # return formatted dataset
  return dataset


# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'cfsravg/' 
avgfile = 'cfsr{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadCFSR(name=dataset_name, period=None, grid=None, resolution='031', varlist=None, varatts=None, 
             folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly CFSR climatology as a DatasetNetCDF. '''
  grid, resolution = checkGridRes(grid=grid, resolution=resolution)
  # load standardized climatology dataset with CFSR-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, resolution=resolution,
                             period=period, grid=grid, shape=None, station=None, 
                             varlist=varlist, varatts=varatts, filelist=filelist, 
                             filepattern=avgfile, lautoregrid=lautoregrid, mode='climatology')
  # return formatted dataset
  return dataset

# function to load station climatologies
def loadCFSR_Stn(name=dataset_name, period=None, station=None, resolution=None, varlist=None, varatts=None, 
                 folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly CFSR climatology at station locations as a DatasetNetCDF. '''
  grid, resolution = checkGridRes(None, resolution); del grid
  # load standardized climatology dataset with -specific parameters
  dataset = loadObservations(name=name, folder=folder, period=period, station=station, shape=None, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, projection=None, mode='climatology')
  # return formatted dataset
  return dataset

# function to load station time-series
def loadCFSR_StnTS(name=dataset_name, station=None, resolution=None, varlist=None, varatts=None, 
                   folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly CFSR time-series at station locations as a DatasetNetCDF. '''
  grid, resolution = checkGridRes(None, resolution); del grid
  # load standardized time-series dataset with -specific parameters
  dataset = loadObservations(name=name, folder=folder, period=None, station=station, shape=None,
                             varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, projection=None, mode='time-series')
  # return formatted dataset
  return dataset

# function to load averaged climatologies
def loadCFSR_Shp(name=dataset_name, period=None, shape=None, resolution=None, varlist=None, varatts=None, 
                 folder=avgfolder, filelist=None, lautoregrid=True, lencl=False):
  ''' Get the pre-processed monthly CFSR climatology averaged over regions as a DatasetNetCDF. '''
  grid, resolution = checkGridRes(None, resolution); del grid
  # load standardized climatology dataset with -specific parameters
  dataset = loadObservations(name=name, folder=folder, period=period, station=None, shape=shape, lencl=lencl, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, projection=None, mode='climatology')
  # return formatted dataset
  return dataset

# function to load averaged time-series
def loadCFSR_ShpTS(name=dataset_name, shape=None, resolution=None, varlist=None, varatts=None, 
                   folder=avgfolder, filelist=None, lautoregrid=True, lencl=False):
  ''' Get the pre-processed monthly CFSR time-series averaged over regions as a DatasetNetCDF. '''
  grid, resolution = checkGridRes(None, resolution); del grid
  # load standardized time-series dataset with -specific parameters
  dataset = loadObservations(name=name, folder=folder, period=None, station=None, shape=shape, lencl=lencl,
                             varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, projection=None, mode='time-series')
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = '{0:s}{1:s}06.gdas.{2:s}.{3:s}.grb2.nc' # filename pattern: type, resolution, variable name, and level 
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: grid, and period
data_folder = avgfolder # folder for user data
grid_def = {'031':CFSR_031_grid, '05':CFSR_05_grid}
LTM_grids = [] # grids that have long-term mean data
TS_grids = ['031','05'] # grids that have time-series data
grid_res = {'031':0.31, '05':0.5} # tag used in climatology files
default_grid = CFSR_031_grid
# grid_def = {0.31:CFSR_031_grid, 0.5:CFSR_05_grid}  # standardized grid dictionary, addressed by grid resolution
# grid_tag = {0.31:'031', 0.5:'05'} # tag used in climatology files
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadClimatology = loadCFSR # pre-processed, standardized climatology
loadTimeSeries = loadCFSR_TS # time-series data
loadStationClimatology = loadCFSR_Stn # climatologies without associated grid (e.g. stations or basins) 
loadStationTimeSeries = loadCFSR_StnTS # time-series without associated grid (e.g. stations or basins)
loadShapeClimatology = loadCFSR_Shp # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries = loadCFSR_ShpTS # time-series without associated grid (e.g. provinces or basins)


## (ab)use main execution for quick test
if __name__ == '__main__':
  
#   mode = 'test_climatology'
  mode = 'average_timeseries'
#   mode = 'test_timeseries'
#   mode = 'test_point_climatology'
#   mode = 'test_point_timeseries'
#   reses = ('05',) # for testing
  reses = ( '031','05',)
#   period = (1979,1984)
#   period = (1979,1989)
  period = (1979,1994)
#   period = (1997,1998)
#   period = (1979,2009)
#   period = (2010,2011) 
#   grid = 'arb1_d01'
  pntset = 'shpavg' # 'ecprecip'
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_climatology':
    
      
      # load averaged climatology file
      print('')
      dataset = loadCFSR(resolution=res,period=period)
      print(dataset)
      print('')
      print(dataset.geotransform)
    
              
    elif mode == 'test_timeseries':
    
      
      # load averaged climatology file
      print('')
      dataset = loadCFSR_TS(resolution=res)
      print(dataset)
#       print('')
#       print(dataset.time)
#       print(dataset.time.coord)
      print('')
      print(dataset.landmask)
      assert dataset.landmask.gdal
    
    elif mode == 'test_point_climatology':
      
      # load station time-series file
      print('')
      if pntset in ('shpavg',): dataset = loadCFSR_Shp(shape=pntset, period=period)
      else: dataset = loadCFSR_Stn(station=pntset, period=period)
      print(dataset)
      print('')
      print(dataset.time)
      print(dataset.time.coord)
  
    elif mode == 'test_point_timeseries':
      
      # load station time-series file
      print('')
      if pntset in ('shpavg',): dataset = loadCFSR_ShpTS(shape=pntset)
      else: dataset = loadCFSR_StnTS(station=pntset)
      print(dataset)
      print('')
      print(dataset.time)
      print(dataset.time.coord)
      assert dataset.time.coord[0] == 0 # Jan 1979
      assert dataset.shape[0] == 1

                  
    elif mode == 'average_timeseries':   
      
      # load source
      periodstr = '{0:4d}-{1:4d}'.format(*period)
      print('\n')
      print('   ***   Processing Resolution %s from %s   ***   '%(res,periodstr))
      print('\n')
      source = loadCFSR_TS(resolution=res)
      print(source)
      print('\n')
      # prepare sink
      filename = avgfile.format('_'+res,'_'+periodstr)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      sink = DatasetNetCDF(name='CFSR Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
      sink.atts.period = periodstr 
      
      # determine averaging interval
      offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, tmp=True)
      
      # start processing climatology
      CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
      
      # shift longitude axis by 180 degrees left (i.e. 0 - 360 -> -180 - 180)
      CPU.Shift(lon=-180, flush=False)
      
      # sync temporary storage with output (sink variable; do not flush!)
      CPU.sync(flush=False)

      # make new masks
      if sink.hasVariable('landmask'):
        sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=True, merge=False)

      # add names and length of months
      sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                          atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
      #print '   ===   month   ===   '
#       sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
#                     atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
      
      # close...
      sink.sync()
      sink.close()
      # print dataset
      print('')
      print(sink)     
      