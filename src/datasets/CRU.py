'''
Created on 2013-09-09

This module contains meta data and access functions for the monthly CRU time-series data. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os
# internal imports
from geodata.base import Variable, Axis
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root 
from datasets.common import loadObservations, grid_folder, transformPrecip, transformDays, timeSlice
from processing.process import CentralProcessingUnit

 
## CRU Meta-data

dataset_name = 'CRU'
root_folder = data_root + dataset_name + '/'

# CRU grid definition           
geotransform = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
size = (720, 360) # (x,y) map size of CRU grid
# make GridDefinition instance
CRU_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform, size=size)

# variable attributes and name
varatts = dict(tmp = dict(name='T2', units='K', offset=273.15), # 2m average temperature
               tmn = dict(name='Tmin', units='K', offset=273.15), # 2m minimum temperature
               tmx = dict(name='Tmax', units='K', offset=273.15), # 2m maximum temperature
               dtr = dict(name='dTd', units='K', offset=0.), # diurnal 2m temperature range
               vap = dict(name='Q2', units='Pa', scalefactor=100.), # 2m water vapor pressure
               pet = dict(name='pet', units='kg/m^2/s', scalefactor=1./86400.), # potential evapo-transpiration
               pre = dict(name='precip', units='mm/month', transform=transformPrecip), # total precipitation
               cld = dict(name='cldfrc', units='', offset=0.), # cloud cover/fraction
               wet = dict(name='wetfrq', units='days', transform=transformDays), # number of wet days
               frs = dict(name='frzfrq', units='days', transform=transformDays), # number of frost days 
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', offset=-28854), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field

# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
tsvaratts = varatts
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    
# variable and file lists settings
nofile = ('lat','lon','time') # variables that don't have their own files


## Functions to load different types of GPCC datasets 

# Time-series (monthly)
orig_ts_folder = root_folder + 'Time-series 3.2/data/' # monthly subfolder
orig_ts_file = 'cru_ts3.20.1901.2011.{0:s}.dat.nc' # file names, need to extend with variable name (original)
tsfile = 'cru{0:s}_monthly.nc' # extend with grid type only
def loadCRU_TS(name=dataset_name, grid=None, varlist=None, resolution=None, varatts=None, filelist=None, 
               folder=None, lautoregrid=None):
  ''' Get a properly formatted  CRU dataset with monthly mean time-series. '''
  if grid is None:
    # load from original time-series files 
    if folder is None: folder = orig_ts_folder
    # translate varlist
    if varatts is None: varatts = tsvaratts.copy()
    if varlist is None: varlist = varatts.keys()
    if varlist and varatts: varlist = translateVarNames(varlist, varatts)
    # assemble filelist
    if filelist is None: # generate default filelist
      filelist = [orig_ts_file.format(var) for var in varlist if var not in nofile]
    # load dataset
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                            multifile=False, ncformat='NETCDF4_CLASSIC')
    # replace time axis with number of month since Jan 1979 
    data = np.arange(0,len(dataset.time),1, dtype='int16') + (1901-1979)*12 # month since 1979 (Jan 1979 = 0)
    timeAxis = Axis(name='time', units='month', coord=data, atts=dict(long_name='Month since 1979-01'))
    dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
    # add projection  
    dataset = addGDALtoDataset(dataset, projection=None, geotransform=None, gridfolder=grid_folder)
    # N.B.: projection should be auto-detected as geographic    
  else:
    # load from neatly formatted and regridded time-series files
    if folder is None: folder = avgfolder
    dataset = loadObservations(name=name, folder=folder, projection=None, resolution=None, grid=grid, 
                               period=None, varlist=varlist, varatts=varatts, filepattern=tsfile, 
                               filelist=filelist, lautoregrid=lautoregrid, mode='time-series')
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'cruavg/' 
avgfile = 'cru{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadCRU(name=dataset_name, period=None, grid=None, resolution=None, varlist=None, varatts=None, 
            folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly CRU climatology as a DatasetNetCDF. '''
  # load standardized climatology dataset with CRU-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, period=period, grid=grid, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                             lautoregrid=lautoregrid, mode='climatology')
  # return formatted dataset
  return dataset

# function to load station climatologies
def loadCRU_Stn(name=dataset_name, period=None, station=None, resolution=None, varlist=None, varatts=None, 
                folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly CRU climatology as a DatasetNetCDF. '''
  # load standardized climatology dataset with CRU-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, period=period, station=station, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                             lautoregrid=False, mode='climatology')
  # return formatted dataset
  return dataset

# function to load station time-series
def loadCRU_StnTS(name=dataset_name, station=None, resolution=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly CRU climatology as a DatasetNetCDF. '''
  # load standardized time-series dataset with CRU-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, period=None, station=station, 
                             varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                             lautoregrid=False, mode='time-series')
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = orig_ts_file # filename pattern: variable name and resolution
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: variable name and resolution
data_folder = avgfolder # folder for user data
grid_def = {'':CRU_grid} # standardized grid dictionary
LTM_grids = [] # grids that have long-term mean data 
TS_grids = [''] # grids that have time-series data
grid_res = {'':0.5} # no special name, since there is only one...
default_grid = CRU_grid
# grid_def = {0.5:CRU_grid} # standardized grid dictionary, addressed by grid resolution
# grid_tag = {0.5:''} # no special name, since there is only one...
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = loadCRU_TS # time-series data
loadClimatology = loadCRU # pre-processed, standardized climatology
loadStationClimatology = loadCRU_Stn # climatologies without associated grid (e.g. stations or basins) 
loadStationTimeSeries = loadCRU_StnTS # time-series without associated grid (e.g. stations or basins)

## (ab)use main execution for quick test
if __name__ == '__main__':
    
#   mode = 'test_climatology'
#   mode = 'test_timeseries'
#   mode = 'test_station_timeseries'
  mode = 'average_timeseries'
#   period = (1971,2001)
  period = (1979,2009)
#   period = (1949,2009)
#   period = (1979,1982)
#   period = (1979,1984)
#   period = (1979,1989)
#   period = (1979,1994)
#   period = (1984,1994)
#   period = (1989,1994)
#   period = (1979,1980)
#   period = (1997,1998)
#   period = (2010,2011)

  if mode == 'test_climatology':
    
    # load averaged climatology file
    print('')
    dataset = loadCRU(period=period)
    print(dataset)
    print('')
    print(dataset.geotransform)
    print(dataset.precip.getArray().mean())
    stnds = loadCRU_Stn(station='ecprecip', period=period)
    print(stnds)
    print('')
    
        
  elif mode == 'test_station_timeseries':
    
    # load station time-series file
    print('')
    dataset = loadCRU_StnTS(station='ectemp')
    print(dataset)
    print('')
    print(dataset.time)
    print(dataset.time.coord)
    assert dataset.time.coord[78*12] == 0 # Jan 1979

        
  elif mode == 'test_timeseries':
    
    # load original time-series file
    print('')
    dataset = loadCRU_TS(grid='arb2_d02')
    print(dataset)
    print('')
    print(dataset.time)
    print(dataset.time.coord)
    print(dataset.time.coord[78*12])

        
  elif mode == 'average_timeseries':
      
    # load source
    periodstr = '%4i-%4i'%period
    print('\n')
    print('   ***   Processing Time-series from %s   ***   '%(periodstr,))
    print('\n')
    source = loadCRU_TS()
    source = source(time=timeSlice(period)) # only get relevant time-slice    
    print(source)
    assert period[0] != 1979 or source.time.coord[0] == 0
    assert len(source.time) == (period[1]-period[0])*12
    print('\n')
    # prepare sink
    filename = avgfile.format('','_'+periodstr,)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
    sink = DatasetNetCDF(name='CRU Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
    sink.atts.period = periodstr 
    
    # determine averaging interval
    offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
    # initialize processing
#     CPU = CentralProcessingUnit(source, sink, varlist=['wetfrq'])
    CPU = CentralProcessingUnit(source, sink)
    # start processing      
    print('')
    print('   +++   processing   +++   ') 
    CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
    # sync temporary storage with output
    CPU.sync(flush=False)   
    print('\n')

    # add landmask
    print '   ===   landmask   ===   '
    tmpatts = dict(name='landmask', units='', long_name='Landmask for Climatology Fields', 
              description='where this mask is non-zero, no data is available')
    # find a masked variable
    for var in sink.variables.itervalues():
      if var.masked and var.gdal: 
        mask = var.getMapMask(); break
    # add variable to dataset
    sink.addVariable(Variable(name='landmask', units='', axes=(sink.lat,sink.lon), 
                  data=mask, atts=tmpatts), asNC=True)
    sink.mask(sink.landmask)            
    # add names and length of months
    sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                        atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
    #print '   ===   month   ===   '
    sink.addVariable(Variable(name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                  atts=dict(name='length_of_month',units='days',long_name='Length of Month')), asNC=True)
    
    # close...
    sink.sync()
    sink.close()
    # print dataset
    print('')
    print(sink)     
    
  