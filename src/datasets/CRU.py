'''
Created on 2013-09-09

This module contains meta data and access functions for the monthly CRU time-series data. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import os
# internal imports
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, GridDefinition
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root, loadClim
from geodata.process import CentralProcessingUnit

 
## CRU Meta-data

dataset_name = 'CRU'

# CRU grid definition           
geotransform = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
size = (720, 360) # (x,y) map size of CRU grid
# make GridDefinition instance
CRU_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform, size=size)

# variable attributes and name
varatts = dict(tmp = dict(name='T2', units='K', offset=273.15), # 2m average temperature
               tmn = dict(name='Tmin', units='K', offset=273.15), # 2m minimum temperature
               tmx = dict(name='Tmax', units='K', offset=273.15), # 2m maximum temperature
               vap = dict(name='Q2', units='Pa', scalefactor=100.), # 2m water vapor pressure
               pet = dict(name='pet', units='kg/m^2/s', scalefactor=1./86400.), # potential evapo-transpiration
               pre = dict(name='precip', units='kg/m^2/s', scalefactor=1./86400.), # total precipitation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', offset=-28854), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field

# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
nofile = ('lat','lon','time') # variables that don't have their own files
filename = 'cru_ts3.20.1901.2011.%s.dat.nc' # file names, need to extend with %varname (original)

# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder


## Functions to load different types of GPCC datasets 

# Time-series (monthly)
tsfolder = root_folder + 'Time-series 3.2/data/' # monthly subfolder
def loadCRU_TS(name=dataset_name, varlist=varlist, varatts=varatts, filelist=None, folder=tsfolder):
  ''' Get a properly formatted  CRU dataset with monthly mean time-series. '''
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # assemble filelist
  if filelist is None: # generate default filelist
    filelist = [filename%var for var in varlist if var not in nofile]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'cruavg/' 
avgfile = 'cru%s_clim%s.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadCRU(name=dataset_name, period=None, grid=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly CRU climatology as a DatasetNetCDF. '''
  # load standardized climatology dataset with CRU-specific parameters
  dataset = loadClim(name=name, folder=folder, projection=None, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
grid_def = {0.5:CRU_grid} # standardized grid dictionary, addressed by grid resolution
grid_tag = {0.5:''} # no special name, since there is only one...
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = loadCRU_TS # time-series data
loadClimatology = loadCRU # pre-processed, standardized climatology

## (ab)use main execution for quick test
if __name__ == '__main__':
    
  mode = 'test_climatology'
#   mode = 'average_timeseries'
  period = (1979,1981)

  if mode == 'test_climatology':
    
    # load averaged climatology file
    print('')
    dataset = loadCRU(period=period)
    print(dataset)
    print('')
    print(dataset.geotransform)

        
  elif mode == 'average_timeseries':
      
    # load source
    periodstr = '%4i-%4i'%period
    print('\n')
    print('   ***   Processing Time-series from %s   ***   '%(periodstr,))
    print('\n')
    source = loadCRU_TS()
    print(source)
    print('\n')
    # prepare sink
    filename = avgfile%('_'+periodstr,)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
    sink = DatasetNetCDF(name='CRU Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
    sink.atts.period = periodstr 
    
    # determine averaging interval
    offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
    # initialize processing
    CPU = CentralProcessingUnit(source, sink)
    # start processing      
    print('')
    print('   +++   processing   +++   ') 
    CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
    print('\n')

    # add landmask
    #print '   ===   landmask   ===   '
    tmpatts = dict(name='landmask', units='', long_name='Landmask for Climatology Fields', 
              description='where this mask is non-zero, no data is available')
    sink += VarNC(sink.dataset, name='landmask', units='', axes=(sink.lat,sink.lon), 
                  data=sink.precip.getMask()[0,:,:], atts=tmpatts)
    sink.mask(sink.landmask)            
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
    
  