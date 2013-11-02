'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import netCDF4 as nc # netcdf python module
import os # check if files are present
import types # to add precip conversion fct. to datasets
from importlib import import_module
# internal imports
from geodata.base import Variable
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition
from geodata.misc import DatasetError
from geodata.nctools import writeNetCDF, add_strvar
from datasets.common import days_per_month, name_of_month, data_root 
from datasets.common import translateVarNames, loadClim, addLandMask, addLengthAndNamesOfMonth, getFileName
from geodata.process import CentralProcessingUnit

## GPCC Meta-data

dataset_name = 'GPCC'

# GPCC grid definition           
geotransform_025 = (-180.0, 0.25, 0.0, -90.0, 0.0, 0.25)
size_025 = (1440,720) # (x,y) map size
geotransform_05 = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
size_05 = (720,360) # (x,y) map size
geotransform_10 = (-180.0, 1.0, 0.0, -90.0, 0.0, 1.0)
size_10 = (360,180) # (x,y) map size
geotransform_25 = (-180.0, 2.5, 0.0, -90.0, 0.0, 2.5)
size_25 = (144,72) # (x,y) map size

# make GridDefinition instance
GPCC_025_grid = GridDefinition(name='GPCC_025',projection=None, geotransform=geotransform_025, size=size_025)
GPCC_05_grid = GridDefinition(name='GPCC_05',projection=None, geotransform=geotransform_05, size=size_05)
GPCC_10_grid = GridDefinition(name='GPCC_10',projection=None, geotransform=geotransform_10, size=size_10)
GPCC_25_grid = GridDefinition(name='GPCC_25',projection=None, geotransform=geotransform_25, size=size_25)


# variable attributes and name
varatts = dict(p    = dict(name='precip', units='mm/month'), # total precipitation rate
               s    = dict(name='stations', units='#'), # number of gauges for observation
               # axes (don't have their own file; listed in axes)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
#                time = dict(name='time', units='days', offset=1)) # time coordinate
# attributes of the time axis depend on type of dataset 
ltmvaratts = dict(time=dict(name='time', units='months', offset=1), **varatts) 
tsvaratts = dict(time=dict(name='time', units='days', offset=-28854), **varatts)
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder


def convertPrecip(precip):
  ''' convert GPCC precip data to SI units (mm/s) '''
  if precip.units == 'kg/m^2/month' or precip.units == 'mm/month':
    precip /= (days_per_month.reshape((12,1,1)) * 86400.) # convert in-place
    precip.units = 'kg/m^2/s'
  return precip      


## Functions to load different types of GPCC datasets 

# climatology
ltmfolder = root_folder + 'climatology/' # climatology subfolder
def loadGPCC_LTM(name=dataset_name, varlist=varlist, resolution='025', varatts=ltmvaratts, filelist=None, folder=ltmfolder):
  ''' Get a properly formatted dataset the monthly accumulated GPCC precipitation climatology. '''
  # prepare input
  if resolution not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
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
  # N.B.: projection should be auto-detected as geographic
  # add method to convert precip from per month to per second
  dataset.convertPrecip = types.MethodType(convertPrecip, dataset.precip)    
  # return formatted dataset
  return dataset

# time-series
tsfolder = root_folder + 'full_data_1900-2010/' # climatology subfolder
def loadGPCC_TS(name=dataset_name, varlist=varlist, resolution='25', varatts=tsvaratts, filelist=None, folder=tsfolder):
  ''' Get a properly formatted dataset with the monthly GPCC time-series. '''
  # prepare input  
  if resolution not in ('05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  if filelist is None: # generate default filelist
    filelist = []
    if 'p' in varlist: filelist.append('full_data_v6_precip_%s.nc'%resolution)
    if 's' in varlist: filelist.append('full_data_v6_statio_%s.nc'%resolution)
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # add method to convert precip from per month to per second
  dataset.convertPrecip = types.MethodType(convertPrecip, dataset.precip)    
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'gpccavg/' 
avgfile = 'gpcc{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadGPCC(name=dataset_name, period=None, grid=None, resolution=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly GPCC climatology as a DatasetNetCDF. '''
  # prepare input
  if grid is not None and grid[0:5].lower() == 'gpcc_': 
    resolution = grid[5:]
    grid = None
  elif resolution is None: 
    resolution = '025' if period is None else '05'
  # check resolution
  if grid is None:
    # check for valid resolution 
    if resolution not in ('025','05', '10', '25'): 
      raise DatasetError, "Selected resolution '%s' is not available!"%resolution  
    if resolution == '025' and period is not None: 
      raise DatasetError, "The highest resolution is only available for the lon-term mean!"
    grid = resolution # grid supersedes resolution  
  # load standardized climatology dataset with GPCC-specific parameters
  dataset = loadClim(name=name, folder=folder, projection=None, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
grid_def = {0.25:GPCC_025_grid, 0.5:GPCC_05_grid, 1.0:GPCC_10_grid, 2.5:GPCC_25_grid}
grid_tag = {0.25:'025', 0.5:'05', 1.0:'10', 2.5:'25'} 
# functions to access specific datasets
loadLongTermMean = loadGPCC_LTM # climatology provided by publisher
loadTimeSeries = loadGPCC_TS # time-series data
loadClimatology = loadGPCC # pre-processed, standardized climatology


## (ab)use main execution for quick test
if __name__ == '__main__':
  
#   mode = 'test_climatology'; reses = ('025',); period = None
  mode = 'average_timeseries'; reses = ('05',) # for testing
#   mode = 'convert_climatology'; reses = ('025',); period = None
  reses = ('025','05', '10', '25')  
#   reses = ('05', '10', '25')
  reses = ('25',)
  period = (1979,1989)
  grid = 'GPCC'
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_climatology':
      
      
      # load averaged climatology file
      print('')
      dataset = loadGPCC(grid='%s_%s'%(grid,res),resolution=res,period=period)
      print(dataset)
      print('')
      print(dataset.geotransform)
      print(dataset.precip.getArray().mean())
      print(dataset.precip.masked)
      
          
    elif mode == 'convert_climatology':      
      
      
      # load dataset
      dataset = loadGPCC_LTM(varlist=['stations','precip'],resolution=res)
      # change meta-data
      dataset.name = 'GPCC'
      dataset.title = 'GPCC Long-term Climatology'
      dataset.atts.resolution = res      
      # load data into memory
      dataset.load()

      # convert precip data to SI units (mm/s)
      dataset.convertPrecip() # convert in-place
      # add landmask
      addLandMask(dataset) # create landmask from precip mask
      dataset.mask(dataset.landmask) # mask all fields using the new landmask      
      # add length and names of month
      addLengthAndNamesOfMonth(dataset, noleap=False) 
      
      # figure out a different filename
      filename = getFileName(grid=res, period=period, name='GPCC', filepattern=avgfile)
      print('\n'+filename+'\n')      
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)      
      # write data and some annotation
      ncset = writeNetCDF(dataset, avgfolder+filename, close=False)
      add_strvar(ncset,'name_of_month', name_of_month, 'time', # add names of month
                 atts=dict(name='name_of_month', units='', long_name='Name of the Month')) 
       
      # close...
      ncset.close()
      dataset.close()
      # print dataset before
      print(dataset)
      print('')           
      
      
    elif mode == 'average_timeseries':
      
      
      # load source
      periodstr = 'Climatology' if period is None else '{0:4d}-{1:4d}'.format(*period)
      print('\n\n   ***   Processing Resolution %s from %s   ***   \n\n'%(res,periodstr))
      if period is None: source = loadGPCC_LTM(varlist=['stations','precip'],resolution=res)
      else: source = loadGPCC_TS(varlist=['stations','precip'],resolution=res)
      source.load()
      print(source)
      print('\n')
            
      # prepare sink
      filename = getFileName(grid=grid, period=period, name='GPCC', filepattern=avgfile)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      atts =dict(period=periodstr, name='GPCC', title='GPCC Climatology') 
      sink = DatasetNetCDF(name='GPCC Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
      
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, tmp=True)

      if period is not None:
        # determine averaging interval
        offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
        # start processing climatology
        CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
      
      # get NARR coordinates
      if grid is not 'GPCC':
        new_grid = import_module(grid[0:4]).__dict__[grid+'_grid']
#       if grid == 'NARR':
#         from datasets.NARR import NARR_grid
        # reproject and resample (regrid) dataset
        CPU.Regrid(griddef=new_grid, flush=False)
            
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift longitude   +++   ') 
#       CPU.Shift(lon=-180, flush=True)
#       print('\n')
#
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift/roll   +++   ') 
#       CPU.Shift(shift=72, axis='lon', byteShift=True, flush=False)
#       print('\n')      

      # get results
      CPU.sync(flush=True)
      # convert precip data to SI units (mm/s) 
      convertPrecip(sink.precip) # convert in-place
      # add landmask
      addLandMask(sink) # create landmask from precip mask
      sink.stations.mask(sink.landmask) # mask all fields using the new landmask
      # add length and names of month
      addLengthAndNamesOfMonth(sink, noleap=False) 
              
#       newvar = sink.precip
#       print
#       print newvar.name, newvar.masked
#       print newvar.fillValue
#       print newvar.data_array.__class__
#       print
      
      # close...
      sink.sync()
      sink.close()
      # print dataset
      print('')
      print(sink)     
      
    