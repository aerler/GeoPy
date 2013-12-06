'''
Created on 2013-11-23

@author: Andre R. Erler
'''

# external imports
import numpy as np
import numpy.ma as ma
#import netCDF4 as nc # netcdf python module
import os
# internal imports
from geodata.base import Variable
from geodata.netcdf import DatasetNetCDF
from datasets.common import days_per_month, name_of_month, getFileName, data_root, loadClim, grid_folder
from geodata.gdal import loadPickledGridDef
from datasets import loadGPCC, loadCRU, loadPRISM
# from geodata.misc import DatasetError
from warnings import warn

## Unity Meta-data

dataset_name = 'Unity'
# N.B.: doesn't have a native grid!

# variable attributes and name (basically no alterations necessary...)
varatts = dict(# PRISM variables
               T2 = dict(name='T2', units='K', atts=dict(long_name='Average 2m Temperature')), # 2m average temperature
               Tmin = dict(name='Tmin', units='K', atts=dict(long_name='Minimum 2m Temperature')), # 2m minimum temperature
               Tmax = dict(name='Tmax', units='K', atts=dict(long_name='Maximum 2m Temperature')), # 2m maximum temperature
               precip = dict(name='precip', units='kg/m^2/s', atts=dict(long_name='Total Precipitation')), # total precipitation
               # precip is merged between PRISM and GPCC
               # the rest is from CRU
               Q2 = dict(name='Q2', units='Pa', scalefactor=100.), # 2m water vapor pressure
               pet = dict(name='pet', units='kg/m^2/s', scalefactor=1./86400.), # potential evapo-transpiration
               cldfrc = dict(name='cldfrc', units='', offset=0.), # cloud cover/fraction
               wetfrq = dict(name='wetfrq', units='', offset=0), # number of wet days
               frsfrq = dict(name='frzfrq', units='', offset=0), # number of frost days 
               # axes (don't have their own file; listed in axes)
               time=dict(name='time', units='month', atts=dict(long_name='Month of the Year')), # time coordinate
               lon  = dict(name='lon', units='deg E', atts=dict(long_name='Longitude')), # geographic longitude field
               lat  = dict(name='lat', units='deg N', atts=dict(long_name='Latitude'))) # geographic latitude field
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    
# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder


## Functions that provide access to well-formatted PRISM NetCDF files

# pre-processed climatology files (varatts etc. should not be necessary)
avgfile = 'unity{0:s}_clim{1:s}.nc' # formatted NetCDF file
avgfolder = root_folder + 'unityavg/' # prefix
# function to load these files...
def loadUnity(name=dataset_name, period=None, grid=None, resolution=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly PRISM climatology as a DatasetNetCDF. '''
  # a climatology is not available
  if period is None: 
    period = (1979,2009)
    warn('A climatology is not available for the Unified Dataset; loading period {0:4d}-{1:4d}.'.format(*period))
  # this dataset has not native/default grid
  if grid is None: 
    grid = 'arb2_d02'
    warn('The Unified Dataset has no native grid; loading {0:s} grid.'.format(grid))
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadClim(name=name, folder=folder, projection=None, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
LTM_grids = None # grids that have long-term mean data 
TS_grids = None # grids that have time-series data
grid_def = {'d02':None,'d01':None} # there are too many... 
grid_res = {'d02':0.13,'d01':3.82} # approximate grid resolution at 45 degrees latitude 
default_grid = None
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = None # time-series data
loadClimatology = loadUnity # pre-processed, standardized climatology


if __name__ == '__main__':
  
  # select mode
  mode = 'merge_datasets'
#   mode = 'test_climatology'
  
  # settings to generate dataset
  grid = 'arb2_d01'
#   grid = 'arb2_d02'
#   grid = 'ARB_small_025'
#   grid = 'cesm1x1'
#   period = (1979,1984)
#   period = (1979,1989)
#   period = (1997,1998)
#   period = (1979,1980)
  period = (1979,2009)

  
  ## do some tests
  if mode == 'test_climatology':  
    
    # load NetCDF dataset
#     dataset = loadUnity(grid='arb2_d02', period=period)
    dataset = loadUnity()
    print(dataset)
    print('')
    print(dataset.geotransform)
    print(dataset.precip.getArray().mean())
    print(dataset.precip.masked)
    print('')
    # display
    import pylab as pyl
    pyl.imshow(np.flipud(dataset.prismmask.getArray()[:,:])) 
    pyl.colorbar(); pyl.show(block=True)

  
  ## begin processing
  if mode == 'merge_datasets':
    # produce a merged dataset for a given time period and grid    
    
    ## load source datasets
    prism = loadPRISM(period=None, grid=grid, varlist=['T2','Tmin','Tmax','precip','datamask','lon2D','lat2D'])
    gpccprd = loadGPCC(period=period, resolution='05', grid=grid, varlist=['precip'])
    gpccclim = loadGPCC(period=None, resolution='05', grid=grid, varlist=['precip'])
    gpcc025 = loadGPCC(period=None, resolution='025', grid=grid, varlist=['precip','landmask'])
    cruprd = loadCRU(period=period, grid=grid, varlist=['T2','Tmin','Tmax','Q2','pet','cldfrc','wetfrq','frzfrq'])
    cruclim = loadCRU(period=(1979,2009), grid=grid, varlist=['T2','Tmin','Tmax','Q2','pet','cldfrc','wetfrq','frzfrq'])
    
    # grid definition
    griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder)
    periodstr = '{0:4d}-{1:4d}'.format(*period)
    
    print('\n   <<<   Merging Climatology from {0:s} on {1:s} Grid  >>>   \n'.format(periodstr,grid,))
    ## prepare target dataset 
    filename = getFileName(grid=griddef.name, period=period, name=None, filepattern=avgfile)
    filepath = avgfolder + filename
    print('\n Saving data to: \'{0:s}\'\n'.format(filepath))
    assert os.path.exists(avgfolder)
    if os.path.exists(filepath): os.remove(filepath) # remove old file
    # set attributes   
    atts=dict() # collect attributes, but add prefixes
    for key,item in prism.atts.iteritems(): atts['PRISM_'+key] = item
    #for key,item in gpcc025.atts.iteritems(): atts['GPCC_'+key] = item # GPCC atts cause problems... 
    for key,item in cruprd.atts.iteritems(): atts['CRU_'+key] = item
    atts['period'] = periodstr; atts['name'] = dataset_name; atts['grid'] = griddef.name
    atts['title'] = 'Unified Climatology from {0:s} on {1:s} Grid'.format(periodstr,griddef.name)
    # make new dataset
    sink = DatasetNetCDF(folder=avgfolder, filelist=[filename], atts=atts, mode='w')
    # add a few variables that will remain unchanged
    for var in [gpcc025.landmask, prism.lon2D, prism.lat2D]:
      var.load(); sink.addVariable(var, asNC=True, copy=True, deepcopy=True); var.unload()
    # PRISM datamask
    prismmask = prism.datamask.copy(); prismmask.name = 'prismmask' 
    prismmask.load(data=prism.datamask.getArray(unmask=True, fillValue=1)) 
    sink.addVariable(prismmask, asNC=True, copy=True, deepcopy=True)
    prism.datamask.unload()
    # sync and write data so far 
    sink.sync()       
            
    ## merge data (create variables)
    
    # precip
    var = prism.precip.copy() # generate variable copy
    # load data
    prism.precip.load(); gpccprd.precip.load(); gpccclim.precip.load(); gpcc025.precip.load() 
    prismarray = prism.precip.getArray(); gpccclimarray = gpccclim.precip.getArray()
    gpccprdarray = gpccprd.precip.getArray(); gpcc025array = gpcc025.precip.getArray()
    # generate climatology
    array = ma.where(prismarray.filled(-999) == -999, gpcc025array, prismarray)        
    array = array - gpccclimarray + gpccprdarray # add temporal variation
    # save variable 
    var.load(data=array)
    sink.addVariable(var, asNC=True, copy=True, deepcopy=True)
    
    # Temperature
    for varname in ['T2', 'Tmin', 'Tmax']:
      var = prism.variables[varname].copy() # generate variable copy
      # load data
      prism.variables[varname].load(); cruprd.variables[varname].load(); cruclim.variables[varname].load() 
      prismarray = prism.variables[varname].getArray(); cruclimarray = cruclim.variables[varname].getArray()
      cruprdarray = cruprd.variables[varname].getArray()
      # generate climatology
      array = ma.where(prismarray.filled(-999) == -999, cruclimarray, prismarray)        
      array = array - cruclimarray + cruprdarray # add temporal variation
      # save variable 
      var.load(data=array)
      sink.addVariable(var, asNC=True, copy=True, deepcopy=True)

    
    ## add remaining CRU data
    for varname in ['Q2','pet','cldfrc','wetfrq','frzfrq']:
      cruprd.variables[varname].load()
      sink.addVariable(cruprd.variables[varname], asNC=True, copy=True, deepcopy=True)
      cruprd.variables[varname].unload()
      sink.variables[varname].atts['source'] = 'CRU'
    
    # add names and length of months
    sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                        atts=dict(name='name_of_month', units='', long_name='Name of the Month'))        
    if not sink.hasVariable('length_of_month'):
      sink += Variable(name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                    atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
    
    # apply higher resolution mask
    sink.mask(sink.landmask, maskSelf=False, varlist=None, skiplist=['prismmask','lon2d','lat2d'], invert=False, merge=True)
        
    # finalize changes
    sink.sync()     
    sink.close()
    print(sink)
    print('\n Writing to: \'{0:s}\'\n'.format(filename))