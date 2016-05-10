'''
Created on 2013-11-23

A unified/merged dataset, constructed from multiple available sources. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
#import netCDF4 as nc # netcdf python module
import os, gc
# internal imports
from geodata.base import Variable
from geodata.netcdf import DatasetNetCDF 
from datasets.common import days_per_month, name_of_month,  shp_params, CRU_vars
from datasets.common import getFileName, data_root, loadObservations, grid_folder
from geodata.gdal import loadPickledGridDef
from datasets.GPCC import loadGPCC, loadGPCC_Shp
from datasets.CRU import loadCRU, loadCRU_Shp, loadCRU_ShpTS
from datasets.PRISM import loadPRISM, loadPRISM_Shp
from datasets.PCIC import loadPCIC, loadPCIC_Shp
from utils.constants import precip_thresholds
# from geodata.utils import DatasetError
from warnings import warn
from plotting.properties import variablePlotatts

## Unity Meta-data

dataset_name = 'Unity'
root_folder = '{:s}/{:s}/'.format(data_root,dataset_name) # the dataset root folder
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
               # additional variables
               dryprec = dict(name='dryprec', units='kg/m^2/s'), # precipitation rate above dry-day threshold (kg/m^2/s)
               wetprec = dict(name='wetprec', units='kg/m^2/s'), # wet-day precipitation rate (kg/m^2/s)
               # axes (don't have their own file; listed in axes)
               time=dict(name='time', units='month', atts=dict(long_name='Month of the Year')), # time coordinate
               lon  = dict(name='lon', units='deg E', atts=dict(long_name='Longitude')), # geographic longitude field
               lat  = dict(name='lat', units='deg N', atts=dict(long_name='Latitude'))) # geographic latitude field
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    
# variable and file lists settings


## Functions that provide access to well-formatted PRISM NetCDF files

# wraper to add special variables
def loadSpecialObs(varlist=None, **kwargs):
  ''' wrapper that adds some special/derived variables '''
  # parse and edit varlist
  if varlist is None:
    ldryprec = True; lwetprec = True; lwetdays = True
  else:
    # cast/copy varlist
    if isinstance(varlist,basestring): varlist = [varlist] # cast as list
    else: varlist = list(varlist) # make copy to avoid interference
    # select options
    ldryprec = False; lwetprec = False; lwetdays = False
    if 'dryprec' in varlist: # really just an alias for precip 
      ldryprec = True; varlist.remove('dryprec')
      if 'precip' not in varlist: varlist.append('precip')
    if 'wetprec' in varlist: 
      lwetprec = True; varlist.remove('wetprec') # wet-day precip
      if 'precip' not in varlist: varlist.append('precip')
      if 'wetfrq' not in varlist: varlist.append('wetfrq')
    if any([var[:6]=='wetfrq' for var in varlist]): 
      lwetdays = True 
      if 'wetfrq' not in varlist: varlist.append('wetfrq')
  # load actual data using standard call to loadObservation
  dataset = loadObservations(varlist=varlist, **kwargs)
  # add new/special variables
  if lwetprec and 'wetprec' not in dataset: 
    wetprec = dataset.precip.load() / dataset.wetfrq.load()
    dataset += wetprec.copy(plot=variablePlotatts['wetprec'], **varatts['wetprec'])
  if ldryprec and 'dryprec' not in dataset: 
    dataset += dataset.precip.copy(plot=variablePlotatts['dryprec'], **varatts['dryprec'])
  if lwetdays:
    for threshold in precip_thresholds:
      varname = 'wetfrq_{:03d}'.format(int(threshold*10))
      if varname not in dataset:
        dataset += dataset.wetfrq.copy(name=varname)
  # return augmented dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'unityavg/' # prefix
avgfile = 'unity{0:s}_clim{1:s}.nc' # formatted NetCDF file
# function to load these files...
def loadUnity(name=dataset_name, period=None, grid=None, varlist=None, varatts=None, 
              folder=avgfolder, filelist=None, lautoregrid=False, resolution=None):
  ''' Get the pre-processed, unified monthly climatology as a DatasetNetCDF. '''
  #if lautoregrid: warn("Auto-regridding is currently not available for the unified dataset - use the generator routine instead.")
  # a climatology is not available
  if period is None: 
    period = (1979,2009)
    warn('A climatology is not available for the Unified Dataset; loading period {0:4d}-{1:4d}.'.format(*period))
  # this dataset has not native/default grid
  if grid is None: 
    grid = 'arb2_d02'
    warn('The Unified Dataset has no native grid; loading {0:s} grid.'.format(grid))
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadSpecialObs(name=name, folder=folder, period=period, grid=grid, shape=None, station=None, 
                           varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                           projection=None, mode='climatology', lautoregrid=False)
  # return formatted dataset
  return dataset

# function to load climatologies at station locations
def loadUnity_Stn(name=dataset_name, period=None, station=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=False, resolution=None, lencl=False):
  ''' Get the pre-processed, unified monthly climatology averaged over shapes as a DatasetNetCDF. '''
  # a climatology is not available
  if period is None: 
    period = (1979,2009)
    warn('A climatology is not available for the Unified Dataset; loading period {0:4d}-{1:4d}.'.format(*period))
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadSpecialObs(name=name, folder=folder, period=period, grid=None, station=station, shape=None, 
                           varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                           projection=None, mode='climatology', lautoregrid=False, lencl=lencl)
  # return formatted dataset
  return dataset

# function to load time-series at station locations
tsfile = 'unity{0:s}_monthly.nc' # formatted NetCDF file
def loadUnity_StnTS(name=dataset_name, station=None, varlist=None, varatts=None, 
                    folder=avgfolder, filelist=None, lautoregrid=False, resolution=None, lencl=False):
  ''' Get the pre-processed, unified monthly climatology averaged over shapes as a DatasetNetCDF. '''
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadSpecialObs(name=name, folder=folder, period=None, grid=None, shape=None, station=station, 
                           varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                           projection=None, mode='time-series', lautoregrid=False, lencl=lencl)
  # return formatted dataset
  return dataset

# function to load these files...
def loadUnity_Shp(name=dataset_name, period=None, shape=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=False, resolution=None, lencl=False):
  ''' Get the pre-processed, unified monthly climatology averaged over shapes as a DatasetNetCDF. '''
  # a climatology is not available
  if period is None: 
    period = (1979,2009)
    warn('A climatology is not available for the Unified Dataset; loading period {0:4d}-{1:4d}.'.format(*period))
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadSpecialObs(name=name, folder=folder, period=period, grid=None, shape=shape, station=None, 
                           varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                           projection=None, mode='climatology', lautoregrid=False, lencl=lencl)
  # return formatted dataset
  return dataset

# function to load shape-averaged time-series
tsfile = 'unity{0:s}_monthly.nc' # formatted NetCDF file
def loadUnity_ShpTS(name=dataset_name, shape=None, varlist=None, varatts=None, 
                    folder=avgfolder, filelist=None, lautoregrid=False, resolution=None, lencl=False):
  ''' Get the pre-processed, unified monthly climatology averaged over shapes as a DatasetNetCDF. '''
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadSpecialObs(name=name, folder=folder, period=None, grid=None, shape=shape, station=None, 
                           varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                           projection=None, mode='time-series', lautoregrid=False, lencl=lencl)
  # return formatted dataset
  return dataset

## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
ts_file_pattern = tsfile
clim_file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
LTM_grids = [] 
TS_grids = ['']
grid_res = {}
grid_def = None # no grid here...
# LTM_grids = ['d02','d01'] # grids that have long-term mean data 
# TS_grids = ['d02','d01'] # grids that have time-series data
# grid_def = {'d02':None,'d01':None} # there are too many... 
# grid_res = {'d02':0.13,'d01':3.82} # approximate grid resolution at 45 degrees latitude 
default_grid = None
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = None # time-series data
loadClimatology = loadUnity # pre-processed, standardized climatology


if __name__ == '__main__':
  
  # select mode
#   mode = 'merge_climatologies'
#   mode = 'merge_timeseries'
#   mode = 'test_climatology'
  mode = 'test_point_climatology'
#   mode = 'test_point_timeseries'
  
  # settings to generate dataset
  grids = []
  grids += ['shpavg']
#   grids += ['arb2_d01']
  grids += ['arb2_d02']
#   grids += ['arb3_d02']
#   grids += ['arb3_d01']
#   grids += ['arb3_d02']
#   grids += ['grb1_d01']
#   grids += ['grb1_d02']
#   grids += ['col1_d01']
#   grids += ['col1_d02'] 
#   grids += ['col1_d03']
#   grids += ['col2_d01']
#   grids += ['col2_d02'] 
#   grids += ['col2_d03'] 
#   grids += ['ARB_small_025']
#   grids += ['ARB_large_025']
#   grids += ['cesm1x1']
#   grids += ['NARR']
  periods = []
#   periods += [(1979,1980)]
#   periods += [(1979,1982)]
#   periods += [(1979,1984)]
#   periods += [(1979,1989)]
  periods += [(1979,1994)]
#   periods += [(1984,1994)]
#   periods += [(1989,1994)]
#   periods += [(1997,1998)]
#   periods += [(1979,2009)]
#   periods += [(1949,2009)]
#   pntset = 'shpavg'
  pntset = 'ecprecip'
  
  ## do some tests
  if mode == 'test_climatology':  
    
    for grid in grids:
      for period in periods: 
        # load NetCDF dataset
        dataset = loadUnity(grid=grid, period=period)
        #dataset = loadUnity()
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

  elif mode == 'test_point_climatology':
            
    # load averaged climatology file
    print('')
    if pntset in ('shpavg',): 
      dataset = loadUnity_Shp(varlist=['dryprec','wetprec','shp_area'], shape=pntset, period=periods[0])
      print(dataset.shp_area.mean())
      print('')
    else: 
      dataset = loadUnity_Stn(varlist=['precip','stn_rec_len'], station=pntset, period=periods[0])
      print(dataset.stn_rec_len.mean())
      print('')
    print(dataset)
    dataset.load()
    if 'precip' in dataset:
      print('')
      print(dataset.precip.mean())
      print(dataset.precip.masked)
    if 'dryprec' in dataset:
      print('')
      print(dataset.dryprec)
      assert dataset.dryprec.mean() == dataset.precip.mean() 
    
    # print time coordinate
    print
    print dataset.time.atts
    print
    print dataset.time.data_array

  elif mode == 'test_point_timeseries':
            
    # load averaged climatology file
    print('')
    if pntset in ('shpavg',): dataset = loadUnity_ShpTS(shape=pntset,varlist=['wetfrq','precip'])
    else: raise NotImplementedError
    print(dataset)
    dataset.load()
    print('')
    print(dataset.precip.mean())
    print(dataset.precip.masked)
    
    # print time coordinate
    print
    print dataset.time.atts
    print
    print dataset.time.data_array


  ## begin processing
  if mode == 'merge_timeseries':
    # produce a merged dataset for a given time period and grid    
    
    for grid in grids:
                
        ## load source datasets
#         period = (1979,2009)
        period = (1979,1994)
        if grid in ('shpavg',):
          # regional averages: shape index as grid
          uclim = loadUnity_Shp(shape=pntset, period=period)
          cruclim = loadCRU_Shp(shape=grid, period=period)
          cruts = loadCRU_ShpTS(shape=grid)           
        else:
          raise NotImplementedError
          
        grid_name = grid
        periodstr = '{0:4d}-{1:4d}'.format(*period)        
        print('\n   ***   Merging Shape-Averaged Time-Series on {:s} Grid  ***   \n'.format(grid,))
        ## prepare target dataset 
        filename = getFileName(grid=grid_name, period=None, name=None, filepattern=tsfile)
        filepath = avgfolder + filename
        print(' Saving data to: \'{0:s}\'\n'.format(filepath))
        assert os.path.exists(avgfolder)
        if os.path.exists(filepath): os.remove(filepath) # remove old file
        # set attributes   
        atts=dict() # collect attributes, but add prefixes
        atts = uclim.atts.copy()
        atts['title'] = 'Corrected Time-sries on {:s} Grid'.format(grid_name)
        # make new dataset
        sink = DatasetNetCDF(folder=avgfolder, filelist=[filename], atts=atts, mode='w')
        # sync and write data so far 
        sink.sync()       
                
        ## correct data (create variables)
        for varname,var in uclim.variables.iteritems():
          print ''
          print varname
          # correct time-series variables
          if var.hasAxis('time'):
            if varname in CRU_vars:
              tsvar = cruts[varname]; climvar = cruclim[varname] 
              assert tsvar.axisIndex('time') == 1, tsvar            
              assert climvar.axisIndex('time') == 1 and var.axisIndex('time') == 1, climvar
              assert len(tsvar.axes[1])%12 == 0, len(tsvar.axes[1])
              assert tsvar.axes[1].coord[0]%12 == 0, tsvar.axes[1].coord[0]
              reps = len(tsvar.axes[1])/12
              unityarray = var.load().data_array.repeat(reps, axis=1)
              climarray = climvar.load().data_array.repeat(reps, axis=1)
              tsarray = tsvar.load().data_array
              assert unityarray.shape == climarray.shape == tsarray.shape, climarray.shape
              array = tsarray - climarray + unityarray
              newvar = var.copy(data=array, axes=tsvar.axes) # generate variable copy
              tsvar.unload(); climvar.unload(); var.unload()
              gc.collect() # not really necessary for shape averages...
            else:
              print var
              newvar = None
          else:
            try:
              var.load()
              newvar = var.copy()
            except ValueError:
              print var
              newvar = None          
          # save variable 
          if newvar is not None: 
            sink.addVariable(newvar, asNC=True, copy=True, deepcopy=True)
            
        # finalize changes
        sink.sync()     
        sink.close()
        print(sink)
        print('\n Writing to: \'{0:s}\'\n'.format(filename))
        
  
  ## begin processing
  if mode == 'merge_climatologies':
    # produce a merged dataset for a given time period and grid    
    
    for grid in grids:
      for period in periods: 
        
        ## load source datasets
        if grid in ('shpavg',):
          lshp = True
          # regional averages: shape index as grid
          # N.B.: currently doesn't work with stations, because station indices are not consistent
          #       for grids of different size (different number of stations included)
          pcic  = loadPCIC_Shp(period=None, shape=grid, lencl=True, 
                               varlist=['T2','Tmin','Tmax','precip','datamask','lon2D','lat2D']+shp_params)
          prism = loadPRISM_Shp(period=None, shape=grid, lencl=True, 
                                varlist=['T2','Tmin','Tmax','precip','datamask','lon2D','lat2D']+shp_params)
          gpccprd = loadGPCC_Shp(period=period, resolution='05', shape=grid, lencl=True, 
                                 varlist=['precip']+shp_params)
          gpccclim = loadGPCC_Shp(period=None, resolution='05', shape=grid, lencl=True, 
                                  varlist=['precip']+shp_params)
          gpcc025 = loadGPCC_Shp(period=None, resolution='025', shape=grid, lencl=True, 
                                 varlist=['precip','landmask']+shp_params)
          cruprd = loadCRU_Shp(period=period, shape=grid, lencl=True, 
                                varlist=['T2','Tmin','Tmax','Q2','pet','cldfrc','wetfrq','frzfrq']+shp_params)
          cruclim = loadCRU_Shp(period=(1979,2009), shape=grid, lencl=True, 
                                varlist=['T2','Tmin','Tmax','Q2','pet','cldfrc','wetfrq','frzfrq']+shp_params)          
        else:
          lshp = False
          # some regular map-type grid 
          pcic  = loadPCIC(period=None, grid=grid, varlist=['T2','Tmin','Tmax','precip','datamask','lon2D','lat2D'], lautoregrid=True)
          prism = loadPRISM(period=None, grid=grid, varlist=['T2','Tmin','Tmax','precip','datamask','lon2D','lat2D'], lautoregrid=True)
          gpccprd = loadGPCC(period=period, resolution='05', grid=grid, varlist=['precip'], lautoregrid=True)
          gpccclim = loadGPCC(period=None, resolution='05', grid=grid, varlist=['precip'], lautoregrid=True)
          gpcc025 = loadGPCC(period=None, resolution='025', grid=grid, varlist=['precip','landmask'], lautoregrid=True)
          cruprd = loadCRU(period=period, grid=grid, varlist=['T2','Tmin','Tmax','Q2','pet','cldfrc','wetfrq','frzfrq'], lautoregrid=True)
          cruclim = loadCRU(period=(1979,2009), grid=grid, varlist=['T2','Tmin','Tmax','Q2','pet','cldfrc','wetfrq','frzfrq'], lautoregrid=True)
          
        # grid definition
        try:
          griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder)
          grid_name = griddef.name
        except IOError:
          griddef = None
          grid_name = grid
        periodstr = '{0:4d}-{1:4d}'.format(*period)
        
        print('\n   ***   Merging Climatology from {0:s} on {1:s} Grid  ***   \n'.format(periodstr,grid,))
        ## prepare target dataset 
        filename = getFileName(grid=grid_name, period=period, name=None, filepattern=avgfile)
        filepath = avgfolder + filename
        print(' Saving data to: \'{0:s}\'\n'.format(filepath))
        assert os.path.exists(avgfolder)
        if os.path.exists(filepath): os.remove(filepath) # remove old file
        # set attributes   
        atts=dict() # collect attributes, but add prefixes
        for key,item in prism.atts.iteritems(): atts['PRISM_'+key] = item
        #for key,item in gpcc025.atts.iteritems(): atts['GPCC_'+key] = item # GPCC atts cause problems... 
        for key,item in cruprd.atts.iteritems(): atts['CRU_'+key] = item
        atts['period'] = periodstr; atts['name'] = dataset_name; atts['grid'] = grid_name
        atts['title'] = 'Unified Climatology from {0:s} on {1:s} Grid'.format(periodstr,grid_name)
        # make new dataset
        sink = DatasetNetCDF(folder=avgfolder, filelist=[filename], atts=atts, mode='w')
        # add a few variables that will remain unchanged
        if griddef is not None:
          for var in [gpcc025.landmask, prism.lon2D, prism.lat2D]:
            var.load(); sink.addVariable(var, asNC=True, copy=True, deepcopy=True); var.unload()
        # add datamasks
        for ds in (prism, pcic):
          datamask = ds.datamask.copy(); datamask.name = ds.name.lower()+'mask' 
          datamask.load(data=ds.datamask.getArray(unmask=True, fillValue=1)) 
          sink.addVariable(datamask, asNC=True, copy=True, deepcopy=True)
          ds.datamask.unload()
        # sync and write data so far 
        sink.sync()       
                
        ## merge data (create variables)
        # precip
        var = pcic.precip.copy() # generate variable copy
        array = pcic.precip.getArray() # start with hi-res PRISM from PCIC  
#         array = ma.masked_where(array == 1., array, copy=False)
        # add low-res PRISM backround
        prismarray = prism.precip.getArray() # start with
        assert isinstance(prismarray,ma.MaskedArray)        
        array = ma.where(array.mask, prismarray, array) # add background climatology
        array = ma.masked_where(prismarray.mask, array, copy=False) # use mask from older PRISM
        # add GPCC background 
        gpccclimarray = gpccclim.precip.getArray()
        gpccprdarray = gpccprd.precip.getArray() 
        gpcc025array = gpcc025.precip.getArray()        
        array = ma.where(array.mask, gpcc025array, array) # add background climatology
#         array = ma.where(array.filled(-999) == -999, gpcc025array, array) # add background climatology        
        array = array - gpccclimarray + gpccprdarray # add temporal variation
        # save variable 
        var.load(data=array)
        sink.addVariable(var, asNC=True, copy=True, deepcopy=True)
        
        # Temperature
        for varname in ['T2', 'Tmin', 'Tmax']:
          # load data
          var = pcic.variables[varname].copy() # generate variable copy
          array = pcic.variables[varname].getArray() # start with hi-res PRISM from PCIC
          # add low-res PRISM backround
          prismarray = prism.variables[varname].getArray()
          array = ma.where(array.mask, prismarray, array) # add background climatology  
          # add CRU background 
          cruclimarray = cruclim.variables[varname].getArray()
          cruprdarray = cruprd.variables[varname].getArray()          
          array = ma.where(array.mask, cruclimarray, array) # generate climatology        
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
        
        ## add station meta data
        if lshp:
          for varname in shp_params:
            var = gpcc025.variables[varname].load()
            sink.addVariable(var, asNC=True, copy=True, deepcopy=True)
            
        # add names and length of months
        sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                            atts=dict(name='name_of_month', units='', long_name='Name of the Month'))        
        if not sink.hasVariable('length_of_month'):
          sink += Variable(name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                        atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
        
        # apply higher resolution mask
        if griddef is not None:
          sink.mask(sink.landmask, maskSelf=False, varlist=None, skiplist=['prismmask','lon2d','lat2d'], invert=False, merge=True)
            
        # finalize changes
        sink.sync()     
        sink.close()
        print(sink)
        print('\n Writing to: \'{0:s}\'\n'.format(filename))