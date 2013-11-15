'''
Created on 2013-09-08

This module contains meta data and access functions for NARR datasets. 

@author: Andre R. Erler, GPL v3
'''

#external imports
#import numpy as np
import os
# from atmdyn.properties import variablePlotatts
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition
from geodata.misc import DatasetError 
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root, loadClim, grid_folder
from processing.process import CentralProcessingUnit


## NARR Meta-data

dataset_name = 'NARR'

# NARR projection
projdict = dict(proj  = 'lcc', # Lambert Conformal Conic  
                lat_1 =   50., # Latitude of first standard parallel
                lat_2 =   50., # Latitude of second standard parallel
                lat_0 =   50., # Latitude of natural origin
                lon_0 = -107.) # Longitude of natural origin
                # 
                # x_0   = 5632642.22547, # False Origin Easting
                # y_0   = 4612545.65137) # False Origin Northing
# NARR grid definition           
projection = getProjFromDict(projdict)
geotransform = (-5648873.5, 32463.0, 0.0, -4628776.5, 0.0, 32463.0)
size = (349, 277) # (x,y) map size of NARR grid
# make GridDefinition instance
NARR_grid = GridDefinition(name=dataset_name, projection=projdict, geotransform=geotransform, size=size)

# variable attributes and name
varatts = dict(air   = dict(name='T2', units='K'), # 2m Temperature
               prate = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
               # LTM-only variables (currently...)
               prmsl = dict(name='pmsl', units='Pa'), # sea-level pressure
               pevap = dict(name='pet', units='kg/m^2'), # monthly accumulated PET (kg/m^2)
               pr_wtr = dict(name='pwtr', units='kg/m^2'), # total precipitable water (kg/m^2)
               # axes (don't have their own file; listed in axes)
               lon   = dict(name='lon2D', units='deg E'), # geographic longitude field
               lat   = dict(name='lat2D', units='deg N'), # geographic latitude field
               time  = dict(name='time', units='days', offset=-1569072, scalefactor=1./24.), # time coordinate
               # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
               #       the time offset is chose such that 1979 begins with the origin (time=0)
               x     = dict(name='x', units='m', offset=-5632642), # projected west-east coordinate
               y     = dict(name='y', units='m', offset=-4612545)) # projected south-north coordinate                 
#                x     = dict(name='x', units='m', offset=-1*projdict['x_0']), # projected west-east coordinate
#                y     = dict(name='y', units='m', offset=-1*projdict['y_0'])) # projected south-north coordinate
# N.B.: At the moment Skin Temperature can not be handled this way due to a name conflict with Air Temperature
# list of variables to load
ltmvarlist = varatts.keys() # also includes coordinate fields    
tsvarlist = ['air', 'prate', 'lon', 'lat'] # 'air' is actually 2m temperature...

# variable and file lists settings
nofile = ('lat','lon','x','y','time') # variables that don't have their own files
special = dict(air='air.2m') # some variables need special treatment

# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder

## Functions to load different types of NARR datasets 

# Climatology (LTM - Long Term Mean)
ltmfolder = root_folder + 'LTM/' # LTM subfolder
def loadNARR_LTM(name=dataset_name, varlist=ltmvarlist, interval='monthly', varatts=varatts, filelist=None, folder=ltmfolder):
  ''' Get a properly formatted dataset of daily or monthly NARR climatologies (LTM). '''
  # prepare input
  if interval == 'monthly': 
    pfx = '.mon.ltm.nc'; tlen = 12
  elif interval == 'daily': 
    pfx = '.day.ltm.nc'; tlen = 365
  else: raise DatasetError, "Selected interval '%s' is not supported!"%interval
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)  
  # axes dictionary, primarily to override time axis 
  axes = dict(time=Axis(name='time',units='day',coord=(1,tlen,tlen)),load=True)
  if filelist is None: # generate default filelist
    filelist = [special[var]+pfx if var in special else var+pfx for var in varlist if var not in nofile]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          axes=axes, atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='{0:s} Coordinate System'.format(name))
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None, folder=grid_folder)
  # return formatted dataset
  return dataset

# Time-series (monthly)
tsfolder = root_folder + 'Monthly/' # monthly subfolder
def loadNARR_TS(name=dataset_name, varlist=tsvarlist, varatts=varatts, filelist=None, folder=tsfolder):
  ''' Get a properly formatted NARR dataset with monthly mean time-series. '''
  # prepare input  
  pfx = '.mon.mean.nc'
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  if filelist is None: # generate default filelist
    filelist = [special[var]+pfx if var in special else var+pfx for var in varlist if var not in nofile]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='{0:s} Coordinate System'.format(name))
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None, folder=grid_folder)
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'narravg/' 
avgfile = 'narr{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadNARR(name=dataset_name, period=None, grid=None, resolution=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly NARR climatology as a DatasetNetCDF. '''
  # load standardized climatology dataset with NARR-specific parameters
  dataset = loadClim(name=name, folder=folder, projection=projection, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset

## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
grid_def = {'':NARR_grid} # no special name since there is only one grid 
LTM_grids = [''] # grids that have long-term mean data 
TS_grids = [''] # grids that have time-series data
grid_res = {'':0.41} # approximate resolution in degrees at 45 degrees latitude
default_grid = NARR_grid
# grid_def = {0.41:NARR_grid} # approximate NARR grid resolution at 45 degrees latitude
# grid_tag = {0.41:''} # no special name, since there is only one... 
# functions to access specific datasets
loadLongTermMean = loadNARR_LTM # climatology provided by publisher
loadTimeSeries = loadNARR_TS # time-series data
loadClimatology = loadNARR # pre-processed, standardized climatology

## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
  mode = 'test_climatology'
#   mode = 'average_timeseries'
#   mode = 'convert_climatology'
  grid = 'NARR'
  period = (1979,1984)
  period = (1979,1989)
  period = (1979,2009)
  
  if mode == 'test_climatology':
    
    # load averaged climatology file
    print('')
    dataset = loadNARR(period=period)
    print(dataset)
    print('')
    print(dataset.geotransform)
    print('')
    print(grid_def[''].scale)
              

  elif mode == 'convert_climatology':      
      
      from datasets.common import addLengthAndNamesOfMonth, getFileName
      from geodata.nctools import writeNetCDF, add_strvar
      
      # load dataset
      dataset = loadNARR_LTM()
      # change meta-data
      dataset.name = 'NARR'
      dataset.title = 'NARR Long-term Climatology'
      # load data into memory
      dataset.load()

#       # add landmask
#       addLandMask(dataset) # create landmask from precip mask
#       dataset.mask(dataset.landmask) # mask all fields using the new landmask      
      # add length and names of month
      addLengthAndNamesOfMonth(dataset, noleap=False) 
      
      # figure out a different filename
      filename = getFileName(grid='NARR', period=None, name='NARR', filepattern=avgfile)
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
      
   
  # generate averaged climatology
  elif mode == 'average_timeseries':
    
    # load source
    periodstr = '%4i-%4i'%period
    print('\n')
    print('   ***   Processing Grid %s from %s   ***   '%(grid,periodstr))
    print('\n')
    source = loadNARR_TS()
    print(source)
    print('\n')
    # prepare sink
    gridstr = '' if grid is 'NARR' else '_'+grid
    filename = avgfile.format(gridstr,'_'+periodstr)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
    sink = DatasetNetCDF(name='NARR Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
    sink.atts.period = periodstr 
    
    # determine averaging interval
    offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
    # initialize processing
#     CPU = CentralProcessingUnit(source, sink, varlist=['precip', 'T2'], tmp=True) # no need for lat/lon
    CPU = CentralProcessingUnit(source, sink, varlist=None, tmp=True) # no need for lat/lon
    
    # start processing climatology
    CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
    
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
    