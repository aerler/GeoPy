'''
Created on 2014-07-09

This module contains meta data and access functions for the updated PRISM climatology as distributed by 
PCIC (Pacific Climate Impact Consortium).

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os
# internal imports
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset
from geodata.nctools import writeNetCDF, add_strvar
from datasets.common import name_of_month, data_root, grid_folder, transformPrecip
from datasets.common import translateVarNames, loadObservations, addLandMask, addLengthAndNamesOfMonth, getFileName
# from geodata.utils import DatasetError
from warnings import warn
from geodata.gdal import GridDefinition

## PCIC PRISM Meta-data

dataset_name = 'PCIC'

# PRISM grid definition
dlat = dlon = 1./120. #  0.0083333333
dlat2 = dlon2 = 1./240. # half step
nlat = 1680 # slat = 14 deg
nlon = 3241 # slon = 27 deg
# N.B.: coordinates refer to grid points (CF convention), commented values refer to box edges (GDAL convention) 
llclat = 48. # 48.0000000000553
# llclat = 48.0000000000553 # 48.
llclon = -140. # -140.0
           
geotransform = (llclon-dlon2, dlon, 0.0, llclat-dlat2, 0.0, dlat)
size = (nlon,nlat) # (x,y) map size of PRISM grid
# make GridDefinition instance
PCIC_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform, size=size)

# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder

## Functions that handle access to the original PCIC NetCDF files

# variable attributes and names in original PCIC files
ltmvaratts = dict(tmin = dict(name='Tmin', units='K', atts=dict(long_name='Minimum 2m Temperature'), offset=273.15), # 2m minimum temperature
               tmax = dict(name='Tmax', units='K', atts=dict(long_name='Maximum 2m Temperature'), offset=273.15), # 2m maximum temperature
               pr   = dict(name='precip', units='mm/month', atts=dict(long_name='Total Precipitation'), 
                           scalefactor=1., transform=transformPrecip), # total precipitation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='days', atts=dict(long_name='days since beginning of year'), offset=-5493), # time coordinate
               lon  = dict(name='lon', units='deg E', atts=dict(long_name='Longitude')), # geographic longitude field
               lat  = dict(name='lat', units='deg N', atts=dict(long_name='Latitude'))) # geographic latitude field
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
ltmvarlist = ltmvaratts.keys() # also includes coordinate fields    

# loads data from original PCIC NetCDF files
# climatology
ltmfolder = root_folder + 'climatology/' # climatology subfolder
ltmfile = '{0:s}_monClim_PRISM_historical_run1_197101-200012.nc' # expand with variable name
def loadPCIC_LTM(name=dataset_name, varlist=None, varatts=ltmvaratts, filelist=None, folder=ltmfolder):
  ''' Get a properly formatted dataset the monthly PCIC PRISM climatology. '''
  # translate varlist
  if varlist is None: varlist = varatts.keys()
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # generate file list
  filelist = [ltmfile.format(var) for var in varlist if var not in ('time','lat','lon')]
  # load variables separately
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, ncformat='NETCDF4')
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None, gridfolder=grid_folder)
  # N.B.: projection should be auto-detected as geographic    
  # return formatted dataset
  return dataset


## Functions that provide access to well-formatted PCIC PRISM NetCDF files

# pre-processed climatology files (varatts etc. should not be necessary)
avgfile = 'pcic{0:s}_clim{1:s}.nc' # formatted NetCDF file
avgfolder = root_folder + 'pcicavg/' # prefix
# function to load these files...
def loadPCIC(name=dataset_name, period=None, grid=None, resolution=None, varlist=None, varatts=None, 
             folder=None, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly PCIC PRISM climatology as a DatasetNetCDF. '''
  if folder is None: folder = avgfolder
  # only the climatology is available
  if period is not None: 
    warn('Only the full climatology is currently available: setting \'period\' to None.')
    period = None
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadObservations(name=name, folder=folder, projection=None, period=period, grid=grid, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                             lautoregrid=lautoregrid, mode='climatology')
#   # make sure all fields are masked
#   dataset.load()
#   dataset.mask(dataset.datamask, maskSelf=False)
  # return formatted dataset
  return dataset

# function to load station data
def loadPCIC_Stn(name=dataset_name, period=None, station=None, resolution=None, varlist=None, 
                 varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly PCIC PRISM climatology at station locations as a DatasetNetCDF. '''
  # only the climatology is available
  if period is not None: 
    warn('Only the full climatology is currently available: setting \'period\' to None.')
    period = None
  # load standardized climatology dataset with PCIC-specific parameters  
  dataset = loadObservations(name=name, folder=folder, grid=None, station=station, shape=None, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, projection=None, 
                             filelist=filelist, lautoregrid=False, period=period, mode='climatology')
  # return formatted dataset
  return dataset

# function to load averaged data
def loadPCIC_Shp(name=dataset_name, period=None, shape=None, resolution=None, varlist=None, 
                 varatts=None, folder=avgfolder, filelist=None, lencl=True):
  ''' Get the pre-processed monthly PCIC PRISM climatology averaged over regions as a DatasetNetCDF. '''
  # only the climatology is available
  if period is not None: 
    warn('Only the full climatology is currently available: setting \'period\' to None.')
    period = None
  # load standardized climatology dataset with PCIC-specific parameters  
  dataset = loadObservations(name=name, folder=folder, grid=None, station=None, shape=shape, lencl=lencl, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, projection=None, 
                             filelist=filelist, lautoregrid=False, period=period, mode='climatology')
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
ts_file_pattern = None
clim_file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
grid_def = {'':PCIC_grid} # no special name, since there is only one...
LTM_grids = [''] # grids that have long-term mean data 
TS_grids = [] # grids that have time-series data
grid_res = {'':0.008} # approximate resolution in degrees at 45 degrees latitude
default_grid = PCIC_grid
# functions to access specific datasets
loadLongTermMean = loadPCIC_LTM # climatology provided by publisher
loadTimeSeries = None # time-series data
loadClimatology = loadPCIC # pre-processed, standardized climatology
loadStationClimatology = loadPCIC_Stn # climatologies without associated grid (e.g. stations or basins)
loadShapeClimatology = loadPCIC_Shp


if __name__ == '__main__':
    
#   mode = 'test_climatology'
  mode = 'test_point_climatology'
#   mode = 'convert_climatology'
  pntset = 'shpavg' # 'ecprecip
    
  # do some tests
  if mode == 'test_climatology':  
    
    # load NetCDF dataset
#     dataset = loadPCIC(grid='arb2_d02')
    dataset = loadPCIC()
    print(dataset)
    print('')
    stnds = loadPCIC_Stn(station='ecprecip')
    print(stnds)
    print('')
    print(dataset.geotransform)
    print(dataset.precip.masked)
    print(dataset.precip.getArray().mean())
    print('')
    # display
    import pylab as pyl
    pyl.imshow(np.flipud(dataset.datamask.getArray()[:,:])) 
    pyl.colorbar(); pyl.show(block=True)


  elif mode == 'test_point_climatology':
    
    # load point climatology
    print('')
    if pntset in ('shpavg',): dataset = loadPCIC_Shp(shape=pntset)
    else: dataset = loadPCIC_Stn(station=pntset)
    print(dataset)
    print('')
    print(dataset.time)
    print(dataset.time.coord)
    
  ## convert PCIC NetCDF files to proper climatology 
  elif mode == 'convert_climatology': 
    
    # load dataset
    source = loadPCIC_LTM()
    # change meta-data
    source.name = 'PCIC'
    source.title = 'PCIC PRISM Climatology'
    # load data into memory (and ignore last time step, which is just the annual average)
#     source.load(time=(0,12)) # exclusive the last index
    source.load(time=(0,12)) # for testing
    # make normal dataset
    dataset = source.copy()
    source.close()
    
    ## add new variables
    # add landmask (it's not really a landmask, thought)
    maskatts = dict(name='datamask', units='', long_name='Mask for Climatology Fields', 
                description='where this mask is non-zero, no data is available')
    addLandMask(dataset, maskname='datamask',atts=maskatts) # create mask from precip mask
    dataset.mask(dataset.datamask) # mask all fields using the new data mask      
    # add length and names of month
    addLengthAndNamesOfMonth(dataset, noleap=False)       
    # add mean temperature
    T2 = dataset.Tmin + dataset.Tmax # average temperature is just the average between min and max
    T2 /= 2.
    T2.name = 'T2'; T2.atts.long_name='Average 2m Temperature'
    print(T2)
    dataset += T2 # add to dataset
    # rewrite time axis
    time = dataset.time
    time.load(data=np.arange(1,13))
    time.units = 'month'; time.atts.long_name='Month of the Year'
    print(time)
    # print diagnostic
    print(dataset)
    print('')
    for var in dataset:
      #print(var)
      print('Mean {0:s}: {1:s} {2:s}'.format(var.atts.long_name, str(var.mean()), var.units))
      #print('')
    print('')
       
    ## create new NetCDF file    
    # figure out a different filename
    filename = getFileName(name='PCIC', filepattern=avgfile)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)      
    # write data and some annotation
    sink = writeNetCDF(dataset, avgfolder+filename, close=False)
    add_strvar(sink,'name_of_month', name_of_month, 'time', # add names of month
               atts=dict(name='name_of_month', units='', long_name='Name of the Month'))          
    sink.close() # close...
    print('Saving Climatology to: '+filename)
    print(avgfolder)
    