'''
Created on Nov. 07, 2020

A module to read ERA5 data; this includes converting GRIB files to NetCDF-4, 
as well as functions to load the converted and aggregated data.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os.path as osp
import pandas as pd
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
from collections import namedtuple
# internal imports
from datasets.common import getRootFolder
from geodata.gdal import GridDefinition
from datasets.misc import loadXRDataset, getFolderFileName


## Meta-vardata

dataset_name = 'C1W'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='NRCan') # get dataset root folder based on environment variables

# C1W projection
projdict = dict(proj='aea', lat_0=40, lon_0=-96, lat_1=20, lat_2=60, x_0=0, y_0=0, ellps='GRS80', towgs84='0,0,0,0,0,0,0', units='m', name=dataset_name)
proj4_string = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +name={} +no_defs'.format(dataset_name)
# Carcajou Watershed, 5 km
ccj1_geotransform = (-1618000, 5000, 0, 3058000, 0, 5000)
ccj1_size = (22, 33)
ccj1_grid = GridDefinition(name=dataset_name, projection=None, geotransform=ccj1_geotransform, size=ccj1_size)

varatts_list = dict()
# attributes of soil variables from reanalysis
varatts_list['C1W_Soil'] = dict(
             # state variables
             soil_temp_0cm_50cm = dict(name='Tsl1',units='K', long_name='Soil Temperature, Layer 1'),
             soil_temp_50cm_100cm = dict(name='Tsl2',units='K', long_name='Soil Temperature, Layer 2'),
             soil_temp_100cm_200cm = dict(name='Tsl3',units='K', long_name='Soil Temperature, Layer 3'),
             swe  = dict(name='snow',  units='kg/m^2', scalefactor=1, long_name='Snow Water Equivalent'),  # water equivalent
             # axes (don't have their own file)
             time_stamp = dict(name='time_stamp', units='', long_name='Time Stamp'),  # readable time stamp (string)
             time = dict(name='time', units='days', long_name='Days'),  # time coordinate
             lon  = dict(name='lon', units='deg', long_name='Longitude'),  # geographic longitude
             lat  = dict(name='lat', units='deg', long_name='Latitude'),  # geographic latitude
             )
# list of variables to load
default_varlists = {name:[atts['name'] for atts in varatts.values()] for name,varatts in varatts_list.items()}
# list of sub-datasets/subsets with titles
DSNT = namedtuple(typename='Dataset', field_names=['name','interval','resolution','title',])
dataset_attributes = dict(C1W_Soil_1deg = DSNT(name='C1W_Soil_1deg',interval='1M', resolution=1.0, title='1 deg. Soil Ensemble',),  # 1 degree Ensemble
                          C1W_Soil = DSNT(name='C1W_Soil',interval='1M', resolution=0.05, title='0.05 deg. Soil Ensemble',),)  # 1 degree Ensemble

# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/' 
avgfile   = 'c1w{0:s}_clim{1:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = 'c1w_{0:s}{1:s}{2:s}_monthly.nc' # extend with biascorrection, variable and grid type
daily_folder    = root_folder + dataset_name.lower()+'_daily/' 
netcdf_filename = 'c1w_{:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float
netcdf_settings = dict() # possibly define chunking


## functions to load NetCDF datasets (using xarray)

def loadC1W(varname=None, varlist=None, dataset=None, subset=None, grid=None, resolution=None, shape=None, station=None,
            resampling=None, varatts=None, varmap=None, lgeoref=True, geoargs=None, aggregation='monthly',
            mode='avg', chunks=True, multi_chunks=None, lxarray=True, lgeospatial=True, **kwargs):
    ''' function to load daily ERA5 data from NetCDF-4 files using xarray and add some projection information '''
    if not (lxarray and lgeospatial):
        raise NotImplementedError("Only loading via geospatial.xarray_tools is currently implemented.")
    if dataset and subset:
        if dataset != subset:
            raise ValueError((dataset, subset))
    elif dataset and not subset:
        subset = dataset
    if resolution is None:
        resolution = 'NA005'  # default
    if varatts is None:
        if grid is None and station is None and shape is None: varatts = varatts_list[subset]  # original files
    default_varlist = default_varlists.get(dataset, None)
    xds = loadXRDataset(varname=varname, varlist=varlist, dataset='C1W', subset=subset, grid=grid, resolution=resolution, shape=shape,
                        station=station, default_varlist=default_varlist, resampling=resampling, varatts=varatts, varmap=varmap, mode=mode,
                        aggregation=aggregation, lgeoref=lgeoref, geoargs=geoargs, chunks=chunks, multi_chunks=multi_chunks, **kwargs)
    # update name and title with sub-dataset
    xds.attrs['name'] = subset
    xds.attrs['title'] = dataset_attributes[subset].title + xds.attrs['title'][len(subset)-1:]
    return xds


## Dataset API

dataset_name  # dataset name
root_folder  # root folder of the dataset
orig_file_pattern = netcdf_filename  # filename pattern: variable name (daily)
ts_file_pattern   = tsfile  # filename pattern: variable name and grid
clim_file_pattern = avgfile  # filename pattern: grid and period
data_folder       = avgfolder  # folder for user data
grid_def  = {'CCJ1':ccj1_grid}  # just one for now...
LTM_grids = []  # grids that have long-term mean data
TS_grids  = ['',]  # grids that have time-series data
grid_res  = {res:0.25 for res in TS_grids}  # no special name, since there is only one...
default_grid = None
# functions to access specific datasets
loadLongTermMean       = None  # climatology provided by publisher
loadDailyTimeSeries    = None  # daily time-series data
# monthly time-series data for batch processing
loadTimeSeries         = loadC1W  # sort of... with defaults
loadClimatology        = None  # pre-processed, standardized climatology
loadStationClimatology = None  # climatologies without associated grid (e.g. stations)
loadStationTimeSeries  = None  # time-series without associated grid (e.g. stations)
loadShapeClimatology   = None  # climatologies without associated grid (e.g. provinces or basins)
loadShapeTimeSeries    = None  # time-series without associated grid (e.g. provinces or basins)


## abuse for testing
if __name__ == '__main__':

  import time, gc, os

  #print('xarray version: '+xr.__version__+'\n')
  xr.set_options(keep_attrs=True)

# import dask
#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=2, memory_limit='1GB')
#   cluster = LocalCluster(n_workers=4, memory_limit='6GB')
#   cluster = LocalCluster(n_workers=1)
#   client = Client(cluster)


  modes = []
  modes += ['load_TimeSeries']
  # modes += ['simple_test']

  grid = None; resampling = None

  dataset = 'C1W_Soil'
  resolution = 'NA005'

  # variable list
  varlist = ['Tsl1']

#   period = (2010,2019)
#   period = (1997,2018)
#   period = (1980,2018)

  # loop over modes 
  for mode in modes:


    if mode == 'simple_test':

        folder = root_folder + dataset.lower()+'avg/' 
        filename = '{}_{}_monthly.nc'.format(dataset.lower(), resolution.lower())

        import xarray as xr
        xds = xr.load_dataset(folder + filename, decode_times=False, chunks={'time':1, 'lat':64, 'lon':64})
        print(xds)

    elif mode == 'load_TimeSeries':

        pass
        lxarray = False
        varname = varlist[0]
        xds = loadC1W(varlist=varlist, resolution=resolution, dataset=dataset, decode_times=False, )
        print(xds)
        print('')
        xv = xds[varname]
        print(xv)
        if lxarray:
            print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
