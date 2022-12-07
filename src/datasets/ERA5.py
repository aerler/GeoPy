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

dataset_name = 'ERA5'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='NRCan') # get dataset root folder based on environment variables

# SnoDAS grid definition
projdict = dict(proj='longlat',lon_0=0,lat_0=0,x_0=0,y_0=0) # wraps at dateline
proj4_string = '+proj=longlat +ellps=WGS84 +datum=WGS84 +lon_0=0 +lat_0=0 +x_0=0 +y_0=0 +name={} +no_defs'.format(dataset_name)
# ERA5-Land
ERA5Land_geotransform = (-180, 0.1, 0, -90, 0, 0.1)
ERA5Land_size = (3600,1800) # (x,y) map size of grid
ERA5Land_grid = GridDefinition(name=dataset_name, projection=None, geotransform=ERA5Land_geotransform, size=ERA5Land_size)
# southern Ontario
SON10_geotransform = (-85, 0.1, 0, 41, 0, 0.1)
SON10_size = (111,61) # (x,y) map size of grid
SON10_grid = GridDefinition(name=dataset_name, projection=None, geotransform=ERA5Land_geotransform, size=ERA5Land_size)

varatts_list = dict()
# attributes of variables in ERA5-Land
varatts_list['ERA5L'] = dict(# forcing/flux variables
             tp   = dict(name='precip', units='kg/m^2/s',scalefactor=1000./86400., long_name='Total Precipitation'), # units of meters water equiv. / day
             pev  = dict(name='pet_era5', units='kg/m^2/s',scalefactor=-1000./86400., long_name='Potential Evapotranspiration'), # units of meters water equiv. / day; negative values
             # state variables
             sd   = dict(name='snow',  units='kg/m^2', scalefactor=1.e3, long_name='Snow Water Equivalent'), # units of meters water equivalent
             # axes (don't have their own file)
             time_stamp = dict(name='time_stamp', units='', long_name='Time Stamp'), # readable time stamp (string)
             time = dict(name='time', units='days', long_name='Days'), # time coordinate
             lon  = dict(name='lon', units='deg', long_name='Longitude'), # geographic longitude
             lat  = dict(name='lat', units='deg', long_name='Latitude'), # geographic latitude
             # derived variables
             dswe = dict(name='dswe',units='kg/m^2/s', long_name='SWE Changes'),
             liqwatflx = dict(name='liqwatflx', units='kg/m^2/s', long_name='Liquid Water Flux'),
             )
# list of variables to load
default_varlists = {name:[atts['name'] for atts in varatts.values()] for name,varatts in varatts_list.items()}
# list of sub-datasets/subsets with titles
DSNT = namedtuple(typename='Dataset', field_names=['name','interval','resolution','title',])
dataset_attributes = dict(ERA5L = DSNT(name='ERA5L',interval='1h', resolution=0.1, title='ERA5-Land',), # downscaled land reanalysis
                          ERA5S = DSNT(name='ERA5S',interval='1h', resolution=0.25, title='ERA5-Sfc',), # regular surface; not verified
                          ERA5A = DSNT(name='ERA5A',interval='6h', resolution=0.25, title='ERA5-Atm',),) # regular 3D; not verified

# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/'
avgfile   = 'era5{0:s}_clim{1:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = 'era5_{0:s}{1:s}{2:s}_monthly.nc' # extend with biascorrection, variable and grid type
daily_folder    = root_folder + dataset_name.lower()+'_daily/'
netcdf_filename = 'era5_{:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float
netcdf_settings = dict(chunksizes=(8,ERA5Land_size[0]/16,ERA5Land_size[1]/32))


## functions to load NetCDF datasets (using xarray)

def loadERA5_Daily(varname=None, varlist=None, dataset=None, subset=None, grid=None, resolution=None, shape=None, station=None,
                   resampling=None, varatts=None, varmap=None, lgeoref=True, geoargs=None, lfliplat=False, aggregation='daily',
                   mode='daily', chunks=True, multi_chunks=None, lxarray=True, lgeospatial=True, **kwargs):
    ''' function to load daily ERA5 data from NetCDF-4 files using xarray and add some projection information '''
    if not ( lxarray and lgeospatial ):
        raise NotImplementedError("Only loading via geospatial.xarray_tools is currently implemented.")
    if dataset and subset:
        if dataset != subset:
            raise ValueError((dataset,subset))
    elif dataset and not subset:
        subset = dataset
    if resolution is None:
        if grid and grid[:3] in ('son','snw',): resolution = 'SON60'
        else: resolution = 'NA10' # default
    if varatts is None:
        if grid is None and station is None and shape is None: varatts = varatts_list[subset] # original files
    default_varlist = default_varlists.get(dataset, None)
    xds = loadXRDataset(varname=varname, varlist=varlist, dataset='ERA5', subset=subset, grid=grid, resolution=resolution, shape=shape,
                        station=station, default_varlist=default_varlist, resampling=resampling, varatts=varatts, varmap=varmap, mode=mode,
                        aggregation=aggregation, lgeoref=lgeoref, geoargs=geoargs, chunks=chunks, multi_chunks=multi_chunks, **kwargs)
    # flip latitude dimension
    if lfliplat and 'latitude' in xds.coords:
        xds = xds.reindex(latitude=xds.latitude[::-1])
    # update name and title with sub-dataset
    xds.attrs['name'] = subset
    xds.attrs['title'] = dataset_attributes[subset].title + xds.attrs['title'][len(subset)-1:]
    return xds


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = netcdf_filename # filename pattern: variable name (daily)
ts_file_pattern   = tsfile # filename pattern: variable name and grid
clim_file_pattern = avgfile # filename pattern: grid and period
data_folder       = avgfolder # folder for user data
grid_def  = {'':ERA5Land_grid} # no special name, since there is only one...
LTM_grids = [] # grids that have long-term mean data
TS_grids  = ['',] # grids that have time-series data
grid_res  = {res:0.25 for res in TS_grids} # no special name, since there is only one...
default_grid = ERA5Land_grid
# functions to access specific datasets
loadLongTermMean       = None # climatology provided by publisher
loadDailyTimeSeries    = loadERA5_Daily # daily time-series data
# monthly time-series data for batch processing
def loadTimeSeries(lxarray=False, **kwargs): raise NotImplementedError(lxarray=lxarray, **kwargs)
loadClimatology        = None # pre-processed, standardized climatology
loadStationClimatology = None # climatologies without associated grid (e.g. stations)
loadStationTimeSeries  = None # time-series without associated grid (e.g. stations)
loadShapeClimatology   = None # climatologies without associated grid (e.g. provinces or basins)
loadShapeTimeSeries    = None # time-series without associated grid (e.g. provinces or basins)


## abuse for testing
if __name__ == '__main__':

  import time, gc, os
  import dask
  from dask.diagnostics import ProgressBar

  # Dask scheduler settings - threading can make debugging very difficult
  dask.config.set(scheduler='threading')  # default scheduler - some parallelization
  # dask.config.set(scheduler='synchronous')  # single-threaded for small workload and debugging

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
#   modes += ['load_Point_Climatology']
#   modes += ['load_Point_Timeseries']
  modes += ['derived_variables'     ]
  # modes += ['load_Daily'            ]
  # modes += ['monthly_mean'          ]
#   modes += ['load_TimeSeries'       ]
  # modes += ['monthly_normal'        ]
#   modes += ['load_Climatology'      ]

  grid = None; resampling = None

  dataset = 'ERA5L'
  # resolution = 'SON10'
  resolution = 'NA10'
  # resolution = 'AU10'

  # variable list
#   varlist = ['snow']
  varlist = ['snow','dswe','precip','pet_era5','liqwatflx']

#   period = (2010,2019)
#   period = (1997,2018)
  period = (1997,1998)
#   period = (1980,2018)

  # loop over modes
  for mode in modes:

    if mode == 'load_Climatology':

        pass
#         lxarray = False
#         ds = loadERA5(varlist=varlist, period=period, grid=grid,
#                         lxarray=lxarray) # load regular GeoPy dataset
#         print(ds)
#         print('')
#         varname = list(ds.variables.keys())[0]
#         var = ds[varname]
#         print(var)
#
#         if lxarray:
#             print(('Size in Memory: {:6.1f} MB'.format(var.nbytes/1024./1024.)))


    elif mode == 'load_Point_Climatology':

        pass
#         # load point climatology
#         print('')
#         if pntset in ('shpavg','glbshp'): dataset = loadERA5_Shp(shape=pntset, period=(2009,2018))
#         elif pntset in ('oncat'): dataset = loadERA5_Shp(shape=pntset, grid=grid, period=(2011,2019))
#         else: raise NotImplementedError(pntset)
#         print(dataset)
#         print('')
#         print((dataset.time))
#         print((dataset.time.coord))


    elif mode == 'load_Point_Timeseries':

        pass
#         # load point climatology
#         print('')
#         if pntset in ('oncat'): dataset = loadERA5_ShpTS(shape=pntset, grid=grid, )
#         else: raise NotImplementedError(pntset)
#         print(dataset)
#         print('')
#         print((dataset.time))
#         print((dataset.time.coord))


    elif mode == 'monthly_normal':

        pass

    elif mode == 'load_TimeSeries':

        pass
#         lxarray = False
#         varname = varlist[0]
#         xds = loadERA5_TS(varlist=varlist,
#                             grid=grid, lxarray=lxarray, geoargs=geoargs) # 32 time chunks may be possible
#         print(xds)
#         print('')
#         xv = xds[varname]
#         print(xv)
#         if lxarray:
#             print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))


    elif mode == 'monthly_mean':

        pass

    elif mode == 'load_Daily':

        varlist = ['snow','dswe']
        xds = loadERA5_Daily(varlist=varlist, resolution=resolution, dataset=None, subset='ERA5L', grid=grid,
                             chunks=True, lgeoref=True, join='override', combine_attrs='no_conflicts')
        print(xds)
#         print('')
        xv = xds.data_vars['snow']
# #         xv = list(xds.data_vars.values())[0]
        xv = xv.loc['2011-06-01':'2012-06-01',:,:]
#   #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(xv.mean())
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))


    elif mode == 'derived_variables':

        start = time.time()

        lexec = True
        lappend_master = False
        ts_name = 'time_stamp'
        dataset = 'ERA5L'
        multi_chunks = 'regular'
        load_chunks = bool(multi_chunks)
        # load variables
        derived_varlist = ['dswe',]; load_list = ['snow']
        # derived_varlist = ['liqwatflx',]; load_list = ['dswe', 'precip']
        varatts = varatts_list[dataset]
        xds = loadERA5_Daily(varlist=load_list, subset=dataset, resolution=resolution,
                             grid=grid, chunks=load_chunks, multi_chunks=multi_chunks,
                             lfliplat=False)
        # N.B.: need to avoid loading derived variables, because they may not have been extended yet (time length)
        print(xds)

        # optional slicing (time slicing completed below)
        start_date = None; end_date = None  # auto-detect available data
        # start_date = '2011-01-01'; end_date = '2012-01-01'

        # slice and load time coordinate
        xds = xds.loc[{'time': slice(start_date, end_date)}]
        if ts_name in xds:
            tsvar = xds[ts_name].load()
        else:
            tax = xds.coords['time']
            ts_data = [pd.to_datetime(dt).strftime('%Y-%m-%d_%H:%M:%S') for dt in tax.data]
            tsvar = xr.DataArray(data=ts_data, coords=(tax,), name='time_stamp', attrs=varatts['time_stamp'])

        # loop over variables
        for varname in derived_varlist:

            # target dataset
            lskip = False
            folder, filename = getFolderFileName(varname=varname, dataset='ERA5', subset=dataset, resolution=resolution, grid=grid,
                                                 resampling=resampling, mode='daily', lcreateFolder=True)
            nc_filepath = '{}/{}'.format(folder, filename)
            if lappend_master and osp.exists(nc_filepath):
                ncds = nc.Dataset(nc_filepath, mode='a')
                ncvar3 = ncds[varname]
                ncts = ncds[ts_name]
                nctc = ncds['time']  # time coordinate
                # update start date for after present data
                start_date = pd.to_datetime(ncts[-1]) + pd.to_timedelta(1,unit='D')
                if end_date is None:
                    end_date = tsvar.data[-1]
                end_date = pd.to_datetime(end_date)
                if start_date > end_date:
                    print(("\nNothing to do - timeseries complete:\n {} > {}".format(start_date,end_date)))
                    ncds.close()
                    lskip = True
                else:
                    lappend = True
                    # update slicing (should not do anything if sliced before)
                    print(("\n Appending data from {} to {}.\n".format(start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))))
                    xds = xds.loc[{'time': slice(start_date, end_date),}]
                    tsvar = tsvar.loc[{'time': slice(start_date, end_date),}]
            else:
                lappend = False

            if not lskip:

                print('\n')
                default_varatts = varatts[varname]  # need to ensure netCDF compatibility
                ## define actual computation
                if varname == 'liqwatflx':
                    ref_var = xds['precip']
                    note = "masked/missing values have been replaced by zero"
                    xvar = ref_var.fillna(0) - xds['dswe'].fillna(0)  # fill missing values with zero
                    # N.B.: missing values are NaN in xarray; we need to fill with 0, or masked/missing values
                    #       in snowmelt will mask/invalidate valid values in precip
                elif varname == 'dswe':
                    ref_var = xds['snow']
                    note = "Rate of Daily SWE Changes"
                    assert ref_var.attrs['units'] == 'kg/m^2', ref_var.attrs['units']
                    # xvar = ref_var.differentiate('time', datetime_unit='s')
                    xvar = ref_var.diff('time', n=1, label='upper') / 86400  # per second
                    # expand time axis
                    xvar = xvar.broadcast_like(ref_var).fillna(0)

                # define/copy metadata
                xvar.attrs = ref_var.attrs.copy()
                xvar = xvar.rename(varname)
                for att in ('name', 'units', 'long_name',):  # don't copy scale factors etc...
                    if att in default_varatts:
                        xvar.attrs[att] = default_varatts[att]
                assert xvar.attrs['name'] == xvar.name, xvar.attrs
                for att in list(xvar.attrs.keys()):
                    if att.startswith('old_') or att in ('original_name',  'standard_name'):
                        del xvar.attrs[att]  # does not apply anymore
                xvar.attrs['note'] = note
                # set chunking for operation
                chunks = ref_var.encoding['chunksizes'] if load_chunks is True else load_chunks.copy()
                if chunks:
                    if isinstance(chunks,dict):
                        chunks = tuple(chunks[dim] for dim in xvar.dims)
                    xvar = xvar.chunk(chunks=chunks)
                # print('Chunks:', xvar.chunks)


          #       # visualize task graph
          #       viz_file = daily_folder+'dask_sum.svg'
          #       xvar3.data.visualize(filename=viz_file)
          #       print(viz_file)


                ## now save data, according to destination/append mode
                if lappend:
                    # append results to an existing file
                    print('\n')
                    # define chunking
                    offset = ncts.shape[0]; t_max = offset + tsvar.shape[0]
                    tc,yc,xc = xvar.chunks # starting points of all blocks...
                    tc = np.concatenate([[0],np.cumsum(tc[:-1], dtype=np.int)])
                    yc = np.concatenate([[0],np.cumsum(yc[:-1], dtype=np.int)])
                    xc = np.concatenate([[0],np.cumsum(xc[:-1], dtype=np.int)])
                    # xvar3 = xvar3.chunk(chunks=(tc,xvar3.shape[1],xvar3.shape[2]))
                    # function to save each block individually (not sure if this works in parallel)
                    dummy = np.zeros((1,1,1), dtype=np.int8)
                    def save_chunk(block, block_id=None):
                        ts = offset + tc[block_id[0]]; te = ts + block.shape[0]
                        ys = yc[block_id[1]]; ye = ys + block.shape[1]
                        xs = xc[block_id[2]]; xe = xs + block.shape[2]
                        #print(((ts,te),(ys,ye),(xs,xe)))
                        #print(block.shape)
                        ncvar3[ts:te, ys:ye, xs:xe] = block
                        return dummy
                    # append to NC variable
                    xvar.data.map_blocks(save_chunk, chunks=dummy.shape, dtype=dummy.dtype).compute() # drop_axis=(0,1,2),
                    # update time stamps and time axis
                    nctc[offset:t_max] = np.arange(offset, t_max)
                    for i in range(tsvar.shape[0]):
                        ncts[i + offset] = tsvar.data[i]
                    ncds.sync()
                    print('\n')
                    print(ncds)
                    ncds.close()
                    del xvar, ncds
                else:
                    # save results in new file
                    nds = xr.Dataset({ts_name: tsvar, varname: xvar}, attrs=xds.attrs.copy())
                    nds.coords['time'].attrs.pop('units', None)  # needs to be free for use by xarray
                    print('\n')
                    print(nds)
                    print(nc_filepath)
                    # write to NetCDF
                    tmp_filepath = nc_filepath + '.tmp'  # use temporary file during creation
                    var_enc = dict(chunksizes=chunks, zlib=True, complevel=1, _FillValue=np.NaN, dtype=netcdf_dtype)
                    task = nds.to_netcdf(tmp_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                                         encoding={varname: var_enc}, compute=False)
                    if lexec:
                        with ProgressBar():
                            task.compute()
                    else:
                        print(var_enc)
                        print(task)
                        task.visualize(filename=folder + 'netcdf.svg')  # This file is never produced
                    del nds, xvar
                    # replace original file
                    if os.path.exists(nc_filepath):
                        os.remove(nc_filepath)
                    os.rename(tmp_filepath, nc_filepath)

                # clean up
                gc.collect()

        # print timing
        end = time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
