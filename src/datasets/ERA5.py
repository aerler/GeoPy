'''
Created on Nov. 07, 2020

A module to read ERA5 data; this includes converting GRIB files to NetCDF-4, 
as well as functions to load the converted and aggregated data.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os
import os.path as osp
import pandas as pd
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
from warnings import warn

# internal imports
from geodata.misc import name_of_month, days_per_month
from utils.nctools import add_var
from datasets.common import getRootFolder, loadObservations
from geodata.gdal import GridDefinition, addGDALtoDataset, grid_folder
from geodata.netcdf import DatasetNetCDF
from datasets.misc import loadXRDataset


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
varatts_list['ERA5L'] = dict(# forcing variables
             tp   = dict(name='precip', units='kg/m^2/s',scalefactor=1./3.6, long_name='Total Precipitation'), # units of meters water equiv. / hour
#              solprec = dict(name='solprec',units='kg/m^2/s',scalefactor=1./3.6, long_name='Solid Precipitation'),
             # state variables
             sd   = dict(name='snow',  units='kg/m^2', scalefactor=1.e3, long_name='Snow Water Equivalent'), # units of meters water equivalent
             # diagnostic variables
#              snwmlt    = dict(name='snwmlt',   units='kg/m^2/s',scalefactor=1./3.6, long_name='Snow Melt Runoff at the Base of the Snow Pack'),
#              evap_snow = dict(name='evap_snow',units='kg/m^2/s',scalefactor=1./3.6, long_name='Sublimation from the Snow Pack'),
             # axes (don't have their own file)
             time_stamp = dict(name='time_stamp', units='', long_name='Time Stamp'), # readable time stamp (string)
             time = dict(name='time', units='days', long_name='Days'), # time coordinate
             lon  = dict(name='lon', units='deg', long_name='Longitude'), # geographic longitude
             lat  = dict(name='lat', units='deg', long_name='Latitude'), # geographic latitude
             # derived variables
             dswe = dict(name='dswe',units='kg/m^2/s',scalefactor=1., long_name='SWE Changes'),
#              liqwatflx = dict(name='liqwatflx',units='kg/m^2/s',scalefactor=1., long_name='Liquid Water Flux'),
             )
# list of variables to load
default_varlists = {name:[atts['name'] for atts in varatts.values()] for name,varatts in varatts_list.items()}

# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/' 
avgfile   = 'era5{0:s}_clim{1:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = 'era5_{0:s}{1:s}{2:s}_monthly.nc' # extend with biascorrection, variable and grid type
daily_folder    = root_folder + dataset_name.lower()+'_daily/' 
netcdf_filename = 'era5_{:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float
netcdf_settings = dict(chunksizes=(8,ERA5Land_size[0]/16,ERA5Land_size[1]/32))


## functions to load NetCDF datasets (using xarray)

def loadERA5_Daily(varname=None, varlist=None, dataset='ERA5', grid=None, resolution=None, shape=None, station=None, 
                   resampling=None, varatts=None, varmap=None, lgeoref=True, geoargs=None,  
                   chunks=None, lautoChunk=False, lxarray=True, lgeospatial=True, **kwargs):
    ''' function to load daily ERA5 data from NetCDF-4 files using xarray and add some projection information '''
    if not ( lxarray and lgeospatial ): 
        raise NotImplementedError("Only loading via geospatial.xarray_tools is currently implemented.")
    if resolution is None: 
        if grid and grid[:3] in ('son','snw',): resolution = 'SON60'
        else: resolution = 'CA12' # default
    if varatts is None:
        if grid is None and station is None and shape is None: varatts = varatts_list[dataset] # original files
    default_varlist = default_varlists.get(dataset, None)
    xds = loadXRDataset(varname=varname, varlist=varlist, dataset='ERA5', filetype=dataset, grid=grid, resolution=resolution, shape=shape,
                        station=station, default_varlist=default_varlist, resampling=resampling, varatts=varatts, varmap=varmap,  
                        lgeoref=lgeoref, geoargs=geoargs, chunks=chunks, lautoChunk=lautoChunk, **kwargs)
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
def loadTimeSeries(lxarray=False, **kwargs): return loadERA5_TS(lxarray=lxarray, **kwargs)
loadClimatology        = loadERA5 # pre-processed, standardized climatology
loadStationClimatology = None # climatologies without associated grid (e.g. stations) 
loadStationTimeSeries  = None # time-series without associated grid (e.g. stations)
loadShapeClimatology   = None # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries    = None # time-series without associated grid (e.g. provinces or basins)


## abuse for testing
if __name__ == '__main__':

  import dask, time, gc 
  
  #print('xarray version: '+xr.__version__+'\n')
  xr.set_options(keep_attrs=True)
        

#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=4, diagnostics_port=18787)
#   client = Client(cluster)

#   from multiprocessing.pool import ThreadPool
#   dask.set_options(pool=ThreadPool(4))

  modes = []
#   modes += ['load_Point_Climatology']
#   modes += ['load_Point_Timeseries']
  modes += ['derived_variables'     ]
#   modes += ['load_Daily'            ]
#   modes += ['monthly_mean'          ]
#   modes += ['load_TimeSeries'       ]
#   modes += ['monthly_normal'        ]
#   modes += ['load_Climatology'      ]

  grid = None

  dataset = 'ERA5L'
  resolution = 'SON10'
  
  # variable list
  varlist = ['snow']
#   varlist = ['snow','dswe']
  
#   period = (2010,2019)
  period = (1997,2018)

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
       
#         varlist = ['liqwatflx','precip','rho_snw']
#         varname = varlist[0]
        xds = loadERA5_Daily(varlist=varlist, resolution=resolution, dataset=dataset, grid=grid) # 32 may be possible
        print(xds)
        print('')
        xv = list(xds.data_vars.values())[0]
        xv = xv.loc['2011-01-01':'2011-02-01',55:45,-80:-70]
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(xv.mean())
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
      
    elif mode == 'add_variables':
      
        lappend_master = True
        start = time.time()
            
        # load variables
        ts_name = 'time_stamp'
        derived_varlist = ['dswe',]
        varatts = varatts_list[dataset]
        xds = loadERA5_Daily(varlist=varlist, dataset=dataset, resolution=resolution, grid=grid)
        # N.B.: need to avoid loading derived variables, because they may not have been extended yet (time length)
        print(xds)
        
        # optional slicing (time slicing completed below)
        start_date = None; end_date = None # auto-detect available data
        start_date = '2011-01-01'; end_date = '2011-01-08'
  
        # slice and load time coordinate
        xds = xds.loc[{'time':slice(start_date,end_date),}]
        tsvar = xds[ts_name].load()
            
        
        # loop over variables
        for var in derived_varlist:
        
            # target dataset
            lexec = True
            var_atts = varatts[var]
            varname = var; folder = daily_folder
            if grid: 
                varname = '{}_{}'.format(varname,grid) # also append non-native grid name to varname
                folder = '{}/{}'.format(folder,grid)
            if biascorrection: varname = '{}_{}'.format(biascorrection,varname) # prepend bias correction method
            nc_filepath = '{}/{}'.format(folder,netcdf_filename.format(varname))
            if lappend_master and osp.exists(nc_filepath):
                ncds = nc.Dataset(nc_filepath, mode='a')
                ncvar3 = ncds[var]
                ncts = ncds[ts_name]
                nctc = ncds['time'] # time coordinate
                # update start date for after present data
                start_date = pd.to_datetime(ncts[-1]) + pd.to_timedelta(1,unit='D')
                if end_date is None: end_date = tsvar.data[-1]
                end_date = pd.to_datetime(end_date)
                if start_date > end_date:
                    print(("\nNothing to do - timeseries complete:\n {} > {}".format(start_date,end_date)))
                    ncds.close()
                    lexec = False
                else:
                    lappend = True
                    # update slicing (should not do anything if sliced before)
                    print(("\n Appending data from {} to {}.\n".format(start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))))
                    xds = xds.loc[{'time':slice(start_date,end_date),}]
                    tsvar = tsvar.loc[{'time':slice(start_date,end_date),}]
            else: 
                lappend = False
                
            if lexec:
              
                print('\n')
                ## define actual computation
                if var == 'liqwatflx':
                    ref_var = 'snwmlt'; note = "masked/missing values have been replaced by zero"
                    xvar = xds['snwmlt'].fillna(0) + xds['liqprec'].fillna(0) # fill missing values with zero
                    # N.B.: missing values are NaN in xarray; we need to fill with 0, or masked/missing values
                    #       in snowmelt will mask/invalidate valid values in precip
                elif var == 'precip':
                    ref_var = 'liqprec'; note = "masked/missing values have been replaced by zero"
                    xvar = xds['liqprec'].fillna(0) + xds['solprec'].fillna(0) # fill missing values with zero
                    # N.B.: missing values are NaN in xarray; we need to fill with 0, or masked/missing values
                    #       in snowmelt will mask/invalidate valid values in precip
                elif var == 'rho_snw':
                    ref_var = 'snow'; note = "SWE divided by snow depth, divided by 1000"
                    xvar = xds['snow'] / xds['snowh']
                    
                # define/copy metadata
                xvar.rename(var)
                xvar.attrs = xds[ref_var].attrs.copy()
                for att in ('name','units','long_name',):
                    xvar.attrs[att] = var_atts[att]
                xvar.attrs['note'] = note
                xvar.chunk(chunks=chunk_settings)
                print(xvar)
          
                
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
          #           xvar3 = xvar3.chunk(chunks=(tc,xvar3.shape[1],xvar3.shape[2]))
                    # function to save each block individually (not sure if this works in parallel)
                    dummy = np.zeros((1,1,1), dtype=np.int8)
                    def save_chunk(block, block_id=None):
                        ts = offset + tc[block_id[0]]; te = ts + block.shape[0]
                        ys = yc[block_id[1]]; ye = ys + block.shape[1]
                        xs = xc[block_id[2]]; xe = xs + block.shape[2]
                        #print(((ts,te),(ys,ye),(xs,xe)))
                        #print(block.shape)
                        ncvar3[ts:te,ys:ye,xs:xe] = block
                        return dummy
                    # append to NC variable
                    xvar.data.map_blocks(save_chunk, chunks=dummy.shape, dtype=dummy.dtype).compute() # drop_axis=(0,1,2), 
                    # update time stamps and time axis
                    nctc[offset:t_max] = np.arange(offset,t_max)
                    for i in range(tsvar.shape[0]): ncts[i+offset] = tsvar.data[i] 
                    ncds.sync()
                    print('\n')
                    print(ncds)
                    ncds.close()
                    del xvar, ncds 
                else:
                    # save results in new file
                    nds = xr.Dataset({ts_name:tsvar, var:xvar,}, attrs=xds.attrs.copy())
      #               print('\n')
      #               print(nds)
                    # write to NetCDF
                    var_enc = dict(zlib=True, complevel=1, _FillValue=-9999, chunksizes=netcdf_settings['chunksizes'])
                    nds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                                  encoding={var:var_enc,}, compute=True)
                    del nds, xvar
                    
                # clean up
                gc.collect()
            
        # print timing
        end =  time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
      