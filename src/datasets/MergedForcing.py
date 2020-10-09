'''
Created on May 25, 2020

A module to merge different high-resolution datesets and load the resulting merged dataset; mainly for hydrological modeling;
a major secondary purpose of this module is also, to keep xarray dependencies out of other modules (in particular, NRCan)

@author: Andre R. Erler, GPL v3
'''



# external imports
import datetime as dt
import pandas as pd
import os
from warnings import warn
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
import dask
from collections import namedtuple
from importlib import import_module
import inspect
# internal imports
from datasets.common import getRootFolder, grid_folder
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset
# for georeferencing
from geospatial.netcdf_tools import autoChunk, addTimeStamps, addNameLengthMonth
from geospatial.xarray_tools import addGeoReference, loadXArray, updateVariableAttrs, computeNormals

## Meta-vardata

dataset_name = 'MergedForcing'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='HGS') # get dataset root folder based on environment variables

# attributes of variables in different collections
# Axes and static variables
axes_varatts = dict(time = dict(name='time', units='hours', long_name='Days'), # time coordinate
                    lon = dict(name='lon', units='deg', long_name='Longitude'), # longitude coordinate
                    lat = dict(name='lat', units='deg', long_name='Latitude'), # latitude coordinate
                    x  = dict(name='x', units='m', long_name='Easting'),
                    y  = dict(name='y', units='m', long_name='Northing'),)
axes_varlist = axes_varatts.keys()
# merged/mixed/derived variables
varatts = dict(liqwatflx = dict(name='liqwatflx', units='kg/m^2/s', long_name='Liquid Water Flux'),
               pet_pt = dict(name='pet_pt', units='kg/m^2/s', long_name='PET (Priestley-Taylor)'),
               pet_pts = dict(name='pet_pts', units='kg/m^2/s', long_name='PET (Priestley-Taylor, approx. LW)'),
               pet_hog = dict(name='pet_hog', units='kg/m^2/s', long_name='PET (Hogg 1997)'),
               pet_har = dict(name='pet_har', units='kg/m^2/s', long_name='PET (Hargeaves)'),
               pet_haa = dict(name='pet_haa', units='kg/m^2/s', long_name='PET (Hargeaves-Allen)'),
               pet_th  = dict(name='pet_th', units='kg/m^2/s', long_name='PET (Thornthwaite)'),
               )
varlist = varatts.keys()
ignore_list = []

# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/' 
avgfile   = '{DS:s}_{GRD:s}_clim_{PRD:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = '{DS:s}_{GRD:s}_monthly.nc' # extend with biascorrection, variable and grid type
# daily data
daily_folder    = root_folder + 'merged_daily/' 
netcdf_filename = 'merged_{VAR:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float

# list of available datasets/collections
DSNT = namedtuple(typename='Dataset', field_names=['name','interval','start_date','end_date',])
dataset_attributes = dict(SnoDAS  = DSNT(name='SnoDAS',interval='1D', start_date=None, end_date=None,  ),                          
                          NRCan   = DSNT(name='NRCan',  interval='1D', start_date=None, end_date=None, ), 
                          #CaSPAr  = DSNT(name='CaSPAr',  interval='6H', start_date='2017-09-11T12', end_date='2019-12-30T12', ),
                          )
dataset_list = list(dataset_attributes.keys())
# N.B.: the effective start date for CaPA and all the rest is '2017-09-11T12'
default_dataset_index = dict(precip='NRCan', T2='NRCan', Tmin='NRCan', Tmax='NRCan', 
                             snow='SnoDAS', dswe='SnoDAS',
                             lat2D='const', lon2D='const', zs='const')


## helper functions

def getFolderFileName(varname=None, dataset=None, grid=None, resampling=None, resolution=None, bias_correction=None, 
                      mode=None, period=None, lcreateFolder=True):
    ''' function to provide the folder and filename for the requested dataset parameters '''
    if mode is None:
        mode = 'clim' if period else 'daily'
    # some default settings
    if dataset is None: 
        dataset = default_dataset_index.get(varname,'MergedForcing')
    # dataset-specific settings
    if dataset.lower() == 'mergedforcing': 
        ds_str = ds_str_rs = 'merged'
    elif dataset.lower() == 'nrcan':
        if not resolution: resolution = 'CA12'
        ds_str = dataset.lower()
        ds_str_rs = ds_str + '_' + resolution.lower()
    else:
        ds_str = ds_str_rs = dataset.lower()
    # construct filename
    gridstr = '_' + grid.lower() if grid else ''
    bcstr = '_' + bias_correction.lower() if bias_correction else ''
    if mode.lower() == 'daily': name_str = bcstr + '_' + varname.lower() + gridstr
    else: name_str = bcstr + gridstr
    mode_str = mode.lower()
    if period is None: pass
    elif isinstance(period,str): mode_str += '_'+period
    elif isinstance(period,(tuple,list)): mode_str += '_{}-{}'.format(*period)
    else: raise NotImplementedError(period)
    filename = '{}{}_{}.nc'.format(ds_str_rs, name_str, mode_str)
    # construct folder
    folder = getRootFolder(dataset_name=dataset, fallback_name='MergedForcing')
    if mode.lower() == 'daily':
        folder += ds_str+'_daily'
        if grid: 
            folder = '{}/{}'.format(folder,grid) # non-native grids are stored in sub-folders
            if resampling: 
                folder = '{}/{}'.format(folder,resampling) # different resampling options are stored in subfolders
                # could auto-detect resampling folders at a later point... 
    else: folder += ds_str+'avg'
    if folder[-1] != '/': folder += '/'
    if lcreateFolder: os.makedirs(folder, exist_ok=True)
    # return folder and filename
    return folder,filename


def addConstantFields(xds, const_list=None, grid=None):
    ''' add constant auxiliary fields like topographic elevation and geographic coordinates to dataset '''
    if const_list is None:
        const_list = ['lat2D', 'lon2D']
    # find horizontal coordinates
    dims = (xds.ylat,xds.xlon)
    for rv in xds.data_vars.values():
        if xds.ylat in rv.dims and xds.xlon in rv.dims: break
    if dask.is_dask_collection(rv):
        chunks = {dim:chk for dim,chk in zip(rv.dims, rv.encoding['chunksizes']) if dim in dims}
    else:
        chunks = None # don't chunk if nothing else is chunked...
    # add constant variables
    if 'lat2D' in const_list or 'lon2D' in const_list:
        # add geographic coordinate fields 
        if grid is None: 
            raise NotImplementedError() # probably a rare case, since datasets need to be on a common grid
        else:
            from geodata.gdal import loadPickledGridDef
            griddef = loadPickledGridDef(grid='son2')
            # add local latitudes
            if 'lat2D' in const_list:
                atts = dict(name='lat2d', long_name='Latitude', units='deg N')
                xvar = xr.DataArray(data=griddef.lat2D, attrs=atts, dims=dims)
                if chunks: xvar = xvar.chunk(chunks=chunks)
                xds['lat2D'] = xvar
            # add local longitudes
            if 'lon2D' in const_list:
                atts = dict(name='lon2d', long_name='Longitude', units='deg E')
                xvar = xr.DataArray(data=griddef.lon2D, attrs=atts, dims=dims)
                if chunks: xvar = xvar.chunk(chunks=chunks)
                xds['lon2D'] = xvar 
    if 'zs' in const_list:
        print("Loading of surface/topographic elevation is not yet implemented")
    return xds        


## functions to load NetCDF datasets (using xarray)


def loadMergedForcing_Daily(varname=None, varlist=None, dataset_index=None, dataset_args=None, time_slice=None, 
                            compat='override', join='inner', fill_value=None, ldebug=False, **kwargs):
    ''' function to load and merge data from different high-resolution datasets (e.g. SnoDAS or NRCan) using xarray;
        typical dataset-agnostic arguments: grid=str, lgeoref=True, geoargs=dict, chunks=dict, lautoChunk=False, 
        typical dataset-specific arguments: folder=str, resampling=str, resolution=str, bias_correction=str '''
    # figure out varlist
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname:
        varlist = [varname] # load a single variable
    elif varlist is None:
        varlist = list(varatts.keys())
    if dataset_args is None: dataset_args = dict()# avoid errors
    # assemble dataset list and arguments
    if isinstance(varlist,dict):
        dataset_varlists = varlist
    else:
        if dataset_index is None: dataset_index = default_dataset_index.copy()
        dataset_varlists = dict()
        for varname in varlist:
            ds_name = dataset_index.get(varname,dataset_name) # default is native (global variable)
            if ds_name not in dataset_varlists: dataset_varlists[ds_name] = [varname] 
            else: dataset_varlists[ds_name].append(varname)
    const_list = dataset_varlists.pop('const', [])
    ## load datasets
    ds_list = []
    for dataset,varlist in dataset_varlists.items():
        if ldebug: print("Loading", dataset, '\n', varlist, '\n')
        # prepare kwargs
        ds_args = kwargs.copy(); 
        if dataset in dataset_args: ds_args.update(dataset_args[dataset])
        if dataset.lower() == dataset_name.lower():
            # native MergedForcing
            ds_args.update(folder=daily_folder, filename_pattern=netcdf_filename)
            argslist = ['grid', ] # specific arguments for merged dataset variables
            if varlist is None: varlist = list(varatts.keys())
            loadFunction = loadXArray
        else:
            # daily data from other datasets
            ds_mod = import_module('datasets.{0:s}'.format(dataset)) # import dataset module
            loadFunction = ds_mod.loadDailyTimeSeries
            argslist = inspect.getfullargspec(loadFunction); argslist = argslist.args # list of actual arguments
        # remove some args that don't apply
        for key in ('resolution','bias_correction'): # list of dataset-specific arguments that have to be controlled
            if key not in argslist and key in ds_args: del ds_args[key]
        # load time series and and apply some formatting to vars
        ds = loadFunction(varlist=varlist, **ds_args)
        # add some dataset attributes to variables, since we will be merging datasets
        for var in ds.variables.values(): var.attrs['dataset_name'] = dataset_name
        if 'resampling' in ds.attrs:
            for var in ds.data_vars.values(): var.attrs['resampling'] = ds.attrs['resampling']
        if 'bias_correction' in ds_args:
            for var in ds.data_vars.values(): var.attrs['bias_correction'] = ds_args['bias_correction']
        if time_slice: ds = ds.loc[{'time':slice(*time_slice),}] # slice time
        ds_list.append(ds)
    # merge datasets and attributed
    if ldebug: print("Merging Datasets:", compat, join, '\n')
    xds = xr.merge(ds_list, compat=compat, join=join, fill_value=fill_value)
    for ds in ds_list[::-1]: xds.attrs.update(ds.attrs) # we want MergedForcing to have precedence
    xds.attrs['name'] = 'MergedForcing'; xds.attrs['title'] = 'Merged Forcing Daily Timeseries'
    if 'resampling' in xds.attrs: del xds.attrs['resampling'] # does not apply to a merged dataset
    ## add additional fields
    if ldebug: print("Adding Constants:", const_list, '\n',)
    xds = addConstantFields(xds, const_list=const_list, grid=kwargs.get('grid',None))
    # return merged dataset
    return xds
loadDailyTimeSeries = loadMergedForcing_Daily


def loadMergedForcing_All(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, 
                          mode=None, period=None, lxarray=True, lmonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    # resolve folder and filename
    file_args = {key:kwargs.pop(key,None) for key in ('resampling', 'resolution', 'bias_correction')}
    folder,filename = getFolderFileName(varname=None, dataset=dataset_name, grid=grid, mode=mode, period=period, 
                                        lcreateFolder=False, **file_args)
    # remove some common arguments that have no meaning
    if name is None: name = dataset_name
    for key in ('resolution','bias_correction'):
        if key in kwargs: del kwargs[key]
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname: varlist = [varname]
    if lxarray: 
        ## load as xarray dataset
        # set options
        if lmonthly: kwargs['decode_times'] = False
        # load  dataset
        xds = xr.open_dataset(folder+filename, **kwargs)
        # update varatts and prune
        xds = updateVariableAttrs(xds, varatts=varatts, varmap=None, varlist=varlist)
        # some attributes
        xds.attrs['name'] = name
        # load time stamps (like coordinate variables)
        if 'time_stamp' in xds: xds['time_stamp'].load()
        # fix time axis (deprecated - should not be necessary anymore)
        if lmonthly:
            warn("'lmonthly=True' should only be used to convert simple monthly indices into 'datetime64' coordinates.")
            # convert a monthly time index into a daily index, anchored at the first day of the month
            tattrs = xds['time'].attrs.copy()
            tattrs['long_name'] = 'Calendar Day'
            tattrs['units'] = tattrs['units'].replace('months','days')
            start_date = pd.to_datetime(' '.join(tattrs['units'].split()[2:]))
            end_date = start_date + pd.Timedelta(len(xds['time'])+1, unit='M')
            tdata = np.arange(start_date,end_date, dtype='datetime64[M]')
            assert len(tdata) == len(xds['time'])
            tvar = xr.DataArray(tdata, dims=('time'), name='time', attrs=tattrs)
            xds = xds.assign_coords(time=tvar)        
        # add projection
        if lgeoref:
            if geoargs is None: geoargs = dict() 
            xds = addGeoReference(xds, **geoargs)
        dataset = xds
    else:
        ## load as GeoPy dataset
        # load dataset
        dataset = DatasetNetCDF(name=name, filelist=[folder+filename], varlist=varlist, multifile=False, 
                                varatts=varatts, **kwargs)
        # fix axes units:
        for ax in ('x','y','lat','lon'):
            if ax in dataset.axes: dataset.axes[ax].atts.update(axes_varatts[ax])
        # add GDAL to dataset
        default_geoargs = dict(griddef=grid, gridfolder=grid_folder)
        if geoargs: default_geoargs.update(geoargs)
        dataset = addGDALtoDataset(dataset, **default_geoargs)
    return dataset


def loadMergedForcing_TS(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None,
                         lxarray=True, lmonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset_name=dataset_name, varatts=varatts, grid=grid, 
                                 mode='monthly', period=None, lxarray=lxarray, lmonthly=lmonthly, lgeoref=lgeoref, geoargs=geoargs, **kwargs)

def loadMergedForcing(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, period=None,
                      lxarray=True, lmonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly normal merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset_name=dataset_name, varatts=varatts, grid=grid, 
                                 mode='clim', period=period, lxarray=lxarray, lmonthly=lmonthly, lgeoref=lgeoref, geoargs=geoargs, **kwargs)


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = netcdf_filename # filename pattern: variable name (daily)
ts_file_pattern   = tsfile # filename pattern: variable name and grid
clim_file_pattern = avgfile # filename pattern: grid and period
data_folder       = avgfolder # folder for user data
grid_def  = {'':None} # no special name, since there is only one...
LTM_grids = [] # grids that have long-term mean data 
TS_grids  = ['','rfbc'] # grids that have time-series data
grid_res  = {res:0.00833333333333333 for res in TS_grids} # no special name, since there is only one...
default_grid = None
# functions to access specific datasets
loadLongTermMean       = None # climatology provided by publisher
loadDailyTimeSeries    = loadMergedForcing_Daily # daily time-series data
# monthly time-series data for batch processing
def loadTimeSeries(lxarray=False, **kwargs): return loadMergedForcing_TS(lxarray=lxarray, **kwargs)
def loadClimatology(lxarray=False, **kwargs): return loadMergedForcing(lxarray=lxarray, **kwargs)
loadStationClimatology = None # climatologies without associated grid (e.g. stations) 
loadStationTimeSeries  = None # time-series without associated grid (e.g. stations)
loadShapeClimatology   = None # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries    = None # time-series without associated grid (e.g. provinces or basins)


## abuse for testing
if __name__ == '__main__':
  
  import time, gc
  from multiprocessing.pool import ThreadPool
  
  print('xarray version: '+xr.__version__+'\n')
        

#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=4, diagnostics_port=18787)
#   client = Client(cluster)

#   from multiprocessing.pool import ThreadPool
#   dask.set_options(pool=ThreadPool(4))

  work_loads = []
#   work_loads += ['print_grid']
#   work_loads += ['compute_derived']
#   work_loads += ['load_Daily']
#   work_loads += ['monthly_mean'          ]
#   work_loads += ['load_TimeSeries'      ]
  work_loads += ['monthly_normal'        ]
  work_loads += ['load_Climatology'      ]

  # some settings
  grid = None
#   grid = 'hd1' # small Quebec grid
  grid = 'son2' # high-res Southern Ontario
 
  ts_name = 'time_stamp'
  
 
  # loop over modes 
  for mode in work_loads:
    
    if mode == 'print_grid':
        
        from geodata.gdal import loadPickledGridDef
        griddef = loadPickledGridDef(grid='son2')
        print(griddef)
        print(griddef.lat2D)
        
    elif mode == 'load_Climatology':
       
        lxarray = False
        varname = 'T2'
        period = (2011,2018); kwargs = dict()
#         period = (1980,2010); kwargs = dict(dataset_name='NRCan', resolution='NA12', varlist=[varname]) # load regular NRCan normals
        xds = loadMergedForcing(grid=grid, lxarray=lxarray, period=period, **kwargs)
        print(xds)
        print('')
        xv = xds[varname]
        print(xv)
        if lxarray:
            print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
    
    elif mode == 'monthly_normal':
  
        # optional slicing (time slicing completed below)
        start_date = '2011-01'; end_date = '2011-12'; varlist = None
#         start_date = '2011-01'; end_date = '2017-12'; varlist = None # date ranges are inclusive
  
        # start operation
        start = time.time()
            
        # load variables object (not data!)
        xds   = loadMergedForcing_TS(varlist=varlist, grid=grid, lxarray=True) # need Dask!
        xds   = xds.loc[{'time':slice(start_date,end_date),}] # slice entire dataset
        print(xds)
        
        # construct period string
        print('\n')
        cds = computeNormals(xds, aggregation='month', time_stamp=ts_name)
        print(cds)
        print('\n')
        prdstr = cds.attrs['period']
        print(prdstr)            
        
        # save resampled dataset
        folder, filename = getFolderFileName(dataset=dataset_name, grid=grid, period=prdstr)
        # write to NetCDF
        var_enc = dict(zlib=True, complevel=1, _FillValue=-9999)
        encoding = {varname:var_enc for varname in cds.data_vars.keys()}
        cds.to_netcdf(folder+filename, mode='w', format='NETCDF4', unlimited_dims=[], engine='netcdf4',
                      encoding=encoding, compute=True)
        
        # add name and length of month (doesn't work properly with xarray)
        ds = nc.Dataset(folder+filename, mode='a')
        ds = addNameLengthMonth(ds, time_dim='time')
        # close NetCDF dataset
        ds.close()
        
        # print timing
        end = time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
  
  
    elif mode == 'load_TimeSeries':
       
        lxarray = True
        varname = 'pet_th'
        xds = loadMergedForcing_TS(varlist=None, grid=grid, lxarray=lxarray)
        print(xds)
        print('')
        xv = xds[varname]
        print(xv)
        if lxarray:
            print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
    
    elif mode == 'monthly_mean':
        
        # settings
        load_chunks = None; lautoChunkLoad = True  # chunking input should not be necessary, if the source files are chunked properly
        chunks = None; lautoChunk = True # auto chunk output - this is necessary to maintain proper chunking!
#         time_slice = ('2011-01-01','2011-12-31') # inclusive
        time_slice = None
        varlist = {dataset:None for dataset in dataset_list+[dataset_name, 'const']} # None means all...
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, bias_correction='rfbc', dataset_args=None, lskip=True, 
                                      lautoChunk=lautoChunkLoad, time_slice=time_slice, ldebug=False)
        print(xds)
        print('')
        
        # start operation
        start = time.time()
        
        # aggregate month
        rds = xds.resample(time='MS',skipna=True,).mean()
        #rds.chunk(chunks=chunk_settings)
        print(rds)
        print('')
        
        # define destination file
        nc_folder, nc_filename = getFolderFileName(dataset=dataset_name, grid=grid, bias_correction=None, mode='monthly')
        nc_filepath = nc_folder + nc_filename
        print("\nExporting to new NetCDF-4 file:\n '{}'".format(nc_filepath))
        # write to NetCDF
        var_enc = dict(chunksizes=chunks, zlib=True, complevel=1, _FillValue=np.NaN,) # should be float
        enc_varlist = rds.data_vars.keys()
        rds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                      encoding={vn:var_enc for vn in enc_varlist}, compute=True)
        # update time information
        print("\nAdding human-readable time-stamp variable ('time_stamp')\n")
        ncds = nc.Dataset(nc_filepath, mode='a')
        ncts = addTimeStamps(ncds, units='month') # add time-stamps        
        ncds.close()
        # print timing
        end = time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
  
                             
    elif mode == 'load_Daily':
       
  #       varlist = netcdf_varlist
#         varlist = ['precip','snow','liqwatflx']
        varlist = {dataset:None for dataset in dataset_list+[dataset_name, 'const']} # None means all...
#         varlist = {'NRCan':None}
        dataset_args = dict(SnoDAS=dict(bias_correction='rfbc'))
#         time_slice = ('2011-01-01','2017-01-01')
        time_slice = None
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, bias_correction='rfbc', dataset_args=dataset_args, 
                                      time_slice=time_slice, lautoChunk=True)
        print(xds)
        print('')
        print(xds.attrs)
        print('')
#         # check for zeros in temperature field... (Kelvin!)
#         for varname in ('T2','Tmin','Tmax'):
#             if varname in xds:
#                 xvar = xds[varname]
#                 zeros = xvar < 100
#                 print(varname,zeros.data.sum())            
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
#         xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(xv.encoding)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
        
    elif mode == 'compute_derived':
      
      with dask.config.set(pool=ThreadPool(4)): # I'm not sure if this works... or it is I/O limited
        
        start = time.time()
        
        # settings
        lexec = True
#         lexec = False
        load_chunks = None; lautoChunkLoad = True  # chunking input should not be necessary, if the source files are chunked properly
        chunks = None; lautoChunk = True # auto chunk output - this is necessary to maintain proper chunking!
        # N.B.: 'lautChunk' is necessary for *loading* data in chunks - otherwise it loads the whole array at once...
        #       !!! Chunking of size (12, 205, 197) requires ~13GB in order to compute T2 (three arrays total) !!!
#         chunks = (9, 59, 59); lautoChunk = False
#         load_chunks = dict(time=chunks[0], y=chunks[1], x=chunks[2])
#         derived_varlist = ['dask_test']; load_list = ['T2']
        derived_varlist = ['pet_pt']; load_list = ['T2']; clim_stns = ['UTM','Elora']
#         derived_varlist = ['pet_pts']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']; clim_stns = ['UTM','Elora']
#         derived_varlist = ['pet_hog']; load_list = ['Tmin', 'Tmax', 'T2']
#         derived_varlist = ['pet_har']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']
#         derived_varlist = ['pet_haa']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # Hargreaves with Allen correction
#         derived_varlist = ['pet_th']; load_list = ['T2', 'lat2D']
#         derived_varlist = ['T2']; load_list = ['Tmin', 'Tmax']
#         derived_varlist = ['liqwatflx']; load_list = ['precip','snow']
#         derived_varlist = ['T2','liqwatflx']; load_list = ['Tmin','Tmax', 'precip','snow']
        grid = 'son2'
        resolution = 'CA12'
        
        # optional slicing (time slicing completed below)
#         start_date = None; end_date = None # auto-detect available data
        start_date = '2011-01-01'; end_date = '2017-12-31' # inclusive
#         start_date = '2011-01-01'; end_date = '2011-04-01'
#         start_date = '2012-11-01'; end_date = '2013-01-31'
#         start_date = '2011-12-01'; end_date = '2012-03-01'
#         start_date = '2011-01-01'; end_date = '2012-12-31'
        # N.B.: it appears slicing is necessary to prevent some weird dtype error with time_stamp...
        
        # load datasets
        time_slice = (start_date,end_date) # slice time
        dataset = loadMergedForcing_Daily(varlist=load_list, grid=grid, resolution=resolution, bias_correction='rfbc', 
                                          resampling=None, time_slice=time_slice, lautoChunk=lautoChunkLoad, chunks=load_chunks)
        
        
        # load time coordinate
        tsvar = dataset[ts_name].load()
               
        print(dataset)
        
        # loop over variables
        for varname in derived_varlist:
            
            print("\n   ***   Processing Variable '{}'   ***   \n".format(varname))
            
            # compute values 
            if varname == 'dask_test':
                default_varatts = dict(name='dask_test', units='kg/m^2/s', long_name='Dask Test') 
                ref_var = dataset['T2']
                note = 'just to test some dask functionality'
                def test_fct(xds, ref_var=None,):
                    ''' dask test function '''
                    ref_var = xds[ref_var]
                    xvar = ref_var**2
                    assert ref_var.dims[0] == 'time', ref_var.dims
                    dt64 = xds['time'].data
                    if not np.issubdtype(dt64.dtype,np.datetime64): raise NotImplementedError()
                    J = 1 + ( ( dt64.astype('datetime64[D]') - dt64.astype('datetime64[Y]') ) / np.timedelta64(1,'D') )
                    xvar += J.reshape(ref_var.shape[:1]+(1,)*(ref_var.ndim-1)) * ref_var
                    xvar.attrs = {}
                    return xvar
                xvar = xr.map_blocks(test_fct, dataset, kwargs=dict(ref_var='T2'))
#                 print(xvar)
            elif varname == 'T2':
                from datasets.NRCan import varatts as ext_varatts
                default_varatts = ext_varatts[varname]; ref_var = dataset['Tmax']
                note = 'simple average of Tmin and Tmax'          
                xvar = dataset['Tmin'] + ref_var
                xvar /= 2                
            elif varname == 'pet_pt' or varname == 'pet_pts':
                default_varatts = varatts[varname]; ref_var = dataset['T2']
                # load radiation data from climate station
                from datasets.ClimateStations import loadClimStn_Daily
                if varname == 'pet_pts': 
                    radvar = 'DNSW'; lnetlw = True  # use only solar radiation and estimate net LW
                else: 
                    radvar = 'netrad'; lnetlw = False # use net radiation timeseries
                stn_ens = [loadClimStn_Daily(station=clim_stn, time_slice=time_slice, lload=True, lxarray=True) for clim_stn in clim_stns]
                # transfer 1D radiation timeseries to 3D dataset
                dataset.attrs['zs'] = np.mean([ds.attrs['zs'] for ds in stn_ens]) # also need approximate elevation - station elevation if fine...
                rad_data = np.nanmean(np.stack([ds[radvar].values for ds in stn_ens], axis=1), axis=1)
                rad_var = xr.DataArray(data=rad_data, coords=(stn_ens[0].time,), name=radvar, attrs=stn_ens[0][radvar].attrs)
                dataset[radvar] = rad_var
                # find missing data
                mia_var = rad_var[np.isnan(rad_var.data)]
                if len(mia_var) > 0:
                    nc_folder,nc_filename = getFolderFileName(varname=varname, dataset='MergedForcing', resolution=resolution, grid=grid, resampling=None,)
                    txt_filename = 'missing_timessteps '+nc_filename[:-3]+'.txt'
                    print("\n   ***   Missing Timesteps   ***   \n   (for Radiation Data)")
                    filepath = nc_folder+txt_filename
                    with open(filepath, mode='w') as fh:
                        for td in mia_var.time.values:
                            line = pd.to_datetime(td).strftime('%Y-%m-%d') 
                            fh.write(line+'\n')
                            print(line)
                        print("   ---   ")
                    print("Wrote missing timesteps to file:\n '{}'".format(filepath))
                # process timeseries
                from processing.newvars import computePotEvapPT
                note = 'PET based on the Priestley-Taylor method using average solar radiation from stations: '
                for stn in clim_stns: note += stn+', '
                kwargs = dict(alpha=1.26, lmeans=False, lrad=True, lA=False, lem=False, lnetlw=lnetlw, 
                              lgrdflx=False, lpmsl=False, lxarray=True,)      
                xvar = xr.map_blocks(computePotEvapPT, dataset, kwargs=kwargs)
                print(xvar)
            elif varname == 'pet_hog':
                from processing.newvars import computePotEvapHog
                default_varatts = varatts[varname]; ref_var = dataset['Tmax']
                note = 'PET based on the Hogg (1997) method using only Tmin and Tmax'
                kwargs = dict(lmeans=False, lq2=None, zs=150, lxarray=True)      
                xvar = xr.map_blocks(computePotEvapHog, dataset, kwargs=kwargs)
            elif varname == 'pet_har' or varname == 'pet_haa':                
                from processing.newvars import computePotEvapHar
                default_varatts = varatts[varname]; ref_var = dataset['Tmax']
                if varname == 'pet_haa':
                    note = 'PET based on the Hargreaves method with Allen correction using only Tmin and Tmax'; lAllen = True
                else: 
                    note = 'PET based on the Hargreaves method using only Tmin and Tmax'; lAllen = False
                kwargs = dict(lmeans=False, lat=None, lAllen=lAllen, l365=False, lxarray=True)      
                xvar = xr.map_blocks(computePotEvapHar, dataset, kwargs=kwargs)
            elif varname == 'pet_th':
                default_varatts = varatts[varname]; ref_var = dataset['T2']
                # load climatological temperature from NRCan
                cds = loadMergedForcing(varname='T2', name='climT2', dataset_name='NRCan', period=(1980,2010), resolution='NA12', 
                                        grid=grid, lxarray=True, lmonthly=False, lgeoref=False)
                clim_chunks = {dim:cnk for dim,cnk in zip(ref_var.dims,ref_var.encoding['chunksizes']) if dim in (ref_var.xlon,ref_var.ylat)}
                dataset['climT2'] = cds['T2'].chunk(chunks=clim_chunks).rename(time='month') # easier not to chunk time dim, since it is small
                # process timeseries
                from processing.newvars import computePotEvapTh
                note = 'PET based on the Thornthwaite method using only T2'
                kwargs = dict(climT2='climT2', lat=None, l365=False, p='center', lxarray=True)      
                xvar = xr.map_blocks(computePotEvapTh, dataset, kwargs=kwargs)
                print(xvar)
            elif varname == 'liqwatflx':
                default_varatts = varatts[varname]
                ref_var = dataset['precip']
                note = 'total precip (NRCan) - SWE changes from RFBC SnoDAS'
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                swe = dataset['snow'].fillna(0) # just pretend there is no snow...
                assert swe.attrs['units'] == 'kg/m^2', swe.attrs['units']
                xvar = ref_var - swe.differentiate('time', datetime_unit='s')
                xvar = xvar.clip(min=0,max=None) # remove negative values
            else:
                raise NotImplementedError(varname)
                
            # define/copy metadata
            xvar.attrs = ref_var.attrs.copy()
            xvar.rename(varname)
            for att in ('name','units','long_name',):
                if att in default_varatts: xvar.attrs[att] = default_varatts[att]
            if 'original_name' in xvar.attrs: del xvar.attrs['original_name'] # does not apply
            xvar.attrs['note'] = note
            # set chunking for operation
            if lautoChunk:                 
                chunks = autoChunk(xvar.shape)
            if chunks: 
                xvar = xvar.chunk(chunks=chunks)
            print('Chunks:',xvar.chunks)
                
            # create a dataset for export to new file
            ds_attrs = dataset.attrs.copy()
            if varname in default_dataset_index:
                orig_ds_name = default_dataset_index[varname]
                ds_attrs['name'] = orig_ds_name 
                resampling = xvar.attrs['resampling']
                ds_attrs['resampling'] = resampling
            else: 
                ds_attrs['name'] = 'MergedForcing'
                if 'resampling' in xvar.attrs: del xvar.attrs['resampling']
                if 'resampling' in ds_attrs: del ds_attrs['resampling']
                resampling = None
            proj4_str = dataset.attrs['proj4']
            nds = xr.Dataset({ts_name:tsvar, varname:xvar,}, attrs=ds_attrs)
            nds = addGeoReference(nds, proj4_string=proj4_str, )
            print('\n')
            print(nds)
            # file path based on variable parameters
            nc_folder,nc_filename = getFolderFileName(varname=varname, dataset=ds_attrs['name'], resolution=resolution, 
                                                      grid=grid, resampling=resampling,)
            nc_filepath = nc_folder + nc_filename
            print("\nExporting to new NetCDF-4 file:\n '{}'".format(nc_filepath))
            # write to NetCDF
            var_enc = dict(chunksizes=chunks, zlib=True, complevel=1, _FillValue=np.NaN, dtype=netcdf_dtype) # should be float
            task = nds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                          encoding={varname:var_enc,}, compute=False)
            if lexec:
                task.compute()
            else:
                print(var_enc)
                print(task)
                task.visualize(filename=nc_folder+'netcdf.svg')  # This file is never produced

        # print timing
        end =  time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
      