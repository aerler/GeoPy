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
import os.path as osp
from warnings import warn
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
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
               pet_hog = dict(name='pet_hog', units='kg/m^2/s', long_name='PET (Hogg 1997)'),
               pet_har = dict(name='pet_har', units='kg/m^2/s', long_name='PET (Hargeaves)'),
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
        

## functions to load NetCDF datasets (using xarray)


def loadMergedForcing_Daily(varname=None, varlist=None, dataset_index=None, dataset_args=None, time_slice=None, 
                            compat='override', join='inner', fill_value=None, **kwargs):
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
    const_list = dataset_varlists.pop('const',[])
    ## load datasets
    ds_list = []
    for dataset,varlist in dataset_varlists.items():
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
    xds = xr.merge(ds_list, compat=compat, join=join, fill_value=fill_value)
    for ds in ds_list[::-1]: xds.attrs.update(ds.attrs) # we want MergedForcing to have precedence
    xds.attrs['name'] = 'MergedForcing'; xds.attrs['title'] = 'Merged Forcing Daily Timeseries'
    if 'resampling' in xds.attrs: del xds.attrs['resampling'] # does not apply to a merged dataset
    ## add additional fields
    if const_list:
        # find horizontal coordinates
        dims = (xds.ylat,xds.xlon)
        for rv in xds.data_vars.values():
            if xds.ylat in rv.dims and xds.xlon in rv.dims: break
        chunks = {dim:chk for dim,chk in zip(rv.dims, rv.encoding['chunksizes']) if dim in dims}
        print(chunks)
        # add constant variables
        if 'lat2D' in const_list or 'lon2D' in const_list:
            # add geographic coordinate fields 
            grid = kwargs.get('grid',None)
            if grid is None: 
                raise NotImplementedError() # probably a rare case, since datasets need to be on a common grid
            else:
                from geodata.gdal import loadPickledGridDef
                griddef = loadPickledGridDef(grid='son2')
                # add local latitudes
                if 'lat2D' in const_list:
                    atts = dict(name='lat2d', long_name='Latitude', units='deg N')
                    xvar = xr.DataArray(data=griddef.lat2D, attrs=atts, dims=dims)
                    xvar = xvar.chunk(chunks=chunks)
                    xds['lat2D'] = xvar
                # add local longitudes
                if 'lon2D' in const_list:
                    atts = dict(name='lon2d', long_name='Longitude', units='deg E')
                    xvar = xr.DataArray(data=griddef.lon2D, attrs=atts, dims=dims)
                    xvar = xvar.chunk(chunks=chunks)
                    xds['lon2D'] = xvar 
        if 'zs' in const_list:
            raise NotImplementedError()
    # return merged dataset
    return xds
loadDailyTimeSeries = loadMergedForcing_Daily


def loadMergedForcing_All(varname=None, varlist=None, name=dataset_name, varatts=None, grid=None, mode=None, period=None,
                          lxarray=True, lmonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    # resolve folder and filename
    folder,filename = getFolderFileName(varname=None, dataset=dataset_name, grid=grid, resampling=None, resolution=None, 
                                        bias_correction=None, mode=mode, period=period, lcreateFolder=False)
    # remove some common arguments that have no meaning
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
        dataset = DatasetNetCDF(name=name, filelist=[folder+filename], varlist=varlist, multifile=False, **kwargs)
        # add GDAL to dataset
        default_geoargs = dict(griddef=grid, gridfolder=grid_folder)
        if geoargs: default_geoargs.update(geoargs)
        dataset = addGDALtoDataset(dataset, **default_geoargs)
    return dataset


def loadMergedForcing_TS(varname=None, varlist=None, name=dataset_name, varatts=None, grid=None,
                         lxarray=True, lmonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, varatts=varatts, grid=grid, mode='monthly', 
                                 period=None, lxarray=lxarray, lmonthly=lmonthly, lgeoref=lgeoref, geoargs=geoargs, **kwargs)

def loadMergedForcing(varname=None, varlist=None, name=dataset_name, varatts=None, grid=None, period=None,
                      lxarray=True, lmonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly normal merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, varatts=varatts, grid=grid, mode='clim', period=period,
                                 lxarray=lxarray, lmonthly=lmonthly, lgeoref=lgeoref, geoargs=geoargs, **kwargs)


## abuse for testing
if __name__ == '__main__':
  
  import dask, time, gc, shutil
  from multiprocessing.pool import ThreadPool
  
  print('xarray version: '+xr.__version__+'\n')
        

#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=4, diagnostics_port=18787)
#   client = Client(cluster)

#   from multiprocessing.pool import ThreadPool
#   dask.set_options(pool=ThreadPool(4))

  modes = []
#   modes += ['print_grid']
  modes += ['compute_derived']
  modes += ['load_Daily']
  modes += ['monthly_mean'          ]
  modes += ['load_TimeSeries'      ]
  modes += ['monthly_normal'        ]
  modes += ['load_Climatology'      ]
#   modes += ['compute_PET']  

  # some settings
  grid = None
#   grid = 'hd1' # small Quebec grid
  grid = 'son2' # high-res Southern Ontario
 
  ts_name = 'time_stamp'
  
 
  # loop over modes 
  for mode in modes:
    
    if mode == 'print_grid':
        
        from geodata.gdal import loadPickledGridDef
        griddef = loadPickledGridDef(grid='son2')
        print(griddef)
        print(griddef.lat2D)
        
    elif mode == 'load_Climatology':
       
        lxarray = True
        varname = 'liqwatflx'
        period = (2011,2018)
        xds = loadMergedForcing(varlist=None, grid=grid, lxarray=lxarray, period=period)
        print(xds)
        print('')
        xv = xds[varname]
        print(xv)
        if lxarray:
            print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
    
    elif mode == 'monthly_normal':
  
        # optional slicing (time slicing completed below)
#         start_date = '2011-01'; end_date = '2011-12'; varlist = ['liqwatflx', ts_name]
        start_date = '2011-01'; end_date = '2018-12'; varlist = None
  
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
        varname = 'pet_hog'
        xds = loadMergedForcing_TS(varlist=None, grid=grid, lxarray=lxarray)
        print(xds)
        print('')
        xv = xds[varname]
        print(xv)
        if lxarray:
            print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
    
    elif mode == 'monthly_mean':
        
        # settings
        load_chunks = None; lautoChunkLoad = False  # chunking input should not be necessary, if the source files are chunked properly
        chunks = None; lautoChunk = True # auto chunk output - this is necessary to maintain proper chunking!
#         time_slice = ('2011-01-01','2018-01-01')
        time_slice = None
        varlist = {dataset:None for dataset in dataset_list+[dataset_name]} # None means all...
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, bias_correction='rfbc', dataset_args=None, lskip=True, 
                                      lautoChunk=False, time_slice=time_slice)
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
        varlist = {dataset:None for dataset in dataset_list+[dataset_name]} # None means all...
        varlist['const'] = ['lat2D']
#         varlist = {'NRCan':None}
        dataset_args = dict(SnoDAS=dict(bias_correction='rfbc'))
#         time_slice = ('2011-01-01','2017-01-01')
        time_slice = None
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, bias_correction='rfbc', dataset_args=dataset_args, 
                                      time_slice=time_slice,)
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
#         for varname,xv in xds.variables.items(): 
#             if xv.ndim == 3: break
#         xv = xds[varname] # get DataArray instead of Variable object
# #         xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
#   #       xv = xv.loc['2011-01-01',:,:]
#         print(xv)
#         print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
        
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
#         derived_varlist = ['pet_hog']; load_list = ['Tmin', 'Tmax', 'T2']
        derived_varlist = ['pet_har']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']
#         derived_varlist = ['T2']; load_list = ['Tmin', 'Tmax']
#         derived_varlist = ['liqwatflx']; load_list = ['precip','snow']
#         derived_varlist = ['T2','liqwatflx']; load_list = ['Tmin','Tmax', 'precip','snow']
        grid = 'son2'
        resolution = 'CA12'
        
        # optional slicing (time slicing completed below)
#         start_date = None; end_date = None # auto-detect available data
        start_date = '2011-01-01'; end_date = '2018-01-01'
#         start_date = '2011-01-01'; end_date = '2011-04-01'
#         start_date = '2011-12-01'; end_date = '2012-03-01'
#         start_date = '2011-01-01'; end_date = '2013-01-01'
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
            elif varname == 'pet_hog':
                from processing.newvars import computePotEvapHog
                default_varatts = varatts[varname]; ref_var = dataset['Tmax']
                note = 'PET based on the Hogg (1997) method using only Tmin and Tmax'
                kwargs = dict(lmeans=False, lq2=None, zs=150, lxarray=True)      
                xvar = xr.map_blocks(computePotEvapHog, dataset, kwargs=kwargs)
            elif varname == 'pet_har':                
                from processing.newvars import computePotEvapHar
                default_varatts = varatts[varname]; ref_var = dataset['Tmax']
                note = 'PET based on the Hargreaves method using only Tmin and Tmax'
                kwargs = dict(lmeans=False, lat=None, lAllen=False, l365=False, lxarray=True)      
                xvar = xr.map_blocks(computePotEvapHar, dataset, kwargs=kwargs)
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
      
        
    elif mode == 'compute_PET':
       
        tic = time.time()
        
        # compute variable list
#         load_variables = dict(CaPA=['precip']); compute_variables = dict(CaPA=['precip'])
#         load_variables = dict(CaLDAS=['snowh','rho_snw']); compute_variables = dict(CaLDAS=['snow'])
#         load_variables = dict(CaLDAS=['snowh','rho_snw'], CaPA=['precip'])
#         compute_variables = dict(CaSPAr=['liqwatflx'])
        load_variables = dict(HRDPS=None) # all
        # HRDPS/PET variable lists
        lderived = True
        derived_valist = ['Rn', 'e_def', 'delta', 'u2', 'gamma', 'T2', 'pet_dgu', 'pet_wnd', 'pet_rad']
#         compute_variables = dict(HRDPS=['Rn',]); lderived = False
#         compute_variables = dict(HRDPS=['Rn', 'e_def', 'delta', 'u2', 'gamma', 'T2']) # 'RH', # first order variables
#         compute_variables = dict(HRDPS=['pet_dgu', 'pet_rad', 'pet_wnd',]) # second order variables
        # second order variables: denominator, radiation and wind terms, PET
#         compute_variables = dict(HRDPS=['pet_dgu',]) # denominator
#         compute_variables = dict(HRDPS=['pet_rad','pet_wnd']) # radiation and wind
#         derived_valist = ['Rn', 'e_def', 'delta', 'u2', 'gamma', 'T2',]
        compute_variables = dict(HRDPS=['pet']) # only PET
        
        drop_variables = 'default' # special keyword
        reference_dataset = next(iter(load_variables)) # just first dataset...
        
        # settings
        ts_name = 'time'
#         period = ('2019-08-19T00','2019-08-19T06')
        folder = folder_6hourly # CaSPAr/caspar_6hourly/
        
        # load multi-file dataset (no time slicing necessary)        
        datasets = dict()
        for dataset,varlist in load_variables.items():
            if lderived:
                datasets[dataset] = loadMergedForcing_Daily(grid=grid, varlist=derived_valist, 
                                                            dataset=dataset, lignore_missing=True)
        ref_ds = datasets[reference_dataset]
        print(ref_ds)
        tsvar = ref_ds[ts_name].load()
#         print(tsvar)
        
        print("\n\n   ***   Computing Derived Variables   ***   ")
        # loop over variables: compute and save to file
        for dataset,varlist in compute_variables.items():
            for varname in varlist:
              
                print('\n\n   ---   {} ({})   ---\n'.format(varname,dataset))
                note = 'derived variable'
                nvar = None; netcdf_dtype = np.dtype('<f4')
                # compute variable
                if dataset == 'CaSPAr':
                    # derived variables 
                    if varname == 'liqwatflx':
                        caldas = datasets['CaLDAS']; capa = datasets['CaPA']
                        ref_var = capa['precip']; ref_ds = capa
                        # check that the data is 6=hourly
                        dt = tsvar.diff(dim='time').values / np.timedelta64(1,'h')
                        assert dt.min() == dt.max() == 6, (dt.min(),dt.max())
                        note = 'total precipitation - SWE differences'
                        swe = caldas['rho_snw'] * caldas['snowh'] 
                        swe1 = xr.concat([swe[{'time':0}],swe], dim='time') # extend for differencing
                        dswe = swe1.diff(dim='time') # the extension should yield correct time indexing
                        dswe /= (6*3600.) # these are 6-hourly differences
                        nvar = capa['precip'].fillna(0) - dswe.fillna(0)
                        nvar = nvar.clip(min=0, max=None) # remove negative (unphysical)
                        del dswe, swe1, swe
                elif dataset == 'CaLDAS':
                    ref_ds = datasets[dataset]
                    # derived CaLDAS 
                    if varname == 'snow':
                        ref_var = ref_ds['snowh']
                        note = 'snow depth x density'
                        nvar = ref_ds['rho_snw'] * ref_ds['snowh']   
                elif dataset == 'HRDPS':
                    ref_ds = datasets[dataset]
                    # derived HRDPS
                    if varname == 'RH':
                        # relative humidity
                        ref_var = ref_ds['Q2']
                        # actual water vapor pressure (from mixing ratio)
                        e_vap = ref_ds['Q2'] * ref_ds['ps'] * ( 28.96 / 18.02 )
                        # saturation vapor pressure (for temperature T2; Magnus Formula)
                        e_sat = 610.8 * np.exp( 17.27 * (ref_ds['T2'] - 273.15) / (ref_ds['T2'] - 35.85) )
                        note = 'e_vap / e_sat (using Magnus Formula)'
                        nvar = e_vap / e_sat
                        del e_sat, e_vap
                    # first order PET variables
                    elif varname == 'Rn':
                        from utils.constants import sig
                        # net radiation
                        ref_var = ref_ds['DNSW']
                        note = '0.23*DNSW + DNLW - 0.93*s*T2**4'
                        nvar = (1-0.23)*ref_ds['DNSW'] + ref_ds['DNLW']- 0.93*sig*ref_ds['T2']**4
                        # N.B.: Albedo 0.23 and emissivity 0.93 are approximate average values...
                    elif varname == 'u2':
                        # wind speed
                        ref_var = ref_ds['U2']
                        note = 'SQRT(U2**2 + V2**2)'
                        nvar = np.sqrt(ref_ds['U2']**2 + ref_ds['V2']**2) # will still be delayed
                    elif varname == 'e_def':
                        # saturation deficit
                        ref_var = ref_ds['Q2']
                        # actual water vapor pressure (from mixing ratio)
                        e_vap = ref_ds['Q2'] * ref_ds['ps'] * ( 28.96 / 18.02 )
                        # saturation vapor pressure (for temperature T2; Magnus Formula)
                        e_sat = 610.8 * np.exp( 17.27 * (ref_ds['T2'] - 273.15) / (ref_ds['T2'] - 35.85) )
                        note = 'e_sat - e_vap (using Magnus Formula)'
                        nvar = e_sat - e_vap
                        del e_sat, e_vap
                    # PET helper variables (still first order)
                    elif varname == 'delta':
                        # slope of saturation vapor pressure (w.r.t. temperature T2; Magnus Formula)
                        ref_var = ref_ds['T2']
                        note = 'd(e_sat)/dT2 (using Magnus Formula)'
                        nvar = 4098 * ( 610.8 * np.exp( 17.27 * (ref_ds['T2'] - 273.15) / (ref_ds['T2'] - 35.85) ) ) / (ref_ds['T2'] - 35.85)**2
                    elif varname == 'gamma':
                        # psychometric constant
                        ref_var = ref_ds['ps']
                        note = '665.e-6 * ps'
                        nvar = 665.e-6 * ref_ds['ps']   
                    # second order PET variables (only depend on first order variables and T2)
                    elif varname == 'pet_dgu':
                        # common denominator for PET calculation
                        ref_var = ref_ds['delta']
                        note = '( D + g * (1 + 0.34 * u2) ) * 86400'
                        nvar = ( ref_ds['delta'] + ref_ds['gamma'] * (1 + 0.34 * ref_ds['u2']) ) * 86400
                    elif varname == 'pet_rad':
                        # radiation term for PET calculation
                        ref_var = ref_ds['Rn']
                        note = '0.0352512 * D * Rn / Dgu'
                        nvar = 0.0352512 * ref_ds['delta'] * ref_ds['Rn'] / ref_ds['pet_dgu']
                    elif varname == 'pet_wnd':
                        # wind/vapor deficit term for PET calculation
                        ref_var = ref_ds['u2']
                        note = 'g * u2 * (es - ea) * 0.9 / T / Dgu'
                        nvar = ref_ds['gamma'] * ref_ds['u2'] * ref_ds['e_def'] * 0.9 / ref_ds['T2'] / ref_ds['pet_dgu']
                    elif varname == 'pet':
                        if 'pet_rad' in ref_ds and 'pet_wnd' in ref_ds:
                            # full PET from individual terms
                            ref_var = ref_ds['pet_rad']
                            note = 'Penman-Monteith (pet_rad + pet_wnd)'
                            nvar = ref_ds['pet_rad'] + ref_ds['pet_wnd']
                        else:
                            # or PET from original derived variables (no terms)
                            ref_var = ref_ds['Rn']
                            note = 'Penman-Monteith from derived variables'
                            nvar = ( 0.0352512 * ref_ds['delta'] * ref_ds['Rn'] + ( ref_ds['gamma'] * ref_ds['u2'] * ref_ds['e_def'] * 0.9 / ref_ds['T2'] ) ) / ( ref_ds['delta'] + ref_ds['gamma'] * (1 + 0.34 * ref_ds['u2']) ) / 86400
                    
                        
                
                # fallback is to copy
                if nvar is None: 
                    if dataset in datasets:
                        # generic operation
                        ref_ds = datasets[dataset]
                        if varname in ref_ds:
                            # generic copy
                            ref_var = ref_ds[varname]
                            nvar = ref_ds[varname].copy()
                        else:
                            raise NotImplementedError("Variable '{}' not found in dataset '{}'".fomat(varname,dataset))
                    else:
                        raise NotImplementedError("No method to compute variable '{}' (dataset '{}')".format(varname,dataset))
                
                nvar = nvar.astype(netcdf_dtype)
                # assign attributes
                nvar.rename(varname)
                nvar.attrs = ref_var.attrs.copy()
                for srcname,varatts in dataset_attributes[dataset].varatts.items():
                    if varatts['name'] == varname: break # use these varatts
                for att in ('name','units','long_name',):
                    nvar.attrs[att] = varatts[att]
                nvar.attrs['note'] = note
                #nvar.chunk(chunks=chunk_settings)
                
                print(nvar)
                
                # save to file            
                nc_filename = filename_6hourly.format(DS=dataset,VAR=varname,GRD=grid)
                nds = xr.Dataset({ts_name:tsvar, varname:nvar,}, attrs=ref_ds.attrs.copy()) # new dataset
                # write to NetCDF
                var_enc = dict(zlib=True, complevel=1, _FillValue=np.NaN,)
                # N.B.: may add chunking for larger grids
                nds.to_netcdf(folder+nc_filename, mode='w', format='NETCDF4', unlimited_dims=['time'], 
                              engine='netcdf4', encoding={varname:var_enc,}, compute=True)
                del nvar, nds, ref_var
            
        # clean up
        gc.collect()            
        
        toc = time.time()
        print("\n\nOverall Timing:",toc-tic)
        
  
