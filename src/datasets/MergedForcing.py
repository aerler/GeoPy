'''
Created on May 25, 2020

A module to merge different high-resolution datesets and load the resulting merged dataset; mainly for hydrological modeling;
a major secondary purpose of this module is also, to keep xarray dependencies out of other modules (in particular, NRCan)

@author: Andre R. Erler, GPL v3
'''



# external imports
import pandas as pd
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
from datasets.misc import getFolderFileName, addConstantFields, loadXRDataset
# for georeferencing
from geospatial.xarray_tools import addGeoReference, updateVariableAttrs, computeNormals, getCommonChunks, saveXArray

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
               liqwatflx_sno = dict(name='liqwatflx_sno', units='kg/m^2/s', long_name='LWF (SnoDAS)'),
               liqwatflx_ne5 = dict(name='liqwatflx_ne5', units='kg/m^2/s', long_name='LWF (ERA5-Land)'),
               pet = dict(name='pet', units='kg/m^2/s', long_name='Potential Evapotranspiration'),
               pet_pt  = dict(name='pet_pt',  units='kg/m^2/s', long_name='PET (Priestley-Taylor)'),
               pet_pts = dict(name='pet_pts', units='kg/m^2/s', long_name='PET (Priestley-Taylor, approx. LW)'),
               pet_hog = dict(name='pet_hog', units='kg/m^2/s', long_name='PET (Hogg 1997)'),
               pet_har = dict(name='pet_har', units='kg/m^2/s', long_name='PET (Hargeaves)'),
               pet_haa = dict(name='pet_haa', units='kg/m^2/s', long_name='PET (Hargeaves-Allen)'),
               pet_th  = dict(name='pet_th',  units='kg/m^2/s', long_name='PET (Thornthwaite)'),
               )
varlist = varatts.keys()
ignore_list = []

# settings for NetCDF-4 files
avgfolder = root_folder + 'mergedavg/' 
avgfile   = '{DS:s}_{GRD:s}_clim_{PRD:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = '{DS:s}_{GRD:s}_monthly.nc' # extend with biascorrection, variable and grid type
# daily data
daily_folder    = root_folder + 'merged_daily/' 
netcdf_filename = 'merged_{VAR:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float

# list of available datasets/collections
DSNT = namedtuple(typename='Dataset', field_names=['name','interval','start_date','end_date',])
dataset_attributes = dict(SnoDAS  = DSNT(name='SnoDAS',interval='1D', start_date='2011-01-01', end_date=None,  ),                          
                          NRCan   = DSNT(name='NRCan',  interval='1D', start_date='1950-01-01', end_date='2017-12-31', ),
                          ERA5L   = DSNT(name='ERA5L',  interval='1D', start_date='1981-01-01', end_date=None, ), 
                          #CaSPAr  = DSNT(name='CaSPAr',  interval='6H', start_date='2017-09-11T12', end_date='2019-12-30T12', ),
                          )
dataset_list = list(dataset_attributes.keys())
# N.B.: the effective start date for CaPA and all the rest is '2017-09-11T12'
default_dataset_index = dict(precip='NRCan', precip_adj='NRCan', Tmin='NRCan', Tmax='NRCan', T2='NRCan', 
                             pet_hog='NRCan', pet_har='NRCan', pet_haa='NRCan', pet_th='NRCan',
                             snow='SnoDAS', dswe='SnoDAS',
                             lat2D='const', lon2D='const', zs='const')
dataset_varlist = {dataset:None for dataset in dataset_list+[dataset_name, 'const']} # None means all...
default_varlist = [varname for varname in varlist if varname not in default_dataset_index]


## functions to load NetCDF datasets (using xarray)


def loadMergedForcing_Daily(varname=None, varlist=None, dataset=None, dataset_index=None, dataset_args=None, dataset_name=dataset_name, 
                            time_slice=None, compat='override', join='inner', fill_value=np.NaN, ldebug=False, **kwargs):
    ''' function to load and merge data from different high-resolution datasets (e.g. SnoDAS or NRCan) using xarray;
        typical dataset-agnostic arguments: grid=str, lgeoref=True, geoargs=dict, chunks=dict, lautoChunk=False, 
        typical dataset-specific arguments: folder=str, resampling=str, resolution=str, bias_correction=str '''
    global_ds_atts_keys = ('resolution','bias_correction','resampling')
    # figure out varlist
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname:
        varlist = [varname] # load a single variable
    elif varlist is None:
        if dataset is None: varlist = list(varatts.keys())
        else: varlist = {dataset:None, 'const':None} # load default list for dataset
    if dataset_args is None: dataset_args = dict() # avoid errors
    # assemble dataset list and arguments
    if isinstance(varlist,dict):
        dataset_varlists = varlist
    elif isinstance(varlist,(list,tuple)):
        if dataset:
            dataset_varlists = {dataset:varlist}
        else:
            if dataset_index is None: dataset_index = default_dataset_index.copy()
            dataset_varlists = dict()
            for varname in varlist:
                ds_name = dataset_index.get(varname,dataset_name) # default is native (global variable)
                if ds_name not in dataset_varlists: dataset_varlists[ds_name] = [varname] 
                else: dataset_varlists[ds_name].append(varname)
    else:
        raise TypeError(varlist)
    const_list = dataset_varlists.pop('const', [])
    ## load datasets
    ds_list = []
    global_ds_atts_dict = dict()
    #print(dataset_name)
    for dataset,varlist in dataset_varlists.items():
        if ldebug: print("Loading", dataset, '\n', varlist, '\n')
        # prepare kwargs
        ds_args = kwargs.copy(); 
        if dataset in dataset_args: ds_args.update(dataset_args[dataset])
        if dataset.lower() == dataset_name.lower():
            # native MergedForcing
            ds_args.update(dataset=dataset)
            argslist = ['grid', ] # specific arguments for merged dataset variables
            if varlist is None: varlist = default_varlist
            loadFunction = loadXRDataset
        else:
            # daily data from other datasets
            ds_mod = import_module('datasets.{0:s}'.format(dataset)) # import dataset module
            loadFunction = ds_mod.loadDailyTimeSeries
            argslist = inspect.getfullargspec(loadFunction); argslist = argslist.args # list of actual arguments
        # remove some args that don't apply
        for key in global_ds_atts_keys: # list of dataset-specific arguments that have to be controlled
            if key not in argslist and key in ds_args: del ds_args[key]
        # load time series and and apply some formatting to vars
        ds = loadFunction(varlist=varlist, compat=compat, join=join, fill_value=fill_value, **ds_args)
        # add some dataset attributes to variables, since we will be merging datasets
        for var in ds.variables.values(): var.attrs['dataset_name'] = dataset_name
        for key in global_ds_atts_keys: # list of dataset-specific arguments that have to be controlled
            if key in ds.attrs: value = ds.attrs[key]
            elif key in ds_args: value = ds_args[key]
            else: value = None
            if value is not None:
                for var in ds.data_vars.values(): var.attrs[key] = value
                if key not in global_ds_atts_dict: 
                    global_ds_atts_dict[key] = value
                elif global_ds_atts_dict[key] != value:
                    global_ds_atts_dict[key] = None # only keep if all equal
        if time_slice: ds = ds.loc[{'time':slice(*time_slice),}] # slice time
        ds_list.append(ds)
    # merge datasets and attributed
    if ldebug: print("Merging Datasets:", compat, join, '\n')
    xds = xr.merge(ds_list, compat=compat, join=join, fill_value=fill_value)
    for ds in ds_list[::-1]: xds.attrs.update(ds.attrs) # we want MergedForcing to have precedence
    xds.attrs['name'] = 'MergedForcing'; xds.attrs['title'] = 'Merged Forcing Daily Timeseries'
    for key,value in global_ds_atts_dict.items():
        if value is not None: xds.attrs[key] = value
    ## add additional fields
    if ldebug: print("Adding Constants:", const_list, '\n',)
    xds = addConstantFields(xds, const_list=const_list, grid=kwargs.get('grid',None))
    # return merged dataset
    return xds
loadDailyTimeSeries = loadMergedForcing_Daily


def loadMergedForcing_All(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, 
                          shape=None, station=None, mode=None, aggregation=None, period=None, lxarray=True, lgeoref=False, 
                          geoargs=None, ltoMonthly=None, lfromMonthly=False, dataset_args=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    if isinstance(varlist, dict):
        raise NotImplementedError("Loading variables from multiple files or datasets is currently not implemented.")
    # resolve folder and filename
    arg_list = ('resampling', 'resolution', 'bias_correction', 'filetype')
    file_args = {key:kwargs.pop(key,None) for key in arg_list}
    if dataset_args is not None and dataset_name in dataset_args: 
        dataset_args = dataset_args[dataset_name]
        for arg in arg_list:
            if arg in dataset_args: file_args[arg] = dataset_args[arg]
    folder,filename = getFolderFileName(varname=None, dataset=dataset_name, grid=grid, mode=mode, aggregation=aggregation, period=period, 
                                        lcreateFolder=False, shape=shape, station=station, dataset_index=default_dataset_index, **file_args)
    #print(folder,filename)
    # remove some common arguments that have no meaning
    if name is None: name = dataset_name
    for key in ('resolution','bias_correction'):
        if key in kwargs: del kwargs[key]
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname: varlist = [varname]
    if lxarray: 
        ## load as xarray dataset
        # set options
        if lfromMonthly: kwargs['decode_times'] = False
        # load  dataset
        xds = xr.open_dataset(folder+filename, **kwargs)
        # update varatts and prune
        xds = updateVariableAttrs(xds, varatts=varatts, varmap=None, varlist=varlist)
        # some attributes
        xds.attrs['name'] = name
        # load time stamps (like coordinate variables)
        if 'time_stamp' in xds: xds['time_stamp'].load()
        # fix time axis (deprecated - should not be necessary anymore)
        if lfromMonthly and xds['time'].attrs['units'].lower() == 'month':
            warn("'ldaily=True' should only be used to convert simple monthly indices into 'datetime64' coordinates.")
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
        # convert daily time axis values to monthly
        time =dataset.time
        if ltoMonthly and 'units' in time.ncvar.ncattrs():
            tunits = time.ncvar.getncattr('units')
            if tunits.startswith('days since'):
                from datetime import datetime
                from dateutil import relativedelta
                from geodata.base import Axis
                print('Rewriting xarray time axis into GeoPy format')
                #print(tunits[11:21])
                startdate = datetime.strptime(tunits[11:21], '%Y-%m-%d'); 
                date1979 = datetime.strptime('1979-01-01', '%Y-%m-%d')
                r = relativedelta.relativedelta(startdate, date1979)
                #print(r.years*12+r.months)
                coord = r.years*12+r.months + np.arange(len(time))
                atts = time.atts.copy()
                atts['long_name'] = 'month since 1979-01'
                atts['units'] = 'month'
                new_time = Axis(coord=coord, atts=atts)
                dataset.replaceAxis(new_time, asNC=False)

        # fix axes units:
        for ax in ('x','y','lat','lon'):
            if ax in dataset.axes: dataset.axes[ax].atts.update(axes_varatts[ax])
        # add GDAL to dataset
        default_geoargs = dict(griddef=grid, gridfolder=grid_folder)
        if geoargs: default_geoargs.update(geoargs)
        dataset = addGDALtoDataset(dataset, **default_geoargs)
    return dataset


def loadMergedForcing_TS(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, mode='avg', 
                         aggregation='monthly', lxarray=False, ltoMonthly=True, lfromMonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset_name=dataset_name, varatts=varatts, grid=grid, 
                                 mode=mode, aggregation=aggregation, period=None, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs, 
                                 ltoMonthly=ltoMonthly, lfromMonthly=lfromMonthly, shape=None, station=None, **kwargs)

def loadMergedForcing(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, period=None, mode='avg', 
                      aggregation='clim', lxarray=False, ltoMonthly=True, lfromMonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly normal merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset_name=dataset_name, varatts=varatts, grid=grid, 
                                 mode=mode, aggregation=aggregation, period=period, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs, 
                                 ltoMonthly=ltoMonthly, lfromMonthly=lfromMonthly, shape=None, station=None, **kwargs)

def loadMergedForcing_ShpTS(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, shape=None, mode='avg', 
                            aggregation='monthly', lxarray=False, ltoMonthly=True, lfromMonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load monthly transient merged forcing data averaged over shapes '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset_name=dataset_name, varatts=varatts, grid=grid,
                                 mode=mode, aggregation=aggregation, period=None, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs, 
                                 ltoMonthly=ltoMonthly, lfromMonthly=lfromMonthly, shape=shape, station=None, **kwargs)

def loadMergedForcing_Shp(varname=None, varlist=None, name=None, dataset_name=dataset_name, varatts=None, grid=None, period=None, mode='avg', 
                          aggregation='clim', shape=None, lxarray=False, ltoMonthly=True, lfromMonthly=False, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load monthly normal merged forcing data averaged over shapes '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset_name=dataset_name, varatts=varatts, grid=grid, 
                                 mode=mode, aggregation=aggregation, period=period, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs, 
                                 ltoMonthly=ltoMonthly, lfromMonthly=lfromMonthly, shape=shape, station=None, **kwargs)

## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = netcdf_filename # filename pattern: variable name (daily)
ts_file_pattern   = tsfile # filename pattern: variable name and grid
clim_file_pattern = avgfile # filename pattern: grid and period
data_folder       = avgfolder # folder for user data
grid_def  = {'':None} # no special name, since there is only one...
LTM_grids = [] # grids that have long-term mean data 
TS_grids  = ['','CA12','SON60'] # grids that have time-series data
grid_res  = {'':0.01,'CA12':1./12.,'SON60':1./60.}
default_grid = None
# functions to access specific datasets
loadLongTermMean       = None # climatology provided by publisher
loadDailyTimeSeries    = loadMergedForcing_Daily # daily time-series data
# monthly time-series data for batch processing
def loadTimeSeries(lxarray=False, **kwargs): return loadMergedForcing_TS(lxarray=lxarray, **kwargs)
def loadClimatology(lxarray=False, **kwargs): return loadMergedForcing(lxarray=lxarray, **kwargs)
loadStationClimatology = None # climatologies without associated grid (e.g. stations) 
loadStationTimeSeries  = None # time-series without associated grid (e.g. stations)
loadShapeClimatology   = loadMergedForcing_Shp # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries    = loadMergedForcing_ShpTS # time-series without associated grid (e.g. provinces or basins)


## abuse for testing
if __name__ == '__main__':
  
  import time, os, gc
#   from multiprocessing.pool import ThreadPool
  
  print('xarray version: '+xr.__version__+'\n')
        
  #gc.set_debug(gc.DEBUG_LEAK)

  import dask
  dask.config.set(**{'array.slicing.split_large_chunks': False}) # suppress warnings about large chunks
  dask.config.set(temporary_directory='G:/Data/TMP/')
  
#   # turn caching off
#   from dask.cache import Cache
#   cache = Cache(0)
#   cache.register() 

#   dask.config.set(scheduler='single-threaded')
#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=1, dashboard_address='rz025:18787',
#                          threads_per_worker=1, memory_limit='2GB')
#   client = Client(cluster)
 
  ts_name = 'time_stamp'
  process_dataset = dataset_name # we can't overwrite dataset_name without causing errors...
  dataset_args = None; filetype = None

  work_loads = []
#   work_loads += ['load_Point_Climatology']
#   work_loads += ['load_Point_Timeseries']  
#   work_loads += ['print_grid']
  # work_loads += ['compute_derived']
  # work_loads += ['load_Daily']
  # work_loads += ['monthly_mean'          ]
  # work_loads += ['load_TimeSeries'      ]
  work_loads += ['monthly_normal'        ]
  work_loads += ['load_Climatology'      ]

  # some settings
  # process_dataset = 'MergedForcing'
  process_dataset = 'NRCan'
#   resolution = 'CA12'
  resolution = 'SON60'
  
#   process_dataset = 'ERA5'; filetype = 'ERA5L'
#   dataset_args = dict(ERA5=dict(filetype='ERA5L', lfliplat=True))
# #   resolution = 'AU10'
#   resolution = 'NA10'
  
  grid = None; bias_correction = None
  grid = 'snw2'
#   grid = 'hd1' # small Quebec grid
#   grid = 'son2'; bias_correction = 'rfbc' # high-res Southern Ontario
#   grid = 'on1'
  pntset = 'sonshp'
  
 
  # loop over modes 
  for mode in work_loads:
    
    if mode == 'print_grid':
        
        from geodata.gdal import loadPickledGridDef
        griddef = loadPickledGridDef(grid='son2')
        print(griddef)
        print(griddef.lat2D)
        
    elif mode == 'load_Point_Climatology':
      
        # load point climatology
        print('')
        if pntset in ('sonshp'): dataset = loadMergedForcing_Shp(shape=pntset, grid=grid, period=(2011,2018))
        else: raise NotImplementedError(pntset)
        print(dataset)
        print('')
        print((dataset.time))
        print((dataset.time.coord))
      
    elif mode == 'load_Point_Timeseries':      
      
        # load point climatology
        print('')
        if pntset in ('sonshp'): dataset = loadMergedForcing_ShpTS(shape=pntset, grid=grid, )
        else: raise NotImplementedError(pntset)
        print(dataset)
        print('')
        print((dataset.time))
        print((dataset.time.coord))

    elif mode == 'load_Climatology':
       
        lxarray = False
#         process_dataset = 'NRCan'; period = (1997,2018); kwargs = dict()
        kwargs = dict()
        period = prdstr # from monthly_normal
#         period = (2011,2018)
#         period = (1985,2015)
        # period = (1997,2018)
#         period = (1980,2010); kwargs = dict(dataset_name='NRCan', resolution='NA12', varlist=[varname]) # load regular NRCan normals
        
        xds = loadMergedForcing(grid=grid, lxarray=lxarray, period=period, dataset_args=dataset_args, mode='daily',
                                dataset_name=process_dataset, resolution=resolution, **kwargs)
        print(xds)
        print('')
        if lxarray:
            xv = list(xds.data_vars.values())[0]
            print(xv)
            if lxarray:
                print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
    
    elif mode == 'monthly_normal':
  
        # optional slicing (time slicing completed below)
#         start_date = '2011-01'; end_date = '2011-12'; varlist = None
#         start_date = '2011-01'; end_date = '2017-12'; varlist = None # date ranges are inclusive
#         start_date = '1985-01'; end_date = '2014-12'; varlist = None # date ranges are inclusive
#         start_date = '1981-01'; end_date = '2010-12'; varlist = None # date ranges are inclusive
        start_date = '2003-01'; end_date = '2017-12'; varlist = None # date ranges are inclusive
        # start_date = None; end_date = None; varlist = None
#         varlist = ['T2','time_stamp']
        process_dataset = 'NRCan'; resolution = 'SON60'

        # just ERA5-land
#         varlist = {'ERA5':['precip','liqwatflx','pet_era5','snow','dswe'], 'const':None}
#         resolution = 'AU10'; filetype = 'ERA5L'
#         dataset_args = dict(ERA5=dict(filetype='ERA5L'))

        
        # start operation
        start = time.time()
            
        # load variables object (not data!)
        xds   = loadMergedForcing_TS(varlist=varlist, grid=grid, dataset_name=process_dataset, resolution=resolution, mode='daily', 
                                     dataset_args=dataset_args, lxarray=True) # need Dask!
        xds   = xds.loc[{'time':slice(start_date,end_date),}] # slice entire dataset
        print(xds)
        chunks = getCommonChunks(xds, method='min') # used later
        
        # construct period string
        print('\n')
        cds = computeNormals(xds, aggregation='month', time_stamp=ts_name)
        print(cds)
        print('\n')
        prdstr = cds.attrs['period']
        print(prdstr)            

        # copy encoding      
        encoding = dict()
        print('Original Chunks:',chunks)
        # save resampled dataset
        nc_folder, nc_filename = getFolderFileName(dataset=process_dataset, resolution=resolution, grid=grid, period=prdstr, mode='daily', 
                                             aggregation='clim', dataset_index=default_dataset_index, filetype=filetype)
        # save to NetCDF file, with all bells and whistles
        saveXArray(cds, filename=nc_filename, folder=nc_folder, mode='write', varlist=None, chunks=chunks, encoding=None, 
                   time_agg='month', laddTime=True, time_dim='time', ltmpfile=True, lcompute=True, lprogress=True, lfeedback=True)
        
#         for varname,cvar in cds.data_vars.items():
#             cks = tuple(1 if dim == 'time' else chunks[dim] for dim in cvar.dims)
#             encoding[varname] = dict(chunksizes=cks, zlib=True, complevel=1, _FillValue=np.NaN,) # should be float            
#         nc_filepath = folder + filename
#         tmp_filepath = nc_filepath + '.tmp' # use temporary file during creation
#         # write to NetCDF
#         cds.to_netcdf(tmp_filepath, mode='w', format='NETCDF4', unlimited_dims=[], engine='netcdf4',
#                       encoding=encoding, compute=True)
#         
#         # add name and length of month (doesn't work properly with xarray)
#         ds = nc.Dataset(tmp_filepath, mode='a')
#         ds = addNameLengthMonth(ds, time_dim='time')
#         ds.close()
#         # rename NetCDF file
#         if os.path.exists(nc_filepath): os.remove(nc_filepath)
#         os.rename(tmp_filepath, nc_filepath)
#         
        # print timing
        end = time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
  
  
    elif mode == 'load_TimeSeries':
       
        lxarray = True
        varname = None
        xds = loadMergedForcing_TS(varlist=None, dataset_name=process_dataset, dataset_args=dataset_args, mode='daily', 
                                   resolution=resolution, grid=grid, lxarray=lxarray)
        print(xds)
        print('')
        if lxarray:
            if varname: xv = xds[varname]
            else:
                for xv in xds.data_vars.values():
                    if xv.ndim > 2: break
            print(xv)
            print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
            print(xv.encoding)
    
    elif mode == 'monthly_mean':
        
        # settings
        lexec = True
        time_slice = None
#         time_slice = ('2011-01-01','2011-12-31') # inclusive
        varlist = None
        dataset_args = None; filetype = None      
        
        # Merged Forcing
        # process_dataset = 'MergedForcing'; varlist = {'MergedForcing':None, 'const':None, }
        # grid = 'son2'; resolution = 'SON60'
        # grid = 'snw2'
        
        # process just NRCan dataset
        process_dataset = 'NRCan'; varlist = {'NRCan':None, 'const':None, 'MergedForcing':['liqwatflx_ne5','liqwatflx']}
        # resolution = 'CA12'; #chunks = dict(time=8, lat=64, lon=63)
        # grid = 'son2'; resolution = 'SON60'
        grid = 'snw2'; resolution = 'SON60'

#         # just ERA5-land
#         process_dataset = 'ERA5'
#         varlist = {'ERA5':['precip','liqwatflx','pet_era5','snow','dswe'], 'const':None}
#         resolution = 'NA10'; filetype = 'ERA5L'
# #         dataset_args = dict(ERA5=dict(filetype=filetype, lfliplat=True))
#         grid = 'son2'; dataset_args = dict(ERA5=dict(filetype=filetype, lfliplat=False))
#         if resolution == 'NA10': chunks = dict(time=8, latitude=61, longitude=62)
#         elif resolution == 'AU10': chunks = dict(time=8, latitude=59, longitude=62)
        
        # auto chunk, but use multiple of chunkf for better workloads (~ 100 MB per chunk)
        multi_chunks = {dim:4 for dim in ('lat','lon','latitude','longitude','x','y')}
        if grid == 'son2': multi_chunks['time'] = 82 # closer to 2 years... time chunk is 9
        else: multi_chunks['time'] = 92 # close to 2 years with time chunk 8 (default)
        
#         xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, bias_correction='rfbc', dataset_args=None, lskip=True,
#                                       resolution=resolution, lautoChunk=lautoChunkLoad, time_slice=time_slice, ldebug=False)
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, dataset_args=dataset_args, join='outer', fill_value=np.NaN, 
                                      bias_correction=bias_correction, resolution=resolution,
                                      time_slice=time_slice, multi_chunks=multi_chunks, chunks=True, lskip=True)
        #xds = xds.unify_chunks()
        chunks = getCommonChunks(xds, method='min') # used later
        print(xds)
        print('')
        # start operation
        start = time.time()
        
        # aggregate month
        rds = xds.resample(time='MS',skipna=True,).mean()
        print(rds)
        print('')
        # collect garbage
        del xds
        gc.collect()

        # define destination file
        nc_folder, nc_filename = getFolderFileName(dataset=process_dataset, grid=grid, resolution=resolution, filetype=filetype,
                                                   bias_correction=bias_correction, mode='daily',aggregation='monthly', 
                                                   dataset_index=default_dataset_index, data_root=None)
        print('Original Chunks:',chunks)
        # save to NetCDF file, with all bells and whistles
        saveXArray(rds, filename=nc_filename, folder=nc_folder, mode='write', varlist=None, chunks=chunks, encoding=None, 
                   time_agg='month', laddTime=True, time_dim='time', ltmpfile=True, lcompute=True, lprogress=True, lfeedback=True)
        
        # print timing
        end = time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
  
                             
    elif mode == 'load_Daily':
       
#         resolution = 'SON60'; grid = None
          
        multi_chunks = 'time'
        multi_chunks = None
  #       varlist = netcdf_varlist
#         varlist = ['precip','snow','liqwatflx']
#         varlist = {dataset:None for dataset in dataset_list+[dataset_name, 'const']} # None means all...
        varlist = {'NRCan':None, 'const':None, 'MergedForcing':['liqwatflx_ne5']}
#         varlist = {'ERA5':['precip'], 'const':None}; dataset_args = dict(ERA5=dict(filetype='ERA5L', lfliplat=True))
#         grid = None; resolution = 'AU10'
#         chunks = dict(time=8, latitude=59, longitude=62)
#         varlist = dataset_varlist
#         varlist = {dataset:None for dataset in ['NRCan',dataset_name, 'const']}
#         varlist = default_varlist
#         dataset_args = dict(SnoDAS=dict(bias_correction='rfbc'))
#         varlist = {'NRCan':['Tmin','Tmax'], 'const':['lat2D']}
#         dataset_args = None
#         time_slice = ('2011-01-01','2017-01-01')
        time_slice = None
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, dataset_args=dataset_args, join='outer', fill_value=np.NaN,
                                      bias_correction=bias_correction, resolution=resolution, chunks=True,
                                      time_slice=time_slice, multi_chunks=multi_chunks, lskip=True)
        #xds.unify_chunks()
        print(xds)
        print('')
        print(xds.attrs)
        print('')
        # check lat2D array for correct order
        if 'lat2D' in xds.variables:
            print('')
            print(xds['lat2D'])
            print('Lat2D[0,:]:')
            print(xds['lat2D'].values[0,0:5])
            print('Lat2D[:,0]:')
            print(xds['lat2D'].values[0:5,0])
            if 'is_projected' not in xds.attrs or xds.attrs['is_projected'] == 0: # projected coords deviate from this
                assert np.diff(xds['lat2D'].values[0,:]).sum() == 0, xds['lat2D'].values[0,0:5]
                assert np.diff(xds['lat2D'].values[:,0]).sum() > 0, xds['lat2D'].values[0:5,0]
#         # check for zeros in temperature field... (Kelvin!)
#         for varname in ('T2','Tmin','Tmax'):
#             if varname in xds:
#                 xvar = xds[varname]
#                 zeros = xvar < 100
#                 print(varname,zeros.data.sum())
        print('')            
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
#         xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(xv.encoding)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
        
    elif mode == 'compute_derived':
      
        #with dask.config.set(pool=ThreadPool(4)): # I'm not sure if this works... or it is I/O limited
        
        start = time.time()
        
        # settings
        lexec = True
        resolution = None; bias_correction = None; dataset_args = dict(ERA5=dict(filetype='ERA5L'))
#         lexec = False
        load_chunks = True # auto chunk output - this is necessary to maintain proper chunking!
        #       !!! Chunking of size (12, 205, 197) requires ~13GB in order to compute T2 (three arrays total) !!!
#         chunks = (9, 59, 59); lautoChunk = False
#         load_chunks = dict(time=chunks[0], y=chunks[1], x=chunks[2])
        clim_stns = ['UTM','Elora']
#         derived_varlist = ['dask_test']; load_list = ['T2']
#         derived_varlist = ['pet_pt']; load_list = ['T2']
#         derived_varlist = ['pet_pts']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']
#         derived_varlist = ['pet_hog']; load_list = ['Tmin', 'Tmax', 'T2']
#         derived_varlist = ['pet_har']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']
#         derived_varlist = ['pet_haa']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # Hargreaves with Allen correction
        # derived_varlist = ['pet_th']; load_list = ['T2', 'lat2D']
#         derived_varlist = ['pet_hog','pet_har','pet_haa','pet_th']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # PET approximations without radiation
#         derived_varlist = ['pet_pts','pet_pt']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # PET approximations with radiation
#         derived_varlist = ['T2']; load_list = ['Tmin', 'Tmax']
#         derived_varlist = ['liqwatflx_sno']; load_list = dict(NRCan=['precip'], SnoDAS=['snow']); bias_correction = 'rfbc'
        derived_varlist = ['liqwatflx_ne5']; load_list = dict(NRCan=['precip',], ERA5=['dswe'])
#         derived_varlist = ['T2','liqwatflx']; load_list = ['Tmin','Tmax', 'precip','snow']
#         bias_correction = 'rfbc'
#         grid = 'son2'; resolution = 'CA12'
#         grid = None; resolution = 'SON60'
#         grid = 'son2'; resolution = 'SON60'; load_chunks = dict(time=8, x=59, y=59)
#         grid = 'son2'
#         grid = 'snw2'; load_chunks = dict(time=8, x=44, y=55)
        # grid = None; load_chunks = dict(time=8, lon=63, lat=64)
        # dataset_args = dict(NRCan=dict(resolution='CA12', grid=None,), 
                            # ERA5=dict(resolution='NA10', grid='ca12', subset='ERA5L'), )
        grid = 'snw2'
        dataset_args = dict(NRCan=dict(resolution='SON60', grid=grid,), 
                            ERA5=dict(resolution='NA10', grid=grid, subset='ERA5L'), )

        
        
        # optional slicing (time slicing completed below)
        start_date = None; end_date = None # auto-detect available data
#         start_date = '2011-01-01'; end_date = '2017-12-31' # inclusive
#         start_date = '2011-01-01'; end_date = '2011-04-01'
#         start_date = '2012-11-01'; end_date = '2013-01-31'
#         start_date = '2011-12-01'; end_date = '2012-03-01'
#         start_date = '2011-01-01'; end_date = '2012-12-31'
        start_date = '1997-01-01'; end_date = '2017-12-31' # inclusive
#         start_date = '1981-01-01'; end_date = '2018-01-01' # apparently not inclusive... 
        # N.B.: it appears slicing is necessary to prevent some weird dtype error with time_stamp...
        
        # load datasets
        time_slice = (start_date,end_date) # slice time
        dataset = loadMergedForcing_Daily(varlist=load_list, grid=grid, resolution=resolution, bias_correction=bias_correction, 
                                          dataset_args=dataset_args, resampling=None, time_slice=time_slice, chunks=load_chunks,
                                          multi_chunks='regular')
#         dataset = dataset.unify_chunks()
        
        
        # load time coordinate
        print(dataset)
        tsvar = dataset[ts_name].load()
        
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
                # align time coords with 3D dataset
                stn_ens = xr.align(dataset, *stn_ens, join='left', copy=False,)[1:] # first is just the reference 
                # N.B.: stations were already clipped to time_slice, so dataset should never be extended (this just pads the station data)
                # transfer 1D radiation timeseries to 3D dataset
                dataset.attrs['zs'] = np.mean([ds.attrs['zs'] for ds in stn_ens]) # also need approximate elevation - station elevation if fine...
                rad_data = np.nanmean(np.stack([ds[radvar].values for ds in stn_ens], axis=1), axis=1)
                rad_var = xr.DataArray(data=rad_data, coords=(stn_ens[0].time,), name=radvar, attrs=stn_ens[0][radvar].attrs)
                dataset[radvar] = rad_var
                # find missing data
                mia_var = rad_var[np.isnan(rad_var.data)]
                if len(mia_var) > 0:
                    nc_folder,nc_filename = getFolderFileName(varname=varname, dataset='MergedForcing', resolution=resolution, grid=grid, 
                                                              resampling=None, dataset_index=default_dataset_index)
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
                kwargs = dict(lmeans=False, lq2=False, zs=400, lxarray=True) # average elevation of Canada is 487 m
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
                if resolution: nrcan_res = resolution
                elif dataset_args and 'NRCan' in dataset_args: 
                    nrcan_res = dataset_args['NRCan'].get('resolution',None)
                else:
                    raise ValueError(dataset_args)
                clim_chunks = ref_var.encoding['chunksizes'] if load_chunks is True else load_chunks.copy()
                clim_chunks['time'] = 12 # need 
                if nrcan_res == 'CA12':
                    T2clim = loadMergedForcing(varname='T2', name='climT2', dataset_name='NRCan', period=(1980,2010), resolution='NA12', 
                                               grid=grid, lxarray=True, lgeoref=False, chunks=None)['T2'].load().rename(time='month')
                    T2clim = T2clim.reindex_like(ref_var, method='nearest', copy=False) # slice out the CA12 grid
                elif nrcan_res == 'SON60':
                    T2clim = loadMergedForcing(varname='T2', name='climT2', dataset_name='NRCan', period=(1997,2018), resolution='SON60', 
                                            grid=grid, lxarray=True, lgeoref=False, chunks=clim_chunks)['T2'].load().rename(time='month')
                print('Size of T2 climatology:',T2clim.nbytes/1024/1024,'MB')
                dataset['climT2'] = T2clim
                # process timeseries
                from processing.newvars import computePotEvapTh
                note = 'PET based on the Thornthwaite method using only T2'
                kwargs = dict(climT2='climT2', lat=None, l365=False, p='center', lxarray=True)      
                xvar = xr.map_blocks(computePotEvapTh, dataset, kwargs=kwargs)
                print(xvar)
            elif varname == 'liqwatflx_sno': # SnoDAS
                default_varatts = varatts[varname]
                ref_var = dataset['precip']
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                swe = dataset['snow'].fillna(0) # just pretend there is no snow...
                assert swe.attrs['units'] == 'kg/m^2', swe.attrs['units']
                dswe = swe.differentiate('time', datetime_unit='s')
                swe_name = 'SnoDAS'
                if bias_correction: swe_name = swe_name + ' ' + bias_correction.upper()
                note = 'total precip (NRCan) - SWE changes from ' + swe_name
                xvar = ref_var - dswe
                xvar = xvar.clip(min=0,max=None) # remove negative values
            elif varname == 'liqwatflx_ne5': # ER5-Land
                default_varatts = varatts[varname]
                ref_var = dataset['precip']
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                dswe = dataset['dswe'].fillna(0)
                assert dswe.attrs['units'] == 'kg/m^2/s', dswe.attrs['units']
                swe_name = 'ERA5-Land'
                if bias_correction: swe_name = swe_name + ' ' + bias_correction.upper()
                note = 'total precip (NRCan) - SWE changes from ' + swe_name
                xvar = ref_var - dswe
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
            assert xvar.shape == ref_var.shape
            chunks = ref_var.encoding['chunksizes'] if load_chunks is True else load_chunks.copy()
            if chunks: 
                xvar = xvar.chunk(chunks=chunks)
            print('Chunks:',chunks)
                
            # create a dataset for export to new file
            ds_attrs = dataset.attrs.copy()
            if varname in default_dataset_index:
                orig_ds_name = default_dataset_index[varname]
                ds_attrs['name'] = orig_ds_name 
                if grid is None:
                    resampling = None
                else:
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
            nc_folder,nc_filename = getFolderFileName(varname=varname, dataset=ds_attrs['name'], resolution=resolution, grid=grid, 
                                                      resampling=resampling, dataset_index=default_dataset_index, mode='daily')
            nc_filepath = nc_folder + nc_filename
            tmp_filepath = nc_filepath + '.tmp' # use temporary file during creation
            print("\nExporting to new NetCDF-4 file:\n '{}'".format(nc_filepath))
            # write to NetCDF
            print(dataset.attrs)
            var_enc = dict(chunksizes=chunks, zlib=True, complevel=1, _FillValue=np.NaN, dtype=netcdf_dtype) # should be float
            task = nds.to_netcdf(tmp_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                          encoding={varname:var_enc,}, compute=False)
            if lexec:
                task.compute()
            else:
                print(var_enc)
                print(task)
                task.visualize(filename=nc_folder+'netcdf.svg')  # This file is never produced
            
            # replace original file
            if os.path.exists(nc_filepath): os.remove(nc_filepath)
            os.rename(tmp_filepath, nc_filepath)
            
        # print timing
        end =  time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
      