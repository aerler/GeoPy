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
from geospatial.xarray_tools import addGeoReference, loadXArray, computeNormals, getCommonChunks, saveXArray
from geospatial.netcdf_tools import addTimeStamps

## Meta-vardata

dataset_name = 'MergedForcing'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='HGS') # get dataset root folder based on environment variables

# attributes of variables in different collections
# Axes and static variables
axes_varatts = dict(time = dict(name='time', long_name='Time'), # time coordinate (units handled by xarray)
                    lon = dict(name='lon', units='deg', long_name='Longitude'), # longitude coordinate
                    lat = dict(name='lat', units='deg', long_name='Latitude'), # latitude coordinate
                    x  = dict(name='x', units='m', long_name='Easting'),
                    y  = dict(name='y', units='m', long_name='Northing'),)
axes_varlist = axes_varatts.keys()
# merged/mixed/derived variables
varatts = dict(liqwatflx = dict(name='liqwatflx', units='kg/m^2/s', long_name='Liquid Water Flux'),
               liqwatflx_sno = dict(name='liqwatflx_sno', units='kg/m^2/s', long_name='LWF (SnoDAS)'),
               liqwatflx_snons = dict(name='liqwatflx_snons', units='kg/m^2/s', long_name='LWF (SnoDAS/no subl.)'),
               liqwatflx_ne5 = dict(name='liqwatflx_ne5', units='kg/m^2/s', long_name='LWF (ERA5-Land)'),
               dswe = dict(name='dswe', units='kg/m^2/s', long_name='Simple SWE Differences'),
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


def loadMergedForcing_Daily(varname=None, varlist=None, dataset=None, dataset_index=None, dataset_args=None,
                            dataset_name=None, time_slice=None, compat='override', join='inner',
                            fill_value=np.NaN, ldebug=False, ldt64=True, **kwargs):
    ''' function to load and merge data from different high-resolution datasets (e.g. SnoDAS or NRCan) using xarray;
        typical dataset-agnostic arguments: grid=str, lgeoref=True, geoargs=dict, chunks=dict, lautoChunk=False,
        typical dataset-specific arguments: folder=str, resampling=str, resolution=str, bias_correction=str '''
    if not ldt64:
        raise NotImplementedError("Daily data are always loaded with a datetime64 axis")
    global_ds_atts_keys = ('resolution', 'bias_correction', 'resampling')
    # figure out varlist
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname:
        varlist = [varname]  # load a single variable
    elif varlist is None:
        if dataset is None: varlist = list(varatts.keys())
        else: varlist = {dataset:None, 'const':None}  # load default list for dataset
    if dataset_args is None: dataset_args = dict()  # avoid errors
    # assemble dataset list and arguments
    if isinstance(varlist, dict):
        dataset_varlists = varlist
    elif isinstance(varlist, (list, tuple)):
        if dataset:
            dataset_varlists = {dataset: varlist}
        else:
            if dataset_index is None: dataset_index = default_dataset_index.copy()
            dataset_varlists = dict()
            for varname in varlist:
                ds_name = dataset_index.get(varname, 'MergedForcing')  # default is native (global variable)
                if ds_name not in dataset_varlists: dataset_varlists[ds_name] = [varname]
                else: dataset_varlists[ds_name].append(varname)
    else:
        raise TypeError(varlist)
    const_list = dataset_varlists.pop('const', [])
    ## load datasets
    ds_list = []
    global_ds_atts_dict = dict()
    #print(dataset_name)
    for dataset, varlist in dataset_varlists.items():
        if ldebug: print("Loading", dataset, '\n', varlist, '\n')
        # prepare kwargs
        ds_args = dict(compat=compat, join=join, fill_value=fill_value)
        ds_args.update(kwargs)
        if dataset in dataset_args: ds_args.update(dataset_args[dataset])
        if dataset.lower() == 'MergedForcing'.lower():
            # native MergedForcing
            ds_args.update(dataset=dataset)
            argslist = ['grid', ]  # specific arguments for merged dataset variables
            if varlist is None: varlist = default_varlist
            loadFunction = loadXRDataset
        else:
            # daily data from other datasets
            ds_mod = import_module('datasets.{0:s}'.format(dataset))  # import dataset module
            loadFunction = ds_mod.loadDailyTimeSeries
            argslist = inspect.getfullargspec(loadFunction); argslist = argslist.args  # list of actual arguments
        # remove some args that don't apply
        for key in global_ds_atts_keys:  # list of dataset-specific arguments that have to be controlled
            if key not in argslist and key in ds_args: del ds_args[key]
        # load time series and and apply some formatting to vars
        ds = loadFunction(varlist=varlist, **ds_args)
        if ldebug: print(ds)
        # add some dataset attributes to variables, since we will be merging datasets
        for var in ds.variables.values(): var.attrs['dataset_name'] = dataset
        for key in global_ds_atts_keys:  # list of dataset-specific arguments that have to be controlled
            if key in ds.attrs: value = ds.attrs[key]
            elif key in ds_args: value = ds_args[key]
            else: value = None
            if value is not None:
                for var in ds.data_vars.values(): var.attrs[key] = value
                if key not in global_ds_atts_dict:
                    global_ds_atts_dict[key] = value
                elif global_ds_atts_dict[key] != value:
                    global_ds_atts_dict[key] = None  # only keep if all equal
        if time_slice: ds = ds.loc[{'time':slice(*time_slice),}]  # slice time
        ds_list.append(ds)
    # merge datasets and attributes
    if len(ds_list) == 1:
        if ldebug: print("Returning single dataset:", dataset, '\n')
        xds = ds_list[0]
        if dataset_name is None: dataset_name = dataset
    else:
        if ldebug: print("Merging Datasets:", compat, join, '\n')
        xds = xr.merge(ds_list, compat=compat, join=join, fill_value=fill_value)
        if ldebug: print(xds)
        for ds in ds_list[::-1]: xds.attrs.update(ds.attrs) # we want MergedForcing to have precedence
        for key, value in global_ds_atts_dict.items():
            if value is not None: xds.attrs[key] = value
        if dataset_name is None: 
            if len(dataset_varlists) == 1:
                dataset_name = next(iter(dataset_varlists))  # get first/only key
            else:
                dataset_name = 'MergedForcing'
    xds.attrs['name'] = dataset_name; xds.attrs['title'] = dataset_name + ' Daily Timeseries'
    ## add additional fields
    if len(const_list) > 0:
        if ldebug: print("Adding Constants:", const_list, '\n',)
        xds = addConstantFields(xds, const_list=const_list, grid=kwargs.get('grid',None))
    else:
        if ldebug: print("No Constants to add.\n")
    # return merged dataset
    return xds
loadDailyTimeSeries = loadMergedForcing_Daily


def loadMergedForcing_All(varname=None, varlist=None, name=None, dataset=None, dataset_name=None, varatts=None,
                          shape=None, station=None, mode=None, aggregation=None, period=None, lxarray=True,
                          ltoMonthly=None, ldt64=True, dataset_args=None, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    if isinstance(varlist, dict):
        raise NotImplementedError("Loading variables from multiple files or datasets is currently not implemented.")
    # resolve folder and filename
    if dataset is None and dataset_name is None:
        dataset_name = 'MergedForcing'
    elif dataset and dataset_name is None:
        dataset_name = dataset
    elif dataset and dataset_name:
        raise ValueError("Cannot use 'dataset' and 'dataset_name' argument at the same time.")
    arg_list = ('resampling', 'resolution', 'bias_correction', 'subset', 'grid')
    file_args = {key:kwargs.pop(key,None) for key in arg_list}
    if dataset_args is not None and dataset_name in dataset_args:
        dataset_args = dataset_args[dataset_name]
        for arg in arg_list:
            if arg in dataset_args: file_args[arg] = dataset_args[arg]
    folder, filename = getFolderFileName(varname=None, dataset=dataset_name, mode=mode, period=period,
                                         aggregation=aggregation, lcreateFolder=False, station=station,
                                         shape=shape, dataset_index=default_dataset_index, **file_args)
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
        if 'decode_times' not in kwargs: kwargs['decode_times'] = True
        if 'decode_timedelta' not in kwargs: kwargs['decode_timedelta'] = True  # does not seem to work
        # load dataset using master function
        xds = loadXArray(varlist=varlist, folder=folder, varatts=varatts, geoargs=geoargs,
                         filelist={'single': filename}, filetypes=['single'],  # mock 'type' for single file
                         default_varlist=None, varname=None, grid=None, filename_pattern=None, **kwargs)
        # load time stamps (like coordinate variables)
        if 'time_stamp' in xds: xds['time_stamp'].load()
        # add timedelta64 axis for monthly date and climatologies
        tax = xds.time
        if ldt64 and not (np.issubdtype(tax.dtype, np.datetime64) or np.issubdtype(tax.dtype, np.timedelta64)):
            if tax.attrs['units'].lower().startswith('month'):
                td_coord = tax.values.astype('timedelta64[M]')
            else:
                raise NotImplementedError(f"Currently only monthly climatologies are implemented; found units '{tax.attrs['units']}'")
            if 'clim' in aggregation.lower() and not np.issubdtype(tax.dtype, np.timedelta64):
                assert len(tax) == 12, tax
                assert xds.time[0] == 1, tax
                assert xds.time[11] == 12, tax
            elif not np.issubdtype(tax.dtype, np.datetime64):
                name_strs = tax.attrs['long_name'].lower().split()
                assert len(name_strs) == 3, name_strs
                assert tax.attrs['units'].lower().startswith('month'), name_strs
                assert name_strs[1] == 'since', name_strs
                td_coord = np.datetime64(name_strs[2], 'M') + td_coord
            xds = xds.assign_coords({'time': td_coord})
            assert np.issubdtype(xds.time.dtype, np.datetime64) or np.issubdtype(xds.time.dtype, np.timedelta64), xds.time.dtype
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
                coord = r.years*12 + r.months + np.arange(len(time))
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


def loadMergedForcing_TS(varname=None, varlist=None, name=None, dataset=None, varatts=None, grid=None, mode='avg',
                         aggregation='monthly', lxarray=False, ltoMonthly=True, ldt64=True, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly transient merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset=dataset, varatts=varatts, grid=grid,
                                 mode=mode, aggregation=aggregation, period=None, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs,
                                 ltoMonthly=ltoMonthly, ldt64=ldt64, shape=None, station=None, **kwargs)

def loadMergedForcing(varname=None, varlist=None, name=None, dataset=None, varatts=None, grid=None, period=None, mode='avg',
                      aggregation='clim', lxarray=False, ltoMonthly=True, ldt64=True, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load gridded monthly normal merged forcing data '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset=dataset, varatts=varatts, grid=grid,
                                 mode=mode, aggregation=aggregation, period=period, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs,
                                 ltoMonthly=ltoMonthly, ldt64=ldt64, shape=None, station=None, **kwargs)

def loadMergedForcing_ShpTS(varname=None, varlist=None, name=None, dataset=None, varatts=None, grid=None, shape=None, mode='avg',
                            aggregation='monthly', lxarray=False, ltoMonthly=True, ldt64=True, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load monthly transient merged forcing data averaged over shapes '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset=dataset, varatts=varatts, grid=grid,
                                 mode=mode, aggregation=aggregation, period=None, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs,
                                 ltoMonthly=ltoMonthly, ldt64=ldt64, shape=shape, station=None, **kwargs)

def loadMergedForcing_Shp(varname=None, varlist=None, name=None, dataset=None, varatts=None, grid=None, period=None, mode='avg',
                          aggregation='clim', shape=None, lxarray=False, ltoMonthly=True, ldt64=True, lgeoref=False, geoargs=None, **kwargs):
    ''' function to load monthly normal merged forcing data averaged over shapes '''
    return loadMergedForcing_All(varname=varname, varlist=varlist, name=name, dataset=dataset, varatts=varatts, grid=grid,
                                 mode=mode, aggregation=aggregation, period=period, lxarray=lxarray, lgeoref=lgeoref, geoargs=geoargs,
                                 ltoMonthly=ltoMonthly, ldt64=ldt64, shape=shape, station=None, **kwargs)

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
  # print('xarray version: '+xr.__version__+'\n')

  #gc.set_debug(gc.DEBUG_LEAK)


  ## Dask config
  lcluster = True

  import dask
  from dask.diagnostics import ProgressBar
  # dask.config.set(**{'array.slicing.split_large_chunks': False}) # suppress warnings about large chunks
  # dask.config.set(temporary_directory='G:/Data/TMP/')

  if lcluster:
      from dask.distributed import Client, LocalCluster
      # cluster = LocalCluster(processes=True, threads_per_worker=1, n_workers=2, memory_limit=0.3)  # for differencing
      cluster = LocalCluster(processes=True, threads_per_worker=2, n_workers=4, memory_limit=0.2)  # for simple computations
      client = Client(cluster)
      print(client)
      comp_args = None
  else:
      comp_args = dict(scheduler='threads', num_workers=4)


  ts_name = 'time_stamp'
  process_dataset = dataset_name # we can't overwrite dataset_name without causing errors...
  dataset_args = None;  filetype = None;  subset = None

  work_loads = []
#   work_loads += ['load_Point_Climatology']
#   work_loads += ['load_Point_Timeseries']
#   work_loads += ['print_grid']
  # work_loads += ['compute_derived']
  # work_loads += ['load_Daily']
  # work_loads += ['monthly_mean']
  # work_loads += ['load_TimeSeries']
  work_loads += ['monthly_normal']
  # work_loads += ['load_Climatology']

  # some settings
  process_dataset = 'MergedForcing'; resolution = None
  process_dataset = 'NRCan'
  resolution = 'NA12'
  # resolution = 'SON60'

  # process_dataset = 'ERA5'; subset = 'ERA5L'
  # dataset_args = dict(ERA5=dict(subset='ERA5L', lfliplat=True))
  # # # resolution = 'AU10'
  # resolution = 'NA10'

  grid = None; bias_correction = None; period = None
  
  # grid = 'snw2'
  # grid = 'son2'
#   grid = 'hd1' # small Quebec grid
#   grid = 'son2'; bias_correction = 'rfbc' # high-res Southern Ontario
#   grid = 'on1'
  # grid = 'na12'
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

        mode = 'avg'
        lxarray = True
#         period = (2011,2018)
        # period = (1997,2018)
        # period = (1981, 2011)
        # period = (1981, 2020)
        # period = (2000, 2020)
        period = period or prdstr  # from monthly_normal

        # process_dataset = 'MergedForcing'
        # process_dataset = 'NRCan'  # can take grid instead of resolution
        # process_dataset = 'ERA5'
        # varlist = ['pet_hog', 'liqwatflx_ne5', 'liqwatflx', 'pet_era5']
        varlist = None

        grid_res = 'na12'
        dataset_args = dict(ERA5=dict(grid='na10', subset='ERA5L'))
        ## N.B.: actually 'NA10' is the 'resolution' for ERA5, but like with NRCan, too,
        #        we can substitute 'grid' for 'resolution', if no such grid exists

        xds = loadMergedForcing(period=period, varlist=varlist, grid=grid_res,
                                dataset_name=process_dataset, dataset_args=dataset_args,
                                mode=mode, aggregation='clim', lxarray=lxarray)
        print(xds)
        print('')
        if lxarray:
            xv = list(xds.data_vars.values())[0]
            print(xv)
            if lxarray:
                print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))

    elif mode == 'monthly_normal':

        mode = 'daily'
        multi_chunks = 'regular'
        # optional slicing (time slicing completed below)
        start_date = None; end_date = None; varlist = None
        start_date = '2010-01'; end_date = '2039-12'; varlist = None
        # start_date = '2011-01'; end_date = '2012-12'; varlist = None
        # start_date = '1981-01'; end_date = '2010-12'; varlist = None # date ranges are inclusive
        # start_date = '2000-01'; end_date = '2019-12'; varlist = None # date ranges are inclusive
        # start_date = '1981-01'; end_date = '2020-12'; varlist = None # date ranges are inclusive        
        # start_date = '1981-01'; end_date = None; varlist = None # date ranges are inclusive
        # start_date = '2003-01'; end_date = '2017-12'; varlist = None # date ranges are inclusive

        # process_dataset = 'NRCan'; resolution = 'NA12'
        # process_dataset = 'MergedForcing'; grid = 'na12'
        # varlist = ['T2','time_stamp']
        
        # Tyler's high-res soil data
        process_dataset = 'C1W_soil'
        mode = 'avg'
        resolution = 'geo005'
        varlist = None
        mode = 'avg'
        multi_chunks = dict(time=1, lon=40, lat=20)


        # # just ERA5-land
        # process_dataset = 'ERA5'; subset = 'ERA5L'
        # #varlist = {'ERA5':['precip','liqwatflx','pet_era5','snow','dswe'], 'const':None}
        # dataset_args = dict(ERA5=dict(subset=subset))
        # # resolution = 'AU10'; grid = 'qel1'
        # # resolution = 'AU10'; grid = None
        # # resolution = 'NA10'; grid = 'snw2'
        # # resolution = 'NA10'; grid = None


        # start operation
        start = time.time()
        
        # load variables object (not data!)
        xds = loadMergedForcing_TS(varlist=varlist, grid=grid, dataset_name=process_dataset,
                                   resolution=resolution, mode=mode, dataset_args=dataset_args,
                                   multi_chunks=multi_chunks, lxarray=True)  # need Dask!
        xds = xds.loc[{'time':slice(start_date,end_date),}] # slice entire dataset
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
        print('Original Chunks:', chunks)
        # save resampled dataset
        nc_folder, nc_filename = getFolderFileName(dataset=process_dataset, resolution=resolution, grid=grid,
                                                   period=prdstr, mode=mode, aggregation='clim',
                                                   dataset_index=default_dataset_index, subset=subset)
        # save to NetCDF file, with all bells and whistles        
        saveXArray(cds, filename=nc_filename, folder=nc_folder, mode='write', varlist=None, chunks=chunks,
                   encoding=None, time_agg='month', laddTime=True, time_dim='time', ltmpfile=True,
                   lcompute=True, lprogress=True, lfeedback=True, comp_args=comp_args)

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
      
        process_dataset = 'C1W_soil'
        mode = 'avg'
        resolution = 'geo005'

        lxarray = True
        varname = None
        xds = loadMergedForcing_TS(varlist=None, dataset_name=process_dataset, dataset_args=dataset_args, mode=mode,
                                   resolution=resolution, grid=grid, lxarray=lxarray, ldropAtts=False)
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
        # time_slice = ('2011-01-01', '2012-12-31')  # inclusive
        varlist = None
        dataset_args = None;  filetype = None

        # # Merged Forcing
        # process_dataset = 'MergedForcing';  resolution = None; subset = None
        # varlist = {'MergedForcing': ['liqwatflx_ne5'], 'const': ['lat2D', 'lon2D']}
        # grid = 'na12'

        # # process just NRCan dataset
        # process_dataset = 'NRCan'
        # varlist = {'NRCan': ['precip', 'T2', 'Tmin', 'Tmax', 'pet_har', 'pet_haa', 'pet_hog'],
        #            'const': ['lat2D', 'lon2D']}
        # # varlist['MergedForcing'] = ['liqwatflx_ne5']
        # resolution = 'NA12'; grid = None
        # # resolution = 'CA12'; #chunks = dict(time=8, lat=64, lon=63)
        # # grid = 'son2'; resolution = 'SON60'
        # # grid = 'snw2'; resolution = 'SON60'

        # just ERA5-land
        process_dataset = 'ERA5'; subset = 'ERA5L'
        varlist = {'ERA5':['precip','liqwatflx','pet_era5','snow','dswe'], 'const':None}
        dataset_args = dict(ERA5=dict(subset=subset, lfliplat=True, combine_attrs='drop_conflicts'))
        #resolution = 'NA10'; grid = 'son2'
        # resolution = 'NA10'; grid = 'snw2'
        resolution = 'NA10'; grid = None
        # resolution = 'AU10'; grid = 'qel1'
        #resolution = 'AU10'; grid = None

        # auto chunk, but use multiple of chunks for better workloads (~ 100 MB per chunk)
        multi_chunks = {dim: 16 for dim in ('lat', 'lon', 'latitude', 'longitude', 'x', 'y')}
        multi_chunks['time'] = 4  # roughly 1 month years...

        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, dataset_args=dataset_args, join='outer', fill_value=np.NaN,
                                      bias_correction=bias_correction, resolution=resolution,
                                      time_slice=time_slice, multi_chunks=multi_chunks, chunks=True, lskip=True)
        # save and remove CRS variable to add after averaging
        if 'crs' in xds:
            crs_var = xds.variables['crs']
            xds = xds.drop('crs')
        else:
            crs_var = None
        #xds = xds.unify_chunks()
        chunks = getCommonChunks(xds, method='min')  # used later
        print(xds)
        print('')
        # start operation
        start = time.time()

        # aggregate month
        rds = xds.resample(time='MS', skipna=True,).mean()
        print(rds)
        print('')
        # collect garbage
        del xds
        gc.collect()

        # add CRS variable if it was there in the first place
        if crs_var:
            rds += crs_var

        # define destination file
        nc_folder, nc_filename = getFolderFileName(dataset=process_dataset, grid=grid, resolution=resolution, subset=subset,
                                                   bias_correction=bias_correction, mode='daily', aggregation='monthly',
                                                   dataset_index=default_dataset_index, data_root=None)
        print('Original Chunks:', chunks)
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
        varlist = dict(
                        # MergedForcing=['liqwatflx_ne5'],
                        # NRCan=['pet_hog'],
                        NRCan=['precip', 'T2', 'Tmin', 'Tmax', 'pet_har', 'pet_haa', 'pet_hog'],
                        # ERA5=['snow', 'dswe'],
                        const=['lat2D', 'lon2D'],
                       )
#         varlist = {dataset:None for dataset in dataset_list+[dataset_name, 'const']} # None means all...
        # varlist = {'NRCan':None, 'const':None, 'MergedForcing':['liqwatflx_ne5']}
#         varlist = {'ERA5':['precip'], 'const':None}; dataset_args = dict(ERA5=dict(filetype='ERA5L', lfliplat=True))
        grid = None; resolution = None; bias_correction = None
        dataset_args = dict(MergedForcing=dict(resolution=None, grid='na12'),
                            NRCan=dict(resolution='NA12', grid=None),
                            ERA5=dict(resolution='NA10', grid='na12', subset='ERA5L'),
                            SnoDAS=dict(bias_correction='rfbc'),
                            )
        time_slice = None
        xds = loadMergedForcing_Daily(varlist=varlist, grid=None, dataset_args=dataset_args,
                                      resolution=None, bias_correction=None,
                                      join='outer', fill_value=np.NaN, chunks=False,
                                      time_slice=time_slice, multi_chunks=multi_chunks, lskip=True)
        print(xds)
        print('')

        # for varname,xv in xds.variables.items():
        #     if xv.ndim == 3: break
        # xv = xds[varname] # get DataArray instead of Variable object
        # print(xv)
        # print(xv.encoding)
        # print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))

        tax = xds.time
        print(tax)
        J = 1 + ( tax.astype('datetime64[D]') - tax.astype('datetime64[Y]') ) / np.timedelta64(1,'D')
        J1980 = J.sel(time=slice('1981-01-01', '1981-12-31')).data
        print(f"Julian Day: {J1980}")

    elif mode == 'compute_derived':

        #with dask.config.set(pool=ThreadPool(4)): # I'm not sure if this works... or it is I/O limited

        start = time.time()

        # settings
        lexec = True
        # lexec = False
        resolution = None; bias_correction = None; dataset_args = dict(ERA5=dict(filetype='ERA5L'))
        # netcdf_chunks = None
        # load_chunks = True # auto chunk output - this is necessary to maintain proper chunking!
        netcdf_chunks = (1, 128, 128)
        load_chunks = {'time': 8*4, 'lat': 256*4, 'lon': 256*4}
        # load_chunks = dict(time=64, lat=480, lon=464)
        clim_stns = ['UTM','Elora']
        # derived_varlist = ['dask_test']; load_list = ['T2']
        # derived_varlist = ['pet_pt']; load_list = ['T2']
        # derived_varlist = ['pet_pts']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']
        # derived_varlist = ['pet_hog']; load_list = ['Tmin', 'Tmax', 'T2']
        # derived_varlist = ['pet_har']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D']
        # derived_varlist = ['pet_haa']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # Hargreaves with Allen correction
        # derived_varlist = ['pet_th']; load_list = ['T2', 'lat2D']
        # derived_varlist = ['pet_hog','pet_har','pet_haa','pet_th']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # PET approximations without radiation
        # derived_varlist = ['pet_pts','pet_pt']; load_list = ['Tmin', 'Tmax', 'T2', 'lat2D'] # PET approximations with radiation
        # derived_varlist = ['T2']; load_list = ['Tmin', 'Tmax']
        derived_varlist = ['liqwatflx_sno']; load_list = dict(NRCan=['precip'], SnoDAS=['dswe', 'evap_snow', 'evap_blow']); bias_correction = None  # 'rfbc'
        # derived_varlist = ['liqwatflx_snons']; load_list = dict(NRCan=['precip'], SnoDAS=['dswe']); bias_correction = None  # 'rfbc'
        # derived_varlist = ['dswe']; load_list = dict(SnoDAS=['snow']); bias_correction = None
        # derived_varlist = ['liqwatflx_ne5']; load_list = dict(NRCan=['precip',], ERA5=['dswe'], )
        # derived_varlist = ['T2','liqwatflx']; load_list = ['Tmin','Tmax', 'precip','snow']
        # grid = 'son2'; resolution = 'CA12'
        # grid = None; resolution = 'SON60'
        # grid = 'son2'; resolution = 'SON60'; load_chunks = dict(time=8, x=59, y=59)
        # grid = 'son2'
        # grid = 'snw2'; load_chunks = dict(time=8, x=44, y=55)
        # grid = None; load_chunks = dict(time=8, lon=63, lat=64)
        # dataset_args = dict(NRCan=dict(resolution='CA12', grid=None,),
                            # ERA5=dict(resolution='NA10', grid='ca12', subset='ERA5L'), )
        # grid = None; resolution = 'NA12'
        # dataset_args = dict(NRCan=dict(resolution='NA12',),
        #                     ERA5=dict(resolution='NA10', subset='ERA5L', grid='na12'), )
        # multi_chunks = 'small'
        dataset_args = dict(NRCan=dict(resolution='NA12', grid='snodas'),
                            SnoDAS=dict(grid=None, ), join='inner')
        # multi_chunks = dict(time=1, lat=8, lon=8)  # for SnoDAS, which has fairly large chunks
        multi_chunks = None
        
        # if grid == 'son2' and load_chunks:
        #     load_chunks = {'time':72, 'y':59, 'x':59}


        # optional slicing (time slicing completed below)
        # start_date = None; end_date = None # auto-detect available data
        start_date = '2009-12-14'; end_date = '2020-12-31' # apparently not inclusive...
        # start_date = '2011-01-01'; end_date = '2017-12-31' # inclusive
        # start_date = '2011-01-01'; end_date = '2011-04-01'
        # start_date = '2012-11-01'; end_date = '2013-01-31'
        # start_date = '2011-12-01'; end_date = '2012-03-01'
        # start_date = '2011-01-01'; end_date = '2012-12-31'
        # start_date = '1997-01-01'; end_date = '2017-12-31' # inclusive
        # start_date = '1981-01-01'; end_date = '2020-08-31'  # currently available ERA5-Land data
        # start_date = '1981-01-01'; end_date = '2020-12-31' # apparently not inclusive...
        # N.B.: it appears slicing is necessary to prevent some weird dtype error with time_stamp...

        # load datasets
        time_slice = (start_date, end_date) # slice time
        dataset = loadMergedForcing_Daily(varlist=load_list, grid=grid, resolution=resolution, bias_correction=bias_correction,
                                          dataset_args=dataset_args, resampling=None, time_slice=time_slice, chunks=load_chunks,
                                          multi_chunks=multi_chunks, join='override', ldebug=False)
        # dataset = dataset.unify_chunks()


        # load time coordinate
        print(dataset)
        tsvar = dataset[ts_name].load() if ts_name in dataset else None

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
                dataset.attrs['zs'] = 400  # average elevation of Canada is 487 m
                kwargs = dict(lmeans=False, lq2=False, lxarray=True)
                xvar = xr.map_blocks(computePotEvapHog, dataset, kwargs=kwargs)
            elif varname == 'pet_har' or varname == 'pet_haa':
                from processing.newvars import computePotEvapHar
                default_varatts = varatts[varname]; ref_var = dataset['Tmax']
                if varname == 'pet_haa':
                    note = 'PET based on the Hargreaves method with Allen correction using only Tmin and Tmax'; lAllen = True
                else:
                    note = 'PET based on the Hargreaves method using only Tmin and Tmax'; lAllen = False
                kwargs = dict(lmeans=False, lAllen=lAllen, l365=False, lxarray=True)
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
                print(clim_chunks)
                clim_chunks['time'] = 12  # need
                if nrcan_res == 'CA12':
                    T2clim = loadMergedForcing(varname='T2', name='climT2', dataset_name='NRCan', period=(1980,2010), resolution='NA12',
                                               grid=grid, lxarray=True, lgeoref=False, chunks=None)['T2'].load().rename(time='month')
                    T2clim = T2clim.reindex_like(ref_var, method='nearest', copy=False) # slice out the CA12 grid
                elif nrcan_res == 'SON60':
                    T2clim = loadMergedForcing(varname='T2', name='climT2', dataset_name='NRCan', period=(1997,2018), resolution='SON60',
                                               grid=grid, lxarray=True, lgeoref=False, chunks=clim_chunks)['T2'].load().rename(time='month')
                else:
                    raise ValueError(f"No climatology available for this resolution/grid: '{nrcan_res}'")
                print('Size of T2 climatology:',T2clim.nbytes/1024/1024,'MB')
                dataset['climT2'] = T2clim
                # process timeseries
                from processing.newvars import computePotEvapTh
                note = 'PET based on the Thornthwaite method using only T2'
                kwargs = dict(climT2='climT2', l365=False, p='center', lxarray=True)
                xvar = xr.map_blocks(computePotEvapTh, dataset, kwargs=kwargs)
                print(xvar)
            elif varname == 'dswe':  # simple 1st-order SWE differences
                default_varatts = varatts[varname]
                ref_var = dataset['snow']
                swe = dataset['snow'].fillna(0)  # just pretend there is no snow...
                assert swe.attrs['units'] == 'kg/m^2', swe.attrs['units']
                xvar = swe.diff('time', n=1, label='upper') / 86400  # per second
                swe_name = dataset.name
                if bias_correction: swe_name = swe_name + ' ' + bias_correction.upper()
                note = 'simple SWE changes from ' + swe_name
                tsvar = tsvar[1:]  # need to clip first day
            elif varname == 'liqwatflx_sno':  # SnoDAS with sublimation
                default_varatts = varatts[varname]
                ref_var = dataset['precip']
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                dswe = dataset['dswe'].fillna(0)
                assert dswe.attrs['units'] == 'kg/m^2/s', dswe.attrs['units']
                evap_snow = dataset['evap_snow'].fillna(0)
                assert evap_snow.attrs['units'] == 'kg/m^2/s', evap_snow.attrs['units']
                evap_blow = dataset['evap_blow'].fillna(0)
                assert evap_blow.attrs['units'] == 'kg/m^2/s', evap_blow.attrs['units']
                swe_name = 'SnoDAS with sublimation'
                if bias_correction: swe_name = swe_name + ' ' + bias_correction.upper()
                note = 'total precip (NRCan) - SWE changes from ' + swe_name
                xvar = ref_var - dswe + evap_snow + evap_blow  # sublimation is negative in SnoDAS
                xvar = xvar.clip(min=0, max=None)  # remove negative values
            elif varname == 'liqwatflx_snons':  # SnoDAS without sublimation
                default_varatts = varatts[varname]
                ref_var = dataset['precip']
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                dswe = dataset['dswe'].fillna(0)
                assert dswe.attrs['units'] == 'kg/m^2/s', dswe.attrs['units']
                swe_name = 'SnoDAS without sublimation'
                if bias_correction: swe_name = swe_name + ' ' + bias_correction.upper()
                note = 'total precip (NRCan) - SWE changes from ' + swe_name
                xvar = ref_var - dswe
                xvar = xvar.clip(min=0, max=None)  # remove negative values
            elif varname == 'liqwatflx_ne5':  # ERA5-Land
                default_varatts = varatts[varname]
                ref_var = dataset['precip']
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                dswe = dataset['dswe'].fillna(0)
                assert dswe.attrs['units'] == 'kg/m^2/s', dswe.attrs['units']
                swe_name = 'ERA5-Land'
                if bias_correction: swe_name = swe_name + ' ' + bias_correction.upper()
                note = 'total precip (NRCan) - SWE changes from ' + swe_name
                xvar = ref_var - dswe
                xvar = xvar.clip(min=0, max=None)  # remove negative values
            else:
                raise NotImplementedError(varname)

            # define/copy metadata
            xvar.attrs = ref_var.attrs.copy()
            if 'history' in xvar.attrs: del xvar.attrs['history']
            xvar.rename(varname)
            for att in ('name', 'units','long_name',):
                if att in default_varatts: xvar.attrs[att] = default_varatts[att]
            if 'original_name' in xvar.attrs: del xvar.attrs['original_name'] # does not apply
            xvar.attrs['note'] = note
            # set chunking for operation
            assert xvar.shape == ref_var.shape or varname == 'dswe'
            chunks = netcdf_chunks if netcdf_chunks else ref_var.encoding['chunksizes']
            # if chunks:
            #     xvar = xvar.chunk(chunks=chunks)
            print('NetCDF Chunks:', chunks)

            # create a dataset for export to new file
            ds_attrs = dataset.attrs.copy()
            if 'history' in ds_attrs: del ds_attrs['history']
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
            if tsvar is None:
                nds = xr.Dataset({varname:xvar,}, attrs=ds_attrs)
            else:
                nds = xr.Dataset({ts_name:tsvar, varname:xvar,}, attrs=ds_attrs)
            nds = addGeoReference(nds, proj4_string=ds_attrs.get('proj4', None), )
            print('\n')
            print(nds)
            # file path based on variable parameters
            nc_folder, nc_filename = getFolderFileName(varname=varname, dataset=ds_attrs['name'], resolution=resolution, grid=grid,
                                                       resampling=resampling, dataset_index=default_dataset_index, mode='daily')
            nc_filepath = nc_folder + nc_filename
            tmp_filepath = nc_filepath + '.tmp' # use temporary file during creation
            print("\nExporting to new NetCDF-4 file:\n '{}'".format(nc_filepath))
            # write to NetCDF
            print(nds.attrs)
            var_enc = dict(chunksizes=chunks, zlib=True, complevel=1, _FillValue=np.NaN, dtype=netcdf_dtype) # should be float
            print(var_enc)
            task = nds.to_netcdf(tmp_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                                 encoding={varname:var_enc}, compute=False)
            if lexec:
                print(f"\nComputing variable '{varname}'...")
                # with ProgressBar():
                    # task.compute(scheduler='threads', num_workers=4)
                task.compute()
            else:
                print(var_enc)
                print(task)
                task.visualize(filename=nc_folder + 'netcdf.svg')  # This file is never produced

            # add timestamps, if necessary
            if tsvar is None:
                ncds = nc.Dataset(tmp_filepath, mode='a', format='NETCDF4', clobber=False)
                addTimeStamps(ncds, time='time', units=None, atts=None)
                ncds.close()

            # replace original file
            if os.path.exists(nc_filepath): os.remove(nc_filepath)
            os.rename(tmp_filepath, nc_filepath)

        # print timing
        end =  time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
