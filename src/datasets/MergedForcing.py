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
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
from collections import namedtuple
# internal imports
from datasets.common import getRootFolder
# for georeferencing
from geospatial.netcdf_tools import autoChunk
from geospatial.xarray_tools import addGeoReference, readCFCRS, loadXArray

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
# merged/mixed variables
varatts = dict(liqwatflx = dict(name='liqwatflx',units='kg/m^2/s',scalefactor=1., 
                                long_name='Liquid Water Flux'),)
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
                          CaSPAr  = DSNT(name='CaSPAr',  interval='6H', start_date='2017-09-11T12', end_date='2019-12-30T12', ),)
dataset_list = list(dataset_attributes.keys())
# N.B.: the effective start date for CaPA and all the rest is '2017-09-11T12'
default_dataset_index = dict(precip='NRCan', T2='NRCan', Tmin='NRCan', Tmax='NRCan', 
                             snow='SnoDAS', dswe='SnoDAS')


## helper functions

def getFolderFileName(varname=None, dataset=None, grid=None, resampling=None, resolution=None, bias_correction=None, 
                      mode='Daily', period=None):
    ''' function to provide the folder and filename for the requested dataset parameters '''
    if period is not None: raise NotImplementedError(period)
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
    filename = '{}_{}_{}.nc'.format(ds_str_rs, bcstr + varname.lower() + gridstr, mode.lower())
    # construct folder
    folder = getRootFolder(dataset_name=dataset, fallback_name='MergedForcing')
    if mode.lower() == 'daily': folder += ds_str+'_daily'
    else: folder += ds_str+'avg'
    if grid: 
        folder = '{}/{}'.format(folder,grid) # non-native grids are stored in sub-folders
        if resampling: 
            folder = '{}/{}'.format(folder,resampling) # different resampling options are stored in subfolders
            # could auto-detect resampling folders at a later point... 
    if folder[-1] != '/': folder += '/'       
    # return folder and filename
    return folder,filename
        

## functions to load NetCDF datasets (using xarray)

def loadSnoDAS_Daily(varname=None, varlist=None, folder=None, grid=None, bias_correction=None, resampling=None,
                     lgeoref=True, geoargs=None, chunks=None, lautoChunk=False, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    from datasets.SnoDAS import daily_folder, netcdf_varlist, netcdf_filename
    if folder is None: folder = daily_folder
    xds = loadXArray(varname=varname, varlist=varlist, folder=folder, grid=grid, bias_correction=bias_correction, resolution=None,
                      filename_pattern=netcdf_filename, default_varlist=netcdf_varlist, resampling=resampling, lgeoref=lgeoref, 
                      geoargs=geoargs, chunks=chunks, lautoChunk=lautoChunk, **kwargs)
    xds.attrs['name'] = 'SnoDAS'; xds.attrs['title'] = xds.attrs['name']+' Daily Timeseries' 
    return xds


def loadNRCan_Daily(varname=None, varlist=None, folder=None, grid=None, resolution='CA12', resampling=None,
                    lgeoref=True, geoargs=None, chunks=None, lautoChunk=False, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    from datasets.NRCan import daily_folder, netcdf_filename
    if folder is None: folder = daily_folder
    if resolution == 'CA12':
        from datasets.NRCan import day12_vardefs, day12_derived
        default_varlist = list(day12_derived) + [atts['name'] for atts in day12_vardefs.values()]
    xds = loadXArray(varname=varname, varlist=varlist, folder=folder, grid=grid, bias_correction=None, resolution=resolution,
                      filename_pattern=netcdf_filename, default_varlist=default_varlist, resampling=resampling, lgeoref=lgeoref, 
                      geoargs=geoargs, chunks=chunks, lautoChunk=lautoChunk, **kwargs)
    xds.attrs['name'] = 'NRCan'; xds.attrs['title'] = xds.attrs['name']+' Daily Timeseries' 
    return xds


def loadMergedForcing_Daily(varname=None, varlist=None, dataset_index=None, dataset_args=None, time_slice=None, 
                            compat='override', join='inner', fill_value=None, **kwargs):
    ''' function to load and merge data from different high-resolution datasets (e.g. SnoDAS or NRCan) using xarray;
        typical dataset-agnostic arguments: grid=str, lgeoref=True, geoargs=dict, chunks=dict, lautoChunk=False, 
        typical dataset-specific arguments: folder=str, resolution=str, resampling=str '''
    # figure out varlist
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname:
        varlist = [varname] # load a single variable
    elif varlist is None:
        varlist = list(varatts.keys())   
    if dataset_args is None: dataset_args = dict()# avoid errors
    # assemble dataset list and arguments
    if dataset_index is None: dataset_index = default_dataset_index.copy()
    dataset_varlists = dict()
    for varname in varlist:
        ds_name = dataset_index.get(varname,dataset_name) # default is native (global variable)
        if ds_name not in dataset_varlists: dataset_varlists[ds_name] = [varname] 
        else: dataset_varlists[ds_name].append(varname)         
    ## load datasets
    ds_list = []
    # Merged Forcing
    if dataset_name in dataset_varlists:
        # prepare kwargs
        ds_args = kwargs.copy(); 
        if dataset_name in dataset_args: ds_args.update(dataset_args[dataset_name])
        ds = loadXArray(varlist=dataset_varlists[dataset_name], folder=daily_folder, 
                        filename_pattern=netcdf_filename, default_varlist=None, **kwargs)
        for var in ds.variables.values(): var.attrs['dataset_name'] = dataset_name
        if time_slice: ds = ds.loc[{'time':slice(*time_slice),}] # slice time
        ds_list.append(ds)
    # SnoDAS
    if 'SnoDAS' in dataset_varlists:
        # prepare kwargs
        ds_args = kwargs.copy()
        if 'SnoDAS' in dataset_args: ds_args.update(dataset_args['SnoDAS'])
        ds = loadSnoDAS_Daily(varlist=dataset_varlists['SnoDAS'], **ds_args)
        for var in ds.variables.values(): var.attrs['dataset_name'] = 'SnoDAS'
        if time_slice: ds = ds.loc[{'time':slice(*time_slice),}] # slice time
        ds_list.append(ds)    
    # NRCan
    if 'NRCan' in dataset_varlists:
        # prepare kwargs
        ds_args = kwargs.copy(); 
        if 'NRCan' in dataset_args: ds_args.update(dataset_args['NRCan'])
        if 'resolution' not in ds_args: ds_args['resolution'] = 'CA12'
        ds = loadNRCan_Daily(varlist=dataset_varlists['NRCan'], **ds_args)
        for var in ds.variables.values(): var.attrs['dataset_name'] = 'NRCan'
        if time_slice: ds = ds.loc[{'time':slice(*time_slice),}] # slice time - helps with merging!
        ds_list.append(ds)
    # TODO: in the future it may make sense to move the dataset load functions back into the original datasets and turn this load-
    #       block into a generic loop and use impotlib to import the respective load functions as needed
    # merge datasets and attributed
    xds = xr.merge(ds_list, compat=compat, join=join, fill_value=fill_value)
    for ds in ds_list[::-1]: xds.attrs.update(ds.attrs) # we want MergedForcing to have precedence
    xds.attrs['name'] = 'MergedForcing'; xds.attrs['title'] = 'Merged Forcing Daily Timeseries'
    # return merged dataset
    return xds
loadDailyTimeSeries = loadMergedForcing_Daily

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
#   modes += ['compute_derived']
#   modes += ['load_NRCan']
  modes += ['load_Daily']
#   modes += ['compute_PET']  

  # some settings
  grid = None
#   grid = 'hd1' # small Quebec grid
  grid = 'son2' # high-res Southern Ontario
  period = ('2011-01-01','2018-01-01')
  
  # loop over modes 
  for mode in modes:
    
    if mode == 'print_grid':
        
        from geodata.gdal import loadPickledGridDef
        griddef = loadPickledGridDef(grid='on1')
        print(griddef)
                             
    elif mode == 'load_Daily':
       
  #       varlist = netcdf_varlist
        varlist = ['precip','snow','liqwatflx']
        dataset_args = dict(SnoDAS=dict(bias_correction='rfbc'))
        time_slice = ('2011-01-01','2017-01-01')
        time_slice = None
        xds = loadMergedForcing_Daily(varlist=varlist, grid=grid, dataset_args=dataset_args, time_slice=time_slice)
        print(xds)
        print('')
        print(xds.attrs)
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
#         xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
    
    elif mode == 'load_NRCan':
       
        varlist = ['liqwatflx',]
        xds = loadNRCan_Daily(varname='precip', grid=None, lautoChunk=True)
        print(xds)
        print('')
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
        #xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
        xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
        
    elif mode == 'compute_derived':
      
      with dask.config.set(pool=ThreadPool(4)): # I'm not sure if this works... or it is I/O limited
        
        start = time.time()
        
        # settings
        lexec = True
#         lexec = False
        load_chunks = None; lautoChunkLoad = False  # chunking input should not be necessary, if the source files are chunked properly
        chunks = None; lautoChunk = True # auto chunk output - this is necessary to maintain proper chunking!
        # N.B.: 'lautChunk' is necessary for *loading* data in chunks - otherwise it loads the whole array at once...
        #       !!! Chunking of size (12, 205, 197) requires ~13GB in order to compute T2 (three arrays total) !!!
#         chunks = (9, 59, 59); lautoChunk = False
#         load_chunks = dict(time=chunks[0], y=chunks[1], x=chunks[2])
#         derived_varlist = ['T2']
        derived_varlist = ['liqwatflx']
        grid = 'son2'
        resolution = 'CA12'
        ts_name = 'time_stamp'
        
        # optional slicing (time slicing completed below)
        start_date = None; end_date = None # auto-detect available data
#         start_date = '2011-01-01'; end_date = '2012-01-01'
        
        # load NRCan data
        from datasets.NRCan import day12_vardefs, varatts
        load_varlist = [atts['name'] for atts in day12_vardefs.values()]
        nrcan = loadNRCan_Daily(varname='precip', grid=grid, resampling='cubic_spline', resolution=resolution, 
                                lautoChunk=lautoChunkLoad, chunks=load_chunks) # take all
        nrcan = nrcan.loc[{'time':slice(start_date,end_date),}] # slice time
        # load bias-corrected SnoDAS data
        snodas = loadSnoDAS_Daily(varname='snow', grid=grid, resampling='bilinear', bias_correction='rfbc', 
                                  lautoChunk=lautoChunkLoad, chunks=load_chunks) # take all
        snodas = snodas.loc[{'time':slice(start_date,end_date),}] # slice time
        
        
        # load time coordinate
        tsvar = nrcan[ts_name].load()
               
        print(snodas)
        
        # loop over variables
        for varname in derived_varlist:
            
            print("\n   ***   Processing Variable '{}'   ***   \n".format(varname))
            
            # compute values 
            ref_ds = None # None means native to MergedForcing, i.e. derived from different products
            if varname == 'T2':
                from datasets.NRCan import varatts as ref_varatts
                ref_varatts = ref_varatts[varname]
                ref_var = nrcan['Tmax']; ref_ds = nrcan
                
                note = 'simple average of Tmin and Tmax'          
                xvar = nrcan['Tmin'] + ref_var
                xvar /= 2                
            elif varname == 'liqwatflx':
                ref_varatts = varatts[varname]
                ref_var = nrcan['precip']
                note = 'total precip (NRCan) - SWE changes from RFBC SnoDAS'
                assert ref_var.attrs['units'] == 'kg/m^2/s', ref_var.attrs['units']
                swe = snodas['snow'].fillna(0) # just pretend there is no snow...
                assert swe.attrs['units'] == 'kg/m^2', swe.attrs['units']
                xvar = ref_var - swe.differentiate('time', datetime_unit='s')
                xvar = xvar.clip(min=0,max=None) # remove negative values
            else:
                raise NotImplementedError(varname)
                
            # define/copy metadata
            xvar.rename(varname)
            xvar.attrs = ref_var.attrs.copy()
            for att in ('name','units','long_name',):
                if att in ref_varatts: xvar.attrs[att] = ref_varatts[att]
            if 'original_name' in xvar.attrs: del xvar.attrs['original_name'] # does not apply
            xvar.attrs['note'] = note
            # set chunking for operation
            if lautoChunk:
                assert chunks is None, chunks 
                chunks = autoChunk(xvar.shape)
            if chunks: 
                xvar = xvar.chunk(chunks=chunks)
            print(xvar)
                
            # create a dataset for export to new file
            if ref_ds: 
                ds_attrs = ref_ds.attrs.copy()
                resampling = ds_attrs['resampling']
                xvar.attrs['note'] = resampling
                proj4_str = ds_attrs['proj4']
            else: 
                ds_attrs = dict(name='MergedForcing')
                resampling = None
                proj4_str = nrcan.attrs['proj4']
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
            if lexec:
                nds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                              encoding={varname:var_enc,}, compute=True)
            else:
                print(var_enc)

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
        
  
