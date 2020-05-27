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
from geospatial.xarray_tools import addGeoReference, readCFCRS, loadXArray
import geodata

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


## functions to load NetCDF datasets (using xarray)

def loadSnoDAS_Daily(varname=None, varlist=None, folder=None, grid=None, biascorrection=None, resampling=None,
                     lgeoref=True, chunks=None, time_chunks=8, geoargs=None, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    from datasets.SnoDAS import daily_folder, netcdf_varlist, netcdf_filename, netcdf_settings
    if folder is None: folder = daily_folder
    xds = loadXArray(varname=varname, varlist=varlist, folder=folder, grid=grid, biascorrection=biascorrection, resolution=None,
                      filename_pattern=netcdf_filename, default_varlist=netcdf_varlist, resampling=resampling, lgeoref=lgeoref, 
                      geoargs=geoargs, chunks=chunks, time_chunks=time_chunks, netcdf_settings=netcdf_settings, **kwargs)
    return xds


def loadNRCan_Daily(varname=None, varlist=None, folder=None, grid=None, resolution='CA12', resampling=None,
                    lgeoref=True, geoargs=None, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    from datasets.NRCan import daily_folder, netcdf_filename, netcdf_settings
    if folder is None: folder = daily_folder
    if resolution == 'CA12':
        from datasets.NRCan import day12_vardefs, day12_derived
        default_varlist = list(day12_derived) + [atts['name'] for atts in day12_vardefs.values()]
    xds = loadXArray(varname=varname, varlist=varlist, folder=folder, grid=grid, biascorrection=None, resolution=resolution,
                      filename_pattern=netcdf_filename, default_varlist=default_varlist, resampling=resampling, lgeoref=lgeoref, 
                      geoargs=geoargs, netcdf_settings=netcdf_settings, **kwargs)
    return xds


def loadMergedForcing_Daily(varname=None, varlist=None, dataset=None, dataset_index=None, folder=None, 
                            lignore_missing=False, grid=None, biascorrection=None, lxarray=True, time_chunks=None, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    if not lxarray: 
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname: varlist = [varname]
    elif varlist is None:
        if dataset is None:
            raise ValueError("Please specify a 'dataset' value in order to load a default variable list.\n"
                             "Supported datasets: {}".format(dataset_list))
        varlist = dataset_attributes[dataset].varatts.keys()
        lignore_missing = True
    # check dataset time intervals/timesteps
    if dataset_index is None: dataset_index = default_dataset_index.copy()
    for varname in varlist:
        ds_name = dataset_index.get(varname,'CaSPAr')
        interval = dataset_attributes[ds_name].interval
        if interval != '6H': 
            raise ValueError(varname,ds_name,interval)
    # load variables
    if biascorrection is None and 'resolution' in kwargs: biascorrection = kwargs['resolution'] # allow backdoor
    if len(varlist) == 1:
        varname = varlist[0]
        # load a single variable
        ds_name = dataset_index.get(varname,'CaSPAr') if dataset is None else dataset
        if biascorrection : ds_name = '{}_{}'.format(ds_name,biascorrection) # append bias correction method
        filepath = '{}/{}'.format(folder,filename_6hourly.format(DS=ds_name,VAR=varname, GRD=grid))
        xds = xr.open_dataset(filepath, **kwargs)
    else:
        # load multifile dataset (variables are in different files)
        filepaths = []
        for varname in varlist:
            ds_name = dataset_index.get(varname,'CaSPAr') if dataset is None else dataset
            if biascorrection : ds_name = '{}_{}'.format(ds_name,biascorrection) # append bias correction method
            filename = filename_6hourly.format(DS=ds_name,VAR=varname, GRD=grid)  
            filepath = '{}/{}'.format(folder,filename)
            if os.path.exists(filepath):
                filepaths.append('{}/{}'.format(folder,filename))
            elif not lignore_missing:
                raise IOError(filepath)
#         xds = xr.open_mfdataset(filepaths, combine='by_coords', concat_dim='time', join='right', parallel=True,   
#                                 data_vars='minimal', compat='override', coords='minimal', **kwargs)
        xds = xr.open_mfdataset(filepaths, combine='by_coords', concat_dim=None, parallel=True,   
                                data_vars='minimal', coords='minimal', join='inner', **kwargs)
        #xds = xr.merge([xr.open_dataset(fp, chunks=chunks, **kwargs) for fp in filepaths])    
    return xds
loadDailyTimeSeries = loadMergedForcing_Daily

## abuse for testing
if __name__ == '__main__':
  
  import dask, time, gc, shutil
  
  print('xarray version: '+xr.__version__+'\n')
        

#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=4, diagnostics_port=18787)
#   client = Client(cluster)

#   from multiprocessing.pool import ThreadPool
#   dask.set_options(pool=ThreadPool(4))

  modes = []
#   modes += ['print_grid']
#   modes += ['load_NRCan']
#   modes += ['load_Daily']
#   modes += ['compute_derived']  
  modes += ['compute_NRCan']

  # some settings
  grid = None
  grid = 'hd1' # small Quebec grid
  period = ('2011-01-01','2018-01-01')
  
  # loop over modes 
  for mode in modes:
    
    if mode == 'print_grid':
        
        from geodata.gdal import loadPickledGridDef
        griddef = loadPickledGridDef(grid='on1')
        print(griddef)
                             
    elif mode == 'load_Daily':
       
  #       varlist = netcdf_varlist
        varlist = ['liqwatflx',]
        xds = loadMergedForcing_Daily(varlist=None, grid=grid, dataset='HRDPS', lignore_missing=True)
        print(xds)
        print('')
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
        xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
    
    elif mode == 'load_NRCan':
       
        varlist = ['liqwatflx',]
        xds = loadNRCan_Daily(varname='precip', grid=None, )
        print(xds)
        print('')
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
        #xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
        xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
        
    elif mode == 'compute_NRCan':
      
        start = time.time()
        
        # settings
        from datasets.NRCan import day12_vardefs, day12_derived, varatts, daily_folder
        derived_varlist = day12_derived
        resolution = 'CA12'
        resampling = 'cubic_spline'
        grid = 'son2'
        ts_name = 'time_stamp'
        
        # load data
        load_varlist = [atts['name'] for atts in day12_vardefs.values()]
        xds = loadNRCan_Daily(varlist=load_varlist, grid=grid, resampling=resampling, resolution=resolution) # take all
        # optional slicing (time slicing completed below)
        start_date = None; end_date = None # auto-detect available data
#         start_date = '2011-01-01'; end_date = '2011-02-01'
        
        # slice and load time coordinate
        xds = xds.loc[{'time':slice(start_date,end_date),}]
        tsvar = xds[ts_name].load()
               
        print(xds)
        
        # loop over variables
        for varname in derived_varlist:

            # compute values 
            if varname == 'T2':
                ref_var = 'Tmax'      
                note = 'simple average of Tmin and Tmax'          
                xvar = xds['Tmin'] + xds['Tmax']
                xvar /= 2                
            else:
                raise NotImplementedError(varname)
                
            # define/copy metadata
            xvar.rename(varname)
            xvar.attrs = xds[ref_var].attrs.copy()
            for att in ('name','units','long_name',):
                if att in varatts: xvar.attrs[att] = varatts[att]
            if 'original_name' in xvar.attrs: del xvar.attrs['original_name'] # does not apply
            xvar.attrs['note'] = note
            #xvar.chunk(chunks=chunk_settings)
            print(xvar)
                
            # create a dataset for export to new file
            nds = xr.Dataset({ts_name:tsvar, varname:xvar,}, attrs=xds.attrs.copy())
            nds = addGeoReference(nds, proj4_string=xds.attrs['proj4'], )
            print('\n')
            print(nds)
            # define variable file path
            var_str = varname
            if grid: var_str += '_'+grid
            nc_filename = '{}_{}_{}_daily.nc'.format('NRCan',resolution,var_str).lower() # should be lower case
            nc_folder = daily_folder + '{}/{}/'.format(grid,resampling)
            nc_filepath = nc_folder + nc_filename
            print("\nExporting to new NetCDF-4 file:\n '{}'".format(nc_filepath))
            # write to NetCDF
            var_enc = dict(zlib=True, complevel=1, ) # _FillValue=-9999, chunksizes=netcdf_settings['chunksizes']
            nds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                          encoding={varname:var_enc,}, compute=True)

        # print timing
        end =  time.time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
      
        
    elif mode == 'compute_variables':
       
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
        
  
