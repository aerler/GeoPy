'''
Created on Feb. 21, 2020

A module to load different CORDEX datasets.

@author: Andre R. Erler, GPL v3
'''

## wget command to download CORDEX simulations from UQAM server:
# for E in CCCma-CanESM2/ MPI-M-MPI-ESM-LR/ MPI-M-MPI-ESM-MR/ UQAM-GEMatm-Can-ESMsea/ UQAM-GEMatm-MPI-ESMsea/ UQAM-GEMatm-MPI LRsea/
#   do 
#     echo ''
#     echo "$E"
#     time wget -e robots=off -m --no-parent --no-check-certificate --reject="index.html*" --http-user=datasca --http-password=obelixa \ 
#           https://data.sca.uqam.ca/winger/CORDEX/NAM-22/UQAM/$E/historical/r1i1p1/UQAM-CRCM5/v1/mon/
#     echo ''
# done
# in short:
# 

# external imports
import datetime as dt
import pandas as pd
import os
import os.path as osp
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
# internal imports
from datasets.common import getRootFolder
from utils.misc import defaultNamedtuple
# for georeferencing
from geospatial.xarray_tools import addGeoReference, readCFCRS, updateVariableAttrs

## Meta-vardata

dataset_name = 'CORDEX'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='WRF') # get dataset root folder based on environment variables

# attributes of variables in different collections
# Axes and static variables
aux_varatts = dict(time = dict(name='time', units='hours', long_name='Days'), # time coordinate
                   lon = dict(name='lon', units='deg', long_name='Longitude'), # longitude coordinate
                   lat = dict(name='lat', units='deg', long_name='Latitude'), # latitude coordinate
                    x  = dict(name='x', units='m', long_name='Easting'),
                    y  = dict(name='y', units='m', long_name='Northing'),)
default_aux_varatts = aux_varatts
# variable attributes - should all follow CF convention
varatts = dict(pr = dict(name='precip', long_name='Total Precipitation'),
               tas = dict(name='T2', long_name='2m Temperature'), )
varmap = {att['name']:vn for vn,att in varatts.items()}
varlist = varmap.keys()
ignore_list = []
default_varatts = varatts; default_varmap = varmap; default_varlist = varlist; default_ignore_list = ignore_list

# netcdf settings
zlib_default = dict(zlib=True, complevel=1, shuffle=True)

# source data
folder_pattern = root_folder + '{DOM:s}/{INS:s}/{GCM:s}/{SCR:s}/{RIP:s}/{RCM:s}/{V:s}/{AGG:s}/{VAR:s}/' 
filename_pattern = '{DS:s}_{GRD:s}_{SCR:s}_{AGG:s}.nc' # original source file
station_dataset_subfolder = 'station_datasets/'

# list of available datasets/collections
dsnt_field_names = ['name','institute','GCM','realization','RCM','version','reanalysis',]
dsnt_defaults = dict(realization='r1i1p1', version='v1', reanalysis=False,)
DSNT = defaultNamedtuple(typename='Dataset', field_names=dsnt_field_names, defaults=dsnt_defaults)
dataset_attributes = {# UQAM CRCM5 datasets
                      'ERAI-CRCM5' : DSNT(institute='UQAM', GCM='ECMWF-ERAINT', RCM='UQAM-CRCM5', reanalysis=True, name='ERAI-CRCM5'),
                      'CanESM-CRCM5' : DSNT(institute='UQAM', GCM='CCCma-CanESM2', RCM='UQAM-CRCM5', name='CanESM-CRCM5'),
                      'MPILR-CRCM5' : DSNT(institute='UQAM', GCM='MPI-M-MPI-ESM-LR', RCM='UQAM-CRCM5', name='MPILR-CRCM5'),
                      'MPIESM-CRCM5' : DSNT(institute='UQAM', GCM='MPI-M-MPI-ESM-MR', RCM='UQAM-CRCM5', name='MPIESM-CRCM5'),
                      'CanESMsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-Can-ESMsea', RCM='UQAM-CRCM5', name='CanESMsea-CRCM5'),
                      'MPILRsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-MPILRsea', RCM='UQAM-CRCM5', name='MPILRsea-CRCM5'),
                      'MPIESMsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-MPI-ESMsea', RCM='UQAM-CRCM5', name='MPIESMsea-CRCM5'), }

dataset_list = list(dataset_attributes.keys())


## helper functions

def datasetAttributes(dataset, dataset_attributes=dataset_attributes, **kwargs):
    ''' helper function to retrieve and update dataset attributes '''
    if isinstance(dataset,str):
        ds_atts = dataset_attributes[dataset]
    elif isinstance(dataset,DSNT): ds_atts = dataset
    else:
        raise TypeError(dataset)
    for key,value in kwargs.items():
        if key in ds_atts.__class__.__dict__: ds_atts.__dict__[key] = value # set as instance attribute
    # return updated named tuple for dataset
    return ds_atts

def expandFolder(dataset, varname='', domain='NAM-22', aggregation='mon', scenario='historical', 
                 folder=folder_pattern, lraise=True, **kwargs):
    ''' helper function to expand dataset folder name with kwargs etc. '''
    # for convenience, we can resolve dataset here
    if isinstance(dataset, str): dataset = datasetAttributes(dataset, **kwargs)
    if dataset.reanalysis: scenario = 'evaluation' # only one option        
    # expand folder name
    folder = folder.format(DOM=domain, INS=dataset.institute, GCM=dataset.GCM, SCR=scenario, RIP=dataset.realization,
                           RCM=dataset.RCM, V=dataset.version, AGG=aggregation, VAR=varname, **kwargs)
    if lraise and not osp.exists(folder): raise IOError(folder)
    # return complete folder
    return folder

def expandFilename(dataset, grid=None, station=None, shape=None, bias_correction=None,
                   aggregation='mon', scenario='historical', filename=filename_pattern, **kwargs):
    ''' helper function to expand dataset filename with kwargs etc. '''
    # for convenience, we can resolve dataset here
    if isinstance(dataset, str): dataset = datasetAttributes(dataset, **kwargs)
    if dataset.reanalysis: scenario = 'evaluation' # only one option  
    # figure out grid string
    str_list = []
    # add parameters in correct order
    if bias_correction: str_list.append(bias_correction)
    if shape and station: raise ValueError(shape,station)
    elif station: str_list.append(station)
    elif shape: str_list.append(shape)
    if grid: str_list.append(grid)
    grd_str = '_'.join(str_list)
    # expand filename
    filename = filename.format(DS=dataset.name, GRD=grd_str, SCR=scenario, AGG=aggregation, **kwargs)
    # return complete filename
    return filename


## load functions

def loadCORDEX_RawVar(dataset=None, varname=None, folder=folder_pattern, lgeoref=True,  
                      domain='NAM-22', aggregation='mon', scenario='historical', 
                      lxarray=True, lcheck_files=True, lmultifile=None, filelist=None, 
                      drop_variables=None, varatts=None, aux_varatts=None, lraw=False, **kwargs):
    ''' function to load CORDEX data from NetCDF-4 files using xarray and add some projection information '''
    if not lxarray: 
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    if dataset is None: 
        raise ValueError('Please specify a dataset name ; valid datasets:\n',dataset_list)
    if isinstance(dataset,str):
        if dataset not in dataset_list: 
            raise ValueError("Dataset name '{}' not recognized; valid datasets:\n".format(dataset),dataset_list)
    elif not isinstance(dataset,DSNT):
        raise TypeError(dataset)
    if varname is None: 
        raise ValueError('Please specify a variable name (CF or local convention).')
    # figure out dataset attributes
    ds_atts = datasetAttributes(dataset, **kwargs)
    if ds_atts.reanalysis: scenario = 'evaluation' # only one option        
    # variable attributes
    if lraw:
        varatts = dict(); varmap = dict() # no translation
    elif varatts is None:
        varatts = default_varatts; varmap = default_varmap
    else:
        if 'name' in varatts: varmap = dict()
        else: varmap = {att['name']:vn for vn,att in varatts.items()}        
    if varname in varmap: varname = varmap[varname] # translate varname
    if 'name' in varatts: 
        varatts = varatts.copy() # just a regular varatts dict
        if varatts['name'] != varname: 
            raise ValueError("Inconsistent varname '{}' and varatts name '{}'.".format(varname,varatts['name']))
    elif varname in varatts: varatts = varatts[varname].copy() # assumed to be a dict of dicts
    else: varatts = dict()
    # construct file list
    if folder:
        folder = expandFolder(dataset=ds_atts, folder=folder, varname=varname, domain=domain, 
                              aggregation=aggregation, scenario=scenario, lraise=lcheck_files, **kwargs)
        if isinstance(filelist,(list,tuple)):
            filelist = [osp.join(folder,filename) for filename in filelist]
        elif isinstance(filelist,str): filelist = folder + '/' + filelist
        elif filelist is None: filelist = folder + '/*.nc'
    lraise = False
    if isinstance(filelist,(list,tuple)):
        for filename in filelist:
            if not osp.exists(filename):
                if not lraise: print("Missing files:")
                print(filename); lraise = True
    if lcheck_files and lraise:
        raise IOError("Some files appear to be missing... see above.")        
    # if folder is None but filelist is not, a list of absolute path is assumed
    if lmultifile is None:
        # auto-detect multi-file dataset
        if isinstance(filelist,(tuple,list)): 
            lmultifile = True # the only exception is a single date string as period
        else:
            # detect regex (assuming string)
            lmultifile = any([char in filelist for char in r'*?[]^'])
    # prepare drop list
    if drop_variables is None: drop_variables = default_ignore_list[:]
    drop_variables = [varmap.get(varname,varname) for varname in drop_variables]
    if lmultifile:
        # load multifile dataset (variables are in different files)
        xds = xr.open_mfdataset(filelist, combine='by_coords', concat_dim='time', join='right', parallel=True,   
                                data_vars='minimal', compat='override', coords='minimal', 
                                drop_variables=drop_variables , **kwargs)
    else:
        # load a single file/timestep
        if isinstance(filelist,(tuple,list)): filename = filelist[0]
        else: filename = filelist
        xds = xr.open_dataset(filename, drop_variables=drop_variables, **kwargs)
    # update attributes
    if not lraw:
        if aux_varatts is None: 
            merged_varatts = default_aux_varatts.copy()
            if 'time' in merged_varatts: del merged_varatts['time'] # handled by xarray
        else:
            merged_varatts = aux_varatts.copy()
        merged_varatts[varname] = varatts # add variable we are loading
        xds = updateVariableAttrs(xds, varatts=merged_varatts)
    # add projection
    if lgeoref:
        proj4_string = readCFCRS(xds, lraise=True, lproj4=True)
        xds = addGeoReference(xds, proj4_string=proj4_string)
    # return xarray Dataset
    return xds


def loadCORDEX_Raw(varlist=None, varname=None, dataset=None, folder=folder_pattern, lgeoref=True,  
                   domain='NAM-22', aggregation='mon', scenario='historical', 
                   lxarray=True, lcheck_files=True, merge_args=None, 
                   drop_variables=None, varatts=None, aux_varatts=None, lraw=False, **kwargs):
    ''' wrapper function to load multiple variables '''
    if varname and varlist: raise ValueError("Can only use either 'varlist' or 'varname'.")
    elif varname: varlist = [varname]
    # figure out dataset attributes
    ds_atts = datasetAttributes(dataset, **kwargs)
    if ds_atts.reanalysis: scenario = 'evaluation' # only one option            
    # figure out folder and potential varlist
    if folder:
        folder = expandFolder(dataset=ds_atts, folder=folder, varname='', domain=domain, 
                              aggregation=aggregation, scenario=scenario, lraise=lcheck_files, **kwargs)
        if varlist is None:
            varlist = []
            for varname in os.listdir(folder):
                if os.path.isdir(os.path.join(folder,varname)) and not varname.endswith('_datasets'):
                    varlist.append(varname)
        folder += '{VAR:s}/' # will be substituted in single-variable function
    # loop over variables
    ds_list = []
    for varname in varlist:
        # load variable
        var_ds = loadCORDEX_RawVar(dataset=ds_atts, varname=varname, folder=folder, lgeoref=False,  
                                   domain=domain, aggregation=aggregation, scenario=scenario, 
                                   lxarray=lxarray, lcheck_files=lcheck_files, drop_variables=drop_variables, 
                                   varatts=varatts, aux_varatts=aux_varatts, lraw=lraw, **kwargs)
        ds_list.append(var_ds)
    # add to merged dataset
    if merge_args is None: merge_args = dict(compat='override')
    xds = xr.merge(ds_list, **merge_args)
    # add meta data
    for att in dsnt_field_names:
        xds.attrs[att] = str(getattr(ds_atts,att))
    xds.attrs['domain'] = domain; xds.attrs['aggregation'] = aggregation; xds.attrs['scenario'] = scenario
    # add projection
    if lgeoref:
        proj4_string = readCFCRS(xds, lraise=True, lproj4=True)
        xds = addGeoReference(xds, proj4_string=proj4_string)
    # return xarray Dataset
    return xds
    

def loadCORDEX_TS(dataset=None, varlist=None, grid=None, station=None, shape=None, bias_correction=None,
                  dataset_subfolder=None, folder=folder_pattern, filename=filename_pattern,
                  domain='NAM-22', aggregation='mon', scenario='historical',
                  varatts=None, aux_varatts=None, lxarray=True, lgeoref=False, lraw=False, **kwargs):
    ''' function to load a consolidated CORDEX dataset '''
    # figure out dataset attributes
    ds_atts = datasetAttributes(dataset, **kwargs)
    if ds_atts.reanalysis: scenario = 'evaluation' # only one option            
    # figure out folder and potential varlist
    filename = expandFilename(dataset=dataset, grid=grid, station=station, bias_correction=bias_correction, 
                              shape=shape, aggregation=aggregation, scenario=scenario, filename=filename)
    if folder:
        folder = expandFolder(dataset=ds_atts, folder=folder, varname=dataset_subfolder, domain=domain, 
                              aggregation=aggregation, scenario=scenario, lraise=True, **kwargs)
        filepath = os.path.join(folder,filename)
    else: filepath = filename
    if not os.path.exists(filepath): 
        raise IOError(filepath)
        # variable attributes
    if lraw:
        varatts = dict(); varmap = dict() # no translation
    elif varatts is None:
        varatts = default_varatts.copy(); varmap = default_varmap.copy()
    else:
        varmap = {att['name']:vn for vn,att in varatts.items()}       
    if lxarray:
        ds = xr.open_dataset(filepath, **kwargs)
        # update attributes
        if not lraw:
            if aux_varatts is None: 
                merged_varatts = default_aux_varatts.copy()
                if 'time' in merged_varatts: del merged_varatts['time'] # handled by xarray
            else: merged_varatts = aux_varatts.copy()
            if varatts is None: merged_varatts.update(default_varatts)
            else: merged_varatts.update(varatts)
            ds = updateVariableAttrs(ds, varatts=merged_varatts)
    else:
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    # return formated dataset
    return ds

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
#   modes += ['load_timeseries']
#   modes = ['extract_timeseries']
  modes += ['load_raw']
#   modes += ['test_georef']  

  # some settings  
  grid = None
  domain = 'NAM-22'
  aggregation = 'mon'
  
  # more settings
  dataset = 'ERAI-CRCM5'
  scenario = 'evaluation'
  station_name = 'MLWC'
  
  # loop over modes 
  for mode in modes:
    
                             
    if mode == 'load_timeseries':
       
  #       varlist = netcdf_varlist
        varlist = ['liqwatflx','precip','snow','test']
        xds = loadCORDEX_TS(dataset=dataset, grid=station_name, varlist=None, 
                            dataset_subfolder=station_dataset_subfolder, 
                            domain=domain, aggregation=aggregation, scenario=scenario, lraw=False)
        print(xds)
        print('')
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
        xv = xv.sel(time=slice('2010-01-01','2011-02-01'),)
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
    
    
    elif mode == 'extract_timeseries':
        
        # coordinates of interest
        
        if station_name == 'MLWC':
            lat,lon = 57.45,-111.4
                
        for dataset in dataset_list:
          
            ds_atts = datasetAttributes(dataset)
            if ds_atts.reanalysis: scenarios = ('evaluation',)
            else: scenarios = ('historical','rcp85') 
            
            for scenario in scenarios:
                
                print("\n   ***   ",dataset,scenario,"   ***   \n")
                xds = loadCORDEX_Raw(dataset=dataset, varlist=None, 
                                     domain=domain, aggregation=aggregation, scenario=scenario,
                                     lgeoref=False, lraw=True)
                
                # compute closest grid point in model
                s = (xds.lat - lat)**2 + (xds.lon - lon)**2
                (j,i) = np.unravel_index(np.argmin(s.values.ravel()), shape=s.shape)
                print('Model I,J:',i,j)
                slat,slon = xds.lat.values[j,i],xds.lon.values[j,i]
                print('Lat/Lon: {:5.2f} / {:5.2f}'.format(slat,slon))
                
                # extract point
                sds = xds.isel(rlat=[j],rlon=[i]).squeeze()
                # add site/station data
                sds.attrs['station_name'] = station_name
                sds.attrs['stn_lat'] = slat
                sds.attrs['stn_lon'] = slon
                sds.attrs['stn_J'] = j
                sds.attrs['stn_I'] = i
                print('')
                print(sds)
                
                # save to NetCDF file
                station_folder = expandFolder(ds_atts, varname=station_dataset_subfolder, domain=domain, aggregation=aggregation, 
                                              scenario=scenario, folder=folder_pattern, lraise=False)
                print('')
                print(station_folder)
                os.makedirs(station_folder, exist_ok=True)
                station_file = filename_pattern.format(DS=dataset, GRD=station_name, SCR=scenario, AGG=aggregation)
                # write file
                encoding = {varname:zlib_default for varname,var in sds.variables.items()}
                sds.to_netcdf(os.path.join(station_folder,station_file), mode='w', 
                              format='NETCDF4', encoding=encoding, compute=True)
                print('')
    
    elif mode == 'load_raw':
       
        tic = time.time()
        xds = loadCORDEX_Raw(dataset=dataset, varlist=None, 
                             aggregation=aggregation, domain=domain, scenario=scenario,
                             lgeoref=False, lraw=False)
        toc = time.time()
        print(toc-tic)
        print(xds)
        print('')
        for name,var in xds.variables.items():
            print(name,var.attrs.get('long_name',None))
        print('')
        dt = xds['time'].diff(dim='time').values / np.timedelta64(1,'D')
        print('Time Delta (days):', dt.min(),dt.max())
        print('')
        print(xds.time)
#         xv = xds['CaPA_fine_exp_A_PR_SFC']
#         xv = xv.loc['2016-06-16T06',:,:]
#         varname = 'time'
#         if varname in xds:
#             xv = xds[varname]
#             print(xv)
#             print("\nMean value:", xv[:].mean().values, xv.attrs['units'])
#             print(('Size in Memory: {:6.1f} kB'.format(xv.nbytes/1024.)))

  
    elif mode == 'test_georef':
      
        import osgeo
        print(osgeo.__version__)
#         from osgeo.osr import SpatialReference, CoordinateTransformation
        from pyproj import Proj, transform
      
        # load single time-step
        xds = loadCORDEX_Raw(dataset='CaPA', grid=grid,  
                             lcheck_files=True, lgeoref=True)
        print(xds)
        
        ## test projection
#         CSR = SpatialReference()
        default_proj4 = "+proj=longlat +lon_0=0 +lat_0=0 +ellps=WGS84 +datum=WGS84"
#         CSR.ImportFromProj4(default_proj4)
        CSR = Proj(default_proj4)
        # coordinate indices
        i,j = 27,50
        y,x = xds.y.data[j],xds.x.data[i]
        pCSR = Proj(xds.attrs['proj4'])
        print("\nSource coordinates (y,x):\n                  {:8.5f}, {:8.5f}".format(y,x))
        # reproject source coordinates
        print("\n   reprojecting...")
#         transform = CoordinateTransformation(rCSR,CSR)
#         slat,slon,z = transform.TransformPoint(rlat.astype(np.float64),rlon.astype(np.float64))
        slon,slat = transform(pCSR, CSR, x, y, radians=False)
        print("\nReprojected coordinates (lat,lon):\n                  {:8.5f}, {:8.5f}".format(slat,slon))
        # compare against recorded coordinates
        lat,lon = xds.lat.data[j,i],xds.lon.data[j,i]
        print("\nActual coordinates:\n                  {:8.5f}, {:8.5f}".format(lat,lon))
    
