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
from geospatial.xarray_tools import addGeoReference, readCFCRS

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

# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/' 
avgfile   = '{DS:s}_{GRD:s}_clim_{PRD:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = '{DS:s}_{GRD:s}_monthly.nc' # extend with biascorrection, variable and grid type
folder_6hourly   = root_folder + dataset_name.lower()+'_6hourly/' 
filename_6hourly = '{DS:s}_{VAR:s}_{GRD:s}_6hourly.nc' # dataset and variable name, grid name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float

# source data
raw_folder = root_folder + '{DOM:s}/{INS:s}/{GCM:s}/{SCR:s}/{RIP:s}/{RCM:s}/{V:s}/{AGG:s}/{VAR:s}/' 
netcdf_filename = '{Y:04d}{M:02d}{D:02d}{H:02d}.nc' # original source file

# list of available datasets/collections
DSNT = defaultNamedtuple(typename='Dataset', field_names=['institute','GCM','realization','RCM','version','reanalysis'],
                         defaults=dict(realization='r1i1p1', version='v1', reanalysis=False))
dataset_attributes = {# UQAM CRCM5 datasets
                      'ERAI-CRCM5' : DSNT(institute='UQAM', GCM='ECMWF-ERAINT', RCM='UQAM-CRCM5', reanalysis=True),
                      'CanESM-CRCM5' : DSNT(institute='UQAM', GCM='CCCma-CanESM2', RCM='UQAM-CRCM5'),
                      'MPILR-CRCM5' : DSNT(institute='UQAM', GCM='MPI-M-MPI-ESM-LR', RCM='UQAM-CRCM5'),
                      'MPIESM-CRCM5' : DSNT(institute='UQAM', GCM='MPI-M-MPI-ESM-MR', RCM='UQAM-CRCM5'),
                      'CanESMsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-Can-ESMsea', RCM='UQAM-CRCM5'),
                      'MPILRsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-MPILRsea', RCM='UQAM-CRCM5'),
                      'MPIESMsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-MPI-ESMsea', RCM='UQAM-CRCM5'), }

dataset_list = list(dataset_attributes.keys())


## load functions

def loadCORDEX_Raw(dataset=None, varname=None, folder=raw_folder, lgeoref=True,  
                   domain='NAM-22', aggregation='mon', scenario='historical', 
                   lxarray=True, lcheck_files=True, lmultifile=None, filelist=None, 
                   drop_variables=None, varatts=None, aux_varatts=None, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    if not lxarray: 
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    if dataset is None: 
        raise ValueError('Please specify a dataset name ; valid datasets:\n',dataset_list)
    if dataset not in dataset_list: 
        raise ValueError("Dataset name '{}' not recognized; valid datasets:\n".format(dataset),dataset_list)
    if varname is None: 
        raise ValueError('Please specify a variable name (CF or local convention).')
    # set dataset attriutes
    ds_atts = dataset_attributes[dataset]
    for key,value in kwargs.items():
        if key in ds_atts.__class__.__dict__: ds_atts.__dict__[key] = value # set as instance attribute
    if ds_atts.reanalysis: scenario = 'evaluation' # only one option
    # variable attributes
    if varatts is None:
        varatts = default_varatts; varmap = default_varmap
    else:
        if 'name' in varatts: varmap = dict()
        else: varmap = {att['name']:vn for vn,att in varatts.itmes()}        
    if varname in varmap: varname = varmap[varname] # translate varname
    if 'name' in varatts: 
        varatts = varatts.copy() # just a regular varatts dict
        if varatts['name'] != varname: 
            raise ValueError("Inconsistent varname '{}' and varatts name '{}'.".format(varname,varatts['name']))
    elif varname in varatts: varatts = varatts[varname].copy() # assumed to be a dict of dicts
    else: varatts = dict()
    # construct file list
    if folder:
        folder = folder.format(DOM=domain, INS=ds_atts.institute, GCM=ds_atts.GCM, SCR=scenario, RIP=ds_atts.realization,
                               RCM=ds_atts.RCM, V=ds_atts.version, AGG=aggregation, VAR=varname, **kwargs)
        if not osp.exists(folder): raise IOError(folder)
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
        if 'lat' not in drop_variables: drop_variables.append('lat')
        if 'lon' not in drop_variables: drop_variables.append('lon')
        xds = xr.open_mfdataset(filelist, combine='by_coords', concat_dim='time', join='right', parallel=True,   
                                data_vars='minimal', compat='override', coords='minimal', 
                                drop_variables=drop_variables , **kwargs)
    else:
        # load a single file/timestep
        if isinstance(filelist,(tuple,list)): filename = filelist[0]
        else: filename = filelist
        xds = xr.open_dataset(filename, drop_variables=drop_variables, **kwargs)
    # update attributes using old names
    if aux_varatts is None: aux_varatts = default_aux_varatts.copy()
    varatts_list = tuple(aux_varatts.items())+((varname,varatts),)
    for varname,atts in varatts_list:
        if varname in xds.variables:
            var = xds.variables[varname]
            atts = atts.copy() # because we will pop scalefactor...
            if 'units' in atts:
              if 'units' not in var.attrs or var.attrs['units'] != atts['units']:
                if 'scalefactor' in atts and atts['scalefactor'] != 1:
                    var *= atts['scalefactor'] # this should execute lazily...
                if 'offset' in atts and atts['offset'] != 0:
                    var += atts['offset'] # this should execute lazily...
            atts.pop('scalefactor',None)
            attrs = var.attrs.copy()
            attrs.update(atts)
            var.attrs = attrs
    # actually rename
    varmap = dict()
    for varname,atts in varatts_list:
        if varname in xds: varmap[varname] = atts['name']  
    xds = xds.rename(varmap)
    # add projection
    if lgeoref:
        proj4_string = readCFCRS(xds, lraise=True, lproj4=True)
        xds = addGeoReference(xds, proj4_string=proj4_string)
    # return xarray Dataset
    return xds



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
  modes += ['load_raw']
#   modes += ['test_georef']  

  # some settings  
  grid = None
  
  # loop over modes 
  for mode in modes:
    
                             
    if mode == 'load_timeseries':
       
  #       varlist = netcdf_varlist
        varlist = ['liqwatflx','precip','snow','test']
        xds = NotImplemented
        print(xds)
        print('')
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
        xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
    
    elif mode == 'load_raw':
       
        tic = time.time()
        xds = loadCORDEX_Raw(dataset='ERAI-CRCM5', varname='precip', aggregation='mon',  
                             lxarray=True, lgeoref=False, lmultifile=None)
        toc = time.time()
        print(toc-tic)
        print(xds)
        print('')
        dt = xds['time'].diff(dim='time').values / np.timedelta64(1,'D')
        print('Time Delta (days):', dt.min(),dt.max())
#         xv = xds['CaPA_fine_exp_A_PR_SFC']
#         xv = xv.loc['2016-06-16T06',:,:]
        varname = 'time'
        if varname in xds:
            xv = xds[varname]
            print(xv)
            print("\nMean value:", xv[:].mean().values, xv.attrs['units'])
            print(('Size in Memory: {:6.1f} kB'.format(xv.nbytes/1024.)))
    
      
  
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
    
