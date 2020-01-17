'''
Created on Jan. 8, 2020

A module to load different datasets obtained from the Canadian Surface Prediction Archive (CaSPAr).

@author: Andre R. Erler, GPL v3
'''

## some CDO commands to process CaSPAr data
# 
# # get timing of commands (with loops)
# time bash -c 'command' # single quotes; execute in same folder
# 
# # ensemble mean for CaLDAS
# for NC in ensemble_members/*_000.nc; do C=${NC%_000.nc}; D=${C#*/}; echo $D; cdo ensmean ensemble_members/${D}_???.nc ${D}.nc; done
# 
# # rename precip variable for experimental CaPA period (before March 2018)
# for NC in *.nc; do echo $NC; ncrename -v .CaPA_fine_exp_A_PR_SFC,CaPA_fine_A_PR_SFC -v .CaPA_fine_exp_A_CFIA_SFC,CaPA_fine_A_CFIA_SFC $NC; done
# 
# # reproject rotated pole grid to custom grid 
# # the CDO grid definition for a Lambert conic conformal projection is stored here:
# lcc_snw_griddef.txt
# 
# # single execution (execute in source folder, write to target folder '../lcc_snw/')
# cdo remapbil,../lcc_snw/lcc_snw_griddef.txt data.nc ../lcc_snw/data.nc
# 
# # paralel execution using GNU Parallel (execute in source folder, write to target folder '../lcc_snw/')
# time ls -1 *.nc | parallel -j 6 --colsep '\t' cdo remapbil,../lcc_snw/lcc_snw_griddef.txt {1} ../lcc_snw/{1} :::: -


# external imports
import datetime as dt
import pandas as pd
import os, gzip
import os.path as osp
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
from warnings import warn
from collections import namedtuple
# internal imports
from geodata.misc import name_of_month, days_per_month
from utils.nctools import add_coord, add_var
from datasets.common import getRootFolder, loadObservations
from geodata.gdal import GridDefinition, addGDALtoDataset, grid_folder
from geodata.netcdf import DatasetNetCDF
# for georeferencing
from geospatial.xarray_tools import addGeoReference, readCFCRS

## Meta-vardata

dataset_name = 'CaSPAr'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='HGS') # get dataset root folder based on environment variables

# attributes of variables in different collections
# Axes and static variables
axes_varatts = dict(time = dict(name='time', units='hours', long_name='Days'), # time coordinate
                    lon = dict(name='lon', units='deg', long_name='Longitude'), # longitude coordinate
                    lat = dict(name='lat', units='deg', long_name='Latitude'), # latitude coordinate
                    x  = dict(name='x', units='m', long_name='Easting'),
                    y  = dict(name='x', units='m', long_name='Northing'),)
axes_varlist = axes_varatts.keys()
# CaSPAr (general/mixed variables)
caspar_varatts = dict(liqwatflx = dict(name='liqwatflx',units='kg/m^2/s',scalefactor=1., 
                                       long_name='Liquid Water Flux'),)
caspar_varlist = caspar_varatts.keys()
caspar_ignore_list = []
# CaPA
capa_varatts = dict(CaPA_fine_A_PR_SFC = dict(name='precip', units='kg/m^2/s', scalefactor=1000./(6*3600.), 
                                              long_name='Total Precipitation'),
                    CaPA_fine_A_CFIA_SFC = dict(name='confidence', units='', scalefactor=1,
                                                long_name='Confidence Index'),)
capa_varlist = capa_varatts.keys()
capa_ignore_list = ['CaPA_fine_A_CFIA_SFC']
# CaLDAS
caldas_varatts = dict(CaLDAS_P_DN_SFC = dict(name='rho_snw', units='kg/m^3', scalefactor=1, long_name='Snow Density'),
                      CaLDAS_A_SD_Avg = dict(name='snowh', units='m', scalefactor=0.01, long_name='Snow Depth'),
                      # derived variables
                      snow = dict(name='snow',  units='kg/m^2', scalefactor=1., long_name='Snow Water Equivalent'),)
caldas_varlist = caldas_varatts.keys()
caldas_ignore_list = ['CaLDAS_P_I2_SFC','CaLDAS_P_SD_Glacier','CaLDAS_A_SD_Veg','CaLDAS_P_SD_OpenWater','CaLDAS_P_SD_IceWater']

# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/' 
avgfile   = '{DS:s}_{GRD:s}_clim_{PRD:s}.nc' # the filename needs to be extended: biascorrection, grid and period
tsfile    = '{DS:s}_{GRD:s}_monthly.nc' # extend with biascorrection, variable and grid type
folder_6hourly   = root_folder + dataset_name.lower()+'_6hourly/' 
filename_6hourly = '{DS:s}_{VAR:s}_{GRD:s}_6hourly.nc' # dataset and variable name, grid name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float

# source data
raw_folder = root_folder + '{DS:s}/{GRD:s}/' 
netcdf_filename = '{Y:04d}{M:02d}{D:02d}{H:02d}.nc' # original source file

# list of available datasets/collections
dataset_list = ['CaPA','CaLDAS','HRDPS']
DSNT = namedtuple(typename='Dataset', 
                  field_names=['name','interval','start_date','end_date','varatts', 'ignore_list'])
end_date = '2019-12-30T12'
dataset_attributes = dict(CaSPAr = DSNT(name='CaSPAr',interval='6H', start_date='2017-09-11T12', end_date=end_date,
                                        varatts=caspar_varatts, ignore_list=caspar_ignore_list),                          
                          CaPA   = DSNT(name='CaPA',  interval='6H', start_date='2016-06-11T12', end_date=end_date,
                                        varatts=capa_varatts, ignore_list=capa_ignore_list),
                          CaLDAS = DSNT(name='CaLDAS',interval='6H', start_date='2017-05-23T00', end_date=end_date,
                                        varatts=caldas_varatts, ignore_list=caldas_ignore_list),
                          HRDPS  = DSNT(name='HRDPS', interval='6H', start_date='2017-05-22T00', end_date=end_date,
                                        varatts=NotImplemented, ignore_list=NotImplemented),)
# N.B.: the effective start date for CaPA and all the rest is '2017-09-11T12'


## load functions

def loadCaSPAr_Raw(dataset=None, filelist=None, folder=raw_folder, grid=None, period=None, biascorrection=None,
                   lxarray=True, lgeoref=True, lcheck_files=True, lmultifile=None, drop_variables='default', **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    if not lxarray: 
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    if biascorrection: 
        raise NotImplementedError("Bias correction is currently not supported: ",biascorrection)
    if dataset is None: 
        raise ValueError('Please specify a dataset name ; valid datasets:\n',dataset_list)
    if dataset not in dataset_list: 
        raise ValueError("Dataset name '{}' not recognized; valid datasets:\n".format(dataset),dataset_list)
    if not folder and not filelist:
        raise IOError("Specify either 'folder' or 'filelist' or both.")
    # handle date ranges (frequency based on dataset)
    ds_atts = dataset_attributes[dataset]
    if period:
        if isinstance(period,str): period = (period,)
        if len(period)==1: 
            period = period*2; lmultifile = False
        date_list = pd.date_range(start=period[0],end=period[1],freq=ds_atts.interval)
        filelist = [netcdf_filename.format(Y=date.year,M=date.month,D=date.day,H=date.hour) for date in date_list]
    # construct file list
    if folder:
        folder = folder.format(DS=dataset,GRD=grid)
        if not osp.exists(folder): raise IOError(folder)
        if isinstance(filelist,(list,tuple)):
            filelist = [osp.join(folder,filename) for filename in filelist]
        elif isinstance(filelist,str): filelist = folder + '/' + filelist
        elif filelist is None: filelist = folder + '/*.nc'
    if isinstance(filelist,(list,tuple)):
        lraise = False
        for filename in filelist:
            if not osp.exists(filename):
                if not lraise: print("Missing files:")
                print(filename); lraise = True
    if lcheck_files and lraise:
        raise IOError("Some files apprear to be missing... see above.")        
    # if folder is None but filelist is not, a list of absolute path is assumed
    if lmultifile is None:
        # auto-detect multi-file dataset
        if isinstance(filelist,(tuple,list)): 
            lmultifile = True # the only exception is a single date string as period
        else:
            # detect regex (assuming string)
            lmultifile = any([char in filelist for char in r'*?[]^'])
    # prepare drop list
    if drop_variables is None: drop_variables = []
    elif isinstance(drop_variables,str) and drop_variables.lower() == 'default': 
        drop_variables = ds_atts.ignore_list[:]
    ravmap = {atts['name']:varname for varname,atts in ds_atts.varatts.items()}
    drop_variables = [ravmap.get(varname,varname) for varname in drop_variables]
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
    for varname,atts in ds_atts.varatts.items():
        if varname in xds.variables:
            var = xds.variables[varname]
            atts = atts.copy() # because we will pop scalefactor...
            if var.attrs['units'] != atts['units']:
                if atts['scalefactor'] != 1:
                    var *= atts['scalefactor'] # this should execute lazily...
            atts.pop('scalefactor',None)
            attrs = var.attrs.copy()
            attrs.update(atts)
            var.attrs = attrs
    # actually rename
    varmap = dict()
    for varname,atts in ds_atts.varatts.items():
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
  
  #print('xarray version: '+xr.__version__+'\n')
  xr.set_options(keep_attrs=True)
        

#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=4, diagnostics_port=18787)
#   client = Client(cluster)

#   from multiprocessing.pool import ThreadPool
#   dask.set_options(pool=ThreadPool(4))

  modes = []
  modes += ['compute_variables']  
#   modes += ['load_raw']
#   modes += ['fix_dataset']
#   modes += ['test_georef']  
  
  
  # loop over modes 
  for mode in modes:
    
    if mode == 'compute_variables':
       
        tic = time.time()
        
        # compute variable list
#         load_variables = dict(CaPA=['precip']); compute_variables = dict(CaPA=['precip'])
#         load_variables = dict(CaLDAS=['snowh','rho_snw']); compute_variables = dict(CaLDAS=['snow'])
        load_variables = dict(CaLDAS=['snowh','rho_snw'], CaPA=['precip'])
        compute_variables = dict(CaSPAr=['liqwatflx'])
        drop_variables = 'default' # special keyword
        reference_dataset = next(iter(load_variables)) # just first dataset...
        
        # settings
        ts_name = 'time'
#         period = ('2019-11-11T12','2019-12-01T12')
        period = ('2017-09-11T12','2019-12-30T12')
        grid = 'lcc_snw'
        folder = folder_6hourly # CaSPAr/caspar_6hourly/
        
        # load multi-file dataset (no time slicing necessary)        
        datasets = dict()
        for dataset,varlist in load_variables.items():
            datasets[dataset] = loadCaSPAr_Raw(dataset=dataset, grid=grid, 
                                               period=period, drop_variables=drop_variables)
        ref_ds = datasets[reference_dataset]
        print(ref_ds)
        tsvar = ref_ds[ts_name].load()
#         print(tsvar)
        
        # loop over variables: compute and save to file
        for dataset,varlist in compute_variables.items():
            for varname in varlist:
              
                note = 'derived variable'
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
                elif dataset in datasets:
                    # generic operation
                    ref_ds = datasets[dataset]
                    if varname in ref_ds:
                        # generic copy
                        ref_var = ref_ds[varname]
                        nvar = ref_ds[varname].copy()
                    else:
                        raise NotImplementedError("Variable '{}' not found in dataset '{}'".fomat(varname,dataset))
                else:
                    raise NotImplementedError("No method to compute variable '{}' (dataset '{}'".fomat(varname,dataset))
                
                # assign attributes
                nvar.rename(varname)
                nvar.attrs = ref_var.attrs.copy()
                varatts = dataset_attributes[dataset].varatts[varname]
                for att in ('name','units','long_name',):
                    nvar.attrs[att] = varatts[att]
                nvar.attrs['note'] = note
                #nvar.chunk(chunks=chunk_settings)
                
                print('\n')
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
        print(toc-tic)
        
                             
    elif mode == 'load_raw':
       
        tic = time.time()
        xds = loadCaSPAr_Raw(dataset='CaLDAS', grid='lcc_snw', #drop_variables=['confidence','test'],
                             period=('2019-11-11T12','2019-12-01T12'), lcheck_files=True,
#                               filelist='2016??????.nc',
#                               period=('2018-03-03T00','2019-12-30T12'), lcheck_files=True,
#                               period=('2017-09-11T12','2019-12-30T12'), lcheck_files=True,
#                               period=('2016-06-11T12','2018-03-02T18'), lcheck_files=True,
#                               period=('2016-06-11T12','2017-09-11T06'), lcheck_files=True,
#                              filelist=['2017091518.nc','2017091600.nc','2017091606.nc','2017091612.nc'])
                             )
        toc = time.time()
        print(toc-tic)
        print(xds)
        print('')
        dt = xds['time'].diff(dim='time').values / np.timedelta64(1,'h')
        assert dt.min() == dt.max() == 6, (dt.min(),dt.max())
#         xv = xds['CaPA_fine_exp_A_PR_SFC']
#         xv = xv.loc['2016-06-16T06',:,:]
        varname = 'precip'
        if varname in xds:
            xv = xds[varname]
            print(xv)
            print("\nMean value:", xv[:].mean().values, xv.attrs['units'])
            print(('Size in Memory: {:6.1f} kB'.format(xv.nbytes/1024.)))
    
      
    elif mode == 'fix_dataset':
        
        dataset = 'CaPA' 
#         dataset = 'CaLDAS'
#         dataset = 'HRDPS'
        grid = 'snw_rotpol'
        ds_atts = dataset_attributes[dataset]
        lmissing = (dataset == 'CaPA') # for CaPA set to missing, for others persist
        missing_value = np.NaN
        grid_mapping_list = ['rotated_pole']
        reference_file = None
        
        folder = raw_folder.format(DS=dataset, GRD=grid)
        os.chdir(folder)
        with open('missing_files.txt',mode='a',newline='\n') as missing_record:
            # loop over dates
            date_list = pd.date_range(start=ds_atts.start_date,end=ds_atts.end_date,freq=ds_atts.interval)
            for date in date_list:
                
                filename = netcdf_filename.format(Y=date.year,M=date.month,D=date.day,H=date.hour)
                # add missing files
                if not osp.exists(filename):
                    # add to record
                    print(filename)
                    missing_record.write(filename+'\n')
                    # handle missing
                    shutil.copy(reference_file,filename) # create new file
                    # set values to missing
                    with nc.Dataset(filename, mode='a') as ds:
                        for varname,variable in ds.variables.items():
                            if varname == 'time':
                                # set time units to hours since present
                                time_str = 'hours since {Y:04d}-{M:02d}-{D:02d} {H:02d}:00:00'.format(Y=date.year,M=date.month,D=date.day,H=date.hour)
                                variable.setncattr('units',time_str)
                            elif lmissing and 'time' in variable.dimensions:
                                # set all values to a fill value
                                variable[:] = missing_value
                                variable.setncattr('missing_value',missing_value)
                        # set flag to indicate dummy nature of file
                        if lmissing:
                            ds.setncattr('DUMMY',"This file was recorded missing and a dummy was generated using the file " +
                                                 reference_file + "' as reference; values have been replaced by " + 
                                                 str(missing_value))
                        else:
                            ds.setncattr('DUMMY',"This file was recorded missing and a dummy was generated using the file " +
                                                 reference_file + "' as reference; values from reference persist")
                # make sure CF attributes are correct
                with nc.Dataset(filename, mode='a') as ds:
                    # rename CaPA experimental precip var
                    if dataset == 'CaPA':
                        if 'CaPA_fine_exp_A_PR_SFC' in ds.variables:
                            ds.renameVariable('CaPA_fine_exp_A_PR_SFC','CaPA_fine_A_PR_SFC')
                        if 'CaPA_fine_exp_A_CFIA_SFC' in ds.variables:
                            ds.renameVariable('CaPA_fine_exp_A_CFIA_SFC','CaPA_fine_A_CFIA_SFC')
                    # fix CF grid mappign attributes
                    grid_mapping = None
                    for grid_mapping in grid_mapping_list:
                        if grid_mapping in ds.variables: break
                    if grid_mapping is None:
                        raise NotImplementedError("No supported 'grid_mapping' detected.")
                    for varname,variable in ds.variables.items():
                        if 'rlat' in variable.dimensions and 'rlon' in variable.dimensions:
                            # verify or set CF projection attributes
                            if 'coordinates' in variable.ncattrs():
                                assert ( variable.getncattr('coordinates',) == 'lon lat' or 
                                         variable.getncattr('coordinates',) == 'lat lon' ), variable.getncattr('coordinates',)
                            else:
                                variable.setncattr('coordinates','lon lat')
                            if 'grid_mapping' in variable.ncattrs():
                                assert variable.getncattr('grid_mapping',) == grid_mapping, variable.getncattr('grid_mapping',)
                            else:
                                variable.setncattr('grid_mapping',grid_mapping)
                        elif varname == 'time':
                            if 'coordinates' in variable.ncattrs(): variable.delncattr('coordinates')
                            if 'grid_mapping' in variable.ncattrs(): variable.delncattr('grid_mapping')
                reference_file = filename # use file as reference for next step
        
        ## N.B.: in order to concatenate the entire time series of experimental and operational high-res CaPA data,
        #  we need to rename the variables in the experimental files using the following command (and a loop):
        #  ncrename -v .CaPA_fine_exp_A_PR_SFC,CaPA_fine_A_PR_SFC -v .CaPA_fine_exp_A_CFIA_SFC,CaPA_fine_A_CFIA_SFC $NC
  
    elif mode == 'test_georef':
      
        import osgeo
        print(osgeo.__version__)
#         from osgeo.osr import SpatialReference, CoordinateTransformation
        from pyproj import Proj, transform
      
        # load single time-step
        xds = loadCaSPAr_Raw(dataset='CaPA', grid='lcc_snw', period='2018-09-11T12', 
                             lcheck_files=True, lgeoref=True)
        print(xds)
#         # proj4 definition for rotated pole (does not work...)
#         RP = xds.rotated_pole
#         o_lon_p = RP.north_pole_grid_longitude #see https://trac.osgeo.org/gdal/ticket/4285
#         o_lat_p = RP.grid_north_pole_latitude 
#         lon_0 = 180. + RP.grid_north_pole_longitude
# #         if lon_0 > 180: lon_0 = lon_0 - 360. 
#         R = RP.earth_radius
#         proj4 = ("+proj=ob_tran +o_proj=longlat" +
# #                   " +to_meter=0.0174532925199 +a=1 " +
# #                   " +m 57.295779506" +
#                  " +o_lon_p={o_lon_p:f} +o_lat_p={o_lat_p:f}".format(o_lon_p=o_lon_p,o_lat_p=o_lat_p) +
#                  " +lon_0={lon_0:f} +R={R:f}".format(lon_0=lon_0, R=R) )
# #         proj4 = " +proj=ob_tran +o_proj=longlat +o_lat_p=90 +o_lon_p=0 +lon_0=0"
# #         proj4 = "+proj=longlat +lon_0=0 +lat_0=0 +ellps=WGS84 +datum=WGS84" # default
#         print(proj4)
#         rCSR = Proj(proj4)
#         print(rCSR.definition_string())
#         rCSR.ImportFromProj4(proj4)
        
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
    
