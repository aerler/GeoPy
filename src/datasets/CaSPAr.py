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
# HRDPS
hrdps_varatts = dict(HRDPS_P_TT_10000 = dict(name='T2', units='K', offset=273.15, long_name='2 m Temperature'),
                     HRDPS_P_HU_10000 = dict(name='Q2', units='kg/kg', long_name='2 m Specific Humidity'),
                     HRDPS_P_UU_10000 = dict(name='U2', units='m/s', scalefactor=1852./3600., long_name='2 m Zonal Wind'),
                     HRDPS_P_VV_10000 = dict(name='V2', units='m/s', scalefactor=1852./3600., long_name='2 m Meridional Wind'),
                     HRDPS_P_GZ_10000 = dict(name='zs', units='m', scalefactor=10., long_name='Surface Geopotential'),
                     HRDPS_P_FB_SFC = dict(name='DNSW', units='W/m^2', long_name='Downward Solar Radiation'),
                     HRDPS_P_FI_SFC = dict(name='DNLW', units='W/m^2', long_name='Downward Longwave Radiation'),
                     HRDPS_P_PN_SFC = dict(name='mslp', units='Pa', scalefactor=100., long_name='Sea-level Pressure'),
                     HRDPS_P_P0_SFC = dict(name='ps', units='Pa', scalefactor=100., long_name='Surface Pressure'),
                     HRDPS_P_TM_SFC = dict(name='SST', units='K', long_name='Sea Surface Temperature'),
                     HRDPS_P_GL_SFC = dict(name='seaice', units='', long_name='Sea Ice Fraction'),
                     HRDPS_P_DN_SFC = dict(name='rho_snw', units='kg/m^3', scalefactor=1, long_name='Snow Density'),
                     HRDPS_P_SD_Avg = dict(name='snowh', units='m', scalefactor=0.01, long_name='Snow Depth'),
                     # derived variables
                     Rn     = dict(name='Rn',    units='W/m^2', long_name='Net Surface Radiation'),
                     e_def  = dict(name='e_def', units='Pa', long_name='Saturation Deficit'),
                     e_vap  = dict(name='e_vap', units='Pa', long_name='Water Vapor Pressure'),
                     RH     = dict(name='RH',    units='', long_name='Relative Humidity'),
                     delta  = dict(name='delta', units='Pa/K', long_name='Saturation Slope'),
                     u2     = dict(name='u2',    units='m/s', long_name='2 m Wind Speed'),
                     gamma  = dict(name='gamma', units='Pa/K', long_name='Psychometric Constant'),
                     pet_dgu = dict(name='pet_dgu', units='Pa/K', long_name='PET Denominator'),
                     pet_rad = dict(name='pet_rad', units='kg/m^2/s', long_name='PET Radiation Term'),
                     pet_wnd = dict(name='pet_wnd', units='kg/m^2/s', long_name='PET Wind Term'),
                     pet   = dict(name='pet', units='kg/m^2/s', long_name='Potential Evapotranspiration'),
                     snow  = dict(name='snow',  units='kg/m^2', long_name='Snow Water Equivalent'),) 
hrdps_varlist = caldas_varatts.keys()
hrdps_ignore_list = ['HRDPS_P_I2_SFC','HRDPS_P_SD_Glaciers','HRDPS_P_SD_Veg','HRDPS_P_SD_OpenWater','HRDPS_P_SD_IceWater', # from CaLDAS (snow); similar, but not identical
                     'HRDPS_P_PT_SFC','HRDPS_P_LA_SFC','HRDPS_P_LO_SFC', # empty variables
                     'HRDPS_P_FSF_SFC', 'HRDPS_P_FSD_SFC', # diffuse/direct radiation
                     'HRDPS_P_N0_SFC','HRDPS_P_RN_SFC','HRDPS_P_PR_SFC','HRDPS_P_AV_SFC',  # water fluxes
                     'HRDPS_P_TT_09950','HRDPS_P_HU_09950','HRDPS_P_GZ_09950', # upper levels (40m)
                     'HRDPS_P_VVC_09950','HRDPS_P_VV_09950','HRDPS_P_UU_09950','HRDPS_P_UUC_09950', # upper levels (40m)
                     'HRDPS_P_VVC_10000','HRDPS_P_UUC_10000'] # geographically corrected winds

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
                                        varatts=hrdps_varatts, ignore_list=hrdps_ignore_list),)
dataset_list = list(dataset_attributes.keys())
# N.B.: the effective start date for CaPA and all the rest is '2017-09-11T12'
default_dataset_index = dict(precip='CaPA', snow='CaLDAS')
for dataset,attributes in dataset_attributes.items():
    for varatts in attributes.varatts.values():
        varname = varatts['name']
        if varname not in default_dataset_index: 
            default_dataset_index[varname] = dataset 


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
    for varname,atts in ds_atts.varatts.items():
        if varname in xds: varmap[varname] = atts['name']  
    xds = xds.rename(varmap)
    # add projection
    if lgeoref:
        proj4_string = readCFCRS(xds, lraise=True, lproj4=True)
        xds = addGeoReference(xds, proj4_string=proj4_string)
    # return xarray Dataset
    return xds


def loadCaSPAr_6hourly(varname=None, varlist=None, dataset=None, dataset_index=None, folder=folder_6hourly, 
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
loadHourlyTimeSeries = loadCaSPAr_6hourly

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
#   modes += ['load_6hourly']
  modes += ['compute_variables']  
#   modes += ['load_raw']
#   modes += ['fix_dataset']
#   modes += ['test_georef']  

  # some settings  
  grid = 'lcc_snw'
  period = ('2017-09-11T12','2019-12-30T12')
  
  # loop over modes 
  for mode in modes:
    
                             
    if mode == 'load_6hourly':
       
  #       varlist = netcdf_varlist
        varlist = ['liqwatflx','precip','snow','test']
        xds = loadCaSPAr_6hourly(varlist=None, grid=grid, dataset='HRDPS', lignore_missing=True)
        print(xds)
        print('')
        for varname,xv in xds.variables.items(): 
            if xv.ndim == 3: break
        xv = xds[varname] # get DataArray instead of Variable object
        xv = xv.sel(time=slice('2018-01-01','2018-02-01'),x=slice(-3500,4500),y=slice(-1000,2000))
  #       xv = xv.loc['2011-01-01',:,:]
        print(xv)
        print(('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.)))
  
    
    elif mode == 'compute_variables':
       
        tic = time.time()
        
        # compute variable list
#         load_variables = dict(CaPA=['precip']); compute_variables = dict(CaPA=['precip'])
#         load_variables = dict(CaLDAS=['snowh','rho_snw']); compute_variables = dict(CaLDAS=['snow'])
#         load_variables = dict(CaLDAS=['snowh','rho_snw'], CaPA=['precip'])
#         compute_variables = dict(CaSPAr=['liqwatflx'])
        load_variables = dict(HRDPS=None) # all
        # HRDPS/PET variable lists
#         compute_variables = dict(HRDPS=['gamma','T2'])
#         compute_variables = dict(HRDPS=['Rn', 'e_def', 'delta', 'u2', 'gamma', 'T2']) # 'RH', # first order variables
#         compute_variables = dict(HRDPS=['pet_dgu', 'pet_rad', 'pet_wnd',]) # second order variables
        lderived = True
        derived_valist = ['Rn', 'e_def', 'delta', 'u2', 'gamma', 'T2', 'pet_dgu', 'pet_wnd', 'pet_rad']
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
                datasets[dataset] = loadCaSPAr_6hourly(grid=grid, varlist=derived_valist, 
                                                       dataset=dataset, lignore_missing=True)
            else:
                datasets[dataset] = loadCaSPAr_Raw(dataset=dataset, grid=grid, 
                                                   period=period, drop_variables=drop_variables)
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
                        nvar = 0.23*ref_ds['DNSW'] + ref_ds['DNLW']- 0.93*sig*ref_ds['T2']**4
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
        
  
    elif mode == 'load_raw':
       
        tic = time.time()
        xds = loadCaSPAr_Raw(dataset='HRDPS', 
#                              period=period, grid=grid,
                              grid='lcc_snw', #drop_variables=['confidence','test'],
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
        varname = 'time'
        if varname in xds:
            xv = xds[varname]
            print(xv)
            print("\nMean value:", xv[:].mean().values, xv.attrs['units'])
            print(('Size in Memory: {:6.1f} kB'.format(xv.nbytes/1024.)))
    
      
    elif mode == 'fix_dataset':
        
        lmissing = False # efault is to persist
        ref_delay = 1
#         dataset = 'CaPA'; lmissing = True # precip can just 'no happen'
#         dataset = 'CaLDAS'
        dataset = 'HRDPS'; ref_delay = 4 # diurnal cycle
        src_grid = 'snw_rotpol'
        ds_atts = dataset_attributes[dataset]        
        missing_value = np.NaN
        grid_mapping_list = ['rotated_pole']
        ref_varlen = None; ref_size = None
        damaged_folder = 'damaged_files/'
        
        folder = raw_folder.format(DS=dataset, GRD=src_grid)
        os.chdir(folder)
        with open('missing_files.txt',mode='a',newline='\n') as missing_record:
            # loop over dates
            date_list = pd.date_range(start=ds_atts.start_date,end=ds_atts.end_date,freq=ds_atts.interval)
            for i,date in enumerate(date_list):
                
                filename = netcdf_filename.format(Y=date.year,M=date.month,D=date.day,H=date.hour)
                # construct reference file
                ref_date = date_list[max(0,i-ref_delay)]
                reference_file = netcdf_filename.format(Y=ref_date.year,M=ref_date.month,D=ref_date.day,H=ref_date.hour)
                # identify damaged files by size (will be moved)
                if ref_size is None:
                    ref_size = os.path.getsize(filename)/2. # first file has to exist!                
                    with nc.Dataset(filename, mode='a') as ds:
                        ref_varlen = len(ds.variables)
                if osp.exists(filename) and os.path.getsize(filename) < ref_size:
                    # count variables
                    with nc.Dataset(filename, mode='a') as ds:
                        varlen = len(ds.variables)
                    if ref_varlen is None:
                        ref_varlen = varlen
                    elif ref_varlen < varlen:
                        raise ValueError("Additional variables detected in file '{}' - check reference file.".format(filename))
                    elif ref_varlen > varlen:
                        # move to separate folder
                        os.makedirs(damaged_folder, exist_ok=True)
                        print(filename,'->',damaged_folder)
                        shutil.move(filename, damaged_folder)                        
                # add missing (and damaged) files
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

        
        ## N.B.: in order to concatenate the entire time series of experimental and operational high-res CaPA data,
        #  we need to rename the variables in the experimental files using the following command (and a loop):
        #  ncrename -v .CaPA_fine_exp_A_PR_SFC,CaPA_fine_A_PR_SFC -v .CaPA_fine_exp_A_CFIA_SFC,CaPA_fine_A_CFIA_SFC $NC
  
    elif mode == 'test_georef':
      
        import osgeo
        print(osgeo.__version__)
#         from osgeo.osr import SpatialReference, CoordinateTransformation
        from pyproj import Proj, transform
      
        # load single time-step
        xds = loadCaSPAr_Raw(dataset='CaPA', grid=grid, period=period[1], 
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
    
