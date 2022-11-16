'''
Created on Feb. 21, 2020

A module to load different CORDEX datasets.

@author: Andre R. Erler, GPL v3
'''

## wget command to download CORDEX simulations from UQAM server:
# for E in CCCma-CanESM2/ MPI-M-MPI-ESM-LR/ MPI-M-MPI-ESM-MR/ UQAM-GEMatm-Can-ESMsea/ UQAM-GEMatm-MPI-ESMsea/ UQAM-GEMatm-MPILRsea/
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
import os
import os.path as osp
import numpy as np
import xarray as xr
import inspect
# internal imports
from geodata.base import Variable
from geodata.netcdf import DatasetNetCDF
from datasets.common import getRootFolder
from utils.misc import defaultNamedtuple
from plotting.properties import getPlotAtts
# for georeferencing
from geospatial.xarray_tools import addGeoReference, readCFCRS, updateVariableAttrs

## Meta-vardata

dataset_name = 'CORDEX'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='WRF') # get dataset root folder based on environment variables

# variable attributes - should all follow CF convention
varatts = dict(# CF compliant variables
               pr  = dict(name='precip', units='kg/m^2/s', long_name='Total Precipitation'),
               snw = dict(name='snow', units='kg/m^2', long_name='Snow Water Equivalent'),
               tas = dict(name='T2', units='K', long_name='2m Temperature'),
               tasmin = dict(name='Tmin', units='K', long_name='Minimum 2m Temperature'),
               tasmax = dict(name='Tmax', units='K', long_name='Maximum 2m Temperature'),
               psl = dict(name='pmsl', units='Pa', long_name='Mean-Sea-Level Pressure'),
               huss = dict(name='q2', units='kg/kg', long_name='2m Specific Humidity'), # mass fraction of water in (moist) air
               rsds = dict(name='SWDNB', units='W/m^2', long_name='Downwelling Solar Radiation'),
               rsus = dict(name='SWUPB', units='W/m^2', long_name='Upwelling Solar Radiation'),
               rlds = dict(name='LWDNB', units='W/m^2', long_name='Downwelling Longwave Radiation'),
               rlus = dict(name='LWUPB', units='W/m^2', long_name='Upwelling Longwave Radiation'),
               sfcWind = dict(name='U10', units='m/s', long_name='10m Wind Speed'),
               # axes and static variables               
               orog = dict(name='zs', units='m', long_name='Surface Elevation'),
               time = dict(name='time', long_name='Time Coordinate'), # time coordinate (keep original units)
               lon = dict(name='lon', units='deg', long_name='Longitude'), # longitude coordinate
               lat = dict(name='lat', units='deg', long_name='Latitude'), # latitude coordinate
               x   = dict(name='x', units='m', long_name='Easting'),
               y   = dict(name='y', units='m', long_name='Northing'),)
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
                      'MPIESMsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-MPI-ESMsea', RCM='UQAM-CRCM5', name='MPIESMsea-CRCM5'),
                      'MPILRsea-CRCM5' : DSNT(institute='UQAM', GCM='UQAM-GEMatm-MPILRsea', RCM='UQAM-CRCM5', name='MPILRsea-CRCM5'),}

dataset_list = list(dataset_attributes.keys())[:6]
# N.B.: the MPILRsea-CRCM5 appears to be incomplete


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
                      drop_variables=None, varatts=None, lraw=False, **kwargs):
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
        varatts = default_varatts.copy(); varmap = default_varmap.copy()
    elif 'name' in varatts: 
        varatts = default_varatts.copy(); varmap = default_varmap.copy()
        varmap[varname] = varatts['name']
        varatts[varname] = varatts.copy()
    else:
        varmap = {att['name']:vn for vn,att in varatts.items()}
        varatts = varatts.copy() # just a regular varatts dict
    # apply varmap
    varname = varmap.get(varname,varname)
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
        if 'time' in varatts and lxarray: del varatts['time'] # handled by xarray
        xds = updateVariableAttrs(xds, varatts=varatts)
    # add projection
    if lgeoref:
        proj4_string = readCFCRS(xds, lraise=True, lproj4=True)
        xds = addGeoReference(xds, proj4_string=proj4_string)
    # return xarray Dataset
    return xds


def loadCORDEX_Raw(varlist=None, varname=None, dataset=None, folder=folder_pattern, lgeoref=True,  
                   domain='NAM-22', aggregation='mon', scenario='historical', lconst=True,
                   lxarray=True, lcheck_files=True, merge_args=None, lsqueeze=True,
                   drop_variables=None, varatts=None, aux_varatts=None, lraw=False, **kwargs):
    ''' wrapper function to load multiple variables '''
    if varname and varlist: raise ValueError("Can only use either 'varlist' or 'varname'.")
    elif varname: varlist = [varname]
    # figure out dataset attributes
    ds_atts = datasetAttributes(dataset, **kwargs)
    if ds_atts.reanalysis: scenario = 'evaluation' # only one option            
    # load variables in individual datasets and merge later
    ds_list = []
    # load constants/fixed fields
    if lconst:
        fixfolder = expandFolder(dataset=ds_atts, folder=folder, varname='', domain=domain, 
                                 aggregation='fx', scenario=scenario, lraise=lcheck_files, **kwargs)
        fixlist = []
        # load all fixed fields
        for varname in os.listdir(fixfolder):
            if osp.isdir(osp.join(fixfolder,varname)) and not varname.endswith('_datasets'):
                fixlist.append(varname)
        fixfolder += '{VAR:s}/' # will be substituted in single-variable function
        # loop over fixed variables
        for fixvar in fixlist:
            # load fixed variable
            fix_ds = loadCORDEX_RawVar(dataset=ds_atts, varname=fixvar, folder=fixfolder, lgeoref=False,  
                                       domain=domain, aggregation='fx', scenario=scenario, 
                                       lxarray=lxarray, lcheck_files=lcheck_files, drop_variables=drop_variables, 
                                       varatts=varatts, lraw=lraw, **kwargs)
            if lsqueeze: fix_ds = fix_ds.squeeze(drop=True) # drop: don't want a meaningless time variable
            ds_list.append(fix_ds)
    # figure out folder and potential varlist
    if folder:
        folder = expandFolder(dataset=ds_atts, folder=folder, varname='', domain=domain, 
                              aggregation=aggregation, scenario=scenario, lraise=lcheck_files, **kwargs)
        if varlist is None:
            varlist = []
            for varname in os.listdir(folder):
                if osp.isdir(osp.join(folder,varname)) and not varname.endswith('_datasets'):
                    varlist.append(varname)
        folder += '{VAR:s}/' # will be substituted in single-variable function
    # loop over variables
    for varname in varlist:
        # load variable
        var_ds = loadCORDEX_RawVar(dataset=ds_atts, varname=varname, folder=folder, lgeoref=False,  
                                   domain=domain, aggregation=aggregation, scenario=scenario, 
                                   lxarray=lxarray, lcheck_files=lcheck_files, drop_variables=drop_variables, 
                                   varatts=varatts, lraw=lraw, **kwargs)
        if lsqueeze: var_ds = var_ds.squeeze()
        ds_list.append(var_ds) # remove singleton dims 
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
                  domain='NAM-22', aggregation='mon', scenario='historical', lfixTime=True, datum_year=1979,
                  varatts=None, lxarray=True, lgeoref=False, lraw=False, lscalars=True, **kwargs):
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
    if lraw: varatts = dict()
    elif varatts is None: varatts = default_varatts.copy()
    else: varatts = varatts.copy()
    # load dataset
    if lxarray:
        # fix arguments to prevent errors (no kwargs)
        argspec, varargs, keywords = inspect.getargs(xr.open_dataset.__code__); del varargs, keywords
        kwargs = {key:value for key,value in kwargs.items() if key in argspec}
        # load dataset using simple single file mode
        ds = xr.open_dataset(filepath, **kwargs)
        # update attributes
        if not lraw:
            ds = updateVariableAttrs(ds, varlist=varlist, varatts=varatts)
        # add projection
        if lgeoref:
            proj4_string = readCFCRS(xds, lraise=True, lproj4=True)
            ds = addGeoReference(xds, proj4_string=proj4_string)
    else:
        ds = DatasetNetCDF(name=dataset, filelist=[filepath], varlist=varlist, varatts=varatts,
                           lscalars=lscalars, **kwargs)
        if lfixTime:
            tax = ds.time
            tunits = 'months since {:04d}-01'.format(datum_year)
            if tunits == tax.units: 
                print("Time Axis '{}' has already been converted to '{}'.".format(tax.name,tax.units))
            elif 'days since 1949-12-01' in tax.units:
                # create varaible with old data
                ds += Variable(name='time_in_days', units=tax.units, axes=(tax,), 
                               data=tax.data_array, atts=tax.atts.copy())
                # fix time axis
                tax.data_array = np.floor( 12*( (tax[:]-30) / 365.2425 - (datum_year-1950) ) )
                tax.units = tunits
                tax.plot = getPlotAtts(name='time', units=tunits, atts=tax.atts,)
            else:
                raise NotImplementedError("Non-standard Datum: '{}'".format(ds.time.units))
        if lgeoref: 
            raise NotImplementedError
    # return formated dataset
    return ds

def loadCORDEX_StnTS(dataset=None, varlist=None, station=None, bias_correction=None, folder=folder_pattern, 
                     dataset_subfolder=station_dataset_subfolder, filename=filename_pattern,
                     domain='NAM-22', aggregation='mon', scenario='historical', load=True, 
                     lfixTime=True, datum_year=1979,
                     varatts=None, lxarray=True, lraw=False, lscalars=True, **kwargs):
    ''' wrapper function to load station/point datasets ''' 
    return loadCORDEX_TS(dataset=dataset, varlist=varlist, station=station, bias_correction=bias_correction,
                  dataset_subfolder=dataset_subfolder, folder=folder, filename=filename,
                  domain=domain, aggregation=aggregation, scenario=scenario, load=load, 
                  lfixTime=lfixTime, datum_year=datum_year,
                  varatts=varatts, lxarray=lxarray, lraw=lraw, lscalars=lscalars, **kwargs)



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
#   modes += ['test_timefix']
#   modes += ['compute_forcing']
#   modes += ['load_timeseries']
  modes = ['extract_timeseries']
#   modes += ['load_raw']
#   modes += ['test_georef']  

  # some settings  
  grid = None
  domain = 'NAM-22'
  aggregation = 'mon'
  
  # more settings
  dataset = 'ERAI-CRCM5'
  scenario = 'evaluation'
#   station_name = 'MLWC'
#   station_name = 'FortMcMurray'
  station_name = 'Bitumont'
  
  # loop over modes 
  for mode in modes:
    

    if mode == 'test_timefix':
       
        dataset = dataset_list[3]; scenario = 'historical'; datum_year = 1979
        varlist = ['precip','pet_th']; lxarray = False
        ds = loadCORDEX_StnTS(dataset=dataset, station=station_name, varlist=varlist, lxarray=lxarray,
                              lfixTime=True, datum_year=datum_year, 
                              domain=domain, aggregation=aggregation, scenario=scenario, lraw=False)
        print(ds)
        print('')
        if not lxarray:
            print(ds.time[:12])
            assert datum_year == 1979, datum_year
            assert scenario == 'historical', scenario
            if dataset == 'ERAI-CRCM5': 
                assert ds.time[0] == 0, ds.time[:12]
            elif len(ds.time) == 672:
                assert ds.time[0] == -29*12, ds.time[:12] 
            elif len(ds.time) == 684:
                assert ds.time[0] == -30*12, ds.time[:12] 
            dt = np.diff(ds.time[:])
            assert dt.min()==1 and dt.max()==1, (dt.min(),dt.max())
    
    elif mode == 'compute_forcing':
        
        # settings
        loverwrite = True
        pet_varlist = ['T2','Tmin','Tmax','pmsl','zs','q2','U10','SWDNB','SWUPB','LWDNB','LWUPB']
        lwf_varlist = ['snow','precip','time_in_days']
        new_varlist = ['pet','petrad','petwnd','pet_th','pet_har','pet_hog','liqwatflx']
      
        print(dataset_list)      
        for dataset in dataset_list:
          
            ds_atts = datasetAttributes(dataset)
            if ds_atts.reanalysis: scenarios = ('evaluation',)
            else: scenarios = ('historical','rcp85') 
            
            for scenario in scenarios:

                print("\n   ***   ",dataset,scenario,"   ***   \n")
       
                # load data using GeoPy/NetCDF4
                ds = loadCORDEX_TS(dataset=dataset, grid=station_name, 
                                   varlist=lwf_varlist+pet_varlist+new_varlist, 
                                   lxarray=False, load=True, mode='rw',
                                   dataset_subfolder=station_dataset_subfolder, 
                                   domain=domain, aggregation=aggregation, scenario=scenario, lraw=False,)
                #print(ds,'\n\n')
                
                if loverwrite or any([varname not in ds for varname in ('pet','petrad','petwnd')]):
                    print("    adding PM PET ('pet') and radiation & wind terms ('petrad' & 'petwnd')")        
                    from processing.newvars import computePotEvapPM
                    # compute PET
                    pet,rad,wnd = computePotEvapPM(ds, lterms=True, lmeans=True, lrad=True, 
                                                   lgrdflx=False, lpmsl=True)
                    pet.data_array = np.clip(pet.data_array, a_min=0, a_max=None) # remove negative PET
                    pet.atts.long_name = 'Potential Evapotranspiration'
                    rad.atts.long_name = 'Radiation Term of PET'
                    wnd.atts.long_name = 'Wind Term of PET'
                    # add to dataset
                    if loverwrite or 'pet' not in ds: ds.addVariable(pet, asNC=True, copy=True, loverwrite=loverwrite)
                    if loverwrite or 'petrad' not in ds: ds.addVariable(rad, asNC=True, copy=True, loverwrite=loverwrite)
                    if loverwrite or 'petwnd' not in ds: ds.addVariable(wnd, asNC=True, copy=True, loverwrite=loverwrite)
                                    
                ## add simplified PET methods
                pet_methods = ['PT', 'Hog', 'Har', 'Th' ]
                if 'note' in ds.time.atts and 'original calendar' in ds.time.atts['note']:
                    l365 = ( '365_day' in ds.time.atts['note'] )
                else: l365 = None
                pet_options = dict(lat=ds.atts.stn_lat, l365=l365, lAllen=False, lgrdflx=False,
                                   climT2=None, time_offset=0, p='center')
                import processing.newvars as pet_mod
                # Priestley-Taylor 1972, Hogg 1997, Hargreaves 1985, Thronthwaite 1948
                for pet_method in pet_methods:
                    varname = 'pet_' + pet_method.lower()
                    if loverwrite or varname not in ds:
                        print("    adding",pet_method,"PET","('{}')".format(varname))        
                        function_name = 'computePotEvap'+pet_method
                        pet_fct = getattr(pet_mod, function_name)
                        # compute PET
                        pet = pet_fct(ds, **pet_options)
                        # add to dataset
                        ds.addVariable(pet, asNC=True, copy=True, loverwrite=loverwrite)

                # compute liquid water flux
                if loverwrite or 'liqwatflx' not in ds:
                    print("    adding Liquid Water Flux ('liqwatflx')")
                    from geodata.base import Variable
                    dswe = np.gradient(ds.snow.data_array, axis=None)
                    dt = np.gradient(ds.time_in_days.data_array, axis=None)*86400.
                    assert 'days' in ds.time_in_days.units, ds.time
                    data = np.clip(ds.precip.data_array - dswe/dt, a_min=0, a_max=None)
                    lwf = Variable(name='liqwatflx', units=ds.precip.units, axes=ds.precip.axes, data=data, 
                                   long_name='Liquid Water Flux')
                    # add to dataset
                    #print(lwf,'\n\n')        
                    ds.addVariable(lwf, asNC=True, copy=True, loverwrite=loverwrite)
                
                # save dataset
                ds.sync()
                print(ds)
                ds.close()
                
    
    elif mode == 'load_timeseries':
       
        varlist = ['precip','T2','tasmax','tasmin']; lxarray = False
        #dataset = 'CanESM-CRCM5'; scenario = 'historical'; lxarray = True
        dataset = 'MPILRsea-CRCM5'; scenario = 'rcp85'
        xds = loadCORDEX_TS(dataset=dataset, grid=station_name, varlist=varlist, lxarray=lxarray, 
                            dataset_subfolder=station_dataset_subfolder, 
                            domain=domain, aggregation=aggregation, scenario=scenario, lraw=False)
        print(xds)
        print('')
        if lxarray:
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
        elif station_name == 'FortMcMurray':
            lat,lon = 56.65,-111.22
        elif station_name == 'Bitumont':
            lat,lon = 57.37,-11.53
                
        for dataset in dataset_list:
          
            ds_atts = datasetAttributes(dataset)
            if ds_atts.reanalysis: scenarios = ('evaluation',)
            else: scenarios = ('historical','rcp85') 
            
            for scenario in scenarios:
                
                print("\n   ***   ",dataset,scenario,"   ***   \n")
                
                # load raw data using xarray
                xds = loadCORDEX_Raw(dataset=dataset, varlist=None, decode_times=False,
                                     domain=domain, aggregation=aggregation, scenario=scenario,
                                     lgeoref=False, lraw=True, lsqueeze=True)
                # N.B.: decode_times=False is necessary, so that the dataset can be written back to
                #       a NetCDF file; for some reason writing from CFTime objects does not work
                
                if xds['time'].attrs['calendar'] == '365_day':
                    time = xds['time']
                    time.attrs['note'] = "original calendar: " + time.attrs['calendar']
                    del time.attrs['calendar']
                    time.values *= 365.2425/365. # correct for missing leap years
                    # this correction is sufficient to prevent drift in monthly data, but not for daily 
                
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
        xds = loadCORDEX_Raw(dataset=dataset, varlist=None, #['precip','T2','tasmax','tasmin'], 
                             aggregation=aggregation, domain=domain, scenario=scenario,
                             lgeoref=False, lraw=False, lconst=True, lsqueeze=True)
        toc = time.time()
        print(toc-tic)
        print(xds)
        print('')
        for name,var in xds.variables.items():
            print(name,var.attrs.get('long_name',None), var.attrs.get('units',None))
        print('')
        dt = xds['time'].diff(dim='time').values / np.timedelta64(1,'D')
        print('Time Delta (days):', dt.min(),dt.max())
        print(xds['time'].values[:10])
        print('')
        print(xds.zs)

  
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
#         slat,slon,z = transform.TransformPoint(rlon.astype(np.float64),rlat.astype(np.float64)) # order changed in GDAL 3
        slon,slat = transform(pCSR, CSR, x, y, radians=False)
        print("\nReprojected coordinates (lat,lon):\n                  {:8.5f}, {:8.5f}".format(slat,slon))
        # compare against recorded coordinates
        lat,lon = xds.lat.data[j,i],xds.lon.data[j,i]
        print("\nActual coordinates:\n                  {:8.5f}, {:8.5f}".format(lat,lon))
    
