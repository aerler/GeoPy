'''
Created on Nov. 25, 2018

A module to read SnoDAS data; this includes reading SnoDAS binary files and conversion to NetCDF-4, 
as well as functions to load the converted and aggregated data.

Changes in SnoDAS data:
  17/02/2010  snowmelt time aggregation label changes
  18/11/2010  Canadian Prairies and Great Lakes are assimilated
  20/08/2012  eastern Canada and Quebec are fully assimilated
  01/10/2013  grid is slightly shifted east (can be ignored)

@author: Andre R. Erler, GPL v3
'''

# external imports
import datetime as dt
import pandas as pd
import os, gzip
import os.path as osp
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
from warnings import warn
# try: import cPickle as pickle
# except: import pickle

# internal imports
from geodata.misc import name_of_month, days_per_month
from utils.nctools import add_coord, add_var
from datasets.common import getRootFolder, loadObservations
from geodata.gdal import GridDefinition
# from geodata.gdal import loadPickledGridDef, addGDALtoDataset
# from geodata.gdal import grid_folder as common_grid_folder


## HGS Meta-vardata

dataset_name = 'SnoDAS'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='HGS') # get dataset root folder based on environment variables

# SnoDAS grid definition
projdict = dict(proj='longlat',lon_0=0,lat_0=0,x_0=0,y_0=0) # wraps at dateline
proj4_string = '+proj=longlat +ellps=WGS84 +datum=WGS84 +lon_0=0 +lat_0=0 +x_0=0 +y_0=0 +name=SnoDAS +no_defs'
# N.B.: this is the new grid (in use since Oct. 2013); the old grid was slightly shifted, but less than 1km
geotransform = (-130.516666666667-0.00416666666666052, 0.00833333333333333, 0, 
                 24.1000000000000-0.00416666666666052, 0, 0.00833333333333333)
size = (8192,4096) # (x,y) map size of SnoDAS grid
# make GridDefinition instance
SnoDAS_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform, size=size)

# attributes of variables in SnoDAS binary files
binary_varatts = dict(# forcing variables (offset_code='P001', downscale_code='S', type_code='v0')
                      liqprec = dict(name='liqprec',units='kg/m^2/day',scalefactor=0.1, product_code='1025',v_code='lL00',t_code='T0024',description='24 hour total, 06:00 UTC-06:00 UTC',variable_type='Driving',long_name='Liquid Precipitation'),
                      solprec = dict(name='solprec',units='kg/m^2/day',scalefactor=0.1, product_code='1025',v_code='lL01',t_code='T0024',description='24 hour total, 06:00 UTC-06:00 UTC',variable_type='Driving',long_name='Solid Precipitation'),
                      # state variables (offset_code='P001', type_code='v1')
                      snow  = dict(name='snow',  units='m', scalefactor=1e-3, product_code='1034',v_code='tS__',t_code='T0001',description='Snapshot at 06:00 UTC',variable_type='state',long_name='Snow Water Equivalent'),
                      snowh = dict(name='snowh', units='m', scalefactor=1e-3, product_code='1036',v_code='tS__',t_code='T0001',description='Snapshot at 06:00 UTC',variable_type='state',long_name='Snow Water Equivalent'),
                      Tsnow = dict(name='Tsnow', units='K', scalefactor=1,    product_code='1038',v_code='wS__',t_code='A0024',description='',variable_type='State',long_name='Snow Pack Average Temperature'),
                      # diagnostic variables (offset_code='P000', type_code='v1')
                      snwmlt    = dict(name='snwmlt',   units='m/day',scalefactor=1e-5,  product_code='1044',v_code='bS__',t_code='T0024',description='Total of 24 per hour melt rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Snow Melt Runoff at the Base of the Snow Pack'),
                      evap_snow = dict(name='evap_snow',units='m/day',scalefactor=-1e-5, product_code='1050',v_code='lL00',t_code='T0024',description='Total of 24 per hour sublimation rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Sublimation from the Snow Pack'),
                      evap_blow = dict(name='evap_blow',units='m/day',scalefactor=-1e-5, product_code='1039',v_code='lL00',t_code='T0024',description='Total of 24 per hour sublimation rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Sublimation of Blowing Snow'),
                      )
# attributes of variables for converted NetCDF files
netcdf_varatts = dict(# forcing variables
                      liqprec = dict(name='liqprec',units='kg/m^2/s',scalefactor=1./86400., long_name='Liquid Precipitation'),
                      solprec = dict(name='solprec',units='kg/m^2/s',scalefactor=1./86400., long_name='Solid Precipitation'),
                      # state variables
                      snow  = dict(name='snow',  units='kg/m^2', scalefactor=1.e3, long_name='Snow Water Equivalent'),
                      snowh = dict(name='snowh', units='m',      scalefactor=1.,   long_name='Snow Height/Depth'),
                      Tsnow = dict(name='Tsnow', units='K',      scalefactor=1.,   long_name='Snow Pack Average Temperature'),
                      # diagnostic variables
                      snwmlt    = dict(name='snwmlt',   units='kg/m^2/s',scalefactor= 1.e3/86400., long_name='Snow Melt Runoff at the Base of the Snow Pack'),
                      evap_snow = dict(name='evap_snow',units='kg/m^2/s',scalefactor=-1.e3/86400., long_name='Sublimation from the Snow Pack'),
                      evap_blow = dict(name='evap_blow',units='kg/m^2/s',scalefactor=-1.e3/86400., long_name='Sublimation of Blowing Snow'),
                      # axes (don't have their own file)
                      time_stamp = dict(name='time_stamp', units='', long_name='Time Stamp'), # readable time stamp (string)
                      time = dict(name='time', units='day', long_name='Days'), # time coordinate
                      lon  = dict(name='lon', units='deg E', long_name='Longitude'), # geographic longitude
                      lat  = dict(name='lat', units='deg N', long_name='Latitude'), # geographic latitude
                      # derived variables
                      liqwatflx = dict(name='liqwatflx',units='kg/m^2/s',scalefactor=1., long_name='Liquid Water Flux'),
                      rho_snw   = dict(name='rho_snw',  units='kg/m^3',  scalefactor=1., long_name='Snow Density'),
                      precip    = dict(name='precip',   units='kg/m^2/s',scalefactor=1., long_name='Total Precipitation'),
                      )
# list of variables to load
binary_varlist = binary_varatts.keys()
netcdf_varlist = [varname for varname in netcdf_varatts.keys() if varname not in ('time','time_stamp','lat','lon')]
# some SnoDAS settings
missing_value = -9999 # missing data flag
snodas_shape2d = (SnoDAS_grid.size[1],SnoDAS_grid.size[0]) # 4096x8192
binary_dtype = np.dtype('>i2') # big-endian 16-bit signed integer
# settings for NetCDF-4 files
avgfolder = root_folder + dataset_name.lower()+'avg/' 
avgfile   = 'snodas{0:s}_clim{1:s}.nc' # the filename needs to be extended grid and period
tsfile    = 'snodas_{0:s}{1:s}_monthly.nc' # extend with variable and grid type
daily_folder    = root_folder + dataset_name.lower()+'_daily/' 
netcdf_filename = 'snodas_{:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float
netcdf_settings = dict(chunksizes=(8,snodas_shape2d[0]/16,snodas_shape2d[1]/32))

## helper functions to handle binary files

class DataError(Exception):
    ''' errors for binary reader '''
    pass

def getFilenameFolder(varname=None, date=None, root_folder=root_folder, lgzip=True):
    ''' simple function to generate the filename and folder of a file based on variable and date '''
    if not isinstance(date,dt.datetime):
        date = pd.to_datetime(date)
    datestr = '{:04d}{:02d}{:02d}'.format(date.year,date.month,date.day)
    varatts = binary_varatts[varname]
    # construct filename
    if varatts['variable_type'].lower() == 'driving': var_code = 'v0' + varatts['product_code'] + 'S'
    else: var_code = 'v1' + varatts['product_code']
    t_code = varatts['t_code']
    if varatts['product_code'] == '1044' and date < dt.datetime(2010,02,17): t_code = 'T0001'
    agg_code = varatts['v_code'] + t_code
    offset_code = 'P000' if varatts['variable_type'].lower() == 'diagnostic' else 'P001'
    I_code = 'H' if t_code.upper() == 'T0001' else 'D'
    filename = 'zz_ssm{:s}{:s}TTNATS{:s}05{:s}{:s}.dat'.format(var_code,agg_code,datestr,I_code,offset_code)        
    if lgzip: filename += '.gz'
    # contruct folder
    mon = '{:02d}_{:s}'.format(date.month,name_of_month[date.month-1][:3].title())
    folder = '{:s}/data/{:04d}/{:s}/SNODAS_unmasked_{:s}/'.format(root_folder,date.year,mon,datestr)
    return folder,filename

def readBinaryData(fobj=None, lstr=True):
    ''' load binary data for variable from file (defiend by file handle) and return a 2D array '''
    # read binary data (16-bit signed integers, big-endian)
    if lstr:
        # read binary data from file stream (basically as string; mainly for gzip files)
        data = np.fromstring(fobj.read(), dtype=binary_dtype, count=-1) # read binary data
    else:
        # read binary data from file system (does not work with gzip files)
        data = np.fromfile(fobj, dtype=binary_dtype, count=-1) # read binary data
    if data.size != snodas_shape2d[0]*snodas_shape2d[1]:
        raise DataError(data)
    data = data.reshape(snodas_shape2d) # assign shape
    return data
  
def readBinaryFile(varname=None, date=None, root_folder=root_folder, lgzip=True, scalefactor=None, 
                   lmask=True, lmissing=True):
    ''' load SnoDAS binary data for one day into a numpy array with proper scaling and unites etc. '''
    # find file
    folder,filename = getFilenameFolder(varname=varname, date=date, root_folder=root_folder, lgzip=lgzip)
    filepath = folder+filename
    # check if present
    try:
        # open file (gzipped or not)
        with gzip.open(filepath, mode='rb') if lgzip else open(filepath, mode='rb') as fobj:              
            data = readBinaryData(fobj=fobj, lstr=lgzip,) # read data        
        # flip y-axis (in file upper-left corner is origin, we want lower-left)
        data = np.flip(data, axis=0)
        # N.B.: the order of the axes is (y,x)
        # format such that we have actual variable values
        if lmask:
            fdata = np.ma.masked_array(data, dtype=netcdf_dtype)
            fdata = np.ma.masked_where(data==-9999, fdata, copy=False)
        else:
            fdata = np.asarray(data, dtype=netcdf_dtype)
        del data
        # apply scalefactor
        if scalefactor is None:
            scalefactor = binary_varatts[varname]['scalefactor']*netcdf_varatts[varname]['scalefactor']
        if scalefactor != 1:
            fdata *= scalefactor
    except IOError:
        if lmissing:
            print("Warning: data for '{}' missing - creating empty array!\n  ('{:s}')".format(date, folder))
            # create empty/masked array
            if lmask: fdata = np.ma.masked_all(snodas_shape2d, dtype=netcdf_dtype)            
            else: fdata = np.zeros(snodas_shape2d, dtype=netcdf_dtype)+missing_value
        else:
            print("Point of failure: {}/{}".format(varname,date))
            raise IOError("Data folder for '{}' is missing:\n  '{:s}'".format(date, folder))
    except DataError:
        if lmissing:
            print("Warning: data for '{}' incomplete - creating empty array!\n  ('{:s}')".format(date, folder))
            # create empty/masked array
            if lmask: fdata = np.ma.masked_all(snodas_shape2d, dtype=netcdf_dtype)            
            else: fdata = np.zeros(snodas_shape2d, dtype=netcdf_dtype)+missing_value
        else:
            print("Point of failure: {}/{}".format(varname,date))
            raise

    return fdata

def creatNetCDF(varname, varatts=None, ncatts=None, data_folder=daily_folder, fillValue=missing_value):
    ''' create a NetCDF-4 file for the given variable, create dimensions and variable, and allocate data;
        also set options for chunking and compression in a sensible manner '''
    if varatts is None: varatts = netcdf_varatts
    if ncatts is None: ncatts = netcdf_settings
    # create Dataset/file    
    filename = netcdf_filename.format(varname.lower())
    filepath = data_folder+filename
    ds = nc.Dataset(filepath, mode='w', format='NETCDF4', clobber=True)
    # add coordinate variables
    # time is the outer-most (record) dimension
    axes = [] # build axes order (need to add in order
    atts = varatts['time']; axes.append(atts['name'])
    add_coord(ds, atts['name'], data=None, length=None, atts=atts,
              dtype=np.dtype('i4'), zlib=True, fillValue=fillValue,) # daily
    # also add a time-stamp variable
    atts = varatts['time_stamp']
    add_var(ds, atts['name'], dims=axes, data=None, shape=(None,), 
            atts=atts, dtype=np.dtype('S'), zlib=True, fillValue=None, lusestr=True) # daily time-stamp
    # latitude (intermediate/regular dimension)
    atts = varatts['lat']; axes.append(atts['name'])
    add_coord(ds, atts['name'], data=SnoDAS_grid.ylat[:], length=SnoDAS_grid.size[1], atts=atts,
              dtype=netcdf_dtype, zlib=True, fillValue=fillValue,)
    # longitude is the inner-most dimension (continuous)
    atts = varatts['lon']; axes.append(atts['name'])
    add_coord(ds, atts['name'], data=SnoDAS_grid.xlon[:], length=SnoDAS_grid.size[0], atts=atts,
              dtype=netcdf_dtype, zlib=True, fillValue=fillValue,)
    # create NC variable
    atts = varatts[varname].copy()
    if 'scalefactor' in atts: del atts['scalefactor']
    add_var(ds, atts['name'], dims=axes, data=None, shape=(None,)+snodas_shape2d, atts=atts, 
            dtype=netcdf_dtype, zlib=True, fillValue=fillValue, lusestr=True, **ncatts)
    # return dataset object
    return ds
  
  
## convenience functions to handle GDAL-style georeferencing

# valid geographic/projected coordinates
x_coords_def = (('lon','long','longitude',), ('x','easting') )
y_coords_def = (('lat','latitude',),         ('y','northing'))

def getGeoCoords(xvar, x_coords=None, y_coords=None):
    ''' temporary helper function to identify lat/lon'''
    if x_coords is None: x_coords = x_coords_def
    if y_coords is None: y_coords = y_coords_def
    # test geographic grid and projected grids separately
    for i in range(len(x_coords)):
        xlon,ylat = None,None
        for name,coord in xvar.coords.items():
            if name.lower() in x_coords[i]: 
                xlon = coord; break
        for name,coord in xvar.coords.items():
            if name.lower() in y_coords[i]: 
                ylat = coord; break
        if xlon is not None and ylat is not None: break
    # return a valid pair of geographic or projected coordinate axis
    return xlon,ylat
  
def isGeoVar(xvar, x_coords=None, y_coords=None):
    ''' temporary helper function to identify lat/lon'''
    if x_coords is None: x_coords = x_coords_def
    if y_coords is None: y_coords = y_coords_def
    # test geographic grid and projected grids separately
    for i in range(len(x_coords)):
        xlon,ylat = False,False
        for name in xvar.coords.keys():
            if name.lower() in x_coords[i]: 
                xlon = True; break
        for name in xvar.coords.keys():
            if name.lower() in y_coords[i]: 
                ylat = True; break
        if xlon and ylat: break
    # if it has a valid pair of geographic or projected coordinate axis
    return ( xlon and ylat )

def addGeoReference(xds, proj4_string=proj4_string, x_coords=None, y_coords=None):
    ''' helper function to add GDAL georeferencing to an xarray dataset '''
    xds.attrs['proj4'] = proj4_string
    xlon,ylat = getGeoCoords(xds, x_coords=x_coords, y_coords=y_coords)
    xds.attrs['xlon'] = xlon.name
    xds.attrs['ylat'] = ylat.name
    for xvar in xds.data_vars.values(): 
        if isGeoVar(xvar):
            xvar.attrs['proj4'] = proj4_string
            xvar.attrs['xlon'] = xlon.name
            xvar.attrs['ylat'] = ylat.name
            xvar.attrs['dim_order'] = int( xvar.dims[-2:] == (ylat.name, xlon.name) )
            # N.B.: the NetCDF-4 backend does not like Python bools
    return xds


## functions to load NetCDF datasets (using xarray)

def loadSnoDAS_Daily(varname=None, varlist=None, folder=daily_folder, lxarray=True, lgeoref=True, 
                     chunks=None, time_chunks=8, geoargs=None, **kwargs):
    ''' function to load daily SnoDAS data from NetCDF-4 files using xarray and add some projection information '''
    if not lxarray: 
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    if time_chunks:
        cks = netcdf_settings['chunksizes'] if chunks is None else chunks
        # use default netCDF chunks or user chunks, but multiply time by time_chunks
        chunks = dict(time=cks[0]*time_chunks,lat=cks[1],lon=cks[2])
    # load variables
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname:
        # load a single variable
        xds = xr.open_dataset(folder+netcdf_filename.format(varname), chunks=chunks, **kwargs)
    else:
        if varlist is None: varlist = netcdf_varlist
        # load multifile dataset (variables are in different files
        filepaths = [folder + netcdf_filename.format(varname) for varname in varlist]
        xds = xr.open_mfdataset(filepaths, chunks=chunks, **kwargs)
        #xds = xr.merge([xr.open_dataset(fp, chunks=chunks, **kwargs) for fp in filepaths])    
    # add projection
    if lgeoref:
        if geoargs is None: geoargs = dict()
        xds = addGeoReference(xds, **geoargs)
    return xds
loadDataset_Daily = loadSnoDAS_Daily # alias


def loadSnoDAS_TS(varname=None, varlist=None, grid=None, folder=avgfolder, tsfile=tsfile, lxarray=True, 
                  lgeoref=True, lmonthly=False, chunks=None, time_chunks=1, geoargs=None, **kwargs):
    ''' function to load monthly transient SnoDAS data '''
    if not lxarray: 
        raise NotImplementedError("Only loading via xarray is currently implemented.")
    if time_chunks:
        if grid: 
            raise NotImplementedError("Default chunks are only available for native lat/lon grid.")
        cks = netcdf_settings['chunksizes'] if chunks is None else chunks
        # use default netCDF chunks or user chunks; set time chunking with time_chunks
        chunks = dict(time=time_chunks,lat=cks[1],lon=cks[2])
    # set options
    grid_str = '_'+grid if grid else ''
    if lmonthly:
        kwargs['decode_times'] = False
    # load variables
    if varname and varlist: raise ValueError(varname,varlist)
    elif varname:
        # load a single variable
        xds = xr.open_dataset(folder + tsfile.format(varname,grid_str), chunks=chunks, **kwargs)
    else:
        if varlist is None: varlist = netcdf_varlist
        # load multifile dataset (variables are in different files
        filepaths = [folder + tsfile.format(varname,grid_str) for varname in varlist]
        xds = xr.open_mfdataset(filepaths, chunks=chunks, **kwargs)
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
        if grid:
            raise NotImplementedError
        xds = addGeoReference(xds, **geoargs)
    return xds
loadTimeSeries = loadSnoDAS_TS # alias


def loadSnoDAS(varname=None, varlist=None, grid=None, period=None, folder=avgfolder, avgfile=avgfile, 
               lxarray=False, lgeoref=True, chunks=None, time_chunks=1, geoargs=None, 
               name=dataset_name, filemode='r', **kwargs):
    ''' function to load monthly transient SnoDAS data '''
    if time_chunks:
        if grid: 
            raise NotImplementedError("Default chunks are only available for native lat/lon grid.")
        cks = netcdf_settings['chunksizes'] if chunks is None else chunks
        # use default netCDF chunks or user chunks; set time chunking with time_chunks
        chunks = dict(time=time_chunks,lat=cks[1],lon=cks[2])
    # set options
    grid_str = '_'+grid if grid else ''
#     if lmonthly:
#         kwargs['decode_times'] = False
    if isinstance(period,basestring): prd_str = '_' + period
    elif isinstance(period,(tuple,list)) and len(period) == 2: 
        prd_str = '_{0:4d}-{1:4d}'.format(*period)
    else: raise TypeError(period) 
    # load dataset
    filepath = folder + avgfile.format(grid_str,prd_str)
    if lxarray:
        if varname and varlist: raise ValueError(varname,varlist)
        elif varname: varlist = [varname]
        # load a single variable
        dataset = xr.open_dataset(filepath, chunks=chunks, **kwargs)
        # load time stamps (like coordinate variables)
        if 'time_stamp' in dataset: dataset['time_stamp'].load()
        # add projection
        if lgeoref:
            if geoargs is None: geoargs = dict()
            if grid:
                raise NotImplementedError
            dataset = addGeoReference(dataset, **geoargs)
    else:
        # load standardized climatology dataset with NRCan-specific parameters
        dataset = loadObservations(name=name, folder=folder, projection=None, resolution=None, filepattern=avgfile, 
                                   period=period, grid=grid, varlist=varlist, varatts=None, griddef=SnoDAS_grid,
                                   filelist=None, lautoregrid=False, mode='climatology', filemode=filemode)
    # return formatted dataset
    return dataset
loadClimatology = loadSnoDAS # alias


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = netcdf_filename # filename pattern: variable name (daily)
ts_file_pattern = tsfile # filename pattern: variable name and grid
clim_file_pattern = avgfile # filename pattern: grid and period
data_folder = avgfolder # folder for user data
grid_def = {'':SnoDAS_grid} # no special name, since there is only one...
LTM_grids = [] # grids that have long-term mean data 
TS_grids = [''] # grids that have time-series data
grid_res = {'':0.00833333333333333} # no special name, since there is only one...
default_grid = SnoDAS_grid
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = loadSnoDAS_TS # time-series data
loadClimatology = loadSnoDAS # pre-processed, standardized climatology
loadStationClimatology = None # climatologies without associated grid (e.g. stations) 
loadStationTimeSeries = None # time-series without associated grid (e.g. stations)
loadShapeClimatology = None # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries = None # time-series without associated grid (e.g. provinces or basins)


## abuse for testing
if __name__ == '__main__':

  import dask, time, gc 
  
  #print('xarray version: '+xr.__version__+'\n')
  xr.set_options(keep_attrs=True)
        

#   from dask.distributed import Client, LocalCluster
#   # force multiprocessing (4 cores)
#   cluster = LocalCluster(n_workers=4, diagnostics_port=18787)
#   client = Client(cluster)

#   from multiprocessing.pool import ThreadPool
#   dask.set_options(pool=ThreadPool(4))

  test_mode = 'load_Climatology'
#   test_mode = 'monthly_normal'
#   test_mode = 'load_TimeSeries'
#   test_mode = 'monthly_mean'
#   test_mode = 'load_daily'
#   test_mode = 'add_variables'
#   test_mode = 'test_binary_reader'
#   test_mode = 'convert_binary'


  if test_mode == 'load_Climatology':
     
      
      varlist = netcdf_varlist
      lxarray = False
      ds = loadSnoDAS(varlist=varlist, period=(2011,2012), lxarray=lxarray) # load regular GeoPy dataset
      print(ds)
      print('')
      varname = ds.variables.keys()[0]
      var = ds[varname]
      print(var)

      if lxarray:
          print('Size in Memory: {:6.1f} MB'.format(var.nbytes/1024./1024.))


  elif test_mode == 'monthly_normal':

     
      # chunk sizes for monthly timeseries
      chunks = (1,)+netcdf_settings['chunksizes'][1:]
      chunk_settings = dict(time=chunks[0],lat=chunks[1],lon=chunks[2])

      # optional slicing (time slicing completed below)
      start_date = None; end_date = None
#       start_date = '2011-01'; end_date = '2011-12'

      ts_name = 'time_stamp'

      # variable list
      varlist = netcdf_varlist
#       varlist = ['rho_snw',]

      # start operation
      start = time.time()
          
      # load variables object (not data!)
      xds   = loadSnoDAS_TS(varlist=varlist)
      xds   = xds.loc[{'time':slice(start_date,end_date),}] # slice entire dataset
      ts_var = xds[ts_name].load()
      print(xds)
      
      # construct period string
      print('\n')
      prdstr = '{:04d}-{:04d}'.format(pd.to_datetime(ts_var.data[0]).year,
                         (pd.to_datetime(ts_var.data[-1])+pd.Timedelta(31, unit='D')).year)
      print(prdstr)
          
      print('\n')
      # compute monthly normals
      nds = xds.groupby('time.month').mean('time')
      assert len(nds['month'])==12, nds
      
      # convert time axis
      nds = nds.rename({'month':'time'}) # the new time axis is named 'month'
      tm = nds['time']
      tm.attrs['name']       = 'time'
      tm.attrs['long_name']  = 'Calendar Month'
      tm.attrs['units']      = 'month'
      tm.attrs['start_date'] = str(ts_var.data[0])
      tm.attrs['end_date']   = str(ts_var.data[-1])
      tm.attrs['period']     = prdstr
      # add attributes to dataset
      nds.attrs['start_date'] = str(ts_var.data[0])
      nds.attrs['end_date']   = str(ts_var.data[-1])
      nds.attrs['period']     = prdstr
      print(nds)

      
      # save resampled dataset
      filepath = avgfolder+avgfile.format('','_'+prdstr) # native grid...
      # write to NetCDF
      var_enc = dict(zlib=True, complevel=1, _FillValue=-9999, chunksizes=chunks)
      encoding = {varname:var_enc for varname in varlist}
      nds.to_netcdf(filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                    encoding=encoding, compute=True)
      
      # add name and length of month (doesn't work properly with xarray)
      ds = nc.Dataset(filepath, mode='a')
      # name of month
      vatts = dict(name='name_of_month', units='', long_name='Name of the Month')
      varnc = add_var(ds, vatts['name'], dims=('time',), data=None, shape=(None,), 
                     atts=vatts, dtype=np.dtype('S'), zlib=True, fillValue=None, lusestr=True) # string variable
      varnc[:] = np.stack(name_of_month, axis=0)
      # length of month
      vatts = dict(name='length_of_month', units='days',long_name='Length of Month')
      varnc = add_var(ds, vatts['name'], dims=('time',), data=None, shape=(None,), 
                     atts=vatts, dtype=np.dtype('float32'), zlib=True, fillValue=None,) # float variable
      varnc[:] = np.stack(days_per_month, axis=0)
      # close NetCDF dataset
      ds.sync(); ds.close()
      
      # print timing
      end = time.time()
      print('\n   Required time:   {:.0f} seconds\n'.format(end-start))


  elif test_mode == 'load_TimeSeries':
     
      
      varlist = netcdf_varlist
      varname = varlist[0]
      xds = loadSnoDAS_TS(varlist=varlist, time_chunks=1, lmonthly=False) # 32 may be possible
#       xds = loadSnoDAS_TS(varname=varname, time_chunks=1, lmonthly=False) # 32 may be possible      
      print(xds)
      print('')
      xv = xds[varname]
      xv = xv.loc['2011-01-01':'2011-01-31',]
#       xv = xv.loc['2011-01-01',:,:]
      print(xv)
      print('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.))

#       print('')
#       print(xds['time'])
      

  elif test_mode == 'monthly_mean':

      
      # chunk sizes for monthly timeseries
      chunks = (1,)+netcdf_settings['chunksizes'][1:]
      chunk_settings = dict(time=chunks[0],lat=chunks[1],lon=chunks[2])

      # optional slicing (time slicing completed below)
#       start_date = '2011-01-20'; end_date = '2011-02-11'
      start_date = None; end_date = None

      # variable list
      varlist = netcdf_varlist
#       varlist = ['precip','rho_snw']

      ts_name = 'time_stamp'
      
      # start operation
      start = time.time()
      
      # loop over variables (processed separately)      
      for varname in varlist:
               
          # load variables object (not data!)
          xds   = loadSnoDAS_Daily(varname=varname, time_chunks=1)
          xds   = xds.loc[{'time':slice(start_date,end_date),}] # slice entire dataset
          #print(xds)
          #print('\n')
          
          # aggregate month
          rds = xds.resample(time='MS',skipna=True,).mean()
          #rds.chunk(chunks=chunk_settings)         
          print(rds)
    
          
          # save resampled dataset
          filepath = avgfolder+tsfile.format(varname,'') # native grid...
          # write to NetCDF
          netcdf_encoding = dict(zlib=True, complevel=1, _FillValue=-9999, chunksizes=chunks)
          rds.to_netcdf(filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                        encoding={varname:netcdf_encoding}, compute=True)
          
          # add time-stamp (doesn't work properly with xarray)
          ds = nc.Dataset(filepath, mode='a')
          atts = netcdf_varatts[ts_name]
          tsnc = add_var(ds, ts_name, dims=('time',), data=None, shape=(None,), 
                         atts=atts, dtype=np.dtype('S'), zlib=True, fillValue=None, lusestr=True) # daily time-stamp
          tsnc[:] = np.stack([str(t) for t in rds['time'].data.astype('datetime64[M]')], axis=0)
          ds.sync(); ds.close()
          
      # print timing
      end = time.time()
      print('\n   Required time:   {:.0f} seconds\n'.format(end-start))


  elif test_mode == 'load_daily':
     
          
#       varlist = netcdf_varlist
      varlist = ['precip','rho_snw']
      varname = varlist[0]
      xds = loadSnoDAS_Daily(varlist=varlist, time_chunks=32) # 32 may be possible
      print(xds)
      print('')
      xv = xds[varname]
      xv = xv.loc['2011-01-01':'2011-02-01',35:45,-100:-80]
#       xv = xv.loc['2011-01-01',:,:]
      print(xv)
      print('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.))


  elif test_mode == 'add_variables':
    
      lappend_master = False
      start = time.time()
          
      # load variables
      time_chunks = 32 # 32 may be possible
      chunks = netcdf_settings['chunksizes']
      chunk_settings = dict(time=chunks[0]*time_chunks,lat=chunks[1],lon=chunks[2])      
      ts_name = 'time_stamp'
      varlist = ['precip','rho_snw',]
      xds = loadSnoDAS_Daily(varlist=None, time_chunks=time_chunks)
      print(xds)
      
      # optional slicing (time slicing completed below)
#       start_date = '2011-01-01'; end_date = '2011-01-08'
      start_date = None; end_date = None
#       lon_min = -85; lon_max = -75; lat_min = 40; lat_max = 45
#       xvar1 = xvar1.loc[:,lat_min:lat_max,lon_min:lon_max]
#       xvar2 = xvar2.loc[:,lat_min:lat_max,lon_min:lon_max]

      # slice and load time coordinate
      xds = xds.loc[{'time':slice(start_date,end_date),}]
      tsvar = xds[ts_name].load()
          
      
      # loop over variables
      for var in varlist:
      
          # target dataset
          var_atts = netcdf_varatts[var]
          nc_filepath = daily_folder + netcdf_filename.format(var)
          if lappend_master and osp.exists(nc_filepath):
              ncds = nc.Dataset(nc_filepath, mode='a')
              ncvar3 = ncds[var]
              ncts = ncds[ts_name]
              nctc = ncds['time'] # time coordinate
              # update start date for after present data
              start_date = pd.to_datetime(ncts[-1]) + pd.to_timedelta(1,unit='D')
              if end_date is None: end_date = tsvar.data[-1]
              end_date = pd.to_datetime(end_date)
              if start_date > end_date:
                  print("\nNothing to do - timeseries complete:\n {} > {}".format(start_date,end_date))
                  ncds.close()
                  exit()
              lappend = True
          else: 
              lappend = False
              
          
          print('\n')
          ## define actual computation
          if var == 'liqwatflx':
              ref_var = 'snwmlt'; note = "masked/missing values have been replaced by zero"
              xvar = xds['snwmlt'].fillna(0) + xds['liqprec'].fillna(0) # fill missing values with zero
              # N.B.: missing values are NaN in xarray; we need to fill with 0, or masked/missing values
              #       in snowmelt will mask/invalidate valid values in precip
          elif var == 'precip':
              ref_var = 'liqprec'; note = "masked/missing values have been replaced by zero"
              xvar = xds['liqprec'].fillna(0) + xds['solprec'].fillna(0) # fill missing values with zero
              # N.B.: missing values are NaN in xarray; we need to fill with 0, or masked/missing values
              #       in snowmelt will mask/invalidate valid values in precip
          elif var == 'rho_snw':
              ref_var = 'snow'; note = "SWE divided by snow depth, divided by 1000"
              xvar = xds['snow'] / xds['snowh']
              
          # define/copy metadata
          xvar.rename(var)
          xvar.attrs = xds[ref_var].attrs.copy()
          for att in ('name','units','long_name',):
              xvar.attrs[att] = var_atts[att]
          xvar.attrs['note'] = note
          xvar.chunk(chunks=chunk_settings)
          print(xvar)
    
          
    #       # visualize task graph
    #       viz_file = daily_folder+'dask_sum.svg'
    #       xvar3.data.visualize(filename=viz_file)
    #       print(viz_file)
          
          
          ## now save data, according to destination/append mode
          if lappend:
              # append results to an existing file
              print('\n')
              # define chunking
              offset = ncts.shape[0]; t_max = offset + tsvar.shape[0]
              tc,yc,xc = xvar.chunks # starting points of all blocks...
              tc = np.concatenate([[0],np.cumsum(tc[:-1], dtype=np.int)])
              yc = np.concatenate([[0],np.cumsum(yc[:-1], dtype=np.int)])
              xc = np.concatenate([[0],np.cumsum(xc[:-1], dtype=np.int)])
    #           xvar3 = xvar3.chunk(chunks=(tc,xvar3.shape[1],xvar3.shape[2]))
              # function to save each block individually (not sure if this works in parallel)
              dummy = np.zeros((1,1,1), dtype=np.int8)
              def save_chunk(block, block_id=None):
                  ts = offset + tc[block_id[0]]; te = ts + block.shape[0]
                  ys = yc[block_id[1]]; ye = ys + block.shape[1]
                  xs = xc[block_id[2]]; xe = xs + block.shape[2]
                  #print((ts,te),(ys,ye),(xs,xe))
                  print(block.shape)
                  ncvar3[ts:te,ys:ye,xs:xe] = block
                  return dummy
              # append to NC variable
              xvar.data.map_blocks(save_chunk, chunks=dummy.shape, dtype=dummy.dtype).compute() # drop_axis=(0,1,2), 
              # update time stamps and time axis
              nctc[offset:t_max] = np.arange(offset,t_max)
              for i in range(tsvar.shape[0]): ncts[i+offset] = tsvar.data[i] 
              ncds.sync()
              print('\n')
              print(ncds)
              ncds.close()
              del xvar, ncds 
          else:
              # save results in new file
              nds = xr.Dataset({ts_name:tsvar, var:xvar,}, attrs=xds.attrs.copy())
#               print('\n')
#               print(nds)
              # write to NetCDF
              var_enc = dict(zlib=True, complevel=1, _FillValue=-9999, chunksizes=netcdf_settings['chunksizes'])
              nds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4',
                            encoding={var:var_enc,}, compute=True)
              del nds, xvar
              
          # clean up
          gc.collect()
          
      # print timing
      end =  time.time()
      print('\n   Required time:   {:.0f} seconds\n'.format(end-start))


  elif test_mode == 'load_daily':
     
          
#       varlist = netcdf_varlist
      varlist = ['evap_snow', 'snwmlt','liqprec','solprec','snowh','Tsnow','liqwatflx','evap_blow',]
#       varlist = ['snwmlt','evap_snow']
      varname = varlist[0]
      xds = loadSnoDAS_Daily(varlist=varlist, time_chunks=32) # 32 may be possible
      print(xds)
      print('')
      xv = xds[varname]
      xv = xv.loc['2011-01-01':'2011-02-01',35:45,-100:-80]
#       xv = xv.loc['2011-01-01',:,:]
      print(xv)
      print('Size in Memory: {:6.1f} MB'.format(xv.nbytes/1024./1024.))


  elif test_mode == 'test_binary_reader':

    
      lgzip  = True
      # current date range
      date = '2009-12-14'
      date = '2018-11-24'
      # date at which snowmelt time aggregation label changes: 17/02/2010
      date = '2010-02-17'
      # date at which Canadian Prairies and Great Lakes are assimilated: 18/11/2010
      date = '2010-11-18'
      # date at which eastern Canada and Quebec are fully assimilated: 23/08/2012
      date = '2012-08-23'
      # a missing date
      date = '2010-02-06'
      date = '2014-07-29'
      

#       for varname in binary_varlist:
      for varname in ['evap_snow']:

          print('\n   ***   {}   ***   '.format(varname))
          # read data
          data = readBinaryFile(varname=varname, date=date, lgzip=lgzip,lmissing=True)
          # some checks
          assert isinstance(data,np.ndarray), data
          assert data.shape == snodas_shape2d, data.shape
          assert data.dtype == netcdf_dtype, data.dtype
          
          # diagnostics
          if not np.all(data.mask):
              print('Min: {:f}, Mean: {:f}, Max: {:f}'.format(data.min(),data.mean(),data.max()))  
          else: 
              print('No data available for timestep')
          print('Size in Memory: {:6.1f} MB'.format(data.nbytes/1024./1024.))            
          # make plot
#           import pylab as pyl
#           pyl.imshow(np.flipud(data[:,:])); pyl.colorbar(); pyl.show(block=True)
    
    
  elif test_mode == 'convert_binary':
    
    
      lappend = True
#       netcdf_settings = dict(chunksizes=(1,snodas_shape2d[0]/4,snodas_shape2d[1]/8))
      nc_time_chunk = netcdf_settings['chunksizes'][0]
      start_date = '2009-12-14'; end_date = '2018-11-24'

      if not osp.isdir(daily_folder): os.mkdir(daily_folder)

      # loop over binary variables (netcdf vars have coordiantes as well...)
      for varname in ['evap_snow']:
#       for varname in binary_varlist:
      

          filename = netcdf_filename.format(varname.lower())
          filepath = daily_folder+filename

          # create or open NetCDF-4 file
          start = time.time()
          if not osp.exists(filepath) or not lappend:
              # create NetCDF-4 dataset
              print("\nCreating new NetCDF-4 dataset for variable '{:s}':\n  '{:s}'".format(varname,filepath))
              ncds = creatNetCDF(varname, varatts=netcdf_varatts, ncatts=netcdf_settings, data_folder=daily_folder)
              ncds.sync()
              ncvar = ncds[varname]
              assert filepath == ncds.filepath(), ncds.filepath()
              ncts  = ncds['time_stamp']; nctc  = ncds['time']
              time_offset = len(ncts)
              assert  time_offset == 0, ncts
              # create datetime axis       
    #           time_array = np.arange('2009-12-14','2009-12-15', dtype='datetime64[D]')
#               time_array = np.arange('2009-12-14','2009-12-14', dtype='datetime64[D]')
              time_array = np.arange(start_date,end_date, dtype='datetime64[D]')
          else:
              # append to existing file
              print("\nOpening existing NetCDF-4 dataset for variable '{:s}':\n  '{:s}'".format(varname,filepath))
              ncds = nc.Dataset(filepath, mode='a', format='NETCDF4', clobber=False)
              ncvar = ncds[varname]
              nc_time_chunk = ncvar.chunking()[0]
              ncts  = ncds['time_stamp']; nctc  = ncds['time']
              time_offset = len(nctc)
              if nctc[0] != 0:
                  if nctc[0] == 1:
                      # some code to correct initial non-CF-compliant units (temporary...)
                      nctc[:] = nctc[:]-1
                      units = 'days since {:s}'.format(ncts[0])
                      print(units)
                      nctc.setncattr_string('units',units)
                  else:
                      raise ValueError(nctc[0])
              # create datetime axis with correct start date
              restart_date = pd.to_datetime(ncts[-1]) + pd.Timedelta(2, unit='D')
              # N.B.: need to add *two* days, because SnoDAS uses end-of-day time-stamp
              time_array = np.arange(restart_date,end_date, dtype='datetime64[D]')
#               time_array = np.arange(start_date,'2009-12-14', dtype='datetime64[D]')
              if len(time_array) > 0:
                  print("First record time-stamp: {:s} (offset={:d}), Time Chunking: {:d}".format(time_array[0],time_offset,nc_time_chunk))  
              else:
                  print("\nSpecified records are already present; exiting.\n")
                  ncds.close()
                  continue # skip ahead to next variable
          
          flush_intervall = 64 if nc_time_chunk==1 else nc_time_chunk*4
          
          # loop over daily records and periodically write to disk
          ii = 0; fi = 0; c = 0; var_chunks = []; tc_chunks = []; ts_chunks = []
          print("\nIterating over daily rasters:\n")
          for i,day in enumerate(time_array):
              
              ii = time_offset + i + 1 # one-based day
              actual_day = day -1
              ## N.B.: look into validity of time-stamp, i.e. end of day or beginning?
    
              # load data and add to variable chunk list
              var_chunks.append(readBinaryFile(varname=varname, date=day,))
              # assign time stamp to time chunk list
              tc_chunks.append(i) # zero-based
              print("  {}".format(actual_day))
              ts_chunks.append(str(actual_day))
              
              # now assign chunk to file
              c += 1
              if c == nc_time_chunk:
                  ic = ii - nc_time_chunk # start of chunk
                  # write data
                  ncvar[ic:ii,:,:] = np.stack(var_chunks, axis=0)
                  nctc[ic:ii] = np.stack(tc_chunks, axis=0)          
                  ncts[ic:ii] = np.stack(ts_chunks, axis=0)
                  # reset intervall counter and lists
                  c = 0; var_chunks = []; tc_chunks = []; ts_chunks = []

              # periodic flushing to disk
              fi += 1
              if fi == flush_intervall:
                  print("Flushing data to disk ({:d}, {:s})".format(ii, actual_day))
                  ncds.sync() # flush data
                  fi = 0 # reset intervall counter
                  gc.collect()
                  
          # write remaining data for incomplete chunk
          if len(var_chunks) > 0:
              ic = ii - c # start of chunk
              # write data
              ncvar[ic:ii,:,:] = np.stack(var_chunks, axis=0)
              nctc[ic:ii] = np.stack(tc_chunks, axis=0)          
              ncts[ic:ii] = np.stack(ts_chunks, axis=0)
              # delete chunk lists lists
              del var_chunks, tc_chunks, ts_chunks
              gc.collect()
          
          print("\nCompleted iteration; read {:d} rasters and created NetCDF-4 variable:\n".format(ii))    
          print(ncvar)

          # make sure time units are set correctly
          assert nctc[0] == 0, nctc
          nctc.setncattr_string('units','days since {:s}'.format(ncts[0]))
          # flush data to disk and close file
          ncds.sync()
          ncds.close()
          
          end =  time.time()
          print('\n   Required time:   {:.0f} seconds\n'.format(end-start))
                
  