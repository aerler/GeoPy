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
try: import cPickle as pickle
except: import pickle

# internal imports
from geodata.misc import ArgumentError, VariableError, DataError, DatasetError, translateSeasons
from geodata.misc import name_of_month
from datasets.common import BatchLoad, getRootFolder
from geodata.base import Dataset, Variable, Axis, concatDatasets
from geodata.gdal import loadPickledGridDef, addGDALtoDataset, GridDefinition
from geodata.gdal import grid_folder as common_grid_folder


## HGS Meta-vardata

dataset_name = 'SnoDAS'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='HGS') # get dataset root folder based on environment variables

# SnoDAS grid definition
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
                      snowh = dict(name='snowh', units='m',      scalefactor=1.,   long_name='Snow Water Equivalent'),
                      Tsnow = dict(name='Tsnow', units='K',      scalefactor=1.,   long_name='Snow Pack Average Temperature'),
                      # diagnostic variables
                      snwmlt    = dict(name='snwmlt',   units='kg/m^2/s',scalefactor= 1.e3/86400., long_name='Snow Melt Runoff at the Base of the Snow Pack'),
                      evap_snow = dict(name='evap_snow',units='kg/m^2/s',scalefactor=-1.e3/86400., long_name='Sublimation from the Snow Pack'),
                      evap_blow = dict(name='evap_blow',units='kg/m^2/s',scalefactor=-1.e3/86400., long_name='Sublimation of Blowing Snow'),
                      )
# list of variables to load
binary_varlist = binary_varatts.keys()

avgfolder = root_folder + dataset_name.lower()+'avg/' 

## helper functions to handle binary files

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
        data = np.fromstring(fobj.read(), dtype=np.dtype('>i2'), count=-1) # read binary data
    else:
        # read binary data from file system (does not work with gzip files)
        data = np.fromfile(fobj, dtype=np.dtype('>i2'), count=-1) # read binary data
    data = data.reshape((SnoDAS_grid.size[1],SnoDAS_grid.size[0])) # assign shape
    return data
  
def readBinaryFile(varname=None, date=None, root_folder=root_folder, lgzip=True, scalefactor=None, lmask=True):
    ''' load SnoDAS binary data for one day into a numpy array with proper scaling and unites etc. '''
    # find file
    folder,filename = getFilenameFolder(varname=varname, date=date, root_folder=root_folder, lgzip=lgzip)
    filepath = folder+filename
    if not osp.exists(filepath): 
        raise IOError(filepath)               
    # open file (gzipped or not)
    with gzip.open(filepath, mode='rb') if lgzip else open(filepath, mode='rb') as fobj:              
        data = readBinaryData(fobj=fobj, lstr=lgzip,) # read data        
    # flip y-axis (in file upper-left corner is origin, we want lower-left)
    data = np.flip(data, axis=0)
    # N.B.: the order of the axes is (y,x)
    # format such that we have actual variable values
    if lmask:
        fdata = np.ma.masked_array(data, dtype=np.dtype('<f4'))
        fdata = np.ma.masked_where(data==-9999, fdata, copy=False)
    else:
        fdata = np.asarray(data, dtype=np.dtype('<f4'))
    del data
    # apply scalefactor
    if scalefactor is None:
        scalefactor = binary_varatts[varname]['scalefactor']*netcdf_varatts[varname]['scalefactor']
    if scalefactor != 1:
        fdata *= scalefactor
    return fdata

## abuse for testing
if __name__ == '__main__':

#   test_mode = 'test_binary_reader'
  test_mode = 'test_convert'
#   test_mode = 'convert_binary'


  if test_mode == 'test_binary_reader':
    
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

      date = '2010-06-14'
      for varname in binary_varlist:
#       for varname in ['liqprec']:

          print('\n   ***   {}   ***   '.format(varname))
          # read data
          data = readBinaryFile(varname=varname, date=date, lgzip=lgzip,)
          # some checks
          assert isinstance(data,np.ndarray), data
          assert data.dtype == np.dtype('<f4'), data.dtype
          
          # diagnostics
          print('Min: {:f}, Mean: {:f}, Max: {:f}'.format(data.min(),data.mean(),data.max()))              
          # make plot
          import pylab as pyl
#           pyl.imshow(np.flipud(data[:,:])); pyl.colorbar(); pyl.show(block=True)
    
  elif test_mode == 'test_convert':
    
      import dask
      import dask.array as da
      import xarray

      datatype = np.dtype('<f4') # little-endian 32-bit float
      shape2d = (SnoDAS_grid.size[1],SnoDAS_grid.size[0])
      varname = 'snow'
      # create datetime axis       
      time_array = np.arange('2009-12-14','2010-01-14', dtype='datetime64[D]')
      
      # loop over days to construct dask execution graph
      data_arrays = []
      for day in time_array:
          # create delayed array slices
          print day
          data2d = dask.delayed(readBinaryFile)(varname=varname, date=day,)
          data_arrays.append(da.from_delayed(data2d, shape=shape2d, dtype=datatype)) 
          
      # construct delayed dask array from list of slices
      data3d = da.stack(data_arrays)
      print(data3d)
      
      # cast into xarray and write netcdf
      data = xarray.DataArray(data3d, dims=['time','lat','lon'])
      print(data)
      data.to_netcdf(avgfolder+'test_daily.nc')
  
  elif test_mode == 'convert_binary':
    
      raise NotImplementedError
    
