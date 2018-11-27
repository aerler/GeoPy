'''
Created on Nov. 25, 2018

A module to read SnoDAS data; this includes reading SnoDAS binary files and conversion to NetCDF-4, 
as well as functions to load the converted and aggregated data.

@author: Andre R. Erler, GPL v3
'''

# external imports
import datetime as dt
import dateutil.parser as dp
import os
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
# N.B.: this is the new grid (in use since 2013); the old grid was slightly shifted, but less than 1km
geotransform = (-130.516666666667-0.00416666666666052, 0.00833333333333333, 0, 
                 24.1000000000000-0.00416666666666052, 0, 0.00833333333333333)
size = (8192,4096) # (x,y) map size of SnoDAS grid
# make GridDefinition instance
SnoDAS_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform, size=size)

# attributes of variables in SnoDAS binary files
binary_varatts = dict(# forcing variables (offset_code='P001', downscale_code='S', type_code='v0')
                      liqprec = dict(name='liqprec',units='kg/m^2',scalefactor=0.1, product_code='1025',v_code='lL00',t_code='T0024',description='24 hour total, 06:00 UTC-06:00 UTC',variable_type='Driving',long_name='Liquid Precipitation'),
                      solprec = dict(name='solprec',units='kg/m^2',scalefactor=0.1, product_code='1025',v_code='lL01',t_code='T0024',description='24 hour total, 06:00 UTC-06:00 UTC',variable_type='Driving',long_name='Solid Precipitation'),
                      # state variables (offset_code='P001', type_code='v1')
                      snow = dict(name='snow', units='m', scalefactor=1e-3,   product_code='1034',v_code='tS__',t_code='T0001',description='Snapshot at 06:00 UTC',variable_type='state',long_name='Snow Water Equivalent'),
                      snowh = dict(name='snowh', units='m', scalefactor=1e-3, product_code='1036',v_code='tS__',t_code='T0001',description='Snapshot at 06:00 UTC',variable_type='state',long_name='Snow Water Equivalent'),
                      Tsnow = dict(name='Tsnow', units='K', scalefactor=1,    product_code='1038',v_code='wS__',t_code='A0024',description='',variable_type='State',long_name='Snow Pack Average Temperature'),
                      # diagnostic variables (offset_code='P000', type_code='v1')
                      snwmlt = dict(name='snwmlt',units='m',scalefactor=1e-5,       product_code='1044',v_code='bS__',t_code='T0024',description='Total of 24 per hour melt rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Snow Melt Runoff at the Base of the Snow Pack'),
                      evap_snow = dict(name='evap_snow',units='m',scalefactor=1e-5, product_code='1050',v_code='lL00',t_code='T0024',description='Total of 24 per hour sublimation rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Sublimation from the Snow Pack'),
                      evap_blow = dict(name='evap_blow',units='m',scalefactor=1e-5, product_code='1039',v_code='lL00',t_code='T0024',description='Total of 24 per hour sublimation rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Sublimation of Blowing Snow'),
                      )
# list of variables to load
binary_varlist = binary_varatts.keys()


## helper functions to handle binary files

def getFilenameFolder(varname=None, date=None, root_folder=root_folder, lgzip=True):
    ''' simple function to generate the filename and folder of a file based on variable and date '''
    if isinstance(date,basestring): 
        date = dp.parse(date)
    if isinstance(date,dt.datetime):
        date = dt.date(date.year,date.month,date.day)
    if not isinstance(date,dt.date):
        raise TypeError(date)
    datestr = '{:04d}{:02d}{:02d}'.format(date.year,date.month,date.day)
    varatts = binary_varatts[varname]
    # construct filename
    if varatts['variable_type'].lower() == 'driving': var_code = 'v0' + varatts['product_code'] + 'S'
    else: var_code = 'v1' + varatts['product_code']
    t_code = varatts['t_code']
    if varatts['product_code'] == '1044' and date < dt.date(2010,02,17): t_code = 'T0001'
    agg_code = varatts['v_code'] + t_code
    offset_code = 'P000' if varatts['variable_type'].lower() == 'diagnostic' else 'P001'
    I_code = 'H' if t_code.upper() == 'T0001' else 'D'
    filename = 'zz_ssm{:s}{:s}TTNATS{:s}05{:s}{:s}.dat'.format(var_code,agg_code,datestr,I_code,offset_code)        
    if lgzip: filename += '.gz'
    # contruct folder
    mon = '{:02d}_{:s}'.format(date.month,name_of_month[date.month-1][:3].title())
    folder = '{:s}/data/{:04d}/{:s}/SNODAS_unmasked_{:s}/'.format(root_folder,date.year,mon,datestr)
    return folder,filename

## abuse for testing
if __name__ == '__main__':

  test_mode = 'test_binary_reader'
#   test_mode = 'convert_binary'


  if test_mode == 'test_binary_reader':
    
      for varname in binary_varlist:
          # select file
#           folder,filename = getFilenameFolder(varname=varname,date='2009-12-14')
#           folder,filename = getFilenameFolder(varname=varname,date='2010-02-16')
#           folder,filename = getFilenameFolder(varname=varname,date='2010-02-17')          
          folder,filename = getFilenameFolder(varname=varname,date='2018-11-24')
          filepath = folder+filename
          print(filepath)
          if not osp.exists(filepath):
              raise IOError(filepath)
    
  elif test_mode == 'convert_binary':
    
      raise NotImplementedError
    
