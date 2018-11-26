'''
Created on Nov. 25, 2018

A module to read SnoDAS data; this includes reading SnoDAS binary files and conversion to NetCDF-4, 
as well as functions to load the converted and aggregated data.

@author: Andre R. Erler, GPL v3
'''

# external imports
import datetime as dt
import numpy as np
try: import cPickle as pickle
except: import pickle

# internal imports
from geodata.misc import ArgumentError, VariableError, DataError, isNumber, DatasetError, translateSeasons
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

# variable attributes and name
binary_varatts = dict(# forcing variables (offset_code='P001', downscale_code='S', type_code='v0')
                      liqprec = dict(name='liqprec',units='kg/m^2',scalefactor=10,atts=dict(product_code='1025',v_code='lL00',t_code='T0024',description='24 hour total, 06:00 UTC-06:00 UTC',variable_type='Driving',long_name='Liquid Precipitation')),
                      solprec = dict(name='solprec',units='kg/m^2',scalefactor=10,atts=dict(product_code='1025',v_code='lL01',t_code='T0024',description='24 hour total, 06:00 UTC-06:00 UTC',variable_type='Driving',long_name='Solid Precipitation')),
                      # state variables (offset_code='P001', type_code='v1')
                      snow = dict(name='snow', units='m', scalefactor=1e-3,   atts=dict(product_code='1034',v_code='tS__',t_code='T0001',description='Snapshot at 06:00 UTC',variable_type='state',long_name='Snow Water Equivalent')),
                      snowh = dict(name='snowh', units='m', scalefactor=1e-3, atts=dict(product_code='1036',v_code='tS__',t_code='T0001',description='Snapshot at 06:00 UTC',variable_type='state',long_name='Snow Water Equivalent')),
                      Tsnow = dict(name='Tsnow', units='K', scalefactor=1,    atts=dict(product_code='1038',v_code='wS__',t_code='A0024',description='',variable_type='State',long_name='Snow Pack Average Temperature')),
                      # diagnostic variables (offset_code='P000', type_code='v1')
                      snwmlt = dict(name='snwmlt',units='m',scalefactor=1e-5,      atts=dict(product_code='1044',v_code='bS__',t_code='T0001',description='Total of 24 per hour melt rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Snow Melt Runoff at the Base of the Snow Pack')),
                      evap_snow = dict(name='evap_snow',units='m',scalefactor=1e-5,atts=dict(product_code='1050',v_code='lL00',t_code='T0024',description='Total of 24 per hour sublimation rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Sublimation from the Snow Pack')),
                      evap_blow = dict(name='evap_blow',units='m',scalefactor=1e-5,atts=dict(product_code='1039',v_code='lL00',t_code='T0024',description='Total of 24 per hour sublimation rates, 06:00 UTC-06:00 UTC',variable_type='diagnostic',long_name='Sublimation of Blowing Snow')),
                      )
# list of variables to load
variable_list = binary_varatts.keys()


## abuse for testing
if __name__ == '__main__':

  test_mode = 'test_binary'
#   test_mode = 'convert_binary'


  if test_mode == 'test_binary':
    
      pass
    
  elif test_mode == 'convert_binary':
    
      raise NotImplementedError
    
