'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

from geodata.netcdf import NetCDFDataset
from geodata.gdal import GDALDataset
from geodata.misc import DatasetError 

## GPCC Meta-data

# variable attributes and name
varatts = dict(p    = dict(name='precip', units='mm/month'), # total precipitation rate
               s    = dict(name='stations', units=''), # number of gauges for observation
               # axes (don't have their own file; listed in axes)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
#                time = dict(name='time', units='days', offset=1)) # time coordinate
# attributes of the time axis depend on type of dataset 
ltmvaratts = dict(time=dict(name='time', units='days', offset=1), **varatts) 
tsvaratts = dict(time=dict(name='time', units='days', offset=-28854), **varatts)
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
folder = '/home/DATA/DATA/GPCC/' # long-term mean folder
nofile = ('lat','lon','time') # variables that don't have their own files
special = dict(air='air.2m') # some variables need special treatment


## Functions to load different types of GPCC datasets 

# climatology
def loadGPCCLTM(varlist=varlist, resolution='025', varatts=ltmvaratts, filelist=None, folder=folder):
  ''' Get a properly formatted dataset the monthly accumulated GPCC precipitation climatology. '''
  # prepare input
  folder += 'climatology/' # climatology subfolder
  if resolution not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts:
    for key,value in varatts.iteritems():
      if value['name'] in varlist: 
        varlist[varlist.index(value['name'])] = key # original name
  if filelist is None: # generate default filelist
    filelist = []
    if 'p' in varlist: filelist.append('normals_v2011_%s.nc'%resolution)
    if 's' in varlist: filelist.append('normals_gauges_v2011_%s.nc'%resolution)
  # load dataset
  dataset = NetCDFDataset(folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')  
  dataset = GDALDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

# time-series
def loadGPCCTS(varlist=varlist, resolution='05', varatts=tsvaratts, filelist=None, folder=folder):
  ''' Get a properly formatted dataset with the monthly GPCC time-series. '''
  # prepare input
  folder += 'full_data_1900-2010/' # climatology subfolder
  if resolution not in ('05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts:
    for key,value in varatts.iteritems():
      if value['name'] in varlist: 
        varlist[varlist.index(value['name'])] = key # original name
  if filelist is None: # generate default filelist
    filelist = []
    if 'p' in varlist: filelist.append('full_data_v6_precip_%s.nc'%resolution)
    if 's' in varlist: filelist.append('full_data_v6_statio_%s.nc'%resolution)
  # load dataset
  dataset = NetCDFDataset(folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')  
  dataset = GDALDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
  
  # load dataset
  dataset = loadGPCCTS(varlist=['stations','precip'])
  
  # print dataset
  print(dataset)
  
  # print time axis
  time = dataset.time
  print
  print time
  print time.offset
  print time[12*79]