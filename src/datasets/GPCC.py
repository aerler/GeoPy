'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import netCDF4 as nc # netcdf python module
# internal imports
from geodata.base import Variable
from geodata.netcdf import NetCDFDataset
from geodata.gdal import GDALDataset
from geodata.misc import DatasetError 
from datasets.misc import translateVarNames, days_per_month
 
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
rootfolder = '/home/DATA/DATA/GPCC/' # long-term mean folder
nofile = ('lat','lon','time') # variables that don't have their own files
special = dict(air='air.2m') # some variables need special treatment


## Functions to load different types of GPCC datasets 

# climatology
ltmfolder = rootfolder + 'climatology/' # climatology subfolder
def loadGPCCLTM(varlist=varlist, resolution='025', varatts=ltmvaratts, filelist=None, folder=ltmfolder):
  ''' Get a properly formatted dataset the monthly accumulated GPCC precipitation climatology. '''
  # prepare input
  if resolution not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # load variables separately
  if 'p' in varlist:
    dataset = NetCDFDataset(folder=folder, filelist=['normals_v2011_%s.nc'%resolution], varlist=['p'], 
                            varatts=varatts, ncformat='NETCDF4_CLASSIC')
  if 's' in varlist: 
    gauges = nc.Dataset(folder+'normals_gauges_v2011_%s.nc'%resolution, mode='r', format='NETCDF4_CLASSIC')
    stations = Variable(data=gauges.variables['p'][0,:,:], axes=(dataset.lat,dataset.lon), **varatts['s'])
  # consolidate dataset
  dataset.addVariable(stations)  
  dataset = GDALDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

# time-series
tsfolder = rootfolder + 'full_data_1900-2010/' # climatology subfolder
def loadGPCCTS(varlist=varlist, resolution='05', varatts=tsvaratts, filelist=None, folder=tsfolder):
  ''' Get a properly formatted dataset with the monthly GPCC time-series. '''
  # prepare input  
  if resolution not in ('05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
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

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = rootfolder + 'gpccavg/' 
avgfile = 'gpcc%s_clim%s.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
def loadGPCC(varlist=None, resolution='025', period=None, folder=avgfolder, filelist=None, varatts=None):
  ''' Get the pre-processed monthly GPCC climatology as a NetCDFDataset. '''
  # prepare input
  if resolution not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  if resolution == '025' and period is not None: raise DatasetError, "The highest resolution is only available for the lon-term mean!"
  # varlist
  if varlist is None: varlist = ['precip', 'stations'] # all variables 
  if varatts is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: 
    if period is None: filelist = [avgfile%('_'+resolution,'')]
    else: filelist = [avgfile%('_'+resolution,'_'+period)]
  # load dataset
  dataset = NetCDFDataset(folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')  
  dataset = GDALDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
  
  # load averaged climatology file
  dataset = loadGPCC()
  print dataset
  
  # generate averaged climatology
  for resolution in (): # ('025', '05', '10', '25'):    
    
    # load dataset
    print('')
    dataset = loadGPCCLTM(varlist=['stations','precip'],resolution=resolution)    
    
    # convert precip data to SI units (mm/s)
    dataset.precip *= days_per_month.reshape((12,1,1)) # convert in-place
    dataset.precip.units = 'kg/m^2/s'
    
    # write data to file
    from geodata.nctools import writeNetCDF
    writeNetCDF(dataset, rootfolder+avgfolder+avgfile%resolution)
    
    # close...
    dataset.close()
    print(dataset) # print dataset before    