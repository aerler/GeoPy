'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc # netcdf python module
import os
# internal imports
from geodata.netcdf import DatasetNetCDF, VarNC, AxisNC
from geodata.gdal import addGDALtoDataset
from geodata.misc import DatasetError 
from datasets.misc import translateVarNames, days_per_month, name_of_month, data_root
 
## GPCC Meta-data

# variable attributes and name
varatts = dict(p    = dict(name='precip', units='mm/month'), # total precipitation rate
               s    = dict(name='stations', units='#'), # number of gauges for observation
               # axes (don't have their own file; listed in axes)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
#                time = dict(name='time', units='days', offset=1)) # time coordinate
# attributes of the time axis depend on type of dataset 
ltmvaratts = dict(time=dict(name='time', units='month', offset=1), **varatts) 
tsvaratts = dict(time=dict(name='time', units='day', offset=-28854), **varatts)
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
rootfolder = data_root + 'GPCC/' # long-term mean folder


## Functions to load different types of GPCC datasets 

# climatology
ltmfolder = rootfolder + 'climatology/' # climatology subfolder
def loadGPCC_LTM(name='GPCC', varlist=varlist, resolution='025', varatts=ltmvaratts, filelist=None, folder=ltmfolder):
  ''' Get a properly formatted dataset the monthly accumulated GPCC precipitation climatology. '''
  # prepare input
  if resolution not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # load variables separately
  if 'p' in varlist:
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=['normals_v2011_%s.nc'%resolution], varlist=['p'], 
                            varatts=varatts, ncformat='NETCDF4_CLASSIC')
  if 's' in varlist: 
    gauges = nc.Dataset(folder+'normals_gauges_v2011_%s.nc'%resolution, mode='r', format='NETCDF4_CLASSIC')
    stations = VarNC(data=gauges.variables['p'][0,:,:], axes=(dataset.lat,dataset.lon), **varatts['s'])
  # consolidate dataset
  dataset.addVariable(stations, copy=False)  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

# time-series
tsfolder = rootfolder + 'full_data_1900-2010/' # climatology subfolder
def loadGPCC_TS(name='GPCC', varlist=varlist, resolution='05', varatts=tsvaratts, filelist=None, folder=tsfolder):
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
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = rootfolder + 'gpccavg/' 
avgfile = 'gpcc%s_clim%s.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
def loadGPCC(name='GPCC', varlist=None, resolution='025', period=None, folder=avgfolder, filelist=None, varatts=None):
  ''' Get the pre-processed monthly GPCC climatology as a DatasetNetCDF. '''
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
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
  
  mode = 'average_timeseries'
  reses = ('25',) # for testing
#   reses = ('05', '10', '25')
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_clim':
      
      # load averaged climatology file
      print('')
      dataset = loadGPCC()
      print(dataset)
          
    elif mode == 'convert_climatology':
      # load dataset
      dataset = loadGPCC_LTM(varlist=['stations','precip'],resolution=res)    
      
      # convert precip data to SI units (mm/s)
      dataset.precip *= days_per_month.reshape((12,1,1)) # convert in-place
      dataset.precip.units = 'kg/m^2/s'
      
      # write data to file
      from geodata.nctools import writeNetCDF
      writeNetCDF(dataset, rootfolder+avgfolder+avgfile%res)
      
      # close...
      dataset.close()
      # print dataset before
      print(dataset)     
      
    elif mode == 'average_timeseries':
      
      # load source
      print('')
      source = loadGPCC_TS(varlist=['stations','precip'],resolution=res)
      print(source)
      # prepare sink
      filename = 'gpcc_%s_clim_avg.nc'%res
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      sink = DatasetNetCDF(name='GPCC Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w') 
      
      # process
      from geodata.process import ClimatologyProcessingUnit
      CPU = ClimatologyProcessingUnit(source, sink)
      CPU.process(flush=False)
      
      # convert precip data to SI units (mm/s)   
      sink.precip /= (days_per_month.reshape((12,1,1)) * 86400.) # convert in-place
      sink.precip.units = 'kg/m^2/s'      

      # add landmask
      sink += VarNC(sink.dataset, name='landmask', units='', axes=('lat','lon'), data=sink.precip.getMask()[0,:,:])
      
      
      # add names of months
      me = len(name_of_month); ne = len(name_of_month[0])
      coord = np.ndarray((me,ne),dtype='S1') # array of strings
      for m in xrange(me): 
        for n in xrange(ne): coord[m,n] = name_of_month[m][n]
      strax = AxisNC(sink.datasets[0], name='string',  units='', length=ne, dtype='S1', mode='w') # character axis
      sink += VarNC(sink.datasets[0], name='name_of_month', units='', axes=(sink.time,strax), 
                    data=coord, dtype='S1', mode='w')
      
      sink.sync()
      newvar = sink.precip
      print
      print newvar.name, newvar.masked
      print newvar.fillValue
      print newvar.data_array.__class__
      print
      
      # close...
#       sink.load()
      sink.sync()
      sink.close()
      # print dataset before
      print('')
      print(sink)     
      
    