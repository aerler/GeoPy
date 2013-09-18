'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import netCDF4 as nc # netcdf python module
import os
# internal imports
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset
from geodata.misc import DatasetError
from geodata.nctools import add_strvar 
from datasets.misc import translateVarNames, days_per_month, name_of_month, data_root
from geodata.process import CentralProcessingUnit

## GPCC Meta-data

# variable attributes and name
varatts = dict(p    = dict(name='precip', units='mm/month'), # total precipitation rate
               s    = dict(name='stations', units='#'), # number of gauges for observation
               # axes (don't have their own file; listed in axes)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
#                time = dict(name='time', units='days', offset=1)) # time coordinate
# attributes of the time axis depend on type of dataset 
ltmvaratts = dict(time=dict(name='time', units='months', offset=1), **varatts) 
tsvaratts = dict(time=dict(name='time', units='days', offset=-28854), **varatts)
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
    stations = Variable(data=gauges.variables['p'][0,:,:], axes=(dataset.lat,dataset.lon), **varatts['s'])
    # consolidate dataset
    dataset.addVariable(stations, asNC=False, copy=True)  
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
  if not isinstance(period,basestring): period = '%4i-%4i'%period  
  # varlist
  if varlist is None: varlist = ['precip', 'stations', 'landmask', 'length_of_month'] # all variables 
  if varatts is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: 
    if period is None: filelist = [avgfile%('_'+resolution,'')]
    else: filelist = [avgfile%('_'+resolution,'_'+period)]
  # load dataset
  #print folder+filelist[0]
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4')  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
  
#   mode = 'test_climatology'
  mode = 'average_timeseries'
#   mode = 'convert_climatology'
  reses = ('25',) # for testing
#   reses = ('025',) # hi-res climatology
#   reses = ('05', '10', '25')
  period = (1979,1981)
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_climatology':
      
      # load averaged climatology file
      print('')
      dataset = loadGPCC(resolution=res,period=period)
      print(dataset)
          
    elif mode == 'convert_climatology':
      
      from geodata.base import Variable
      from geodata.nctools import writeNetCDF
      
      # load dataset
      dataset = loadGPCC_LTM(varlist=['stations','precip'],resolution=res)
      # change meta-data
      dataset.name = 'GPCC'
      dataset.title = 'GPCC Long-term Climatology'
      dataset.atts.resolution = res
      
      # load data into memory
      dataset.load()
#       newvar = dataset.time
#       print
#       print newvar.name, newvar.data
#       print newvar.data_array
#       print

      # convert precip data to SI units (mm/s)
      dataset.precip *= days_per_month.reshape((12,1,1)) # convert in-place
      dataset.precip.units = 'kg/m^2/s'

      # add landmask
      tmpatts = dict(name='landmask', units='', long_name='Landmask for Climatology Fields', 
                description='where this mask is non-zero, no data is available')
      dataset += Variable(name='landmask', units='', axes=(dataset.lat,dataset.lon), 
                          data=dataset.precip.getMask()[0,:,:], atts=tmpatts)
      dataset.mask(dataset.landmask)            
      # add names and length of months
      dataset += Variable(name='length_of_month', units='days', axes=(dataset.time,), data=days_per_month, dtype='i4', 
                          atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
      
      # write data to a different file
      filename = avgfile%('_'+res,'')
      print('') 
      print(filename)
      print('')
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      ncset = writeNetCDF(dataset, avgfolder+filename, close=False)
      add_strvar(ncset,'name_of_month', name_of_month, 'time', # add names of month
                 atts=dict(name='name_of_month', units='', long_name='Name of the Month')) 
      
      # close...
      ncset.close()
      dataset.close()
      # print dataset before
      print(dataset)
      print('')           
      
    elif mode == 'average_timeseries':
      
      # load source
      periodstr = '%4i-%4i'%period
      print('\n')
      print('   ***   Processing Resolution %s from %s   ***   '%(res,periodstr))
      print('\n')
      source = loadGPCC_TS(varlist=['stations','precip'],resolution=res)
      print(source)
      print('\n')
      # prepare sink
      filename = avgfile%('_'+res,'_'+periodstr)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      atts =dict(period=periodstr, name='GPCC', title='GPCC Climatology') 
      sink = DatasetNetCDF(name='GPCC Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
      
      # determine averaging interval
      offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, tmp=True)
            
      # start processing climatology
      print('')
      print('   +++   processing climatology   +++   ') 
      CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
      print('\n')
      
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift/roll   +++   ') 
#       CPU.Shift(lon=-180, flush=False)
#       print('\n')
#
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift/roll   +++   ') 
#       CPU.Shift(shift=72, axis='lon', byteShift=True, flush=False)
#       print('\n')      


#       newvar = sink.time
#       print
#       print newvar.name, newvar.data 
#       print newvar.shape
#       print newvar.coord
#       print

      # get results
      CPU.sync(flush=False, deepcopy=False)
#       sink = CPU.getTmp(asNC=True, filename=avgfolder+filename, atts=atts)
      
      # convert precip data to SI units (mm/s)   
      sink.precip /= (days_per_month.reshape((12,1,1)) * 86400.) # convert in-place
      sink.precip.units = 'kg/m^2/s'      

      # add landmask
      #print '   ===   landmask   ===   '
      tmpatts = dict(name='landmask', units='', long_name='Landmask for Climatology Fields', 
                description='where this mask is non-zero, no data is available')
      sink += VarNC(sink.dataset, name='landmask', units='', axes=(sink.lat,sink.lon), 
                    data=sink.precip.getMask()[0,:,:], atts=tmpatts)
      sink.mask(sink.landmask)            
      # add names and length of months
      sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                          atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
      #print '   ===   month   ===   '
      sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                    atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
              
#       newvar = sink.precip
#       print
#       print newvar.name, newvar.masked
#       print newvar.fillValue
#       print newvar.data_array.__class__
#       print
      
      # close...
      sink.sync()
      sink.close()
      # print dataset
      print('')
      print(sink)     
      
    