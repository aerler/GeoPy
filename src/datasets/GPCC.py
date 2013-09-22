'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import netCDF4 as nc # netcdf python module
import os
# internal imports
from geodata.base import Variable
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, addGDALtoVar, GridDefinition
from geodata.misc import DatasetError
from geodata.nctools import add_strvar 
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root, loadClim
from geodata.process import CentralProcessingUnit

## GPCC Meta-data

# GPCC grid definition           
geotransform_025 = (-180.0, 0.25, 0.0, -90.0, 0.0, 0.25)
size_025 = (1440,720) # (x,y) map size
geotransform_05 = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
size_05 = (720,360) # (x,y) map size
geotransform_10 = (-180.0, 1.0, 0.0, -90.0, 0.0, 1.0)
size_10 = (360,180) # (x,y) map size
geotransform_25 = (-180.0, 2.5, 0.0, -90.0, 0.0, 2.5)
size_25 = (144,72) # (x,y) map size

# make GridDefinition instance
GPCC_025_grid = GridDefinition(projection=None, geotransform=geotransform_025, size=size_025)
GPCC_05_grid = GridDefinition(projection=None, geotransform=geotransform_05, size=size_05)
GPCC_10_grid = GridDefinition(projection=None, geotransform=geotransform_10, size=size_10)
GPCC_25_grid = GridDefinition(projection=None, geotransform=geotransform_25, size=size_25)

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
# function to load these files...
def loadGPCC(name='GPCC', period=None, grid=None, resolution=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly GPCC climatology as a DatasetNetCDF. '''
  # prepare input
  if grid is not None and grid[0:5].lower() == 'gpcc_': 
    resolution = grid[5:]
    grid = None
  elif resolution is None: 
    resolution = '025' if period is None else '05'
  # check resolution
  if grid is None:
    # check for valid resolution 
    if resolution not in ('025','05', '10', '25'): 
      raise DatasetError, "Selected resolution '%s' is not available!"%resolution  
    if resolution == '025' and period is not None: 
      raise DatasetError, "The highest resolution is only available for the lon-term mean!"
    grid = resolution # grid supersedes resolution  
  # load standardized climatology dataset with GPCC-specific parameters
  dataset = loadClim(name=name, folder=folder, projection=None, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
  
  mode = 'test_climatology'
#   mode = 'average_timeseries'
#   mode = 'convert_climatology'
  reses = ('25',) # for testing
#   reses = ('025',) # hi-res climatology
#   reses = ('05', '10', '25')
  period = None #(1979,1981)
  grid = 'GPCC'
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_climatology':
      
      # load averaged climatology file
      print('')
      dataset = loadGPCC(grid='%s_%s'%(grid,res),resolution=res,period=period)
      print(dataset)
      print('')
      print(dataset.geotransform)
          
    elif mode == 'convert_climatology':
      
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
      # add landmask
      #print '   ===   landmask   ===   '
      tmp = source; source.load()
      tmpatts = dict(name='landmask', units='', long_name='Landmask for Climatology Fields', 
                description='where this mask is non-zero, no data is available')
      lnd = Variable(name='landmask', units='', axes=(tmp.lat,tmp.lon), dtype='int16', 
                    data=tmp.precip.getMask()[0,:,:], atts=tmpatts)
      lnd = addGDALtoVar(lnd, projection=source.projection, geotransform=source.geotransform)
      tmp += lnd
      print(source)
      print('\n')
#       newvar = source.landmask
#       print
#       print newvar.name #, newvar.gdal 
#       print newvar.shape
#       print newvar.data
#       print
      
            
      # prepare sink
      if grid == 'GPCC': filename = avgfile%('_'+res,'_'+periodstr)
      if grid == 'NARR': filename = avgfile%('_narr','_'+periodstr)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      atts =dict(period=periodstr, name='GPCC', title='GPCC Climatology') 
      sink = DatasetNetCDF(name='GPCC Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
      
      # determine averaging interval
      offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, tmp=True)
#       CPU = CentralProcessingUnit(source, tmp=True)
            
      # start processing climatology
      print('')
      print('   +++   processing climatology   +++   ') 
      CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
      print('\n')

      
      # define new coordinates
#       import numpy as np
#       from geodata.base import Axis    
#       dlon = dlat = 2.5 # resolution
#       slon, slat, elon, elat =    -179.75, 0.75, -72.25, 85.75
#       assert (elon-slon) % dlon == 0 
#       lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
#       assert (elat-slat) % dlat == 0
#       lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
#       # add new geographic coordinate axes for projected map
#       xlon = Axis(coord=lon, atts=dict(name='lon', long_name='longitude', units='deg E'))
#       ylat = Axis(coord=lat, atts=dict(name='lat', long_name='latitude', units='deg N'))

      # get NARR coordinates
      if grid == 'NARR':
        from datasets.NARR import loadNARR_TS, projdict
        from geodata.gdal import getProjFromDict
        narr = loadNARR_TS()
        x = narr.x.copy(deepcopy=True); y = narr.y.copy(deepcopy=True)
        geot = narr.geotransform
        proj = getProjFromDict(projdict)
        # reproject and resample (regrid) dataset
        print('')
        print('   +++   processing regidding   +++   ') 
        CPU.Regrid(projection=proj, geotransform=geot, xlon=x, ylat=y, flush=False)
        print('\n')
      
      
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift longitude   +++   ') 
#       CPU.Shift(lon=-180, flush=True)
#       print('\n')
#
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift/roll   +++   ') 
#       CPU.Shift(shift=72, axis='lon', byteShift=True, flush=False)
#       print('\n')      


      # get results
      CPU.sync(flush=True, deepcopy=False)
#       sink = CPU.getTmp(asNC=True, filename=avgfolder+filename, atts=atts)
      # print dataset
#       print('')
#       print(sink)     

      # convert precip data to SI units (mm/s)   
      sink.precip /= (days_per_month.reshape((12,1,1)) * 86400.) # convert in-place
      sink.precip.units = 'kg/m^2/s'      

      newvar = sink.stations
      print
      print newvar.name, newvar.masked 
      print newvar.shape, newvar.data
      print newvar.data_array.mean()
      print newvar.data_array.__class__, newvar.fillValue
            
      # add landmask
      #print '   ===   landmask   ===   '
#       tmpatts = dict(name='landmask', units='', long_name='Landmask for Climatology Fields', 
#                 description='where this mask is non-zero, no data is available')
#       sink += VarNC(sink.dataset, name='landmask', units='', axes=(sink.lat,sink.lon), 
#                     data=sink.precip.getMask()[0,:,:], atts=tmpatts)
      sink.stations.mask(sink.landmask)            
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
      
    