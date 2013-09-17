'''
Created on 2013-09-09

This module contains meta data and access functions for the monthly CRU time-series data. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import os
# internal imports
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset
from datasets.misc import translateVarNames, days_per_month, name_of_month, data_root

 
## CRU Meta-data

# variable attributes and name
varatts = dict(tmp = dict(name='T2', units='K', offset=273.15), # 2m average temperature
               tmn = dict(name='Tmin', units='K', offset=273.15), # 2m minimum temperature
               tmx = dict(name='Tmax', units='K', offset=273.15), # 2m maximum temperature
               vap = dict(name='Q2', units='Pa', scalefactor=100.), # 2m water vapor pressure
               pet = dict(name='pet', units='kg/m^2/s', scalefactor=1./86400.), # potential evapo-transpiration
               pre = dict(name='precip', units='kg/m^2/s', scalefactor=1./86400.), # total precipitation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', offset=-28854), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field

# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
crufolder = data_root + 'CRU/' # long-term mean folder
nofile = ('lat','lon','time') # variables that don't have their own files
filename = 'cru_ts3.20.1901.2011.%s.dat.nc' # file names, need to extend with %varname (original)

## Functions to load different types of GPCC datasets 

# Time-series (monthly)
tsfolder = crufolder + 'Time-series 3.2/data/' # monthly subfolder
def loadCRU_TS(name='CRU', varlist=varlist, varatts=varatts, filelist=None, folder=tsfolder):
  ''' Get a properly formatted  CRU dataset with monthly mean time-series. '''
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # assemble filelist
  if filelist is None: # generate default filelist
    filelist = [filename%var for var in varlist if var not in nofile]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = crufolder + 'cruavg/' 
avgfile = 'cru_clim%s.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
def loadCRU(name='CRU', varlist=None, period=None, folder=avgfolder, filelist=None, varatts=None):
  ''' Get the pre-processed monthly CRU climatology as a DatasetNetCDF. '''
  # prepare input
  if not isinstance(period,basestring): period = '%4i-%4i'%period  
  # varlist
  if varlist is None: varlist = ['precip', 'pet', 'Q2', 'T2', 'Tmin', 'Tmax', 'landmask', 'length_of_month'] # all variables 
  if varatts is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: 
    if period is None: filelist = [avgfile%('',)]
    else: filelist = [avgfile%('_'+period,)]  
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          multifile=False, ncformat='NETCDF4', load=False)  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
    
#   mode = 'test_climatology'
  mode = 'average_timeseries'
  period = (1979,1981)

  if mode == 'test_climatology':
    
    # load averaged climatology file
    print('')
    dataset = loadCRU(period=period)
    print(dataset)
        
  elif mode == 'average_timeseries':
      
    # load source
    periodstr = '%4i-%4i'%period
    print('\n')
    print('   ***   Processing Time-series from %s   ***   '%(periodstr,))
    print('\n')
    source = loadCRU_TS()
    print(source)
    print('\n')
    # prepare sink
    filename = avgfile%('_'+periodstr,)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
    sink = DatasetNetCDF(name='CRU Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
    sink.atts.period = periodstr 
    
    # determin averaging itnerval
    offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979
    #print offset 
    # initialize processing
    from geodata.process import ClimatologyProcessingUnit
    CPU = ClimatologyProcessingUnit(source, sink, period=period[1]-period[0], offset=offset)
    # start processing
    print('')
    #print('   ...   processing   ...   ') 
    CPU.process(flush=False)
    print('')

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
    
    # close...
    sink.sync()
    sink.close()
    # print dataset
    print('')
    print(sink)     
    
  