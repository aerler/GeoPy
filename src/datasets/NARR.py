'''
Created on 2013-09-08

This module contains meta data and access functions for NARR datasets. 

@author: Andre R. Erler, GPL v3
'''

#external imports
import numpy as np
import os
# from atmdyn.properties import variablePlotatts
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, getProjFromDict
from geodata.misc import DatasetError 
from datasets.misc import translateVarNames, days_per_month, name_of_month, data_root
from geodata.process import CentralProcessingUnit


## NARR Meta-data

# projection
projdict = dict(proj  = 'lcc', # Lambert Conformal Conic  
                lat_1 =   50., # Latitude of first standard parallel
                lat_2 =   50., # Latitude of second standard parallel
                lat_0 =   50., # Latitude of natural origin
                lon_0 = -107., # Longitude of natural origin
                x_0   = 5632642.22547, # False Origin Easting
                y_0   = 4612545.65137) # False Origin Northing

# variable attributes and name
varatts = dict(air   = dict(name='T2', units='K'), # 2m Temperature
               prate = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
               # LTM-only variables (currently...)
               prmsl = dict(name='pmsl', units='Pa'), # sea-level pressure
               pevap = dict(name='pet', units='kg/m^2'), # monthly accumulated PET (kg/m^2)
               pr_wtr = dict(name='pwtr', units='kg/m^2'), # total precipitable water (kg/m^2)
               # axes (don't have their own file; listed in axes)
               lon   = dict(name='lon', units='deg E'), # geographic longitude field
               lat   = dict(name='lat', units='deg N'), # geographic latitude field
               time  = dict(name='time', units='days', offset=-1569072, scalefactor=1./24.), # time coordinate
               # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
               #       the time offset is chose such that 1979 begins with the origin (time=0)   
               x     = dict(name='x', units='m', offset=-1*projdict['x_0']), # projected west-east coordinate
               y     = dict(name='y', units='m', offset=-1*projdict['y_0'])) # projected south-north coordinate
# N.B.: At the moment Skin Temperature can not be handled this way due to a name conflict with Air Temperature
# list of variables to load
ltmvarlist = varatts.keys() # also includes coordinate fields    
tsvarlist = ['air', 'prate', 'lon', 'lat'] # 'air' is actually 2m temperature...

# variable and file lists settings
narrfolder = data_root + 'NARR/' # root folder
nofile = ('lat','lon','x','y','time') # variables that don't have their own files
special = dict(air='air.2m') # some variables need special treatment


## Functions to load different types of NARR datasets 

# Climatology (LTM - Long Term Mean)
ltmfolder = narrfolder + 'LTM/' # LTM subfolder
def loadNARR_LTM(name='NARR', varlist=ltmvarlist, interval='monthly', varatts=varatts, filelist=None, folder=ltmfolder):
  ''' Get a properly formatted dataset of daily or monthly NARR climatologies (LTM). '''
  # prepare input
  if interval == 'monthly': 
    pfx = '.mon.ltm.nc'; tlen = 12
  elif interval == 'daily': 
    pfx = '.day.ltm.nc'; tlen = 365
  else: raise DatasetError, "Selected interval '%s' is not supported!"%interval
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)  
  # axes dictionary, primarily to override time axis 
  axes = dict(time=Axis(name='time',units='day',coord=(1,tlen,tlen)),load=True)
  if filelist is None: # generate default filelist
    filelist = [special[var]+pfx if var in special else var+pfx for var in varlist if var not in nofile]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          axes=axes, atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='{0:s} Coordinate System'.format(name))
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None)
  # return formatted dataset
  return dataset

# Time-series (monthly)
tsfolder = narrfolder + 'Monthly/' # monthly subfolder
def loadNARR_TS(name='NARR', varlist=tsvarlist, varatts=varatts, filelist=None, folder=tsfolder):
  ''' Get a properly formatted NARR dataset with monthly mean time-series. '''
  # prepare input  
  pfx = '.mon.mean.nc'
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  if filelist is None: # generate default filelist
    filelist = [special[var]+pfx if var in special else var+pfx for var in varlist if var not in nofile]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='{0:s} Coordinate System'.format(name))
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None)
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = narrfolder + 'narravg/' 
avgfile = 'narr%s_clim%s.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
def loadNARR(name='NARR', varlist=None, period=None, folder=avgfolder, filelist=None, varatts=None):
  ''' Get the pre-processed monthly NARR climatology as a DatasetNetCDF. '''
  # prepare input
  if not isinstance(period,basestring): period = '%4i-%4i'%period  
  # varlist
  if varlist is None: varlist = ['precip', 'T2', 'lon', 'lat', 'time', 'length_of_month'] # all variables 
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
  grid = 'NARR'
  period = (1979,1981)
  
  if mode == 'test_climatology':
    
    # load averaged climatology file
    print('')
    dataset = loadNARR(period=period)
    print(dataset)
              
  # generate averaged climatology
  elif mode == 'average_timeseries':
    
    # load source
    periodstr = '%4i-%4i'%period
    print('\n')
    print('   ***   Processing Grid %s from %s   ***   '%(grid,periodstr))
    print('\n')
    source = loadNARR_TS()
    print(source)
    print('\n')
    # prepare sink
    gridstr = '' if grid is 'NARR' else '_'+grid
    filename = avgfile%(gridstr,'_'+periodstr)
    if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
    sink = DatasetNetCDF(name='NARR Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
    sink.atts.period = periodstr 
    
    # determine averaging interval
    offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, varlist=['precip', 'T2'], tmp=True)
    
    # start processing climatology
    print('')
    print('   +++   processing climatology   +++   ') 
    CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
    print('\n')      
    
    # sync temporary storage with output
    CPU.sync(flush=False)
    # find new coordinate arrays
    dlon = dlat = 0.5 # resolution
#     slon = sink.lon.getArray().min(); elon = sink.lon.getArray().max()
#     print slon,elon
#     slat = sink.lat.getArray().min(); elat = sink.lat.getArray().max()
#     print slat,elat
    slon, slat, elon, elat =    -179.75, 0.75, -0.25, 85.75 
    lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
    lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
    # add new geographic coordinate axes for projected map
    xlon = Axis(coord=lon, atts=dict(name='lon', long_name='longitude', units='deg E'))
    ylat = Axis(coord=lat, atts=dict(name='lat', long_name='latitude', units='deg N'))
    
    # reproject and resample (regrid) dataset
    print('')
    print('   +++   processing regidding   +++   ') 
    CPU.Regrid(xlon=xlon, ylat=ylat, flush=False)
    print('\n')      
    
    # sync temporary storage with output
    CPU.sync(flush=True)

#     # make new masks
#     sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=True, merge=False)

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
    