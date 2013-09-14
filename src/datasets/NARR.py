'''
Created on 2013-09-08

This module contains meta data and access functions for NARR datasets. 

@author: Andre R. Erler, GPL v3
'''

# from atmdyn.properties import variablePlotatts
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, getProjFromDict
from geodata.misc import DatasetError 
from datasets.misc import translateVarNames, days_per_month, name_of_month, data_root


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
               prmsl = dict(name='pmsl', units='Pa'), # sea-level pressure
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
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
rootfolder = data_root + 'NARR/' # root folder
nofile = ('lat','lon','x','y','time') # variables that don't have their own files
special = dict(air='air.2m') # some variables need special treatment


## Functions to load different types of NARR datasets 

# Climatology (LTM - Long Term Mean)
ltmfolder = rootfolder + 'LTM/' # LTM subfolder
def loadNARR_LTM(name='NARR', varlist=varlist, interval='monthly', varatts=varatts, filelist=None, folder=ltmfolder):
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
tsfolder = rootfolder + 'Monthly/' # monthly subfolder
def loadNARR_TS(name='NARR', varlist=varlist, varatts=varatts, filelist=None, folder=tsfolder):
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

## (ab)use main execution for quick test
if __name__ == '__main__':
    
    # load dataset
    dataset = loadNARR_TS(varlist=['T2','precip'])
    
    # print dataset
    print(dataset)
    
    # print time axis
    time = dataset.time
    print
    print time
    print time.scalefactor
#     print time[:]