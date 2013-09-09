'''
Created on 2013-09-08

This module contains meta data and access functions for NARR datasets. 

@author: Andre R. Erler, GPL v3
'''

# from atmdyn.properties import variablePlotatts
from geodata.base import Axis
from geodata.netcdf import NetCDFDataset
from geodata.gdal import GDALDataset, getProjFromDict
from geodata.misc import DatasetError 

## NARR Climatology (LTM - Long Term Mean)

# projection
projdict = dict(proj = 'lcc', # Lambert Conformal Conic  
                lat1 =   50., # Latitude of first standard parallel
                lat2 =   50., # Latitude of second standard parallel
                lat0 =   50., # Latitude of natural origin
                lon0 = -107., # Longitude of natural origin
                x0   = 5632642.22547, # False Origin Easting
                y0   = 4612545.65137) # False Origin Northing

# variable attributes and name
varatts = dict(air   = dict(name='T2', units='K'), # 2m Temperature
               prate = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
               prmsl = dict(name='pmsl', units='Pa'), # sea-level pressure
               # axes (don't have their own file; listed in axes)
               lon   = dict(name='lon', units='deg E'), # geographic longitude field
               lat   = dict(name='lat', units='deg N'), # geographic latitude field
               time  = dict(name='time', units='days', offset=-1569072, scalefactor=1./24.), # time coordinate
               # N.B.: the time coordinate is only used for the monthly data, not the LTM, which starts in 1979   
               x     = dict(name='x', units='m', offset=-1*projdict['x0']), # projected west-east coordinate
               y     = dict(name='y', units='m', offset=-1*projdict['y0'])) # projected south-north coordinate
# N.B.: At the moment Skin Temperature can not be handled this way due to a name conflict with Air Temperature
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
folder = '/home/DATA/DATA/NARR/' # long-term mean folder
nofile = ('lat','lon','x','y','time') # variables that don't have their own files
special = dict(air='air.2m') # some variables need special treatment


## Functions to load different types of NARR datasets 

def loadNARRLTM(varlist=varlist, interval='monthly', varatts=varatts, projection=projdict, filelist=None, folder=folder):
  ''' Get a properly formatted dataset of daily or monthly NARR climatologies (LTM). '''
  # prepare input
  folder += 'LTM/' # LTM subfolder
  if interval == 'monthly': 
    pfx = '.mon.ltm.nc'; tlen = 12
  elif interval == 'daily': 
    pfx = '.day.ltm.nc'; tlen = 365
  else: raise DatasetError, "Selected interval '%s' is not supported!"%interval
  # translate varlist
  if varlist and varatts:
    for key,value in varatts.iteritems():
      if value['name'] in varlist: 
        varlist[varlist.index(value['name'])] = key # original name
  # axes dictionary, primarily to override time axis 
  axes = dict(time=Axis(name='time',units='day',coord=(1,tlen,tlen)),load=True)
  if filelist is None: # generate default filelist
    filelist = [special[var]+pfx if var in special else var+pfx for var in varlist if var not in nofile]
  # load dataset
  dataset = NetCDFDataset(folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          axes=axes, atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='NARR Coordinate System')
  dataset = GDALDataset(dataset, projection=projection, geotransform=None)
  # return formatted dataset
  return dataset

def loadNARR(varlist=varlist, varatts=varatts, projection=projdict, filelist=None, folder=folder):
  ''' Get a properly formatted  NARR dataset with monthly mean time-series. '''
  # prepare input
  folder += 'Monthly/' # monthly subfolder
  pfx = '.mon.mean.nc'
  # translate varlist
  if varlist and varatts:
    for key,value in varatts.iteritems():
      if value['name'] in varlist: 
        varlist[varlist.index(value['name'])] = key # original name
  if filelist is None: # generate default filelist
    filelist = [special[var]+pfx if var in special else var+pfx for var in varlist if var not in nofile]
  # load dataset
  dataset = NetCDFDataset(folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='NARR Coordinate System')
  dataset = GDALDataset(dataset, projection=projection, geotransform=None)
  # return formatted dataset
  return dataset

## (ab)use main execution for quick test
if __name__ == '__main__':
    
    # load dataset
    dataset = loadNARR(varlist=['T2','precip'])
    
    # print dataset
    print(dataset)
    
    # print time axis
    time = dataset.time
    print
    print time
    print time.scalefactor
    print time[:]