'''
Created on 2013-09-09

This module contains meta data and access functions for the monthly CRU time-series data. 

@author: Andre R. Erler, GPL v3
'''

# internal imports
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset
from datasets.misc import translateVarNames, days_per_month, months_names, data_root

 
## CRU Meta-data

# variable attributes and name
varatts = dict(tmp = dict(name='T2', units='K', offset=273.15), # 2m average temperature
               tmn = dict(name='Tmin', units='K', offset=273.15), # 2m minimum temperature
               tmx = dict(name='Tmax', units='K', offset=273.15), # 2m maximum temperature
               vap = dict(name='Q2', units='Pa', scalefactor=100.), # 2m water vapor pressure
               pet = dict(name='pet', units='kg/m^2/s', scalefactor=86400.), # potential evapo-transpiration
               pre = dict(name='precip', units='kg/m^2/s', scalefactor=86400.), # total precipitation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', offset=-28854), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field

# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
rootfolder = data_root + 'CRU/' # long-term mean folder
nofile = ('lat','lon','time') # variables that don't have their own files
filename = 'cru_ts3.20.1901.2011.%s.dat.nc' # file names, need to extend with %varname (original)

## Functions to load different types of GPCC datasets 

# Time-series (monthly)
tsfolder = rootfolder + 'Time-series 3.2/data/' # monthly subfolder
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


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  # load dataset
  dataset = loadCRU_TS()
  
  # print dataset
  print(dataset)

  print 
  print dataset.pet
  print dataset.pet[45,:,:]         