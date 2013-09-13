'''
Created on 2013-09-12

This module contains meta data and access functions for the monthly CFSR time-series data.

@author: Andre R. Erler, GPL v3
'''

# internal imports
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset
from datasets.misc import translateVarNames, days_per_month, months_names, data_root


## CRU Meta-data

# variable attributes and name
varatts = dict(TMP_L103_Avg = dict(name='T2', units='K'), # 2m average temperature
               TMP_L1 = dict(name='Ts', units='K'), # average skin temperature
               PRATE_L1 = dict(name='precip', units='kg/m^2/s'), # total precipitation
               PRES_L1 = dict(name='ps', units='Pa'), # surface pressure
               WEASD_L1 = dict(name='snow', units='kg/m^2'), # snow water equivalent
               SNO_D_L1 = dict(name='snowh', units='m'), # snow depth
               # lower resolution
               PRMSL_L101 = dict(name='pmsl', units='Pa'), # sea-level pressure
               # static fields (full resolution)
               LAND_L1 = dict(name='lnd', units=''), # land mask
               HGT_L1 = dict(name='zs', units='m'), # surface elevation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', scalefactor=1/24.), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
# N.B.: the time-series begins in 1979 (time=0), so no offset is necessary

# variable and file lists settings
rootfolder = data_root + 'CFSR/' # long-term mean folder
nofile = ('lat','lon','time') # variables that don't have their own files
# file names by variable
hiresfiles = dict(TMP_L103_Avg='TMP.2m', TMP_L1='TMP.SFC', PRATE_L1='PRATE.SFC', PRES_L1='PRES.SFC',
                 WEASD_L1='WEASD.SFC', SNO_D_L1='SNO_D.SFC')
hiresstatic = dict(LAND_L1='LAND.SFC', HGT_L1='HGT.SFC')
lowresfiles = dict(PRMSL_L101='PRMSL.MSL')
lowresstatic = dict() # currently none available
hiresfiles = {key:'flxf06.gdas.{0:s}.grb2.nc'.format(value) for key,value in hiresfiles.iteritems()}
hiresstatic = {key:'flxf06.gdas.{0:s}.grb2.nc'.format(value) for key,value in hiresstatic.iteritems()}
lowresfiles = {key:'pgbh06.gdas.{0:s}.grb2.nc'.format(value) for key,value in lowresfiles.iteritems()}
lowresstatic = {key:'pgbh06.gdas.{0:s}.grb2.nc'.format(value) for key,value in lowresstatic.iteritems()}
# list of variables to load
varlist = hiresfiles.keys() + hiresstatic.keys() + list(nofile) # hires + coordinates    


## Functions to load different types of CFSR datasets 

tsfolder = rootfolder + 'Monthly/'
def loadCFSR_TS(name='CFSR', varlist=varlist, varatts=varatts, resolution='high', filelist=None, folder=tsfolder):
  ''' Get a properly formatted CFSR dataset with monthly mean time-series. '''
  # translate varlist
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  if filelist is None: # generate default filelist
    if resolution == 'high': files = [hiresfiles[var] for var in varlist if var in hiresfiles]
    elif resolution == 'low': files = [lowresfiles[var] for var in varlist if var in lowresfiles]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=files, varlist=varlist, varatts=varatts, 
                          check_override=['time'], multifile=False, ncformat='NETCDF4_CLASSIC')
  # load static data
  if filelist is None: # generate default filelist
    if resolution == 'high': files = [hiresstatic[var] for var in varlist if var in hiresstatic]
    elif resolution == 'low': files = [lowresstatic[var] for var in varlist if var in lowresstatic]
    # create singleton time axis
    staticdata = DatasetNetCDF(name=name, folder=folder, filelist=files, varlist=varlist, varatts=varatts, 
                               axes=dict(lon=dataset.lon, lat=dataset.lat), multifile=False, ncformat='NETCDF4_CLASSIC')
    # N.B.: need to override the axes, so that the datasets are consistent
  if len(staticdata.variables) > 0:
    if staticdata.hasAxis('time'): staticdata.time.name = 'singleton_time'
    for var in staticdata.variables.values(): 
      if not dataset.hasVariable(var.name):
        # var.load() # need to load variables into memory, in order to copy
        var.squeeze()
        dataset.addVariable(var, copy=False) # no need to copy...
  # add projection  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  # load dataset
  dataset = loadCFSR_TS()
  
  # print dataset
  print(dataset)

  print 
  print dataset.lnd
  print dataset.lnd[45,:]         