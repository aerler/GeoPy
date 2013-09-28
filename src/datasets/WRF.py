'''
Created on 2013-09-28

This module contains meta data and access functions for WRF model output. 

@author: Andre R. Erler, GPL v3
'''

#external imports
import numpy as np
import os
# from atmdyn.properties import variablePlotatts
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, getProjFromDict, GridDefinition
from geodata.misc import DatasetError 
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root
from geodata.process import CentralProcessingUnit


# N.B.: Unlike with observational datasets, model Meta-data depends on the experiment and has to be 
#       loaded from the NetCFD-file; a few conventions have to be defied, however.

# include these variables in monthly means 
varlist = ['ps','T2','Ts','snow','snowh','rainnc','rainc','snownc','graupelnc',
           'Q2','evap','hfx','lhfx','OLR','GLW','SWDOWN'] # 'SWNORM'
varmap = dict(ps='PSFC',Q2='Q2',T2='T2',Ts='TSK',snow='SNOW',snowh='SNOWH', # original (WRF) names of variables
              rainnc='RAINNC',rainc='RAINC',rainsh='RAINSH',snownc='SNOWNC',graupelnc='GRAUPELNC',
              hfx='HFX',lhfx='LH',evap='QFX',OLR='OLR',GLW='GLW',SWD='SWDOWN',SWN='SWNORM') 


## variable attributes and name
# constants
constatts = dict(HGT    = dict(name='zs', units='m'), # surface elevation
                 XLONG  = dict(name='lon2D', units='deg E'), # geographic longitude field
                 XLAT   = dict(name='lat2D', units='deg N')) # geographic latitude field
constvars = constatts.keys()    
avgconstfile = 'wrfconst_%s%s.nc' # the filename needs to be extended by %(domain,'_'+grid)
# surface variables
srfcatts  = dict(T2   = dict(name='T2', units='K'), # 2m Temperature
                 Q2   = dict(name='Q2', units='Pa'), # 2m water vapor pressure
                 # rain = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
                 rainnc = dict(name='precnc', units='kg/m^2/s'), # grid-scale precipitation rate (kg/m^2/s)
                 rainc = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate (kg/m^2/s)
                 snow = dict(name='snow', units='kg/m^2'), # snow water equivalent
                 snowh = dict(name='snowh', units='m'), # snow depth
                 ps = dict(name='ps', units='Pa'), # surface pressure
                 evap = dict(name='evap', units='kg/m^2/s')) # monthly accumulated PET (kg/m^2)
srfcvars = srfcatts.keys()    
avgsrfcfile = 'wrfsrfc_%s%s_clim%s.nc' # the filename needs to be extended by %(domain,'_'+grid,'_'+period)
# axes (don't have their own file)
axesatts =  dict(time  = dict(name='time', units='days', scalefactor=1./1440.), # time coordinate
                 # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                 #       the time offset is chose such that 1979 begins with the origin (time=0)
                 x     = dict(name='x', units='m', offset=None), # projected west-east coordinate
                 y     = dict(name='y', units='m', offset=None)) # projected south-north coordinate                 
axesvars = axesatts.keys()

# the projection and grid configuration will be inferred from the source file upon loading;
# the axes variables are created on-the-fly and coordinate values are inferred from the source dimensions     


# data source/location
root_folder = data_root + 'WRF/Downscaling/' # long-term mean folder


## Functions to load different types of NARR datasets 

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'narravg/' 
# function to load these files...
def loadWRF(name=None, domain=None, period=None, grid=None, varlist=None):
  ''' Get the pre-processed monthly NARR climatology as a DatasetNetCDF. '''
  avgfolder = data_root + name + '/' # long-term mean folder
  # prepare input
  if domain not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
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
  # add method to convert precip from per month to per second
  dataset.convertPrecip = types.MethodType(convertPrecip, dataset)    
  # return formatted dataset
  return dataset


if __name__ == '__main__':
    pass