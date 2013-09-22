'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
# internal imports
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset

# days per month
days_per_month = np.array([31,28.2425,31,30,31,30,31,31,30,31,30,31]) # 97 leap days every 400 years
# N.B.: the Gregorian calendar repeats every 400 years
days_per_month_365 = np.array([31,28,31,30,31,30,31,31,30,31,30,31]) # no leap day
# human-readable names
name_of_month = ['January  ', 'February ', 'March    ', 'April    ', 'May      ', 'June     ', #
                 'July     ', 'August   ', 'September', 'October  ', 'November ', 'December ']


# attributes for variables in standardized climatologies 
# variable attributes and name
default_varatts = dict(pmsl     = dict(name='pmsl', units='Pa'), # sea-level pressure
                       ps       = dict(name='ps', units='Pa'), # surface pressure
                       Ts       = dict(name='Ts', units='K'), # average skin temperature
                       T2       = dict(name='T2', units='K'), # 2m average temperature
                       Tmin     = dict(name='Tmin', units='K'), # 2m minimum temperature
                       Tmax     = dict(name='Tmax', units='K'), # 2m maximum temperature
                       Q2       = dict(name='Q2', units='Pa'), # 2m water vapor pressure
                       pet      = dict(name='pet', units='kg/m^2/s'), # potential evapo-transpiration
                       precip   = dict(name='precip', units='kg/m^2/s'), # total precipitation
                       pwtr     = dict(name='pwtr', units='kg/m^2'), # total precipitable water (kg/m^2)
                       snow     = dict(name='snow', units='kg/m^2'), # snow water equivalent
                       snowh    = dict(name='snowh', units='m'), # snow depth
                       stations = dict(name='stations', units='#'), # number of gauges for observation
                       zs       = dict(name='zs', units='m'), # surface elevation
                       landmask = dict(name='landmask', units=''), # land mask
                       lon2D    = dict(name='lon2D', units='deg E'), # geographic longitude field
                       lat2D    = dict(name='lat2D', units='deg N'), # geographic latitude field
                       # axes (don't have their own file; listed in axes)
                       time     = dict(name='time', units='month'), # time coordinate for climatology
                       lon      = dict(name='lon', units='deg E'), # geographic longitude field
                       lat      = dict(name='lat', units='deg N'), # geographic latitude field
                       x        = dict(name='x', units='m'), # projected west-east coordinate
                       y        = dict(name='y', units='m')) # projected south-north coordinate                                


# data root folder
import socket
hostname = socket.gethostname()
if hostname=='komputer':
  data_root = '/home/DATA/DATA/'
#  root = '/media/tmp/' # RAM disk for development
else:
  data_root = '/home/me/DATA/PRISM/'


# convenience function to invert variable name mappings
def translateVarNames(varlist, varatts):
  ''' Simple function to replace names in a variable list with their original names as inferred from the 
      attributes dictionary. Note that this requires the dictionary to have the field 'name'. '''
  if not isinstance(varlist,list) or not isinstance(varatts,dict): raise TypeError  
  # cycle over names in variable attributes (i.e. final names, not original names)  
  for key,atts in varatts.iteritems():
    if 'name' in atts and atts['name'] in varlist: 
      varlist[varlist.index(atts['name'])] = key # original name is used as key in the attributes dict
  # return varlist with final names replaced by original names
  return varlist


# universal load function that will be imported by datasets
def loadClim(name, folder, projection=None, period=None, grid=None, varlist=None, varatts=None, filepattern=None, filelist=None):
  ''' A function to load standardized climatology datasets. '''
  # prepare input
  # varlist (varlist = None means all variables)
  if varatts is None: varatts = default_varatts
  if varlist is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: 
    # grid
    if grid is None or grid == name: gridstr = ''
    else: gridstr = '_%s'%grid 
    # period
    if isinstance(period,(tuple,list)): period = '%4i-%4i'%tuple(period)  
    if period is None or period == '': periodstr = ''
    else: periodstr = '_%s'%period
    # assemble filename/list
    if filepattern is None: filepattern = name.lower() + '%s_clim%s.nc' 
    filelist = [filepattern%(gridstr,periodstr)] 
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          multifile=False, ncformat='NETCDF4')  
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  return dataset

