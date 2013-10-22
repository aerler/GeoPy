'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
# internal imports
from geodata.misc import AxisError, DatasetError
from geodata.base import Dataset, Variable
from geodata.netcdf import DatasetNetCDF, VarNC
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
  data_root = '/data/'
#  root = '/media/tmp/' # RAM disk for development
elif hostname=='cryo':
  data_root = '/scratch/marcdo/Data/'
else:
  data_root = '/home/me/DATA/PRISM/'
  

# convenience function to extract landmask variable from another masked variable
def addLandMask(dataset, varname='precip', maskname='landmask', atts=None):
  ''' Add a landmask variable with meta data from a masked variable to a dataset. '''
  # check
  if not isinstance(dataset,Dataset): raise TypeError
  if dataset.hasVariable(maskname): 
    raise DatasetError, "The Dataset '%s' already has a field called '%s'."%(dataset.name,maskname)
  # attributes and meta data
  if atts is None:
    atts = default_varatts[maskname] 
    atts['long_name'] = 'Geographic Mask for Climatology Fields' 
    atts['description'] = 'data are valid where this mask is zero'
  # axes and data
  var = dataset.variables[varname]
  axes = var.axes[-2:] # last two axes (i.e. map axes)
  data = var.getMask().__getitem__((0,)*(var.ndim-2)+(slice(None),)*2)
  if 'gdal' in dataset.__dict__ and dataset.gdal:
    if dataset.xlon not in axes or dataset.ylat not in axes: raise AxisError
  if not all([ax.name in ('x','y','lon','lat') for ax in axes]): raise AxisError
  # create variable and add to dataset
  if isinstance(dataset, DatasetNetCDF) and 'w' in dataset.mode: 
    dataset += VarNC(dataset.dataset, axes=axes, data=data, atts=atts)
  else: dataset += Variable(axes=axes, data=data, atts=atts)
  # return mask variable
  return dataset.variables[maskname]


# annotate dataset with names and length of months (for climatology mostly)
def addLengthAndNamesOfMonth(dataset, noleap=False, length=None, names=None):
  ''' Function to add the names and length of month to a NetCDF dataset. '''
  if not isinstance(dataset,Dataset): raise TypeError
  # attributes
  lenatts = dict(name='length_of_month', units='days',long_name='Length of Month')
  stratts = dict(name='name_of_month', units='', long_name='Name of the Month')
  # data
  if length is None: # leap year or no leap year
    if noleap: length = days_per_month_365
    else: length = days_per_month
  if names is None: names = name_of_month
  # create variables
  if isinstance(dataset, DatasetNetCDF) and 'w' in dataset.mode: 
    dataset += VarNC(dataset.dataset, axes=(dataset.time,), data=length, atts=lenatts)
    dataset.axisAnnotation(stratts['name'], names, 'time', atts=stratts)
  else:
    # N.B.: char/string arrays are currently not supported as Variables
    dataset += Variable(axes=(dataset.time,), data=length, atts=lenatts)
  # return length variable
  return dataset.variables[lenatts['name']]


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


# universal function to generate file names for climatologies
def getFileName(grid=None, period=None, name=None, filepattern=None):
  ''' A function to generate a standardized filename for climatology files, based on grid type and period.  '''
  if name is None: name = ''
  # grid
  if grid is None or grid == name: gridstr = ''
  else: gridstr = '_%s'%grid.lower() # only use lower case for filenames 
  # period
  if isinstance(period,(tuple,list)): period = '%4i-%4i'%tuple(period)  
  if period is None or period == '': periodstr = ''
  else: periodstr = '_%s'%period
  # assemble filename/list
  if filepattern is None: filepattern = name.lower() + '%s_clim%s.nc' 
  filename = filepattern%(gridstr,periodstr)
  # return final name
  assert filename == filename.lower(), "By convention, climatology files only have lower-case names!"
  return filename
  
  
# universal load function that will be imported by datasets
def loadClim(name, folder, period=None, grid=None, varlist=None, varatts=None, filepattern=None, filelist=None, 
             projection=None, geotransform=None, axes=None):
  ''' A function to load standardized climatology datasets. '''
  # prepare input
  # varlist (varlist = None means all variables)
  if varatts is None: varatts = default_varatts
  if varlist is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: filelist = [getFileName(grid=grid, period=period, name=name, filepattern=filepattern)]   
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          axes=axes, multifile=False, ncformat='NETCDF4')  
  dataset = addGDALtoDataset(dataset, projection=projection, geotransform=geotransform)
  # N.B.: projection should be auto-detected as geographic
  return dataset

