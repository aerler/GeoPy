'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

# external imports
from importlib import import_module
import numpy as np
import pickle
import os
# internal imports
from geodata.misc import AxisError, DatasetError
from geodata.base import Dataset, Variable, Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, GridDefinition

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
                       T        = dict(name='T', units='K'), # average temperature
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
  else: gridstr = '_{0:s}'.format(grid.lower()) # only use lower case for filenames 
  # period
  if isinstance(period,(tuple,list)): period = '{0:4d}-{1:4d}'.format(*period)  
  if period is None or period == '': periodstr = ''
  else: periodstr = '_{0:s}'.format(period)
  # assemble filename/list
  if filepattern is None: filepattern = name.lower() + '{0:s}_clim{1:s}.nc' 
  filename = filepattern.format(gridstr,periodstr)
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

# function to return grid definitions for some common grids
def getCommonGrid(grid, res=None):
  ''' return grid definitions of some commonly used grids '''
  # look in known datasets first
  try :
    dataset = import_module(grid)
    if res is None:
      griddef = dataset.default_grid
    else:
      griddef = dataset.grid_def[res]
  except ImportError:
    lgrid = True
    # select grid
    if grid == 'ARB_small':   slon, slat, elon, elat = -160.25, 32.75, -90.25, 72.75
    elif grid == 'ARB_large': slon, slat, elon, elat = -179.75, 3.75, -69.75, 83.75
    else: lgrid = False
    # select resolution:
    lres = True
    if res is None: res = '025' # default    
    if res == '025':   dlon = dlat = 0.25 # resolution
    elif res == '05':  dlon = dlat = 0.5
    elif res == '10':  dlon = dlat = 1.0
    elif res == '25':  dlon = dlat = 2.5
    else: lres = False
    if lgrid and lres:    
      assert (elon-slon) % dlon == 0 
      lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
      assert (elat-slat) % dlat == 0
      lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
      # add new geographic coordinate axes for projected map
      xlon = Axis(coord=lon, atts=dict(name='lon', long_name='longitude', units='deg E'))
      ylat = Axis(coord=lat, atts=dict(name='lat', long_name='latitude', units='deg N'))
      griddef = GridDefinition(name=grid, projection=None, xlon=xlon, ylat=ylat) # projection=None >> lat/lon
    else: 
      griddef = None
  # return grid definition object
  return griddef
  
# function to load pickled grid definitions
grid_folder = data_root + '/grids/' # folder for pickled grids
grid_pickle = '{0:s}_griddef.pickle' # file pattern for pickled grids
def loadPickledGridDef(grid, res=None, folder=grid_folder):
  ''' function to load pickled datasets '''
  gridstr = grid if res is None else '{0:s}_{1:s}'.format(grid,res)
  filename = '{0:s}/{1:s}'.format(folder,grid_pickle.format(gridstr))
  if os.path.exists(filename):
    filehandle = open(filename, 'r')
    griddef = pickle.load(filehandle)
    filehandle.close()
  else:
    griddef = None
  return griddef


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
#   mode = 'test_climatology'
#   mode = 'test_timeseries'
  mode = 'pickle_grid'
  grids = dict( ARB_small=['025','05','10','25'],
                ARB_large=['025','05','10','25'],
                CFSR=['031','05'],
                GPCC=['025','05','10','25'],
                CRU=[''],NARR=[''],PRISM=[''])
    
  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,reses in grids.items():
      
      if reses is None: reses = [None] # default grid
      
      for res in reses: 
      
        print('')        
        if res is None:
          gridstr = grid
          print('   ***   Pickling Grid Definition for {0:s}   ***   '.format(grid))
        else:
          gridstr = '{0:s}_{1:s}'.format(grid,res)  
          print('   ***   Pickling Grid Definition for {0:s} Resolution {1:s}   ***   '.format(grid,res))
        print('')
        
        # load GridDefinition      
        griddef = getCommonGrid(grid,res)         
        
        if griddef is None:
          print('GridDefinition object for {0:s} not found!'.format(gridstr))         
        else:
          # save pickle
          filename = '{0:s}/{1:s}'.format(grid_folder,grid_pickle.format(gridstr))
          filehandle = open(filename, 'w')
          pickle.dump(griddef, filehandle)
          filehandle.close()
          
          print('   Saving Pickle to \'{0:s}\''.format(filename))
          print('')
          
          # load pickle to make sure it is right
          del griddef
          griddef = loadPickledGridDef(grid, res=res, folder=grid_folder)
          print(griddef)
        print('')
