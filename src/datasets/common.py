'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

# external imports
from importlib import import_module
import inspect
import numpy as np
import pickle
import os
from operator import isCallable
# internal imports
from utils.misc import expandArgumentList
from geodata.misc import AxisError, DatasetError, DateError, ArgumentError
from geodata.base import Dataset, Variable, Axis, Ensemble
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import GDALError, addGDALtoDataset, GridDefinition, loadPickledGridDef, griddef_pickle
# import some calendar defintions
from geodata.misc import name_of_month, days_per_month, days_per_month_365, seconds_per_month, seconds_per_month_365
from matplotlib.delaunay.testfuncs import plotallfuncs




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
                       evap     = dict(name='evap', units='kg/m^2/s'), # actual evapo-transpiration
                       precip   = dict(name='precip', units='kg/m^2/s'), # total precipitation                       
                       solprec  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation
                       liqprec  = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation
                       pwtr     = dict(name='pwtr', units='kg/m^2'), # total precipitable water (kg/m^2)
                       snow     = dict(name='snow', units='kg/m^2'), # snow water equivalent
                       snowh    = dict(name='snowh', units='m'), # snow depth
                       sfroff   = dict(name='sfroff', units='kg/m^2/s'), # surface run-off                      
                       ugroff   = dict(name='ugroff', units='kg/m^2/s'), # sub-surface/underground run-off      
                       runoff   = dict(name='runoff', units='kg/m^2/s'), # total surface and sub-surface run-off
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
default_varatts['p-et'] = dict(name='p-et', units='kg/m^2/s') # net precipitation; only legal as a string                                

# parameters used in shape files
shp_params = ['shape_name','shp_long_name','shp_type','shp_mask','shp_area','shp_encl','shp_full','shp_empty']
# parameters used in station files
stn_params = ['station_name', 'stn_prov', 'stn_rec_len', 'zs_err', 'stn_lat', 'stn_lon', 'cluster_id']

# data root folder
import socket
hostname = socket.gethostname()
if hostname=='komputer':
  data_root = '/data/'  
#  root = '/media/tmp/' # RAM disk for development
elif hostname=='cryo':
  data_root = '/scratch/marcdo/Data/'
elif hostname=='erlkoenig':
  data_root = '/media/me/data-2/Data/'
else:
  raise NotImplementedError, "No 'data_root' folder set!"
# standard folder for grids and shapefiles  
grid_folder = data_root + '/grids/' # folder for pickled grids
shape_folder = data_root + '/shapes/' # folder for pickled grids


## utility functions for datasets

# convenience method to convert a period tuple into a monthly coordinate tuple 
def timeSlice(period):
  ''' convenience method to convert a period tuple into a monthly coordinate tuple '''
  return (period[0]-1979)*12, (period[1]-1979)*12-1 


# convenience function to extract landmask variable from another masked variable
def addLandMask(dataset, varname='precip', maskname='landmask', atts=None):
  ''' Add a landmask variable with meta data from a masked variable to a dataset. '''
  # check
  if not isinstance(dataset,Dataset): raise TypeError
  if dataset.hasVariable(maskname): 
    raise DatasetError, "The Dataset '%s' already has a field called '%s'."%(dataset.name,maskname)
  # attributes and meta data
  if atts is None:
    atts = default_varatts[maskname].copy()
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
    dataset.addVariable(Variable(axes=axes, name=maskname, data=data, atts=atts), asNC=True)
  else: dataset.addVariable(Variable(axes=axes, name=maskname, data=data, atts=atts))
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
    dataset.addVariable(Variable(axes=(dataset.time,), data=length, atts=lenatts), asNC=True)
    dataset.addVariable(Variable(axes=(dataset.time,), data=names, atts=stratts), asNC=True)
  else:
    # N.B.: char/string arrays are currently not supported as Variables
    dataset.addVariable(Variable(axes=(dataset.time,), data=length, atts=lenatts))
    dataset.addVariable(Variable(axes=(dataset.time,), data=names, atts=stratts))
  # return length variable
  return dataset.variables[lenatts['name']]


# helper function to convert monthly precip amount into precip rate
# def convertPrecip(precip):
#   ''' convert monthly precip amount to SI units (mm/s) '''
#   warn("Use of method 'convertPrecip' is depricated; use the on-the-fly transformPrecip function instead")
#   if precip.units == 'kg/m^2/month' or precip.units == 'mm/month':
#     precip /= (days_per_month.reshape((12,1,1)) * 86400.) # convert in-place
#     precip.units = 'kg/m^2/s'
#   return precip


# transform function to convert monthly precip amount into precip rate on-the-fly
def transformPrecip(data, l365=False, var=None, slc=None):
  ''' convert monthly precip amount to SI units (mm/s) '''
  if not isinstance(var,VarNC): raise TypeError
  if var.units == 'kg/m^2/month' or var.units == 'mm/month':
    assert data.ndim == var.ndim
    tax = var.axisIndex('time')
    # expand slices
    if slc is None or isinstance(slc,slice): tslc = slc
    elif isinstance(slc,(list,tuple)): tslc = slc[tax]
    # handle sliced or non-sliced axis
    if tslc is None or tslc == slice(None):
      # trivial case
      te = len(var.time)
      if not ( data.shape[tax] == te and te%12 == 0 ): raise NotImplementedError, "The record has to start and end at a full year!"
    else:  
      # special treatment if time axis was sliced
      tlc = slc[tax]
      ts = tlc.start or 0 
      te = ( tlc.stop or len(var.time) ) - ts
      if not ( ts%12 == 0 and te%12 == 0 ): raise NotImplementedError, "The record has to start and end at a full year!"
      assert data.shape[tax] == te
      # assuming the record starts some year in January, and we always need to load full years
    shape = [1,]*data.ndim; shape[tax] = te # dimensions of length 1 will be expanded as needed
    spm = seconds_per_month_365 if l365 else seconds_per_month
    data /= np.tile(spm, te/12).reshape(shape) # convert in-place
    var.units = 'kg/m^2/s'
  return data      
      
# transform function to convert days per month into a ratio
def transformDays(data, l365=False, var=None, slc=None):
  ''' convert days per month to fraction '''
  if not isinstance(var,VarNC): raise TypeError
  if var.units == 'days':
    assert data.ndim == var.ndim
    tax = var.axisIndex('time')
    # expand slices
    if slc is None or isinstance(slc,slice): tslc = slc
    elif isinstance(slc,(list,tuple)): tslc = slc[tax]
    # handle sliced or non-sliced axis
    if tslc is None or tslc == slice(None):
      # trivial case
      te = len(var.time)
      if not ( data.shape[tax] == te and te%12 == 0 ): 
        raise NotImplementedError, "The record has to start and end at a full year!"
    else:  
      # special treatment if time axis was sliced
      tlc = slc[tax]
      ts = tlc.start or 0 
      te = ( tlc.stop or len(var.time) ) - ts
      if not ( ts%12 == 0 and te%12 == 0 ): 
        raise NotImplementedError, "The record has to start and end at a full year!"
      assert data.shape[tax] == te
      # assuming the record starts some year in January, and we always need to load full years
    shape = [1,]*data.ndim; shape[tax] = te # dimensions of length 1 will be expanded as needed
    spm = days_per_month_365 if l365 else days_per_month
    data /= np.tile(spm, te/12).reshape(shape) # convert in-place
    var.units = '' # fraction
  return data      
      
      
## functions to load a dataset

# convenience function to invert variable name mappings
def translateVarNames(varlist, varatts):
  ''' Simple function to replace names in a variable list with their original names as inferred from the 
      attributes dictionary. Note that this requires the dictionary to have the field 'name'. '''
  if isinstance(varlist,basestring): varlist = [varlist]
  if not isinstance(varlist,(list,tuple,set)) or not isinstance(varatts,dict): raise TypeError 
  varlist = list(varlist) # make copy, since operation is in-place 
  # cycle over names in variable attributes (i.e. final names, not original names)  
  for key,atts in varatts.iteritems():
    if 'name' in atts and atts['name'] in varlist: varlist.append(key)
#       varlist[varlist.index(atts['name'])] = key # original name is used as key in the attributes dict
  # return varlist with final names replaced by original names
  return varlist


# universal function to generate file names for climatologies and time-series
def getFileName(name=None, resolution=None, period=None, filetype='climatology', grid=None, filepattern=None):
  ''' A function to generate a standardized filename for climatology and time-series files, based on grid type and period.  '''
  if name is None: name = ''
  # grid (this is a *non-native grid*)
  if grid is None or grid == name: gridstr = ''
  else: gridstr = '_{0:s}'.format(grid.lower()) # only use lower case for filenames
  # resolution is the native resolution (behind dataset name, prepended to the grid 
  if resolution: gridstr = '_{0:s}{1:s}'.format(resolution,gridstr)
  # period
  if filetype == 'time-series':
    # assemble filename
    if filepattern is None: filepattern = name.lower() + '{0:s}_monthly.nc' 
    filename = filepattern.format(gridstr)
  elif filetype == 'climatology':
    if isinstance(period,(tuple,list)): pass
    elif isinstance(period,basestring): pass
    elif period is None: pass
    elif isinstance(period,(int,np.integer)):
      period = (1979, 1979+period)
    else: raise DateError   
    if period is None or period == '': periodstr = ''
    elif isinstance(period,basestring): periodstr = '_{0:s}'.format(period)
    else: periodstr = '_{0:4d}-{1:4d}'.format(*period)  
    # assemble filename
    if filepattern is None: filepattern = name.lower() + '{0:s}_clim{1:s}.nc' 
    filename = filepattern.format(gridstr,periodstr)
  else: raise NotImplementedError, "Unrecognized filetype: '{:s}'".format(filetype)
  # return final name
  assert filename == filename.lower(), "By convention, climatology files only have lower-case names!"
  return filename
  
  
# common climatology load function that will be imported by datasets (for backwards compatibility)
def loadObs(name=None, folder=None, resolution=None, period=None, grid=None, varlist=None, 
             varatts=None, filepattern=None, filelist=None, projection=None, geotransform=None, 
             axes=None, lautoregrid=None):
  ''' A function to load standardized observational climatologies. '''
  return loadObservations(name=name, folder=folder, resolution=resolution, period=period, grid=grid, station=None, 
                          varlist=varlist, varatts=varatts, filepattern=filepattern, filelist=filelist, 
                          projection=projection, geotransform=geotransform, axes=axes, 
                          lautoregrid=lautoregrid, mode='climatology')

# common climatology load function that will be imported by datasets (for backwards compatibility)
def loadObs_StnTS(name=None, folder=None, resolution=None, varlist=None, station=None, 
                  varatts=None, filepattern=None, filelist=None, axes=None):
    ''' A function to load standardized observational time-series at station locations. '''
    return loadObservations(name=name, folder=folder, resolution=resolution, station=station, 
                          varlist=varlist, varatts=varatts, filepattern=filepattern, filelist=filelist, 
                          projection=None, geotransform=None, axes=axes, period=None, grid=None,
                          lautoregrid=False, mode='time-series')
  
# universal load function that will be imported by datasets
def loadObservations(name=None, folder=None, period=None, grid=None, station=None, shape=None, lencl=False, 
                     varlist=None, varatts=None, filepattern=None, filelist=None, resolution=None, 
                     projection=None, geotransform=None, axes=None, lautoregrid=None, mode='climatology'):
  ''' A function to load standardized observational datasets. '''
  # prepare input
  if mode.lower() == 'climatology': # post-processed climatology files
    # transform period
    if period is None or period == '':
      if name not in ('PCIC','PRISM','GPCC','NARR'): 
        raise ValueError, "A period is required to load observational climatologies."
    elif isinstance(period,basestring):
      period = tuple([int(prd) for prd in period.split('-')]) 
    elif not isinstance(period,(int,np.integer)) and ( not isinstance(period,tuple) and len(period) == 2 ): 
      raise TypeError
  elif mode.lower() in ('time-series','timeseries'): # concatenated time-series files
    period = None # to indicate time-series (but for safety, the input must be more explicit)
    if lautoregrid is None: lautoregrid = False # this can take very long!
  # figure out station and shape options
  if station and shape: raise ArgumentError
  elif station or shape: 
    if grid is not None: raise NotImplementedError, 'Currently observational station data can only be loaded from the native grid.'
    if lautoregrid: raise GDALError, 'Station data can not be regridded, since it is not map data.'   
    lstation = bool(station); lshape = bool(shape)
    grid = station if lstation else shape
  else:
    lstation = False; lshape = False
  # varlist (varlist = None means all variables)
  if varatts is None: varatts = default_varatts.copy()
  if varlist is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: 
    filename = getFileName(name=name, resolution=resolution, period=period, grid=grid, filepattern=filepattern)
    # check existance
    filepath = '{:s}/{:s}'.format(folder,filename)
    if not os.path.exists(filepath):
      nativename = getFileName(name=name, resolution=resolution, period=period, grid=None, filepattern=filepattern)
      nativepath = '{:s}/{:s}'.format(folder,nativename)
      if os.path.exists(nativepath):
        if lautoregrid: 
          from processing.regrid import performRegridding # causes circular reference if imported earlier
          griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder)
          dataargs = dict(period=period, resolution=resolution)
          performRegridding(name, 'climatology',griddef, dataargs) # default kwargs
        else: raise IOError, "The dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.".format(filename,grid) 
      else: raise IOError, "The dataset file '{:s}' does not exits!\n('{:s}')".format(filename,filepath)
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=[filename], varlist=varlist, varatts=varatts, 
                          axes=axes, multifile=False, ncformat='NETCDF4')
  # mask all shapes that are incomplete in dataset
  if shape and lencl and 'shp_encl' in dataset: 
    dataset.load() # need to load data before masking; is cheap for shape averages, anyway
    dataset.mask(mask='shp_encl', invert=True, skiplist=shp_params)
  # correct ordinal number of shape (should start at 1, not 0)
  if lshape:
    if dataset.hasAxis('shapes'): raise AxisError, "Axis 'shapes' should be renamed to 'shape'!"
    if not dataset.hasAxis('shape'): 
      raise AxisError
    if dataset.shape.coord[0] == 0: dataset.shape.coord += 1
# figure out grid
  if not lstation and not lshape:
    if grid is None or grid == name:
      dataset = addGDALtoDataset(dataset, projection=projection, geotransform=geotransform, gridfolder=grid_folder)
    elif isinstance(grid,basestring): # load from pickle file
  #     griddef = loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder)
      # add GDAL functionality to dataset 
      dataset = addGDALtoDataset(dataset, griddef=grid, gridfolder=grid_folder)
    else: raise TypeError
    # N.B.: projection should be auto-detected, if geographic (lat/lon)
  return dataset


## functions to load multiple datasets

# decorator class for batch-loading datasets into an ensemble using a custom load function
class BatchLoad(object):
  ''' A decorator class that wraps custom functions to load specific datasets. List arguments can be
      expanded to load multiple datasets and places them in a list or Ensemble. 
      Keyword arguments are passed on to the dataset load functions; arguments listed in load_list 
      are applied to the datasets according to expansion rules, otherwise they are applied to all. '''
  
  def __init__(self, load_fct):
    ''' initialize wrapping of original operation '''
    self.load_fct = load_fct
    
  def __call__(self, load_list=None, lproduct='outer', lensemble=None, ens_name=None, ens_title=None, **kwargs):
    ''' wrap original function: expand argument list, execute load_fct over argument list, 
        and return a list or Ensemble of datasets '''
    # decide, what to do
    if load_list is None:
      # normal operation: no expansion      
      datasets =  self.load_fct(**kwargs)
    else:
      # expansion required
      lensemble = ens_name is not None if lensemble is None else lensemble
      # figure out arguments
      kwargs_list = expandArgumentList(expand_list=load_list, lproduct=lproduct, **kwargs)
      # load datasets
      datasets = []
      for kwargs in kwargs_list:    
        # load dataset
        datasets.append(self.load_fct(**kwargs))    
      # construct ensemble
      if lensemble:
        datasets = Ensemble(members=datasets, name=ens_name, title=ens_title, basetype='Dataset')
    # return list or ensemble of datasets
    return datasets


# convenience shortcut to load only climatologies 
@BatchLoad
def loadClim(name=None, **kwargs):
  ''' A function to load any standardized climatologies; identifies source by name heuristics '''
  return loadDataset(name=name, station=None, mode='climatology', **kwargs)

# convenience shortcut to load only staton time-series
@BatchLoad
def loadStnTS(name=None, station=None, **kwargs):
    ''' A function to load any standardized time-series at station locations. '''
    return loadDataset(name=name, station=station, shape=None, mode='time-series', **kwargs)

# convenience shortcut to load only regionally averaged time-series
@BatchLoad
def loadShpTS(name=None, shape=None, **kwargs):
    ''' A function to load any standardized time-series averaged over regions. '''
    return loadDataset(name=name, station=None, shape=shape, mode='time-series', **kwargs)
  
# universal load function that will be imported by datasets
@BatchLoad
def loadDataset(name=None, station=None, shape=None, mode='climatology', **kwargs):
  ''' A function to load any datasets; identifies source by name heuristics. '''
  # some private imports (prevent import errors)  
  from projects.WRF_experiments import WRF_exps, WRF_experiments
  from datasets.CESM import CESM_exps, CESM_experiments
  orig_name = name
  # identify dataset source
  lensemble = False; lobs = False
  if mode.upper() == 'CVDP':
    # this is a special case for observational data in the CVDP package
    if name.lower() == 'obs': name = 'HadISST' # default to SST modes
    elif name.lower() in ('hadisst','mlost','20thc_reanv2','gpcp'): lobs = True
    elif name.isupper(): lobs = True; name = 'HadISST' # load ocean modes for comparison 
    # resolve WRF experiments to parent CESM runs for CVDP
    elif name in WRF_exps: name = WRF_exps[name].parent
    elif name in WRF_experiments: name = WRF_experiments[name].parent
    elif name[:-4] in WRF_exps: name = WRF_exps[name[:-4]].parent
    elif name[:-4] in WRF_experiments: name = WRF_experiments[name[:-4]].parent
    elif not name in CESM_exps or name in CESM_experiments: 
      raise ArgumentError, "No CVDP dataset matching '{:s}' found.".format(name)
    # nothing to do for CESM runs
    dataset_name = 'CESM'
    import datasets.CESM as dataset # also in CESM module
  elif ( name in WRF_exps or name in WRF_experiments or 
       name[:-4] in WRF_exps or name[:-4] in WRF_experiments ):
    # this is most likely a WRF experiment or ensemble
    import datasets.WRF as dataset
    from projects.WRF_experiments import WRF_ens
    dataset_name = 'WRF'    
    lensemble = name in WRF_ens or name[:-4] in WRF_ens
  elif name in CESM_exps or name in CESM_experiments:
    # this is most likely a CESM experiment or ensemble
    import datasets.CESM as dataset
    from datasets.CESM import CESM_ens
    dataset_name = 'CESM'
    lensemble = name in  CESM_ens
  else:
    # this is most likely an observational dataset
    if name[:3].lower() == 'obs': dataset_name = 'Unity' # alias... 
    else: dataset_name = name 
    try: dataset = import_module('datasets.{0:s}'.format(dataset_name))
    except ImportError: raise ArgumentError, "No dataset matching '{:s}' found.".format(dataset_name)
  # identify load function  
  if mode.upper() in ('CVDP',):
    load_fct = 'loadCVDP'
    if lobs: load_fct += '_Obs'
  else:
    load_fct = 'load{:s}'.format(dataset_name)
    if mode.lower() in ('climatology',):
      if lensemble and station: raise ArgumentError
      if station: load_fct += '_Stn'
      elif shape: load_fct += '_Shp'
    elif mode.lower() in ('time-series','timeseries',):
      if lensemble:
        if station: load_fct += '_StnEns'
        elif shape: load_fct += '_ShpEns'
        else: load_fct += '_Ensemble'
      else:
        if station: load_fct += '_StnTS'
        elif shape: load_fct += '_ShpTS'
        else: load_fct += '_TS'      
  # load dataset
  if load_fct in dataset.__dict__: 
    load_fct = dataset.__dict__[load_fct]
  else: 
    raise ArgumentError, "Dataset '{:s}' has no method '{:s}'".format(dataset_name,load_fct)
  if not inspect.isfunction(load_fct): 
    raise ArgumentError, "Attribute '{:s}' in module '{:s}' is not a function".format(load_fct.__name__,dataset_name)
  # generate and check arguments
  kwargs.update(name=name, station=station, shape=shape, mode=mode)
  argspec, varargs, keywords, defaults = inspect.getargspec(load_fct); del varargs, keywords, defaults
  kwargs = {key:value for key,value in kwargs.iteritems() if key in argspec}
  # load dataset
  dataset = load_fct(**kwargs)
  if orig_name == name: 
    if dataset.name != name: raise DatasetError, load_fct.__name__
  else: dataset.name = orig_name
  # return dataset
  return dataset


# function to extract common points that meet a specific criterion from a list of datasets
def selectElements(datasets, axis, testFct=None, imaster=None, linplace=True, lall=False):
  ''' Extract common points that meet a specific criterion from a list of datasets. 
      The test function has to accept the following input: index, dataset, axis'''
  # check input
  if not isinstance(datasets, (list,tuple,Ensemble)): raise TypeError
  if not all(isinstance(dataset,Dataset) for dataset in datasets): raise TypeError 
  if not isCallable(testFct) and testFct is not None: raise TypeError
  if isinstance(axis, Axis): axis = axis.name
  if not isinstance(axis, basestring): raise TypeError
  if lall and imaster is not None: raise ArgumentError, "The options 'lall' and 'imaster' are mutually exclusive!"
  # save some ensemble parameters for later  
  lnotest = testFct is None
  lens = isinstance(datasets,Ensemble)
  if lens:
    enskwargs = dict(basetype=datasets.basetype, idkey=datasets.idkey, 
                     name=datasets.name, title=datasets.title) 
  # use dataset with shortest axis as master sample (more efficient)
  axes = [dataset.getAxis(axis) for dataset in datasets]
  if imaster is None: imaster = np.argmin([len(ax) for ax in axes]) # find shortest axis
  elif not isinstance(imaster,(int,np.integer)): raise TypeError
  elif imaster >= len(datasets) or imaster < 0: raise ValueError 
  maxis = axes.pop(imaster) # extraxt shortest axis for loop
  if lall: 
    tmpds = tuple(datasets)
    if imaster != 0: tmpds = (tmpds[imaster],)+tmpds[:imaster]+tmpds[imaster+1:]
    test_fct = lambda i,ds: testFct(i, ds, axis) # prepare test function arguments
  else: 
    test_fct = lambda i: testFct(i, datasets[imaster], axis) 
  # loop over coordinate axis
  itpls = [] # list of valid index tuple
  for i,x in enumerate(maxis.coord):
    # check other axes
    if all([x in ax.coord for ax in axes]): # only the other axes
      # no condition
      if lnotest:
        # just find and add indices
        itpls.append((i,)+tuple(ax.coord.searchsorted(x) for ax in axes))
      # check condition using shortest dataset
      elif lall: 
        # check test condition on all datasets (slower)
        tmpidx = (i,)+tuple(ax.coord.searchsorted(x) for ax in axes)
        if all(test_fct(ii,ds) for ii,ds in zip(tmpidx,tmpds)):
          # add corresponding indices in each dataset to list
          itpls.append(tmpidx)
      else:
        # check test condition on only one dataset (faster, default)
        if test_fct(i):
          # add corresponding indices in each dataset to list
          itpls.append((i,)+tuple(ax.coord.searchsorted(x) for ax in axes))
          # N.B.: since we can expect exact matches, plain searchsorted is fastest (side='left') 
  # check if there is anything left...
  if len(itpls) == 0: raise DatasetError, "Aborting: no data points match all criteria!"
  # construct axis indices for each dataset (need to remember to move shortest axis back in line)
  idxs = [[] for ds in datasets] # create unique empty lists
  for itpl in itpls:
    for i,idx in enumerate(itpl): idxs[i].append(idx)
  idxs.insert(imaster,idxs.pop(0)) # move first element back in line (where shortest axis was)
  idxs = [np.asarray(idxlst, dtype='int') for idxlst in idxs]      
  # slice datasets using only positive results  
  datasets = [ds(lidx=True, linplace=linplace, **{axis:idx}) for ds,idx in zip(datasets,idxs)]
  if lens: datasets = Ensemble(*datasets, **enskwargs)
  # return datasets
  return datasets


# a function to load station data
@BatchLoad
def loadEnsembleTS(names=None, name=None, title=None, varlist=None, aggregation=None, season=None, 
                   slices=None, obsslices=None, reduction=None, shape=None, station=None, prov=None, 
                   constraints=None, filetypes=None, domain=None, ldataset=False, lcheckVar=False, 
                   lwrite=False, name_tags=None, dataset_mode='time-series', 
                   ensemble_list=None, ensemble_product='inner', **kwargs):
  ''' a convenience function to load an ensemble of time-series, based on certain criteria; works 
      with either stations or regions; seasonal/climatological aggregation is also supported '''
  # prepare ensemble
  if varlist is not None:
    varlist = list(varlist)[:] # copy list
    if station: 
      for var in stn_params: # necessary to select stations
        if var not in varlist: varlist.append(var)
    if shape: 
      for var in shp_params: # necessary to select shapes
        if var not in varlist: varlist.append(var)
  # perpare ensemble and arguments
  if ldataset and ensemble_list: raise ArgumentError 
  elif not ldataset: ensemble = Ensemble(name=name, title=title, basetype='Dataset')
  # expand argument list
  if ensemble_list is None: ensemble_list = ['names'] if not ldataset else None
  loadargs = expandArgumentList(names=names, station=station, prov=prov, shape=shape, varlist=varlist, 
                                mode=dataset_mode, filetypes=filetypes, domains=domain, lwrite=lwrite,
                                slices=slices, obsslices=obsslices, name_tags=name_tags, 
                                expand_list=ensemble_list, lproduct=ensemble_product)
  for loadarg in loadargs:
    # clean up argumetns
    name = loadarg.pop('names',None); name_tag = loadarg.pop('name_tags',None)
    slcs = loadarg.pop('slices',None); obsslcs = loadarg.pop('obsslices',None)    
    # load individual dataset
    dataset = loadDataset(name=name, **loadarg)
    if name_tag is not None: 
      if name_tag[0] == '_': dataset.name += name_tag
      else: dataset.name = name_tag
    # apply slicing
    if obsslcs and ( dataset.name[:3].lower() == 'obs' or dataset.name.isupper() ):
      if slcs is None: slcs = obsslcs
      else: slcs.update(**obsslcs) # add special slices for obs
      # N.B.: currently VarNC's can only be sliced once, because we can't combine slices yet
    if slcs: dataset = dataset(**slcs) # slice immediately 
    if not ldataset: ensemble += dataset.load() # load data and add to ensemble
  # if input was not a list, just return dataset
  if ldataset: ensemble = dataset.load() # load data
  # select specific stations (if applicable)
  if not ldataset and station and constraints:
    from datasets.EC import selectStations
    ensemble = selectStations(ensemble, stnaxis='station', imaster=None, linplace=False, lall=True,
                              lcheckVar=lcheckVar, **constraints)
  # make sure all have cluster meta data  
  for varname in stn_params + shp_params:
    # find valid instance
    var = None
    for ds in ensemble: 
      if varname in ds: var = ds[varname]; break
    # give to those who have not
    if var is not None:
      var.load() # load data and add as regular variable (not VarNC)
      for ds in ensemble: 
        if varname not in ds: ds.addVariable(var.copy()) 
  # apply general reduction operations
  if reduction is not None:
    for ax,op in reduction.iteritems():
      ensemble = getattr(ensemble,op)(axis=ax)
  # extract seasonal/climatological values/extrema
  # N.B.: the operations below should work with Ensembles as well as Datasets 
  if aggregation:
    if season is not None:
      ensemble = getattr(ensemble,'seasonal'+aggregation.title())(season=season, taxis='time', **kwargs)
    else:
      ensemble = getattr(ensemble,'clim'+aggregation.title())(taxis='time', **kwargs)
  elif season: # but not aggregation
    ensemble = ensemble.extractSeason(season=season)
  # return dataset
  return ensemble


## Miscellaneous utility functions

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
      gridstr = '{0:s}_{1:s}'.format(grid,res) if res is not None else grid
      griddef = GridDefinition(name=gridstr, projection=None, xlon=xlon, ylat=ylat) # projection=None >> lat/lon
    else: 
      griddef = None
  # return grid definition object
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
                CRU=[None],NARR=[None],PRISM=[None],PCIC=[None])
  #grids = dict( CFSR=['031','05'],)
    
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
          filename = '{0:s}/{1:s}'.format(grid_folder,griddef_pickle.format(gridstr))
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
