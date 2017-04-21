'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

# external imports
from importlib import import_module
from warnings import warn
import inspect
import numpy as np
import os
import functools
# internal imports
from utils.misc import expandArgumentList
from geodata.misc import AxisError, DatasetError, DateError, ArgumentError, EmptyDatasetError, DataError,\
  VariableError
from geodata.base import Dataset, Variable, Axis, Ensemble
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import GDALError, addGDALtoDataset, loadPickledGridDef, grid_folder, shape_folder, data_root
# import some calendar definitions
from geodata.misc import name_of_month, days_per_month, days_per_month_365, seconds_per_month, seconds_per_month_365


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
                       snwmlt   = dict(name='snwmlt', units='kg/m^2/s'), # snow melt (rate)
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
shp_params = ['shape_name','shp_long_name','shp_type','shp_area','shp_encl','shp_full','shp_empty']
# N.B.: 'shp_mask' should not be loaded by default, because it can not be concatenated, if the grid is different 
# parameters used in station files
stn_params = ['station_name', 'stn_prov', 'zs_err', 'stn_zs', 'stn_lat', 'stn_lon', 'stn_rec_len', 'stn_begin_date', 'stn_end_date']
# variables contained in the CRU dataset
CRU_vars = ['T2','Tmin','Tmax','Q2','pet','precip','cldfrc','wetfrq','frzfrq']
# list of reanalysis, station, and gridded observational datasets currently available
reanalysis_datasets = ['CFSR','NARR']
station_obs_datasets = ['EC','GHCN','WSC']
gridded_obs_datasets = ['CRU','GPCC','NRCan','PCIC','PRISM','Unity']
observational_datasets = reanalysis_datasets + station_obs_datasets + gridded_obs_datasets
timeseries_datasets = ['CFSR','NARR','EC','GHCN','WSC','CRU','GPCC',]


## utility functions for datasets


def addLoadFcts(namespace, dataset, comment=" (Experiment and Ensemble lists are already set.)", **kwargs):
  ''' function to add dataset load functions to the local namespace, which already have a fixed experiments dictionary '''
  # search namespace for load functions
  vardict = dataset if isinstance(dataset, dict) else dataset.__dict__
  for name,fct in vardict.iteritems():
    if inspect.isfunction(fct) and name[:4] == 'load':
      # check valid arguments (omit others)
      arglist = inspect.getargs(fct.func_code)
      arglist = None if arglist[2] is not None else arglist[0]
      fctargs = {key:value for key,value in kwargs.iteritems() if arglist is None or  key in arglist}
      # apply arguments and update doc-string
      newfct = functools.partial(fct, **fctargs)
      newfct.__doc__ = fct.__doc__ + comment # copy doc-string with comment
      namespace[name] = newfct
  # not really necessary, since dicts are passed by reference
  return namespace


def nullNaN(data, var=None, slc=None):
  ''' transform function to remove month with no precip from data (replace by NaN) '''
  return np.where(data == 0., np.NaN, data)
  

# convenience method to convert a period tuple into a monthly coordinate tuple 
def timeSlice(period):
  ''' convenience method to convert a period tuple into a monthly coordinate tuple '''
  return (period[0]-1979)*12, (period[1]-1979)*12-1 


# convenience function to extract landmask variable from another masked variable
def addLandMask(dataset, varname='precip', maskname='landmask', atts=None):
  ''' Add a landmask variable with meta data from a masked variable to a dataset. '''
  # check
  if not isinstance(dataset,Dataset): raise TypeError(dataset)
  if dataset.hasVariable(maskname): 
    raise DatasetError("The Dataset '%s' already has a field called '%s'."%(dataset.name,maskname))
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
    if dataset.xlon not in axes or dataset.ylat not in axes: raise AxisError(dataset.axes)
  if not all([ax.name in ('x','y','lon','lat') for ax in axes]): raise AxisError(dataset.axes)
  # create variable and add to dataset
  if isinstance(dataset, DatasetNetCDF) and 'w' in dataset.mode: 
    dataset.addVariable(Variable(axes=axes, name=maskname, data=data, atts=atts), asNC=True)
  else: dataset.addVariable(Variable(axes=axes, name=maskname, data=data, atts=atts))
  # return mask variable
  return dataset.variables[maskname]


# annotate dataset with names and length of months (for climatology mostly)
def addLengthAndNamesOfMonth(dataset, noleap=False, length=None, names=None):
  ''' Function to add the names and length of month to a NetCDF dataset. '''
  if not isinstance(dataset,Dataset): raise TypeError(dataset)
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


# apply a periodic monthly scalefactor or offset
def monthlyTransform(var=None, data=None, slc=None, time_axis='time', lvar=None, scalefactor=None, offset=None, linplace=True):
    ''' apply a periodic monthly scalefactor or offset to a variable '''
    # check input
    if data is None: 
        data = var.data_array
        if lvar is None: lvar = True
    elif not isinstance(data,np.ndarray): raise TypeError(data)
    if slc is None and ( data.ndim != var.ndim or data.shape != var.shape ):
        raise DataError("Dimensions of data array and Variable are incompatible!\n {} != {}".format(data.shape,var.shape))
    tax = var.axisIndex(time_axis)
    if scalefactor is not None and not len(scalefactor) == 12: 
        raise ArgumentError("the 'scalefactor' array/list needs to have length 12 (one entry for each month).\n {}".format(scalefactor))
    if offset is not None and not len(offset) == 12: 
        raise ArgumentError("the 'offset' array/list needs to have length 12 (one entry for each month).\n {}".format(offset))
    # expand slices
    if slc is None or isinstance(slc,slice): tslc = slc
    elif isinstance(slc,(list,tuple)): tslc = slc[tax]
    # handle sliced or non-sliced axis
    if tslc is None or tslc == slice(None):
        # trivial case
        te = len(var.axes[tax])
        if not ( data.shape[tax] == te and te%12 == 0 ): 
          raise NotImplementedError("The record has to start and end at a full year!")
    else:  
        # special treatment if time axis was sliced
        tlc = slc[tax]
        ts = tlc.start or 0 
        te = ( tlc.stop or len(var.axes[tax]) ) - ts
        if not ( ts%12 == 0 and te%12 == 0 ): raise NotImplementedError("The record has to start and end at a full year!")
        assert data.shape[tax] == te
        # assuming the record starts some year in January, and we always need to load full years
    shape = [1,]*data.ndim; shape[tax] = te # dimensions of length 1 will be expanded as needed
    if not linplace: data = data.copy() # make copy if not inplace
    if scalefactor is not None: data *= np.tile(scalefactor, te/12).reshape(shape) # scale in-place
    if offset is not None: data += np.tile(offset, te/12).reshape(shape) # shift in-place
    # return Variable, not just array
    if lvar:
        var.data_array = data
        data = var
    # return data array (default) or Variable instance
    return data

# transform function to convert monthly precip amount into precip rate on-the-fly
def transformMonthly(data=None, var=None, slc=None, time_axis='time', l365=False, lvar=None, linplace=True):
    ''' convert monthly amount to rate in SI units (e.g. mm/month to mm/s) '''
    # check input, makesure this makes sense
    if not isinstance(var,Variable): raise TypeError(var)
    elif var.units[-6:].lower() != '/month':
        raise VariableError("Units check failed: this function converts from month accumulations to rates in SI units.")
    # prepare scalefactor
    spm = 1. / (seconds_per_month_365 if l365 else seconds_per_month) # divide by seconds
    # actually scale by month
    data = monthlyTransform(var=var, data=data, slc=slc, time_axis=time_axis, lvar=lvar, 
                            scalefactor=spm, offset=None, linplace=linplace)
    # return data array (default) or Variable instance
    var.units = var.units[:-6]+'/s' # per second
    return data      
transformPrecip = transformMonthly # for backwards compatibility
      
# transform function to convert days per month into a ratio
def transformDays(data=None, var=None, slc=None, l365=False, lvar=None, linplace=True):
    ''' convert days per month to fraction '''
    # check input, makesure this makes sense
    if not isinstance(var,Variable): raise TypeError(var)
    elif var.units[:4].lower() != 'days':
        raise VariableError("Units check failed: this function converts from days per month to fractions.")
    # prepare scalefactor
    dpm = 1. / (days_per_month_365 if l365 else days_per_month ) # divide by days
    # actually scale by month
    data = monthlyTransform(var=var, data=data, slc=slc, lvar=lvar, scalefactor=dpm, offset=None, linplace=linplace)
        # return data array (default) or Variable instance
    var.units = '' # fraction
    return data      
      
      
## functions to load a dataset

# convenience function to invert variable name mappings
def translateVarNames(varlist, varatts):
  ''' Simple function to replace names in a variable list with their original names as inferred from the 
      attributes dictionary. Note that this requires the dictionary to have the field 'name'. '''
  warn("WARNING: this function is deprecated - the functionality is not handled by DatasetNetCDF directly")
  if isinstance(varlist,basestring): varlist = [varlist]
  if not isinstance(varlist,(list,tuple,set)) or not isinstance(varatts,dict): raise TypeError(varlist)
  varlist = list(varlist) # make copy, since operation is in-place, and to avoid interference
  # cycle over names in variable attributes (i.e. final names, not original names)  
  for key,atts in varatts.iteritems():
    if 'name' in atts and atts['name'] in varlist: varlist.append(key)
#       varlist[varlist.index(atts['name'])] = key # original name is used as key in the attributes dict
  # return varlist with final names replaced by original names
  return varlist


# universal function to generate file names for climatologies and time-series
def getFileName(name=None, resolution=None, period=None, grid=None, shape=None, station=None, 
                filetype='climatology', filepattern=None):
  ''' A function to generate a standardized filename for climatology and time-series files, based on grid type and period.  '''
  if name is None: name = ''
  # grid (this is a *non-native grid*)
  if grid is None or grid == name: gridstr = ''
  else: gridstr = '_{0:s}'.format(grid.lower()) # only use lower case for filenames
  # prepend shape or station type before grid 
  if shape and station: raise ArgumentError
  elif shape: gridstr = '_{0:s}{1:s}'.format(shape,gridstr)
  elif station: gridstr = '_{0:s}{1:s}'.format(station,gridstr)
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
  else: raise NotImplementedError("Unrecognized filetype/mode: '{:s}'".format(filetype))
  # return final name
  return filename.lower() # By convention, climatology files only have lower-case names
  
  
# common climatology load function that will be imported by datasets (for backwards compatibility)
def loadObs(name=None, folder=None, resolution=None, period=None, grid=None, varlist=None, 
             varatts=None, filepattern=None, filelist=None, filemode='r', projection=None, geotransform=None, 
             griddef=None, axes=None, lautoregrid=None):
  ''' A function to load standardized observational climatologies. '''
  return loadObservations(name=name, folder=folder, resolution=resolution, period=period, grid=grid, station=None, 
                          varlist=varlist, varatts=varatts, filepattern=filepattern, filelist=filelist, 
                          projection=projection, geotransform=geotransform, axes=axes, griddef=griddef, 
                          lautoregrid=lautoregrid, mode='climatology', filemode=filemode)

# common climatology load function that will be imported by datasets (for backwards compatibility)
def loadObs_StnTS(name=None, folder=None, resolution=None, varlist=None, station=None, 
                  varatts=None, filepattern=None, filelist=None, filemode='r', axes=None):
    ''' A function to load standardized observational time-series at station locations. '''
    return loadObservations(name=name, folder=folder, resolution=resolution, station=station, 
                          varlist=varlist, varatts=varatts, filepattern=filepattern, filelist=filelist, 
                          projection=None, geotransform=None, griddef=None, axes=axes, period=None, grid=None,
                          lautoregrid=False, mode='time-series', filemode=filemode)
  
# universal load function that will be imported by datasets
def loadObservations(name=None, folder=None, period=None, grid=None, station=None, shape=None, lencl=False, 
                     varlist=None, varatts=None, filepattern=None, filelist=None, filemode='r', resolution=None,
                     projection=None, geotransform=None, griddef=None, axes=None, lautoregrid=None, mode='climatology'):
  ''' A function to load standardized observational datasets. '''
  # prepare input
  if mode.lower() == 'climatology': # post-processed climatology files
    # transform period
    if period is None or period == '': pass
#       if name not in ('PCIC','PRISM','GPCC','NARR'): 
#         raise ValueError("A period is required to load observational climatologies.")
    elif isinstance(period,basestring):
      period = tuple([int(prd) for prd in period.split('-')]) 
    elif not isinstance(period,(int,np.integer)) and ( not isinstance(period,tuple) and len(period) == 2 ): 
      raise TypeError(period)
  elif mode.lower() in ('time-series','timeseries'): # concatenated time-series files
    period = None # to indicate time-series (but for safety, the input must be more explicit)
    if lautoregrid is None: lautoregrid = False # this can take very long!
  # cast/copy varlist
  if isinstance(varlist,basestring): varlist = [varlist] # cast as list
  elif varlist is not None: varlist = list(varlist) # make copy to avoid interference
  # figure out station and shape options
  if station and shape: raise ArgumentError()
  elif station or shape: 
    if grid is not None: raise NotImplementedError('Currently observational station data can only be loaded from the native grid.')
    if lautoregrid: raise GDALError('Station data can not be regridded, since it is not map data.')
    lstation = bool(station); lshape = bool(shape)
    grid = station if lstation else shape
    # add station/shape parameters
    if varlist:
      params = stn_params if lstation else shp_params
      for param in params:
        if param not in varlist: varlist.append(param)    
  else:
    lstation = False; lshape = False
  # varlist (varlist = None means all variables)
  if varatts is None: varatts = default_varatts.copy()
  #if varlist is not None: varlist = translateVarNames(varlist, varatts)
  # N.B.: renaming of variables in the varlist is now handled in theDatasetNetCDF initialization routine
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
        else: raise IOError("The dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.".format(filename,grid) )
      else: raise IOError("The dataset file '{:s}' does not exits!\n('{:s}')".format(filename,filepath))
    filelist = [filename]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                          axes=axes, multifile=False, ncformat='NETCDF4', mode=filemode)
  # mask all shapes that are incomplete in dataset
  if shape and lencl and 'shp_encl' in dataset: 
    dataset.load() # need to load data before masking; is cheap for shape averages, anyway
    dataset.mask(mask='shp_encl', invert=True, skiplist=shp_params)
  # correct ordinal number of shape (should start at 1, not 0)
  if lshape:
    if dataset.hasAxis('shapes'): raise AxisError("Axis 'shapes' should be renamed to 'shape'!")
    if not dataset.hasAxis('shape'): 
      raise AxisError()
    if dataset.shape.coord[0] == 0: dataset.shape.coord += 1
  # figure out grid
  if not lstation and not lshape:
    if grid is None or grid == name:
      dataset = addGDALtoDataset(dataset, projection=projection, geotransform=geotransform, griddef=griddef, gridfolder=grid_folder)
    elif isinstance(grid,basestring): # load from pickle file
  #     griddef = loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder)
      # add GDAL functionality to dataset 
      dataset = addGDALtoDataset(dataset, griddef=grid, gridfolder=grid_folder)
    else: raise TypeError(dataset)
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
    
  def __call__(self, load_list=None, lproduct='outer', inner_list=None, outer_list=None, 
               lensemble=None, ens_name=None, ens_title=None, **kwargs):
    ''' wrap original function: expand argument list, execute load_fct over argument list, 
        and return a list or Ensemble of datasets '''
    # decide, what to do
    if load_list is None and inner_list is None and outer_list is None:
      # normal operation: no expansion      
      datasets =  self.load_fct(**kwargs)
    else:
      # expansion required
      lensemble = ens_name is not None if lensemble is None else lensemble
      # figure out arguments
      kwargs_list = expandArgumentList(expand_list=load_list, lproduct=lproduct, 
                                       inner_list=inner_list, outer_list=outer_list, **kwargs)
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

  
# universal load function that will be imported by datasets
# @BatchLoad
def loadDataset(name=None, station=None, shape=None, mode='climatology', basin_list=None,
                WRF_exps=None, CESM_exps=None, WRF_ens=None, CESM_ens=None, **kwargs):
  ''' A function to load any datasets; identifies source by name heuristics. '''
  # some private imports (prevent import errors)  
  orig_name = name
  # identify dataset source
  lensemble = False; lobs = False
  if mode.upper() == 'CVDP':
    # resolve WRF experiments to parent CESM runs or Reanalysis for CVDP
    if WRF_exps:
      if name in WRF_exps: name = WRF_exps[name].parent
      elif name[:-4] in WRF_exps: name = WRF_exps[name[:-4]].parent
      # N.B.: a WRF ensemble should directly reference a CESM ensemble
    # special case for observational data in the CVDP package (also applies to Reanalysis)
    if name.lower() in ('hadisst','mlost','20thc_reanv2','gpcp'): lobs = True
    elif name.lower()[:3] == 'obs': # load observational data for comparison (also includes reanalysis)
      lobs = True; name = None # select dataset based on variable list (in loadCVDP_Obs)
    elif not name in CESM_exps: 
      raise ArgumentError("No CVDP dataset matching '{:s}' found.".format(name))
    # nothing to do for CESM runs
    dataset_name = 'CESM' # also in CESM module
  elif WRF_ens and ( name in WRF_ens or name[:-4] in WRF_ens ):
    # this is most likely a WRF ensemble
    dataset_name = 'WRF'; lensemble = True
  elif WRF_exps and ( name in WRF_exps or name[:-4] in WRF_exps ):
    # this is most likely a WRF experiment
    dataset_name = 'WRF'
  elif CESM_ens and name in CESM_ens:
    # this is most likely a CESM ensemble
    dataset_name = 'CESM'; lensemble = True
  elif CESM_exps and name in CESM_exps:
    # this is most likely a CESM experiment
    dataset_name = 'CESM'
  else:
    # this is most likely an observational dataset
    dataset_name = name
#     if name[:3].lower() == 'obs': dataset_name = 'EC' if station else 'Unity' # alias... 
#     else: dataset_name = name 
  # import dataset based on name
  try: dataset = import_module('datasets.{0:s}'.format(dataset_name))
  except ImportError: raise ArgumentError("No dataset matching '{:s}' found.".format(dataset_name))
  # identify load function  
  if mode.upper() in ('CVDP',):
    load_fct = 'loadCVDP'
    if lobs: load_fct += '_Obs'
  else:
    load_fct = 'load{:s}'.format(dataset_name)
    if mode.lower() in ('climatology',):
      if lensemble and station: raise ArgumentError(station)
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
    raise ArgumentError("Dataset '{:s}' has no method '{:s}'".format(dataset_name,load_fct))
  if not inspect.isfunction(load_fct): 
    raise ArgumentError("Attribute '{:s}' in module '{:s}' is not a function".format(load_fct.__name__,dataset_name))
    # N.B.: for example, inspect does not work properly on functools.partial objects, and functools.partial does not return a function 
  # generate and check arguments
  kwargs.update(name=name, station=station, shape=shape, mode=mode, basin_list=basin_list,
                WRF_exps=WRF_exps, CESM_exps=CESM_exps, WRF_ens=WRF_ens, CESM_ens=CESM_ens)
  if dataset_name == 'WRF': kwargs.update(exps=WRF_exps, enses=WRF_ens)
  elif dataset_name == 'CESM': kwargs.update(exps=CESM_exps, enses=CESM_ens)
  argspec, varargs, keywords = inspect.getargs(load_fct.func_code); del varargs, keywords
  kwargs = {key:value for key,value in kwargs.iteritems() if key in argspec}
  # load dataset
  dataset = load_fct(**kwargs)
  if orig_name == name: 
    if dataset.name != name: raise DatasetError(load_fct.__name__)
  else: dataset.name = orig_name
  # return dataset
  return dataset

# loadDataset version with BatchLoad capability
loadDatasets = BatchLoad(loadDataset)

# function to extract common points that meet a specific criterion from a list of datasets
def selectElements(datasets, axis, testFct=None, master=None, linplace=False, lall=False):
  ''' Extract common points that meet a specific criterion from a list of datasets. 
      The test function has to accept the following input: index, dataset, axis'''
  if linplace: raise NotImplementedError("Option 'linplace' does not work currently.")
  # check input
  if not isinstance(datasets, (list,tuple,Ensemble)): raise TypeError(datasets)
  if not all(isinstance(dataset,Dataset) for dataset in datasets): raise TypeError(dataset)
  if not callable(testFct) and testFct is not None: raise TypeError(testFct)
  if isinstance(axis, Axis): axis = axis.name
  if not isinstance(axis, basestring): raise TypeError(axis)
  if lall and master is not None: raise ArgumentError("The options 'lall' and 'imaster' are mutually exclusive!")
  # save some ensemble parameters for later  
  lnotest = testFct is None
  lens = isinstance(datasets,Ensemble)
  if lens:
    enskwargs = dict(basetype=datasets.basetype, idkey=datasets.idkey, 
                     name=datasets.name, title=datasets.title) 
  # use dataset with shortest axis as master sample (more efficient)
  axes = [dataset.getAxis(axis) for dataset in datasets]
  if master is None: imaster = np.argmin([len(ax) for ax in axes]) # find shortest axis
  elif isinstance(master,basestring): 
    # translate name of dataset into index
    imaster = None
    for i,dataset in enumerate(datasets): 
      if dataset.name == master: 
        imaster = i; break
    if imaster is None: raise ArgumentError("Master '{:s}' not found in datasets".format(master))
  else: imaster = master
  if not imaster is None and not isinstance(imaster,(int,np.integer)): raise TypeError(imaster)
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
  if len(itpls) == 0: raise DatasetError("Aborting: no data points match all criteria!")
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
def loadEnsemble(names=None, name=None, title=None, varlist=None, aggregation=None, season=None, prov=None, 
                 shape=None, station=None, slices=None, obsslices=None, years=None, period=None, obs_period=None, 
                 reduction=None, constraints=None, filetypes=None, domain=None, grid=None, ldataset=False, 
                 lcheckVar=False, lwrite=False, ltrimT=True, name_tags=None, dataset_mode='time-series', 
                 lminmax=False, master=None, lall=True, ensemble_list=None, ensemble_product='inner', 
                 lensembleAxis=False, WRF_exps=None, CESM_exps=None, WRF_ens=None, CESM_ens=None, 
                 bias_correction=None, obs_list=observational_datasets, basin_list=None, **kwargs):
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
  if ldataset and ensemble_list: raise ArgumentError()
  elif not ldataset: ensemble = Ensemble(name=name, title=title, basetype='Dataset')
  # expand argument list
  if ensemble_list is None: ensemble_list = ['names'] if not ldataset else None
  elif 'aggregation' in ensemble_list: raise ArgumentError("'aggregation' can not be expanded")
  loadargs = expandArgumentList(names=names, station=station, prov=prov, shape=shape, varlist=varlist, 
                                mode=dataset_mode, filetypes=filetypes, domains=domain, grid=grid, lwrite=lwrite, 
                                slices=slices, obsslices=obsslices, period=period, obs_period=obs_period, 
                                years=years, name_tags=name_tags, ltrimT=ltrimT, bias_correction=bias_correction, 
                                lensembleAxis=lensembleAxis, expand_list=ensemble_list, lproduct=ensemble_product,)
  for loadarg in loadargs:
    # clean up arguments
    name = loadarg.pop('names',None); name_tag = loadarg.pop('name_tags',None)
    slcs = loadarg.pop('slices',None); obsslcs = loadarg.pop('obsslices',None)
    slcs = dict() if slcs is None else slcs.copy(); 
    prd = loadarg.pop('period',None); obsprd = loadarg.pop('obs_period',None)
    if name in obs_list: prd = obsprd or prd 
    # special handling of periods for time-series: user for slicing by year!
    mode = loadarg['mode'].lower(); lts = 'time' in mode and 'series' in mode 
    if lts and prd: slcs['years'] = prd          
    if obsslcs and name in obs_list: slcs.update(**obsslcs) # add special slices for obs
    if not lts: 
      if prd: loadarg['period'] = prd
      elif 'years' in slcs: loadarg['period'] = slcs['years']
      if 'years' in slcs: del slcs['years'] # will cause an error with climatologies
    # N.B.: currently VarNC's can only be sliced once, because we can't combine slices yet
    # load individual dataset
    dataset = loadDataset(name=name, WRF_exps=WRF_exps, CESM_exps=CESM_exps, WRF_ens=WRF_ens, 
                          CESM_ens=CESM_ens, basin_list=basin_list, slices=slcs, **loadarg)
    if name_tag is not None: 
      if name_tag.startswith('_'): dataset.name += name_tag
      else: dataset.name = name_tag
    # apply slicing
    if slcs: dataset = dataset(lminmax=lminmax, **slcs) # slice immediately 
    if not ldataset: ensemble += dataset.load() # load data and add to ensemble
  # if input was not a list, just return dataset
  if ldataset: ensemble = dataset.load() # load data
  # select specific stations (if applicable)
  if not ldataset and station and constraints:
    from datasets.EC import selectStations
    ensemble = selectStations(ensemble, stnaxis='station', master=master, linplace=False, lall=lall,
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
      if isinstance(op, basestring): ensemble = getattr(ensemble,op)(axis=ax)
      elif isinstance(op, (int,np.integer,float,np.inexact)): ensemble = ensemble(**{ax:op})
  # extract seasonal/climatological values/extrema
  if (ldataset and len(ensemble)==0): raise EmptyDatasetError(varlist)
  if not ldataset and any([len(ds)==0 for ds in ensemble]): raise EmptyDatasetError(ensemble)
  # N.B.: the operations below should work with Ensembles as well as Datasets 
  if aggregation:
    method = aggregation if aggregation.isupper() else aggregation.title() 
    if season is None:
      ensemble = getattr(ensemble,'clim'+method)(taxis='time', **kwargs)
    else:
      ensemble = getattr(ensemble,'seasonal'+method)(season=season, taxis='time', **kwargs)
  elif season: # but not aggregation
    ensemble = ensemble.seasonalSample(season=season)
  # return dataset
  return ensemble

# convenience versions of loadEnsemble with BatchLoad capability or 'TS' suffix (for backwards-compatibility)
loadEnsembles = BatchLoad(loadEnsemble)
loadEnsembleTS = loadEnsemble # for backwards-compatibility


## Miscellaneous utility functions

# function to return grid definitions for some common grids
def getCommonGrid(grid, res=None, lfilepath=False):
  ''' return definitions of commonly used grids (either from datasets or pickles) '''
  # try pickle first
  griddef = loadPickledGridDef(grid=grid, res=res, folder=grid_folder, 
                               check=False, lfilepath=lfilepath)
  # alternatively look in known datasets
  if griddef is None:
    try:
      dataset = import_module(grid)
      if res is None: griddef = dataset.default_grid
      else: griddef = dataset.grid_def[res]
      if lfilepath: griddef.filepath = None # monkey-patch...
    except ImportError:
      griddef = None
#       assert (elon-slon) % dlon == 0 
#       lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
#       assert (elat-slat) % dlat == 0
#       lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
#       # add new geographic coordinate axes for projected map
#       xlon = Axis(coord=lon, atts=dict(grid='lon', long_name='longitude', units='deg E'))
#       ylat = Axis(coord=lat, atts=dict(grid='lat', long_name='latitude', units='deg N'))
#       gridstr = '{0:s}_{1:s}'.format(grid,res) if res is not None else grid
  # return grid definition object (or None, if none found)
  return griddef


## (ab)use main execution for quick test
if __name__ == '__main__':
    
  from geodata.gdal import GridDefinition, pickleGridDef
  
#   mode = 'pickle_grid'
  mode = 'create_grid'
  grids = dict( 
                CFSR=['031','05'],
                GPCC=['025','05','10','25'],
                NRCan=['NA12','CA12','CA24'],
                CRU=[None],NARR=[None],PRISM=[None],PCIC=[None]
               )
    
  ## pickle grid definition
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
          filename = pickleGridDef(griddef, lfeedback=True, loverwrite=True, lgzip=True)
          
          print('   Saving Pickle to \'{0:s}\''.format(filename))
          print('')
          
          # load pickle to make sure it is right
          del griddef
          griddef = loadPickledGridDef(grid, res=res)
          print(griddef)
        print('')

  ## create a new grid
  elif mode == 'create_grid':
    
    ## parameters for UTM 17 GRW grids
#     name = 'grw1' # 1km resolution
#     geotransform = [500.e3,1.e3,0,4740.e3,0,1.e3]; size = (132,162)
    name = 'grw2' # 5km resolution
    geotransform = [500.e3,5.e3,0,4740.e3,0,5.e3]; size = (27,33)
    projection = "+proj=utm +zone=17 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    ## parameters for UTM 17 Assiniboine River Basin grids
    # Bird River, a subbasin of the Assiniboine River Basin
#     name = 'brd1' # 5km resolution 
#     geotransform = [246749.8, 5.e3, 0., 5524545., 0., 5.e3]; size = ((438573.1-246749.8)/5.e3,(5682634.-5524545.)/5.e3)
#     print size
#     size = tuple(int(i) for i in size)
#     print size
#     geotransform = (245.e3, 5.e3, 0., 5524.e3, 0., 5.e3); size = (39,32)
#     projection = "+proj=utm +zone=14 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
#     ## parameters for Canada-wide Lambert Azimuthal Equal-area
#     name = 'can1' # 5km resolution
#     llx = -3500000; lly = -425000; urx = 3000000; ury = 4000000; dx = dy = 5.e3
#     geotransform = [llx, dx, 0., lly, 0., dy]; size = ((urx-llx)/dx,(ury-lly)/dy)
#     size = tuple(int(i) for i in size)
#     projection = "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs"
    # N.B.: (x_0, dx, 0, y_0, 0, dy); (xl,yl)
    #       GT(0),GT(3) are the coordinates of the bottom left corner
    #       GT(1) & GT(5) are pixel width and height
    #       GT(2) & GT(4) are usually zero for North-up, non-rotated maps
    # create grid
    griddef = GridDefinition(name=name, projection=projection, geotransform=geotransform, size=size, 
                             xlon=None, ylat=None, lwrap360=False, geolocator=True, convention='Proj4')

    # save pickle to standard location
    filepath = pickleGridDef(griddef, folder=grid_folder, filename=None, lfeedback=True)
    assert os.path.exists(filepath)
    print('')
    
    # load pickle to make sure it is right
    del griddef
    griddef = loadPickledGridDef(grid=name, res=None, folder=grid_folder)
    print(griddef)
