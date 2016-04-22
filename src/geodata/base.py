'''
Created on 2013-08-19

Variable and Dataset classes for handling geographical datasets.

@author: Andre R. Erler, GPL v3
'''

# numpy imports
import numpy as np
import numpy.ma as ma # masked arrays
from numpy.lib.stride_tricks import as_strided
import scipy.stats as ss
import numbers
import functools
import gc # garbage collection
from warnings import warn
# my own imports
import utils.nanfunctions as nf
from plotting.properties import getPlotAtts, variablePlotatts # import plot properties from different file
from geodata.misc import checkIndex, isEqual, isInt, isNumber, AttrDict, joinDicts, floateps
from geodata.misc import genStrArray, translateSeasons
from geodata.misc import VariableError, AxisError, DataError, DatasetError, ArgumentError
from processing.multiprocess import apply_along_axis
from utils.misc import histogram, binedges, detrend, percentile, tabulate
     
# used for climatology and seasons
monthlyUnitsList = ('month','months','month of the year')

class UnaryCheckAndCreateVar(object):
  ''' Decorator class for unary arithmetic operations that implements some sanity checks and 
      handles in-place operation or creation of a new Variable instance. '''
  def __init__(self, op):
    ''' Save original operation. '''
    self.op = op
  def __call__(self, orig, asVar=True, linplace=False, **kwargs):
    ''' Perform sanity checks, then execute operation, and return result. '''
    if not orig.data: orig.load()
    # apply operation
    tmp = self.op(orig, linplace=linplace, **kwargs)
    # check for invalid returns (e.g. from applying arithmetic to strings)
    if tmp is None: return None # return immediately (invalid operation)
    else: data, name, units = tmp # regular processing
    # create new Variable or assign other return
    if linplace:
      var = orig
      var.data_array = data
      var.units = units # don't change name, though
    elif asVar: var = orig.copy(name=name, units=units, data=data)
    else: var = data
    if asVar and not isinstance(var,Variable): raise TypeError
    # return function result
    return var
  def __get__(self, instance, klass):
    ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
    # N.B.: similar implementation to 'partial': need to return a callable that behaves like the instance method
    # def f(arg):
    #  return self.__call__(instance, arg)
    # return f    
    return functools.partial(self.__call__, instance) # but using 'partial' is simpler


def BinaryCheckAndCreateVar(sameUnits=True, linplace=False):
  ''' A decorator function that accepts arguments and returns a decorator class with fixed parameter values. '''
  class BinaryCheckAndCreateVar_Class(object):
    ''' A decorator to perform similarity checks before binary operations and create a new variable instance 
      afterwards; name and units are modified; only non-conflicting attributes are kept. '''
    def __init__(self, binOp):
      ''' Save original operation and parameters. '''
      self.binOp = binOp
#       self.sameUnits = sameUnits # passed to constructor function
    # define method wrapper (this is now the actual decorator)
    def __call__(self, orig, other, sameUnits=sameUnits, asVar=True, linplace=linplace, **kwargs):
      ''' Perform sanity checks, then execute operation, and return result. '''
      if isinstance(other,Variable): # raise TypeError, 'Can only add two Variable instances!' 
        if orig.shape != other.shape:
          mind = min(orig.ndim,other.ndim)
          if orig.ndim == other.ndim or orig.shape[:mind] != other.shape[:mind]:
            raise AxisError, 'Variables need to have the same shape and compatible axes!'
          # else we can broadcast
        if sameUnits and orig.units != other.units: 
          raise VariableError, 'Variable units have to be identical for addition!'
        for lax,rax in zip(orig.axes,other.axes):
          if (lax.coord != rax.coord).any(): raise AxisError,  'Variables need to have identical coordinate arrays!'
        if not other.data: other.load()
      elif not isinstance(other, (np.ndarray,numbers.Number,np.integer,np.inexact)): 
        raise TypeError, 'Can only operate with Variables or numerical types!'
        # N.B.: don't check ndarray shapes, because we want to allow broadcasting        
      if not orig.data: orig.load()
      # prepare arguments
      if isinstance(other, Variable):
        otherdata = other.data_array
        othername = other.name      
        otherunits = other.units
        if other.ndim < orig.ndim: 
          otherdata = otherdata.reshape(other.shape+(1,)*(orig.ndim-other.ndim))
        elif other.ndim < orig.ndim: 
          raise NotImplementedError, "Can only broadcast second Variable in binary operation. "
      else:
        otherdata = other
        othername = str(other)
        otherunits = None
      # call original method
      data, name, units = self.binOp(orig, otherdata, othername=othername, otherunits=otherunits, 
                                     linplace=linplace, **kwargs)
      if linplace:
        var = orig # in-place operation should already have changed data_array 
      elif asVar:
        # construct resulting variable (copy from orig)
        if hasattr(other,'atts'): atts = joinDicts(orig.atts, other.atts)
        else: atts = orig.atts.copy()
        atts['name'] = name; atts['units'] = units
        var = orig.copy(data=data, atts=atts)
      else:
        var = data
      if asVar and not isinstance(var,Variable): raise TypeError
      # return new variable instance or data
      return var
    def __get__(self, instance, klass):
      ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
      return functools.partial(self.__call__, instance)
  # return decorator class  
  return BinaryCheckAndCreateVar_Class


class ReduceVar(object): # not a Variable child!!!
  ''' Decorator class that implements some sanity checks for reduction operations. '''
  def __init__(self, reduceop):
    ''' Save original operation. '''
    self.reduceop = reduceop
  def __call__(self, var, asVar=None, axis=None, axes=None, lcheckVar=True, lcheckAxis=True,
                          fillValue=None, lrecursive=False, lall=True, **kwaxes):
    ''' Figure out axes, perform sanity checks, then execute operation, and return result as a Variable 
        instance. Axes are specified either in a list ('axes') or as keyword arguments with corresponding
        slices. '''
    # this really only works for numeric types
    if var.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Reduction does not work with string Variables!"
      else: return None
    # extract axes and keyword arguments      
    slcaxes = dict(); kwargs = dict()
    for key,value in kwaxes.iteritems():
      if var.hasAxis(key): slcaxes[key] = value # use for slicing axes
      else: kwargs[key] = value # pass this to reduction operator
    if len(slcaxes) > 0:
      if axes is None:
        axes = set()
        if axis is not None:
          if axis not in axes: axes.add(axis)
          axis = None
      axes =axes.union(slcaxes.keys())
    # take shortcut?
    if axis is None and axes is None:
      # simple and quick, less overhead
      if not var.data: var.load()
      # remove mask, if fill value is given (some operations don't work with masked arrays)
      if fillValue is not None and var.masked: data = var.data_array.filled(fillValue)
      else: data = var.data_array
      # apply operation without arguments, i.e. over all axes
      data, name, units = self.reduceop(var, data, **kwargs)
      # whether or not to cast as Variable (default: No)
      if asVar is None: asVar = False # default for total reduction
      if asVar: newaxes = tuple()
    else:
      ## figure out reduction axis/axes and slices
      # add axes list to dictionary
      if axis is not None and axes is not None: 
        raise ArgumentError
      elif axis is not None: 
        axes = [axis]
        if not var.hasAxis(axis): 
          if lcheckAxis: raise AxisError, axis
          else: return None # nothing to do
      elif axes is not None: 
        if lcheckAxis:
          for ax in axes: 
            if not var.hasAxis(ax): raise AxisError, ax 
        else:
          if not var.hasAxis(axes, lany=not lall, lall=lall):
            return None # nothing to do
            # if not lrecursive, missing axes will be skipped 
      # N.B.: leave checking of slices to var.slicing (below)
      ## get data from Variable  
      # use overloaded call method to index with coordinate values directly 
      if slcaxes: var = var(asVar=True, **slcaxes)
      # N.B.: call can also accept index values and slices (set options accordingly!)
      # apply reduction operation over axes 
      if lrecursive: # legacy mode
        # remove mask, if fill value is given (some operations don't work with masked arrays)
        #if fillValue is not None and var.masked: data = data.filled(fillValue)
        data = var.getArray(unmask= not fillValue is None, fillValue=fillValue)
        ## compute reduction
        # get proper order of axes
        axlist = [ax.name for ax in var.axes if ax.name in axes]
        axlist.reverse() # start from the back  
        name = var.name; units = var.units # defaults, in case axlist is empty...    
        for axis in axlist:
          # apply reduction operation with axis argument, looping over axes
          data, name, units = self.reduceop(var, data, axidx=var.axisIndex(axis), **kwargs)
      else:
        if len(axes) > 1:
          # merge axes
          sample_axis = 'internal_reduction_axis'
          tmp = var.mergeAxes(axes=axes, new_axis=sample_axis, asVar=True, linplace=False, 
                              lcheckAxis=lcheckAxis, lvarall=lall)
          axes = [sample_axis] # from now on...
          if tmp is None: raise AssertionError, var
          else: var = tmp
          data = var.getArray(unmask= not fillValue is None, fillValue=fillValue)
          # apply reduction operation over merged axes
          data, name, units = self.reduceop(var, data, axidx=var.axisIndex(sample_axis), **kwargs)
        else:
          # apply reduction operation over axis
          data = var.getArray(unmask= not fillValue is None, fillValue=fillValue)
          data, name, units = self.reduceop(var, data, axidx=var.axisIndex(tuple(axes)[0]), **kwargs)
      # squeeze removed dimension (but no other!)
      newshape = [len(ax) for ax in var.axes if not ax.name in axes]
      #print data.shape, newshape
      data = data.reshape(newshape)
      # N.B.: other singleton dimensions will have been removed, too
    ## cast into Variable
    # whether or not to cast as Variable (default: Yes)
    if asVar is None: asVar = True # default for iterative reduction
    if asVar: newaxes = [ax for ax in var.axes if not ax.name in axes]
    if asVar: 
      redvar = var.copy(name=name, units=units, axes=newaxes, data=data)
#       redvar = Variable(name=var.name, units=var.units, axes=newaxes, data=data, 
#                      fillValue=var.fillValue, atts=var.atts.copy(), plot=var.plot.copy())
    else: redvar = data
    return redvar # return function result
  
  def __get__(self, instance, klass):
    ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
    return functools.partial(self.__call__, instance) # but using 'partial' is simpler


class ApplyTestOverList(object):
  ''' Decorator for class methods that accepts a list as main argument, applys the class method to
      all elements of the list, and returns an aggregated boolean result; if the first argument is 
      not (tuple,list,np.ndarray), the method is executed normally.
      N.B.: this only works with methods that take a single non-keyword argument.'''
  def __init__(self, op):
    ''' Save original operation. '''
    self.op = op
  def __call__(self, orig, args, lany=None, lall=None, **kwargs):
    ''' perform test over list and aggregate results. '''
    if isinstance(args, (tuple,list,np.ndarray)):
      if lany is None and lall is None: lall,lany = True,False # default
      # apply operation over list
      results = [self.op(orig, arg, **kwargs) for arg in args]
      # aggregate results
      if lall and not lany: result = np.all(results)
      elif not lall and lany: result = np.any(results)
      else: raise ArgumentError
    else:
      # just apply method to argument
      result = self.op(orig, args, **kwargs)
    # return test result
    return result
  def __get__(self, instance, klass):
    ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
    # N.B.: similar implementation to 'partial': need to return a callable that behaves like the instance method
    # def f(arg):
    #  return self.__call__(instance, arg)
    # return f    
    return functools.partial(self.__call__, instance) # but using 'partial' is simpler


## Variable class and derivatives 

class Variable(object):
  ''' 
    The basic variable class; it mainly implements arithmetic operations and indexing/slicing.
  '''
  
  def __init__(self, name=None, units=None, axes=None, data=None, dtype=None, mask=None, fillValue=None, 
               atts=None, plot=None):
    ''' 
      Initialize variable and attributes.
      
      Basic Attributes:
        name = @property # short name, e.g. used in datasets (links to atts dictionary)
        units = @property # physical units (links to atts dictionary)
        data = @property # logical indicating whether a data array is present/loaded 
        axes = None # a tuple of references to coordinate variables (also Variable instances)
        data_array = None # actual data array (None if not loaded)
        shape = @property # length of dimensions, like an array
        ndim = @property # number of dimensions
        dtype = @property # data type (string)
        
      Optional/Advanced Attributes:
        masked = @property # whether or not the array in self.data is a masked array
        fillValue = @property # value to fill in for masked values
        dataset = None # parent dataset the variable belongs to
        atts = None # dictionary with additional attributes
        plot = None # attributed used for displaying the data       
    '''
    # basic input check
    if data is None:
      ldata = False; shape = None
    else:
      if isinstance(data,(list,tuple)) and isinstance(data[0],basestring):
        data = genStrArray(data) # more checks inside function
      if not isinstance(data,np.ndarray): data = np.asarray(data) # 'The data argument must be a numpy array!'
      ldata = True; shape = data.shape; 
      if dtype:
        dtype = np.dtype(dtype) # make sure it is properly formatted.. 
        if dtype is not data.dtype: data = data.astype(dtype) # recast as new type        
#         raise TypeError, "Declared data type '{:s}' does not match the data type of the array ({:s}).".format(str(dtype),str(data.dtype))
      else: dtype = data.dtype
      if np.issubdtype(dtype, np.inexact) and not isinstance(data, ma.masked_array):
        data = ma.masked_invalid(data, copy=False) 
      if axes is not None and len(axes) != data.ndim: 
        raise AxisError, 'Dimensions of data array and axes are not compatible!'
    # for completeness of MRO...
    super(Variable,self).__init__()
    ## set basic variable 
    # cast attributes dicts as AttrDict to facilitate easy access 
    if atts is None: atts = dict()
    # set name in atts
    if name is not None: atts['name'] = name # name can also be accessed directly as a property
    elif 'name' not in atts: atts['name'] = 'N/A'
    # set units in atts
    if units is not None: atts['units'] = units # units can also be accessed directly as a property
    elif 'units' not in atts: atts['units'] = 'N/A'
    # sync fillValue with atts
    if 'fillValue' in atts:
      if fillValue is None: fillValue = atts['fillValue']
      else: atts['fillValue'] = fillValue
#     if fillValue is not None: atts['missing_value'] = fillValue # slightly irregular treatment...
    self.__dict__['atts'] = AttrDict(**atts)
    # try to find sensible default values
    self.__dict__['plot'] = getPlotAtts(name=name, units=units, atts=atts, plot=plot)
    # set defaults - make all of them instance variables! (atts and plot are set below)
    self.__dict__['data_array'] = None
    self.__dict__['_dtype'] = dtype
    self.__dict__['_dataset'] = None # set by addVariable() method of Dataset  
    ## figure out axes
    if axes is not None:
      assert isinstance(axes, (list, tuple))
      if all([isinstance(ax,Axis) for ax in axes]):
        if ldata: 
          for ax,n in zip(axes,shape): ax.len = n
        if not ldata and all([len(ax) for ax in axes]): # length not zero
          self.__dict__['shape'] = tuple([len(ax) for ax in axes]) # get shape from axes
      elif all([isinstance(ax,basestring) for ax in axes]):
        if ldata: axes = [Axis(name=ax, length=n) for ax,n in zip(axes,shape)] # use shape from data
        else: axes = [Axis(name=ax) for ax in axes] # initialize without shape
    else: 
      raise VariableError, 'Cannot initialize {:s} instance \'{:s}\': no axes declared'.format(self.var.__class__.__name__,self.name)
    self.__dict__['axes'] = tuple(axes) 
    # create shortcuts to axes (using names as member attributes) 
    for ax in axes: self.__dict__[ax.name] = ax
    # assign data, if present (can initialize without data)
    if data is not None: 
      self.load(data, mask=mask, fillValue=fillValue) # member method defined below
      assert self.data == ldata # should be loaded now
      
  @property
  def name(self):
    ''' The Variable name (stored in the atts dictionary). '''
    return self.atts['name']  
  @name.setter
  def name(self, name):
    self.atts['name'] = name
    
  @property
  def units(self):
    ''' The Variable units (stored in the atts dictionary). '''
    return self.atts['units']  
  @units.setter
  def units(self, units):
    self.atts['units'] = units

  @property
  def dataset(self):
    ''' The Dataset the Variable belongs to (or None). '''
    return self._dataset  
  @dataset.setter
  def dataset(self, dataset):
    self._dataset = dataset
    self.atts['dataset_name'] = dataset.name

  @property
  def dataset_name(self):
    ''' The name of the Dataset the Variable belongs to (or None). '''
    if self.dataset is not None: 
      dataset_name = self.dataset.name
      if dataset_name != self.atts['dataset_name']:
        warn("Detected name change of Dataset from {:s} to {:s} - correcting automatically.".format(self.atts['dataset_name'],dataset_name))
        self.atts['dataset_name'] = dataset_name
#         raise DatasetError, "{:s} != {:s}".format(self.dataset.name, self.atts['dataset_name'])
    elif 'dataset_name' in self.atts: 
      dataset_name = self.atts['dataset_name']
    else: dataset_name = None
    return dataset_name
  @dataset_name.setter
  def dataset_name(self, dataset_name):
    if self.dataset is None: self.atts['dataset_name'] = dataset_name
    else: raise ValueError, self.dataset
    
  @property
  def data(self):
    ''' A flag indicating if data is loaded. '''
    return False if self.data_array is None else True   
  
  @property
  def dtype(self):
    ''' The data type of the Variable (inferred from data). '''
    dtype = self._dtype
    if self.data and dtype != self.data_array.dtype:
      DataError, "Dtype mismatch!"
    return dtype   
  @dtype.setter
  def dtype(self, dtype):
    if self.data:
      self.data_array = self.data_array.astype(dtype)
    self._dtype = dtype

  @property
  def strvar(self):
    ''' If the data type is a String kind '''
    return self.dtype.kind == 'S'   

  @property
  def strlen(self):
    ''' The length/itemsize of a String variable  '''
    return self.dtype.itemsize if self.strvar else None
  
  @property
  def ndim(self):
    ''' The number of dimensions (inferred from axes). '''
    ndim = len(self.axes)
    if self.data and ndim != self.data_array.ndim: raise DataError, 'Dimension mismatch!' 
    return ndim   
  
  @property
  def shape(self):
    ''' The length of each dimension (shape of data; inferred from axes). '''
    shape = tuple(len(ax) for ax in self.axes)
    if self.data and shape != self.data_array.shape: 
      raise DataError, 'Shape mismatch!'
    return shape
  
  @property
  def masked(self):
    ''' A flag indicating if the data is masked. '''
    if self.data: masked = isinstance(self.data_array,ma.MaskedArray)
    else: masked = self.atts.get('fillValue',None) is not None
    return masked
  
  @property
  def fillValue(self):
    ''' The fillValue for masks (stored in the atts dictionary). '''
    fillValue = self.atts.get('fillValue',None)
    if self.data and self.masked:
      if  fillValue is None:
        raise DataError, 'Invalid FillValue: None!'      
      elif fillValue != self.data_array._fill_value:
        if  not np.isnan(fillValue) or not np.isnan(self.data_array._fill_value):
          raise DataError, 'FillValue mismatch!' # N.B.: NaN's are never equal           
    return fillValue
  @fillValue.setter
  def fillValue(self, fillValue):
    self.atts['fillValue'] = fillValue
    if self.data and self.masked:
      self.data_array._fill_value = fillValue
      # I'm not sure which one does work, but this seems to work more reliably!
#       self.data_array.set_fill_value = fillValue
#       ma.set_fill_value(self.data_array,fillValue)
    
  
  def __str__(self):
    ''' Built-in method; we just overwrite to call 'prettyPrint()'. '''
    return self.prettyPrint(short=False) # print is a reserved word  

  def prettyPrint(self, short=False):
    ''' Print a string representation of the Variable. '''
    if short: 
      name = "{0:s} [{1:s}]".format(self.name,self.units) # name and units
      shape = '(' # shape
      for l in self.shape: shape += '{0:d},'.format(l)
      shape += ')'
      if not self in self.axes: # this is to avoid code duplication in Axis class
        axes = 'Axes: '
        for ax in self.axes: axes += '{0:s},'.format(ax.name)
      else: axes = self.__class__.__name__ # Class name (usually an Axis)
      substr = name + ' '*(max(1,35-len(name)-len(shape))) + shape # the field is 35 wide, with at least 1 space
      string = '{:<35s}  {:s}'.format(substr,axes)
    else:
      string = '{0:s} \'{1:s}\' [{2:s}]   {3:s}\n'.format(self.__class__.__name__,self.name,self.units,self.__class__)
      for ax in self.axes: string += '  {0:s}\n'.format(ax.prettyPrint(short=True))
      string += 'Attributes: {0:s}\n'.format(str(self.atts))
      string += 'Plot Attributes: {0:s}'.format(str(self.plot))
    return string
  
  def tabulate(self, row=None, column=None, header=None, labels=None, cell_str='{}', cell_axis=None,
               cell_fct=None, lflatten=False, mode='latex', filename=None, folder=None, **kwargs):
    ''' Create a nicely formatted table in the selected format ('mylatex' or call tabulate); 
        cell_str controls formatting of each cell, and also supports multiple arguments along 
        an axis. lflatten skips cell axis checking and lumps all remaining axes together. '''
    # check input
    if cell_axis:
      if not self.ndim == 3: raise AxisError, self.axes
    elif lflatten:
      if not self.ndim >= 2: raise AxisError, self.axes
    elif not self.ndim == 2: raise AxisError, self.axes
    if not isinstance(cell_str,basestring): raise TypeError, cell_str
    if not self.hasAxis(row): raise AxisError, row
    if not self.hasAxis(column): raise AxisError, column
    if cell_axis and not self.hasAxis(cell_axis): raise AxisError, cell_axis
    if self.data: data = self.getArray(copy=False)
    else: raise DataError
    irow = self.axisIndex(row); icol = self.axisIndex(column)
    cell_idx = None if cell_axis is None else self.axisIndex(cell_axis)
    # call general tabulate method
    string = tabulate(data=data, col_idx=icol, row_idx=irow, header=header, labels=labels, cell_str=cell_str, 
                      cell_idx=cell_idx, cell_fct=cell_fct, lflatten=lflatten, mode=mode, 
                      filename=filename, folder=folder)
    # return string for printing
    return string
  
  def squeeze(self):
    ''' A method to remove singleton dimensions (in-place). '''
    # new axes tuple: only the ones longer than one element
    axes = []; retour = [] 
    for ax in self.axes:
      if len(ax) > 1: axes.append(ax)
      else: retour.append(ax)
    self.axes = tuple(axes)
    # squeeze data array, if necessary
    if self.data:
      self.data_array = self.data_array.squeeze()
      assert self.ndim == self.data_array.ndim
      assert self.shape == self.data_array.shape        
    # return squeezed dimensions
    assert self.ndim == len(axes)    
    assert self.shape == tuple([len(ax) for ax in self.axes])
    return self
  
  def copy(self, deepcopy=False, **newargs): # this methods will have to be overloaded, if class-specific behavior is desired
    ''' A method to copy the Variable with just a link to the data. '''
    if deepcopy:
      var = self.deepcopy( **newargs)
    else:
      # N.B.: don't pass name and units as they just link to atts anyway, and if passed directly, they overwrite user atts
      args = dict(axes=self.axes, data=self.data_array, dtype=self.dtype,
                  mask=None, atts=self.atts.copy(), plot=self.plot.copy())
      if 'data' in newargs and newargs['data'] is not None: 
        newargs['dtype'] = newargs['data'].dtype
      args.update(newargs) # apply custom arguments (also arguments related to subclasses)      
      var = Variable(**args) # create a new basic Variable instance
    # N.B.: this function will be called, in a way, recursively, and collect all necessary arguments along the way
    return var
  
  def deepcopy(self, **newargs): # in almost all cases, this methods will be inherited from here
    ''' A method to generate an entirely independent variable instance (copy meta data, data array, and axes). '''
    # copy axes (generating ordinary Axis instances with coordinate arrays)
    if 'axes' not in newargs: newargs['axes'] = tuple([ax.deepcopy() for ax in self.axes]) # allow override though
    # replace link with new copy of data array
    if self.data: data = self.data_array.copy()
    else: data = None
    # copy meta data
    var = self.copy(data=data, **newargs) # use instance copy() - this method can be overloaded!   
    # N.B.: using load() and getArray() should automatically take care of any special needs 
    return var

  @ApplyTestOverList
  def hasAxis(self, axis, strict=True):
    ''' Check if the variable instance has a particular axis. '''
    if not strict and isinstance(axis,Variable): axis = axis.name
    if isinstance(axis,basestring): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return True
    elif isinstance(axis,Variable): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i] == axis: return True    
    elif isinstance(axis,Variable): # by object ID
      for i in xrange(len(self.axes)):
        if self.axes[i] == axis: return True
    # if all fails
    return False
  
  def replaceAxis(self, oldaxis, newaxis=None):
    ''' Replace an existing axis with a different one with similar general properties. '''
    if newaxis is None: 
      newaxis = oldaxis; oldaxis = oldaxis.name # i.e. replace old axis with the same name'
    # check axis
    if isinstance(oldaxis,Axis): oldname = oldaxis.name # just go by name
    else: oldname = oldaxis
    oldaxis = self.axes[self.axisIndex(oldname)]
    if not self.hasAxis(oldaxis): raise AxisError
    if len(oldaxis) != len(newaxis): 
      raise AxisError # length has to be the same!
    #if oldaxis.data != newaxis.data: raise DataError # make sure data status is the same
    # replace old axis
    self.axes = tuple([newaxis if ax.name == oldname else ax for ax in self.axes])
    self.__dict__[oldname] = newaxis # update this reference as well
    assert len(self.axes) == self.ndim
    assert tuple([len(ax) for ax in self.axes]) == self.shape
    # return confirmation, i.e. True, if replacement was successful
    return self.hasAxis(newaxis)

#   def __contains__(self, axis):
#     ''' Check if the variable instance has a particular axis. '''
#     # same as class method
#     return self.hasAxis(axis)
#   
#   def __len__(self):
#     ''' Return number of dimensions. '''
#     return self.__dict__['ndim']

  @ApplyTestOverList
  def getAxis(self, axis, lcheck=True):
    ''' Return a reference to the Axis object or one with the same name. '''
    lhas = self.hasAxis(axis)        
    if not lhas and lcheck:
      if isinstance(axis,Axis): axis = axis.name  
      raise AxisError, "Variable '{:s}' has no axis named '{:s}'".format(self.name,axis) 
    if lhas: ax = self.axes[self.axisIndex(axis)]
    else: ax = None
    return ax
  
  @ApplyTestOverList
  def axisIndex(self, axis, strict=True, lcheck=True):
    ''' Return the index of a particular axis. (return None if not found) '''
    if isinstance(axis,basestring): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return i
    elif isinstance(axis,Axis): # by object ID
      for i in xrange(len(self.axes)):
        if strict and self.axes[i] == axis: return i
        elif not strict and self.axes[i].name == axis.name: return i
    # if all fails
    elif lcheck: raise AxisError, "Axis '{:s}' not found!".format(str(axis))
    return None
        
  def __getitem__(self, slc):
    ''' Method implementing direct access to the data (returns array, not Variable). '''
    # check if data is loaded      
    if not self.data:
      raise DataError, "Variable instance '{:s}' has no associated data array or it is not loaded!".format(self.name)
    # if the data is scalar, just return it
    if len(self.shape) == 0 and slc == slice(None): 
      data = self.data_array 
    # determine what to do with numpy arrays
    elif all(checkIndex(slc, floatOK=True)):       
      # array indexing: return array slice
      data = self.data_array.__getitem__(slc) # valid array slicing
    else:    
      # if nothing applies, raise index error
      #print slc
      raise IndexError, "Invalid index type for class '{:s}'!".format(self.__class__.__name__)
    # return data, if no error
    return data
  
  def __setitem__(self, slc, data):
    ''' Method implementing write access to data array'''
    if self.data:
      # pass on to array 
      self.data_array.__setitem__(slc, data)
      # N.B.: slice doesn't have to match data, since we can just assign a subset 
    else: 
      # here we are assigning an entire, new array, so do some type and shape checking
      if self.dtype is not None and not np.issubdtype(data.dtype, self.dtype):
        raise DataError, "Dtypes of Variable and array are inconsistent."
      else: self.dtype = data.dtype
      if isinstance(slc,slice): slc = (slc,)*self.ndim
      def slen(a,o,e): 
        return (o-a)/e
      shape = tuple([slen(*s.indices(len(ax))) for s,ax in zip(slc,self.axes)])
      if shape != self.shape: 
        raise NotImplementedError, "Implicit slicing during date assignment is currently not supported."
      elif self.shape != data.shape: 
        raise DataError, "Data array shape does not match variable shape\n(slice was ignored, since no data array was present before)."
      else: self.data_array = data         
    
  def slicing(self, lidx=None, lrng=None, years=None, listAxis=None, asVar=None, lfirst=False, lminmax=False,
              lsqueeze=True, lcheck=False, lcopy=False, lslices=False, linplace=False, **axes):
    ''' This method implements access to slices via coordinate values and returns Variable objects. 
        Default behavior for different argument types: 
          - index by coordinate value, not array index, except if argument is a Slice object
          - interprete tuples of length 2 or 3 as ranges
          - treat lists and arrays as coordinate lists (can specify new list axis)
          - for backwards compatibility, None values are accepted and indicate the entire range 
        Type-based defaults are ignored if appropriate keyword arguments are specified. '''
    ## check axes and input and determine modes
    # parse axes for pseudo-axes, using info from parent dataset
    lpseudo = False; lnonpseudo = False
    if self.dataset is not None:
      dataset = self.dataset 
      axes = axes.copy() # might be changed...
      for key,val in axes.iteritems():
        if val is not None and dataset.hasVariable(key):
          del axes[key] # remove pseudo axis
          var = dataset.getVariable(key)
          if isinstance(val,(tuple,list,np.ndarray)): 
            raise NotImplementedError, "Currently only single coordiante values/indices are supported for pseudo-axes."        
          if var.ndim == 1: # possibly valid pseudo-axis!
            lpseudo = True
            coord = var.findValues(val, lidx=lidx, lfirst=lfirst, lminmax=lminmax, lflatten=False)          
          else: raise AxisError, "Pseudo-axis can only have one axis!"
          axes[var.axes[0].name] = coord # create new entry with actual axis
          # N.B.: not that this automatically squeezes the pseudo-axis, since it is just a values...
        elif val is not None and dataset.hasAxis(key): lnonpseudo = True
      # pseudo-axes work with ranges, but return indices, and we can't mix ranges with indices!
      if lrng and lpseudo: 
        lrng = False
        if lnonpseudo: 
          raise ArgumentError, "'lrng' was reset to False, but other variables are not yet processed."          
      
    # resolve special key words
    if years is not None:
      if self.hasAxis('time'):
        time = self.getAxis('time')
        # make sure the time axis is well-formatted, because we are making a lot of assumptions!
        if not time.units.lower() in ('month','months'): 
          raise NotImplementedError, "Time units='month' required for keyword 'years'!"
        if 'long_name' not in time.atts: raise KeyError, self.prettyPrint(short=True)
        if time.coord[0]%12 != 0: 
          raise AxisError, "Time-axis has to start at a full year for keyword 'years'!"
        if lidx: raise ArgumentError, "Keyword 'years' only works with coordinate indexing, not direct indexing!"
        # convert years to time-axis coordinates
        if '1979' in time.atts.long_name: offset = 1979
        else: offset = 0
        if isinstance(years,np.number): months = (years - offset)*12
        elif isinstance(years,(list,tuple)): months = [(yr - offset)*12 for yr in years]
        if isinstance(years,tuple): 
          if len(months) == 2: months = (months[0],months[1]-1)
          elif len(months) == 3: months = (months[0],months[1]-1,months[2]) # passed to np.linspace
          else: raise NotImplementedError
        axes['time'] = months
      if self.hasAxis('year'):
        year = self.getAxis('year')
        # make sure the time axis is well-formatted, because we are making a lot of assumptions!
        if not year.units.lower() in ('year','years'): 
          raise NotImplementedError, "Year units='year' required for keyword 'years'!"
        if 'long_name' not in year.atts: raise KeyError, self.prettyPrint(short=True)
        if lidx: raise ArgumentError, "Keyword 'years' only works with coordinate indexing, not direct indexing!"
        # convert years to time-axis coordinates
        if '1979' in year.atts.long_name: offset = 1979
        else: offset = 0
        if isinstance(years,np.number): new_years = years - offset
        elif isinstance(years,(list,tuple)): new_years = [(yr - offset) for yr in years]
        if isinstance(years,tuple): 
          if len(new_years) == 2: new_years = (new_years[0],new_years[1]-1)
          elif len(new_years) == 3: new_years = (new_years[0],new_years[1]-1,new_years[2]) # passed to np.linspace
          else: raise NotImplementedError
        axes['year'] = new_years
      elif lcheck: 
        raise AxisError, "Axis 'time' required for keyword 'years'!"
      # N.B.: if lcheck is False, these keywords will just be ignored without a time-axis
    varaxes = dict(); idxmodes = dict() ; rngmodes = dict(); lstmodes = dict()
    # parse axes arguments and determine slicing
    for key,val in axes.iteritems():
      if val is not None and self.hasAxis(key): # extract axes that are relevant        
        if not isinstance(val,(tuple,list,np.ndarray,slice,int,float,np.integer,np.inexact)): 
          raise TypeError, "Can only use numbers, tuples, lists, arrays, and slice objects for indexing!"        
        if isinstance(val,np.ndarray) and val.ndim > 1: raise TypeError, "Can only use 1-D arrays for indexing!"
        varaxes[key] = val
        idxmodes[key] = isinstance(val,slice) if lidx is None else lidx  # only slices, except if set manually
        if isinstance(val,slice) and not idxmodes[key]: raise TypeError, "Slice can not be used for indexing by coordinate value."
        if lrng is None:
          rngmodes[key] = isinstance(val,tuple) and 2<=len(val)<=3 # all others are not ranges  
        else:
          if isinstance(val,slice) and lrng: raise ArgumentError, "A Slice ibject does not require range expansion."
          if len(val) < 2: raise ArgumentError, "Need at least two values to define a range." 
          if len(val) > 3: raise ArgumentError, "Can not expand more than three values to range."
          if len(val) == 3 and lrng and not idxmodes[key]: 
            raise NotImplementedError, "Coordinate ranges with custom spacing are not implemented yet."  
          rngmodes[key] = lrng
        lstmodes[key] = not rngmodes[key] and isinstance(val,(tuple,list,np.ndarray))
      elif lcheck: # default is to ignore axes that the variable doesn't have
        raise AxisError, "Variable '{:s}' has no Axis '{:s}'.".format(self.name,key)
    ## create Slice tuple to slice data array and axes
    slcs = []; lists = [] # lists need special treatment
    # loop over axes of variable
    for ax in self.axes:
      if ax.name in varaxes:
        axval = varaxes[ax.name]; idxmod = idxmodes[ax.name]; rngmod = rngmodes[ax.name]
        # figure out indexing method
        if not idxmod:
          # values and ranges are in coordinate values
          # translate coordinate values into indices
          if isinstance(axval,(tuple,list,np.ndarray)): 
            idxslc = [ax.getIndex(idx, outOfBounds=True) for idx in axval]
            # expand ranges
            if rngmod:
              # coordinate values are inclusive, unlike indices
              if idxslc[1] is not None: idxslc[1] += 1 
              # out-of-bounds values (None) should be handled correctly by slice              
              slcs.append(slice(*idxslc)) # use slice with index bounds 
            else:
              #idxslc = [idx for idx in idxslc if idx is not None] # remove out-of-bounds values              
              slcs.append(idxslc) # use list of converted indices
          else:
            idxslc = ax.getIndex(axval, outOfBounds=True)
            if idxslc is None: raise AxisError, "Coordinate value {:s} out of bounds for axis '{:s}'.".format(str(axval),ax.name)  
            slcs.append(idxslc)
        else:
          # values and ranges are indices
          if rngmod: # need to expand ranges
            if axval[1] is not None: 
              axval = list(axval)
              if axval[1] == -1: axval[1] = None # special for last index
              else: axval[1] += 1 # make behavior consistent with coordinates 
            slcs.append(slice(*axval)) # this also works with arrays!
          else: # use as is
            slcs.append(axval)          
        # list mode identifier
        if lstmodes[ax.name]: lists.append(slcs[-1]) # add most recent element if this is a list    
      else:
        # if not present, no slicing...
        slcs.append(slice(None))
    assert len(slcs) == len(self.axes)
    ## check for lists and ensure consistency
    assert lists == [slc for slc in slcs if isinstance(slc,(tuple,list,np.ndarray))] 
    if len(lists) > 0:   
      lstlen = len(lists[0]) # if list indexing is used, we need to check the length
      if not all([len(lst)==lstlen for lst in lists]): raise ArgumentError
      # check if elements are out-of-bounds and remove those
      for i in xrange(lstlen-1,-1,-1):
        if any([lst[i] is None for lst in lists]):
          for lst in lists: 
            if isinstance(slc,list): del lst[i]
            else: raise IndexError, "Element of immutable index sequence out of bounds."
            # N.B.: one might recast as a list, but that's maybe not necessary    
      if not all([len(lst)==lstlen for lst in lists]): raise ArgumentError
      # return checked lists to slices
      lstidx = -1 # index of first list (where list axis will be inserted)
      lstcnt = 0 # count number of lists
      for i in xrange(len(slcs)):
        if isinstance(slcs[i],(tuple,list,np.ndarray)):
          slcs[i] = lists.pop(0) # return checked lists
          if lstidx == -1: lstidx = i 
          lstcnt += 1
      # create generic list axis or use listAxis
      if listAxis is None:
        # if only one list, just use old axis and slice (done below) 
        if lstcnt > 1: listAxis = Axis(name='list', units='n/a', length=lstlen)
      elif not isinstance(listAxis,Axis): raise TypeError
    else:
      assert any(lstmodes.values()) == False   
    ## create new Variable object
    # slice data using the variables __getitem__ method (which can also be overloaded)
    if self.data:
      if self.ndim > 0:
        data = self.data_array.__getitem__(slcs) # just pass list of slices
      else: data = self.data_array.__getitem__(None)
      if lcopy and isinstance(data,np.ndarray): data = data.copy() # copy array
      if lsqueeze: data = np.squeeze(data) # squeeze
    else: data = None
    # create a Variable object by default, unless data is scalar
    if asVar or linplace or ( asVar is None and 
                               ( data is None or isinstance(data,np.ndarray) ) ):
      # create axes for new variable
      newaxes = []
      for i,ax,idxslc in zip(xrange(self.ndim),self.axes,slcs):
        # use slice object from before to get coordinate data
        coord = ax.coord.__getitem__(idxslc)
        if isinstance(coord,np.ndarray) and (len(coord) > 1 or not lsqueeze):
          # N.B.: when indexing with scalars, it gets squeezed anyway
          if listAxis is not None and ax.name in lstmodes and lstmodes[ax.name]:
            if i == lstidx:
              # add list axis, but only the first time!
              newaxes.append(listAxis) # this is always a new axis
          else:
            # make new axis object from old, using axis' copy method
            if linplace:
              ax.coord = coord 
              newaxes.append(ax) 
            else: 
              newaxes.append(ax.copy(coord=coord.copy()))            
        # if this axis will be squeezed, we can just omit it
      # create new variable object from old, using variables copy method
      if linplace:
        self.axes = newaxes # need to sneak in new axes, or shape mismatch will cause Error
        if data is not None: self.load(data=data)
        newvar = self
      else: 
        newvar = self.copy(data=data, axes=newaxes)
    else:
      # N.B.: this is the default for scalar results  
      newvar = data # just return data, like __getitem__
    # return results and slices, if requested
    if lslices: return newvar, slcs
    else: return newvar
  # a convenient alias to slicing
  def __call__(self, lslices=False, **kwargs):
    ''' Defers the call to the 'slicing' method. '''
    if lslices: 
      newvar, slcs = self.slicing(lslices=True, **kwargs)
      return newvar, slcs
    else: return self.slicing(lslices=False, **kwargs)
  
  def load(self, data=None, mask=None, fillValue=None, lrecast=False, **axes):
    ''' Method to attach numpy data array to variable instance (also used in constructor). '''
    # optional slicing
    if any([self.hasAxis(ax) for ax in axes.iterkeys()]):
      self, slcs = self.slicing(asVar=True, lslices=True, linplace=True, **axes) # this is poorly tested...
      if data is not None and data.shape != self.shape: 
        data = data.__getitem__(slcs) # slice input data, if appropriate     
    # now load data       
    if data is None:
      if not self.data:
        raise DataError, 'No data loaded and no external data supplied!'        
    else:   
      # check types   
      if not isinstance(data,np.ndarray): raise TypeError, 'The data argument must be a numpy array!'
      if self.dtype is None: self.dtype = data.dtype
      elif data.dtype == self.dtype: pass
      elif np.issubdtype(data.dtype, self.dtype): data = data.astype(self.dtype) 
      else: 
        if lrecast: data = data.astype(self.dtype)
        else: raise DataError, "Dtypes of Variable and array are inconsistent."
      if np.issubdtype(data.dtype, np.inexact) and not isinstance(data, ma.masked_array):
        data = ma.masked_invalid(data, copy=False)     
        data._fill_value = fillValue if fillValue is not None else self.fillValue
#         ma.set_fill_value(data, fillValue if fillValue is not None else self.fillValue) # this seems to work more reliably!
      # handle/apply mask
      if mask is not None: data = ma.array(data, mask=mask) 
      if isinstance(data,ma.MaskedArray): # figure out fill value for masked array
        if fillValue is not None: # override variable preset 
          if isinstance(fillValue,np.generic): fillValue = fillValue.astype(self.dtype)
          self.atts['fillValue'] = fillValue
          data._fill_value =  fillValue
          # I'm not sure which one does work, but this seems to work more reliably!
        elif 'fillValue' in self.atts and self.atts['fillValue'] is not None: # use variable preset
          data._fill_value = self.atts['fillValue'] 
          # I'm not sure which one does work, but this seems to work more reliably!
#           ma.set_fill_value(data,self.atts['fillValue'])
#           data.set_fill_value(self.atts['fillValue']) 
        else: # use data default
          self.atts['fillValue'] = data._fill_value if data._fill_value is not None else data.fill_value
        if self.atts['fillValue'] is None or not ( self.atts['fillValue'] == data._fill_value  
              or (np.isnan(self.fillValue) and np.isnan(data._fill_value)) ):
          raise AssertionError, "{:s}, {:s}, {:s}".format(self.atts['fillValue'], data._fill_value, fillValue)
      # assign data to instance attribute array 
      self.__dict__['data_array'] = data
      # check shape consistency
      if len(self.shape) != self.ndim and (self.ndim != 0 or data.size != 1):
        raise DataError, 'Variable dimensions and data dimensions incompatible!'
      # N.B.: the second statement is necessary, so that scalars don't cause a crash
      # some more checks
      # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple; hence
      #       the coordinate vector has to be assigned before the dimensions size can be checked 
      if len(self.axes) == len(self.shape): # update length is all we can do without a coordinate vector
        for ax,n in zip(self.axes,self.shape): 
          if ax.data and ax.len != n: 
            raise DataError, "Length of axis '{:s} incompatible with data dimensions ({:d} vs. {:d}).".format(ax.name,ax.len,n) 
          else: ax.len = n 
      else: # this should only happen with scalar variables!
        if not ( self.ndim == 0 and data.size == 1 ): 
          raise DataError, 'Dimensions of data array and variable must be identical, except for scalars!'
    # just return itself, operations are in-place  
    return self       
     
  def unload(self):
    ''' Method to unlink data array. (also calls garbage collection)'''
    del self.__dict__['data_array'] # delete array
    self.__dict__['data_array'] = None # unlink data array
    # self.__dict__['shape'] = None # retain shape for later use
    gc.collect() # enforce garbage collection
      
  def getArray(self, idx=None, axes=None, broadcast=False, unmask=False, fillValue=None, copy=True):
    ''' Copy the entire data array or a slice; option to unmask and to reorder/reshape to specified axes. '''
    # without data, this will fail
    if self.data:
      if copy: datacopy = self.data_array.copy() # copy, if desired
      else: datacopy = self.data_array 
      # unmask    
      if unmask and self.masked: 
        # N.B.: if no data is loaded, self.mask is usually false...
        if fillValue is None: fillValue = self.fillValue
        datacopy = datacopy.filled(fill_value=fillValue) # I don't know if this generates a copy or not...
      # reorder and reshape to match axes (add missing dimensions as singleton dimensions)
      if axes is not None:
        if idx is not None: raise NotImplementedError
        for ax in self.axes:
          assert (ax in axes) or (ax.name in axes), "Can not broadcast Variable '{:s}' to dimension '{:s}' ".format(self.name,ax.name)
        # order dimensions as in broadcast axes list
        order = [self.axisIndex(ax) for ax in axes if self.hasAxis(ax)] # indices of broadcast list axes in instance axes list (self.axes)
        datacopy = np.transpose(datacopy,axes=order) # reorder dimensions to match broadcast list
        # adapt shape for broadcasting (i.e. expand shape with singleton dimensions)
        shape = [1]*len(axes); z = 0
        for i in xrange(len(axes)):
          if self.hasAxis(axes[i]): 
            shape[i] = datacopy.shape[z] # indices of instance axes in broadcast axes list
            z += 1
        assert z == datacopy.ndim 
        datacopy = datacopy.reshape(shape)
      # true broadcasting: extend array to match given axes and dimensions
      if broadcast:
        assert all([isinstance(ax,Axis) and len(ax)>0 for ax in axes]),\
           'All axes need to have a defined length in order broadcast the array.'
        # get tiling list
        tiling = [len(ax) if l == 1 else 1 for ax,l in zip(axes,datacopy.shape)]
        datacopy = np.tile(datacopy, reps=tiling)
    else:
      raise DataError, "No data loaded (Variable '{:s}')".format(self.name)
    # return array
    return datacopy
    
  def mask(self, mask=None, maskValue=None, fillValue=None, invert=False, merge=True):
    ''' A method to add a mask to an unmasked array, or extend or replace an existing mask. '''
    if mask is not None:
      assert isinstance(mask,np.ndarray) or isinstance(mask,Variable), 'Mask has to be a numpy array or a Variable instance!'
      # 'mask' can be a variable
      if isinstance(mask,Variable): mask = mask.getArray(unmask=True,axes=self.axes,broadcast=True)
      assert isinstance(mask,np.ndarray), 'Mask has to be convertible to a numpy array!'      
      # if 'mask' has less dimensions than the variable, it can be extended      
      if len(self.shape) < len(mask.shape): raise AxisError, 'Data array needs to have the same number of dimensions or more than the mask!'
      if self.shape[self.ndim-mask.ndim:] != mask.shape: raise AxisError, 'Data array and mask have to be of the same shape!'
      # convert to a boolean numpy array
      if invert: mask = ( mask == 0 ) # mask where zero or False 
      else: mask = ( mask != 0 ) # mask where non-zero or True
      # broadcast mask to data array
      mask = np.broadcast_arrays(mask,self.data_array)[0] # only need first element (the broadcasted mask)
      # create new data array
      if merge and self.masked: # the first mask is usually the land-sea mask, which we want to keep
        data = self.getArray(unmask=False) # get data with mask
        mask = ma.mask_or(data.mask, mask, copy=True, shrink=False) # merge masks
      else:         
        data = self.getArray(unmask=False) # don't fill missing values!
        if self.masked: data.mask = ma.nomask # unmask, sort of...
      self.__dict__['data_array'] = ma.array(data, mask=mask)
    elif maskValue is not None:
      if isinstance(self.dtype,(int,bool,np.integer,np.bool)): 
        self.__dict__['data_array'] = ma.masked_equal(self.data_array, maskValue, copy=False)
      elif isinstance(self.dtype,(float,np.inexact)):
        self.__dict__['data_array'] = ma.masked_values(self.data_array, maskValue, copy=False)
    # update fill value (stored in atts dict)
    self.fillValue = fillValue or ( self.data_array.fill_value if self.data_array._fill_value is None
                                    else self.data_array._fill_value )
    # as usual, return self
    return self
    
  def unmask(self, fillValue=None):
    ''' A method to remove an existing mask and fill the gaps with fillValue. '''
    if self.masked:
      if fillValue is None: fillValue = self.fillValue # default
      self.__dict__['data_array'] = self.data_array.filled(fill_value=fillValue)
    # as usual, return self
    return self
      
  def getMask(self, nomask=False, axes=None, strict=False):
    ''' Get the mask of a masked array or return a boolean array of False (no mask); axes has to be a 
        tuple, list or set of Axis instances or names; 'strict' refers to matching of axes. '''
    if axes is not None and not isinstance(axes,(list,tuple,set)): raise TypeError
    # get mask    
    if nomask: mask = ma.getmask(self.data_array)
    else: mask = ma.getmaskarray(self.data_array)
    # select axes (reduce)    
    if axes is not None:
      axes = set([ax if strict or not isinstance(ax,basestring) else self.getAxis(ax) for ax in axes])
      slices = [slice(None) if ax in axes else 0 for ax in self.axes]
      mask = mask.__getitem__(slices)
    # return mask
    return mask
  
  def findValues(self, value, lidx=False, lfirst=False, lminmax=False, lstrip=True, 
                 lflatten=False, lsqueeze=True):
    ''' Method to find all or only the first occurence of a value or a range of values and return the 
        coordinate or index values; The single occurence algorithm is actually pretty slow but works 
        for all types of data; the main difference to the Axis method getIndex is that it does not 
        require sorted data.
        Currently this is only tested with single-axis Variables ('pseudo-axes'), or with flattened 
        arrays, not with multi-dimensional fields.  '''
    if not self.data: self.load()
    if self.ndim != 1 and not lflatten: 
      raise NotImplementedError, "findValue() currently only works with single-axis 'pseudo-axes', not with multi-dimensional fields"
    if lflatten: data = self.data_array.ravel() # just a 'view', flatten() returns a copy
    else: data = self.data_array
    lstrip = lstrip and self.strvar # doesn't apply to non-string variables
    if lfirst or lstrip:
      # generate list of matches by iterating over elements
      # N.B.: usually this will be used for categorical data like int or str anyway...    
      idx = None if lfirst else []
      if isinstance(value,(tuple,list,np.ndarray)):
        # now scan through the values to extract matching index
        if lstrip: 
          vlen = min(len(val) for val in value)
          lstrip = vlen < self.dtype.itemsize
          if vlen > self.dtype.itemsize: 
            raise ValueError, "Value is longer than string length: {:d} > {:d}".format(vlen,self.dtype.itemsize)
        for i,vv in enumerate(data):
          if lstrip and len(vv) > vlen: vv = vv.rstrip() # strip trailing spaces (strvars get padded)
          if vv in value: 
            if lfirst: idx = i; break # terminate at first match
            else: idx.append(i) # add to list of hits
      else:
        if lstrip: 
          vlen = len(value)
          lstrip = vlen < self.dtype.itemsize 
          if vlen > self.dtype.itemsize: 
            raise ValueError, "Value is longer than string length: {:d} > {:d}".format(vlen,self.dtype.itemsize)
        for i,vv in enumerate(data):
          # N.B.: this way we avoid false positives due to too short strings
          if lstrip and len(vv) > vlen: vv = vv.rstrip() # strip trailing spaces (strvars get padded)
          if vv == value: 
            if lfirst: idx = i; break # terminate at first match
            else: idx.append(i) # add to list of hits
    else:
      # use numpy's nonzero-function
      if isinstance(value,(tuple,list,np.ndarray)):
        if lminmax:
          if len(value) != 2: raise ArgumentError, "Lists can only be interpreted as upper and lower bounds."
          tmp = np.ones_like(data, dtype=np.bool_)
          if value[0] is not None: tmp *= (value[0] < data)
          if value[1] is not None: tmp *= (data < value[1]) 
          idx = np.nonzero(tmp)[0]
        else:
          tmp = np.ones(self.shape, dtype=np.bool_)
          for val in value: tmp *= ( val == data ) # loop over all values and take 'and'
          idx = np.nonzero(tmp)[0] # tuple with indices for each dimension...
      else:
        idx = np.nonzero(data == value)[0]
    # cast as integer array
    idx = np.asarray(idx, dtype=np.int_)
    if idx is None:
      # possible problems with floats
      if np.issubdtype(self.dtype, np.inexact): 
        warn("The current implementation may fail for floats due to machine precision differences (non-exact match).") 
      #raise AxisError, "Value '{:s}' not found in Variable '{:s}'.".format(str(value),self.name)
    else:
      if lsqueeze:
        idx = idx.squeeze()
        if idx.size == 1: idx = np.asscalar(idx) 
      if not lidx: idx = self.axes[0].coord[idx] # convert to coordinates
    # return index of coordinate value
    return idx
  

  def findValuesND(self, values, lidx=False, lflatten=False, ltranspose=False):
    ''' Method to find exact matches of values and return their coordinate or index values; 
        unlike findValue, this method works for multiple dimensions, but is less flexible. '''
    if not self.data: self.load()
    if not isinstance(values,(tuple,list,np.ndarray)): raise 
    TypeError                
    if lflatten: data = self.data_array.ravel() # just a 'view'
    else: data = self.data_array
    # N.B.: this function really only finds exact matches
    # pad strings with spaces
    if self.strvar: values = [value+' '*(self.strlen-len(value)) for value in values]
    # now use numpy to extract matching index
    idxs = np.in1d(data.ravel(), values).reshape(self.shape) # find exact matches
    idxs = np.nonzero(idxs) # find indices of matches
    if len(idxs[0]) == 0:
      # possible problems with floats
      if np.issubdtype(self.dtype, np.inexact): 
        warn("The current implementation may fail for floats due to machine precision differences (non-exact match).") 
      #raise AxisError, "Value '{:s}' not found in Variable '{:s}'.".format(str(value),self.name)
    elif not lidx:
      idxs = tuple(ax.coord[ix] for ix,ax in zip(idxs,self.axes))
    # transpose results
    if ltranspose: idxs = [tpl for tpl in zip(*idxs)]        
    # return index of coordinate value
    return idxs
  
  def limits(self):
    ''' A convenience function to return a min,max tuple to indicate the data range. '''
    if self.data:
      mn,mx = self.data_array.min(), self.data_array.max()
      if np.isnan(mn): mn = np.nanmin(self.data_array)
      if np.isnan(mx): mx = np.nanmax(self.data_array)
      return mn,mx
    else: 
      return None
    
  # decorator arguments: slcaxes are passed on to slicing, axis and axes are converted to axidx
  #                      (axes is a list of reduction axes that are applied in sequence)
  # ReduceVar(asVar=None, axis=None, axes=None, lcheckAxis=True, **slcaxes)
  
  @ReduceVar
  def sum(self, data, axidx=None):
    data = nf.nansum(data, axis=axidx)
    name = '{:s}_sum'.format(self.name)
    units = self.units
    return data, name, units
  
  @ReduceVar
  def mean(self, data, axidx=None):
    data = nf.nanmean(data, axis=axidx)
    name = '{:s}_mean'.format(self.name)
    units = self.units
    return data, name, units
  
  @ReduceVar
  def std(self, data, axidx=None, ddof=0):
    data = nf.nanstd(data, axis=axidx, ddof=ddof) # ddof: degrees of freedom
    name = '{:s}_std'.format(self.name)
    units = self.units # variance has squared units    
    return data, name, units
 
  @ReduceVar
  def sem(self, data, axidx=None, ddof=0):
    data = nf.nansem(data, axis=axidx, ddof=ddof) # ddof: degrees of freedom
    name = '{:s}_sem'.format(self.name)
    units = self.units # variance has squared units    
    return data, name, units
  
  @ReduceVar
  def var(self, data, axidx=None, ddof=0):
    data = nf.nanvar(data, axis=axidx, ddof=ddof) # ddof: degrees of freedom
    name = '{:s}_var'.format(self.name)
    units = '({:s})^2'.format(self.units) # variance has squared units    
    return data, name, units
  
  @ReduceVar
  def max(self, data, axidx=None):
    data = nf.nanmax(data, axis=axidx)
    name = '{:s}_max'.format(self.name)
    units = self.units
    return data, name, units
  
  @ReduceVar
  def min(self, data, axidx=None):
    data = nf.nanmin(data, axis=axidx)
    name = '{:s}_min'.format(self.name)
    units = self.units
    return data, name, units
  
  def reduce(self, operation, blklen=None, blkidx=None, axis=None, mode=None, offset=0, 
                  asVar=None, axatts=None, varatts=None, fillValue=None, data_view=None,
                  lcheckVar=True, lcheckAxis=True, **kwargs):
    ''' Reduce a time-series; there are two modes:
          'block'     reduce to one value representing each block, e.g. from monthly to yearly averages;
                      specify a subset of elements from each block with blkidx
          'periodic'  reduce to one values representing each element of a block,
                      e.g. a monthly seasonal cycle from monthly data ;
                      specify a subset of block with blkidx, but use all elements in each block
                      '''
    ## check input
    lblk = False; lperi = False; lall = False
    if mode == 'block': lblk = True
    elif mode == 'periodic': lperi = True
    elif mode == 'all': lall = True
    else: raise ArgumentError 
    # check variables
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Reduction does not work with string Variables!"
      else: return None
    # reduction axis
    if not self.hasAxis(axis): 
      if lcheckAxis: raise AxisError, 'Reduction operations require a reduction axis!'
      else: return None # just do nothing and return original Variable
    if isinstance(axis,basestring): axis = self.getAxis(axis)
    if not isinstance(axis,Axis): raise TypeError
    axlen = len(axis); iax = self.axisIndex(axis)
    # offset definition: start blocks at this index
    if offset != 0: raise NotImplementedError
    # block definition
    if blklen is not None:
      if isinstance(blkidx,(list,tuple,np.ndarray)): 
        blkidx = np.asarray(blkidx, dtype='int')
      elif blkidx is not None: raise TypeError
    else: raise ArgumentError
    # N.B.: for mode 'all', blklen is simply the length of the new dimension
    if not isInt(blklen): raise TypeError
    if blkidx is not None:
      if blklen < np.max(blkidx) or np.min(blkidx) < 0: ArgumentError 
    if lblk or lperi: nblks = axlen/blklen # number of blocks
    # more checks
    if ( lblk or lperi ) and axlen%blklen != 0: 
      raise NotImplementedError, 'Currently seasonal means only work for full years.'
    if not self.data: self.load()
    ## massage data
    # get actual data and reshape
    if data_view is None: odata = self.getArray()
    else: odata = data_view
    # move reduction axis to the end, so that it is fastes varying
    if iax < self.ndim-1: odata = np.rollaxis(odata, axis=iax, start=self.ndim)
    # reshape
    oshape = odata.shape
    # make length of blocks the last axis, the number of blocks second to last
    if lblk or lperi: 
      odata = odata.reshape(oshape[:-1]+(nblks,blklen,))
      if lperi: odata = np.swapaxes(odata, -1, -2) # swap last and second to last
    # predict resultign shape
    if lblk: # use as is 
      rshape = oshape[:-1] + (nblks,) # shape of results array
    elif lperi or lall: 
      rshape = oshape[:-1] + (blklen,) if blklen > 0 else oshape[:-1] # shape of results array
    # extract block slice
    if blkidx is not None: tdata = odata.take(blkidx, axis=-1)
    else: tdata = odata
    # N.B.: this does different things depending on the mode:
    #       block: use a subset of elements from each block, but use all blocks
    #       periodic: use a subset of blocks, but all elements in each block 
    ## apply operation
    if fillValue is not None and self.masked: tdata = tdata.filled(fillValue)
    rdata = operation(tdata, axis=-1, **kwargs)
    assert rdata.shape == rshape
    # return new variable
    if iax < self.ndim-1 and blklen > 0: 
      rdata = np.rollaxis(rdata, axis=self.ndim-1, start=iax) # move reduction axis back
    # cast as variable
    if asVar:      
      # create new time axis (yearly)
      oaxis = self.axes[iax]
      raxatts = oaxis.atts.copy()      
      if axatts is not None: raxatts.update(axatts)      
      # define coordinates for new axis
      if 'coord' in raxatts:
        coord = raxatts.pop('coord') # use user-defiend coordinates 
      elif lblk: # use the beginning of each block as new coordinates (not divided by block length!) 
        coord = oaxis.coord.reshape(nblks,blklen)[:,0].copy()
      elif lperi or lall: # just enumerate block elements
        coord = np.arange(blklen) if blklen > 0 else None
      axes = list(self.axes)
      if blklen > 0: axes[iax] = Axis(coord=coord, atts=raxatts)
      else: del axes[iax] # just remove this axis, if blklen is zero
      # create new variable
      vatts = self.atts.copy()
      if varatts is not None: vatts.update(varatts)
      rvar = self.copy(data=rdata, axes=axes, atts=vatts)      
    else: # just return data array 
      rvar = rdata
    # return results
    return rvar
  
  def histogram(self, bins=None, binedgs=None, ldensity=True, asVar=True, name=None, axis=None, axis_idx=None, 
                lflatten=False, lcheckVar=True, lcheckAxis=True, haxatts=None, hvaratts=None, fillValue=None, **kwargs):
    ''' Generate a histogram of along a given axis and preserve the other axes. '''
    # some input checking
    if lflatten and axis is not None: raise ArgumentError
    if not lflatten and axis is None: 
      if self.ndim > 1: raise ArgumentError
      else: lflatten = True # treat both as flat arrays...
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Histogram does not work with string Variables!"
      else: return None
    if axis_idx is not None and axis is None: axis = self.axes[axis_idx]
    elif axis_idx is None and axis is not None: axis_idx = self.axisIndex(axis)
    elif not lflatten: raise ArgumentError    
    if axis is not None and not self.hasAxis(axis):
      if lcheckAxis: raise AxisError, "Variable '{:s}' has no axis '{:s}'.".format(self.name, axis)
      else: return None
    kwargs['density'] = ldensity # overwrite parameter
    # figure out bins
    bins, binedgs = binedges(bins=bins, binedgs=binedgs, limits=self.limits(), lcheckVar=lcheckVar)
    # setup histogram axis and variable attributes (special case)
    if asVar:
      axatts = self.atts.copy() # variable values become axis
      axatts['name'] = '{:s}_bins'.format(self.name) # '_bins' suffix to indicate histogram axis
      axatts['units'] = self.units # the histogram axis has the same units as the variable
      axatts['long_name'] = '{:s} Axis'.format(self.atts.get('long_name',self.name.title()))    
      if haxatts is not None: axatts.update(haxatts)
      varatts = self.atts.copy() # this is either density or frequency
      if ldensity:
        varatts['name'] = name or '{:s}_pdf'.format(self.name)
        varatts['long_name'] = 'PDF of {:s}'.format(self.atts.get('long_name',self.name.title()))
        varatts['units'] = '' # density
      else:
        varatts['name'] = name or '{:s}_hist'.format(self.name)
        varatts['long_name'] = 'Histogram of {:s}'.format(self.atts.get('long_name',self.name.title()))
        varatts['units'] = '#' # count    
      if hvaratts is not None: varatts.update(hvaratts)
    else:
      varatts = None; axatts = dict() # axatts is used later
    # choose a fillValue, because np.histogram does not ignore masked values but does ignore NaNs
    if fillValue is None and self.masked:
      if np.issubdtype(self.dtype,np.integer): fillValue = binedgs[-1]+1
      elif np.issubdtype(self.dtype,np.inexact): fillValue = np.NaN
      else: raise NotImplementedError
    # define functions that perform actual computation
    # N.B.: these "operations" will be called through the reduce method (see above for details)
    if lflatten: # totally by-pass reduce()...
      # this is actually the default behavior of np.histogram()
      if self.masked: data = self.data_array.filled(fillValue)
      else: data = self.data_array
      # N.B.: to ignore masked values they have to be replaced by NaNs or out-of-bounds values 
      hdata, bin_edges = np.histogram(data, bins=binedgs, **kwargs) # will flatten automatically
      assert isEqual(bin_edges, binedgs)
      assert hdata.shape == (len(binedgs)-1,)
      # create new Axis and Variable objects (1-D)
      if asVar: hvar = Variable(data=hdata, axes=(Axis(coord=bins, atts=axatts),), atts=varatts)
      else: hvar = hdata
    else: # use reduce to only apply to selected axis      
      # create a helper function that apllies the histogram along the specified axis
      def histfct(data, axis=None):
        if axis < 0: axis += data.ndim
        fct = functools.partial(histogram, bins=binedgs, **kwargs)         
        hdata = apply_along_axis(fct, axis, data,)
        # N.B.: the additional output of np.histogram must be suppressed, or np.apply_along_axis will
        #       expand everything as tuples!
        assert hdata.shape[axis] == len(binedgs)-1
        assert hdata.shape[:axis] == data.shape[:axis]
        assert hdata.shape[axis+1:] == data.shape[axis+1:]
        return hdata
      # call reduce to perform operation
      axatts['coord'] = bins # reduce() reads this and uses it as new axis coordinates
      hvar = self.reduce(operation=histfct, blklen=len(bins), blkidx=None, axis=axis, mode='all', 
                         offset=0, asVar=asVar, axatts=axatts, varatts=varatts, fillValue=fillValue)
    if asVar:
      if ldensity: hvar.plot = variablePlotatts['pdf'].copy()
      else: hvar.plot = variablePlotatts['hist'].copy()
    # return new variable instance (or data)
    return hvar
    
  def CDF(self, bins=None, binedgs=None, lnormalize=True, asVar=True, name=None, axis=None, axis_idx=None, 
          lflatten=False, lcheckVar=True, lcheckAxis=True, caxatts=None, cvaratts=None, fillValue=None, **kwargs):
    ''' Generate a histogram of along a given axis and preserve the other axes. '''
    # some input checking
    if lflatten and axis is not None: raise ArgumentError
    if not lflatten and axis is None: 
      if self.ndim > 1: raise ArgumentError
      else: lflatten = True # treat both as flat arrays...
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "CDF does not work with string Variables!"
      else: return None
    if axis_idx is not None and axis is None: axis = self.axes[axis_idx]
    elif axis_idx is None and axis is not None: axis_idx = self.axisIndex(axis)
    elif not lflatten: raise ArgumentError
    if axis is not None and not self.hasAxis(axis):
      if lcheckAxis: raise AxisError, "Variable '{:s}' has no axis '{:s}'.".format(self.name, axis)
      else: return None
    # let histogram worry about the bins...
    # setup CDF variable attributes (special case)
    if asVar:
      varatts = self.atts.copy() # this is either density or frequency
      varatts['name'] = name or '{:s}_cdf'.format(self.name)
      varatts['long_name'] = 'CDF of {:s}'.format(self.atts.get('long_name',self.name.title()))
      varatts['units'] = '' if lnormalize else '#' # count    
      if cvaratts is not None: varatts.update(cvaratts)
    else: varatts = None
    axatts = None # axis is the same as histogram
    # let histogram handle fill values and other stuff    
    # call histogram to perform the computation
    cvar = self.histogram(bins=bins, binedgs=binedgs, ldensity=False, asVar=asVar, name=name, 
                          axis=axis, lflatten=lflatten, lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, 
                          haxatts=axatts, hvaratts=varatts, fillValue=fillValue, **kwargs)
    # compute actual CDF
    if asVar: data = cvar.data_array
    else: data = cvar
    if lnormalize: normsum = np.sum(data, axis=axis_idx) 
    np.cumsum(data, axis=axis_idx, out=data) # do this in-place!
#     data = np.cumsum(data, axis=axis_idx)
    if lnormalize: data /= normsum 
    # update and polish variable
    if asVar:
      cvar.load(data) # load new CDF data
      cvar.plot = variablePlotatts['cdf'].copy()
    else: cvar = data
    # return new variable instance (or data)
    return cvar

  def percentile(self, q=None, asVar=True, name=None, axis=None, axis_idx=None, lflatten=False,  
                 lcheckVar=True, lcheckAxis=True, qaxatts=None, qvaratts=None, fillValue=None, **kwargs):
    ''' Compute percentiles along a given axis and preserve the other axes. 
        N.B.: this involves sorting the array and hence makes a full copy of the data '''
    # some input checking
    if lflatten and axis is not None: raise ArgumentError
    if not lflatten and axis is None: 
      if self.ndim > 1: raise ArgumentError
      else: lflatten = True # treat both as flat arrays...
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Percentiles do not work with string Variables!"
      else: return None
    if axis_idx is not None and axis is None: axis = self.axes[axis_idx]
    elif axis_idx is None and axis is not None: axis_idx = self.axisIndex(axis)
    elif not lflatten: raise ArgumentError    
    if axis is not None and not self.hasAxis(axis):
      if lcheckAxis: raise AxisError, "Variable '{:s}' has no axis '{:s}'.".format(self.name, axis)
      else: return None
    # setup histogram axis and variable attributes (special case)
    if asVar:
      axatts = self.atts.copy() # variable values become axis
      axatts['name'] = 'percentile'.format(self.name) # '_q' suffix to indicate percentiles axis
      axatts['units'] = '' # the histogram axis has the same units as the variable
      axatts['long_name'] = 'Percentile Axis'    
      if qaxatts is not None: axatts.update(qaxatts)
      varatts = self.atts.copy() # 
      varatts['name'] = name or '{:s}_q'.format(self.name)
      varatts['long_name'] = 'Certain Percentiles of {:s}'.format(self.atts.get('long_name',self.name.title()))
      varatts['units'] = self.units # count    
      if qvaratts is not None: varatts.update(qvaratts)
    else:
      varatts = None; axatts = dict() # axatts is used later
    # choose a fillValue, because np.histogram does not ignore masked values but does ignore NaNs
    if fillValue is None and self.masked:
      if np.issubdtype(self.dtype,np.integer): raise NotImplementedError # convert to float and use NaN?
      elif np.issubdtype(self.dtype,np.inexact): fillValue = np.NaN
      else: raise NotImplementedError
    # check percentiles
    if isinstance(q, (list,np.ndarray)): q = tuple(q) # scalar values  
    if not isinstance(q,tuple): raise TypeError, "percentiles have to be a sequence"
    qcoord = np.asarray(q) # for percentile axis
    if qcoord.max() > 1 or qcoord.min() < 0: raise ValueError
    q = tuple(100.*qq for qq in q) # for percentile function
    # define functions that perform actual computation
    # N.B.: these "operations" will be called through the reduce method (see above for details)
    if lflatten: # totally by-pass reduce()...
      # this is actually the default behavior of np.histogram()
      if self.masked: data = self.data_array.filled(fillValue)
      else: data = self.data_array
      # N.B.: to ignore masked values they have to be replaced by NaNs or out-of-bounds values 
      qdata = percentile(data.ravel(), q, axis)
      # create new Axis and Variable objects (1-D)
      if asVar: qvar = Variable(data=qdata, axes=(Axis(coord=qcoord, atts=axatts),), atts=varatts)
      else: qvar = qdata
    else: # use reduce to only apply to selected axis      
      # create a helper function that apllies the histogram along the specified axis
      def qfct(data, axis=None):
        if axis < 0: axis += data.ndim
        fct = functools.partial(percentile, q=q, axis=axis)         
        qdata = apply_along_axis(fct, axis, data, laax=False)
        return qdata
      # call reduce to perform operation
      axatts['coord'] = qcoord # reduce() reads this and uses it as new axis coordinates
      qvar = self.reduce(operation=qfct, blklen=len(q), blkidx=None, axis=axis, mode='all', 
                         offset=0, asVar=asVar, axatts=axatts, varatts=varatts, fillValue=fillValue)
#     if asVar: qvar.plot = variablePlotatts[self.name].copy()
    # return new variable instance (or data)
    return qvar
  
  def apply_stat_test(self, asVar=True, name=None, axis=None, axis_idx=None, test=None, dist='norm', 
                      lflatten=False, lstatistic=False, lonesided=False, fillValue=None, ignoreNaN=None,
                      lcheckVar=True, lcheckAxis=True, paxatts=None, pvaratts=None, **kwargs):
    ''' Apply a statistical test along a axis and return a variable containing the resulting p-values. '''
    if ignoreNaN is None:
      ignoreNaN = test.lower() != 'normaltest' or lflatten
    # some input checking
    if lflatten and axis is not None: raise ArgumentError
    if not lflatten and axis is None: 
      if self.ndim > 1: raise ArgumentError
      else: lflatten = True # treat both as flat arrays...
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Statistical tests does not work with string Variables!"
      else: return None
    var = self # to avoid confusion later
    if axis_idx is not None and axis is None: axis = self.axes[axis_idx]
    elif axis_idx is None and axis is not None: 
      if isinstance(axis,(tuple,list)):
        # if sample includes multiple axes, merge them and introduce new sample axis
        var = var.mergeAxes(axes=axis, new_axis='sample', asVar=True, linplace=False, lcheckAxis=lcheckAxis)
        if var is None: return None
        axis = 'sample' # now operate on the new sample axis, as if it was just one    
      axis_idx = var.axisIndex(axis)
    elif not lflatten: raise ArgumentError    
    if axis is not None and not var.hasAxis(axis):
      if lcheckAxis: raise AxisError, "Variable '{:s}' has no axis '{:s}'.".format(self.name, axis)
      else: return None
    if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
    # setup histogram axis and variable attributes (special case)
    if asVar:
      if paxatts is not None: raise NotImplementedError
      varatts = self.atts.copy()
      varatts['name'] = name or '{:s}_{:s}_pval'.format(self.name,dist)
      varatts['long_name'] = "p-value of {:s} for '{:s}'-distribution".format(
                              self.atts.get('long_name',self.name.title()),dist)
      varatts['units'] = '' # p-value / probability
      if pvaratts is not None: varatts.update(pvaratts)
    else:
      varatts = None # axatts = dict() # axatts is used later
    # choose a fillValue, because np.histogram does not ignore masked values but does ignore NaNs
    if fillValue is None and var.masked:
      if np.issubdtype(var.dtype,np.integer): fillValue = 0
      elif np.issubdtype(var.dtype,np.inexact): fillValue = np.NaN
      else: raise NotImplementedError
    # import test wrappers (need to do here, to prevent circular reference)
    from geodata.stats import anderson_wrapper, kstest_wrapper, normaltest_wrapper, shapiro_wrapper
    if lflatten: # totally by-pass reduce()...
      # get data
      if isinstance(var.data_array,ma.MaskedArray): # DistVars treat masks slightly differently
        data = var.data_array.filled(fillValue).ravel()
      else: data = var.data_array.ravel()
      # N.B.: to ignore masked values they have to be replaced by NaNs or out-of-bounds values 
      # select test function for flat/1D test
      if test.lower() in ('anderson',): 
        pval = anderson_wrapper(data, dist=dist, ignoreNaN=ignoreNaN)
      elif test.lower() in ('kstest',):
        pval = kstest_wrapper(data, dist=dist, ignoreNaN=ignoreNaN, **kwargs)
      elif test.lower() in ('normaltest',):
        pval = normaltest_wrapper(data, axis=None, ignoreNaN=ignoreNaN)
      elif test.lower() in ('shapiro',):
        pval = shapiro_wrapper(data, ignoreNaN=ignoreNaN, **kwargs) # runs only once
      else: raise NotImplementedError, test
      # create new Axis and Variable objects (1-D)
      if asVar: 
        raise NotImplementedError, "Cannot return a single scalar as a Variable object."
      else: pvar = pval
    else: # use reduce to only apply to selected axis      
      # select test function for multi-dimensional test
      # N.B.: these "operations" will be called through the reduce method (see above for details)
      if test.lower() in ('normaltest',):
        laax = False # don't need to use Numpy's apply_along_axis
        if ignoreNaN: 
          raise NotImplementedError, "NaN-removal does not work with 'normaltest' and multi-dimensional arrays."
        testfct = normaltest_wrapper
        # N.B.: the normaltest just works on multi-dimensional data
      else:
        laax = True # have to use Numpy's apply_along_axis
        if test.lower() in ('anderson',): 
          testfct = functools.partial(anderson_wrapper, dist=dist, ignoreNaN=ignoreNaN)
        elif test.lower() in ('kstest',):
          testfct = functools.partial(kstest_wrapper, dist=dist, ignoreNaN=ignoreNaN, **kwargs)
        elif test.lower() in ('shapiro',):
          lreta = kwargs.pop('reta',True) # default: True
          if lreta: 
            global shapiro_a # global variable to retain parameters
            shapiro_a = None # reset, just to be safe!
          testfct = functools.partial(shapiro_wrapper, reta=lreta, ignoreNaN=ignoreNaN)
        else: raise NotImplementedError, test
      # create a helper function that apllies the histogram along the specified axis
      def aaa_testfct(data, axis=None):
        if axis < 0: axis += data.ndim
        pval = apply_along_axis(testfct, axis, data, laax=laax)
        return pval
      # call reduce to perform operation
      pvar = var.reduce(operation=aaa_testfct, blklen=0, blkidx=None, axis=axis, mode='all', 
                         offset=0, asVar=asVar, axatts=None, varatts=varatts, fillValue=fillValue)
    if asVar: pvar.plot = variablePlotatts['pval'].copy()
    # return new variable instance (or data)
    return pvar
  
  def _createVar(self, axes=None, data=None, atts=None, linplace=False):
    ''' helper function to create new Variable objects from old ones '''
    if linplace:
      # replace axes and data
      self.axes = tuple(axes)
      self.data_array = np.asanyarray(data) if data is not None else None
      if atts is not None: self.atts.update(atts)
      newvar = self
    else:
      # create new variable
      newvar = self.copy(data=data, axes=axes, atts=self.atts if atts is None else atts)
    return newvar
    
  def _checkMonthlyAxis(self, taxis='time', lbegin=True, lclim=False):
    ''' helper function to check certain assumptions about the time axis '''
    time = self.getAxis(taxis); tcoord = time.coord
    # make sure the time axis is well-formatted, because we are making a lot of assumptions!
    if not time.units.lower() in monthlyUnitsList: 
      raise NotImplementedError, "Time units='month' required to extract seasons! (got '{:s}')".format(time.units)
    #if 'long_name' not in time.atts and lstrictCheck: raise KeyError, time.prettyPrint(short=False)
    if lbegin and ( (not lclim and tcoord[0]%12 != 0) or 
                    (lclim and tcoord[0]%12 != 1) ): 
      raise NotImplementedError, "Time-axis has to start in January; front-padding is currently not supported."
    # TODO: implement front-padding in _getCompleteYears(); maybe return front offset here
    if np.any( np.diff(tcoord, axis=0) != 1 ): 
      raise NotImplementedError, "Time-axis cannot have missing coordinates (month)!"
    # no return value - just throw exceptions
    
  def seasonalSample(self, season=None, asVar=True, lcheckAxis=False, lcheckVar=True, linplace=False, 
                    lstrict=True, taxis='time', svaratts=None, saxatts=None):
    ''' A method to extract a subset of month from a monthly timeseries and return a concatenated 
        time-series of values for the specified season. '''
    # check input
    if season is not None and self.hasAxis(taxis):
      time = self.getAxis(taxis); itime = self.axisIndex(taxis); tcoord = time.coord
      if lstrict: self._checkMonthlyAxis(taxis=taxis, lbegin=True)
      # translate season string
      idx = translateSeasons(season) # does most of the remining input/type checking
      # extend the list of indices to the length of the time axis
      idxlen = idx.size; tlen = tcoord.size; yrlen = tlen//12; tover = tlen%12
      # basically, construct a 2D array of years and month, and flatten afterwards
      idxarr = np.repeat(np.arange(0,yrlen*12,12, dtype=np.int32).reshape((yrlen,1)), repeats=idxlen, axis=1)
      idxarr = ( idxarr + idx ).ravel() # addition should broadcast automatically       
      # add incomplete year at the end (extend array)
      idxover = np.asarray([yrlen*12+i for i in idx if i < tover], dtype=np.int32)
      idxarr = np.concatenate((idxarr,idxover), axis=0)
      assert idxarr.min() >= 0 and idxarr.max() < tlen
      # slice data and coordinate vector
      data = self.data_array.take(idxarr, axis=itime)
      assert data.dtype == self.dtype
      assert data.shape == self.shape[:itime]+(len(idxarr),)+self.shape[itime+1:]
      if asVar:
        # create new axes
        coord = tcoord.take(idxarr, axis=0)
        assert len(coord) == len(idxarr)
        saxis = time.copy(coord=coord, atts=time.atts if saxatts is None else saxatts)        
        axes = self.axes[:itime]+(saxis,)+self.axes[itime+1:]
        # create new variable or modify old one in-place
        svar = self._createVar(axes=axes, data=data, atts=svaratts, linplace=linplace)
      else:
        if linplace: raise ArgumentError, linplace
        svar = data # just return new array      
    else:
      if lcheckAxis: raise AxisError, "Axis '{:s}' required to extract seasonal values!".format(taxis)
      else: svar = None # basically just skip and return None
    # return results
    return svar

  def _getCompleteYears(self, taxis='time', ltrim=False, lfront=False, lback=True, asVar=False, 
                        lcheck=True, lclim=False):
    ''' helper function that generates a trimmed or padded view of the data, either extending or 
        trimming the time axis ''' 
    if asVar: raise NotImplementedError, 'currently we can only return a bare array view, not a variable'
    # N.B.: to return a Variable, we would also have to trim/pad the time axis 
    if lfront: raise NotImplementedError, 'currently we can only pad the back, not the front'
    time = self.getAxis(taxis); itime = self.axisIndex(taxis); tcoord = time.coord
    if lcheck: self._checkMonthlyAxis(taxis=taxis, lbegin=not lfront, lclim=lclim)
    # define new shape
    tlen = tcoord.size; slen = tlen//12; tover = tlen%12 # determine dimension length
    if not self.data: raise DataError, 'Need to load data for trimming and padding.'
    data_view = self.data_array
    # handle incomplete years
    if tover > 0:
      if ltrim: 
        data_view = data_view.take(range(tlen-tover), axis=itime) # only use complete years 
      else:
        pshape = self.shape[:itime]+(12-tover,)+self.shape[itime+1:] # padding shape  
        padding = np.ones(pshape)*self.fillValue # create padding using fillValue 
        data_view = np.concatenate((data_view,padding), axis=itime) # append paddign array
        if self.masked and data_view.mask: 
          data_view.mask[...,tlen-tover:] = True # also mask pad, if applicable
        slen += 1
    assert data_view.shape[itime]%12 == 0, data_view.shape # should be divisible by 12 now
    # return padded/trimmed view
    return data_view
  
  def climSample(self, lstrict=True, ltrim=False, asVar=True, lcheckAxis=False, lcheckVar=True, linplace=False, 
                 taxis='time', saxis=None, saxatts=None, caxis=None, caxatts=None, svaratts=None):
    ''' A method to reshaped a Variable to a seasonal cycle axis and all samples (years) for each month
        along a new sample axis. '''    
    # check input
    if self.hasAxis(taxis):
      # get array view with complete years
      data = self._getCompleteYears(taxis=taxis, ltrim=ltrim, asVar=False, lcheck=lstrict)
      time = self.getAxis(taxis); itime = self.axisIndex(taxis); tcoord = time.coord
      assert data.shape[itime]%12 == 0, data.shape # should be divisible by 12 now      
      # define new shape
      tlen = tcoord.size; slen = tlen//12 
      nshape = self.shape[:itime]+(slen,12)+self.shape[itime+1:] # new shape
      # reshape data
      data = data.reshape(nshape)
      assert data.dtype == self.dtype and data.shape == nshape
      # create new axes and variable
      if asVar:
        if not caxis:
          # create new climatology axis
          caxis = time.copy(coord=np.arange(1,13), atts=time.atts if caxatts is None else caxatts)
        else:
          if not isinstance(caxis, Axis): raise TypeError, caxis 
          if len(caxis) != 12: raise Axis, caxis
        if saxis is None or isinstance(saxis,basestring):
          # create new sample axis (yearly)
          satts = self.time.atts.copy()
          satts['name'] = saxis if isinstance(saxis,basestring) else 'sample'
          satts['units'] = 'year' # defaults
          if saxatts is not None: satts.update(saxatts) 
          scoord = np.arange(tcoord[0]//12,tcoord[0]//12+slen)
          saxis = time.copy(coord=scoord, atts=satts)
        elif not isinstance(saxis,Axis): raise TypeError, saxis
        # insert new axes
        axes = self.axes[:itime]+(saxis,caxis)+self.axes[itime+1:]
        # create new variable or modify old one in-place
        svar = self._createVar(axes=axes, data=data, atts=svaratts, linplace=linplace)
      else:
        if linplace: raise ArgumentError, linplace
        svar = data # just return new array      
    else:
      if lcheckAxis: raise AxisError, "Axis '{:s}' required to extract seasonal values!".format(taxis)
      else: svar = None # basically just skip and return None
    # return results
    return svar
  
  def reduceToAnnual(self, season, operation, asVar=False, name=None, offset=0, taxis='time', 
                     checkUnits=True, lcheckVar=True, lcheckAxis=True, taxatts=None, varatts=None, 
                     mean_list=None, ltrim=False, lstrict=True, lclim=False, **kwargs):
    ''' Reduce a monthly time-series to an annual time-series, using mean/min/max over a subset of month or seasons. '''
    if not self.hasAxis(taxis): 
      if lcheckAxis: raise AxisError, 'Seasonal reduction requires a time axis!'
      else: return None # just skip and do nothing
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Seasonal reduction does not work with string Variables!"
      else: return None
    data_view = self._getCompleteYears(taxis=taxis, ltrim=ltrim, asVar=False, lcheck=lstrict, lclim=lclim)
    taxis = self.getAxis(taxis); te = len(taxis); tax = self.axisIndex(taxis.name)
    assert data_view.shape[self.axisIndex(taxis)]%12 == 0, data_view.shape # should be divisible by 12 now          
#     if checkUnits and not taxis.units.lower() in monthlyUnitsList: 
#       raise AxisError, "Seasonal reduction requires monthly data! (time units: '{:s}')".format(taxis.units)
#     te = len(taxis); tax = self.axisIndex(taxis.name)
#     if te%12 != 0 or not (taxis.coord[0]%12 == 0 or taxis.coord[0]%12 == 1): 
#       raise NotImplementedError, 'Currently seasonal reduction only works with full years.'
    # hadling of exceptions: some variables in Datasets should only be averaged
    if mean_list is not None and self.name in mean_list: operation = np.nanmean    
    # modify variable
    if asVar:      
      # create new time axis (yearly)
      tatts = self.time.atts.copy()
      tatts['name'] = 'year'; tatts['units'] = 'year' # defaults
      if taxatts is not None: tatts.update(taxatts)      
      # create new variable
      vatts = self.atts.copy()
      if name is not None: vatts['name'] = name
      elif isinstance(season,basestring): 
        vatts['name'] = '{:s}_{:s}'.format(self.name,season); # default 
      else: vatts['name'] = self.name
      vatts['units'] = self.units
      if varatts is not None: vatts.update(varatts)
    else: tatts = None; varatts = None # irrelevant
    # translate definiton of month or season
    idx = translateSeasons(season)
    # call general reduction function
    avar =  self.reduce(operation, blklen=12, blkidx=idx, axis=taxis, mode='block', offset=offset, 
                        asVar=asVar, axatts=tatts, varatts=varatts, data_view=data_view, 
                        lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, **kwargs)
    # check shape of annual variable
    assert avar.shape == self.shape[:tax]+(te/12,)+self.shape[tax+1:]
    # convert time coordinate to years (from month)
    if asVar:
      if tatts['units'].lower() == 'year' and taxis.units.lower() in monthlyUnitsList:
        raxis = avar.getAxis(tatts['name'])
        if taxis.coord[0]%12 == 1: # special treatment, if we start counting at 1(instead of 0)
          raxis.coord -= 1; raxis.coord /= 12; raxis.coord += 1  
        else: raxis.coord /= 12 # just divide by 12, assuming we count from 0
    # return data
    return avar
  
  def seasonalMean(self, season='annual', **kwargs):
    ''' Return a time-series of annual averages of the specified season. '''    
    return self.reduceToAnnual(season=season, operation=nf.nanmean, **kwargs)
  
  def seasonalSum(self, season='annual', **kwargs):
    ''' Return a time-series of annual sums of the specified season. '''    
    return self.reduceToAnnual(season=season, operation=nf.nansum, **kwargs)
  
  def seasonalStd(self, season='annual', **kwargs):
    ''' Return a time-series of annual root-mean-variances (of the specified season/months). '''    
    return self.reduceToAnnual(season=season, operation=nf.nanstd, **kwargs)
  
  def seasonalSEM(self, season='annual', **kwargs):
    ''' Return a time-series of annual standard errros of seasonal/monthly means. '''
    return self.reduceToAnnual(season=season, operation=nf.nansem, **kwargs)
  
  def seasonalVar(self, season='annual', **kwargs):
    ''' Return a time-series of annual root-mean-variances (of the specified season/months). '''    
    return self.reduceToAnnual(season=season, operation=nf.nanvar, **kwargs)
  
  def seasonalMax(self, season='annual', **kwargs):
    ''' Return a time-series of annual averages of the specified season. '''    
    return self.reduceToAnnual(season=season, operation=nf.nanmax, **kwargs)
  
  def seasonalMin(self, season='annual', **kwargs):
    ''' Return a time-series of annual averages of the specified season. '''    
    return self.reduceToAnnual(season=season, operation=nf.nanmin, **kwargs)
  
  def reduceToClimatology(self, operation, yridx=None, asVar=True, name=None, offset=0, taxis='time', 
                          lcheckVar=True, lcheckAxis=True, checkUnits=True, taxatts=None, varatts=None, 
                          mean_list=None, ltrim=False, lstrict=True, **kwargs):
    ''' Reduce a monthly time-series to an annual climatology; use 'yridx' to limit the reduction to 
        a set of years (identified by index) '''
    if not self.hasAxis(taxis): 
      if lcheckAxis: raise AxisError, 'Reduction to climatology requires a time axis!'
      else: return None # just skip and do nothing
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Reduction to climatology does not work with string Variables!"
      else: return None
    data_view = self._getCompleteYears(taxis=taxis, ltrim=ltrim, asVar=False, lcheck=lstrict)
    taxis = self.getAxis(taxis); tax = self.axisIndex(taxis.name)
    assert data_view.shape[self.axisIndex(taxis)]%12 == 0, data_view.shape # should be divisible by 12 now          
#     taxis = self.getAxis(taxis)    
#     if checkUnits and not taxis.units.lower() in monthlyUnitsList: 
#       raise AxisError, "Reduction to climatology requires monthly data! (time units: '{:s}')".format(taxis.units)
#     te = len(taxis); tax = self.axisIndex(taxis.name)
#     if te%12 != 0 or not (taxis.coord[0]%12 == 0 or taxis.coord[0]%12 == 1): 
#       raise NotImplementedError, 'Currently reduction to climatology only works with full years.'
    # hadling of exceptions: some variables in Datasets should only be averaged
    if mean_list is not None and self.name in mean_list: operation = np.nanmean    
    # modify variable
    if asVar:      
      # create new time axis (still monthly)
      tatts = self.time.atts.copy()
      if taxatts is not None: tatts.update(taxatts)      
      # create new variable
      vatts = self.atts.copy()
      if name is not None: vatts['name'] = name
      else: vatts['name'] = self.name
      vatts['units'] = self.units
      if varatts is not None: vatts.update(varatts)
    else: tatts = None; varatts = None # irrelevant
    # call general reduction function
    avar =  self.reduce(operation, blklen=12, blkidx=yridx, axis=taxis, mode='periodic',
                        offset=offset, asVar=asVar, axatts=tatts, varatts=varatts, 
                        lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, **kwargs)
    # check shape of annual variable
    assert avar.shape == self.shape[:tax]+(12,)+self.shape[tax+1:]
    # construct time coordinate
    if asVar:
      if tatts['units'].lower() in monthlyUnitsList:
        raxis = avar.getAxis(tatts['name'])
        if raxis.coord[0] == 0: raxis.coord += 1 # customarily, month are counted, starting at 1, not 0 
    # return data
    return avar
  
  def climMean(self, yridx=None, **kwargs):
    ''' Return a climatology of averages of monthly data. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nanmean, **kwargs)
  
  def climSum(self, yridx=None, **kwargs):
    ''' Return a climatology of sums of monthly data. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nansum, **kwargs)

  def climStd(self, yridx=None, **kwargs):
    ''' Return a climatology of root-mean-variances/standard deviation of monthly data. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nanstd, **kwargs)   
  
  def climSEM(self, yridx=None, **kwargs):
    ''' Return a climatology of the standard errros of monthly/climatological means. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nansem, **kwargs)   
  
  def climVar(self, yridx=None, **kwargs):
    ''' Return a climatology of (square) variances of monthly data. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nanvar, **kwargs)
  
  def climMax(self, yridx=None, **kwargs):
    ''' Return a climatology of maxima of monthly data. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nanmax, **kwargs)
  
  def climMin(self, yridx=None, **kwargs):
    ''' Return a climatology of minima of monthly data. '''    
    return self.reduceToClimatology(yridx=yridx, operation=nf.nanmin, **kwargs)
  
  def reorderAxes(self, axes=None, asVar=True, linplace=False, lcheckAxis=False):
    ''' reorder the axes of a Variable and replace the data array with an array view with 
        appropriately reordered dimensions '''
    # check and translate input (axes list)
    iaxes = [] 
    for ax in axes:
      if isinstance(ax,(basestring,Axis)): 
        if self.hasAxis(ax): iaxes.append(self.axisIndex(ax))
        elif lcheckAxis: raise AxisError, ax # default is to just skip
      else: iaxes.append(int(ax))
    # reorder data (create view)
    if self.data: data = self.data_array.transpose(iaxes)      
    # create new axes tuple and new variable
    if asVar: 
      axes = [self.axes[i] for i in iaxes] # direct list indexing doesn't seem to work
      var = self._createVar(axes=axes, data=data, linplace=linplace)
    else: var = data
    # return variable
    return var
  
  def _findAxes(self, axes=None, lcheckAxis=False, lstrict=False):
    ''' find indices for given axes; also return indices of remaining axes '''
    maxes = set() # removes duplicates 
    for ax in axes:
      if isinstance(ax,(basestring,Axis)): 
        if self.hasAxis(ax, strict=lstrict): maxes.add(self.axisIndex(ax, strict=lstrict))
        elif lcheckAxis: raise AxisError, ax # default is to just skip
      else: maxes.add(int(ax))
    raxes = set(range(self.ndim)).difference(maxes) # remaining axes that are not merged
    raxes = list(raxes); raxes.sort()
    maxes = list(maxes); maxes.sort() # get a sorted list
    # return indices of requested axes and remaining axes
    return maxes, raxes 
    
  def mergeAxes(self, axes=None, new_axis=None, axatts=None, asVar=True, linplace=False, 
                lcheckAxis=True, lcheckVar=None, lvarall=True, ldsall=False, lstrict=False):
    ''' merge several axes into one new axis; reorder axes to make them contiguous, if necessary '''
    # check dataset axes or remove missing axes
    if self.dataset: 
      tmp_axes = []; ds = self.dataset
      for ax in axes:
        if ds.hasAxis(ax, strict=lstrict): tmp_axes.append(ax)
        elif ldsall: 
          raise DatasetError, "Dataset '{:s}' does not have Axis '{:s}'.".format(ds.name,
                                                      ax.name if isinstance(ax,Axis) else ax)
      axes = tmp_axes
    # check axes determin reordering
    if len(axes) == 0: lcheck = False
    elif lvarall: lcheck = all(self.hasAxis(ax, strict=lstrict) for ax in axes)
    else: lcheck = any(self.hasAxis(ax, strict=lstrict) for ax in axes)
    # N.B.: if some variable have all axes and others only a subset, the resulting merged axes will
    #       have a different shape, so that added all of them to a Dataset will cause an error
    # process, if merge axes are present, otherwise skip
    if not lcheck:
      if lcheckAxis: raise AxisError, "No merge axis found in Variable {:s}".format(self.name)
      else: return None
    else:
      maxes,raxes = self._findAxes(axes=axes, lcheckAxis=lcheckAxis, lstrict=lstrict)
      # insert sorted merge axes at their first index
      imin = maxes[0]; imax = maxes[-1]; mdim = len(maxes)
      iin = 0 if imin == 0 else raxes.index(imin-1)+1
      iaxes = raxes[:iin] + maxes + raxes[iin:] # proper axes order
      assert imin == iaxes[iin] and imax == iaxes[iin+mdim-1], iaxes
      # reorder and make copy of variable, as necessary
      if iaxes == range(self.ndim): # no reordering necessary 
        var = self if linplace else self.copy()
      else: # call reordering method
        var = self.reorderAxes(axes=iaxes, asVar=True, linplace=linplace, lcheckAxis=lcheckAxis)
      # use variable instead of self from now on - more flexibility    
      # determine merge axes again (after reordering)
      maxes,raxes = var._findAxes(axes=axes, lcheckAxis=lcheckAxis)
      # reshape data (create view)
      nlen = 1 # total length of merged axes
      for i in maxes: nlen *= var.shape[i]
      if var.data:
        rshape = tuple(var.shape[i] for i in raxes)
        nshape = rshape[:imin]+(nlen,)+rshape[imin:]
        data = var.data_array.reshape(nshape)
      # create new axes tuple and new variable
      if asVar:
        # create new axis
        if new_axis is None or isinstance(new_axis,basestring):
          # create new sample axis (yearly)
          satts = dict(name='sample' if new_axis is None else new_axis, units='') # defaults
          if axatts is not None: satts.update(axatts) 
          new_axis = Axis(coord=np.arange(nlen), atts=satts)
        else:
          if not isinstance(new_axis,Axis): raise TypeError, new_axis
          if not len(new_axis)==nlen: raise AxisError, new_axis 
        varatts = {new_axis.name:tuple(ax.name if isinstance(ax,Axis) else ax for ax in axes)} # save merge history
        axes = var.axes[:imin]+(new_axis,)+var.axes[imin+mdim:] # finally, assemble new axes
        var = var._createVar(axes=axes, data=data, linplace=True, atts=varatts) # already made a copy
        assert var.shape == nshape, var.shape
      else: var = data
    # return variable
    return var
  
  def insertAxis(self, axis=None, iaxis=0, length=None, req_axes=None, asVar=True, lcheckVar=None, 
                 lcheckAxis=True, lstrict=False, linplace=False, lcopy=True, ltile=True):
    ''' Insert a dummy axis (and 'tile' the data) at a given location; to allow meaningful 
        application to datasets, a required axes 'req_axes' set can be specified; an Axis instance
        can be created on-the-fly from a name ('axis') and a length ('length'). '''
    if req_axes is None or all(self.hasAxis(ax, strict=lstrict) for ax in req_axes):
      # generate new Axis instance, if necessary
      if isinstance(axis,basestring):
        if not isinstance(length, (np.integer,int)) and length>0: raise AxisError, axis
        axis = Axis(name=axis, length=length, coord=np.arange(length), atts=dict(note='dummy axis')) 
      if not isinstance(axis,Axis): raise TypeError, axis      
      # generate new_axes argument (insert new axis into old axes tuple)
      new_axes = self.axes[:iaxis] + (axis,) + self.axes[iaxis:]
      # call insertAxes method (some checks have already been performed here)
      var = self.insertAxes(new_axes=new_axes, req_axes=None, asVar=asVar, lcheckVar=lcheckVar, 
                            lcheckAxis=False, lstrict=lstrict, linplace=linplace, 
                            lcopy=lcopy, ltile=ltile)
    else:
      if lcheckAxis: raise AxisError, req_axes 
      else: var = None # don't do anything
    # return new variable
    return var

  def insertAxes(self, new_axes=None, req_axes=None, asVar=True, lcheckVar=None, lcheckAxis=True, 
                 lcopy=True, ltile=True, lstrict=False, linplace=False):
    ''' Insert dummy axes (and 'tile' the data) to match the axes in 'new_axes'; to allow meaningful 
        application to datasets, a required axes 'req_axes' set can be specified '''
    if not lcopy: warn("Replicating an array without duplicating memory is not save w.r.t. reshaping and aggregating!")
    if req_axes is None or all(self.hasAxis(ax, strict=lstrict) for ax in req_axes):
      axes = [] # store new axes
      ldata = self.data
      if ldata:
        reshape_axes = []; new_shape = []
        if lcopy: tile_axes = []
        else: axes_strides = [] 
        data = self.data_array.copy() # view on data array
      else: data=None
      # loop over new axes
      if not lcopy and ldata: i = 0
      for ax in new_axes:
        # add new axis or keep existing axis
        if self.hasAxis(ax, strict=lstrict):
          if isinstance(ax,Axis):
            own_axis = self.axes[self.axisIndex(ax.name)]
            if len(own_axis) != len(ax) and lcheckAxis: 
              raise AxisError, "New and old axes have incompatible length!"
          else: own_axis = self.axes[self.axisIndex(ax)]
          axes.append(own_axis) # keep old axis
          if ldata: 
            reshape_axes.append(len(own_axis))
            new_shape.append(len(own_axis))
            if lcopy: tile_axes.append(1) # don't tile along existing axes
            else: 
              axes_strides.append(data.strides[i]); i += 1 # copy strides from original array
        else:
          if not isinstance(ax,Axis): raise AxisError, "A new axis has to be an Axis instance."
          axes.append(ax) # add new axis
          if ldata: 
            reshape_axes.append(1) # insert new singleton dimension
            new_shape.append(len(ax))
            if lcopy: tile_axes.append(len(ax)) # tile along new axes
            else: axes_strides.append(0) 
            # N.B.: if we set the stride to 0, we always get back to the same section in memory
      assert len(axes) == len(new_axes)
      assert not lcopy or tuple(new_shape) == tuple(tl if rl==1 else rl for rl,tl in zip(reshape_axes,tile_axes))
      assert lcopy or not ldata or i == data.ndim      
      # reshape/tile data
      if ldata:
        reshape_axes = tuple(reshape_axes); new_shape = tuple(new_shape)
        data = data.reshape(reshape_axes) # insert new dimensions
        if lcopy: 
          if ltile: data = np.tile(data, tuple(tile_axes)) # repeat along new (singleton) dimensions
          else:
            for iax,lax in enumerate(tile_axes):
              if lax>1: data = data.repeat(lax, axis=iax)
        else: data = as_strided(data, shape=new_shape, strides=tuple(axes_strides)) # special numpy trick...
        assert data.ndim == len(new_axes) and data.shape == new_shape 
        #assert isEqual(np.ma.mean(data),np.ma.mean(self.data_array)), '{}, {}'.format(np.ma.mean(data),np.ma.mean(self.data_array))
      if asVar:
        # construct new variable with new axes and reshaped/repeated data
        var = self._createVar(axes=axes, data=data, atts=self.atts, linplace=linplace)
        assert var.ndim == len(new_axes) and var.shape == new_shape
        assert not ldata or var[:].strides == data.strides, data.strides 
      else:
        var = data # just return reshaped/repeated data
    else:
      if lcheckAxis: raise AxisError, req_axes 
      else: var = None # don't do anything
    # return new variable
    return var
  
  @UnaryCheckAndCreateVar
  def standardize(self, axis=None, name=None, linplace=False, lcheckVar=True, lcheckAxis=True,
                  lstandardize=True, ldetrend=False, ltrend=False, lsmooth=False, lresidual=False, 
                  degree=1, rcond=None, w=None, window_len=11, window='hanning'):
    ''' Standardize Variable, i.e. subtract mean and divide by standard deviation '''
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Standardization does not work with string Variables!"
      else: return None
    if linplace: data = self.data_array # this is just a link
    else: data = self.data_array.copy()
    # apply detrending and smoothing
    if axis is None:
      iaxis = None
      if lsmooth and data.size <= window_len: window_len = data.size-1 # shrink window, if data too short
      # apply over flattened array
      data = detrend(data, ax=None, lcopy=False, 
                     ldetrend=ldetrend, ltrend=ltrend, degree=degree, rcond=rcond, w=w, 
                     lsmooth=lsmooth, lresidual=lresidual, window_len=window_len, window=window)
    else:
      iaxis = self.axisIndex(axis, lcheck=lcheckAxis)
      # get axis coordinates
      ax = self.axes[iaxis].coord if ldetrend else None
      if ldetrend or lsmooth:
        if lsmooth and ax.size <= window_len: window_len = ax.size-1 # shrink window, if data too short
        # prepare parallel function call
        aaa_fct = functools.partial(detrend, ax=ax, lcopy=False, 
                                    ldetrend=ldetrend, ltrend=ltrend, degree=degree, rcond=rcond, w=w, 
                                    lsmooth=lsmooth, lresidual=lresidual, window_len=window_len, window=window)
        # apply along selected axis
        data = apply_along_axis(aaa_fct, iaxis, data, laax=True)
        # reassigning array (apply_along_axis creates a new results array)
    # standardize (subtract mean and divide by standard deviation)
    if lstandardize:
      data -= np.nanmean(data, axis=iaxis, keepdims=True)
      data /= np.nanstd(data, axis=iaxis, keepdims=True)
    # N.B.: we standardize last, so as to ensure that the standard deviation is indeed unity
    # meta data
    if name is None: name = '{:s}_std' # allow use of original name in function
    name = name.format(self.name) # only used for new variables
    units = '' if lstandardize else self.units # no units after normalization
    # return results to decorator/wrapper
    return data, name, units
    
  def fitDist(self, axis='time', nsamples=None, dist=None, lflatten=False, name=None, atts=None, 
              var_dists=None, lsuffix=False, asVar=True, lcheckVar=True, lcheckAxis=True, 
              lcrossval=False, ncv=0.2, lignoreParams=False, **kwargs):
    ''' Generate a DistVar of type 'dist' from a Variable 'var'; use dimension 'axis' as sample axis; 
        if no 'dist' is specified, some heuristics are used to determine a suitable distribution; 
        dictionary lookup is also supported through 'var_dists'. '''
    from geodata.stats import asDistVar
    from datasets.common import shp_params, stn_params
    if self.name in shp_params or self.name in stn_params:
      dvar = self
    else:
      dvar = asDistVar(self, axis=axis, dist=dist, lflatten=lflatten, name=name, atts=atts, 
                       var_dists=var_dists, lcrossval=lcrossval, ncv=ncv, nsamples=nsamples,
                       lsuffix=lsuffix, asVar=asVar, lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, **kwargs)
    # return new variable
    return dvar
  
  @UnaryCheckAndCreateVar
  def _apply_ufunc(self, ufunc=None, linplace=False, lwarn=True, **kwargs):
    ''' Apply ufunc to data and return new Variable instance '''
    uname = ufunc.__name__
    lunits = len(self.units) > 0
    # a word of warning...
    if lwarn and lunits: 
      warn("Applying ufunc '{:s}' to '{:s}' data with units '{:s}' may require normalization.".format(uname,self.name,self.units))
    # compute ufunc
    if linplace: data = self.data_array
    else: data = self.data_array.copy()
    data = ufunc(data, **kwargs)
    # figure out meta data
    name = '{:s}({:s})'.format(uname,self.name)
    units = '{:s}({:s})'.format(uname,self.units) if lunits else ''
    # return results to decorator/wrapper
    return data, name, units    
  
  def __getattr__(self, attr):
    ''' If the call is a numpy ufunc method that is not implemented by Variable, call the ufunc method
        on data using _apply_ufunc; if the call is a scipy.stats distribution or test that is supported
        by the geodata.stats module, generate a DistVar object from the variable or apply the test 
        selected to the variable. '''
    # N.B.: this method is only called as a fallback, if no class/instance attribute exists,
    #       i.e. Variable methods and attributes will always have precedent 
    # check if a ufunc of that name exists
    if hasattr(np,attr):
      ufunc = getattr(np,attr)
      if isinstance(ufunc,np.ufunc):
        # call function on data, using _apply_ufunc
        return functools.partial(self._apply_ufunc, ufunc=ufunc)
      else:
        raise AttributeError, "The numpy function '{:s}' is not supported by class '{:s}'! (only ufunc's are supported)".format(attr,self.__class__.__name__)
    elif hasattr(nf,attr):
      ufunc = getattr(nf,attr)
      if isinstance(ufunc,np.ufunc):
      # call function on data, using _apply_ufunc
        return functools.partial(self._apply_ufunc, ufunc=ufunc)      
      else:
        raise AttributeError, "The numpy function '{:s}' is not supported by class '{:s}'! (only ufunc's are supported)".format(attr,self.__class__.__name__)
    elif hasattr(ss,attr): # either a distribution or a statistical test
      dist = getattr(ss, attr)
      if isinstance(dist,ss.rv_discrete):
        raise NotImplementedError, "Discrete distributions are not yet supported."
        # N.B.: DistVar's have not been tested with descrete distributions, but it might just work
      elif isinstance(dist,(ss.rv_continuous)) or attr.lower() == 'kde':
        from geodata.stats import asDistVar
        # call function on variable (self)
        return functools.partial(asDistVar, self, dist=attr)
      elif callable(dist):
        if attr in ('anderson','kstest','normaltest','shapiro'): # a function; assuming a statistical test
          # one of the implemented statistical tests
          return functools.partial(self.apply_stat_test, test=attr)
        else:
          raise NotImplementedError, "The statistical function '{:s}' is not supported by class '{:s}'!".format(attr,self.__class__.__name__)
      else:
        raise AttributeError, "The scipy.stats attribute '{:s}' is not supported by class '{:s}'!".format(attr,self.__class__.__name__)
    else: 
      raise AttributeError, "Attribute/method '{:s}' not found in class '{:s}'!".format(attr,self.__class__.__name__)
    
  @BinaryCheckAndCreateVar(sameUnits=True, linplace=True)    
  def __iadd__(self, a, othername=None, otherunits=None, linplace=True):
    ''' Add a number or an array to the existing data. '''
    self.data_array += a    
    assert linplace, 'This is strictly an in-place operation!'      
    return self.data_array, self.name, self.units # return array as result

  @BinaryCheckAndCreateVar(sameUnits=True, linplace=True)
  def __isub__(self, a, othername=None, otherunits=None, linplace=True):
    ''' Subtract a number or an array from the existing data. '''      
    self.data_array -= a
    assert linplace, 'This is strictly an in-place operation!'      
    return self.data_array, self.name, self.units # return array as result

  @BinaryCheckAndCreateVar(sameUnits=False, linplace=True)
  def __imul__(self, a, othername=None, otherunits=None, linplace=True):
    ''' Multiply the existing data with a number or an array. '''      
    self.data_array *= a
    assert linplace, 'This is strictly an in-place operation!'      
    return self.data_array, self.name, self.units # return array as result

  @BinaryCheckAndCreateVar(sameUnits=False, linplace=True)
  def __idiv__(self, a, othername=None, otherunits=None, linplace=True):
    ''' Divide the existing data by a number or an array. '''      
    self.data_array /= a
    assert linplace, 'This is strictly an in-place operation!'      
    return self.data_array, self.name, self.units # return array as result

  @BinaryCheckAndCreateVar(sameUnits=False, linplace=True)
  def __ipow__(self, a, othername=None, otherunits=None, linplace=True):
    ''' Take a power of the existing data with a number or an array. '''      
    self.data_array **= a
    # generally we don't change the name for in-place operations
    units = '{:s}^{:s}'.format(self.units,str(a)) # units change
    assert linplace, 'This is strictly an in-place operation!'      
    return self.data_array, self.name, units # return array as result

  @BinaryCheckAndCreateVar(sameUnits=True, linplace=False)
  def __add__(self, a, othername=None, otherunits=None, linplace=False):
    ''' Add two variables and return a new variable. '''
    data = self.data_array + a
    name = '{:s} + {:s}'.format(self.name,othername)
    units = self.units
    assert not linplace, 'This operation is strictly not in-place!'
    return data, name, units

  @BinaryCheckAndCreateVar(sameUnits=True, linplace=False)
  def __sub__(self, a, othername=None, otherunits=None, linplace=False):
    ''' Subtract two variables and return a new variable. '''
    data = self.data_array - a
    name = '{:s} - {:s}'.format(self.name,othername)
    units = self.units
    assert not linplace, 'This operation is strictly not in-place!'
    return data, name, units
  
  @BinaryCheckAndCreateVar(sameUnits=False, linplace=False)
  def __mul__(self, a, othername=None, otherunits=None, linplace=False):
    ''' Multiply two variables and return a new variable. '''
    data = self.data_array * a
    name = '{:s} x {:s}'.format(self.name,othername)
    if otherunits is None: units = self.units 
    else: units = '{:s} {:s}'.format(self.units,otherunits)
    assert not linplace, 'This operation is strictly not in-place!'
    return data, name, units

  @BinaryCheckAndCreateVar(sameUnits=False, linplace=False)
  def __div__(self, a, othername=None, otherunits=None, linplace=False):
    ''' Divide two variables and return a new variable. '''
    data = self.data_array / a
    name = '{:s} / {:s}'.format(self.name,othername)
    if otherunits is None: units = self.units
    else:
      if self.units == otherunits: units = ''
      else: units = '{:s} / ({:s})'.format(self.units,otherunits)
    assert not linplace, 'This operation is strictly not in-place!'
    return data, name, units

  @BinaryCheckAndCreateVar(sameUnits=False, linplace=False)
  def __pow__(self, a, othername=None, otherunits=None, linplace=False):
    ''' Raise a variable to a power and return a new variable. '''
    data = self.data_array ** a
    astr = '{}'.format(str(a))
    name = '{:s}^{:s}'.format(self.name,astr)
    units = '{:s}^{:s}'.format(self.units,astr)
    assert not linplace, 'This operation is strictly not in-place!'
    return data, name, units
     

class Axis(Variable):
  '''
    A special class of 1-dimensional variables for coordinate variables.
     
    It is essential that this class does not overload any class methods of Variable, 
    so that new Axis sub-classes can be derived from new Variable sub-classes via 
    multiple inheritance from the Variable sub-class and this class. 
  '''
  
  def __init__(self, length=0, coord=None, axes=None, **varargs):
    ''' Initialize a coordinate axis with appropriate values.
        
        Attributes: 
          coord = @property # managed access to the coordinate vector
          len = @property # the current length of the dimension (integer value)
          ascending = bool # if coordinates incease or decrease
    '''
    # initialize dimensions
    if axes is None: axes = (self,)
    elif not isinstance(axes,(list,tuple)) and len(axes) == 1:
      raise ArgumentError
#     axes = (self,)
    # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple
    if coord is not None: 
      data = self._transformCoord(coord)
      if length > 0:
        if data.size != length: raise AxisError, "Specified length and coordinate vector are incompatible!"
      else: length = data.size
    else: data = None
    self.__dict__['_len'] = length
    # initialize as a subclass of Variable, depending on the multiple inheritance chain    
    super(Axis, self).__init__(axes=axes, data=data, **varargs)
    # add coordinate vector
    if data is not None: 
      self.coord = data
      assert self.data == True
    # determine direction of ascend
    if self.coord is not None:
      if all(np.diff(self.coord) > 0): self.ascending = True
      elif all(np.diff(self.coord) < 0): self.ascending = False
#       else: self.ascending = None
      else: raise AxisError, "Coordinates must be strictly monotonically increasing or decreasing."

  def _transformCoord(self, data):
    ''' a coordinate vector will be converted, based on input conventions '''
    if isinstance(data,tuple) and ( 0 < len(data) < 4):
      data = np.linspace(*data)
    elif isinstance(data,np.ndarray) and data.ndim == 1:
      data = data
    elif isinstance(data,(list,tuple)):
      data = np.asarray(data)
    elif isinstance(data,slice):
      if not self.data: raise DataError, 'Cannot slice coordinate when coordinate vector is empty!'
      if ( data.stop is None and data.start is None) or len(self.data_array) > data.stop-data.start: 
        data = self.data_array.__getitem__(data)
      else: data = self.data_array
      # N.B.: this is necessary to prevent shrinking of the coordinate vector after successive slicing
    else: #data = data
      raise TypeError, 'Data type not supported for coordinate values.'
    return data

  @property
  def coord(self):
    ''' An alias for the data_array variable that is specific to coordiante vectors. '''
    return self.data_array
  @coord.setter
  def coord(self, data):
    ''' Update the coordinate vector of an axis based on certain conventions. '''
    # resolve coordinates
    if data is None:
      # this means the coordinate vector/data is going to be deleted 
      self.unload()
    else:
      # transform input based on conventions
      data = self._transformCoord(data)
      # load data
      self._len = data.size    
      self.load(data=data, mask=None)

  @property
  def len(self):
    ''' The length of the axis; if a coordinate vector is present, it is the length of that vector. '''
    if self.data and self._len != self.coord.size: 
      raise AxisError, "Length of axis '{:s}' and coordinate size do not match!".format(self.name)
    return self._len    
  @len.setter
  def len(self, length):
    ''' Update the length, or check for conflict if a coordinate vector is present. (Default length is 0)'''
    if self.data and length != self.coord.size:
      raise AxisError, 'Axis instance \'{:s}\' already has a coordinate vector of length {:d} ({:d} given)'.format(self.name,len(self),length)        
    self._len = length
  
  def __len__(self):
    ''' Return length of dimension. '''
    return self.len 
    
  def copy(self, deepcopy=False, **newargs): # with multiple inheritance, this method will override all others
    ''' A method to copy the Axis with just a link to the data. '''
    if deepcopy: 
      ax = self.deepcopy(**newargs)
    else:
      args = dict(name=self.name, units=self.units, length=self.len, coord=None, mask=None,  
                  dtype=self.dtype, atts=self.atts.copy(), plot=self.plot.copy())
      #if self.data: args['data'] = self.data_array
      if self.data: args['coord'] = self.coord # btw. don't pass axes to and Axis constructor!
      if 'axes' in newargs:
        axes = newargs.pop('axes') # this will cause a crash when creating an Axis instance
        if not ( isinstance(axes, (list,tuple)) and len(axes) == 1 ): raise ArgumentError
        axis = axes[0] # template axis 
        if not isinstance(axis, Axis): raise TypeError
        # take values from passed axis
        if axis.data: newargs['coord'] = axis.coord # btw. don't pass axes to and Axis constructor!       
      if 'coord' in newargs and newargs['coord'] is not None: 
        newargs['length'] = 0 # avoid conflict (0 is like None here)
      if 'atts' in newargs and newargs['atts'] is not None:
        atts = newargs.pop('atts')
        args['atts'].update(atts) # update original atts dict
        for key in ('name','units'): # also update name/units explicitly, or they get overridden
          if key in atts and atts[key] is not None: args[key] = atts[key]
      args.update(newargs) # apply custom arguments (also arguments related to subclasses)
      ax = Axis(**args) # create a new basic Axis instance
    # N.B.: this function will be called, in a way, recursively, and collect all necessary arguments along the way
    return ax
  
  def deepcopy(self, **newargs): # in almost all cases, this methods will be inherited from here
    ''' A method to copy the Axis and also copy data array. '''
    ax = self.copy(**newargs) # copy meta data
    # replace link with new copy of data array
    if self.data: ax.load(data=self.getArray(unmask=False,copy=True))
    # N.B.: using load() and getArray() should automatically take care of any special needs 
    return ax

  def getIndex(self, value, mode='closest', outOfBounds=None):
    ''' Return the coordinate index that is closest to the value or suitable for index ranges (left/right). '''
    if not self.data: raise DataError
    if outOfBounds is None:
      if mode.lower() == 'closest': outOfBounds = False
#       elif mode.lower() in ('left','right'): outOfBounds = False # return lowest/highest index if out of bounds
      else: outOfBounds = True # return None if value out of bounds
    # check coordinate order
    coord = self.coord
    if self.ascending: 
      if outOfBounds and ( value < coord[0] or value > coord[-1] ): return None # check bounds
    else: 
      if outOfBounds and ( value > coord[0] or value < coord[-1] ): return None # check bounds before reversing
      coord = coord[::-1] # reverse order
      # also swap left and right
      if mode.lower() == 'left': mode = 'right'
      elif mode.lower() == 'right': mode = 'left'    
    # behavior depends on mode
    if mode.lower() == 'left':
      # returns value suitable for beginning of range (inclusive)
#       return coord.searchsorted(value, side='left') 
      idx = max(coord.searchsorted(value, side='right')-1,0)
    elif mode.lower() == 'right':    
      # returns value suitable for end of range (inclusive)
      idx = coord.searchsorted(value, side='right')
      if idx > 0 and coord[idx-1] == value: idx -= 1 # special case...
    elif mode.lower() == 'closest':      
      # search for closest index
      idx = coord.searchsorted(value, side='right') # returns value 
      # refine search
      if idx <= 0: 
        idx = 0
      elif idx >= self.len: 
        idx = self.len-1
      else:
        dl = value - coord[idx-1]
        dr = coord[idx] - value
        assert dl >= 0 and dr >= 0
        if dr < dl: 
          idx = idx
        else: 
          idx = idx-1 # can't be 0 at this point 
    else: 
      raise ValueError, "Mode '{:s}' unknown.".format(mode)      
    # return
    if not self.ascending: idx = self.len - idx -1 # flip again
    return idx 

  def getIndices(self, coords):
    ''' Method to find occurences of coords and return their index values. '''
    if not self.data: self.load()
    if not isinstance(coords,(tuple,list,np.ndarray)): raise TypeError                
    # now use numpy to extract matching index
    idxs = np.where(np.in1d(self.coord, coords)) # find exact matches and indices of matches
    idxs = idxs[0] # where() always returns a tuple, we just want an array
    if len(idxs) == 0:
      # possible problems with floats
      if np.issubdtype(self.dtype, np.inexact): 
        warn("The current implementation may fail for floats due to machine precision differences (non-exact match).") 
    # return index of coordinate value
    return idxs
                  

class Dataset(object):
  '''
    A container class for variable and axes objects, as well as some meta information. This class also 
    implements collective operations on all variables in the dataset.
  '''
  
  def __init__(self, name=None, title=None, varlist=None, atts=None):
    ''' 
      Create a dataset from a list of variables. The basic dataset class has no capability to create variables.
      
      Basic Attributes:
        name = @property # short name (links to name in atts)
        title = @property # descriptive name (links to name in atts)
        variables = dict() # dictionary holding Variable instances
        axes = dict() # dictionary holding Axis instances (inferred from Variables)
        atts = AttrDict() # dictionary containing global attributes / meta data
    '''
    # create instance attributes
    self.__dict__['variables'] = dict()
    self.__dict__['axes'] = dict()
    # set properties in atts
    if atts is None: atts = dict()
    if name is not None: atts['name'] = name
    elif 'name' not in atts: atts['name'] = 'N/A'
    if title is not None: atts['title'] = title
    elif 'title' not in atts: atts['title'] = atts['name']
    # load global attributes, if given
    if atts: self.__dict__['atts'] = AttrDict(**atts)
    else: self.__dict__['atts'] = AttrDict()
    # load variables (automatically adds axes linked to variables)
    if varlist is None: varlist = []
    # add variables (ApplyTestOverList-decorator takes care of iteration)
    assert self.addVariable(varlist, copy=False, lall=True)
#     print '\n'
#     varnames = [var.name for var in varlist]
#     varnames.sort()
#     for var in varnames:
#       print var
#     check = True
#     for var in varlist:
#       print var.name
#       check = check and self.addVariable(var, copy=False) # don't make copies of new variables!
#     assert check
      
  @property
  def name(self):
    ''' A short name, stored in the atts dictionary. '''
    return self.atts['name']  
  @name.setter
  def name(self, name):
    self.atts['name'] = name
  
  @property
  def title(self):
    ''' A long, descriptive name, stored in the atts dictionary. '''
    return self.atts['title']  
  @title.setter
  def title(self, title):
    self.atts['title'] = title
  
  @ApplyTestOverList
  def addAxis(self, ax, copy=False, loverwrite=False):
    ''' Method to add an Axis to the Dataset. If the Axis is already present, check that it is the same. '''
    if not isinstance(ax,Axis): raise TypeError
    if not self.hasAxis(ax.name): # add new axis, if it does not already exist        
      if ax.name in self.__dict__: 
        raise AttributeError, "Cannot add Axis '{:s}' to Dataset, because an attribute of the same name already exits!".format(ax.name)
      # add variable to dictionary
      if copy: self.axes[ax.name] = ax.copy()
      else: self.axes[ax.name] = ax
      self.__dict__[ax.name] = self.axes[ax.name] # create shortcut
    else: # make sure the axes are consistent between variable (i.e. same name, same axis)
      if loverwrite:
        self.replaceAxis(ax)
      elif not ax is self.axes[ax.name]:        
        if len(ax) != len(self.axes[ax.name]): 
          raise AxisError, "Error: Axis '{:s}' from Variable and Dataset are different!".format(ax.name)
        if ax.data and self.axes[ax.name].data:
          if not isEqual(ax.coord,self.axes[ax.name].coord): raise DataError
    # set dataset attribute
    self.axes[ax.name].dataset = self
    # double-check
    return self.axes.has_key(ax.name)       
    
  @ApplyTestOverList
  def addVariable(self, var, copy=False, deepcopy=False, loverwrite=False, lautoTrim=False, **kwargs):
    ''' Method to add a Variable to the Dataset. If the variable is already present, abort. '''
    if not isinstance(var,Variable): raise TypeError
    if var.name in self.__dict__: 
      if loverwrite and self.hasVariable(var.name): self.replaceVariable(var)
      else: raise AttributeError, "Cannot add Variable '{:s}' to Dataset, because an attribute of the same name already exits!".format(var.name)      
    else:       
      # optionally, slice variable to conform to axes
      if lautoTrim:
        trimaxes = dict()
        for ax in var.axes: 
          if self.hasAxis(ax.name):
            dsax = self.axes[ax.name]
            if len(ax) > len(dsax) : trimaxes[ax.name] = (dsax[0],dsax[-1])
            elif len(ax) < len(dsax): 
              raise AxisError, "Can only trim Variable axes, not extend: {:s}".format(ax)
        # slice variable
        var = var(linplace=False, lidx=False, lrng=True, **trimaxes)
      # add new axes, or check, if already present; if present, replace, if different
      for ax in var.axes: 
        if not self.hasAxis(ax.name):
          self.addAxis(ax, copy=copy) # add new axis          
        elif ax is not self.axes[ax.name]:
          var.replaceAxis(ax, self.axes[ax.name]) # or use old one of the same name
        # N.B.: replacing the axes in the variable is to ensure consistent axes within the dataset 
      # finally, if everything is OK, add variable
      if copy: self.variables[var.name] = var.copy(deepcopy=deepcopy)
      else: self.variables[var.name] = var
      self.__dict__[var.name] = self.variables[var.name] # create shortcut
    # set dataset attribute
    self.variables[var.name].dataset = self
    # double-check
    return self.variables.has_key(var.name) 
    
  @ApplyTestOverList
  def removeAxis(self, ax, force=False):
      ''' Method to remove an Axis from the Dataset, provided it is no longer needed. '''
      if isinstance(ax,basestring): ax = self.axes[ax] # only work with Axis objects
      assert isinstance(ax,Axis), "Argument 'ax' has to be an Axis instance or a string representing the name of an axis." 
      if ax.name in self.axes: # remove axis, if it does exist
        # make sure no variable still needs axis
        if force or not any([var.hasAxis(ax) for var in self.variables.itervalues()]):
          # delete axis from dataset   
          del self.axes[ax.name]
          del self.__dict__[ax.name]
          # this just removes the references, not the object; we still rely on garbage collection 
          # to remove the object, if it is no longer needed (which is not necessarily the case!) 
        # don't delete, if still needed
      # double-check (return True, if axis is not present, False, if it is)
      return not self.axes.has_key(ax.name)
    
  @ApplyTestOverList
  def removeVariable(self, var):
    ''' Method to remove a Variable from the Dataset. '''
    if isinstance(var,Variable): var = var.name
    if not isinstance(var,basestring): raise TypeError, "Argument 'var' has to be a Variable instance or a string representing the name of a variable."
    if var in self.variables: # add new variable if it does not already exist
      # delete variable from dataset   
      del self.variables[var]
      del self.__dict__[var]
    # double-check (return True, if variable is not present, False, if it is)
    return not self.variables.has_key(var)
  
  def replaceAxis(self, oldaxis, newaxis=None, **kwargs):    
    ''' Replace an existing axis with a different one with similar general properties. '''
    if newaxis is None: 
      newaxis = oldaxis; oldaxis = newaxis.name # i.e. replace old axis with the same name'
    # check axis
    if isinstance(oldaxis,Axis): oldname = oldaxis.name # just go by name
    else: oldname = oldaxis
    oldaxis = self.axes[oldname]
    if not self.hasAxis(oldaxis): raise AxisError
    if len(oldaxis) != len(newaxis): raise AxisError # length has to be the same!    
#     if oldaxis.data != newaxis.data: raise DataError # make sure data status is the same
    # remove old axis and add new to dataset
    self.removeAxis(oldaxis, force=True)
    self.addAxis(newaxis, copy=False)
    newaxis = self.axes[newaxis.name] # update reference
    # loop over variables with this axis    
    for var in self.variables.values():
      if var.hasAxis(oldname): 
        var.replaceAxis(oldname,newaxis) 
    # return verification
    return self.hasAxis(newaxis)    

  def replaceVariable(self, oldvar, newvar=None, **kwargs):
    ''' Replace an existing Variable with a different one and transfer NetCDF reference and axes. '''
    if newvar is None: 
      newvar = oldvar; oldvar = newvar.name # i.e. replace old var with the same name
    # check var
    if not self.hasVariable(oldvar): raise VariableError
    if isinstance(oldvar,Variable): oldname = oldvar.name # just go by name
    else: oldname = oldvar
    oldvar = self.variables[oldname]
    #if oldvar.shape != newvar.shape: raise AxisError # shape has to be the same!
    # N.B.: the shape of a variable in a NetCDF file can't change!
    # remove old variable from dataset...
    self.removeVariable(oldvar)
    # ... and add new axis to dataset
    self.addVariable(newvar, copy=False)    
    # return status of variable
    return self.hasVariable(newvar)  

  def squeeze(self):
    ''' Remove singleton axes from all variables; return axes that were entirely removed. '''
    axes = set()
    # squeeze variables
    for var in self.variable.values():
      var.squeeze() # get axes that were removed
      axes.add(var.axes) # collect axes that are still needed
    # remove axes that are no longer needed
    retour = []
    for ax in self.axes:
      if ax not in axes: 
        self.removeAxis(ax)
        retour.append(ax)        
    # return axes that were removed
    return retour
  
  @ApplyTestOverList
  def hasVariable(self, var, strict=True):
    ''' Method to check, if a Variable is present in the Dataset. '''
    if isinstance(var,basestring):
      return self.variables.has_key(var) # look up by name
    elif isinstance(var,Variable):
      if self.variables.has_key(var.name):
        if strict: return self.variables[var.name] is var # verify identity
        return True # name found and identity verified 
      else: return False # not found
    else: # invalid input
      raise DatasetError, "Need a Variable instance or name to check for a Variable in the Dataset!"
    
  @ApplyTestOverList
  def getVariable(self, varname, check=True):
    ''' Method to return a Variable by name '''
    if not isinstance(varname,basestring): raise TypeError
    if self.hasVariable(varname):
      return self.variables[varname]
    else:
      if check: raise VariableError, "Variable '{:s}' not found!".format(varname)
      else: return None
  
  @ApplyTestOverList
  def hasAxis(self, ax, strict=True):
    ''' Method to check, if an Axis is present in the Dataset; if strict=False, only names are compared. '''
    if isinstance(ax,basestring):
      return self.axes.has_key(ax) # look up by name
    elif isinstance(ax,Axis):
      if self.axes.has_key(ax.name):
        if strict: return self.axes[ax.name] is ax # verify identity 
        else: return True
      else: return False # not found
    else: # invalid input
      raise DatasetError, "Need a Axis instance or name to check for an Axis in the Dataset!"
  
  @ApplyTestOverList
  def getAxis(self, axname, check=True):
    ''' Method to return an Axis by name '''
    if not isinstance(axname,basestring): raise TypeError
    if self.hasAxis(axname):
      return self.axes[axname]
    else:
      if check: raise AxisError, "Axis '{:s}' not found!".format(axname)
      else: return None
      
  def slicing(self, lidx=None, lrng=None, lminmax=False, lsqueeze=True, lcopy=False, years=None, lfirst=False, 
               listAxis=None, lrmOther=False, lcpOther=False, **axes):
    ''' This method implements access to slices via coordinate values and returns a Dataset object; the 
        method relies on the Variable method for actual slicing but preserves the dataset integrity.
        Default behavior for different argument types: 
          - index by coordinate value, not array index, except if argument is a Slice object
          - interprete tuples of length 2 or 3 as ranges
          - treat lists and arrays as coordinate lists (can specify new list axis)
          - for backwards compatibility, None values are accepted and indicate the entire range 
        Type-based defaults are ignored if appropriate keyword arguments are specified. '''
    # process variables
    slicevars = {}; othervars = {} # variables that will get sliced and others that are unaffected
    sliceaxes = {}; otheraxes = {}
    singlevaratts = self.atts.copy() # variables that collapse to scalars are added as attributes (with the value)
    # parse axes for pseudo-axes
    # N.B.: we do this once for all variables, because this operation can be somewhat slow
    axes = axes.copy() # might be changed...
    lpseudo = False; lnonpseudo = False
    for key,val in axes.iteritems():
      if val is not None and self.hasVariable(key):
        del axes[key] # remove pseudo axis
        var = self.getVariable(key)
        if var.ndim == 1: # possibly valid pseudo-axis!
          coord = var.findValues(val, lidx=lidx, lfirst=lfirst, lminmax=lminmax, lflatten=False)          
        else: raise AxisError, "Pseudo-axis can only have one axis!"
        axes[var.axes[0].name] = coord # create new entry with actual axis
        # N.B.: not that this automatically squeezes the pseudo-axis, since it is just a values...
    # pseudo-axes work with ranges, but return indices, and we can't mix ranges with indices!
    if lrng and lpseudo: 
      lrng = False
      if lnonpseudo: 
        raise ArgumentError, "'lrng' was reset to False, but other variables are not yet processed."          
              
    # loop over variables
    for var in self.variables.itervalues():
      if ( all(ax.name in axes for ax in var.axes) and 
           all(not isinstance(axes[ax.name],(slice,tuple,list,np.ndarray)) for ax in var.axes) ):
        # just extract value and add as attribute
        if var.name in singlevaratts: 
          raise NotImplementedError, "Name collision between attribute and singleton variable '{:s}'".format(var.name)
        if not var.data: var.load()
        attval = var(lidx=lidx, lrng=lrng, asVar=False, lcheck=False, lsqueeze=True, 
                     lcopy=False, years=years, listAxis=listAxis, **axes)    
        # convert to Python scalar (of sorts)
        if isinstance(attval,np.ndarray):
          if np.issubdtype(attval.dtype, np.str): attval = str(attval).rstrip()    
          elif np.issubdtype(attval.dtype, np.integer): attval = int(attval)
          elif np.issubdtype(attval.dtype, np.float): attval = float(attval)
          else: raise TypeError, attval
        singlevaratts[var.name] = attval
      else:
        # properly slice variable
        newvar = var(lidx=lidx, lrng=lrng, asVar=True, lcheck=False, lsqueeze=lsqueeze, 
                     lcopy=lcopy, years=years, listAxis=listAxis, **axes)
        # save variable
        if var.ndim == newvar.ndim and var.shape == newvar.shape: 
          othervars[newvar.name] = newvar         
        else: slicevars[newvar.name] = newvar
        # save axes
        for ax in newvar.axes:
          if ax.name in otheraxes and len(ax) == len(self.axes[ax.name]): otheraxes[ax.name] = ax
          else: sliceaxes[ax.name] = ax       
    # figure out what to copy
    axes = otheraxes.copy(); axes.update(sliceaxes) # sliced axes overwrite old axes
    variables = slicevars.copy()
    varlist = variables.keys()
    if not lrmOther:
      varlist += othervars.keys()
      if lcpOther: variables.update(othervars)
    # copy dataset
    return self.copy(axes=axes, variables=variables, varlist=varlist, atts=singlevaratts,
                     varargs=None, axesdeep=True, varsdeep=False)
  # a convenient alias for slicing
  def __call__(self, **kwargs):
    ''' Defers the call to the 'slicing' method. '''
    return self.slicing(**kwargs)


  def copy(self, axes=None, variables=None, varlist=None, varargs=None, axesdeep=True, varsdeep=False, 
           **kwargs): # this methods will have to be overloaded, if class-specific behavior is desired
    ''' A method to copy the Axes and Variables in a Dataset with just a link to the data arrays. '''
    # copy axes (shallow copy)    
    newaxes = {name:ax.copy(deepcopy=axesdeep) for name,ax in self.axes.iteritems()}
    if axes is not None: # allow override
      if not isinstance(axes,dict): raise TypeError # check input
      for name,ax in axes.iteritems():
        if not isinstance(ax,Axis): raise TypeError
      newaxes.update(axes) # axes overwrites the ones copied from old dataset
    # check attributes
    if varargs is None: varargs=dict() 
    if not isinstance(varargs,dict): raise TypeError
    # copy variables
    if variables is None: variables = dict()
    if not isinstance(variables,dict): raise TypeError
    if varlist is None: varlist = self.variables.keys()
    if not isinstance(varlist,(list,tuple)): raise TypeError
    newvars = []; oldvars = []
    for varname in varlist:
      # select variable
      if varname in variables: 
        # skip variables that are set to None
        var = variables[varname]
        if var is not None: newvars.append(var)
        # N.B.: don't make a copy, or we loose any NetCDF reference!
      else: 
        var = self.variables[varname]
        # change axes and attributes
        axes = tuple([newaxes.get(ax.name,ax) for ax in var.axes])
        if varname in varargs: # check input again
          if isinstance(varargs[varname],dict): args = varargs[varname]  
          else: raise TypeError
        else: args = dict()
        # copy variables
        oldvars.append(var.copy(axes=axes, deepcopy=varsdeep, **args))
    # determine attributes
    tmp = kwargs.pop('atts',None) 
    if isinstance(tmp,dict): atts = tmp
    else: atts = self.atts.copy()
    # make new dataset with new variables
    dataset = Dataset(varlist=newvars, atts=atts, **kwargs)
    for var in oldvars: 
      try: dataset.addVariable(var, copy=False, loverwrite=False, lrename=True)
      except AxisError: pass # ignore incompatible axes
    # N.B.: this function will be called, in a way, recursively, and collect all necessary arguments along the way
    return dataset
  
  def deepcopy(self, **kwargs): # ideally this does not have to be overloaded, but can just call copy()
    ''' A method to generate an entirely independent Dataset instance (deepcopy of all data, variables, and axes). '''
    kwargs['axesdeep'] = True; kwargs['varsdeep'] = True     
    dataset = self.copy(**kwargs) 
    return dataset
  
  def __str__(self):
    ''' Built-in method; we just overwrite to call 'prettyPrint()'. '''
    return self.prettyPrint(short=False) # print is a reserved word  

  def prettyPrint(self, short=False):
    ''' Print a string representation of the Dataset. '''
    if short:
      title = self.title or self.name
      variables = '{:3d} Vars'.format(len(self.variables)) # number of variables
      axes = '{:2d} Axes'.format(len(self.axes)) # number of axes
      klass = self.__class__.__name__ 
      string = '{:<20s} {:s}, {:s} ({:s})'.format(title,variables,axes,klass)
    else:
      string = '{0:s} \'{1:s}\'   {2:s}\n'.format(self.__class__.__name__,self.title or self.name, self.__class__)
      # print variables (sorted alphabetically)
      string += 'Variables:\n'
      varlisting = ['  {0:s}\n'.format(var.prettyPrint(short=True)) for var in self.variables.values()]
      varlisting.sort() # sorting variables strings in-place 
      for varstr in varlisting: string += varstr 
      # print axes (sorted alphabetically) 
      string += 'Axes:\n'
      axlisting = ['  {0:s}\n'.format(ax.prettyPrint(short=True)) for ax in self.axes.values()]
      axlisting.sort()
      for axstr in axlisting: string += axstr
      string += 'Attributes: {0:s}'.format(str(self.atts))
    return string
    
  def __getitem__(self, varname):
    ''' Yet another way to access variables by name... conforming to the container protocol. '''
    if not isinstance(varname, basestring): raise TypeError, varname
    if not self.hasVariable(varname): raise KeyError, varname
    return self.variables[varname]
  
  def __setitem__(self, varname, var):
    ''' Yet another way to add a variable, this time by name... conforming to the container protocol. '''
    if not isinstance(var, Variable) or not isinstance(varname, basestring): raise TypeError, varname
    var.name = varname # change name to varname
    if self.hasVariable(varname, strict=False):
      check = self.replaceVariable(var)
    else:
      check = self.addVariable(var) # add variable
    if not check: raise KeyError, varname # raise error if variable is not present
    
  def __delitem__(self, varname):
    ''' A way to delete variables by name... conforming to the container protocol. '''
    if not isinstance(varname, basestring): raise TypeError, varname
    if not self.hasVariable(varname): raise KeyError, varname
    check = self.removeVariable(varname)
    if not check: raise KeyError, varname # raise error if variable has not disappeared
  
  def __iter__(self):
    ''' Return an iterator over all variables... conforming to the container protocol. '''
    return self.variables.itervalues() # just the iterator from the variables dictionary
    
  def __contains__(self, var):
    ''' Check if the Dataset instance has a particular Variable... conforming to the container protocol. '''
    if not (isinstance(var, Variable) or isinstance(var, basestring)): raise TypeError, var
    return self.hasVariable(var) # variable only
  
  def __len__(self):
    ''' Get the number of Variables in the Dataset. '''
    return len(self.variables)
  
  def __add__(self, other):
    ''' create new dataset from two existing ones '''
    name = self.name if self.name == other.name else "{:s} & {:s}".format(self.name,other.name)
    title = self.title if self.title == other.title else "{:s} & {:s}".format(self.title,other.title)
    varlist = self.variables.values() + other.variables.values()
    atts = other.atts.copy(); atts.update(self.atts)
    return Dataset(name=name,title=title,varlist=varlist,atts=atts)
    
  def __iadd__(self, var):
    ''' Add a Variable to an existing dataset. '''      
    assert self.addVariable(var), "A problem occurred adding Variable '{:s}' to Dataset.".format(var.name)    
    return self # return self as result

  def __isub__(self, var):
    ''' Remove a Variable to an existing dataset. '''      
    assert self.removeVariable(var), "A proble occurred removing Variable '{:s}' from Dataset.".format(var.name)
    return self # return self as result
  
  def load(self, **kwargs):
    ''' Issue load() command to all variable; pass on any keyword arguments. '''
    for var in self.variables.itervalues():
      if not var.data: var.load(**kwargs) # there is only one argument to the base method
    return self
      
  def unload(self, **kwargs):
    ''' Unload all data arrays currently loaded in memory. '''
    for var in self.variables.itervalues():
      var.unload(**kwargs)
      
  def mask(self, mask=None, maskSelf=False, varlist=None, skiplist=None, invert=False, merge=True, **kwargs):
    ''' Apply 'mask' to all variables and add the mask, if it is a variable. '''
    # figure out variable list
    if isinstance(mask,basestring): mask = self.variables[mask]
    if skiplist is None: skiplist = []
    if not isinstance(skiplist,(list,tuple)): raise TypeError
    if varlist is None: varlist = self.variables.keys()  
    if not isinstance(varlist,(list,tuple)): raise TypeError
    varlist = [var for var in varlist if var not in skiplist and var in self.variables]
    # iterate over all variables (not axes!) 
    for varname in varlist:
      var = self.variables[varname]
      if var.ndim >= mask.ndim:
        if not var.data: raise DataError, "Need to have data to process mask!"
        # need to have data and also the right number of dimensions 
        lOK = False
        if isinstance(mask,np.ndarray):
          if mask.shape == var.shape[var.ndim-mask.ndim:]: lOK = True
        elif isinstance(mask,Variable):
          if mask is var or mask.name == var.name: lOK = maskSelf # default: False
          elif all([var.hasAxis(ax) for ax in mask.axes]): lOK = True
        if lOK: var.mask(mask=mask, invert=invert, merge=merge, **kwargs) # if everything is OK, apply mask
    # add mask do dataset
    if isinstance(mask,Variable) and not self.hasVariable(mask): self.addVariable(mask)
    
  def unmask(self, fillValue=None, **kwargs):
    ''' Unmask all Variables in the Dataset. '''
    for var in self.variables.itervalues():
      var.load(fillValue=fillValue, **kwargs)
      
  def _apply_to_all(self, fctsdict, asVar=True, dsatts=None, copyother=True, deepcopy=False, 
                    lcheckVar=False, lcheckAxis=False, lkeepName=True, **kwargs):
    ''' Apply functions from fctsdict to variables in dataset and return a new dataset. '''
    # separate axes from kwargs
    axes = {axname:ax for axname,ax in kwargs.iteritems() if self.hasAxis(axname)}
    for axname in axes.iterkeys(): del kwargs[axname] 
    # loop over variables
    newvars = dict() 
    for varname,var in self.variables.iteritems():
      if varname in fctsdict and fctsdict[varname] is not None:
        # figure out, which axes apply
        tmpargs = kwargs.copy()
        if axes: tmpargs.update({key:value for key,value in axes.iteritems() if var.hasAxis(key)})
        newvars[varname] = fctsdict[varname](asVar=asVar, lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, **tmpargs)
        if lkeepName and asVar and newvars[varname] is not None: # can also be None or np.ndarray
          newvars[varname].name = varname # don't change names, since we are creating a new dataset         
      elif copyother and asVar:
        newvars[varname] = var.copy(deepcopy=deepcopy)
    # assemble new dataset
    if copyother: # varname:None means the variable is omitted
      newvars = {varname:var for varname,var in newvars.iteritems() if var is not None}
    if asVar: newset = self.copy(variables=newvars, atts=dsatts) # use copy method of dataset
    else: newset = newvars # just return resulting dictionary
    # return new dataset
    return newset

  def __getattr__(self, attr):
    ''' if the call is a Variable method that is not provided by Dataset, call the Variable method
        on all Variables using _apply_to_all '''
    # N.B.: this method is only called as a fallback, if no class/instance attribute exists,
    #       i.e. Dataset methods and attributes will always have precedent 
    # check if Variables have this attribute
    if any([hasattr(var,attr) for var in self.variables.itervalues()]):
      # get all attributes into a dict, using None if not present
      attrdict = {varname:getattr(var,attr) for varname,var in self.variables.iteritems() 
                  if hasattr(var, attr)}
      # determine if this is a Variable method
      if any([callable(fct) for fct in attrdict.itervalues()]):
        # call function on all variables, using _apply_to_all
        return functools.partial(self._apply_to_all, attrdict) 
      else:
        # treat as simple attributes and return dict with values
        return attrdict
    else: raise AttributeError, attr # raise previous exception
      

def concatVars(variables, axis=None, coordlim=None, idxlim=None, asVar=True, offset=None, 
               name=None, units=None, axatts=None, varatts=None, lcheckAxis=True, lensembleAxis=None):
  ''' A function to concatenate Variables from different sources along a given axis;
      this is useful to generate a continuous time series from an ensemble. '''
  if lensembleAxis and axis is None: axis = 'ensemble'
  elif isinstance(axis,(Axis,basestring)) and not any([var.hasAxis(axis) for var in variables]):
    if lensembleAxis is None: lensembleAxis = True
    elif lensembleAxis is False: raise ArgumentError        
  if all([not var.hasAxis(axis, strict=False) for var in variables]): lnew = True
  elif all([var.hasAxis(axis, strict=False) for var in variables]): lnew = False
  else: raise AxisError, axis 
  if isinstance(axis,Axis):
    axis_obj = axis 
    axis = axis.name
  else: axis_obj = None
  if not isinstance(axis,basestring): raise TypeError
  if not all([isinstance(var,Variable) for var in variables]): raise TypeError
  var0 = variables[0] # shortcut
  if not all([var.shape == var0.shape  for var in variables]): 
    raise AxisError, "All Variables need to have the same shape for concatenation!"
  if not var0.data: var.load()
  # get some axis info
  if lnew:
    tax = 0 # add ensemble axis as first axis (assuming C order)
  else:
    axt = var0.getAxis(axis)
    tax = var0.axisIndex(axis)  
    if offset is None: offset = axt.coord[0] 
    if not isNumber(offset): raise TypeError
    delta = axt.coord[1] - axt.coord[0]
    for var in variables:
      axdiff = np.diff(axt.coord)
      if lcheckAxis and not (axdiff.min()+100*floateps >= delta >=  axdiff.max()-100*floateps): 
        raise AxisError, "Concatenation axis has to be evenly spaced!"
  # slicing options
  lcoordlim = False; lidxlim = False
  if coordlim is not None and idxlim is not None: 
    raise ValueError, "Can only define either'coordlim' or 'idxlim', not both!"
  elif coordlim is not None:
    lcoordlim = True
    if isinstance(coordlim,(tuple,list)): coordlim = {axis:coordlim}
    else: raise TypeError    
  elif idxlim is not None:
    lidxlim = True
    if isinstance(idxlim,slice): idxslc = idxlim
    elif isinstance(idxlim,(tuple,list)): idxslc= slice(*idxlim)
    else: raise TypeError, idxlim
  # check dimensions
  shapes = []; tes = []
  for var in variables:
    shp = list(var.shape)
    if lcoordlim: tes.append(len(axt(**coordlim)))
    elif lidxlim:       
      tmpidx = idxslc.indices(len(axt))
      idxlen = 1 + (tmpidx[1] -1 - tmpidx[0]) / tmpidx[2]
      assert idxlen == len(axt[idxslc])
      tes.append(idxlen)
    else: tes.append(shp[tax])
    del shp[tax]
    shapes.append(shp)
  if not all([s == shp for s in shapes]): 
    raise AxisError, shapes
  if lnew:
    tlen = len(tes) # number of ensemble members
    assert tlen == len(variables)
    newshape = (tlen,)+var0.shape
  else:
    tlen = 0 # length of concatenation axis
    for te in tes: tlen += te
    newshape = list(var0.shape)
    newshape[tax] = tlen
    newshape = tuple(newshape)
  # load data
  data = []
  for var in variables:
    if not var.data: var.load()
    if lcoordlim: 
      array = var(**coordlim)     
    elif lidxlim:
      array = var.getArray().take(xrange(*idxslc.indices(len(axt))), axis=tax)
    else: 
      array = var.getArray()
    if lnew: array = array.reshape((1,)+array.shape) # add singleton dimension to concatenate over
    data.append(array)
  # concatenate
  data = np.concatenate(data, axis=tax)
  assert data.shape[tax] == tlen
  assert data.shape == newshape
  # cast as variable
  if asVar:      
    # create new concatenation axis
    if axis_obj is not None:
      if not isinstance(axis_obj, Axis): raise TypeError, axis_obj
      if len(axis_obj) != tlen: raise AxisError, axis_obj
      axes = (axis_obj,) + var0.axes  
    elif lnew: 
      # new ensemble axis (just enumerate)
      tmpatts = dict(name=axis, units='#')
      if axatts is not None: tmpatts.update(axatts)      
      axes = (Axis(coord=np.arange(tlen), atts=tmpatts),) + var0.axes 
    else:
      # extended old axis  
      tmpatts = axt.atts.copy()
      tmpatts['name'] = axt.name; tmpatts['units'] = axt.units
      # N.B.: offset is a parameter to simply shift the time axis origin
      coord = np.arange(offset,tlen*delta+offset,delta) 
      if axatts is not None: tmpatts.update(axatts)      
      axes = list(var0.axes); axes[tax] = Axis(coord=coord, atts=tmpatts)
    # create new variable
    vatts = var0.atts.copy()
    vatts['name'] = name or var0.name; vatts['units'] = units or var0.units
    if varatts is not None: vatts.update(varatts)
    return Variable(data=data, axes=axes, atts=vatts)
    # or return data
  else: return data
  
  
def concatDatasets(datasets, name=None, axis=None, coordlim=None, idxlim=None, offset=None, axatts=None,
                   title=None, lensembleAxis=None, lignoreConst=True, time_axes=None, check_vars=None,
                   lcpOther=True, lcpAny=False, ldeepcopy=True, lcheckVars=True, lcheckAxis=True):
  ''' A function to concatenate Datasets from different sources along a given axis; this
      function essentially applies concatVars to every Variable and creates a new dataset. 
      When concatenating station or shape arrays, use check_vars with an array of unique ID's
      to make sure they are all in the same order (since only the first axis and ID variable
      (pseudo-axis) will be retained. '''
  if lensembleAxis and axis is None: axis = 'ensemble'
  if lignoreConst and time_axes is None: time_axes = ('time','year')
  elif isinstance(axis,(Axis,basestring)) and not any([ds.hasAxis(axis) for ds in datasets]):
    if lensembleAxis is None: lensembleAxis = True
    elif lensembleAxis is False: raise ArgumentError        
  if isinstance(axis,(Axis,basestring)): axislist = (axis,)
  else: axislist = axis
  nax = len(axislist)
  if isinstance(coordlim,(tuple,list)):
    if isinstance(coordlim[0],(tuple,list)): climlist = coordlim
    elif len(coordlim) == 2: climlist = (coordlim,)*nax
    else: raise TypeError
  elif coordlim is None: climlist = (None,)*nax
  else: raise TypeError
  if isinstance(idxlim,(tuple,list)):
    if isinstance(idxlim[0],(tuple,list)) and len(idxlim[0]): ilimlist = idxlim
    elif isinstance(idxlim[0],slice): ilimlist = idxlim
    elif len(idxlim) == 2: ilimlist = (idxlim,)*nax
  elif idxlim is None: ilimlist = (None,)*nax
  else: raise TypeError
  if not isinstance(offset,(tuple,list)): oslist = (offset,)*nax
  else: oslist = offset
  variables = dict() # variables for new dataset
  axes = dict() # new axes
  # verify that check_var's are the same
  if check_vars:
    if not isinstance(check_vars, (list,tuple)): check_vars = (check_vars,)
    for check_var  in check_vars:
      ds = datasets[0] # master dataset
      if not ds.hasVariable(check_var): 
        raise VariableError, "Check Var '{:s}' not found in Dataset '{:s}'.".format(check_var, ds.name)      
      varunits = ds.variables[check_var].units
      varvalues = ds.variables[check_var][:]
      for ds in datasets[1:]:
        # check preliminaries
        if not ds.hasVariable(check_var):
          raise VariableError, "Check Var '{:s}' not found in Dataset '{:s}'.".format(check_var, ds.name)
        if ds.variables[check_var].units != varunits: 
          raise VariableError, "Check Var '{:s}' units don't match in Dataset '{:s}' ('{:s}' != '{:s}').".format(check_var, ds.name, ds.variables[check_var].units, varunits)
        # check values
        if not np.all( ds.variables[check_var][:] == varvalues ): 
          raise VariableError, "Check Var '{:s}' values don't match in Dataset '{:s}'.".format(check_var, ds.name)
  # loop over axes
  for axis,coordlim,idxlim,offset in zip(axislist,climlist,ilimlist,oslist):
    if isinstance(axis,Axis): axis = axis.name
    if not isinstance(axis,basestring): raise TypeError
    if not lensembleAxis and not any([ds.hasAxis(axis) for ds in datasets]): pass # for convenience
    elif not lensembleAxis and not all([ds.hasAxis(axis) for ds in datasets]): 
      raise DatasetError, "Some datasets don't have axis '{:s}' - aborting!".format(axis)
    else: # go ahead
      # figure out complete set of variables
      varlist = []
      for dataset in datasets:
        varlist += [varname for varname in dataset.variables.iterkeys() if not varname in varlist]
      # process variables
      for varname in varlist:
        lall = all([ds.hasVariable(varname) for ds in datasets]) # check if all have this variable
        # find first occurence
        c = 0; dataset = datasets[c] # try first    
        while not dataset.hasVariable(varname):
          c += 1; dataset = datasets[c] # try next
        varobj = dataset.variables[varname]
        # N.B.: one has to have it, otherwise it would not be in the list    
        # decide what to do        
        lconst = lignoreConst and not any([varobj.hasAxis(ax) for ax in time_axes])
        if varobj.hasAxis(axis) or ( lensembleAxis and not lconst ) : # concatenate
          if lall: 
            variables[varname] = concatVars([ds.variables[varname] for ds in datasets], axis=axis, asVar=True,
                                            coordlim=coordlim, idxlim=idxlim, offset=offset, axatts=axatts,
                                            lcheckAxis=lcheckAxis, lensembleAxis=lensembleAxis)
          else:
            if lcheckVars:       
              raise DatasetError, "Variable '{:s}' is not present in all Datasets!".format(varname)
            else: variables[varname] = None # removes the variable from the target dataset
            # N.B.: copying variables that have the concat axis but are not being concatenated,
            #       will lead to axis conflicts
        elif lcpOther and varname not in variables: # either add as is, or skip... 
          if lcpAny or lall: # add if all are present or flag to ignore is set
            variables[varname] = dataset.variables[varname].copy(deepcopy=ldeepcopy)  
      # find new concatenated axis
      c = 0; catax = None
      while catax is None:
        catax = variables.values()[c].getAxis(axis, lcheck=False); c += 1 # return None if not present
      axes[axis] = catax # add new concatenation axis
    # copy first dataset and replace concatenation axis and variables
  return datasets[0].copy(axes=axes, name=name, title=title, variables=variables, varlist=None, 
                          varargs=None, axesdeep=True, varsdeep=False)


class Ensemble(object):
  '''
    A container class that holds several datasets ("members" of the ensemble),
    furthermore, the Ensemble class provides functionality to execute Dataset
    class methods collectively for all members, and return the results in a tuple.
  '''
  members   = None    # list of members of the ensemble
  basetype  = Dataset # base class of the ensemble members
  idkey     = 'name'  # property of members used for unique identification
  ens_name  = ''      # name of the ensemble
  ens_title = ''      # printable title used for the ensemble
  
  def __init__(self, *members, **kwargs):
    ''' Initialize an ensemble from a list of members (the list arguments);
    keyword arguments are added as attributes (key = attribute name, 
    value = attribute value).
    
  Attributes:
    members      = list/tuple of members of the ensemble
    basetype     = class of the ensemble members
    idkey        = property of members used for unique identification
    ens_name     = name of the ensemble (string)
    ens_title    = printable title used for the ensemble (string)
    '''
    # add members
    self.members = list(members)
    # add certain properties
    self.ens_name = kwargs.pop('name','')
    self.ens_title = kwargs.pop('title','')
    # no need to be too restrictive
    if 'basetype' in kwargs:
      self.basetype = kwargs.pop('basetype') # don't want to add that later! 
      if isinstance(self.basetype,basestring):
        self.basetype = globals()[self.basetype]
    elif isinstance(members[0],Dataset): self.basetype = Dataset
    elif isinstance(members[0],Variable): self.basetype = Variable
    else: self.basetype = members[0].__class__
    if len(members) > 0 and not all(isinstance(member,self.basetype) for member in members):
      raise TypeError, "Not all members conform to selected type '{}'".format(self.basetype.__name__)
    self.idkey = kwargs.get('idkey','name')
    # add keywords as attributes
    for key,value in kwargs.iteritems():
      self.__dict__[key] = value
    # add short-cuts and keys
    self.idkeys = []
    for member in self.members:
      memid = getattr(member, self.idkey)
      self.idkeys.append(memid)
      if not isinstance(memid, basestring): raise TypeError, "Member ID key '{:s}' should be a string-type, but received '{:s}'.".format(str(memid),memid.__class__)
      if memid in self.__dict__:
        raise AttributeError, "Cannot overwrite existing attribute '{:s}'.".format(memid)
      self.__dict__[memid] = member
      
  def _recastList(self, fs):
    ''' internal helper method to decide if a list or Ensemble should be returned '''
    if all(f is None for f in fs): return # suppress list of None's
    elif all([not callable(f) and not isinstance(f, (Variable,Dataset)) for f in fs]): return fs  
    elif all([isinstance(f, (Variable,Dataset)) for f in fs]):
      # N.B.: technically, Variable instances are callable, but that's not what we want here...
      if all([isinstance(f, Axis) for f in fs]): 
        return fs
      # N.B.: axes are often shared, so we can't have an ensemble
      elif all([isinstance(f, Variable) for f in fs]): 
        # check for unique keys
        if len(fs) == len(set([f.name for f in fs if f.name is not None])): 
          return Ensemble(*fs, idkey='name') # basetype=Variable,
        elif len(fs) == len(set([f.dataset.name for f in fs if f.dataset is not None])): 
#           for f in fs: f.dataset_name = f.dataset.name 
          return Ensemble(*fs, idkey='dataset_name') # basetype=Variable, 
        else:
          #raise KeyError, "No unique keys found for Ensemble members (Variables)"
          # just re-use current keys
          for f,member in zip(fs,self.members):
            if self.idkey == 'dataset_name':
              if f.dataset_name is None: 
                f.dataset_name = member.dataset_name
              elif not f.dataset_name == member.dataset_name: 
                raise DatasetError, f.dataset_name
            elif not hasattr(f, self.idkey): 
              setattr(f, self.idkey, getattr(member,self.idkey))
            else: raise DatasetError, self.idkey
#             f.__dict__[self.idkey] = getattr(member,self.idkey)
          return Ensemble(*fs, idkey=self.idkey) # axes from several variables can be the same objects
      elif all([isinstance(f, Dataset) for f in fs]): 
        # check for unique keys
        if len(fs) == len(set([f.name for f in fs if f.name is not None])): 
          return Ensemble(*fs, idkey='name') # basetype=Variable,
        else:
#           raise KeyError, "No unique keys found for Ensemble members (Datasets)"
          # just re-use current keys
          for f,member in zip(fs,self.members): 
            f.name = getattr(member,self.idkey)
          return Ensemble(*fs, idkey=self.idkey) # axes from several variables can be the same objects
      else:
        raise TypeError, "Resulting Ensemble members have inconsisent type."
  
  def __call__(self, *args, **kwargs):
    ''' Overloading the call method allows coordinate slicing on Ensembles. '''
    return self.__getattr__('slicing')(*args, **kwargs)
  
  def __getattr__(self, attr):
    ''' This is where all the magic happens: defer calls to methods etc. to the 
        ensemble members and return a list of values. '''
    # intercept some list methods
    #print dir(self.members), attr, attr in dir(self.members)
    # determine whether we need a wrapper
    fs = [getattr(member,attr) for member in self.members]
    if all([callable(f) and not isinstance(f, (Variable,Dataset)) for f in fs]):
      # for callable objects, return a wrapper that can read argument lists      
      def wrapper( *args, **kwargs):
        # either distribute args or give the same to everyone
        lens = len(self)
        if all([len(arg)==lens and isinstance(arg,(tuple,list,Ensemble)) for arg in args]):
          argslists = [list() for i in xrange(lens)] 
          for arg in args: # swap nested list order ("transpose") 
            for i in xrange(len(argslists)): 
              argslists[i].append(arg[i])
          res = [f(*args, **kwargs) for args,f in zip(argslists,fs)]
        else:
          res = [f(*args, **kwargs) for f in fs]
        return self._recastList(res) # code is reused, hens pulled out
      # return function wrapper
      return wrapper
    else:
      # regular object
      return self._recastList(fs)
    
  def __str__(self):
    ''' Built-in method; we just overwrite to call 'prettyPrint()'. '''
    return self.prettyPrint(short=False) # print is a reserved word  

  def prettyPrint(self, short=False):
    ''' Print a string representation of the Ensemble. '''
    if short:      
      string = '{0:s} {1:s}'.format(self.__class__.__name__,self.ens_name)
      string += ', {:2d} Members ({:s})'.format(len(self.members),self.basetype.__name__)
    else:
      string = '{0:s}   {1:s}\n'.format(self.__class__.__name__,str(self.__class__))
      string += 'Name: {0:s},  '.format(self.ens_name)
      string += 'Title: {0:s}\n'.format(self.ens_title)
      string += 'Members:\n'
      for member in self.members: string += ' {0:s}\n'.format(member.prettyPrint(short=True))
      string += 'Basetype: {0:s},  '.format(self.basetype.__name__)
      string += 'ID Key: {0:s}'.format(self.idkey)
    return string

  def hasMember(self, member):
    ''' check if member is part of the ensemble; also perform consistency checks '''
    if isinstance(member, self.basetype):
      # basetype instance
      memid = getattr(member,self.idkey)
      if member in self.members:
        assert memid in self.__dict__
        assert member == self.__dict__[memid]
        return True
      else: 
        assert memid not in self.__dict__
        return False
    elif isinstance(member, basestring):
      # assume it is the idkey
      if member in self.__dict__:
        assert self.__dict__[member] in self.members
        assert getattr(self.__dict__[member],self.idkey) == member
        return True
      else: 
        assert member not in [getattr(m,self.idkey) for m in self.members]
        return False
    else: raise TypeError, "Argument has to be of '{:s}' of 'basestring' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)       
      
  def addMember(self, member):
    ''' add a new member to the ensemble '''
    if not isinstance(member, self.basetype): 
      raise TypeError, "Ensemble members have to be of '{:s}' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)       
    self.members.append(member)
    self.__dict__[getattr(member,self.idkey)] = member
    return self.hasMember(member)
  
  def removeMember(self, member):
    ''' remove a member from the ensemble '''
    if not isinstance(member, (self.basetype,basestring)): 
      raise TypeError, "Argument has to be of '{:s}' of 'basestring' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)
    if self.hasMember(member):
      if isinstance(member, basestring): 
        memid = member
        member = self.__dict__[memid]
      else: memid = getattr(member,self.idkey)
      assert isinstance(member,self.basetype)
      # remove from dict 
      del self.__dict__[memid]
      # remove from list
      del self.members[self.members.index(member)]
    # return check
    return not self.hasMember(member)
  
  def __mul__(self, n):
    ''' how to combine with other objects '''
    if isInt(n):
      return self.members*n
    else:
      raise TypeError

  def __add__(self, other):
    ''' how to combine with other objects '''
    if isinstance(other, Ensemble):
      for member in other: self.addMember(member)
      return self
    elif isinstance(other, list):
      return self.members + other
    elif isinstance(other, tuple):
      return tuple(self.members) * other
    else:
      raise TypeError

  def __radd__(self, other):
    ''' how to combine with other objects '''
    if isinstance(other, Ensemble):
      for member in other: self.addMember(member)
      return self
    elif isinstance(other, list):
      return other + self.members
    elif isinstance(other, tuple):
      return other + tuple(self.members)
    else:
      raise TypeError

  def __getitem__(self, item):
    ''' Yet another way to access members by name... conforming to the container protocol. 
        If argument is not a member, it is called with __getattr__.'''
    if isinstance(item, basestring): 
      if self.hasMember(item):
        # access members like dictionary
        return self.__dict__[item] # members were added as attributes
      else:
        try:
          # dispatch to member attributes 
          atts = [getattr(member,item) for member in self.members]
          if any([callable(att) and not isinstance(att, (Variable,Dataset)) for att in atts]): raise AttributeError
          return self._recastList(atts)
          # N.B.: this is useful to load different Variables from Datasets by name, 
          #       without having to use getattr()
        except AttributeError:
          if self.basetype is Dataset: raise DatasetError, item
          elif self.basetype is Variable: raise VariableError, item
          else: raise AttributeError, item
        #return self.__getattr__(item) # call like an attribute
    elif isinstance(item, (int,np.integer,slice)):
      # access members like list/tuple 
      return self.members[item]
    elif isinstance(item, (list,tuple,np.ndarray)):
      # index/label list like ndarray
      members = [self[i] for i in item] # select members
      kwargs = dict(basetype=self.basetype, idkey=self.idkey, name=self.ens_name, title=self.ens_title)
      return Ensemble(*members,**kwargs) # return new ensemble with selected members
    else: raise TypeError
  
  def __setitem__(self, name, member):
    ''' Yet another way to add a member, this time by name... conforming to the container protocol. '''
    idkey = getattr(member,self.idkey)
    if idkey != name: raise KeyError, "The member ID '{:s}' is not consistent with the supplied key '{:s}'".format(idkey,name)
    return self.addMember(member) # add member
    
  def __delitem__(self, member):
    ''' A way to delete members by name... conforming to the container protocol. '''
    if not isinstance(member, basestring): raise TypeError
    if not self.hasMember(member): raise KeyError
    return self.removeMember(member)
  
  def __iter__(self):
    ''' Return an iterator over all members... conforming to the container protocol. '''
    return self.members.__iter__() # just the iterator from the member list
    
  def __contains__(self, member):
    ''' Check if the Ensemble instance has a particular member Dataset... conforming to the container protocol. '''
    return self.hasMember(member)

  def __len__(self):
    ''' return number of ensemble members '''
    return len(self.members)
  
  def __iadd__(self, member):
    ''' Add a Dataset to an existing Ensemble. '''
    if isinstance(member, self.basetype):
      assert self.addMember(member), "A problem occurred adding Dataset '{:s}' to Ensemble.".format(member.name)    
    elif isinstance(member, Variable):
      assert all(self.addVariable(member)), "A problem occurred adding Variable '{:s}' to Ensemble Members.".format(member.name)    
    elif all([isinstance(m, Variable) for m in member]):
      assert all(self.addVariable(member)), "A problem occurred adding Variable '{:s}' to Ensemble Members.".format(member.name)    
    return self # return self as result

  def __isub__(self, member):
    ''' Remove a Dataset to an existing Ensemble. '''      
    if isinstance(member, basestring) and self.hasMember(member):
      assert self.removeMember(member), "A proble occurred removing Dataset '{:s}' from Ensemble.".format(member)    
    elif isinstance(member, self.basetype):
      assert self.removeMember(member), "A proble occurred removing Dataset '{:s}' from Ensemble.".format(member.name)
    elif isinstance(member, (basestring,Variable)):
      assert all(self.removeVariable(member)), "A problem occurred removing Variable '{:s}' from Ensemble Members.".format(member.name)    
    return self # return self as result

  
## run a test    
if __name__ == '__main__':

  pass