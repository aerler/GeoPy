'''
Created on 2013-08-19

Variable and Dataset classes for handling geographical datasets.

@author: Andre R. Erler, GPL v3
'''

# numpy imports
import numpy as np
import numpy.ma as ma # masked arrays
# my own imports
from atmdyn.properties import variablePlotatts # import plot properties from different file
from misc import checkIndex, isEqual, isFloat, AttrDict, joinDicts
from misc import VariableError, AxisError, DataError, DatasetError

import numbers
import functools

class UnaryCheck(object):
  ''' Decorator class that implements some sanity checks for unary arithmetic operations. '''
  def __init__(self, op):
    ''' Save original operation. '''
    self.op = op
  def __call__(self, orig, arg):
    ''' Perform sanity checks, then execute operation, and return result. '''
    if isinstance(arg,np.ndarray): 
      pass # allow broadcasting and type-casting
#       assert orig.shape == arg.shape, 'Arrays need to have the same shape!' 
#       assert orig.dtype == arg.dtype, 'Arrays need to have the same type!'
    else: assert isinstance(arg, numbers.Number), 'Can only operate with numerical types!'
    if not orig.data: orig.load()
    var = self.op(orig,arg)
    assert isinstance(var,Variable)
    return var # return function result
  def __get__(self, instance, klass):
    ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
    # N.B.: similar implementation to 'partial': need to return a callable that behaves like the instance method
    # def f(arg):
    #  return self.__call__(instance, arg)
    # return f    
    return functools.partial(self.__call__, instance) # but using 'partial' is simpler

def BinaryCheckAndCreateVar(sameUnits=True):
  ''' A decorator function that accepts arguments and returns a decorator class with fixed parameter values. '''
  class BinaryCheckAndCreateVar_Class(object):
    ''' A decorator to perform similarity checks before binary operations and create a new variable instance 
      afterwards; name and units are modified; only non-conflicting attributes are kept. '''
    def __init__(self, binOp):
      ''' Save original operation and parameters. '''
      self.binOp = binOp
      self.sameUnits = sameUnits # passed to constructor function
    # define method wrapper (this is now the actual decorator)
    def __call__(self, orig, other):
      ''' Perform sanity checks, then execute operation, and return result. '''
      assert isinstance(other,Variable), 'Can only add two \'Variable\' instances!' 
      if self.sameUnits: assert orig.units == other.units, 'Variable units have to be identical for addition!'
      assert orig.shape == other.shape, 'Variables need to have the same shape and compatible axes!'
      if not orig.data: orig.load()
      if not other.data: other.load()
      for lax,rax in zip(orig.axes,other.axes):
        assert (lax.coord == rax.coord).all(), 'Variables need to have identical coordinate arrays!'
      # call original method
      data, name, units = self.binOp(orig, other)
      # construct common dict of attributes
      atts = joinDicts(orig.atts, other.atts)
      atts['name'] = name; atts['units'] = units
      # assign axes (copy from orig)
      axes = [ax for ax in orig.axes]
      var = Variable(name=name, units=units, axes=axes, data=data, atts=atts)
      return var # return new variable instance
    def __get__(self, instance, klass):
      ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
      return functools.partial(self.__call__, instance)
  # return decorator class  
  return BinaryCheckAndCreateVar_Class


## Variable class and derivatives 

class Variable(object):
  ''' 
    The basic variable class; it mainly implements arithmetic operations and indexing/slicing.
  '''
  
  def __init__(self, name=None, units=None, axes=None, data=None, dtype='', mask=None, fillValue=None, atts=None, plot=None):
    ''' 
      Initialize variable and attributes.
      
      Basic Attributes:
        name = @property # short name, e.g. used in datasets (links to atts dictionary)
        units = @property # physical units (links to atts dictionary)
        data = False # logical indicating whether a data array is present/loaded 
        axes = None # a tuple of references to coordinate variables (also Variable instances)
        data_array = None # actual data array (None if not loaded)
        shape = None # length of dimensions, like an array
        ndim = None # number of dimensions
        dtype = '' # data type (string)
        
      Optional/Advanced Attributes:
        masked = False # whether or not the array in self.data is a masked array
        fillValue = None # value to fill in for masked values
        atts = None # dictionary with additional attributes
        plot = None # attributed used for displaying the data       
    '''
    # basic input check
    if data is None:
      ldata = False; shape = None
    else:
      assert isinstance(data,np.ndarray), 'The data argument must be a numpy array!'
      ldata = True; shape = data.shape; 
      if dtype:
        dtype = np.dtype(dtype) # make sure it is properly formatted.. 
        if dtype is not data.dtype: data = data.astype(dtype) # recast as new type        
#         raise TypeError, "Declared data type '%s' does not match the data type of the array (%s)."%(str(dtype),str(data.dtype))
      if axes is not None:
        assert len(axes) == data.ndim, 'Dimensions of data array and axes are note compatible!'
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
    if fillValue is not None: atts['missing_value'] = fillValue # slightly irregular treatment...
    self.__dict__['fillValue'] = fillValue
    self.__dict__['atts'] = AttrDict(**atts)
    if plot is None: # try to find sensible default values 
      if variablePlotatts.has_key(self.name): plot = variablePlotatts[self.name]
      else: plot = dict(plotname=self.name, plotunits=self.units, plottitle=self.name) 
    self.__dict__['plot'] = AttrDict(**plot)
    # set defaults - make all of them instance variables! (atts and plot are set below)
    self.__dict__['data_array'] = None
    self.__dict__['data'] = ldata
    self.__dict__['shape'] = shape
    self.__dict__['dtype'] = dtype
    self.__dict__['masked'] = False # handled in self.load() method    
    ## figure out axes
    if axes is not None:
      assert isinstance(axes, (list, tuple))
      if all([isinstance(ax,Axis) for ax in axes]):
        if ldata: 
          for ax,n in zip(axes,shape): ax.updateLength(n)
        if not ldata and all([len(ax) for ax in axes]): # length not zero
          self.__dict__['shape'] = tuple([len(ax) for ax in axes]) # get shape from axes
      elif all([isinstance(ax,basestring) for ax in axes]):
        if ldata: axes = [Axis(name=ax, length=n) for ax,n in zip(axes,shape)] # use shape from data
        else: axes = [Axis(name=ax) for ax in axes] # initialize without shape
    else: 
      raise VariableError, 'Cannot initialize %s instance \'%s\': no axes declared'%(self.var.__class__.__name__,self.name)
    self.__dict__['axes'] = tuple(axes) 
    self.__dict__['ndim'] = len(axes)  
    # create shortcuts to axes (using names as member attributes) 
    for ax in axes: self.__dict__[ax.name] = ax
    # assign data, if present (can initialize without data)
    if data is not None: 
      self.load(data, mask=mask, fillValue=fillValue) # member method defined below
      
  @property
  def name(self):
    ''' The name stored in the atts dictionary. '''
    return self.atts['name']  
  @name.setter
  def name(self, name):
    self.atts['name'] = name
  
  @property
  def units(self):
    ''' The units stored in the atts dictionary. '''
    return self.atts['units']  
  @units.setter
  def units(self, units):
    self.atts['units'] = units
  
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
      string = '{0:s} {1:s} [{2:s}]   {3:s}\n'.format(self.__class__.__name__,self.name,self.units,self.__class__)
      for ax in self.axes: string += '  {0:s}\n'.format(ax.prettyPrint(short=True))
      string += 'Attributes: {0:s}\n'.format(str(self.atts))
      string += 'Plot Attributes: {0:s}'.format(str(self.plot))
    return string
  
  def squeeze(self):
    ''' A method to remove singleton dimensions. '''
    # new axes tuple: only the ones longer than one element
    axes = []; retour = []
    for ax in self.axes:
      if len(ax) > 1: axes.append(ax)
      else: retour.append(ax)
    self.axes = tuple(axes)
    self.shape = tuple([len(ax) for ax in self.axes])
    assert self.ndim == len(axes) + len(retour)
    self.ndim = len(self.axes)    
    # squeeze data array, if necessary
    if self.data:
      self.data_array = self.data_array.squeeze()
      assert self.ndim == self.data_array.ndim
      assert self.shape == self.data_array.shape        
    # return squeezed dimensions
    return retour
  
  def copy(self, deepcopy=False, **newargs): # this methods will have to be overloaded, if class-specific behavior is desired
    ''' A method to copy the Variable with just a link to the data. '''
    if deepcopy:
      var = self.deepcopy( **newargs)
    else:
      args = dict(name=self.name, units=self.units, axes=self.axes, data=self.data_array, dtype=self.dtype,
                  mask=None, fillValue=self.fillValue, atts=self.atts.copy(), plot=self.plot.copy())
      args.update(newargs) # apply custom arguments (also arguments related to subclasses)
      var = Variable(**args) # create a new basic Variable instance
    # N.B.: this function will be called, in a way, recursively, and collect all necessary arguments along the way
    return var
  
  def deepcopy(self, **newargs): # in almost all cases, this methods will be inherited from here
    ''' A method to generate an entirely independent variable instance (copy meta data, data array, and axes). '''
    # copy axes (generating ordinary Axis instances with coordinate arrays)
    if 'axes' not in newargs: newargs['axes'] = tuple([ax.deepcopy() for ax in self.axes]) # allow override though
    # copy meta data
    var = self.copy(**newargs) # use instance copy() - this method can be overloaded!   
    # replace link with new copy of data array
    if self.data: var.load(data=self.getArray(unmask=False,copy=True))
    # N.B.: using load() and getArray() should automatically take care of any special needs 
    return var

  def hasAxis(self, axis):
    ''' Check if the variable instance has a particular axis. '''
    if isinstance(axis,basestring): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return True
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
    if not self.hasAxis(oldaxis): raise AxisError
    else: oldname = oldaxis
    oldaxis = self.axes[self.axisIndex(oldname)]
    if len(oldaxis) != len(newaxis): raise AxisError # length has to be the same!
    if oldaxis.data != newaxis.data: raise DataError # make sure data status is the same
    # replace old axis
    self.axes = tuple([ax if ax is not oldaxis else newaxis for ax in self.axes])
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

  def getAxis(self, axis):
    ''' Return a reference to the Axis object or one with the same name. '''
    return self.axes[self.axisIndex(axis)]
  
  def axisIndex(self, axis):
    ''' Return the index of a particular axis. (return None if not found) '''
    if isinstance(axis,basestring): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return i
    elif isinstance(axis,Variable): # by object ID
      for i in xrange(len(self.axes)):
        if self.axes[i] == axis: return i
    # if all fails
    return None
        
  def __getitem__(self, idx=None):
    ''' Method implementing access to the actual data. '''
    # default
    if idx is None: idx = slice(None,None,None) # first, last, step     
    # determine what to do
    if all(checkIndex(idx, floatOK=True)):
      # check if data is loaded      
      if not self.data:
        raise DataError, 'Variable instance \'%s\' has no associated data array or it is not loaded!'%(self.name) 
      # array indexing: return array slice
      if any(isFloat(idx)): raise NotImplementedError, \
        'Floating-point indexing is not implemented yet for \'%s\' class.'%(self.__class__.__name__)
      return self.data_array.__getitem__(idx) # valid array slicing
    else:    
      # if nothing applies, raise index error
      raise IndexError, 'Invalid index/key type for class \'%s\'!'%(self.__class__.__name__)
  
  def __call__(self, **kwargs):
    ''' This method implements access to slices via coordinate values (as opposed to indices). '''
    # loop over arguments and find indices of coordinate values
    slices = dict()
    for axname,coord in kwargs.iteritems():
      if self.hasAxis(axname):
        ax = self.getAxis(axname)
        if ax.data:
          if isinstance(coord,(list,tuple)):
            if len(coord) == 2:
              #l = max(ax.data_array.searchsorted(coord[0],side='right')-1,0) # choose such as to bracket coords
              #r = ax.data_array.searchsorted(coord[1],side='left') # same value or higher index
              slices[axname] = slice(ax.getIndex(coord[0]),ax.getIndex(coord[1]))
            elif len(coord) == 1: slices[axname] = ax.getIndex(coord[0])
            else: raise IndexError
          elif isinstance(coord,np.number):
            slices[axname] = ax.getIndex(coord)
          else: raise TypeError
    # assemble index tuple for axes
    idx = tuple([slices.get(ax.name,slice(None)) for ax in self.axes])
    print idx
    return self.__getitem__(idx=idx) # pass on to getitem

  def load(self, data=None, mask=None, fillValue=None):
    ''' Method to attach numpy data array to variable instance (also used in constructor). '''
    if data is None: raise DataError, 'A basic \'Variable\' instance requires external data to load!'
    if not isinstance(data,np.ndarray): raise TypeError, 'The data argument must be a numpy array!'
    # apply mask
    if mask: data = ma.array(data, mask=mask) 
    if isinstance(data, ma.MaskedArray): 
      self.__dict__['masked'] = True # set masked flag
    else: self.__dict__['masked'] = False
    if self.masked: # figure out fill value for masked array
      if fillValue is not None: # override variable preset 
        self.__dict__['fillValue'] = fillValue
        data.set_fill_value = fillValue
      elif self.fillValue is not None: # use variable preset
        data.set_fill_value = self.fillValue
      else: # use data default
        self.__dict__['fillValue'] = data.get_fill_value()
    # more meta data
    self.__dict__['data'] = True
    self.__dict__['dtype'] = data.dtype
    self.__dict__['shape'] = data.shape
    if len(self.shape) != self.ndim and (self.ndim != 0 or data.size != 1):
      raise DataError, 'Variable dimensions and data dimensions incompatible!'
    # N.B.: the second statement is necessary, so that scalars don't cause a crash
    # assign data to instance attribute array 
    self.__dict__['data_array'] = data
    # some more checks
    # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple; hence
    #       the coordinate vector has to be assigned before the dimensions size can be checked 
    if len(self.axes) == len(self.shape): # update length is all we can do without a coordinate vector
      for ax,n in zip(self.axes,self.shape): ax.updateLength(n) 
    else: # this should only happen with scalar variables!
      assert self.ndim == 0 and data.size == 1, 'Dimensions of data array and variable must be identical, except for scalars!'       
     
  def unload(self):
    ''' Method to unlink data array. '''
    self.__dict__['data_array'] = None # unlink data array
    self.__dict__['data'] = False # set data flag
    self.__dict__['fillValue'] = None
    # self.__dict__['shape'] = None # retain shape for later use
    
  def getArray(self, idx=None, axes=None, broadcast=False, unmask=False, fillValue=None, copy=True):
    ''' Copy the entire data array or a slice; option to unmask and to reorder/reshape to specified axes. '''
    # get data (idx=None will return the entire data array)
    if copy: datacopy = self.__getitem__(idx).copy() # use __getitem__ to get slice
    else: datacopy = self.__getitem__(idx) # just get a view
    # unmask    
    if unmask and self.masked:
      if fillValue is None: fillValue=self.fillValue
      datacopy = datacopy.filled(fill_value=fillValue) # I don't know if this generates a copy or not...
    elif not self.masked and isinstance(datacopy, ma.MaskedArray): 
      self.__dict__['masked'] = True # update masked flag
    # reorder and reshape to match axes (add missing dimensions as singleton dimensions)
    if axes is not None:
      if idx is not None: raise NotImplementedError
      for ax in self.axes:
        assert (ax in axes) or (ax.name in axes), "Can not broadcast Variable '%s' to dimension '%s' "%(self.name,ax.name)
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
    # return array
    return datacopy
    
  def mask(self, mask=None, maskedValue=None, fillValue=None, invert=False, merge=True):
    ''' A method to add a mask to an unmasked array, or extend or replace an existing mask. '''
    if mask is not None:
      assert isinstance(mask,np.ndarray) or isinstance(mask,Variable), 'Mask has to be a numpy array or a Variable instance!'
      # 'mask' can be a variable
      if isinstance(mask,Variable): mask = mask.getArray(unmask=True,axes=self.axes,broadcast=True)
      if not isinstance(mask,np.ndarray): raise TypeError, 'Mask has to be convertible to a numpy array!'      
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
        data = self.getArray(unmask=True) # get data without mask
      self.__dict__['data_array'] = ma.array(data, mask=mask)
    elif maskedValue is not None:
      if isinstance(self.dtype,(int,bool,np.integer,np.bool)): 
        self.__dict__['data_array'] = ma.masked_equal(self.data_array, maskedValue, copy=False)
      elif isinstance(self.dtype,(float,np.inexact)):
        self.__dict__['data_array'] = ma.masked_values(self.data_array, maskedValue, copy=False)
      if fillValue is not None: self.data_array.set_fill_value(fillValue)
    if isinstance(self.data_array,ma.MaskedArray):
      # change meta data
      self.__dict__['masked'] = True
      if fillValue:
        # external fill value has priority 
        self.data_array.set_fill_value(fillValue)
        self.__dict__['fillValue'] = fillValue
      elif self.fillValue:
        # use fill value we already have
        self.data_array.set_fill_value(self.fillValue)
      else:          
        self.__dict__['fillValue'] = self.data_array.get_fill_value() # probably just the default
    
  def unmask(self, fillValue=None):
    ''' A method to remove and existing mask and fill the gaps with fillValue. '''
    if self.masked:
      if fillValue is None: fillValue = self.fillValue # default
      self.__dict__['data_array'] = self.data_array.filled(fill_value=fillValue)
      # change meta data
      self.__dict__['masked'] = False
      self.__dict__['fillValue'] = None  
    
  def getMask(self, nomask=False):
    ''' Get the mask of a masked array or return a boolean array of False (no mask). '''
    if nomask: return ma.getmask(self.data_array)
    else: return ma.getmaskarray(self.data_array)    

  @UnaryCheck    
  def __iadd__(self, a):
    ''' Add a number or an array to the existing data. '''      
    self.data_array += a    
    return self # return self as result

  @UnaryCheck
  def __isub__(self, a):
    ''' Subtract a number or an array from the existing data. '''      
    self.data_array -= a
    return self # return self as result
  
  @UnaryCheck
  def __imul__(self, a):
    ''' Multiply the existing data with a number or an array. '''      
    self.data_array *= a
    return self # return self as result

  @UnaryCheck
  def __idiv__(self, a):
    ''' Divide the existing data by a number or an array. '''      
    self.data_array /= a
    return self # return self as result
  
  @BinaryCheckAndCreateVar(sameUnits=True)
  def __add__(self, other):
    ''' Add two variables and return a new variable. '''
    data = self.data_array + other.data_array
    name = '%s + %s'%(self.name,other.name)
    units = self.units
    return data, name, units

  @BinaryCheckAndCreateVar(sameUnits=True)
  def __sub__(self, other):
    ''' Subtract two variables and return a new variable. '''
    data = self.data_array - other.data_array
    name = '%s - %s'%(self.name,other.name)
    units = self.units
    return data, name, units
  
  @BinaryCheckAndCreateVar(sameUnits=False)
  def __mul__(self, other):
    ''' Multiply two variables and return a new variable. '''
    data = self.data_array * other.data_array
    name = '%s x %s'%(self.name,other.name)
    units = '%s %s'%(self.units,other.units)
    return data, name, units

  @BinaryCheckAndCreateVar(sameUnits=False)
  def __div__(self, other):
    ''' Divide two variables and return a new variable. '''
    data = self.data_array / other.data_array
    name = '%s / %s'%(self.name,other.name)
    units = '%s / (%s)'%(self.units,other.units)
    return data, name, units
     
#   def __getattr__(self, name):
#     ''' Return contents of atts or plot dictionaries as if they were attributes. '''
#     # N.B.: before this method is called, instance attributes are checked automatically
#     if self.__dict__.has_key(name): # check instance attributes first
#       return self.__dict__[name]
#     elif self.__dict__['atts'].has_key(name): # try atts second
#       return self.__dict__['atts'][name] 
#     elif self.__dict__['plot'].has_key(name): # then try plot
#       return self.__dict__['plot'][name]
#     else: # or throw attribute error
#       raise AttributeError, '\'%s\' object has no attribute \'%s\''%(self.__class__.__name__,name)
    
#   def __setattr__(self, name, value):
#     ''' Change the value of class existing class attributes, atts, or plot entries,
#       or store a new attribute in the 'atts' dictionary. '''
#     if self.__dict__.has_key(name): # class attributes come first
#       self.__dict__[name] = value # need to use __dict__ to prevent recursive function call
#     elif self.__dict__['atts'].has_key(name): # try atts second
#       self.__dict__['atts'][name] = value    
#     elif self.__dict__['plot'].has_key(name): # then try plot
#       self.__dict__['plot'][name] = value
#     else: # if the attribute does not exist yet, add it to atts or plot
#       if name[0:4] == 'plot':
#         self.plot[name] = value
#       else:
#         self.atts[name] = value


class Axis(Variable):
  '''
    A special class of 1-dimensional variables for coordinate variables.
     
    It is essential that this class does not overload any class methods of Variable, 
    so that new Axis sub-classes can be derived from new Variable sub-classes via 
    multiple inheritance from the Variable sub-class and this class. 
  '''
  
  def __init__(self, length=0, coord=None, **varargs):
    ''' Initialize a coordinate axis with appropriate values.
        
        Attributes: 
          coord = None # the coordinate vector (also accessible as data_array)
          len = 0 # the length of the dimension (integer value)
    '''
    # initialize dimensions
    axes = (self,)
    # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple
    self.__dict__['coord'] = None
    self.__dict__['len'] = length
    # initialize as a subclass of Variable, depending on the multiple inheritance chain    
    super(Axis, self).__init__(axes=axes, **varargs)
    # add coordinate vector
    if coord is not None: self.updateCoord(coord)
    elif length > 0: self.updateLength(length)
  
  def copy(self, deepcopy=False, **newargs): # with multiple inheritance, this method will override all others
    ''' A method to copy the Axis with just a link to the data. '''
    if deepcopy: 
      ax = self.deepcopy(**newargs)
    else:
      args = dict(name=self.name, units=self.units, length=self.len, data=None, coord=None, 
                  dtype=self.dtype, mask=None, fillValue=self.fillValue, atts=self.atts.copy(), plot=self.plot.copy())
      if self.data: args['data'] = self.data_array
      if self.data: args['coord'] = self.coord # btw. don't pass axes to and Axis constructor!
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
    
  def load(self, *args, **kwargs):
    ''' Load a coordinate vector into an axis and update related attributes. '''
    # load data
    super(Axis,self).load(*args, **kwargs) # call load of base variable (Variable subclass)
    # update attributes
    self.__dict__['coord'] = self.data_array
    self.__dict__['len'] = self.data_array.shape[0]
    
  def unload(self):
    ''' Remove the coordinate vector of an axis but keep length attribute. '''
    # load data
    super(Axis,self).unload() # call unload of base variable (Variable subclass)
    # update attributes
    self.__dict__['coord'] = None
#     self.__dict__['len'] = 0

  def getIndex(self, value):
    ''' Return the coordinate index that is closest the value. '''
    if not self.data: raise DataError
    # search for close index
    idx = self.coord.searchsorted(value)
    # refine search
    if idx <= 0: return 0
    elif idx >= self.len: return self.len-1
    else:
      dl = value - self.coord[idx-1]
      dr = self.coord[idx] - value
      if dr < dl: return idx
      else: return idx-1 # can't be 0 at this point 
      
    
  def updateCoord(self, coord=None, **varargs):
    ''' Update the coordinate vector of an axis based on certain conventions. '''
    # resolve coordinates
    if coord is None:
      # this means the coordinate vector/data is going to be deleted 
      self.unload()
    else:
      # a coordinate vector will be created and loaded, based on input conventions
      if isinstance(coord,tuple) and ( 0 < len(coord) < 4):
        data = np.linspace(*coord)
      elif isinstance(coord,np.ndarray) and coord.ndim == 1:
        data = coord
      elif isinstance(coord,(list,tuple)):
        data = np.asarray(coord)
      else: #data = coord
        raise TypeError, 'Data type not supported for coordinate values.'
      # load data
      self.load(data=data, mask=None, **varargs)
      

  def __len__(self):
    ''' Return length of dimension. '''
    return self.__dict__['len'] 
    
  def updateLength(self, length=0):
    ''' Update the length, or check for conflict if a coordinate vector is present. (Default is length=0)'''
    if self.data:
      assert length == self.shape[0], \
        'Coordinate vector of Axis instance \'%s\' is incompatible with given length: %i != %i'%(self.name,len(self),length)
    else:
      self.__dict__['len'] = length
      

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
    if name is not None: atts['name'] = name
    if title is not None: atts['title'] = title
    # load global attributes, if given
    if atts: self.__dict__['atts'] = AttrDict(**atts)
    else: self.__dict__['atts'] = AttrDict()
    # load variables (automatically adds axes linked to varaibles)
    for var in varlist:
      #print var.name
      self.addVariable(var)
      
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
    
  def addAxis(self, ax, copy=False, overwrite=False):
    ''' Method to add an Axis to the Dataset. If the Axis is already present, check that it is the same. '''
    if not isinstance(ax,Axis): raise TypeError
    if not self.hasAxis(ax.name): # add new axis, if it does not already exist        
      if ax.name in self.__dict__: 
        raise AttributeError, "Cannot add Axis '%s' to Dataset, because an attribute of the same name already exits!"%(ax.name)
      # add variable to dictionary
      if copy: self.axes[ax.name] = ax.copy()
      else: self.axes[ax.name] = ax
      self.__dict__[ax.name] = self.axes[ax.name] # create shortcut
    else: # make sure the axes are consistent between variable (i.e. same name, same axis)
      if overwrite:
        self.replaceAxis(ax)
      elif not ax is self.axes[ax.name]:        
        if len(ax) != len(self.axes[ax.name]): 
          raise AxisError, "Error: Axis '%s' from Variable and Dataset are different!"%ax.name
        if ax.data and self.axes[ax.name].data:
          if not isEqual(ax.coord,self.axes[ax.name].coord): raise DataError
    # double-check
    return self.axes.has_key(ax.name)       
    
  def addVariable(self, var, copy=False, deepcopy=False, overwrite=False):
    ''' Method to add a Variable to the Dataset. If the variable is already present, abort. '''
    if not isinstance(var,Variable): raise TypeError
    if var.name in self.__dict__: 
      if overwrite and self.hasVariable(var.name): self.replaceVariable(var)
      else: raise AttributeError, "Cannot add Variable '%s' to Dataset, because an attribute of the same name already exits!"%(var.name)      
    else:       
      # add new axes, or check, if already present; if present, replace, if different
      for ax in var.axes: 
        if not self.hasAxis(ax.name):
          self.addAxis(ax) # add new axis
        elif ax is not self.axes[ax.name]: 
          #print '   >>>   replacing a axis',ax.name
          var.replaceAxis(ax, self.axes[ax.name]) # or use old one of the same name
        # N.B.: replacing the axes in the variable is to ensure consistent axes within the dataset 
      # finally, if everything is OK, add variable
      if copy: self.variables[var.name] = var.copy(deepcopy=deepcopy)
      else: self.variables[var.name] = var
      self.__dict__[var.name] = self.variables[var.name] # create shortcut
    # double-check
    return self.variables.has_key(var.name) 
    
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
    
  def removeVariable(self, var):
    ''' Method to remove a Variable from the Dataset. '''
    if isinstance(var,basestring): var = self.variables[var] # only work with Variable objects
    assert isinstance(var,Variable), "Argument 'var' has to be a Variable instance or a string representing the name of a variable."
    if var.name in self.variables: # add new variable if it does not already exist
      # delete variable from dataset   
      del self.variables[var.name]
      del self.__dict__[var.name]
    # double-check (return True, if variable is not present, False, if it is)
    return not self.variables.has_key(var.name)
  
  def replaceAxis(self, oldaxis, newaxis=None):    
    ''' Replace an existing axis with a different one with similar general properties. '''
    if newaxis is None: 
      newaxis = oldaxis; oldaxis = newaxis.name # i.e. replace old axis with the same name'
    # check axis
    if not self.hasAxis(oldaxis): raise AxisError
    if isinstance(oldaxis,Axis): oldname = oldaxis.name # just go by name
    else: oldname = oldaxis
    oldaxis = self.axes[oldname]
    if len(oldaxis) != len(newaxis): raise AxisError # length has to be the same!
    if oldaxis.data != newaxis.data: raise DataError # make sure data status is the same
    # remove old axis and add new to dataset
    self.removeAxis(oldaxis, force=True)
    self.addAxis(newaxis, copy=False)
    newaxis = self.axes[newaxis.name] # update reference
    # loop over variables with this axis    
    for var in self.variables.values():
      if var.hasAxis(oldname): var.replaceAxis(oldname,newaxis)    
    # return verification
    return self.hasAxis(newaxis)    

  def replaceVariable(self, oldvar, newvar=None):
    ''' Replace an existing Variable with a different one and transfer NetCDF reference and axes. '''
    if newvar is None: 
      newvar = oldvar; oldvar = newvar.name # i.e. replace old var with the same name
    # check var
    if not self.hasVariable(oldvar): raise VariableError
    if isinstance(oldvar,Variable): oldname = oldvar.name # just go by name
    else: oldname = oldvar
    oldvar = self.variables[oldname]
    if oldvar.shape != newvar.shape: raise AxisError # shape has to be the same!
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
    
  def copy(self, axes=None, varlist=None, varargs=None, axesdeep=True, varsdeep=False, **kwargs): # this methods will have to be overloaded, if class-specific behavior is desired
    ''' A method to copy the Axes and Variables in a Dataset with just a link to the data arrays. '''
    # copy axes (shallow copy)    
    if axes is None: # allow override
      if axesdeep: newaxes = {name:ax.deepcopy() for name,ax in self.axes.iteritems()}
      else: newaxes = {name:ax.copy() for name,ax in self.axes.iteritems()}
    else: 
      if not isinstance(axes,dict): raise TypeError # check input
      newaxes = {name:ax.copy() for name,ax in axes.iteritems()}
    # check attributes
    if varargs is None: varargs=dict() 
    if not isinstance(varargs,dict): raise TypeError
    # copy variables
    if varlist is None: varlist = self.variables.keys() 
    variables = []
    for varname in varlist:
      var = self.variables[varname]
      axes = tuple([newaxes[ax.name] for ax in var.axes])
      if varname in varargs: # check input again
        if isinstance(varargs[varname],dict): args = varargs[varname]  
        else: raise TypeError
      else: args = dict()
      newvar = var.deepcopy(axes=axes, deepcopy=varsdeep, **args)
      variables.append(newvar)
    # determine attributes
    kwargs['varlist'] = variables
    if 'atts' not in kwargs: kwargs['atts'] = self.atts.copy() 
    # make new dataset
    dataset = Dataset(**kwargs)
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
    if short: pass 
    else:
      string = '{0:s}   {1:s}\n'.format(self.__class__.__name__,str(self.__class__))
      string += 'Variables:\n'
      for var in self.variables.values(): string += '  {0:s}\n'.format(var.prettyPrint(short=True))
      string += 'Axes:\n'
      for ax in self.axes.values(): string += '  {0:s}\n'.format(ax.prettyPrint(short=True))
      string += 'Attributes: {0:s}'.format(str(self.atts))
    return string
    
  def __getitem__(self, varname):
    ''' Yet another way to access variables by name... conforming to the container protocol. '''
    if not isinstance(varname, basestring): raise TypeError
    if not self.hasVariable(varname): raise KeyError
    return self.variables[varname]
  
  def __setitem__(self, varname, var):
    ''' Yet another way to add a variable, this time by name... conforming to the container protocol. '''
    if not isinstance(var, Variable) or not isinstance(varname, basestring): raise TypeError
    var.name = varname # change name to varname
    if 'name' in var.atts: var.atts['name'] = varname
    check = self.addVariable(var) # add variable
    if not check: raise KeyError # raise error if variable is not present
    
  def __delitem__(self, varname):
    ''' A way to delete variables by name... conforming to the container protocol. '''
    if not isinstance(varname, basestring): raise TypeError
    if not self.hasVariable(varname): raise KeyError
    check = self.removeVariable(varname)
    if not check: raise KeyError # raise error if variable has not disappeared
  
  def __iter__(self):
    ''' Return an iterator over all variables... conforming to the container protocol. '''
    return self.variables.itervalues() # just the iterator from the variables dictionary
    
  def __contains__(self, var):
    ''' Check if the Dataset instance has a particular Variable... conforming to the container protocol. '''
    if not (isinstance(var, Variable) or isinstance(var, basestring)): raise TypeError
    return self.hasVariable(var) # variable only
  
  def __len__(self):
    ''' Get the number of Variables in the Dataset. '''
    return len(self.variables)
    
  def __iadd__(self, var):
    ''' Add a Variable to an existing dataset. '''      
    assert self.addVariable(var), "A proble occurred adding Variable '%s' to Dataset."%(var.name)    
    return self # return self as result

  def __isub__(self, var):
    ''' Remove a Variable to an existing dataset. '''      
    assert self.removeVariable(var), "A proble occurred removing Variable '%s' from Dataset."%(var.name)
    return self # return self as result
  
  def load(self, data=None, **kwargs):
    ''' Issue load() command to all variable; pass on any keyword arguments. '''
    for var in self.variables.itervalues():
      var.load(data=data, **kwargs) # there is only one argument to the base method
      
  def unload(self, **kwargs):
    ''' Unload all data arrays currently loaded in memory. '''
    for var in self.variables.itervalues():
      var.unload(**kwargs)
      
  def mask(self, mask=None, maskSelf=False, varlist=None, skiplist=None, invert=False, merge=True, **kwargs):
    ''' Apply 'mask' to all variables and add the mask, if it is a variable. '''
    # figure out variable list
    if skiplist is None: skiplist = []
    if not isinstance(skiplist,(list,tuple)): raise TypeError
    if varlist is None: varlist = self.variables.keys()  
    if not isinstance(varlist,(list,tuple)): raise TypeError
    varlist = [var for var in varlist if var not in skiplist and var in self.variables]
    # iterate over all variables (not axes!) 
    for varname in varlist:
      var = self.variables[varname]
      if var.data and var.ndim >= mask.ndim:
        # need to have data and also the right number of dimensions 
        lOK = False
        if isinstance(mask,np.ndarray):
          if all(mask.shape == var.shape[var.ndim-mask.ndim:]): lOK = True
        elif isinstance(mask,Variable):
          if mask is var or mask.name == var.name: lOK = maskSelf # default: False
          elif all([var.hasAxis(ax) for ax in mask.axes]): lOK = True
        if lOK: var.mask(mask=mask,  invert=invert, merge=merge, **kwargs) # if everything is OK, apply mask
    # add mask do dataset
    if isinstance(mask,Variable) and not self.hasVariable(mask): self.addVariable(mask)
    
  def unmask(self, fillValue=None, **kwargs):
    ''' Unmask all Variables in the Dataset. '''
    for var in self.variables.itervalues():
      var.load(fillValue=fillValue, **kwargs)
      

## run a test    
if __name__ == '__main__':

  pass