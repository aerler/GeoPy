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
from misc import VariableError, DataError, checkIndex, isFloat

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
      assert orig.shape == arg.shape, 'Arrays need to have the same shape!' 
      assert orig.dtype == arg.dtype, 'Arrays need to have the same type!'
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
      tmp = orig.atts.copy(); tmp.update(other.atts)
      atts = {key:value for key,value in tmp.iteritems() if value == orig.atts[key]}
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


# def BinaryCheckAndCreateVar(sameUnits=True):
#   ''' A decorator to perform similarity checks before binary operations and create a new variable instance 
#       afterwards; name and units are modified; only non-conflicting attributes are kept. '''
#   # first decorator, that processes arguments 
#   # N.B.: basically the first wrapper produces a decorator function with fixed parameter values
#   def decorator_wrapper(binOp):
#     # define method wrapper (this is now the actual decorator)
#     def function_wrapper(self, other):
#       # initial sanity checks
#       assert isinstance(other,Variable), 'Can only add two \'Variable\' instances!' 
#       if sameUnits: assert self.units == other.units, 'Variable units have to be identical for addition!'
#       assert self.shape == other.shape, 'Variables need to have the same shape and compatible axes!'
#       if not self.data: self.load()
#       if not other.data: other.load()
#       for lax,rax in zip(self.axes,other.axes):
#         assert (lax.coord == rax.coord).all(), 'Variables need to have identical coordinate arrays!'
#       # call original method
#       data, name, units = binOp(self, other)
#       # construct common dict of attributes
#       tmp = self.atts.copy(); tmp.update(other.atts)
#       atts = {key:value for key,value in tmp.iteritems() if value == self.atts[key]}
#       atts['name'] = name; atts['units'] = units
#       # assign axes (copy from self)
#       axes = [ax for ax in self.axes]
#       var = Variable(name=name, units=units, axes=axes, data=data, atts=atts)
#       return var # return new variable instance
#     # return wrapper
#     return function_wrapper
#   return decorator_wrapper


## Variable class and derivatives 

class Variable(object):
  ''' 
    The basic variable class; it mainly implements arithmetic operations and indexing/slicing.
  '''
  
  def __init__(self, name='N/A', units='N/A', axes=None, data=None, mask=None, fillValue=None, atts=None, plotatts=None):
    ''' 
      Initialize variable and attributes.
      
      Basic Attributes:
        name = '' # short name, e.g. used in datasets
        units = '' # physical units
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
        plotatts = None # attributed used for displaying the data       
    '''
    # basic input check
    if data is None:
      ldata = False; shape = None; ndim = None; dtype = ''
    else:
      assert isinstance(data,np.ndarray), 'The data argument must be a numpy array!'
      ldata = True; shape = data.shape; dtype = data.dtype
      if axes is not None:
        assert len(axes) == data.ndim, 'Dimensions of data array and axes are note compatible!'
    # for completeness of MRO...
    super(Variable,self).__init__()
    # set basic variable 
    self.__dict__['name'] = name
    self.__dict__['units'] = units
    # set defaults - make all of them instance variables! (atts and plotatts are set below)
    self.__dict__['data_array'] = None
    self.__dict__['data'] = ldata
    self.__dict__['shape'] = shape
    self.__dict__['dtype'] = dtype
    self.__dict__['masked'] = False # handled in self.load() method    
    # figure out axes
    if axes is not None:
      assert isinstance(axes, (list, tuple))
      if all([isinstance(ax,Axis) for ax in axes]):
        if ldata: 
          for ax,n in zip(axes,shape): ax.updateLength(n)
        if not ldata and all([len(ax) for ax in axes]):
          self.__dict__['shape'] = [len(ax) for ax in axes] # get shape from axes
      elif all([isinstance(ax,str) for ax in axes]):
        if ldata: axes = [Axis(name=ax, len=n) for ax,n in zip(axes,shape)] # use shape from data
        else: axes = [Axis(name=ax) for ax in axes] # initialize without shape
    else: 
      raise VariableError, 'Cannot initialize %s instance \'%s\': no axes declared'%(self.var.__class__.__name__,self.name)
    self.__dict__['axes'] = tuple(axes) 
    self.__dict__['ndim'] = len(axes)  
    # create shortcuts to axes (using names as member attributes) 
    for ax in axes: self.__dict__[ax.name] = ax
    # assign attributes
    if atts is None: atts = dict(name=self.name, units=self.units)
    self.__dict__['atts'] = atts
    if plotatts is None: # try to find sensible default values 
      if variablePlotatts.has_key(self.name): plotatts = variablePlotatts[self.name]
      else: plotatts = dict(plotname=self.name, plotunits=self.units, plottitle=self.name) 
    self.__dict__['plotatts'] = plotatts
    # guess fillValue
    if fillValue is None:
      if 'fillValue' in atts: fillValue = atts['fillValue']
      elif '_fillValue' in atts: fillValue = atts['_fillValue']
      else: fillValue = None
    self.__dict__['fillValue'] = fillValue
    # assign data, if present (can initialize without data)
    if data is not None: 
      self.load(data, mask=mask, fillValue=fillValue) # member method defined below
    
    
  def __getattr__(self, name):
    ''' Return contents of atts or plotatts dictionaries as if they were attributes. '''
    # N.B.: before this method is called, instance attributes are checked automatically
    if self.__dict__.has_key(name): # check instance attributes first
      return self.__dict__[name]
    elif self.__dict__['atts'].has_key(name): # try atts second
      return self.__dict__['atts'][name] 
    elif self.__dict__['plotatts'].has_key(name): # then try plotatts
      return self.__dict__['plotatts'][name]
    else: # or throw attribute error
      raise AttributeError, '\'%s\' object has no attribute \'%s\''%(self.__class__.__name__,name)
    
  def __setattr__(self, name, value):
    ''' Change the value of class existing class attributes, atts, or plotatts entries,
      or store a new attribute in the 'atts' dictionary. '''
    if self.__dict__.has_key(name): # class attributes come first
      self.__dict__[name] = value # need to use __dict__ to prevent recursive function call
    elif self.__dict__['atts'].has_key(name): # try atts second
      self.__dict__['atts'][name] = value    
    elif self.__dict__['plotatts'].has_key(name): # then try plotatts
      self.__dict__['plotatts'][name] = value
    else: # if the attribute does not exist yet, add it to atts or plotatts
      if name[0:4] == 'plot':
        self.plotatts[name] = value
      else:
        self.atts[name] = value
  
  def hasAxis(self, axis):
    ''' Check if the variable instance has a particular axis. '''
    if isinstance(axis,str): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return True
    elif isinstance(axis,Variable): # by object ID
      for i in xrange(len(self.axes)):
        if self.axes[i] == axis: return True
    # if all fails
    return False

  def __contains__(self, axis):
    ''' Check if the variable instance has a particular axis. '''
    # same as class method
    return self.hasAxis(axis)
  
  def __len__(self):
    ''' Return number of dimensions. '''
    return self.__dict__['ndim']
  
  def axisIndex(self, axis):
    ''' Return the index of a particular axis. (return None if not found) '''
    if isinstance(axis,str): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return i
    elif isinstance(axis,Variable): # by object ID
      for i in xrange(len(self.axes)):
        if self.axes[i] == axis: return i
    # if all fails
    return None
        
  def __getitem__(self, idx):
    ''' Method implementing access to the actual data, plus some extras. '''          
    # determine what to do
    if all(checkIndex(idx, floatOK=True)):      
      # array indexing: return array slice
      if self.data:
        if any(isFloat(idx)): raise NotImplementedError, \
          'Floating-point indexing is not implemented yet for \'%s\' class.'%(self.__class__.__name__)
        return self.data_array.__getitem__(idx) # valid array slicing
      else: 
        raise IndexError, 'Variable instance \'%s\' has no associated data array!'%(self.name) 
    elif isinstance(idx,str) or isinstance(idx,Axis):
      # dictionary-type key: return index of dimension with that name
      return self.axisIndex(idx)
    else:    
      # if nothing applies, raise index error
      raise IndexError, 'Invalid index/key type for class \'%s\'!'%(self.__class__.__name__)

  def load(self, data=None, mask=None, fillValue=None):
    ''' Method to attach numpy data array to variable instance (also used in constructor). '''
    assert data is not None, 'A basic \'Variable\' instance requires external data to load!'
    assert isinstance(data,np.ndarray), 'The data argument must be a numpy array!'          
    if mask: 
      self.__dict__['data_array'] = ma.array(data, mask=mask)
    else: 
      self.__dict__['data_array'] = data
    if isinstance(self.data_array, ma.MaskedArray): 
      self.__dict__['masked'] = True # set masked flag
    self.__dict__['data'] = True
    self.__dict__['shape'] = data.shape
    assert len(self.shape) == self.ndim, 'Variable dimensions and data dimensions incompatible!'
    self.__dict__['dtype'] = data.dtype
    if self.masked: # figure out fill value for masked array
      if fillValue is None: self.__dict__['fillValue'] = ma.default_fill_value(data)
      else: self.__dict__['fillValue'] = fillValue
    # some more checks
    # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple; hence
    #       the coordinate vector has to be assigned before the dimensions size can be checked 
    assert len(self.axes) == len(self.shape), 'Dimensions of data array and variable must be identical!'
    for ax,n in zip(self.axes,self.shape): 
      ax.updateLength(n) # update length is all we can do without a coordinate vector       
     
  def unload(self):
    ''' Method to unlink data array. '''
    self.__dict__['data_array'] = None # unlink data array
    self.__dict__['data'] = False # set data flag
    self.__dict__['fillValue'] = None
    # self.__dict__['shape'] = None # retain shape for later use
    
  def get(self, idx=None, unmask=True, fillValue=None):
    ''' Copy the entire data array or a slice; with some extra options. '''
    if all(checkIndex(idx, floatOK=True)): datacopy = self.__getitem__(idx).copy() # use __getitem__ to get slice
    else: datacopy = self.data_array.copy()
    if unmask and self.masked:
      if fillValue is None: fillValue=self.fillValue
      datacopy = datacopy.filled(fill_value=fillValue)
    return datacopy
    
  def mask(self, mask=None, fillValue=None, merge=True):
    ''' A method to add a mask to an unmasked array, or extend or replace an existing mask. '''
    if mask is not None:
      assert isinstance(mask,np.ndarray), 'Mask has to be a numpy array!'  
      assert len(self.shape) == len(mask.shape) and self.shape == mask.shape, 'Data array and mask have to be of the same shape!'
      # create new data array
      if merge and self.masked: # the first mask is usually the land-sea mask, which we want to keep
        data = self.get(unmask=False) # get data with mask
        mask = ma.mask_or(data.mask, mask, copy=True, shrink=False) # merge masks
      else: 
        data = self.get(unmask=True) # get data without mask
      self.__dict__['data_array'] = ma.array(data, mask=mask)
      # change meta data
      self.__dict__['masked'] = True
      if fillValue: 
        self.data_array.set_fill_value(fillValue)
        self.__dict__['fillValue'] = fillValue
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


class Axis(Variable):
  '''
    A special class of 1-dimensional variables for coordinate variables.
     
    It is essential that this class does not overload any class methods of Variable, 
    so that new Axis sub-classes can be derived from new Variable sub-classes via 
    multiple inheritance from the Variable sub-class and this class. 
  '''
  
  coord = None # the coordinate vector (also accessible as data_array)
  len = 0 # the length of the dimension (integer value)
  
  def __init__(self, length=0, coord=None, **varargs):
    ''' Initialize a coordinate axis with appropriate values. '''
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
      elif isinstance(coord,tuple) or isinstance(coord,list):
        data = np.asarray(coord)
      else: #data = coord
        raise TypeError, 'Data type not supported for coordinate values.'
      # load data
      self.load(data, mask=None, **varargs)
      

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

## run a test    
if __name__ == '__main__':

  # initialize test objects
  x = Axis(name='x', units='none', coord=(1,5,5))
  y = Axis(name='y', units='none', coord=(1,5,5))
  var = Variable(name='test',units='none',axes=(x,y),data=np.zeros((5,5)),atts=dict(_FillValue=-9999))
  
  # variable test
  print
  var += 1
  # test getattr
  print 'Name: %s, Units: %s, Missing Values: %s'%(var.name, var.units, var._FillValue)
  # test setattr
  var.Comments = 'test'; var.plotComments = 'test' 
  print 'Comments: %s, Plot Comments: %s'%(var.Comments,var.plotatts['plotComments'])
#   print var[:]
  # indexing (getitem) test
  print var.shape, var[2,2:5:2]
  var.unload()
#   print var.data
  
  # axis test
  print 
  # test contains 
  print var[x]
  for ax in (x,y):
    if ax in var: print '%s is the %i. axis and has length %i'%(ax.name,var[ax]+1,len(ax))
