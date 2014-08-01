'''
Created on 2013-08-19

Variable and Dataset classes for handling geographical datasets.

@author: Andre R. Erler, GPL v3
'''

# numpy imports
import numpy as np
import numpy.ma as ma # masked arrays
# my own imports
from plotting.properties import variablePlotatts # import plot properties from different file
from misc import checkIndex, isEqual, isInt, isFloat, AttrDict, joinDicts, printList
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
    elif not isinstance(arg, numbers.Number): raise TypeError, 'Can only operate with numerical types!'
    if not orig.data: orig.load()
    var = self.op(orig,arg)
    if not isinstance(var,Variable): raise TypeError
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
      if not isinstance(other,Variable): raise TypeError, 'Can only add two \'Variable\' instances!' 
      if self.sameUnits: assert orig.units == other.units, 'Variable units have to be identical for addition!'
      if orig.shape != other.shape: raise AxisError, 'Variables need to have the same shape and compatible axes!'
      if not orig.data: orig.load()
      if not other.data: other.load()
      for lax,rax in zip(orig.axes,other.axes):
        if (lax.coord != rax.coord).any(): raise AxisError,  'Variables need to have identical coordinate arrays!'
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

class ReduceVar(object):
  ''' Decorator class that implements some sanity checks for reduction operations. '''
  def __init__(self, redop):
    ''' Save original operation. '''
    self.redop = redop
  def __call__(self, orig, asVar=None, checkAxis=True, coordIndex=True, axis=None, axes=None, **kwaxes):
    ''' Figure out axes, perform sanity checks, then execute operation, and return result as a Variable 
        instance. Axes are specified either in a list ('axes') or as keyword arguments with corresponding
        slices. '''
    if axis is None and axes is None and len(kwaxes) == 0:
      if not orig.data: orig.load()
      # apply operation without arguments, i.e. over all axes
      data = self.redop(orig, orig.data_array)
      # whether or not to cast as Variable (default: No)
      if asVar is None: asVar = False # default for total reduction
      if asVar: newaxes = tuple()
    else:
      # add axes list to dictionary
      if axis is not None: 
        kwaxes[axis] = None
        assert axes is None 
      elif axes is not None: 
        for ax in axes: kwaxes[ax] = None
      # check for axes in keyword arguments       
      for ax,slc in kwaxes.iteritems():
        #print ax,slc
        if checkAxis and not orig.hasAxis(ax): raise AxisError
        if slc is not None and not isinstance(slc,(slice,list,tuple,int,np.integer)): raise TypeError
      # order axes and get indices
      axlist = [ax.name for ax in orig.axes if ax.name in kwaxes]
      # get data  
      if coordIndex:
        # use overloaded call method to index with coordinate values directly 
        data = orig.__call__(**kwaxes)
      else: 
        # sort slices accordign to axes
        idx = [None]*orig.ndim
        for ax,slc in kwaxes.iteritems():
          idx[orig.axisIndex(ax)] = slc
        # get slices the usual way
        orig.__getitem__(idx=idx)
      # compute mean
      axlist.reverse() # start from the back      
      for axis in axlist:
        # apply reduction operation with axis argument, looping over axes
        data = self.redop(orig, data, axidx=orig.axisIndex(axis))
      # squeeze removed dimension (but no other!)
      newshape = [len(ax) for ax in orig.axes if not ax.name in axlist]
      data = data.reshape(newshape)
      # whether or not to cast as Variable (default: Yes)
      if asVar is None: asVar = True # default for iterative reduction
      if asVar: newaxes = [ax for ax in orig.axes if not ax.name in axlist] 
    # N.B.: other singleton dimensions will have been removed, too
    # cast into Variable
    if asVar: 
      #print self.name, data.__class__.__name__
      #print data, newaxes
      var = Variable(name=orig.name, units=orig.units, axes=newaxes, data=data, 
                     fillValue=orig.fillValue, atts=orig.atts.copy(), plot=orig.plot.copy())
    else: var = data
    return var # return function result
  def __get__(self, instance, klass):
    ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
    return functools.partial(self.__call__, instance) # but using 'partial' is simpler


## Variable class and derivatives 

class Variable(object):
  ''' 
    The basic variable class; it mainly implements arithmetic operations and indexing/slicing.
  '''
  
  def __init__(self, name=None, units=None, axes=None, data=None, dtype=None, mask=None, fillValue=None, atts=None, plot=None):
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
        dataset = None # parent dataset the variable belongs to
        atts = None # dictionary with additional attributes
        plot = None # attributed used for displaying the data       
    '''
    # basic input check
    if data is None:
      ldata = False; shape = None
    else:
      if not isinstance(data,np.ndarray): data = np.asarray(data) # 'The data argument must be a numpy array!'
      ldata = True; shape = data.shape; 
      if dtype:
        dtype = np.dtype(dtype) # make sure it is properly formatted.. 
        if dtype is not data.dtype: data = data.astype(dtype) # recast as new type        
#         raise TypeError, "Declared data type '%s' does not match the data type of the array (%s)."%(str(dtype),str(data.dtype))
      if axes is not None and len(axes) != data.ndim: 
        raise AxisError, 'Dimensions of data array and axes are note compatible!'
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
    self.__dict__['data'] = False # data has not been loaded yet
    self.__dict__['shape'] = shape
    self.__dict__['dtype'] = dtype
    self.__dict__['masked'] = False # handled in self.load() method    
    self.__dict__['dataset'] = None # set by addVariable() method of Dataset  
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
      raise VariableError, 'Cannot initialize %s instance \'%s\': no axes declared'%(self.var.__class__.__name__,self.name)
    self.__dict__['axes'] = tuple(axes) 
    self.__dict__['ndim'] = len(axes)  
    # create shortcuts to axes (using names as member attributes) 
    for ax in axes: self.__dict__[ax.name] = ax
    # assign data, if present (can initialize without data)
    if data is not None: 
      self.load(data, mask=mask, fillValue=fillValue) # member method defined below
      assert self.data == ldata # should be loaded now
      
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
    else: oldname = oldaxis
    oldaxis = self.axes[self.axisIndex(oldname)]
    if not self.hasAxis(oldaxis): raise AxisError
    if len(oldaxis) != len(newaxis): raise AxisError # length has to be the same!
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

  def getAxis(self, axis, lcheck=True):
    ''' Return a reference to the Axis object or one with the same name. '''
    return self.axes[self.axisIndex(axis)]
  
  def axisIndex(self, axis):
    ''' Return the index of a particular axis. (return None if not found) '''
    if isinstance(axis,basestring): # by name
      for i in xrange(len(self.axes)):
        if self.axes[i].name == axis: return i
    elif isinstance(axis,Axis): # by object ID
      for i in xrange(len(self.axes)):
        if self.axes[i] == axis: return i
    # if all fails
    elif lcheck: raise AxisError, "Axis '{:s}' not found!".format(str(axis))
    return None
        
  def __getitem__(self, idx=None):
    ''' Method implementing access to the actual data. '''
    # default
    if idx is None: 
      if len(self.shape) > 0: idx = slice(None,None,None) # first, last, step
      elif isinstance(self.data_array,(np.ndarray,numbers.Number)): 
        return self.data_array # if the data is scalar, just return it
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
  
  def __call__(self, years=None, lgetIndex=True, lcoordList=False, lcheckaxis=True, **kwargs):
    ''' This method implements access to slices via coordinate values (as opposed to indices). '''
    # loop over arguments and find indices of coordinate values
    slices = dict()
    # resolve special key words
    if years is not None:
      if self.hasAxis('time'):
        time = self.getAxis('time')
        if not time.units.lower() in ('month','months'): raise NotImplementedError, 'Can only convert years to month!'
        if '1979' in time.atts.long_name: offset = 1979
        else: offset = 0
        if isinstance(years,np.number): months = (years - offset)*12
        elif isinstance(years,(list,tuple)): months = [ (yr - offset)*12 for yr in years]
        kwargs['time'] = months
      elif lcheckaxis: raise AxisError, "Axis '{}' not found!".format('time')
    # search for regular dimensions 
    for axname,coord in kwargs.iteritems():      
      if self.hasAxis(axname):
        ax = self.getAxis(axname)
        if ax.data:
          if lgetIndex:
            if isinstance(coord,(list,tuple)):
              if lcoordList:
                # N.B.: this feature is poorly tested...
                slices[axname] = [ax.getIndex(cv) for cv in coord] # just look up the indices for the coordinate values
              else:
                if len(coord) == 1: slices[axname] = ax.getIndex(coord[0])
                elif len(coord) == 2:
                  #l = max(ax.data_array.searchsorted(coord[0],side='right')-1,0) # choose such as to bracket coords
                  #r = ax.data_array.searchsorted(coord[1],side='left') # same value or higher index
                  # N.B.: I am not sure what the above version was supposed to achieve, but a simple version that works well is below (commented.
                  # l = ax.data_array.searchsorted(coord[0],side='left')
                  # r = ax.data_array.searchsorted(coord[1],side='right')                  
                  slices[axname] = slice(ax.getIndex(coord[0], mode='left'),ax.getIndex(coord[1], mode='right'))
                elif len(coord) == 3:  
                  coord = np.linspace(*coord) # expand into linearly spaced list
                  slices[axname] = [ax.getIndex(cv) for cv in coord] # and look up the indices
                else: raise IndexError
            elif isinstance(coord,np.number):
              slices[axname] = ax.getIndex(coord)
            elif coord is None:
              slices[axname] = slice(None)
            else: raise TypeError
          else: # means index has to be passed directly (no coordinate look-up
            # N.B.: this feature was not tested...
            if isinstance(coord,(list,tuple)):
              if not all(isInt(coord)): raise IndexError, 'Only integers can be used as indices.'
              slices[axname] = coord
            elif isinstance(coord,slice):
              slices[axname] = coord
            elif coord is None:
              slices[axname] = slice(None)
            else: raise TypeError
        else: 
          raise AxisError, "Axis '{}' has no coordinate vector!".format(ax.name)
      else:
        if lcheckaxis: raise AxisError, "Axis '{}' not found!".format(axname)
    # assemble index tuple for axes
    idx = tuple([slices.get(ax.name,slice(None)) for ax in self.axes])
    return self.__getitem__(idx=idx) # pass on to getitem

  def load(self, data=None, mask=None, fillValue=None):
    ''' Method to attach numpy data array to variable instance (also used in constructor). '''
    if data is None:
      if not self.data:
        raise DataError, 'No data loaded and no external data supplied!'
    else:
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
        for ax,n in zip(self.axes,self.shape): ax.len = n 
      else: # this should only happen with scalar variables!
        assert self.ndim == 0 and data.size == 1, 'Dimensions of data array and variable must be identical, except for scalars!'       
     
  def unload(self):
    ''' Method to unlink data array. '''
    del self.__dict__['data_array'] # delete array
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
    if unmask and isinstance(datacopy, ma.MaskedArray): 
      # N.B.: if no data is loaded, self.mask is usually false...
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
    ''' A method to remove an existing mask and fill the gaps with fillValue. '''
    if self.masked:
      if fillValue is None: fillValue = self.fillValue # default
      self.__dict__['data_array'] = self.data_array.filled(fill_value=fillValue)
      # change meta data
      self.__dict__['masked'] = False
      self.__dict__['fillValue'] = None  
    
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
  
  def limits(self):
    ''' A convenience function to return a min,max tuple to indicate the data range. '''
    if self.data:
      return self.data_array.min(), self.data_array.max()
    else: 
      return None
  
  @ReduceVar
  def mean(self, data, axidx=None):
    return data.mean(axis=axidx)
  
  @ReduceVar
  def max(self, data, axidx=None):
    return data.max(axis=axidx)
  
  @ReduceVar
  def min(self, data, axidx=None):
    return data.min(axis=axidx)
    
  def reduceToAnnual(self, season, operation, asVar=False, offset=0, taxis='time', checkUnits=True, taxatts=None, varatts=None):
    ''' Reduce a monthly time-series to an annual time-series, using mean/min/max over a subset of month or seasons. '''
    #if not isinstance(season,basestring): raise TypeError
    if not self.data: raise DataError
    if not self.hasAxis(taxis): raise AxisError, 'Seasonal reduction requires a time axis!'
    taxis = self.getAxis(taxis)
    if checkUnits and not taxis.units.lower() in ('month','months'): raise AxisError, 'Seasonal reduction requires monthly data!'
    te = len(taxis); tax = self.axisIndex(taxis.name)
    if te%12 != 0: raise NotImplementedError, 'Currently seasonal means only work for full years.'
    # determine season
    if isinstance(season,(int,np.integer)): idx = np.asarray([season])
    elif isinstance(season,(list,tuple)):
      if all([isinstance(s,(int,np.integer)) for s in season]): 
	idx = np.asarray(season)
      else: raise TypeError      
    elif isinstance(season,basestring):
      ssn = season.lower() # ignore case
      year = 'jfmamjjasondjfmamjjasond' # all month, twice
      # N.B.: regular Python indexing, starting at 0 for Jan and going to 11 for Dec
      if ssn == 'jfmamjjasond' or ssn == 'annual': idx = np.arange(12)
      elif ssn == 'jja' or ssn == 'summer': idx = np.asarray([5,6,7])
      elif ssn == 'djf' or ssn == 'winter': idx = np.asarray([11,0,1])
      elif ssn == 'mam' or ssn == 'spring': idx = np.asarray([2,3,4])
      elif ssn == 'son' or ssn == 'fall'  or ssn == 'autumn': idx = np.asarray([8,9,10])
      elif ssn == 'mamjja' or ssn == 'warm': idx = np.asarray([2,3,4,5,6,7])
      elif ssn == 'sondjf' or ssn == 'cold': idx = np.asarray([8,9,10,11,0,1,])
      elif ssn == 'amj' or ssn == 'melt': idx = np.asarray([3,4,5,])
      elif ssn in year: 
	s = year.find(ssn) # find first occurrence of sequence
	idx = np.arange(s,s+len(ssn))%12 # and use range of months
      else: raise ValueError, "Unknown key word/season: '{:s}'".format(str(season))
    else: raise TypeError, "Unknown identifier for season: '{:s}'".format(str(season))
    # get actual data and reshape
    mdata = self.getArray()
    if tax > 0: np.rollaxis(mdata, axis=tax, start=0) # move time axis to front
    # reshape for annual array
    oldshape = mdata.shape
    y = te/12 # number of years
    newshape = (y,)+oldshape[1:]
    #print oldshape, (y,12)+oldshape[1:]
    mdata = mdata.reshape((y,12)+oldshape[1:])
    # compute mean/min/max
    if mdata.ndim > 2: tmp = mdata[:,idx,:]
    else: tmp = mdata[:,idx]
    if operation == 'mean': ydata = tmp.mean(axis=1)
    elif operation == 'max': ydata = tmp.max(axis=1)
    elif operation == 'min': ydata = tmp.min(axis=1)
    else: raise NotImplementedError, "Unknown operation: '{:s}'".format(operation)
    assert ydata.shape == newshape
    # return new variable
    if tax > 0: np.rollaxis(mdata, axis=0, start=tax+1) # move time axis to front
    # cast as variable
    if asVar:      
      # create new time axis (yearly)
      tatts = self.time.atts.copy()
      tatts['name'] = 'year'; tatts['units'] = 'year'
      # N.B.: offset is a parameter to simply shift the time axis origin
      coord = np.linspace(int(taxis.coord[11]/12), int(taxis.coord[-1]/12), int(len(taxis)/12)) + offset
      # N.B.: this should preserve 1-ness or 0-ness
      if taxatts is not None: tatts.update(taxatts)      
      axes = list(self.axes); axes[tax] = Axis(coord=coord, dtype='int', atts=tatts)
      # create new variable
      vatts = self.atts.copy()
      vatts['name'] = self.name; vatts['units'] = self.units
      if varatts is not None: vatts.update(varatts)
      return Variable(data=ydata, axes=axes, atts=vatts)
      # or return data
    else: return ydata
  
  def seasonalMean(self, season, asVar=False, offset=0, taxis='time', checkUnits=True):
    ''' Return a time-series of annual averages of the specified season. '''    
    return self.reduceToAnnual(season=season, operation='mean', asVar=asVar, offset=offset, taxis=taxis, checkUnits=checkUnits)
  
  def seasonalMax(self, season, asVar=False, offset=0, taxis='time', checkUnits=True):
    ''' Return a time-series of annual averages of the specified season. '''    
    return self.reduceToAnnual(season=season, operation='max', asVar=asVar, offset=offset, taxis=taxis, checkUnits=checkUnits)
  
  def seasonalMin(self, season, asVar=False, offset=0, taxis='time', checkUnits=True):
    ''' Return a time-series of annual averages of the specified season. '''    
    return self.reduceToAnnual(season=season, operation='min', asVar=asVar, offset=offset, taxis=taxis, checkUnits=checkUnits)
  
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
          coord = @property # managed access to the coordinate vector
          len = @property # the current length of the dimension (integer value)
    '''
    # initialize dimensions
    axes = (self,)
    # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple
    self.__dict__['_len'] = length
    # initialize as a subclass of Variable, depending on the multiple inheritance chain    
    super(Axis, self).__init__(axes=axes, **varargs)
    # add coordinate vector
    if coord is not None: self.coord = coord
    elif length > 0: self.len = length

  @property
  def coord(self):
    ''' An alias for the data_array variable that is specific to coordiante vectors. '''
    return self.getArray() # unmask=True ?
  @coord.setter
  def coord(self, data):
    ''' Update the coordinate vector of an axis based on certain conventions. '''
    # resolve coordinates
    if data is None:
      # this means the coordinate vector/data is going to be deleted 
      self.unload()
    else:
      # a coordinate vector will be created and loaded, based on input conventions
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
      # load data
      self.load(data=data, mask=None)

  @property
  def len(self):
    ''' The length of the axis; if a coordinate vector is present, it is the length of that vector. '''
    if self.data: return self.coord.size
    else: return self._len    
  @len.setter
  def len(self, length):
    ''' Update the length, or check for conflict if a coordinate vector is present. (Default length is 0)'''
    if self.data and length != self.coord.size:
      raise AxisError, 'Axis instance \'{:s}\' already has a coordinate vector of length {:d} ({:d} given)'.format(self.name,len(self),length)        
    else: self.__dict__['_len'] = length
  
  def __len__(self):
    ''' Return length of dimension. '''
    return self.len 
    
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

  def getIndex(self, value, mode='closest'):
    ''' Return the coordinate index that is closest to the value or suitable for index ranges (left/right). '''
    if not self.data: raise DataError
    # behavior depends on mode
    if mode == 'left':
      # returns value suitable for beginning of range (inclusive)
      return self.coord.searchsorted(value, side='left')
    elif mode == 'right':    
      # returns value suitable for end of range (inclusive)
      return self.coord.searchsorted(value, side='right')
    elif mode == 'closest':      
      # search for closest index
      idx = self.coord.searchsorted(value) # returns value 
      # refine search
      if idx <= 0: return 0
      elif idx >= self.len: return self.len-1
      else:
	dl = value - self.coord[idx-1]
	dr = self.coord[idx] - value
	if dr < dl: return idx
	else: return idx-1 # can't be 0 at this point 
    else: raise ValueError, "Mode '{:s}' unknown.".format(mode)
                  

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
    else: atts['name'] = ''
    if title is not None: atts['title'] = title
    else: atts['title'] = ''
    # load global attributes, if given
    if atts: self.__dict__['atts'] = AttrDict(**atts)
    else: self.__dict__['atts'] = AttrDict()
    # load variables (automatically adds axes linked to varaibles)
    for var in varlist:
      #print var.name
      self.addVariable(var, copy=False) # don't make copies of new variables!
      
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
    # set dataset attribute
    self.axes[ax.name].dataset = self
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
    # set dataset attribute
    self.variables[var.name].dataset = self
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
    
  def getVariable(self, varname, check=True):
    ''' Method to return a Variable by name '''
    if not isinstance(varname,basestring): raise TypeError
    if self.hasVariable(varname):
      return self.variables[varname]
    else:
      if check: raise VariableError, "Variable '{:s}' not found!".format(varname)
      else: return None
  
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
    
  def getAxis(self, axname, check=True):
    ''' Method to return an Axis by name '''
    if not isinstance(axname,basestring): raise TypeError
    if self.hasAxis(axname):
      return self.axes[axname]
    else:
      if check: raise AxisError, "Axis '{:s}' not found!".format(axname)
      else: return None

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
    if short:
      title = self.title or self.name
      variables = '{:3d} Vars'.format(len(self.variables)) # number of variables
      axes = '{:2d} Axes'.format(len(self.axes)) # number of axes
      klass = self.__class__.__name__ 
      string = '{:<20s} {:s}, {:s} ({:s})'.format(title,variables,axes,klass)
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
    assert self.addVariable(var), "A problem occurred adding Variable '{:s}' to Dataset.".format(var.name)    
    return self # return self as result

  def __isub__(self, var):
    ''' Remove a Variable to an existing dataset. '''      
    assert self.removeVariable(var), "A proble occurred removing Variable '{:s}' from Dataset.".format(var.name)
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
          if mask.shape == var.shape[var.ndim-mask.ndim:]: lOK = True
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
      
  def mean(self, squeeze=True, checkAxis=True, coordIndex=True, **axes):
    ''' Average entire dataset, and return a new (reduced) one. '''
    newset = Dataset(name=self.name, title=self.title, varlist=[], atts=self.attscopy())    
    # loop over variables
    for var in self.variable:
      # figure out, which axes apply
      tmpax = {key:value for key,value in axes.iteritems() if var.hasAxis(key)}
      # get averaged variable
      if len(tmpax) > 0:
        self.addVariable(var.mean(**tmpax), copy=False) # new variable/values anyway
      else: 
        self.addVariable(var, copy=True, deepcopy=True) # copy values
    # add some record
    for key,value in axes.iteritem():
      if isinstance(value,(list,tuple)): newset.atts[key] = printList(value)
      elif isinstance(value,np.number): newset.atts[key] = str(value)      
      else: newset.atts[key] = 'n/a'
    # return new dataset
    return newset
      

def concatVars(variables, axis='time', axlim=None, asVar=True, offset=0, axatts=None, varatts=None):
  ''' A function to concatenate variables from different sources along a given axis;
      this is useful to generate a continuous time series from an ensemble. '''
  if not all([isinstance(var,Variable) for var in variables]): raise TypeError
  if not all([var.hasAxis(axis) for var in variables]): raise AxisError  
  var0 = variables[0] # shortcut
  axt = var0.getAxis(axis)
  tax = var0.axisIndex(axis)
  if axlim is not None:
    laxlim = True
    axlim = {axis:axlim}
  # check dimensions
  shapes = []; tes = []
  for var in variables:
    shp = list(var.shape)
    if laxlim: tes.append(len(axt(**axlim)))
    else: tes.append(shp[tax])
    del shp[tax]
    shapes.append(shp)
  if not all([s == shp for s in shapes]): raise AxisError
  tlen = 0
  for te in tes: tlen += te
  newshape = list(var0.shape)
  newshape[tax] = tlen
  newshape = tuple(newshape)
  # load data
  data = []
  for var in variables:
    if not var.data: var.load()
    if laxlim is not None: 
      array = var(**axlim)      
    else: 
      array = var.getArray()
      assert array.shape == var.shape
    data.append(array)
  # concatenate
  data = np.concatenate(data, axis=tax)
  assert data.shape[tax] == tlen
  #print data.shape, newshape
  assert data.shape == newshape
  # cast as variable
  if asVar:      
    # create new time axis (yearly)    
    axatts = axt.atts.copy()
    axatts['name'] = axt.name; axatts['units'] = axt.units
    # N.B.: offset is a parameter to simply shift the time axis origin
    coord = np.arange(tlen) + offset
    if axatts is not None: axatts.update(axatts)      
    axes = list(var0.axes); axes[tax] = Axis(coord=coord, dtype='int', atts=axatts)
    # create new variable
    vatts = var0.atts.copy()
    vatts['name'] = var0.name; vatts['units'] = var0.units
    if varatts is not None: vatts.update(varatts)
    return Variable(data=data, axes=axes, atts=vatts)
    # or return data
  else: return data

class Ensemble(object):
  '''
    A container class that holds several datasets ("members" of the ensemble),
    furthermore, the Ensemble class provides functionality to execute Dataset
    class methods collectively for all members, and return the results in a tuple.
  '''
  members  = None    # list of members of the ensemble
  basetype = Dataset # base class of the ensemble members
  idkey    = 'name'  # property of members used for unique identification
  name     = ''      # name of the ensemble
  title    = ''      # printable title used for the ensemble
  
  def __init__(self, *datasets, **kwargs):
    ''' Initialize an ensemble from a list of datasets (the list arguments);
    keyword arguments are added as attributes (key = attribute name, 
    value = attribute value).
    
	Attributes:
	  members  = list/tuple of members of the ensemble
	  basetype = class of the ensemble members
	  idkey    = property of members used for unique identification
	  name     = name of the ensemble (string)
	  title    = printable title used for the ensemble (string)
    '''
    # add members
    self.members = list(datasets)
    # add certain properties
    self.name = kwargs.get('name','')
    self.title = kwargs.get('title','')
    self.basetype = kwargs.get('basetype',datasets[0].__class__)
    self.idkey = kwargs.get('idkey','name')
    # add keywords as attributes
    for key,value in kwargs.iteritems():
      self.__dict__[key] = value
    # add short-cuts
    for member in self.members:
      memid = getattr(member, self.idkey)
      if not isinstance(memid, basestring): raise TypeError, "Member ID key '{:s}' should be a string-type, but received '{:s}'.".format(str(memid),memid.__class__)
      if memid in self.__dict__:
	raise AttributeError, "Cannot overwrite existing attribute '{:s}'.".format(memid)
      self.__dict__[memid] = member
      
  def __getattr__(self, attr):
    ''' This is where all the magic happens: defer calls to methods etc. to the 
	ensemble members and return a list of values. '''
    # intercept some list methods
    #print dir(self.members), attr, attr in dir(self.members)
    # determine whether we need a wrapper
    fs = [getattr(member,attr) for member in self.members]
    if all([not callable(f) or isinstance(f, Variable) for f in fs]):
      # N.B.: technically, Variable instances are callable, but that's not what we want here...
      # simple values, not callable
      #if all([isinstance(f, Variable) and not isinstance(f, Axis) for f in fs]):
      # check for unique keys
      if len(fs) == len(set([f.name for f in fs if f is not None])): 
	return Ensemble(*fs, basetype=Variable, idkey='name')
      elif len(fs) == len(set([f.dataset.name for f in fs if f is not None])): 
	for f in fs: f.dataset_name = f.dataset.name 
	return Ensemble(*fs, idkey='dataset_name') #basetype=Variable, 
      #elif all([isinstance(f, Variable) for f in fs]):
	#for f,m in zip(fs,self.members): f.dataset_name = getattr(m,self.idkey)
	#return Ensemble(*fs, idkey='dataset_name') #basetype=Variable, 
      else: return fs
    else:
      # for callable objects, return a wrapper that can read argument lists      
      def wrapper( *args, **kwargs):
	lensvar = kwargs.pop('lensvar',True)
	res = [f(*args, **kwargs) for f in fs]
	if lensvar and all([isinstance(f, Variable) for f in res]):
	  if len(res) == len(set([f.name for f in res])): 
	    return Ensemble(*res, basetype=Variable, idkey='name')
	  elif len(res) == len(set([f.dataset.name for f in res if f.dataset is not None])): 
	    for f in res: f.dataset_name = f.dataset.name 
	    return Ensemble(*res, idkey='dataset_name') #basetype=Variable, 
	  elif all([isinstance(f, Variable) for f in res]):
	    for f,m in zip(res,self.members): f.dataset_name = getattr(m,self.idkey)
	    return Ensemble(*res, idkey='dataset_name') #basetype=Variable,
	  else: raise VariableError
	else: 
	  return res      
      # return function wrapper
      return wrapper
    
  def __str__(self):
    ''' Built-in method; we just overwrite to call 'prettyPrint()'. '''
    return self.prettyPrint(short=False) # print is a reserved word  

  def prettyPrint(self, short=False):
    ''' Print a string representation of the Ensemble. '''
    if short:      
      string = '{0:s} {1:s}'.format(self.__class__.__name__,self.name)
      string += ', {:2d} Members ({:s})'.format(len(self.members),self.basetype.__name__)
    else:
      string = '{0:s}   {1:s}\n'.format(self.__class__.__name__,str(self.__class__))
      string += 'Name: {0:s},  '.format(self.name)
      string += 'Title: {0:s}\n'.format(self.title)
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
  
  def __getitem__(self, member):
    ''' Yet another way to access members by name... conforming to the container protocol. '''
    if not isinstance(member, basestring): raise TypeError
    if not self.hasVariable(member): raise KeyError
    return self.__dict__[member]
  
  def __setitem__(self, name, member):
    ''' Yet another way to add a member, this time by name... conforming to the container protocol. '''
    idkey = getattr(member,self.idkey)
    if idkey != name: raise KeyError, "The member ID '{:s}' is not consistent with the supplied key '{:s}'".format(idkey,name)
    return self.addMember(member) # add member
    
  def __delitem__(self, member):
    ''' A way to delete members by name... conforming to the container protocol. '''
    if not isinstance(varname, basestring): raise TypeError
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
    return self # return self as result

  def __isub__(self, member):
    ''' Remove a Dataset to an existing Ensemble. '''      
    if isinstance(member, basestring) and self.hasMember(member):
      assert self.removeMember(member), "A proble occurred removing Dataset '{:s}' from Ensemble.".format(member)    
    elif isinstance(member, self.basetype):
      assert self.removeMember(member), "A proble occurred removing Dataset '{:s}' from Ensemble.".format(member.name)
    elif isinstance(member, Variable):
      assert all(self.removeVariable(member)), "A problem occurred removing Variable '{:s}' from Ensemble Members.".format(member.name)    
    return self # return self as result

  
## run a test    
if __name__ == '__main__':

  pass