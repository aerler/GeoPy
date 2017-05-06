'''
Created on 2013-08-23

A module that provides access to NetCDF datasets for GeoData datasets and variables.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import collections as col
import netCDF4 as nc # netcdf python module
import os, functools

# import all base functionality from PyGeoDat
# from nctools import * # my own netcdf toolkit
from geodata.base import Variable, Axis, Dataset, ApplyTestOverList
from geodata.misc import checkIndex, isEqual, joinDicts
from geodata.misc import DatasetError, DataError, AxisError, NetCDFError, PermissionError, FileError, VariableError, ArgumentError 
from utils.nctools import coerceAtts, writeNetCDF, add_var, add_coord, checkFillValue


def asVarNC(var=None, ncvar=None, mode='rw', axes=None, deepcopy=False, **kwargs):
  ''' Simple function to cast a Variable instance as a VarNC (NetCDF-capable Variable subclass). '''
  # figure out axes
  if axes:
    if isinstance(axes,dict):
      # use dictionary entries to replace axes of the same name with the new ones
      axes = [axes[ax.name] if ax.name in axes else ax for ax in var.axes] 
    elif isinstance(axes,(list,tuple)): 
      # just use the new set of axes
      if not len(axes) == var.ndim: raise AxisError
      if var.shape is not None and not var.shape == tuple([len(ax) for ax in axes]): raise AxisError  
    else: 
      raise TypeError, "Argument 'axes' has to be of type dict, list, or tuple."
  else: axes = var.axes
  # create new VarNC instance (using the ncvar NetCDF Variable instance as file reference)
  if not isinstance(var,Variable): raise TypeError
  if not isinstance(ncvar,(nc.Variable,nc.Dataset)): raise TypeError
  atts = kwargs.pop('atts',var.atts.copy()) # name and units are also stored in atts!
  plot = kwargs.pop('plot',var.plot.copy())
  varnc = VarNC(ncvar, axes=axes, atts=atts, plot=plot, dtype=var.dtype, mode=mode, **kwargs)
  # copy data
  if var.data: varnc.load(data=var.getArray(unmask=False,copy=deepcopy))
  # return VarNC
  return varnc

def asAxisNC(ax=None, ncvar=None, mode='rw', deepcopy=True, **kwargs):
  ''' Simple function to cast an Axis instance as a AxisNC (NetCDF-capable Axis subclass). '''
  # create new AxisNC instance (using the ncvar NetCDF Variable instance as file reference)
  if not isinstance(ax,Axis): raise TypeError
  if not isinstance(ncvar,(nc.Variable,nc.Dataset)): raise TypeError # this is for the coordinate variable, not the dimension
  # axes are handled automatically (self-reference)  )
  atts = kwargs.pop('atts',ax.atts.copy()) # name and units are also stored in atts!
  plot = kwargs.pop('plot',ax.plot.copy())
  axisnc = AxisNC(ncvar, atts=atts, plot=plot, length=len(ax), coord=ax.coord, dtype=ax.dtype, 
                  mode=mode, **kwargs)
  # return AxisNC
  return axisnc

def asDatasetNC(dataset=None, ncfile=None, mode='rw', deepcopy=False, writeData=True, ncformat='NETCDF4', zlib=True, **kwargs):
  ''' Simple function to copy a dataset and cast it as a DatasetNetCDF (NetCDF-capable Dataset subclass). '''
  if not isinstance(dataset,Dataset): raise TypeError
  if not (mode == 'w' or mode == 'r' or mode == 'rw' or mode == 'wr'):  raise PermissionError
  # create NetCDF file
  ncfile = writeNetCDF(dataset, ncfile, ncformat=ncformat, zlib=zlib, writeData=writeData, close=False)
  # initialize new dataset - kwargs: varlist, varatts, axes, check_override, atts
  atts = kwargs.pop('atts',dataset.atts.copy()) # name and title are also stored in atts!
  newset = DatasetNetCDF(dataset=ncfile, atts=atts, mode=mode, ncformat=ncformat, **kwargs)
  # copy axes data/coordinates
  for axname,ax in dataset.axes.iteritems():
    if ax.data: newset.axes[axname].coord = ax.getArray(unmask=False, copy=True)
  # copy variable data
  for varname,var in dataset.variables.iteritems():
    if var.data: newset.variables[varname].load(data=var.getArray(unmask=False, copy=deepcopy))
  # return dataset
  return newset


class NoNetCDF(object):
  ''' Decorator class for Variable methods that don't work with VarNC instances, and thus have to return
      a regular Variable copy. '''
  def __init__(self, op):
    ''' Save original methods. '''
    self.op = op
  def __call__(self, ncvar, asNC=False, asVar=True, linplace=False, **kwargs):
    ''' Perform sanity checks, then execute operation, and return result. '''
    # input checks
    if linplace or asNC: raise NotImplementedError, "This operation does not work on VarNC instances!"
    # create regular Variable instance
    if not ncvar.data: ncvar.load() # load data
    nonc = ncvar.copy(asNC=False, deepcopy=False)
    # apply operation (assumed to have linplace keyword option)
    var = self.op(nonc, linplace=True, asVar=asVar, **kwargs)
    assert not isinstance(var, VarNC)
    # check for invalid returns (e.g. from applying arithmetic to strings)
    if var is None: return None # return immediately (invalid operation)
    if asVar and not isinstance(var,Variable): raise TypeError
    # return function result
    return var
  def __get__(self, instance, klass):
    ''' Support instance methods. This is necessary, so that this class can be bound to the parent instance. '''
    # N.B.: similar implementation to 'partial': need to return a callable that behaves like the instance method
    return functools.partial(self.__call__, instance) # but using 'partial' is simpler

class VarNC(Variable):
  '''
    A variable class that implements access to data from a NetCDF variable object.
  '''
  
  def __init__(self, ncvar, name=None, units=None, axes=None, data=None, dtype=None, scalefactor=1, 
               offset=0, transform=None, atts=None, plot=None, fillValue=None, mode='r', load=False, 
               squeeze=False, slices=None):
    ''' 
      Initialize Variable instance based on NetCDF variable.
      
      New Instance Attributes:
        mode = 'r' # a string indicating whether read ('r') or write ('w') actions are intended/permitted
        ncvar = None # the associated netcdf variable
        scalefactor = 1 # linear scale factor w.r.t. values in netcdf file
        offset = 0 # constant offset w.r.t. values in netcdf file
        transform = None # function that can perform non-trivial transforms upon load
        squeezed = False # if True, all singleton dimensions in NetCDF Variable are silently ignored
        slices = None # slice with respect to NetCDF Variable
    '''
    # check mode
    if not (mode == 'w' or mode == 'r' or mode == 'rw' or mode == 'wr'):  raise PermissionError  
    # write-only actions
    if isinstance(ncvar,nc.Dataset):
      #if 'w' not in mode: mode += 'w'
      if name is None and isinstance(atts,dict): name = atts.get('name',None)      
      dims = [ax if isinstance(ax,basestring) else ax.name for ax in axes] # list axes names
      dimshape = [None if isinstance(ax,basestring) else len(ax) for ax in axes]
      # construct a new netcdf variable in the given dataset and determine dtype
      if dtype is None and data is not None: dtype = data.dtype
      if name in ncvar.variables: 
        ncvar = ncvar.variable[name] # hope it is the right one...
        if dtype is None: dtype = ncvar.dtype
      else: 
        if dtype is None: raise TypeError, "No data (-type) to construct NetCDF variable!"
        ncvar = add_var(ncvar, name, dims=dims, shape=dimshape, atts=atts, dtype=dtype, fillValue=fillValue, zlib=True)
    elif isinstance(ncvar,nc.Variable):
      if dtype is None: dtype = ncvar.dtype
    if dtype is not None: dtype = np.dtype(dtype) # proper formatting
    # some type checking
    if not isinstance(ncvar,nc.Variable): raise TypeError, "Argument 'ncvar' has to be a NetCDF Variable or Dataset."        
    if data is not None and slices is None and data.shape != ncvar.shape: raise DataError
    if data is not None and slices is not None and len(slices) != data.ndim:
      raise DataError, "Data and slice have incompatible dimensions!"      
    lstrvar = False; strlen = None
    if dtype is not None: 
      if dtype.kind == 'S' and dtype.itemsize > 1:
        lstrvar = ncvar.dtype == np.dtype('|S1')
        strlen = ncvar.shape[-1] # last dimension
      elif not np.issubdtype(ncvar.dtype,dtype):
        if 'scale_factor' not in ncvar.ncattrs(): # data is not being scaled in NetCDF module
          raise DataError, "NetCDF data dtype does not match Variable dtype (ncvar.dtype={:s})".format(ncvar.dtype)
    # read actions
    if 'r' in mode: 
      # construct attribute dictionary from netcdf attributes
      ncatts = { key : ncvar.getncattr(key) for key in ncvar.ncattrs() }
      fillValue = ncatts.pop('_FillValue', fillValue) # this value should always be removed
      for key in ['scale_factor', 'add_offset']: ncatts.pop(key,None) # already handled by NetCDf Python interface
      # update netcdf attributes with custom override
      if atts is not None: ncatts.update(atts)
      # handle some netcdf conventions
      if name is None: name = ncatts.get('name',ncvar._name) # name in attributes has precedence
      else: ncatts['name'] = name
      if units is None: units = ncatts.get('units','') # units are not mandatory
      else: ncatts['units'] = units
      # construct axes, based on netcdf dimensions
      if axes is None: 
        if lstrvar: axes = tuple([str(dim) for dim in ncvar.dimensions[:-1]]) # omit last dim
        else: axes = tuple([str(dim) for dim in ncvar.dimensions]) # have to get rid of unicode
      elif lstrvar:
        if len(ncvar.shape[:-1]) != len(axes) and len(ncvar.shape[:-1]) != len(slices): raise AxisError,ncvar
        if ncvar.shape[-1] != dtype.itemsize: raise AxisError, ncvar
        assert strlen == dtype.itemsize
      elif slices is not None: 
        if not isinstance(slices, (list,tuple)): raise TypeError
        elif not squeeze and ncvar.ndim != len(slices): raise AxisError, (slices,ncvar)
        elif squeeze and len([l for l in ncvar.shape if l > 1]) != len(slices): raise AxisError, (slices,ncvar)
      else:
        axshape = tuple(ax._len for ax in axes)
        # N.B.: Because this constructor is also used in Axis initialization, and the axis of an Axis 
        #       is the Axis itself, not all class attributes are initialized yet, but self._len is!
        if ncvar.ndim != len(axes) or ncvar.shape != axshape:
          sqshape = tuple(axl for axl in ncvar.shape if axl > 1)
          if len(sqshape) != len(axshape) or sqshape != axshape: 
            raise AxisError, ncvar          
      # N.B.: slicing with index lists can change the shape
    else: ncatts = atts
    # check transform
    if transform is not None and not callable(transform): raise TypeError
    # call parent constructor
    super(VarNC,self).__init__(name=name, units=units, axes=axes, data=None, dtype=dtype, 
                               mask=None, fillValue=fillValue, atts=ncatts, plot=plot)
    # assign special attributes
    self.__dict__['ncvar'] = ncvar
    self.__dict__['mode'] = mode
    self.__dict__['offset'] = offset
    self.__dict__['scalefactor'] = scalefactor
    self.__dict__['transform'] = transform
    self.__dict__['squeezed'] = False
    self.__dict__['slices'] = slices # initial default (i.e. everything)
    assert self.strvar == lstrvar
    assert self.strlen == strlen
    if squeeze: self.squeeze() # may set 'squeezed' to True
    # handle data
    if load and data is not None: raise DataError, "Arguments 'load' and 'data' are mutually exclusive, i.e. only one can be used!"
    elif load and 'r' in self.mode: self.load(data=None) # load data from file
    # N.B.: load will automatically load teh specified slice
    elif data is not None and 'w' in self.mode: self.load(data=data) # load data from array
    # sync?
    if 'w' in self.mode: self.sync() 
  
  def __getitem__(self, slcs):
    ''' Method implementing access to the actual data; if data is not loaded, give direct access to NetCDF file. '''
    # determine what to do
    if self.data:
      # call parent method
      data = super(VarNC,self).__getitem__(slcs) # load actual data using parent method      
    else:
      # provide direct access to netcdf data on file
      if isinstance(slcs,(list,tuple)):
        if (not self.strvar and len(slcs) != self.ncvar.ndim) or (self.strvar and len(slcs)+1 != self.ncvar.ndim): 
          raise AxisError(slcs)
        slcs = list(slcs) # need to insert items
        # NetCDF can't deal wit negative list indices
        for i,slc in enumerate(slcs):
          lendim = self.ncvar.shape[i] # add dimension length to negative values
          if isinstance(slc,(list,tuple)):
            slcs[i] = [idx+lendim if idx < 0 else idx for idx in slc]
          elif isinstance(slc,np.ndarray):
            slcs[i] = np.where(slc<0,slc+lendim,slc) 
      else: 
        slcs = [slcs,]*self.ndim # trivial case: expand slices to all axes
      # handle squeezed vars
      if self.squeezed:
        # figure out slices
        for i in xrange(self.ncvar.ndim):
          if self.ncvar.shape[i] == 1: slcs.insert(i, 0) # '0' automatically squeezes out this dimension upon retrieval
      # check for existing slicing directive
      if self.slices:
        assert isinstance(self.slices,(list,tuple)) and isinstance(slcs,list)
        # substitute None-slices with the preset slicing directive
        slcs = [sslc if oslc == slice(None) else oslc for oslc,sslc in zip(slcs,self.slices)]
      # finally, get data!
      data = self.ncvar.__getitem__(slcs) # exceptions handled by netcdf module
      if self.dtype is not None and not np.issubdtype(data.dtype,self.dtype):
        if 'scale_factor' in self.ncvar.ncattrs():
          self.dtype = data.dtype # data was scaled automatically in NetCDF module
          if isinstance(data,np.ma.MaskedArray): self.fillValue = data.fill_value # possibly scaled
        else: 
          raise DataError, "NetCDF data dtype does not match Variable dtype (ncvar.dtype={:s})".format(self.ncvar.dtype) 
      if self.strvar: data = nc.chartostring(data)
      #assert self.ndim == data.ndim # make sure that squeezing works!
      # N.B.: the shape and even dimension number can change dynamically when a slice is loaded, so don't check for that, or it will fail!
      # apply transformations (try in-place first)
      if self.scalefactor != 1: 
        try: data *= self.scalefactor
        except TypeError: data = data * self.scalefactor
      if self.offset != 0: 
        try: data += self.offset
        except TypeError: data = data + self.offset
      if self.transform is not None: 
        data = self.transform(data=data, var=self, slc=slcs)
      # make sure dtype is correct, since it may have changed
      if not np.issubdtype(data.dtype,self.dtype): self.dtype = data.dtype
    # return data
    return data
  
  def slicing(self, lidx=None, lrng=None, years=None, listAxis=None, asVar=None, lsqueeze=True, 
              lcheck=False, lcopy=False, lslices=False, linplace=False, asNC=None, **axes):
    ''' This method implements access to slices via coordinate values and returns Variable objects. 
        Default behavior for different argument types: 
          - index by coordinate value, not array index, except if argument is a Slice object
          - interprete tuples of length 2 or 3 as ranges (passes to linspace)
          - treat lists and arrays as coordinate lists (can specify new list axis)
          - None values are accepted and indicate the entire range (i.e. no slicing) 
        Type-based defaults are ignored if appropriate keyword arguments are specified. 
        N.B.: this VarNC implementation will by default return another VarNC object, 
              referencing the original NetCDF variable, but with a new slice. '''
    newvar,slcs = super(VarNC,self).slicing(lidx=lidx, lrng=lrng, years=years, listAxis=listAxis, 
                                        asVar=asVar, lsqueeze=lsqueeze, lcheck=lcheck, 
                                        lcopy=lcopy, lslices=True, linplace=linplace, **axes)
    # transform sliced Variable into VarNC
    asNC = isinstance(newvar,Variable) and not linplace if asNC is None else asNC
    if asNC:
      #for ax in newvar.axes: ax.unload() # will retain its slice, just for test
      axes = []
      for newax,slc in zip(newvar.axes,slcs):
        if self.hasAxis(newax.name):
          ncax = self.getAxis(newax.name) # transform to sliced NetCDF
          if isinstance(ncax,AxisNC):
            axes.append(asAxisNC(newax, ncvar=ncax.ncvar, mode=ncax.mode, slices=(slc,)))
          else: axes.append(newax) # keep as is
        else: axes.append(newax) # this can be a coordinate list axis
      # figure out slices
      
      # create new VarNC instance with different slices
      newvar = asVarNC(newvar, self.ncvar, mode=self.mode, axes=axes, slices=slcs, squeeze=lsqueeze,
                       scalefactor=self.scalefactor, offset=self.offset, transform=self.transform)
    # N.B.: the copy method can also cast as VarNC and it is called in slicing; however, slicing
    #       can not communicate slices correctly, so that casting as VarNC has to happen here
    if lslices: return newvar, slcs
    else: return newvar
  
  def getArray(self, idx=None, axes=None, broadcast=False, unmask=False, fillValue=None, dtype=None,  copy=True):
    ''' Copy the entire data array or a slice; option to unmask and to reorder/reshape to specified axes. '''
    # use __getitem__ to get slice
    if not self.data: self.load()       
    return super(VarNC,self).getArray(idx=idx, axes=axes, broadcast=broadcast, unmask=unmask, dtype=dtype, 
                                      fillValue=fillValue, copy=copy) # just call superior
   
  def squeeze(self, **kwargs):
    ''' A method to remove singleton dimensions; special handling of __getitem__() is necessary, 
        because NetCDF Variables cannot be squeezed directly. '''
    self.squeezed = True
    return super(VarNC,self).squeeze(**kwargs) # just call superior  
  
  def copy(self, asNC=None, deepcopy=False, **newargs):
    ''' A method to copy the Variable with just a link to the data.
        N.B.: if we return a VarNC object, it will be attached to the same NetCDF file/variable;
              to get a new NetCDF file with new variables, create a DatasetNetCDF instance with a 
              new NetCDF file (e.g. using writeNetCDF), and add the new variable to the new dataset, 
              using addVariable with the asNC=True and the copy=True / deepcopy=True option '''
    if deepcopy and not self.data: self.load() # need data for deepcopy 
    copyvar = super(VarNC,self).copy(deepcopy=deepcopy, **newargs) # just call superior - returns a regular Variable instance
    if asNC is None: asNC = not deepcopy and not 'data' in newargs 
    # N.B.: copy as VarNC, if no deepcopy and no data provided, otherwise as regular Variable;
    #       this method is also called in slicing, but since sliced data is passed without a slice-
    #       argument, it has to be casted as a regular Variable and converted later
    if asNC: 
      if 'scalefactor' not in newargs: newargs['scalefactor'] = self.scalefactor
      if 'transform' not in newargs: newargs['transform'] = self.transform
      if 'offset' not in newargs: newargs['offset'] = self.offset
      if 'slices' not in newargs: newargs['slices'] = self.slices
      copyvar = asVarNC(var=copyvar, ncvar=self.ncvar, mode=self.mode, **newargs)
    else:
      if not copyvar.data and not 'data' in newargs: 
        copyvar.load(self.__getitem__(None)) # don't want to loose data, unless it was explicitly reset 
    return copyvar
  
  # some methods that don't work with VarNC's and need to return regular Variables: basically everything
  # that changes the axes/shape (implemented through NoNetCDF-decorator, above)
  @NoNetCDF
  def reorderAxes(self, axes=None, asVar=True, linplace=False, lcheckAxis=False):
    return Variable.reorderAxes(self, axes=axes, asVar=asVar, linplace=linplace, lcheckAxis=lcheckAxis)
  @NoNetCDF
  def insertAxis(self, axis=None, iaxis=0, length=None, req_axes=None, asVar=True, lcheckVar=None, 
                 lcheckAxis=True, lstrict=False, linplace=False, lcopy=True, ltile=True):
    return Variable.insertAxis(self, axis=axis, iaxis=iaxis, length=length, req_axes=req_axes, asVar=asVar, 
                               lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, lstrict=lstrict, linplace=linplace, 
                               lcopy=lcopy, ltile=ltile)
  @NoNetCDF
  def insertAxes(self, new_axes=None, req_axes=None, asVar=True, lcheckVar=None, lcheckAxis=True, 
                 lstrict=False, linplace=False, lcopy=True, ltile=True):
    return Variable.insertAxes(self, new_axes=new_axes, req_axes=req_axes, asVar=asVar, 
                               lcheckVar=lcheckVar, lcheckAxis=lcheckAxis, lstrict=lstrict, linplace=linplace, 
                               lcopy=lcopy, ltile=ltile)
  @NoNetCDF
  def mergeAxes(self, axes=None, new_axis=None, axatts=None, asVar=True, linplace=False, 
                lcheckAxis=True, lcheckVar=None, lvarall=True, ldsall=False, lstrict=False):
    return Variable.mergeAxes(self, axes=axes, new_axis=new_axis, axatts=axatts, asVar=asVar, linplace=linplace, 
                              lcheckAxis=lcheckAxis, lcheckVar=lcheckVar, lvarall=lvarall, ldsall=ldsall, lstrict=lstrict)
#   def mergeAxes(self, axes=None, new_axis=None, axatts=None, asVar=True, linplace=False, asNC=False, 
#     lcheckAxis=False, lcheckVar=None, lall=True):
#     ''' this doesn't work with NetCDF Variables, so we need to make a regular copy '''
#     if asNC: raise NotImplementedError, "Merging axes doesn't work with NetCDF Variables."
#     if not self.data: self.load()
#     nonc = self.copy(asNC=False, deepcopy=False)
#     nonc = nonc.mergeAxes(axes=axes, new_axis=new_axis, axatts=axatts, asVar=asVar, 
#                           linplace=linplace, lcheckAxis=lcheckAxis, lcheckVar=lcheckVar, lall=lall) 
#     return nonc
    
  def load(self, data=None, **kwargs):
    ''' Method to load data from NetCDF file into RAM. '''
    slcs = self.slices
    # optional slicing
    if any([self.hasAxis(ax) for ax in kwargs.iterkeys()]):
      if slcs is not None: raise NotImplementedError, "Currently, VarNC instances can only be sliced once."
      # extract axes; remove axes from kwargs to avoid slicing again in super-call
      axes = {ax:kwargs.pop(ax) for ax in kwargs.iterkeys() if self.hasAxis(ax)}
      if len(axes) > 0: 
        self, slcs = self.slicing(asVar=True, lslices=True, linplace=True, **axes) # this is poorly tested...
        if data is not None and data.shape != self.shape: data = data.__getitem__(slcs) # slice input data, if appropriate 
    if data is None:
      if self.data: 
        return self # do nothing         
      else: # use slices to load data
        if slcs is None: slcs = slice(None)
        data = self.__getitem__(slcs) # load everything
    elif isinstance(data,np.ndarray):
      data = data
    elif all(checkIndex(data)):
      if isinstance(data,(list,tuple)):
        if len(data) != len(self.shape): 
          raise IndexError, 'Length of index tuple has to equal to the number of dimensions!'       
        for ax,idx in zip(self.axes,data): ax.coord = idx
        data = self.__getitem__(data) # load slice
      else: 
        if self.ndim != 1: raise IndexError, 'Multi-dimensional variable have to be indexed using tuples!'
        if self != self.axes[0]: ax.coord = data # prevent infinite loop due to self-reference
        data = self.__getitem__(idx=data) # load slice
    else: 
      raise TypeError
    # load data and return itself (this allows for some convenient syntax)
    return super(VarNC,self).load(data=data, **kwargs) # load actual data using parent method    
    
  def sync(self):
    ''' Method to make sure, data in NetCDF variable and Variable instance are consistent. '''
    ncvar = self.ncvar
    # update netcdf variable    
    if 'w' in self.mode:
      if self.strvar and ncvar.shape[:-1] == self.shape: pass
      elif not self.squeezed and ncvar.shape == self.shape: pass
      elif self.squeezed and tuple([n for n in ncvar.shape if n > 1]) == self.shape: pass
      else: 
        raise NetCDFError, "Cannot write to NetCDF variable: array shape in memory and on disk are inconsistent!"
      if self.data:
        fillValue = self.fillValue
        # special handling of some data types
        if isinstance(self.data_array,np.bool_): 
          ncvar[:] = self.data_array.astype('i1') # cast boolean as 8-bit integers
          if fillValue is not None: fillValue = 1 if fillValue else 0
        elif self.strvar:
          ncvar[:] = nc.stringtochar(self.data_array) # transform string array to char array with one more dimension
          if fillValue is not None: raise NotImplementedError
        else: ncvar[:] = self.data_array # masking should be handled by the NetCDF module
        # reset scale factors etc.
        self.scalefactor = 1; self.offset = 0
        fillValue = checkFillValue(fillValue, self.dtype)
        if fillValue is not None:
          ncvar.setncattr('missing_value',fillValue) 
      # update NetCDF attributes
      ncvar.setncatts(coerceAtts(self.atts))
      ncattrs = ncvar.ncattrs() # list of current NC attributes
      ncvar.set_auto_maskandscale(True) # automatic handling of missing values and scaling and offset
      if 'scale_factor' in ncattrs: ncvar.delncattr('scale_factor',ncvar.getncattr('scale_factor'))
      if 'add_offset' in ncattrs: ncvar.delncattr('add_offset',ncvar.getncattr('add_offset'))
      # set other attributes like in variable
      ncvar.setncattr('name',self.name)
      ncvar.setncattr('units',self.units)
      # now sync dataset
      ncvar.group().sync()     
    else: 
      raise PermissionError, "Cannot write to NetCDF variable: writing (mode = 'w') not enabled!"
    # for convenience...
    return self
     
  def unload(self):
    ''' Method to sync the currently loaded data to file and free up memory (discard data in memory) '''
    # synchronize data with NetCDF file
    if 'w' in self.mode: self.sync() # only if we have write permission, of course
    # discard NetCDF Variable object (contains a reference to the data)
    ncds = self.ncvar.group(); ncname = self.ncvar._name # this is the actual netcdf name
    del self.ncvar; self.ncvar = ncds.variables[ncname] # reattach (hopefully without the data array)
    # discard data array the usual way
    super(VarNC,self).unload()
    # return itself- this allows for some convenient syntax
    return self

class AxisNC(Axis,VarNC):
  '''
    A NetCDF Variable representing a coordinate axis.
  '''
  
  def __init__(self, ncvar, name=None, length=0, coord=None, dtype=None, atts=None, fillValue=None, mode='r', load=None, **axargs):
    ''' Initialize a coordinate axis with appropriate values. '''
    if isinstance(ncvar,nc.Dataset):
      if 'w' not in mode: mode += 'w'
      if name is None and isinstance(atts,dict): name = atts.pop('name',None) 
      # construct a new netcdf coordinate variable in the given dataset
      if isinstance(ncvar,nc.Dataset) and name in ncvar.variables: ncvar = ncvar.variables[name] 
      else: ncvar = add_coord(ncvar, name, length=length, data=coord, dtype=dtype, fillValue=fillValue, atts=atts, zlib=True)    
    if length == 0: length = ncvar.shape[0] # necessary to allow shape checks during creation
    if load is None: load = True if coord is None else False 
    # initialize as an Axis subclass and pass arguments down the inheritance chain
    super(AxisNC,self).__init__(ncvar=ncvar, name=name, length=length, coord=coord, dtype=dtype, atts=atts, 
                                fillValue=fillValue, mode=mode, load=load, **axargs)
    # synchronize coordinate array with netcdf variable
    if 'w' in mode: self.sync()    
      

class DatasetNetCDF(Dataset):
  '''
    A Dataset Class that provides access to variables in one or more NetCDF files. The class supports reading
    and writing, as well as the creation of new NetCDF files.
  '''
  
  def __init__(self, name=None, title=None, dataset=None, filelist=None, varlist=None, variables=None,
      	       varatts=None, atts=None, axes=None, multifile=False, check_override=None, ignore_list=None, 
               folder='', mode='r', ncformat='NETCDF4', squeeze=True, load=False, check_vars=None):
    ''' 
      Create a Dataset from one or more NetCDF files; Variables are created from NetCDF variables. 
      Alternatively, create a netcdf file from an existing Dataset (Variables can be added as well).  
      Arguments:
        name           : a simple name for the dateset (string)
        title          : name, formatted for printing (string
        dataset        : an existing NetCDF dataset (instead of file list) or an existing PyGeoDat Dataset, if mode = 'w'
        filelist       : a list/tuple of NetCDF files (relative for folder, see below); may be created, if mode = 'w'
        varlist        : if mode = 'r', a list of variables to be loaded, if mode = 'w', a list/tuple of existing PyGeoDat Variables  
        varatts        : dict of dicts with arguments for the Variable/Axis constructor (for each variable/axis) 
        atts           : dict with attributes for the new dataset
        axes           : list/tuple of axes to use (Axis or AxisNC); overrides axes of same name in NetCDF file 
        multifile      : open file list using the netCDF4 MFDataset multi-file option (logical)
        check_override : overrides consistency check for axes of same name for listed names (list/tuple of strings) 
        ignore_list    : ignore listed variables and dimensions and any variables that depend on listed dimensions (list/tuple/set of strings; original names)
        folder         : root folder for file list (string); this path is prepended to all filenames
        mode           : file mode: whether read ('r') or write ('w') actions are intended/permitted (string; passed to netCDF4.Dataset)
        ncformat       : format of NetCDF file, i.e. NETCDF3 NETCDF4 or NETCDF_CLASSIC (string; passed to netCDF4.Dataset)
        squeeze        : squeeze singleton dimensions from all variables
        load           : load data from disk immediately (passed on to VarNC)
                       
      NetCDF Attributes:
        mode           = 'r' # a string indicating whether read ('r') or write ('w') actions are intended/permitted
        datasets       = [] # list of NetCDF datasets
        dataset        = @property # shortcut to first element of self.datasets
        filelist       = [] # files used to create datasets (absolute path)
      Basic Attributes:        
        variables      = dict() # dictionary holding Variable instances
        axes           = dict() # dictionary holding Axis instances (inferred from Variables)
        atts           = AttrDict() # dictionary containing global attributes / meta data
    '''
    if len(folder) > 0 and folder[-1] != '/': folder += '/'
    if variables is None:
      # either use available NetCDF datasets directly, ...  
      if isinstance(dataset,nc.Dataset):
        datasets = [dataset]  # datasets is used later
        #if hasattr(dataset,'filepath'): filelist = [dataset.filepath()] # only available in newer versions
        # N.B.: apparently filepath() tends to cause the netCDF library to crash... need to find a workaround...
      elif isinstance(dataset,(list,tuple)):
        if not all([isinstance(ds,nc.Dataset) for ds in dataset]): raise TypeError
        datasets = dataset
        #filelist = [dataset.filepath() for dataset in datasets if hasattr(dataset,'filepath')]
        # N.B.: apparently filepath() tends to cause the netCDF library to crash... need to find a workaround...
      # ... create a new NetCDF file, ...
      elif isinstance(mode,basestring) and 'w' == mode and filelist:    
        if isinstance(filelist,col.Iterable): filelist = filelist[0]
        filename = folder + filelist; filelist = [filename] # filelist is used later
        if os.path.exists(filename): raise NetCDFError, "File '%s' already exits - aborting!"%filename
        if dataset: # add variables in dataset
          if not isinstance(dataset,Dataset): raise TypeError        
          dataset.atts.update(coerceAtts(atts))
        else: # if no dataset is provided, make one
          dataset = Dataset(varlist=[], atts=atts)
        if axes: # add remaining axes
          if not isinstance(axes,(list,tuple)): raise TypeError
          for ax in axes:
            if not isinstance(ax,Axis): raise TypeError 
            dataset.addAxis(ax)
        if varlist: # add remaining variables  
          if not isinstance(varlist,(list,tuple)): raise TypeError
          for var in varlist: 
            if not isinstance(var,Variable): raise TypeError
            dataset.addVariable(var)      
        # create netcdf dataset/file
        dataset = writeNetCDF(dataset, filename, ncformat='NETCDF4', zlib=True, writeData=False, close=False, feedback=False)
        datasets = [dataset]
      # ... or open datasets from filelist
      else:
        # translate modes
        ncmode = 'a' if 'r' in mode and 'w' in mode else mode # 'rw' -> 'a' for "append"     
        # open netcdf datasets from netcdf files
        if not isinstance(filelist,col.Iterable): raise TypeError, filelist
        # check if file exists
        for filename in filelist:
          if not os.path.exists(folder+filename): 
            raise FileError, "File {0:s} not found in folder {1:s}".format(filename,folder)     
        datasets = []; filenames = []
        for ncfile in filelist:        
          try: # NetCDF4 error messages are not very helpful...
            if multifile: # open a netCDF4 multi-file dataset 
              if isinstance(ncfile,(list,tuple)): tmpfile = [folder+ncf for ncf in ncfile]
              else: tmpfile = folder+ncfile # multifile via regular expressions
              datasets.append(nc.MFDataset(tmpfile), mode=ncmode, format=ncformat, clobber=False)
            else: # open a simple single-file dataset
              tmpfile = folder+ncfile
              datasets.append(nc.Dataset(tmpfile, mode=ncmode, format=ncformat, clobber=False))
          except RuntimeError:
            raise NetCDFError, "Error reading file '{0:s}' in folder {1:s}".format(ncfile,folder)
          filenames.append(tmpfile)
        filelist = filenames # original file list, absolute path        
      # from here on, dataset creation is based on the netcdf-Dataset(s) in 'datasets'
      # figure out per-dataset varatts and ignore_lists 
      if varatts is None: varatts_list = [dict()]*len(datasets) # empty dictionary means no parameters...
      elif isinstance(varatts,dict): varatts_list = [varatts]*len(datasets)
      elif isinstance(varatts,(tuple,list)):
        if len(varatts) != len(datasets): 
          raise ArgumentError("{} != {}".format(len(varatts),len(datasets)))
        else: varatts_list = varatts
      else: 
        raise TypeError("'varatts' has to be a dictionary or a list of dictionaries; found: {}".format(varatts))
      if ignore_list is None: ignore_lists  = [set()]*len(datasets) # empty set means no none...
      elif isinstance(ignore_list,set): ignore_lists = [ignore_list]*len(datasets)
      elif isinstance(ignore_list,(tuple,list)):
        if all(isinstance(e,basestring) for e in ignore_list):
          ignore_lists = [set((ignore_list,))]*len(datasets)
        elif all(isinstance(e,(list,tuple,set)) for e in ignore_list):
          ignore_lists = [set(ignore_list)]*len(datasets)
        else:
          raise TypeError("'ignore_list' has to be a set or a list of sets; found: {}".format(ignore_lists))
      else: 
        raise TypeError("'ignore_list' has to be a set or a list of sets; found: {}".format(ignore_lists))
      if len(ignore_lists) != len(datasets): 
        raise ArgumentError("{} != {}".format(len(ignore_lists),len(datasets)))
      # generate list of variables that have already been converted
      check_rename_list = []
      for varatts in varatts_list:
        check_rename = dict()
        for varname,varatt in varatts.items():
          if 'name' in varatt: # if name is not in varatt, there is no renaming, hence no need to record 
            check_rename[varatt['name']] = dict(units=varatt.get('units',''), old_name=varname)
        check_rename_list.append(check_rename)
      if check_override is None: check_override = [] # list of variables (and axes) that is not checked for consistency
      # N.B.: check_override may be necessary to combine datasets from different files with inconsistent axis instances 
      assert len(datasets) == len(varatts_list) == len(ignore_lists) == len(check_rename_list)
      # create axes from netcdf dimensions and coordinate variables
      if axes is None: axes = dict()
      else: check_override += axes.keys() # don't check externally provided axes   
      if not isinstance(axes,dict): raise TypeError
      for ds,varatts,ignore_list,check_rename in zip(datasets,varatts_list,ignore_lists,check_rename_list):
        for dim in ds.dimensions.keys():
          if dim not in ignore_list:
            if dim[:8] == 'str_dim_': pass # dimensions added to store strings as charater arrays        
            elif dim in ds.variables: # dimensions with an associated coordinate variable           
              if dim in axes: # if already present, make sure axes are essentially the same
                tmpax = AxisNC(ncvar=ds.variables[dim], mode='r', **varatts.get(dim,{})) # apply all correction factors...
                if dim not in check_override and not isEqual(axes[dim][:],tmpax[:]): 
                  raise DatasetError, "Error constructing Dataset: NetCDF files have incompatible {:s} dimension.".format(dim)
              else: # if this is a new axis, add it to the list
                if ds.variables[dim].dtype == '|S1': pass # Variables of type char are currently not implemented
                else: axes[dim] = AxisNC(ncvar=ds.variables[dim], mode=mode, **varatts.get(dim,{})) # also use overrride parameters
            else: # initialize dimensions without associated variable as regular Axis (not AxisNC)
              if dim in axes: # if already present, make sure axes are essentially the same
                if len(axes[dim]) != len(ds.dimensions[dim]): 
                  raise DatasetError, "Error constructing Dataset: NetCDF files have incompatible '{:s}' dimensions: {:d} != {:d}".format(dim,len(axes[dim]),len(ds.dimensions[dim])) 
              else: # if this is a new axis, add it to the list
                params = dict(name=dim,coord=np.arange(len(ds.dimensions[dim]))); params.update(varatts.get(dim,{}))
                axes[dim] = Axis(**params) # also use overrride parameters          
      # create variables from netcdf variables    
      variables = dict()
      if not isinstance(check_vars, (list,tuple)): check_vars = (check_vars,)
      for ds,varatts,ignore_list,check_rename in zip(datasets,varatts_list,ignore_lists,check_rename_list):
        # figure out desired variables
        dsvars = []
        for var in ds.variables.keys():
          if var in varatts:
            varatt = varatts[var]
            if 'name' in  varatt: alt = varatt['name']
            elif 'atts' in  varatt and 'name' in  varatt['atts']: alt = varatt['atts']['name']  
            else: alt = None
          else: alt = None
          if varlist is None:
            if var not in ignore_list and alt not in ignore_list: dsvars.append(var)
            # N.B.: ignored variables usually don't have varatts...
          else:   
            if var in varlist or alt in varlist: dsvars.append(var)
        # loop over variables in dataset
        for var in dsvars:
          ncvar= ds.variables[var]
          if var in axes: pass # do not treat coordinate variables as real variables 
          elif ncvar.ndim == 0: pass # also ignore scalars for now...
          elif var in variables: # if already present, make sure variables are essentially the same
            varobj = variables[var] 
            if var not in check_override:
              # check shape (don't load)
              if varobj.strvar and varobj.ndim == ncvar.ndim-1:
                if varobj.shape != ncvar.shape[:-1] or varobj.ncvar.dimensions != ncvar.dimensions:
                  raise DatasetError, "Error constructing Dataset: Variables '{:s}' from different files have incompatible dimensions.".format(var)
              else: 
                if varobj.shape != ncvar.shape or varobj.ncvar.dimensions != ncvar.dimensions:              
                  raise DatasetError, "Error constructing Dataset: Variables '{:s}' from different files have incompatible dimensions.".format(var)
                if 'units' in ncvar.ncattrs() and not ( varobj.ncvar.units == ncvar.units or varobj.units == ncvar.units ): # check units as well              
                  raise DatasetError, "Error constructing Dataset: Variables '{:s}' from different files have incompatible units.".format(var)
              # check values only of requested
              if var in check_vars:
                if np.any(varobj.ncvar[:] != ncvar[:]):              
                  raise DatasetError, "Error constructing Dataset: Variables '{:s}' from different files have incompatible values.".format(var)                
          else: # if this is a new variable, add it to the list
            ncunits = ncvar.units if hasattr(ncvar,'units') else ''
            if var in check_rename and ncunits == check_rename[var]['units']:
              # check both, name and units, to minimize confusion
              tmpatts = varatts[check_rename[var]['old_name']] # must be in varatts, too
              old_name = check_rename[var]['old_name'] # also store old name
              # N.B.: if variable has likely already been renamed, apply new attributes anyway,
              #       but remove scale factors and transforms, since the have already been applied
              for att in ('offset','scalefactor','transform'):
                  if att in tmpatts: del tmpatts[att]
            elif var in varatts: 
              tmpatts = varatts[var] # rename and apply new attributes
              old_name = var # store old name
            else: 
              tmpatts = dict(name=var, units=ncunits)
              old_name = '' # no old name
            if 'atts' in tmpatts: tmpatts['atts']['old_name'] = old_name
            else: tmpatts['atts'] = dict(old_name=old_name)
            if ncvar.dtype == '|S1' and all([dim in axes for dim in ncvar.dimensions[:-1]]): # string variable
              varaxes = [axes[dim] for dim in ncvar.dimensions[:-1]] # collect axes (except last)
              strtype = np.dtype('|S{:d}'.format(ncvar.shape[-1])) # string with length of string dimension
              # N.B.: apparently len(dim) does not work properly - ncvar.shape is more reliable
              # create new variable using the override parameters in varatts
              variables[tmpatts['name']] = VarNC(ncvar=ncvar, axes=varaxes, dtype=strtype, 
                                                 mode=mode, squeeze=squeeze, load=load, **tmpatts)
            elif all([dim in axes for dim in ncvar.dimensions]):
              varaxes = [axes[dim] for dim in ncvar.dimensions] # collect axes
              # create new variable using the override parameters in varatts
              variables[tmpatts['name']] = VarNC(ncvar=ncvar, axes=varaxes, 
                                                 mode=mode, squeeze=squeeze, load=load, **tmpatts)
              # N.B.: using tmpatts['name'] as key is more reliable in preventing duplicate variables,
              #       because it also works when NetCDF names are different across files
            elif not any([dim in ignore_list for dim in ncvar.dimensions]): # legitimate omission
              raise DatasetError, 'Error constructing Variable: Axes/coordinates not found:\n {:s}, {:s}'.format(str(var), str(ncvar.dimensions))
      variables = variables.values()
    else:
      if isinstance(variables,dict): variables = variables.values()
      if filelist is None: raise ArgumentError, filelist
      if folder: filelist = [folder+filename for filename in filelist]
      if isinstance(dataset,nc.Dataset):
        datasets = [dataset]  # datasets is used later
        #if hasattr(dataset,'filepath'): filelist = [dataset.filepath()] # only available in newer versions
        # N.B.: apparently filepath() tends to cause the netCDF library to crash... need to find a workaround...
        if len(filelist) != 1: raise ValueError, filelist
      elif isinstance(dataset,(list,tuple)):
        if not all([isinstance(ds,nc.Dataset) for ds in dataset]): raise TypeError
        datasets = dataset
        #filelist = [dataset.filepath() for dataset in datasets if hasattr(dataset,'filepath')]
        # N.B.: apparently filepath() tends to cause the netCDF library to crash... need to find a workaround...
        if len(filelist) == 0: raise ValueError, filelist
      else: raise ArgumentError, dataset
      mode = 'r' # for now, only allow read
    # get attributes from NetCDF dataset
    ncattrs = joinDicts(*[ds.__dict__ for ds in datasets])
    # update NC atts with attributes passed to constructor
    if atts is not None: ncattrs.update(atts) # update with attributes passed to constructor
    self.__dict__['mode'] = mode
    # add NetCDF attributes
    self.__dict__['datasets'] = datasets
    self.__dict__['filelist'] = filelist
    # initialize Dataset using parent constructor
    #if axes: axes = tuple(set(axes.values())) # same axis can have multiple names here
    super(DatasetNetCDF,self).__init__(name=name, title=title, varlist=variables, axes=None, atts=ncattrs)
    # N.B.: don't pass axes explicitly, otherwise we are adding a lot of unneccessary axes, which causes confusion
    #       (in particular, the singular Time axis from constant files will be loaded, which causes problems)
    # check that stuff was loaded
#     if len(self.variables) == 0 and mode != 'w': raise EmptyDatasetError
    # catch exception if an empty dataset is OK
    
  @property
  def dataset(self):
    ''' The first element of the datasets list. '''
    return self.datasets[0] 
  
  @ApplyTestOverList
  def addAxis(self, ax, asNC=None, copy=True, loverwrite=False, deepcopy=False):
    ''' Method to add an Axis to the Dataset. (If the Axis is already present, check that it is the same.) '''   
    if asNC is None: asNC = copy
    if ax.name in self.__dict__: 
      # replace axes, if permitted; need to use NetCDF method immediately, though
      if loverwrite and self.hasVariable(ax.name): 
        return self.replaceAxis(ax.name, ax, deepcopy=deepcopy)
      else: 
        raise AttributeError, "Cannot add Variable '{:s}' to Dataset, because an attribute of the same name already exits!".format(ax.name)      
    else:             
      if copy: # make a new instance or add it as is 
        # cast Axis instance as AxisNC (sort of implies copying)    
        if asNC and 'w' in self.mode: 
          ax = asAxisNC(ax=ax, ncvar=self.datasets[0], mode=self.mode, deepcopy=deepcopy)
        elif copy: 
          ax = ax.copy(deepcopy=deepcopy) # make a new instance or add it as is
      else:
          if asNC and not isinstance(ax,AxisNC): 
            raise ArgumentError, "Cannot create NC variable without copying!"
    # hand-off to parent method and return status
    return super(DatasetNetCDF,self).addAxis(ax, copy=False, loverwrite=loverwrite) # already copied above
      
  def replaceAxis(self, oldaxis, newaxis=None, asNC=True, deepcopy=True):    
    ''' Replace an existing axis with a different one and transfer NetCDF reference to new axis. '''
    if newaxis is None: 
      newaxis = oldaxis; oldaxis = newaxis.name # i.e. replace old axis with the same name
    # check axis
    if not self.hasAxis(oldaxis): raise AxisError
    # special treatment for VarNC: transfer of ownership of NetCDF variable
    if asNC or isinstance(newaxis,AxisNC):
      if isinstance(oldaxis,Axis): oldname = oldaxis.name # just go by name
      else: oldname = oldaxis
      oldaxis = self.axes[oldname]
      if len(oldaxis) != len(newaxis): raise AxisError # length has to be the same!
      # N.B.: the length of a dimension in a NetCDF file can't change!
      if oldaxis.data != newaxis.data: raise DataError # make sure data status is the same
      # remove old axis from dataset...
      self.removeAxis(oldaxis, force=True)
      # cast new axis as AxisNC and transfer old ncvar reference
      newaxis = asAxisNC(ax=newaxis, ncvar=oldaxis.ncvar, mode=oldaxis.mode, deepcopy=deepcopy)
      # ... and add new axis to dataset
      self.addAxis(newaxis, copy=False)
      # loop over variables with this axis    
      newaxis = self.axes[newaxis.name] # update reference
      for var in self.variables.values():
        if var.hasAxis(oldname): var.replaceAxis(oldname,newaxis)    
    else: # no need for special treatment...
      super(DatasetNetCDF,self).replaceAxis(oldaxis, newaxis)
    # return verification
    return self.hasAxis(newaxis)        
  
  @ApplyTestOverList
  def addVariable(self, var, asNC=None, copy=True, loverwrite=False, lautoTrim=False, deepcopy=False):
    ''' Method to add a new Variable to the Dataset. '''
    if asNC is None: asNC = copy and 'w' in self.mode
    if asNC and 'w' not in self.mode: 
        raise NetCDFError("Cannot add new NetCDF Variables in read-only mode; open in write mode.")
    if var.name in self.__dict__: 
      # replace axes, if permitted; need to use NetCDF method immediately, though
      if loverwrite and self.hasVariable(var.name): 
        return self.replaceVariable(var.name, var, deepcopy=deepcopy)
      else: raise AttributeError, "Cannot add Variable '{:s}' to Dataset, because an attribute of the same name already exits!".format(var.name)      
    else:             
      if deepcopy: copy=True   
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
      # cast Axis instance as AxisNC
      if copy: # make a new instance or add it as is 
        if asNC and 'w' in self.mode:
          for ax in var.axes:
            if not self.hasAxis(ax.name): 
              self.addAxis(ax, asNC=asNC, copy=copy, loverwrite=loverwrite, deepcopy=deepcopy)
          # add variable as a NetCDF variable             
          var = asVarNC(var=var,ncvar=self.datasets[0], axes=self.axes, mode=self.mode, deepcopy=deepcopy)
        else: 
          var = var.copy(deepcopy=deepcopy) # or just add as a normal Variable
      else:
        if asNC and not isinstance(var,VarNC): 
          raise ArgumentError, "Cannot create NC variable without copying!"
      # hand-off to parent method and return status
      check = super(DatasetNetCDF,self).addVariable(var, copy=False, loverwrite=loverwrite)
      assert var.dataset == self
      return check
  
  def replaceVariable(self, oldvar, newvar=None, asNC=False, deepcopy=False):
    ''' Replace an existing Variable with a different one and transfer NetCDF reference and axes. '''
    if newvar is None: 
      newvar = oldvar; oldvar = newvar.name # i.e. replace old var with the same name
    # check var
    if not self.hasVariable(oldvar): raise VariableError
    # special treatment for VarNC: transfer of ownership of NetCDF variable
    if asNC or isinstance(newvar,VarNC):
      # resolve names
      if isinstance(oldvar,Variable): oldname = oldvar.name # just go by name
      else: oldname = oldvar
      oldvar = self.variables[oldname]
      if oldvar.shape != newvar.shape: raise AxisError # shape has to be the same!
      # N.B.: the shape of a variable in a NetCDF file can't change!
      # remove old variable from dataset...
      self.removeVariable(oldvar) # this is actually the ordinary Dataset class method that doesn't do anything to the NetCDF file
      # cast new variable as VarNC and transfer old ncvar reference and axes    
      newvar = asVarNC(var=newvar,ncvar=oldvar.ncvar, axes=oldvar.axes, mode=oldvar.mode, deepcopy=deepcopy)
      # ... and add new axis to dataset
      self.addVariable(newvar, copy=False, loverwrite=False)
    else: # no need for special treatment...
      super(DatasetNetCDF,self).replaceVariable(oldvar, newvar)
    # return status of variable
    return self.hasVariable(newvar)  
  
#   def load(self, **slices):
#     ''' Load all VarNC's and AxisNC's using the slices specified as keyword arguments. '''
#     # make slices
#     for key,value in slices.iteritems():
#       if isinstance(value,col.Iterable): 
#         slices[key] = slice(*value)
#       else: 
#         if not isinstance(value,(int,np.integer)): raise TypeError
#     # load variables
#     for var in self.variables.values():
#       if isinstance(var,VarNC):
#         idx = [slices.get(ax.name,slice(None)) for ax in var.axes]
#         var.load(data=idx) # load slice, along with relevant dimensions
#     # no return value...     
#     # return itself- this allows for some convenient syntax
#     return self
    
  def sync(self):
    ''' Synchronize variables and axes/coordinates with their associated NetCDF variables. '''
    # only if writing is enabled
    if 'w' in self.mode:
      # sync coordinates with ncvars
      for ax in self.axes.values(): 
        if isinstance(ax,AxisNC): ax.sync() 
      # sync variables with ncvars
      for var in self.variables.values(): 
        if isinstance(var,VarNC): var.sync()
      # synchronize NetCDF datasets with file system
      for dataset in self.datasets: 
        dataset.setncatts(coerceAtts(self.atts)) # synchronize attributes with NetCDF dataset
        dataset.sync() # synchronize data
    else: 
      raise PermissionError, "Cannot write to NetCDF Dataset: writing (mode = 'w') not enabled!"
    
  def unload(self):
    ''' Method to sync the currently loaded dataset to file and free up memory (discard data in memory) '''
    # synchronize data with NetCDF file
    if 'w' in self.mode: self.sync() # only if we have write permission, of course
    # unload all variables
    super(DatasetNetCDF,self).unload()  
    # return itself- this allows for some convenient syntax
    return self
    
  def copy (self, asNC=True, filename=None, varsdeep=False, varargs=None, **newargs):
    ''' Copy a DatasetNetCDF, either into a normal Dataset or into a DatasetNetCDF (requires a filename). '''
    if asNC and filename is not None:
      writeData = newargs.pop('lwriteData',True)
      if writeData or varsdeep: self.load() 
      # N.B.: we need to pre-load, so the data can be written later
    # figure out how variables will be copied (NC, or not NC)   
    if varargs is None: varargs = dict()
    for varname,var in self.variables.iteritems():
      if isinstance(var, VarNC):
        vararg = varargs.get(varname,dict())
        if not isinstance(vararg,dict): raise TypeError, vararg
        if 'asNC' not in vararg: vararg['asNC'] = asNC
        varargs[varname] = vararg  
    # first invoke parent method, to make regular copy
    dataset = super(DatasetNetCDF,self).copy(varsdeep=varsdeep, varargs=varargs, **newargs)
    # now handle NetCDF stuff
    if asNC:
      if filename is None:
        mode = 'r' if newargs.pop('lwrite',False) else 'r' 
        dataset = DatasetNetCDF(mode=mode, filelist=self.filelist, dataset=self.datasets, 
                                atts=dataset.atts, variables=dataset.variables)
      else:
        #mode = 'wr' if 'r' in self.mode else 'w'      
        ncformat = newargs.pop('ncformat','NETCDF4')
        zlib = newargs.pop('zlib',True)
        dataset = asDatasetNC(dataset, ncfile=filename, mode='wr', deepcopy=varsdeep, 
                              writeData=writeData, ncformat=ncformat, zlib=zlib)        
    # return
    return dataset  
    
  def axisAnnotation(self, name, strlist, dim, atts=None):
    ''' Add a list of string values along the specified axis. '''
    # figure out dimensions
    if len(strlist) != len(self.axes[dim]) if isinstance(dim,basestring) else len(dim): raise AxisError
    if isinstance(strlist,(list,tuple)): strlist = np.array(strlist)
    elif not isinstance(strlist,np.ndarray) and strlist.dtype.kind == 'S': raise TypeError
    # create netcdf dimension and variable
    add_var(self.dataset, name, (dim,), data=strlist, atts=atts)
    #dimname = dim if isinstance(dim,basestring) else dim.name
    #add_strvar(self.dataset, name, strlist, dimname, atts=atts)    
    
  def close(self):
    ''' Call this method before deleting the Dataset: close netcdf files; if in write mode, also synchronizes with file system before closing. '''
    # synchronize data
    if 'w' in self.mode: self.sync() # 'if mode' is a precaution 
    # close files
    for ds in self.datasets: ds.close()

## run a test    
if __name__ == '__main__':
  
  pass
