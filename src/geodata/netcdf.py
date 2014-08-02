'''
Created on 2013-08-23

A module that provides GDAL functionality to GeoData datasets and variables, 
and exposes some other GDAL functionality, such as regriding.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import collections as col
import netCDF4 as nc # netcdf python module
import os

# import all base functionality from PyGeoDat
# from nctools import * # my own netcdf toolkit
from geodata.base import Variable, Axis, Dataset
from geodata.misc import checkIndex, isEqual, joinDicts
from geodata.misc import DatasetError, DataError, AxisError, NetCDFError, PermissionError, FileError, VariableError 
from geodata.nctools import coerceAtts, writeNetCDF, add_var, add_coord, add_strvar

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
  varnc = VarNC(ncvar, axes=axes, atts=atts, plot=var.plot.copy(), fillValue=var.fillValue, 
                dtype=var.dtype, mode=mode, **kwargs)
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
  axisnc = AxisNC(ncvar, atts=atts, plot=ax.plot.copy(), length=len(ax), coord=None, dtype=ax.dtype, 
                  mode=mode, **kwargs)
  # copy data  
  if ax.data:    
    axisnc.coord = ax.getArray(copy=deepcopy)
  # return AxisNC
  return axisnc

def asDatasetNC(dataset=None, ncfile=None, mode='rw', deepcopy=False, writeData=False, ncformat='NETCDF4', zlib=True, **kwargs):
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

class VarNC(Variable):
  '''
    A variable class that implements access to data from a NetCDF variable object.
  '''
  
  def __init__(self, ncvar, name=None, units=None, axes=None, data=None, dtype=None, scalefactor=1, offset=0, 
               transform=None, atts=None, plot=None, fillValue=None, mode='r', load=False, squeeze=False):
    ''' 
      Initialize Variable instance based on NetCDF variable.
      
      New Instance Attributes:
        mode = 'r' # a string indicating whether read ('r') or write ('w') actions are intended/permitted
        ncvar = None # the associated netcdf variable
        scalefactor = 1 # linear scale factor w.r.t. values in netcdf file
        offset = 0 # constant offset w.r.t. values in netcdf file
        transform = None # function that can perform non-trivial transforms upon load
        squeezed = False # if True, all singleton dimensions in NetCDF Variable are silently ignored
    '''
    # check mode
    if not (mode == 'w' or mode == 'r' or mode == 'rw' or mode == 'wr'):  raise PermissionError  
    # write-only actions
    if isinstance(ncvar,nc.Dataset):
      if 'w' not in mode: mode += 'w'
      if name is None and isinstance(atts,dict): name = atts.get('name',None)      
      dims = [ax if isinstance(ax,basestring) else ax.name for ax in axes] # list axes names
      dimshape = [None if isinstance(ax,basestring) else len(ax) for ax in axes]
      if dtype is None: 
        if data is not None: dtype = data.dtype
        else: raise TypeError, "No data (-type) to construct NetCDF variable!"
      else: dtype = np.dtype(dtype)
      # construct a new netcdf variable in the given dataset
      ncvar = add_var(ncvar, name, dims=dims, shape=dimshape, atts=atts, dtype=dtype, fillValue=fillValue, zlib=True)
    # some type checking
    if not isinstance(ncvar,nc.Variable): raise TypeError, "Argument 'ncvar' has to be a NetCDF Variable or Dataset."
    if dtype and dtype != ncvar.dtype: raise TypeError    
    if data is not None and data.shape != ncvar.shape: raise DataError
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
        axes = tuple([str(dim) for dim in ncvar.dimensions]) # have to get rid of unicode
      elif len(ncvar.dimensions) != len(axes): raise AxisError
    else: ncatts = atts
    # check transform
    if transform is not None and not callable(transform): raise TypeError
    # call parent constructor
    super(VarNC,self).__init__(name=name, units=units, axes=axes, data=None, dtype=ncvar.dtype, 
                               mask=None, fillValue=fillValue, atts=ncatts, plot=plot)
    # assign special attributes
    self.__dict__['ncvar'] = ncvar
    self.__dict__['mode'] = mode
    self.__dict__['offset'] = offset
    self.__dict__['scalefactor'] = scalefactor
    self.__dict__['transform'] = transform
    self.__dict__['squeezed'] = False
    if squeeze: self.squeeze() # may set 'squeezed' to True
    # handle data
    if load and data: raise DataError, "Arguments 'load' and 'data' are mutually exclusive, i.e. only one can be used!"
    elif load and 'r' in self.mode: self.load(data=None) # load data from file
    elif data is not None and 'w' in self.mode: self.load(data=data) # load data from array
    # sync?
    if 'w' in self.mode: self.sync() 
  
  def __getitem__(self, idx=None):
    ''' Method implementing access to the actual data; if data is not loaded, give direct access to NetCDF file. '''
    # default
    if idx is None: idx = [slice(None,None,None),]*self.ndim # first, last, step          
    # determine what to do
    if self.data:
      # call parent method     
      data = super(VarNC,self).__getitem__(idx) # load actual data using parent method      
    else:
      # provide direct access to netcdf data on file
      if isinstance(idx,(list,tuple)):
        if len(idx) != self.ndim: raise AxisError
        if self.squeezed:
          # figure out slices
          idx = list(idx) # need to insert items
          for i in xrange(self.ncvar.ndim):
            if self.ncvar.shape[i] == 1: idx.insert(i, 0) # '0' automatically squeezes out this dimension upon retrieval
      else: idx = (idx,)
      data = self.ncvar.__getitem__(idx) # exceptions handled by netcdf module
      #assert self.ndim == data.ndim # make sure that squeezing works!
      # N.B.: the shape can change dynamically when a slice is loaded, so don't check for that, or it will fail!
      # apply scalefactor and offset
      if self.offset != 0: data += self.offset
      if self.scalefactor != 1: data *= self.scalefactor
      if self.transform is not None: data = self.transform(data, var=self, slc=idx)
    # return data
    return data  
  
  def squeeze(self, **kwargs):
    ''' A method to remove singleton dimensions; special handling of __getitem__() is necessary, 
        because NetCDF Variables cannot be squeezed directly. '''
    self.squeezed = True
    return super(VarNC,self).squeeze(**kwargs) # just call superior  
  
  def copy(self, **newargs):
    ''' A method to copy the Variable with just a link to the data. '''
    # N.B.: we can't really return a VarNC object, since it would have to be attached to a new NetCDF file;
    #       instead, create a DatasetNetCDF instance with a new NetCDF file, and add the variable to the
    #       new dataset, using addVariable with the asNC=True and the copy=True / deepcopy=True option
    return super(VarNC,self).copy(**newargs) # just call superior - returns a regular Variable instance
    
  def load(self, data=None, **kwargs):
    ''' Method to load data from NetCDF file into RAM. '''
    if data is None: 
      data = self.__getitem__() # load everything
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
    # load data    
    super(VarNC,self).load(data=data, **kwargs) # load actual data using parent method
    # no need to return anything...
    
  def sync(self):
    ''' Method to make sure, data in NetCDF variable and Variable instance are consistent. '''
    ncvar = self.ncvar
    # update netcdf variable    
    if 'w' in self.mode:      
      if not self.squeezed and ncvar.shape != self.shape: 
        raise NetCDFError, "Cannot write to NetCDF variable: array shape in memory and on disk are inconsistent!"
      if self.squeezed and tuple([n for n in ncvar.shape if n > 1]) != self.shape: 
        raise NetCDFError, "Cannot write to NetCDF variable: array shape in memory and on disk are inconsistent!"
      if self.data:
        # special handling of numpy bools: cast as 8-bit integers
        if isinstance(self.data_array,np.bool_): ncvar[:] = self.data_array.astype('i1')
        else: ncvar[:] = self.data_array # masking should be handled by the NetCDF module
        # reset scale factors etc.
        self.scalefactor = 1; self.offset = 0
      # update NetCDF attributes
      ncvar.setncatts(coerceAtts(self.atts))
      ncattrs = ncvar.ncattrs() # list of current NC attributes
      ncvar.set_auto_maskandscale(True) # automatic handling of missing values and scaling and offset
      if self.fillValue: 
        ncvar.setncattr('missing_value',self.fillValue)        
      if 'scale_factor' in ncattrs: ncvar.delncattr('scale_factor',ncvar.getncattr('scale_factor'))
      if 'add_offset' in ncattrs: ncvar.delncattr('add_offset',ncvar.getncattr('add_offset'))
      # set other attributes like in variable
      ncvar.setncattr('name',self.name)
      ncvar.setncattr('units',self.units)
      # now sync dataset
      ncvar.group().sync()     
    else: 
      raise PermissionError, "Cannot write to NetCDF variable: writing (mode = 'w') not enabled!"
     
  def unload(self):
    ''' Method to sync the currently loaded data to file and free up memory (discard data in memory) '''
    # synchronize data with NetCDF file
    if 'w' in self.mode: self.sync() # only if we have write permission, of course
    # discard NetCDF Variable object (contains a reference to the data)
    ncds = self.ncvar.group(); ncname = self.ncvar._name # this is the actual netcdf name
    del self.ncvar; self.ncvar = ncds.variables[ncname] # reattach (hopefully without the data array)
    # discard data array the usual way
    super(VarNC,self).unload()

class AxisNC(Axis,VarNC):
  '''
    A NetCDF Variable representing a coordinate axis.
  '''
  
  def __init__(self, ncvar, name=None, length=0, coord=None, dtype=None, atts=None, fillValue=None, mode='r', load=True, **axargs):
    ''' Initialize a coordinate axis with appropriate values. '''
    if isinstance(ncvar,nc.Dataset):
      if 'w' not in mode: mode += 'w'
      if name is None and isinstance(atts,dict): name = atts.pop('name',None) 
      # construct a new netcdf coordinate variable in the given dataset
      ncvar = add_coord(ncvar, name, length=length, data=coord, dtype=dtype, fillValue=fillValue, atts=atts, zlib=True)
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
  
  def __init__(self, name=None, title=None, dataset=None, filelist=None, varlist=None, varatts=None, atts=None, axes=None, 
               multifile=False, check_override=None, ignore_list=None, folder='', mode='r', ncformat='NETCDF4', squeeze=True):
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
    # create a new NetCDF file
    if 'w' == mode and filelist:    
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
      dataset = writeNetCDF(dataset, filename, ncformat='NETCDF4', zlib=True, writeData=False, close=False)
    # either use available NetCDF datasets directly, or open datasets from filelist  
    if isinstance(dataset,nc.Dataset): 
      datasets = [dataset]  # datasets is used later
      if 'filepath' in dir(dataset): filelist = [dataset.filepath] # only available in newer versions
    elif isinstance(dataset,(list,tuple)): 
      if not all([isinstance(ds,nc.Dataset) for ds in dataset]): raise TypeError
      datasets = dataset
      filelist = [dataset.filepath for dataset in datasets if 'filepath' in dir(dataset)]
    else:
      # open netcdf datasets from netcdf files
      if not isinstance(filelist,col.Iterable): raise TypeError
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
            datasets.append(nc.MFDataset(tmpfile), mode='r', format=ncformat)
          else: # open a simple single-file dataset
            tmpfile = folder+ncfile
            datasets.append(nc.Dataset(tmpfile, mode='r', format=ncformat))
        except RuntimeError:
          raise NetCDFError, "Error reading file '{0:s}' in folder {1:s}".format(ncfile,folder)
        filenames.append(tmpfile)
      filelist = filenames # original file list, absolute path        
    # from here on, dataset creation is based on the netcdf-Dataset(s) in 'datasets'
    if ignore_list is not None:
      if isinstance(ignore_list,(list,tuple,set)): ignore_list = set(ignore_list) # order doesn't matter
      elif not isinstance(ignore_list,set): raise TypeError         
    else: ignore_list = set()
    # create axes from netcdf dimensions and coordinate variables
    if varatts is None: varatts = dict() # empty dictionary means no parameters...
    if check_override is None: check_override = [] # list of variables (and axes) that is not checked for consistency
    # N.B.: check_override may be necessary to combine datasets from different files with inconsistent axis instances 
    if axes is None: axes = dict()
    else: check_override += axes.keys() # don't check externally provided axes   
    if not isinstance(axes,dict): raise TypeError
    for ds in datasets:
      for dim in ds.dimensions.keys():
        if dim not in ignore_list:        
          if dim in ds.variables: # dimensions with an associated coordinate variable           
            if dim in axes: # if already present, make sure axes are essentially the same
              tmpax = AxisNC(ncvar=ds.variables[dim], mode='r', **varatts.get(dim,{})) # apply all correction factors...
              if dim not in check_override and not isEqual(axes[dim][:],tmpax[:]): 
                raise DatasetError, "Error constructing Dataset: NetCDF files have incompatible {:s} dimension.".format(dim)
            else: # if this is a new axis, add it to the list
              if ds.variables[dim].dtype == '|S1': pass # Variables of type char are currently not implemented
              else:      
                axes[dim] = AxisNC(ncvar=ds.variables[dim], mode=mode, **varatts.get(dim,{})) # also use overrride parameters
          else: # initialize dimensions without associated variable as regular Axis (not AxisNC)
            if dim in axes: # if already present, make sure axes are essentially the same
              if len(axes[dim]) != len(ds.dimensions[dim]): 
                raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible dimensions.' 
            else: # if this is a new axis, add it to the list
              params = dict(name=dim,coord=np.arange(len(ds.dimensions[dim]))); params.update(varatts.get(dim,{}))
              axes[dim] = Axis(**params) # also use overrride parameters          
    # create variables from netcdf variables    
    variables = dict()
    for ds in datasets:
      if varlist is None: dsvars = [var for var in ds.variables.keys() if not var in ignore_list] 
      else: dsvars = [var for var in varlist if ds.variables.has_key(var)] # varlist overrides ignore_list 
      # loop over variables in dataset
      for var in dsvars:
        if var in axes: pass # do not treat coordinate variables as real variables 
        elif ds.variables[var].dtype == '|S1': pass # just ignore string variables for now... 
        elif ds.variables[var].ndim == 0: pass # also ignore scalars for now...
          #raise NotImplementedError # Variables of type char are currently not implemented
        elif var in variables: # if already present, make sure variables are essentially the same
          if var not in check_override: 
            if ( (variables[var].shape != ds.variables[var].shape) or
                 (variables[var].ncvar.dimensions != ds.variables[var].dimensions) ): 
              raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible variables.' 
        else: # if this is a new variable, add it to the list
          if all([axes.has_key(dim) for dim in ds.variables[var].dimensions]):
            varaxes = [axes[dim] for dim in ds.variables[var].dimensions] # collect axes
            # create new variable using the override parameters in varatts
            variables[var] = VarNC(ncvar=ds.variables[var], axes=varaxes, mode=mode, squeeze=squeeze, **varatts.get(var,{}))
          elif not any([dim in ignore_list for dim in ds.variables[var].dimensions]): # legitimate omission
            print var, ds.variables[var].dimensions
            raise DatasetError, 'Error constructing Variable: Axes/coordinates not found.'
    # get attributes from NetCDF dataset
    ncattrs = joinDicts(*[ds.__dict__ for ds in datasets])
    if atts is not None: ncattrs.update(atts) # update with attributes passed to constructor
    self.__dict__['mode'] = mode
    # add NetCDF attributes
    self.__dict__['datasets'] = datasets
    self.__dict__['filelist'] = filelist
    # initialize Dataset using parent constructor
    super(DatasetNetCDF,self).__init__(name=name, title=title, varlist=variables.values(), atts=ncattrs)
    
  @property
  def dataset(self):
    ''' The first element of the datasets list. '''
    return self.datasets[0] 
  
  def addAxis(self, ax, asNC=True, copy=False, overwrite=False):
    ''' Method to add an Axis to the Dataset. (If the Axis is already present, check that it is the same.) '''   
    # cast Axis instance as AxisNC
    if copy: # make a new instance or add it as is
      if asNC and 'w' in self.mode: ax = asAxisNC(ax=ax, ncvar=self.datasets[0], mode=self.mode, deepcopy=True)
      else: ax = ax.copy(deepcopy=True)
    # hand-off to parent method and return status
    return super(DatasetNetCDF,self).addAxis(ax=ax, copy=False, overwrite=overwrite) # already copied above
  
  def addVariable(self, var, asNC=True, copy=True, overwrite=False, deepcopy=False):
    ''' Method to add a new Variable to the Dataset. '''
    if var.name in self.__dict__: 
      # replace axes, if permitted; need to use NetCDF method immediately, though
      if overwrite and self.hasVariable(var.name): 
        return self.replaceVariable(var.name, var, deepcopy=deepcopy)
      else: raise AttributeError, "Cannot add Variable '%s' to Dataset, because an attribute of the same name already exits!"%(var.name)      
    else:             
      if deepcopy: copy=True   
      # cast Axis instance as AxisNC
      if copy: # make a new instance or add it as is 
        if asNC and 'w' in self.mode:
          for ax in var.axes:
            if not self.hasAxis(ax.name): 
              self.addAxis(ax, asNC=True, copy=True)
          # add variable as a NetCDF variable             
          var = asVarNC(var=var,ncvar=self.datasets[0], axes=self.axes, mode=self.mode, deepcopy=deepcopy)
        else: 
          var = var.copy(deepcopy=deepcopy) # or just add as a normal Variable
      # hand-off to parent method and return status
      return super(DatasetNetCDF,self).addVariable(var=var)
      
  def repalceAxis(self, oldaxis, newaxis=None, asNC=True, deepcopy=True):    
    ''' Replace an existing axis with a different one and transfer NetCDF reference to new axis. '''
    if newaxis is None: 
      newaxis = oldaxis; oldaxis = newaxis.name # i.e. replace old axis with the same name'
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
  
  def replaceVariable(self, oldvar, newvar=None, asNC=True, deepcopy=False):
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
      self.addVariable(newvar, copy=False, overwrite=False)
    else: # no need for special treatment...
      super(DatasetNetCDF,self).replaceVariable(oldvar, newvar)
    # return status of variable
    return self.hasVariable(newvar)  
  
  def load(self, **slices):
    ''' Load all VarNC's and AxisNC's using the slices specified as keyword arguments. '''
    # make slices
    for key,value in slices.iteritems():
      if isinstance(value,col.Iterable): 
        slices[key] = slice(*value)
      else: 
        if not isinstance(value,(int,np.integer)): raise TypeError
    # load variables
    for var in self.variables.values():
      if isinstance(var,VarNC):
        idx = [slices.get(ax.name,slice(None)) for ax in var.axes]
        var.load(data=idx) # load slice, along with relevant dimensions
    # no return value...        
    
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
    
  def copy (self, asNC=False, filename=None, varsdeep=False, **newargs):
    ''' Copy a DatasetNetCDF, either into a normal Dataset or into a DatasetNetCDF (requires a filename). '''
    if asNC:
      #mode = 'wr' if 'r' in self.mode else 'w'      
      writeData = newargs.pop('writeData',False)
      ncformat = newargs.pop('ncformat','NETCDF4')
      zlib = newargs.pop('zlib',True)
      dataset = asDatasetNC(self, ncfile=filename, mode='wr', deepcopy=varsdeep, 
                            writeData=writeData, ncformat=ncformat, zlib=zlib)
    else: # just invoke parent method
      dataset = super(DatasetNetCDF,self).copy(varsdeep=varsdeep, **newargs)
    # return
    return dataset  
    
  def axisAnnotation(self, name, strlist, dim, atts=None):
    ''' Add a list of string values along the specified axis. '''
    # figure out dimensions
    dimname = dim if isinstance(dim,basestring) else dim.name
    if len(strlist) != len(self.axes[dim]) if isinstance(dim,basestring) else len(dim): raise AxisError
    # create netcdf dimension and variable
    add_strvar(self.dataset, name, strlist, dimname, atts=atts)    
    
  def close(self):
    ''' Call this method before deleting the Dataset: close netcdf files; if in write mode, also synchronizes with file system before closing. '''
    # synchronize data
    if 'w' in self.mode: self.sync() # 'if mode' is a precaution 
    # close files
    for ds in self.datasets: ds.close()

## run a test    
if __name__ == '__main__':
  
  pass