'''
Created on 2013-08-23

A module that provides GDAL functionality to GeoData datasets and variables, 
and exposes some other GDAL functionality, such as regriding.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc # netcdf python module

# import all base functionality from PyGeoDat
# from nctools import * # my own netcdf toolkit
from geodata.base import Variable, Axis, Dataset
from geodata.misc import checkIndex, isEqual, DatasetError, joinDicts

class VarNC(Variable):
  '''
    A variable class that implements access to data from a NetCDF variable object.
  '''
  
  def __init__(self, ncvar, name=None, units=None, axes=None, scalefactor=1, offset=0, atts=None, plot=None, fillValue=None, load=False):
    ''' 
      Initialize Variable instance based on NetCDF variable.
      
      New Instance Attributes:
        ncvar = None # the associated netcdf variable
        scalefactor = 1 # linear scale factor w.r.t. values in netcdf file
        offset = 0 # constant offset w.r.t. values in netcdf file 
    '''
    # construct attribute dictionary from netcdf attributes
    ncatts = { key : ncvar.getncattr(key) for key in ncvar.ncattrs() }
    # handle some netcdf conventions
    if fillValue is not None: fillValue = ncatts.pop('fillValue',ncatts.pop('_fillValue',None))
    for key in ['scale_factor', 'add_offset']: ncatts.pop(key,None) # already handled by NetCDf Python interface
    if name is None: name = ncatts.get('name',ncvar._name) # name in attributes has precedence
    else: ncatts[name] = name
    if units is None: units = ncatts.get('units','') # units are not mandatory
    else: ncatts[units] = units
    # update netcdf attributes with custom override
    if atts is not None: ncatts.update(atts)
    # construct axes, based on netcdf dimensions
    if axes is None: axes = tuple([str(dim) for dim in ncvar.dimensions]) # have to get rid of unicode
    # call parent constructor
    super(VarNC,self).__init__(name=name, units=units, axes=axes, data=None, mask=None, fillValue=fillValue, atts=ncatts, plot=plot)
    # assign special attributes
    self.__dict__['ncvar'] = ncvar
    self.__dict__['scalefactor'] = scalefactor
    self.__dict__['offset'] = offset
    if load: self.load() # load data here 
  
  def __getitem__(self, idx=None):
    ''' Method implementing access to the actual data; if data is not loaded, give direct access to NetCDF file. '''
    # default
    if idx is None: idx = slice(None,None,None) # first, last, step          
    # determine what to do
    if self.data:
      # call parent method
      data = super(VarNC,self).__getitem__(idx) # load actual data using parent method
    else:
      # provide direct access to netcdf data on file
      data = self.ncvar.__getitem__(idx) # exceptions handled by netcdf module
    # return data
    return data
    
  def load(self, data=None, **kwargs):
    ''' Method to load data from NetCDF file into RAM. '''
    if data is None: 
      data = self.ncvar[:] # load everything
    elif all(checkIndex(data)):
      if isinstance(data,tuple):
        assert len(data)==len(self.shape), 'Length of index tuple has to equal to the number of dimensions!'       
        for ax,idx in zip(self.axes,data): ax.updateCoord(idx)
        data = self.ncvar.__getitem__(data) # load slice
      else: 
        assert 1==len(self.shape), 'Multi-dimensional variable have to be indexed using tuples!'
        if self != self.axes[0]: ax.updateCoord(data) # prevent infinite loop due to self-reference 
        data = self.ncvar.__getitem__(data) # load slice
    else:
      assert isinstance(data,np.ndarray) 
      data = data
    # apply scalefactor and offset
    if self.scalefactor != 1: data *= self.scalefactor
    if self.offset != 0: data += self.offset
    # load data    
    super(VarNC,self).load(data=data, **kwargs) # load actual data using parent method
    # no need to return anything...


class AxisNC(Axis,VarNC):
  '''
    A NetCDF Variable representing a coordinate axis.
  '''
  
  def __init__(self, ncvar, load=True, **axargs):
    ''' Initialize a coordinate axis with appropriate values. '''
    # initialize as an Axis subclass and pass arguments down the inheritance chain
    super(AxisNC,self).__init__(ncvar=ncvar, load=load, **axargs)
    
  def updateCoord(self, coord=None):
    ''' Update the coordinate vector from NetCDF file. '''    
    # resolve coordinates
    if isinstance(coord,slice):      
      # load data using VarNC load function and coord slice
      super(AxisNC,self).load(data=coord)
      # update attributes
      self.__dict__['coord'] = self.data_array
      self.__dict__['len'] = self.shape[0]    
    else:
      # use parent constructor
      super(AxisNC,self).updateCoord(coord=coord)
      

class NetCDFDataset(Dataset):
  '''
    A container class for variable and axes objects, as well as some meta information. This class also 
    implements collective operations on all variables in the dataset.
  '''
  
  def __init__(self, folder='', dataset=None, filelist=None, varlist=None, varatts=None, atts=None, multifile=False, ncformat='NETCDF4'):
    ''' 
      Create a Dataset from one or more NetCDF files; Variables are created from NetCDF variables. 
      
      NetCDF Attributes:
        datasets = [] # list of NetCDF datasets
        filelist = None # files used to create datasets 
      Basic Attributes:        
        variables = dict() # dictionary holding Variable instances
        axes = dict() # dictionary holding Axis instances (inferred from Variables)
        atts = AttrDict() # dictionary containing global attributes / meta data
    '''
    # create netcdf datasets
    if dataset: # from netcdf datasets
      assert filelist is None
      if isinstance(dataset,nc.Dataset): datasets = [dataset]
      elif isinstance(dataset,(list,tuple)): 
        assert all([isinstance(ds,nc.Dataset) for ds in dataset])
        datasets = dataset
    else: # or directly from netcdf files
      assert filelist is not None
      datasets = []; files = []
      for ncfile in filelist:
        if multifile: 
          if isinstance(ncfile,(list,tuple)): tmpfile = [folder+ncf for ncf in ncfile]
          else: tmpfile = folder+ncfile # multifile via regular expressions
          datasets.append(nc.MFDataset(tmpfile), mode='r', format=ncformat)
        else: 
          tmpfile = folder+ncfile
          datasets.append(nc.Dataset(tmpfile, mode='r', format=ncformat))
        files.append(tmpfile)
      filelist = files # original file list, including folders        
    # create axes from netcdf dimensions and coordinate variables
    if varatts is None: varatts = dict() # empty dictionary means no parameters... 
    axes = dict()
    for ds in datasets:
      for dim in ds.dimensions.keys():
        if dim in ds.variables: # dimensions with an associated coordinate variable 
          if dim in axes: # if already present, make sure axes are essentially the same
            if not isEqual(axes[dim][:],ds.variables[dim][:]): 
              raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible dimensions.' 
          else: # if this is a new axis, add it to the list
            axes[dim] = AxisNC(ncvar=ds.variables[dim], **varatts.get(dim,{})) # also use overrride parameters
        else: # initialize dimensions without associated variable as regular Axis (not AxisNC)
          if dim in axes: # if already present, make sure axes are essentially the same
            if len(axes[dim]) != len(ds.dimensions[dim]): 
              raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible dimensions.' 
          else: # if this is a new axis, add it to the list
            params = dict(name=dim,length=len(ds.dimensions[dim])); params.update(varatts.get(dim,{})) 
            axes[dim] = Axis(**params) # also use overrride parameters          
    # create variables from netcdf variables
    variables = dict()
    for ds in datasets:
      if varlist is None: dsvars = ds.variables.keys()
      else: dsvars = [var for var in varlist if ds.variables.has_key(var)]
      # loop over variables in dataset
      for var in dsvars:
        if var in axes: pass # do not treat coordinate variables as real variables 
        elif var in variables: # if already present, make sure variables are essentially the same
          if (variables[var].shape != ds.variables[var].shape) or \
             (variables[var].ncvar.dimensions != ds.variables[var].dimensions): 
            raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible variables.' 
        else: # if this is a new variable, add it to the list
          if all([axes.has_key(dim) for dim in ds.variables[var].dimensions]):
            varaxes = [axes[dim] for dim in ds.variables[var].dimensions] # collect axes
            # create new variable using the override parameters in varatts
            variables[var] = VarNC(ncvar=ds.variables[var], axes=varaxes, **varatts.get(var,{}))
          else: raise DatasetError, 'Error constructing Variable: Axes/coordinates not found.'
    # get attributes from NetCDF dataset
    ncattrs = joinDicts(*[ds.__dict__ for ds in datasets])
    if atts: ncattrs.update(atts) # update with attributes passed to constructor
    # initialize Dataset using parent constructor
    super(NetCDFDataset,self).__init__(varlist=variables.values(), atts=ncattrs)
    # add NetCDF attributes
    self.__dict__['datasets'] = datasets
    self.__dict__['filelist'] = filelist
    
  def close(self):
    ''' Call this method before deleting the Dataset: close netcdf files. '''
    for ds in self.datasets: ds.close()

## run a test    
if __name__ == '__main__':
  
  pass