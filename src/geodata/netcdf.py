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
  
  def __init__(self, ncvar, axes=None, mask=None, plotatts=None, load=False):
    ''' 
      Initialize Variable instance based on NetCDF variable.
      
      New Instance Attributes:
        ncvar = None # the associated netcdf variable 
    '''
    # construct attribute dictionary from netcdf attributes
    atts = { key : ncvar.getncattr(key) for key in ncvar.ncattrs() }
    # handle some netcdf conventions
    fillValue = atts.pop('fillValue',atts.pop('_fillValue',None))
    for key in ['scale_factor', 'add_offset']: atts.pop(key,None) # already handled by NetCDf Python interface
    name = ncvar.__dict__.get('name',ncvar._name) # name in attributes has precedence
    units = ncvar.__dict__.get('units','') # units are not mandatory
    # construct axes, based on netcdf dimensions
    if axes is None: axes = tuple([str(dim) for dim in ncvar.dimensions]) # have to get rid of unicode
    # call parent constructor
    super(VarNC,self).__init__(name=name, units=units, axes=axes, data=None, mask=mask, fillValue=fillValue, atts=atts, plotatts=plotatts)
    # assign special attributes
    self.__dict__['ncvar'] = ncvar
    if load: self.load() # load data here 
    
  def load(self, data=None, scale=True):
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
    # apply scale factor and offset
    if scale:
      if 'scale_factor' in self.atts: data *= self.atts['scale_factor']
      if 'add_offset' in self.atts: data += self.atts['add_offset']
    # load data
    super(VarNC,self).load(data=data, mask=None) # load actual data using parent method
    # no need to return anything...


class AxisNC(Axis,VarNC):
  '''
    A NetCDF Variable representing a coordinate axis.
  '''
  
  def __init__(self, ncvar, plotatts=None, load=True, **axargs):
    ''' Initialize a coordinate axis with appropriate values. '''
    # initialize as an Axis subclass and pass arguments down the inheritance chain
    super(AxisNC,self).__init__(ncvar=ncvar, plotatts=plotatts, load=load, **axargs)
#   def __init__(self, ncvar, plotatts=None, load=True):
#     ''' Initialize a coordinate axis with appropriate values. '''
#     # initialize dimensions
#     axes = (self,)
#     # N.B.: Axis objects carry a circular reference to themselves in the dimensions tuple
#     self.__dict__['coord'] = None
#     self.__dict__['len'] = ncvar.shape[0] 
#     # initialize as netcdf variable
# #     super(AxisNC,self).__init__(ncvar, axes=axes, mask=None, plotatts=plotatts, load=False)
#     super(AxisNC,self).__init__(ncvar, axes=axes, mask=None, plotatts=plotatts, load=False)
#     ## N.B.: Apparently the super(Class,self)-syntax does not work with multiple inheritance...
#     # add coordinate vector
#     if load: self.updateCoord(slice(ncvar.shape[0]))
#     self.updateLength(ncvar.shape[0])
    
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
  
  def __init__(self, folder='', dataset=None, filelist=None, varlist=None, varatts=None, atts=None, multifile=False):
    ''' 
      Create a Dataset from one or more NetCDF files; Variables are created from NetCDF variables. 
      
      NetCDF Attributes:
        datasets = [] # list of NetCDF datasets 
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
      datasets = []
      for ncfile in filelist:
        if multifile: 
          if isinstance(ncfile,(list,tuple)): tmpfile = [folder+ncf for ncf in ncfile]
          else: tmpfile = folder+ncfile  
          datasets.append(nc.MFDataset(tmpfile))        
        else:
          datasets.append(nc.Dataset(folder+ncfile))
    # create axes from netcdf dimensions and coordinate variables
    axes = dict()
    for ds in datasets:
      for dim in ds.dimensions.keys():
        if dim in ds.variables: # skip dimensions that have no associated variable 
          if axes.has_key(dim): # if already present, make sure axes are essentially the same
            if not isEqual(axes[dim][:],ds.variables[dim][:]): 
              raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible dimensions.' 
          else: # if this is a new axis, add it to the list
            axes[dim] = AxisNC(ncvar=ds.variables[dim])
    # create variables from netcdf variables
    variables = dict()
    for ds in datasets:
      for var in ds.variables.keys():
        if axes.has_key(var): pass # do not treat coordinate variables as real variables 
        elif variables.has_key(var): # if already present, make sure variables are essentially the same
          if not variables[var].shape == ds.variables[var].shape: 
            raise DatasetError, 'Error constructing Dataset: NetCDF files have incompatible variables.' 
        else: # if this is a new variable, add it to the list
          if all([axes.has_key(dim) for dim in ds.variables[var].dimensions]):
            varaxes = [axes[dim] for dim in ds.variables[var].dimensions]
            variables[var] = VarNC(ncvar=ds.variables[var], axes=varaxes)
    # get attributes from NetCDF dataset
    ncattrs = joinDicts(*[ds.__dict__ for ds in datasets])
    if atts: ncattrs.update(atts) # update with attributes passed to constructor
    # initialize Dataset using parent constructor
    super(NetCDFDataset,self).__init__(varlist=variables.values(), atts=ncattrs)
    # add NetCDF attributes
    self.__dict__['datasets'] = datasets
    
  def close(self):
    ''' Call this method before deleting the Dataset: close netcdf files. '''
    for ds in self.datasets: ds.close()

## run a test    
if __name__ == '__main__':
  
  pass