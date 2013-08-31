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
from geodata.base import Variable, Axis
from geodata.misc import checkIndex

class VarNC(Variable):
  '''
    A variable class that implements access to data from a NetCDF variable object.
  '''
  
  ncvar = None # the associated netcdf variable
  
  def __init__(self, ncvar, axes=None, mask=None, plotatts=None, load=False):
    ''' Initialize Variable instance based on NetCDF variable. '''
    # construct attribute dictionary from netcdf attributes
    atts = { key : ncvar.getncattr(key) for key in ncvar.ncattrs() } 
    name = ncvar.__dict__.get('name',ncvar._name)
    units = ncvar.__dict__.get('units','') # units are not mandatory
    # construct axes, based on netcdf dimensions
    if axes is None: axes = tuple([str(dim) for dim in ncvar.dimensions]) # have to get rid of unicode
    # call parent constructor
    super(VarNC,self).__init__(name=name, units=units, axes=axes, data=None, mask=mask, atts=atts, plotatts=plotatts)
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
    super(VarNC,self).load(data, mask=None) # load actual data using parent method
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

## run a test    
if __name__ == '__main__':
  
  # initialize a netcdf variable
  ncdata = nc.Dataset('/media/tmp/gpccavg/gpcc_25_clim_1979-1988.nc',mode='r')
  time = AxisNC(ncdata.variables['time'], length=len(ncdata.dimensions['time'])) 
  lat = AxisNC(ncdata.variables['lat'], length=len(ncdata.dimensions['lat']))
  lon = AxisNC(ncdata.variables['lon'], length=len(ncdata.dimensions['lon']))
  ncvar = VarNC(ncdata.variables['rain'], axes=(time,lat,lon))
  # NetCDF test
  ncvar.load((slice(0,12,1),slice(20,50,5),slice(70,140,15)))
#   ncvar.load((slice(20,50,5),slice(70,140,15)))
  print 'NetCDF variable:'
  print ncvar
  print 'VarNC MRO:', VarNC.mro()
  print 'NetCDF axis:'
  print time
  print 'AxisNC MRO:', AxisNC.mro()
  print time[:]
  
  # variable test
  print
  ncvar += np.ones(1)
  # test getattr
#   print ncvar.ncvar
  ncvar._FillValue = 0
  print 'Name: %s, Units: %s, Missing Values: %s'%(ncvar.name, ncvar.units, ncvar._FillValue)
  # test setattr
  ncvar.Comments = 'test'; ncvar.plotComments = 'test' 
  print 'Comments: %s, Plot Comments: %s'%(ncvar.Comments,ncvar.plotatts['plotComments'])
#   print var[:]
  # indexing (getitem) test
#   print ncvar.shape, ncvar[:,:], lon[:], lat[:] # [20:50:5,70:140:15]
  ncvar.unload()
#   print ncvar.data

  # axis test
  print 
  # test contains 
#   print ncvar[lon]
  for ax in (lon,lat):
    if ax in ncvar: print '%s is the %i. axis and has length %i'%(ax.name,ncvar[ax]+1,len(ax))
