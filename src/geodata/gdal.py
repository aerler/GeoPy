'''
Created on 2013-08-23

A module that provides GDAL functionality to GeoData datasets and variables, 
and exposes some more GDAL functionality, such as regriding.

The GDAL functionality for Variables is implemented as a decorator that adds GDAL attributes 
and methods to an existing object/class instance.    

@author: Andre R. Erler, GPL v3
'''

import numpy as np
import types # needed to bind functions to objects

# gdal imports
from osgeo import gdal, osr
# register RAM driver
ramdrv = gdal.GetDriverByName('MEM')

# import all base functionality from PyGeoDat
from geodata.base import Variable, Axis
from misc import isEqual, isZero, isFloat, DataError


def addGDAL(var, projection=None, geotransform=None):
  ''' 
    A function that adds GDAL-based geographic projection features to an existing Variable instance.
    
    New Instance Attributes: 
      gdal = False # whether or not this instance possesses any GDAL functionality and is map-like
      isProjected = False # whether lat/lon spherical (False) or a geographic projection (True)
      projection = None # a GDAL spatial reference object
      geotransform = None # a GDAL geotransform vector (can e inferred from coordinate vectors)
      mapSize = None # size of horizontal dimensions
      bands = None # all dimensions except, the map coordinates 
      xlon = None # West-East axis
      ylat = None # South-North axis  
  '''
  # check some special conditions
  assert isinstance(var,Variable), 'This function can only be used to add GDAL functionality to \'Variable\' instances!'
  lgdal = False; isProjected = None; mapSize = None; bands = None; xlon = None; ylat = None # defaults
  # only for 2D variables!
  shape = var.shape; ndim = var.ndim
  if ndim >= 2: # else not a map-type
    mapSize = var.shape[-2:]
    assert all(mapSize), 'Horizontal dimensions have to be of finite, non-zero length.'
    if ndim==2: bands = 1 
    else: bands = np.prod(shape[:-2])
    if projection is not None: # figure out projection 
      assert isinstance(projection,osr.SpatialReference), '\'projection\' has to be a GDAL SpatialReference object.'              
      isProjected =  projection.IsProjected
      if isProjected: 
        assert ('x' in self) and ('y' in self), 'Horizontal axes for projected GDAL variables have to \'x\' and \'y\'.'
        xlon = var.x; ylat = var.y
      else: 
        assert ('lon' in self) and ('lat' in self), 'Horizontal axes for non-projected GDAL variables have to \'lon\' and \'lat\''
        xlon = var.lon; ylat = var.lat    
    else: # can still infer some useful info
      if ('x' in var) and ('y' in var):
        isProjected = True; xlon = var.x; ylat = var.y
      elif ('lon' in var) and ('lat' in var):
        isProjected = False; xlon = var.lon; ylat = var.lat
        projection = osr.SpatialReference(); projection.ImportFromEPSG(4326) # normal lat/lon projection       
    # if the variable is map-like, add GDAL properties
    if xlon is not None and ylat is not None:
      lgdal = True
      # check axes
      assert (var[xlon] in [var.ndim-1,var.ndim-2]) and (var[ylat] in [var.ndim-1,var.ndim-2]),\
         'Horizontal axes (\'lon\' and \'lat\') have to be the innermost indices.'
      if isProjected: assert isinstance(xlon,Axis) and isinstance(ylat,Axis), 'Error: attributes \'x\' and \'y\' have to be axes.'
      else: assert isinstance(xlon,Axis) and isinstance(ylat,Axis), 'Error: attributes \'lon\' and \'lat\' have to be axes.'   

  # modify Variable instance
  var.__dict__['gdal'] = lgdal # all variables have this after going through this process
  if lgdal:
    
    # infer or check geotransform
    if geotransform is None: 
      # infer GDAL geotransform vector from  coordinate vectors (axes)
      dx = xlon[1]-xlon[0]; dy = ylat[1]-ylat[0]
      assert (np.diff(xlon) == dx).all() and (np.diff(ylat) == dy).all(), 'Coordinate vectors have to be uniform!'
      ulx = xlon[0]-dx/2.; uly = ylat[0]-dy/2. # coordinates of upper left corner (same for source and sink)
      # GT(2) & GT(4) are zero for North-up; GT(1) & GT(5) are pixel width and height; (GT(0),GT(3)) is the top left corner
      geotransform = (ulx, dx, 0., uly, 0., dy)
    elif xlon.data or ylat.data:
      # check if GDAL geotransform vector is consistent with coordinate vectors
      assert len(geotransform) == 6, '\'geotransform\' has to be a vector or list with 6 elements.'
      dx = geotransform[1]; dy = geotransform[5]; ulx = geotransform[0]; uly = geotransform[3] 
      assert isZero(np.diff(xlon)-dx) and isZero(np.diff(ylat)-dy), 'Coordinate vectors have to be compatible with geotransform!'
      assert isEqual(uly+dx/2,xlon[0]) and isEqual(uly+dy/2,ylat[0]) # coordinates of upper left corner (same for source and sink)       
    else: assert len(geotransform) == 6 and all(isFloat(geotransform)), '\'geotransform\' has to be a vector or list of 6 floating-point numbers.'
    # add new instance attributes (projection parameters)
    var.__dict__['isProjected'] = isProjected    
    var.__dict__['projection'] = projection
    var.__dict__['geotransform'] = geotransform
    var.__dict__['mapSize'] = mapSize
    var.__dict__['bands'] = bands
    var.__dict__['xlon'] = xlon
    var.__dict__['ylat'] = ylat
      
    # define GDAL-related 'class methods'  
    def getGDAL(self, load=True):
      ''' Method that returns a gdal dataset, ready for use with GDAL routines. '''
      if self.gdal:
        # determine GDAL data type
        if self.dtype == 'float32': gdt = gdal.GDT_Float32
        elif self.dtype == 'float64': gdt = gdal.GDT_Float64
        elif self.dtype == 'int16': gdt = gdal.GDT_Int16
        elif self.dtype == 'int32': gdt = gdal.GDT_Int32
        else: raise TypeError, 'Cannot translate numpy data type into GDAL data type!'        
        # create GDAL dataset 
        xe = len(self.xlon); ye = len(self.ylat) 
        dataset = ramdrv.Create(self.name, int(xe), int(ye), int(self.bands), int(gdt)) 
        # N.B.: for some reason a dataset is always initialized with 6 bands
        # set projection parameters
        dataset.SetGeoTransform(self.geotransform) # does the order matter?
        dataset.SetProjection(self.projection.ExportToWkt()) # is .ExportToWkt() necessary?        
        if load:
          if not self.data: self.load()
          if not self.data: raise DataError, 'Need data in Variable instance in order to load data into GDAL dataset!'
          data = self.get(unmask=True) # get unmasked data
          data = data.reshape(self.bands,self.mapSize[0],self.mapSize[1]) # reshape to fit bands
          # assign data
          for i in xrange(self.bands):
            dataset.GetRasterBand(i+1).WriteArray(data[i,:,:])
            if self.masked: dataset.GetRasterBand(i+1).SetNoDataValue(float(self.fillValue))
      else: dataset = None
      # return dataset
      return dataset
    # add new method to object
    var.getGDAL = types.MethodType(getGDAL,var)
    
    # update new instance attributes
    def load(self, data):
      ''' Load new data array. '''
      super(var.__class__,self).load(data, mask=None)    
      if len(data.shape) >= 2: # 2D or more
        self.__dict__['mapSize'] = data.shape[-2:] # need to update
      else: # less than 2D can't be GDAL enabled
        self.__dict__['mapSize'] = None
        self.__dict__['gdal'] = False
    # add new method to object
    var.load = types.MethodType(load,var)
    
    # maybe needed in the future...
    def unload(self):
      ''' Remove coordinate vector. '''
      super(var.__class__,self).unload()      
    # add new method to object
    var.unload = types.MethodType(unload,var)
  
  # the return value is actually not necessary, since the object is modified immediately
  return var

class VarGDAL(Variable):
  '''
    An extension to the Variable class that adds some GDAL-based geographic projection features.
  '''
  gdal = False # whether or not this instance possesses any GDAL functionality
  isProjected = False # whether lat/lon spherical (False) or a geographic projection (True)
  projection = None # a GDAL spatial reference object
  geotransform = None # a GDAL geotransform vector
  mapSize = None # size of horizontal dimensions
  bands = None # all dimensions except, the map coordinates 
  xlon = None # West-East axis
  ylat = None # South-North axis
  
  def __init__(self, name='N/A', units='N/A', axes=None, data=None, mask=None, projection=None, geotransform=None, atts=None, plotatts=None):
    ''' Initialize Variable Instance with GDAL projection features. '''
    # initialize standard Variable
    super(VarGDAL,self).__init__(name=name, units=units, axes=axes, data=data, mask=mask, atts=atts, plotatts=plotatts)
    # check some special conditions
    lgdal = False; isProjected = None; mapSize = None; bands = None; xlon = None; ylat = None # defaults
    # only for 2D variables!
    shape = self.__dict__['shape']
    if shape is not None and shape >= 2:
      mapSize = self.__dict__['shape'][-2:]
      if len(shape)==2: bands = 1 
      else: bands = np.prod(shape[:-2])
      if projection is not None: # figure out projection 
        assert isinstance(projection,osr.SpatialReference), '\'projection\' has to be a GDAL SpatialReference object.'              
        lgdal = True   
        isProjected =  projection.IsProjected
        if isProjected: 
          assert ('x' in self) and ('y' in self), 'Horizontal axes for projected GDAL variables have to \'x\' and \'y\'.'
          xlon = self.__dict__['x']; ylat = self.__dict__['y']
        else: 
          assert ('lon' in self) and ('lat' in self), 'Horizontal axes for non-projected GDAL variables have to \'lon\' and \'lat\''
          xlon = self.__dict__['lon']; ylat = self.__dict__['lat']    
        assert (self[xlon] in [0,1]) and (self[ylat] in [0,1]), \
          'Horizontal axes (\'lon\' and \'lat\') have to be the innermost indices.'
      else: # can still infer some useful info
        if ('x' in self) and ('y' in self):
          isProjected = True; xlon = self.__dict__['x']; ylat = self.__dict__['y']
        elif ('lon' in self) and ('lat' in self):
          isProjected = False; xlon = self.__dict__['lon']; ylat = self.__dict__['lat']
    # quickly check geotransform
    if geotransform is not None:
      assert len(geotransform) == 6, '\'projection\' has to be a vector or list with 6 elements.'
    else: lgdal = False  
    # finally, also need to have non-zero horizontal dimensions
    lgdal = lgdal and all(mapSize)
    # add projection parameters to instance dict
    self.__dict__['gdal'] = lgdal
    self.__dict__['isProjected'] = isProjected
    self.__dict__['projection'] = projection
    self.__dict__['geotransform'] = geotransform
    self.__dict__['mapSize'] = mapSize
    self.__dict__['bands'] = bands
    self.__dict__['xlon'] = xlon
    self.__dict__['ylat'] = ylat
      
  def getGDAL(self, load=False):
    ''' Method that returns a gdal dataset, ready for use with GDAL routines. '''
    if self.gdal:
      # determine GDAL data type
      if self.dtype == 'float32': gdt = gdal.GDT_Float32
      # create GDAL dataset 
      dataset = ramdrv.Create(self.name, int(self.mapSize[0]), int(self.mapSize[1]), int(self.bands), int(gdt)) 
      # N.B.: for some reason a dataset is always initialized with 6 bands
      # set projection parameters
      dataset.SetGeoTransform(self.geotransform) # does the order matter?
      dataset.SetProjection(self.projection.ExportToWkt()) # is .ExportToWkt() necessary?
      if load:
        data = self.data_array.reshape(self.bands,self.mapSize[0],self.mapSize[1])
        missing = False    
        if self.masked:
          data = data.filled() # default fill value
          missing = data.fill_value
        # assign data
        for i in xrange(self.bands):
          dataset.GetRasterBand(i+1).WriteArray(data[i,:,:])
          if missing: dataset.GetRasterBand(i+1).SetNoDataValue(missing)
    else: dataset = None
    # return dataset
    return dataset
    
  def load(self, data):
    ''' Load new data array. '''
    super(VarGDAL,self).load(data, mask=None)    
    if len(data.shape) >= 2: # 2D or more
      self.__dict__['mapSize'] = data.shape[-2:] # need to update
    else: # less than 2D can't be GDAL enabled
      self.__dict__['mapSize'] = None
      self.__dict__['gdal'] = False

  def unload(self):
    ''' Remove coordinate vector. '''
    super(VarGDAL,self).unload()      
  

## run a test    
if __name__ == '__main__':

  # initialize test objects
  x = Axis(name='x', units='none', coord=(1,5,5))
  y = Axis(name='y', units='none', coord=(1,5,5))
  var = Variable(name='test',units='none',axes=(x,y),data=np.zeros((5,5)),atts=dict(_FillValue=-9999))
    
  # GDAL test
  gdal = var.getGDAL(load=True)
  print 'GDAL projection object:'
  print gdal
  
  # variable test
  print
  var += np.ones(1)
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
