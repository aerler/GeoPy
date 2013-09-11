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
# use exceptions (off by default)
gdal.UseExceptions()

# import all base functionality from PyGeoDat
from geodata.base import Variable, Axis, Dataset
from misc import isEqual, isZero, isFloat, DataError

## function to generate a projection object from a dictionary
# (geographic coordinates are handled automatically, if lat/lon vectors are supplied)

def getProjFromDict(projdict, name='', GeoCS='WGS84'):
  ''' Initialize a projected OSR SpatialReference instance from a dictionary using Proj4 conventions. 
      Valid parameters are documented here: http://trac.osgeo.org/proj/wiki/GenParms
      Projections are described here: http://www.remotesensing.org/geotiff/proj_list/ '''
  # start with projection, which is usually a string
  projstr = '+proj={0:s}'.format(projdict['proj']) 
  # loop over entries
  for key,value in projdict.iteritems():
    if key is not 'proj':
      if not isinstance(key,str): raise TypeError
      if not isinstance(value,float): raise TypeError
      # translate dict entries to string
      projstr = '{0:s} +{1:s}={2:f}'.format(projstr,key,value)
  # initialize
  projection = osr.SpatialReference()
  projection.ImportFromProj4(projstr)
  # more meta data
  projection.SetProjCS(name) # establish that this is a projected system
  projection.SetWellKnownGeogCS(GeoCS) # default reference datum/geoid
  # return finished projection object (geotransform can be inferred from coordinate vectors)
  return projection

## functions to add GDAL functionality

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
      # figure out projection
      if isinstance(projection,dict): projection = getProjFromDict(projection)
      # assume projection is set
      assert isinstance(projection,osr.SpatialReference), '\'projection\' has to be a GDAL SpatialReference object.'              
      isProjected =  projection.IsProjected()
      if isProjected: 
        assert var.hasAxis('x') and var.hasAxis('y'), 'Horizontal axes for projected GDAL variables have to \'x\' and \'y\'.'
        xlon = var.x; ylat = var.y
      else: 
        assert var.hasAxis('lon') and var.hasAxis('lat'), 'Horizontal axes for non-projected GDAL variables have to \'lon\' and \'lat\''
        xlon = var.lon; ylat = var.lat    
    else: # can still infer some useful info
      if var.hasAxis('x') and var.hasAxis('y'):
        isProjected = True; xlon = var.x; ylat = var.y
      elif var.hasAxis('lon') and var.hasAxis('lat'):
        isProjected = False; xlon = var.lon; ylat = var.lat
        projection = osr.SpatialReference() 
        projection.SetWellKnownGeogCS('WGS84') # normal lat/lon projection       
    # if the variable is map-like, add GDAL properties
    if xlon is not None and ylat is not None:
      lgdal = True
      # check axes
      assert (var.axisIndex(xlon) in [var.ndim-1,var.ndim-2]) and (var.axisIndex(ylat) in [var.ndim-1,var.ndim-2]),\
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
#       assert (np.diff(xlon).mean() == dx).all() and (np.diff(ylat).mean() == dy).all(), 'Coordinate vectors have to be uniform!'
      ulx = xlon[0]-dx/2.; uly = ylat[0]-dy/2. # coordinates of upper left corner (same for source and sink)
      # GT(2) & GT(4) are zero for North-up; GT(1) & GT(5) are pixel width and height; (GT(0),GT(3)) is the top left corner
      geotransform = (ulx, dx, 0., uly, 0., dy)
    elif xlon.data or ylat.data:
      # check if GDAL geotransform vector is consistent with coordinate vectors
      assert len(geotransform) == 6, '\'geotransform\' has to be a vector or list with 6 elements.'
      dx = geotransform[1]; dy = geotransform[5]; ulx = geotransform[0]; uly = geotransform[3] 
#       assert isZero(np.diff(xlon)-dx) and isZero(np.diff(ylat)-dy), 'Coordinate vectors have to be compatible with geotransform!'
      assert isEqual(ulx+dx/2,xlon[0]) and isEqual(uly+dy/2,ylat[0]) # coordinates of upper left corner (same for source and sink)       
    else: assert len(geotransform) == 6 and all(isFloat(geotransform)), '\'geotransform\' has to be a vector or list of 6 floating-point numbers.'
    # add new instance attributes (projection parameters)
    var.__dict__['isProjected'] = isProjected    
    var.__dict__['projection'] = projection
    var.__dict__['geotransform'] = geotransform
    var.__dict__['mapSize'] = mapSize
    var.__dict__['bands'] = bands
    var.__dict__['xlon'] = xlon
    var.__dict__['ylat'] = ylat
    
    # append projection info  
    def prettyPrint(self, short=False):
      ''' Add projection information in to string in long format. '''
      string = var.__class__.prettyPrint(self, short=short)
      if not short:
        if var.projection is not None:
          string += '\nProjection: {0:s}'.format(self.projection.ExportToWkt())
      return string
    # add new method to object
    var.prettyPrint = types.MethodType(prettyPrint,var)
    
    def copy(self, **newargs):
      ''' A method to copy the Variable with just a link to the data. '''
      var = self.__class__.copy(self, **newargs) # use class copy() function
      var = addGDAL(var, projection=self.projection, geotransform=self.geotransform) # add GDAL functionality      
      return var
    # add new method to object
    var.copy = types.MethodType(copy,var)
        
    # define GDAL-related 'class methods'  
    def getGDAL(self, load=True):
      ''' Method that returns a gdal dataset, ready for use with GDAL routines. '''
      if self.gdal and self.projection is not None:
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
          data = self.getArray(unmask=True) # get unmasked data
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
    def load(self, data=None, mask=None):
      ''' Load new data array. '''
      var.__class__.load(self, data=data, mask=mask)    
      if len(self.shape) >= 2: # 2D or more
        self.__dict__['mapSize'] = self.shape[-2:] # need to update
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
  
  ## the return value is actually not necessary, since the object is modified immediately
  return var

def DatasetGDAL(dataset, projection=None, geotransform=None):
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
  assert isinstance(dataset,Dataset), 'This function can only be used to add GDAL functionality to \'Variable\' instances!'
  lgdal = False; isProjected = None; xlon = None; ylat = None # defaults
  # only for 2D variables!
  if len(dataset.axes) >= 2: # else not a map-type
    if projection is not None: # figure out projection 
      # figure out projection
      if isinstance(projection,dict): projection = getProjFromDict(projection)
      # assume projection is set
      assert isinstance(projection,osr.SpatialReference), '\'projection\' has to be a GDAL SpatialReference object.'              
      isProjected =  projection.IsProjected
      if isProjected: 
        assert dataset.hasAxis('x') and dataset.hasAxis('y'), 'Horizontal axes for projected GDAL variables have to \'x\' and \'y\'.'
        xlon = dataset.x; ylat = dataset.y
      else: 
        assert dataset.hasAxis('lon') and dataset.hasAxis('lat'), 'Horizontal axes for non-projected GDAL variables have to \'lon\' and \'lat\''
        xlon = dataset.lon; ylat = dataset.lat    
    else: # can still infer some useful info
      if dataset.hasAxis('x') and dataset.hasAxis('y'):
        isProjected = True; xlon = dataset.x; ylat = dataset.y
      elif dataset.hasAxis('lon') and dataset.hasAxis('lat'):
        isProjected = False; xlon = dataset.lon; ylat = dataset.lat
        projection = osr.SpatialReference() 
        projection.SetWellKnownGeogCS('WGS84') # normal lat/lon projection       
    # if the variable is map-like, add GDAL properties
    if xlon is not None and ylat is not None:
      lgdal = True
      # check axes
      if isProjected: assert isinstance(xlon,Axis) and isinstance(ylat,Axis), 'Error: attributes \'x\' and \'y\' have to be axes.'
      else: assert isinstance(xlon,Axis) and isinstance(ylat,Axis), 'Error: attributes \'lon\' and \'lat\' have to be axes.'   

  # modify Variable instance
  dataset.__dict__['gdal'] = lgdal # all variables have this after going through this process
  if lgdal:
    
    # infer or check geotransform
    if geotransform is None: 
      # infer GDAL geotransform vector from  coordinate vectors (axes)
      dx = xlon[1]-xlon[0]; dy = ylat[1]-ylat[0]
#       assert (np.diff(xlon) == dx).all() and (np.diff(ylat) == dy).all(), 'Coordinate vectors have to be uniform!'
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
    dataset.__dict__['isProjected'] = isProjected    
    dataset.__dict__['projection'] = projection
    dataset.__dict__['geotransform'] = geotransform
    dataset.__dict__['xlon'] = xlon
    dataset.__dict__['ylat'] = ylat
    
    # append projection info  
    def __str__(self, short=False):
      ''' Add projection information in to string in long format. '''
      string = super(dataset.__class__,self).__str__(short=short)
      if not short:
        string += '\nProjection: {0:s}'.format(self.projection.ExportToWkt)
    # add new method to object
    dataset.__str__ = types.MethodType(__str__,dataset)
    
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
          data = self.getArray(unmask=True) # get unmasked data
          data = data.reshape(self.bands,self.mapSize[0],self.mapSize[1]) # reshape to fit bands
          # assign data
          for i in xrange(self.bands):
            dataset.GetRasterBand(i+1).WriteArray(data[i,:,:])
            if self.masked: dataset.GetRasterBand(i+1).SetNoDataValue(float(self.fillValue))
      else: dataset = None
      # return dataset
      return dataset
    # add new method to object
    dataset.getGDAL = types.MethodType(getGDAL,dataset)
    
    # append projection info  
    def prettyPrint(self, short=False):
      ''' Add projection information in to string in long format. '''
      string = super(dataset.__class__,self).prettyPrint(short=short)
      if not short:
        if dataset.projection is not None:
          string += '\nProjection: {0:s}'.format(self.projection.ExportToWkt())
      return string
    # add new method to object
    dataset.prettyPrint = types.MethodType(prettyPrint,dataset)
            
  ## the return value is actually not necessary, since the object is modified immediately
  return dataset
  

## run a test    
if __name__ == '__main__':

  pass