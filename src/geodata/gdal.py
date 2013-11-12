'''
Created on 2013-08-23

A module that provides GDAL functionality to GeoData datasets and variables, 
and exposes some more GDAL functionality, such as regriding.

The GDAL functionality for Variables is implemented as a decorator that adds GDAL attributes 
and methods to an existing object/class instance.    

@author: Andre R. Erler, GPL v3
'''

import numpy as np
import numpy.ma as ma
import types  # needed to bind functions to objects

# gdal imports
from osgeo import gdal, osr
# register RAM driver
ramdrv = gdal.GetDriverByName('MEM')
# use exceptions (off by default)
gdal.UseExceptions()

# import all base functionality from PyGeoDat
from geodata.base import Variable, Axis, Dataset
from geodata.misc import printList, isEqual, isInt, isFloat, isNumber, DataError, AxisError, GDALError


# # utility functions and classes to handle projection information and related meta data

class GridDefinition(object):
  ''' 
    A class that encapsulates all necessary information to fully define a grid.
    That includes GDAL spatial references and map-Axis instances with coordinates.
  '''
  name = '' # a name for the grid...
  scale = None # approximate resolution of the grid in degrees at the domain center
  projection = None  # GDAL SpatialReference instance
  isProjected = None  # logical; indicates whether the coordiante system is projected or geographic 
  xlon = None  # Axis instance; the west-east axis
  ylat = None  # Axis instance; the south-north axis
  geotransform = None  # 6-element vector defining a GDAL GeoTransform
  size = None  # tuple, defining the size of the x/lon and y/lat axes
      
  def __init__(self, name='', projection=None, geotransform=None, size=None, xlon=None, ylat=None):
    ''' This class can be initialized in several ways. Some form of projections has to be defined (using a 
        GDAL SpatialReference, WKT, EPSG code, or Proj4 conventions), or a simple geographic (lat/lon) 
        coordinate system will be assumed. 
        The horizontal grid can be defined by either specifying the geotransform and the size of the 
        horizontal dimensions (and standard axes will be constructed from this information), or the horizontal 
        Axis instances can be specified directly (and the map size and geotransform will be inferred from the 
        axes). '''
    self.name = name # just a name...
    # check projection (default is WSG84)
    if isinstance(projection, osr.SpatialReference):
      gdalsr = projection # use as is
    else:
      gdalsr = osr.SpatialReference() 
      gdalsr.SetWellKnownGeogCS('WGS84')           
      if projection is None: 
        pass  # normal lat/lon projection
      elif isinstance(projection, dict): 
        gdalsr = getProjFromDict(projdict=projection, name='', GeoCS='WGS84')  # get projection from dictionary
      elif isinstance(projection, basestring):
        gdalsr.ImportFromWkt(projection)  # from Well-Known-Text
      elif isinstance(projection, np.number):
        gdalsr.ImportFromEpsg(projection)  # from EPSG code    
      else: 
        raise TypeError, '\'projection\' has to be a GDAL SpatialReference object.'              
    # set projection attributes
    self.projection = gdalsr
    self.isProjected = gdalsr.IsProjected()
    # figure out geotransform and axes (axes have precedence)
    if xlon is not None or ylat is not None:
      # use axes and check consistency with geotransform and size, if applicable
      if xlon is None or ylat is None: raise TypeError
      if not isinstance(xlon, Axis) or not isinstance(ylat, Axis): raise TypeError  
      if size is not None:
        if not (len(xlon), len(ylat)) == size: raise AxisError
      else: size = (len(xlon), len(ylat))
      geotransform = getGeotransform(xlon=xlon, ylat=ylat, geotransform=geotransform)  
    elif geotransform is not None and size is not None:
      # generate new axes from size and geotransform
      if not isinstance(geotransform, (list, tuple)) and isNumber(geotransform) and len(geotransform) == 6: raise TypeError
      if not isinstance(size, (list, tuple)) and isInt(geotransform) and len(geotransform) == 2: raise TypeError
      xlon, ylat = getAxes(geotransform, xlen=size[0], ylen=size[1], projected=self.isProjected)
    # N.B.: [x_0, dx, 0, y_0, 0, dy]
    #       GT(0),GT(3) are the coordinates of the bottom left corner
    #       GT(1) & GT(5) are pixel width and height
    #       GT(2) & GT(4) are usually zero for North-up, non-rotated maps
    # estimate scale in degrees
    if self.isProjected:
      latlon = osr.SpatialReference() 
      latlon.SetWellKnownGeogCS('WGS84') # a normal lat/lon coordinate system
      tx = osr.CoordinateTransformation(gdalsr,latlon)
      frac = 1./5. # the fraction that is used to calculate the effective resolution (at the domain center)
      xs = int(size[0]*(0.5-frac/2.)); ys = int(size[1]*(0.5-frac/2.))
      xe = int(size[0]*(0.5+frac/2.)); ye = int(size[1]*((0.5+frac/2.)))
      (llx,lly,llz) = tx.TransformPoint(float(xlon.coord[xs]),float(ylat.coord[ys]))
      (urx,ury,urz) = tx.TransformPoint(float(xlon.coord[xe]),float(ylat.coord[ye]))
      # N.B.: for some reason GDAL is very sensitive to type and does not understand numpy types
      dlon = ( urx - llx ) / ( xe - xs ); dlat = ( ury - lly ) / ( ye - ys )       
      self.scale = ( dlon + dlat ) / 2
    else:
      self.scale = ( geotransform[1] + geotransform[5] ) / 2 # pretty straight forward 
    # set geotransform/axes attributes
    self.xlon = xlon
    self.ylat = ylat
    self.geotransform = geotransform
    self.size = size
    
  def __str__(self):
    ''' A string representation of the grid definition '''
    string = '{0:s}   {1:s}\n'.format(self.__class__.__name__,self.name)
    string += 'Size: {0:s}\n'.format(printList(self.size))
    string += 'GeoTransform: {0:s}\n'.format(printList(self.geotransform))
    string += '  {0:s}\n'.format(self.xlon.prettyPrint(short=True))
    string += '  {0:s}\n'.format(self.ylat.prettyPrint(short=True))
    string += 'Projection: {0:s}\n'.format(self.projection.ExportToWkt())
    return string
  
  def __getstate__(self):
    ''' support pickling, necessary for multiprocessing: GDAL is not pickable '''
    pickle = self.__dict__.copy()
    pickle['_projection'] =  self.projection.ExportToWkt()  # to Well-Known-Text format
    del pickle['projection'] # remove offensive GDAL object
    # return instance dict to pickle
    return pickle
  
  def __setstate__(self, pickle):
    ''' support pickling, necessary for multiprocessing: GDAL is not pickable '''
    self.projection = osr.SpatialReference() 
    self.projection.SetWellKnownGeogCS('WGS84')           
    self.projection.ImportFromWkt(pickle['_projection'])  # from Well-Known-Text
    del pickle['_projection'] # not actually an attribute
    self.__dict__.update(pickle)
    
    
def getGridDef(var):
  ''' Get a GridDefinition instance from a GDAL enabled Variable of Dataset. '''
  if 'gdal' not in var.__dict__: raise GDALError
  # instantiate GridDefinition
  return GridDefinition(name=var.name+'_grid', projection=var.projection, geotransform=var.geotransform, 
                        size=var.mapSize, xlon=var.xlon, ylat=var.ylat)


# determine GDAL interpolation
def gdalInterp(interpolation):
  if interpolation == 'bilinear': gdal_interp = gdal.GRA_Bilinear
  elif interpolation == 'nearest': gdal_interp = gdal.GRA_NearestNeighbour
  elif interpolation == 'lanczos': gdal_interp = gdal.GRA_Lanczos
  elif interpolation == 'convolution': gdal_interp = gdal.GRA_Cubic # cubic convolution
  elif interpolation == 'cubicspline': gdal_interp = gdal.GRA_CubicSpline # cubic spline
  else: raise GDALError, 'Unknown interpolation method: %s'%interpolation
  return gdal_interp
         

def getProjFromDict(projdict, name='', GeoCS='WGS84', convention='Proj4'):
  ''' Initialize a projected OSR SpatialReference instance from a dictionary using Proj4 conventions. 
      Valid parameters are documented here: http://trac.osgeo.org/proj/wiki/GenParms
      Projections are described here: http://www.remotesensing.org/geotiff/proj_list/ '''
  # initialize
  projection = osr.SpatialReference()
  # interpret dictionary according to convention
  if convention == 'Proj4':
    # start with projection, which is usually a string
    projstr = '+proj={0:s}'.format(projdict['proj']) 
    # loop over entries
    for key, value in projdict.iteritems():
      if key is not 'proj':
        if not isinstance(key, str): raise TypeError
        if not isinstance(value, (float,np.inexact)): raise TypeError
        # translate dict entries to string
        projstr = '{0:s} +{1:s}={2:f}'.format(projstr, key, float(value))
    # load projection from proj4 string
    projection.ImportFromProj4(projstr)
  else:
    raise NotImplementedError
  # more meta data
  projection.SetProjCS(name)  # establish that this is a projected system
  projection.SetWellKnownGeogCS(GeoCS)  # default reference datum/geoid
  # return finished projection object (geotransform can be inferred from coordinate vectors)
  return projection


def getProjection(var, projection=None):
  ''' Function to infere GDAL parameters from a Variable or Dataset '''
  if not isinstance(var, (Variable, Dataset)): raise TypeError
  # infer map axes and projection parameters
  if projection is None:  # can still infer some useful info
    if var.hasAxis('x') and var.hasAxis('y'):
      isProjected = True; xlon = var.x; ylat = var.y
    elif var.hasAxis('lon') and var.hasAxis('lat'):
      isProjected = False; xlon = var.lon; ylat = var.lat
      projection = osr.SpatialReference() 
      projection.SetWellKnownGeogCS('WGS84')  # normal lat/lon projection
    else: xlon = None; ylat = None
  else: 
    # figure out projection
    if isinstance(projection, dict): projection = getProjFromDict(projection)
    # assume projection is set
    if not isinstance(projection, osr.SpatialReference): 
      raise TypeError, '\'projection\' has to be a GDAL SpatialReference object.'              
    isProjected = projection.IsProjected()
    if isProjected: 
      if not var.hasAxis('x') and var.hasAxis('y'): 
        raise AxisError, 'Horizontal axes for projected GDAL variables have to \'x\' and \'y\'.'
      if var.hasAxis('x') and var.hasAxis('y'):
        xlon = var.x; ylat = var.y
      else: xlon = None; ylat = None
    else: 
      if not var.hasAxis('lon') and var.hasAxis('lat'):
        raise AxisError, 'Horizontal axes for non-projected GDAL variables have to \'lon\' and \'lat\''
      if var.hasAxis('lon') and var.hasAxis('lat'):
        xlon = var.lon; ylat = var.lat
      else: xlon = None; ylat = None    
  # if the variable is map-like, add GDAL properties
  if xlon is not None and ylat is not None:
    lgdal = True
    # check axes
    if isProjected: axstr = "'x' and 'y'"
    else: axstr = "'lon' and 'lat'"
    if isinstance(var.axes, (tuple, list)):  # only check this, if axes are ordered, as in Variables
      dmtpl = (var.ndim -1, var.ndim -2)
      if (var.axisIndex(xlon) not in dmtpl) or (var.axisIndex(ylat) not in dmtpl):
        raise AxisError, "Horizontal axes (%s) have to be the innermost indices."%axstr
    if not isinstance(xlon, Axis) and not isinstance(ylat, Axis): 
      raise AxisError, "Error: attributes %s have to be axes."%axstr
  else: lgdal = False   
  # return
  return lgdal, projection, isProjected, xlon, ylat


def getAxes(geotransform, xlen=0, ylen=0, projected=False):
  ''' Generate a set of axes based on the given geotransform and length information. '''
  if projected: 
    xatts = dict(name='x', long_name='easting', units='m')
    yatts = dict(name='y', long_name='northing', units='m')
  else: 
    xatts = dict(name='lon', long_name='longitude', units='deg E')
    yatts = dict(name='lat', long_name='latitude', units='deg N')
  # create axes    
  (x0, dx, s, y0, t, dy) = geotransform; del s,t
  xlon = Axis(length=xlen, coord=np.arange(x0 + dx / 2, x0+xlen*dx, dx), atts=xatts)
  ylat = Axis(length=ylen, coord=np.arange(y0 + dy / 2, y0+ylen*dy, dy), atts=yatts)
  # return tuple of axes
  return xlon, ylat


def getGeotransform(xlon=None, ylat=None, geotransform=None):
  ''' Function to check or infer GDAL geotransform from coordinate axes. '''
  if geotransform is None:  # infer geotransform from axes
    if not isinstance(ylat, Axis) or not isinstance(xlon, Axis): raise TypeError     
    if xlon.data and ylat.data:
      # infer GDAL geotransform vector from  coordinate vectors (axes)
      dx = xlon[1] - xlon[0]; dy = ylat[1] - ylat[0]
      # assert (np.diff(xlon).mean() == dx).all() and (np.diff(ylat).mean() == dy).all(), 'Coordinate vectors have to be uniform!'
      ulx = xlon[0] - dx / 2.; uly = ylat[0] - dy / 2.  # coordinates of upper left corner (same for source and sink)
      # GT(2) & GT(4) are zero for North-up; GT(1) & GT(5) are pixel width and height; (GT(0),GT(3)) is the top left corner
      geotransform = (ulx, dx, 0., uly, 0., dy)
    else: raise DataError, "Coordinate vectors are required to infer GDAL geotransform vector."
  else:  # check given geotransform
    if xlon.data or ylat.data:
      # check if GDAL geotransform vector is consistent with coordinate vectors
      assert len(geotransform) == 6, '\'geotransform\' has to be a vector or list with 6 elements.'
      dx = geotransform[1]; dy = geotransform[5]; ulx = geotransform[0]; uly = geotransform[3] 
      # assert isZero(np.diff(xlon)-dx) and isZero(np.diff(ylat)-dy), 'Coordinate vectors have to be compatible with geotransform!'
      assert isEqual(ulx + dx / 2, xlon[0]) and isEqual(uly + dy / 2, ylat[0])  # coordinates of upper left corner (same for source and sink)       
    else: 
      assert len(geotransform) == 6 and all(isFloat(geotransform)), '\'geotransform\' has to be a vector or list of 6 floating-point numbers.'
  # return results
  return geotransform


# # functions to add GDAL functionality to existing Variable and Dataset instances

def addGDALtoVar(var, projection=None, geotransform=None):
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
  if not isinstance(var, Variable): 
    raise TypeError, 'This function can only be used to add GDAL functionality to \'Variable\' instances!'
  # only for 2D variables!
  if var.ndim >= 2:  # map-type: GDAL potential
    # infer or check projection and related parameters       
    lgdal, projection, isProjected, xlon, ylat = getProjection(var, projection=projection)
  else: lgdal = False
  # add result to Variable instance
  var.__dict__['gdal'] = lgdal  # all variables have this after going through this process

  if lgdal:    
    # determine gdal-relevant shape parameters
    mapSize = var.shape[-2:]
    if not all(mapSize): raise AxisError, 'Horizontal dimensions have to be of finite, non-zero length.'
    if var.ndim == 2: bands = 1 
    else: bands = np.prod(var.shape[:-2])          
    # infer or check geotransform
    geotransform = getGeotransform(xlon, ylat, geotransform=geotransform)
    # add new instance attributes (projection parameters)
    var.__dict__['isProjected'] = isProjected    
    var.__dict__['projection'] = projection
    var.__dict__['geotransform'] = geotransform
    var.__dict__['mapSize'] = mapSize
    var.__dict__['bands'] = bands
    var.__dict__['xlon'] = xlon
    var.__dict__['ylat'] = ylat
    
    # get grid definition object
    var.getGridDef = types.MethodType(getGridDef, var)
    
    # append projection info  
    def prettyPrint(self, short=False):
      ''' Add projection information in to string in long format. '''
      string = var.__class__.prettyPrint(self, short=short)
      if not short:
        if var.projection is not None:
          string += '\nProjection: {0:s}'.format(self.projection.ExportToWkt())
      return string
    # add new method to object
    var.prettyPrint = types.MethodType(prettyPrint, var)
    
    def copy(self, projection=None, geotransform=None, **newargs):
      ''' A method to copy the Variable with just a link to the data. '''
      var = self.__class__.copy(self, **newargs)  # use class copy() function
      # handle geotransform
      if geotransform is None:
        if 'axes' in newargs:  # if axes were changed, geotransform can change!
          geotransform = None  # infer from new axes
#           if var.hasAxis(self.xlon) and var.hasAxis(self.ylat): geotransform = self.geotransform
        else: geotransform = self.geotransform
      # handle projection
      if projection is None: projection = self.projection
      var = addGDALtoVar(var, projection=projection, geotransform=geotransform)  # add GDAL functionality      
      return var
    # add new method to object
    var.copy = types.MethodType(copy, var)
        
    # define GDAL-related 'class methods'  
    def getGDAL(self, load=True, allocate=True, fillValue=None):
      ''' Method that returns a gdal dataset, ready for use with GDAL routines. '''
      lperi = False
      if self.gdal and self.projection is not None:
        if load:
          lperi = False
          if not self.data: self.load()
          if not self.data: raise DataError, 'Need data in Variable instance in order to load data into GDAL dataset!'
          data = self.getArray(unmask=True)  # get unmasked data
          data = data.reshape(self.bands, self.mapSize[0], self.mapSize[1])  # reshape to fit bands
          if lperi: 
            tmp = np.zeros((self.bands, self.mapSize[0], self.mapSize[1]+1))
            tmp[:,:,0:-1] = data; tmp[:,:,-1] = data[:,:,0]
            data = tmp
        elif allocate: 
          if fillValue is None and self.fillValue is not None: fillValue = self.fillValue  # use default 
          if self.fillValue is None: fillValue = ma.default_fill_value(self.dtype)
          data = np.zeros((self.bands,) + self.mapSize, dtype=self.dtype) + fillValue
        # determine GDAL data type        
        if self.dtype == 'float32': gdt = gdal.GDT_Float32
        elif self.dtype == 'float64': gdt = gdal.GDT_Float64
        elif self.dtype == 'int16': gdt = gdal.GDT_Int16
        elif self.dtype == 'int32': gdt = gdal.GDT_Int32
        elif np.issubdtype(self.dtype,(float,np.inexact)):
          data = data.astype('f4'); gdt = gdal.GDT_Float32          
        elif np.issubdtype(self.dtype,(int,np.integer)):
          data = data.astype('i2'); gdt = gdal.GDT_Int16  
        elif np.issubdtype(self.dtype,(bool,np.bool)):
          data = data.astype('i2'); gdt = gdal.GDT_Int16  
        else: raise TypeError, 'Cannot translate numpy data type into GDAL data type!'
        #print self.name, self.dtype, data.dtype
        # create GDAL dataset 
        xe = len(self.xlon); ye = len(self.ylat) 
        if lperi: dataset = ramdrv.Create(self.name, int(xe)+1, int(ye), int(self.bands), int(gdt))
        else: dataset = ramdrv.Create(self.name, int(xe), int(ye), int(self.bands), int(gdt)) 
        # N.B.: for some reason a dataset is always initialized with 6 bands
        # set projection parameters
        dataset.SetGeoTransform(self.geotransform)  # does the order matter?
        dataset.SetProjection(self.projection.ExportToWkt())  # is .ExportToWkt() necessary?        
        if load or allocate: 
          # assign data
          for i in xrange(self.bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i, :, :])
            if self.masked: dataset.GetRasterBand(i + 1).SetNoDataValue(float(self.fillValue))
      else: dataset = None
      # return dataset
      return dataset
    # add new method to object
    var.getGDAL = types.MethodType(getGDAL, var)
    
    def loadGDAL(self, dataset, mask=True, fillValue=None):
      ''' Load data from the bands of a GDAL dataset into the variable. '''
      # check input
      if not isinstance(dataset, gdal.Dataset): raise TypeError
      if self.gdal:
        # get data field
        if self.bands == 1: data = dataset.ReadAsArray()[:, :]  # for 2D fields
        else: data = dataset.ReadAsArray()[0:self.bands, :, :]  # ReadAsArray(0,0,xe,ye)
        # convert data, if necessary
        if self.dtype is not data.dtype: data = data.astype(self.dtype)          
#         print data.__class__
#         print self.masked, self.fillValue
        # adjust shape (unravel bands)
        if self.ndim == 2: data = data.squeeze()
        else: data = data.reshape(self.shape)
        # shift missing value to zero (for some reason ReprojectImage treats missing values as 0)                      
        if mask:
          # mask array where zero (in accord with ReprojectImage convention)
          if fillValue is None and self.fillValue is not None: fillValue = self.fillValue  # use default 
          if self.fillValue is None: fillValue = ma.default_fill_value(data.dtype)
          data = ma.masked_values(data, fillValue)
        # load data)
        self.load(data=data)
      # return verification
      return self.data
    # add new method to object
    var.loadGDAL = types.MethodType(loadGDAL, var)    
    
    # update new instance attributes
    def load(self, data=None, mask=None):
      ''' Load new data array. '''
      var.__class__.load(self, data=data, mask=mask)    
      if len(self.shape) >= 2:  # 2D or more
        self.__dict__['mapSize'] = self.shape[-2:]  # need to update
      else:  # less than 2D can't be GDAL enabled
        self.__dict__['mapSize'] = None
        self.__dict__['gdal'] = False
    # add new method to object
    var.load = types.MethodType(load, var)
    
    # maybe needed in the future...
    def unload(self):
      ''' Remove coordinate vector. '''
      var.__class__.unload(self)      
    # add new method to object
    var.unload = types.MethodType(unload, var)
  
  # # the return value is actually not necessary, since the object is modified immediately
  return var

def addGDALtoDataset(dataset, projection=None, geotransform=None):
  ''' 
    A function that adds GDAL-based geographic projection features to an existing Dataset instance
    and all its Variables.
    
    New Instance Attributes: 
      gdal = False # whether or not this instance possesses any GDAL functionality and is map-like
      isProjected = False # whether lat/lon spherical (False) or a geographic projection (True)
      projection = None # a GDAL spatial reference object
      geotransform = None # a GDAL geotransform vector (can e inferred from coordinate vectors)
      xlon = None # West-East axis
      ylat = None # South-North axis  
  '''
  # check some special conditions
  assert isinstance(dataset, Dataset), 'This function can only be used to add GDAL functionality to a \'Dataset\' instance!'
  # only for 2D variables!
  if len(dataset.axes) >= 2:  # else not a map-type
    # infer or check projection and related parameters       
    lgdal, projection, isProjected, xlon, ylat = getProjection(dataset, projection=projection)
  else: lgdal = False
  # modify Variable instance
  dataset.__dict__['gdal'] = lgdal  # all variables have this after going through this process
  
  if lgdal:  
    # infer or check geotransform
    geotransform = getGeotransform(xlon, ylat, geotransform=geotransform)    
    # add new instance attributes (projection parameters)
    dataset.__dict__['isProjected'] = isProjected    
    dataset.__dict__['projection'] = projection
    dataset.__dict__['geotransform'] = geotransform
    dataset.__dict__['xlon'] = xlon
    dataset.__dict__['ylat'] = ylat
    dataset.__dict__['mapSize'] = (len(xlon),len(ylat))
    
    # add GDAL functionality to all variables!
    for var in dataset.variables.values():
      # call variable 'constructor' for all variables
      var = addGDALtoVar(var, projection=dataset.projection, geotransform=dataset.geotransform)
      # check result
      assert (var.ndim >= 2 and var.hasAxis(dataset.xlon) and var.hasAxis(dataset.ylat)) == var.gdal    
    
    # get grid definition object
    dataset.getGridDef = types.MethodType(getGridDef, dataset)
        
    # append projection info  
    def prettyPrint(self, short=False):
      ''' Add projection information in to string in long format. '''
      string = dataset.__class__.prettyPrint(self, short=short)
      if not short:
        if dataset.projection is not None:
          string += '\nProjection: {0:s}'.format(self.projection.ExportToWkt())
      return string
    # add new method to object
    dataset.prettyPrint = types.MethodType(prettyPrint, dataset)

    def copy(self, griddef=None, projection=None, geotransform=None, **newargs):
      ''' A method to copy the Dataset with just a link to the data. Also supports new projections. '''
      # griddef supersedes all other arguments
      if griddef is not None:
        projection = griddef.projection
        geotransform = griddef.geotransform
        if not 'axes' in newargs: newargs['axes'] = dict()
        newargs['axes'][self.xlon.name] = griddef.xlon
        newargs['axes'][self.ylat.name] = griddef.ylat
      # invoke class copy() function to copy dataset
      dataset = self.__class__.copy(self, **newargs)
      # handle geotransform
      if geotransform is None:
        if 'axes' in newargs:  # if axes were changed, geotransform can change!
          geotransform = None  # infer from new axes
        else: geotransform = self.geotransform
      # handle projection
      if projection is None: projection = self.projection
      dataset = addGDALtoDataset(dataset, projection=projection, geotransform=geotransform)  # add GDAL functionality      
      return dataset
    # add new method to object
    dataset.copy = types.MethodType(copy, dataset)
            
  # # the return value is actually not necessary, since the object is modified immediately
  return dataset
  

# # run a test    
if __name__ == '__main__':

  pass
