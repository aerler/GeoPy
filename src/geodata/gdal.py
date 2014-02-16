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
import os, pickle
# gdal imports
from osgeo import gdal, osr, ogr
# register RAM driver
ramdrv = gdal.GetDriverByName('MEM')
# use exceptions (off by default)
gdal.UseExceptions()
osr.UseExceptions()
ogr.UseExceptions()

# import all base functionality from PyGeoDat
from geodata.base import Variable, Axis, Dataset
from geodata.misc import printList, isEqual, isInt, isFloat, isNumber, DataError, AxisError, GDALError, DatasetError


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
  geolocator = False # whether or not geolocator arrays are available
  lon2D = None # 2D field of longitude at each grid point
  lat2D = None # 2D field of latitude at each grid point
      
  def __init__(self, name='', projection=None, geotransform=None, size=None, xlon=None, ylat=None, geolocator=True):
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
    # calculate projection properties    
    if self.isProjected:
      latlon = osr.SpatialReference() 
      latlon.SetWellKnownGeogCS('WGS84') # a normal lat/lon coordinate system
      tx = osr.CoordinateTransformation(gdalsr,latlon)      
      # estimate scale in degrees
      frac = 1./5. # the fraction that is used to calculate the effective resolution (at the domain center)
      xs = int(size[0]*(0.5-frac/2.)); ys = int(size[1]*(0.5-frac/2.))
      xe = int(size[0]*(0.5+frac/2.)); ye = int(size[1]*((0.5+frac/2.)))
      (llx,lly,llz) = tx.TransformPoint(xlon.coord[xs].astype(np.float64),ylat.coord[ys].astype(np.float64))
      (urx,ury,urz) = tx.TransformPoint(xlon.coord[xe].astype(np.float64),ylat.coord[ye].astype(np.float64))
      # N.B.: for some reason GDAL is very sensitive to type and does not understand numpy types
      dlon = ( urx - llx ) / ( xe - xs ); dlat = ( ury - lly ) / ( ye - ys )       
      self.scale = ( dlon + dlat ) / 2
      # add geolocator arrays
      if geolocator:
        x2D, y2D = np.meshgrid(xlon.coord, ylat.coord) # if we have x/y arrays
        xx = x2D.flatten().astype(np.float64); yy = y2D.flatten().astype(np.float64)
        lon2D = np.zeros_like(xx); lat2D = np.zeros_like(yy) 
        for i in xrange(xx.size):
          (lon2D[i],lat2D[i],zzz) = tx.TransformPoint(xx[i],yy[i])
        # N.B.: apparently TransformPoints does not work at the moment... and the loop is not too bad...
        #(lon2D,lat2D,zzz) = tx.TransformPoints((x2D.flatten().astype(np.float64),y2D.flatten().astype(np.float64))) 
        lon2D = lon2D.reshape(x2D.shape); lat2D = lat2D.reshape(y2D.shape)
    else:
      self.scale = ( geotransform[1] + geotransform[5] ) / 2 # pretty straight forward
      if geolocator:
        lon2D, lat2D = np.meshgrid(xlon.coord, ylat.coord) # if we have x/y arrays
    # set geotransform/axes attributes
    self.xlon = xlon
    self.ylat = ylat
    self.geotransform = geotransform
    self.size = size
    self.geolocator = geolocator
    if geolocator: self.lon2D = lon2D; self.lat2D = lat2D
    else: self.lon2D = None; self.lat2D = None
    
  def getProjection(self):
    ''' Convenience method that emulates behavior of the function of the same name '''
    return self.projection, self.isProjected, self.xlon, self.ylat
    
  def __str__(self):
    ''' A string representation of the grid definition '''
    string = '{0:s}   {1:s}\n'.format(self.__class__.__name__,self.name)
    string += 'Size: {0:s}\n'.format(printList(self.size))
    string += 'GeoTransform: {0:s}\n'.format(printList(self.geotransform))
    string += '  {0:s}\n'.format(self.xlon.prettyPrint(short=True))
    string += '  {0:s}\n'.format(self.ylat.prettyPrint(short=True))
    string += 'Projection: {0:s}\n'.format(self.projection.ExportToWkt())
    if self.geolocator:
      string += 'Geolocator Center (lon/lat): {0:3.1f} / {1:3.1f}\n'.format(self.lon2D.mean(),self.lat2D.mean())
    return string
  
  def __getstate__(self):
    ''' support pickling, necessary for multiprocessing: GDAL is not pickable '''
    pickle = self.__dict__.copy()
    # handle projection
    pickle['_projection'] =  self.projection.ExportToWkt()  # to Well-Known-Text format
    del pickle['projection'] # remove offensive GDAL object
    # handle axes
    pickle['_geotransform'] = self.geotransform
    pickle['_isProjected'] = self.isProjected
    pickle['_xlon'] = len(self.xlon) 
    pickle['_ylat'] = len(self.ylat)
    del pickle['geotransform'], pickle['isProjected'], pickle['xlon'], pickle['ylat']
    # return instance dict to pickle
    return pickle
  
  def __setstate__(self, pickle):
    ''' support pickling, necessary for multiprocessing: GDAL is not pickable '''
    # handle projection
    self.projection = osr.SpatialReference() 
    self.projection.SetWellKnownGeogCS('WGS84')           
    self.projection.ImportFromWkt(pickle['_projection'])  # from Well-Known-Text
    del pickle['_projection'] # not actually an attribute
    # handle axes
    self.geotransform = pickle['_geotransform']
    self.isProjected = pickle['_isProjected']
    xlon, ylat = getAxes(geotransform=self.geotransform, xlen=pickle['_xlon'], ylen=pickle['_ylat'], 
                         projected=self.isProjected)
    self.xlon = xlon; self.ylat = ylat
    del pickle['_geotransform'], pickle['_isProjected'], pickle['_xlon'], pickle['_ylat']
    # update instance dict with pickle dict
    self.__dict__.update(pickle)
    
    
def getGridDef(var):
  ''' Get a GridDefinition instance from a GDAL enabled Variable of Dataset. '''
  if 'gdal' not in var.__dict__: raise GDALError
  # instantiate GridDefinition
  return GridDefinition(name=var.name+'_grid', projection=var.projection, geotransform=var.geotransform, 
                        size=var.mapSize, xlon=var.xlon, ylat=var.ylat)


# function to load pickled grid definitions
griddef_pickle = '{0:s}_griddef.pickle' # file pattern for pickled grids
def loadPickledGridDef(grid=None, res=None, filename=None, folder=None, check=True):
  ''' function to load pickled datasets '''
  if grid is not None and not isinstance(grid,basestring): raise TypeError
  if res is not None and not isinstance(res,basestring): raise TypeError
  if filename is not None and not isinstance(filename,basestring): raise TypeError
  if folder is not None and not isinstance(folder,basestring): raise TypeError
  # figure out filename
  if filename is None:
    tmp = '{0:s}_{1:s}'.format(grid,res) if res else grid
    filename = griddef_pickle.format(tmp)
  if folder is not None: 
    filename = '{0:s}/{1:s}'.format(folder,filename)
  # load pickle
  if os.path.exists(filename):
    filehandle = open(filename, 'r')
    griddef = pickle.load(filehandle)
    filehandle.close()
  elif check: 
    raise IOError, "GridDefinition pickle file '{0:s}' not found!".format(filename) 
  else:
    griddef = None
  # return
  return griddef


# a utility function
def addGeoLocator(dataset, griddef=None, lcheck=True, asNC=True, lgdal=False, lreplace=False):
  ''' add 2D geolocator arrays to geographic or projected datasets '''
  # add geolocator arrays as variables
  if griddef is None: griddef = getGridDef(dataset) # make temporary griddef from dataset      
  axes = (griddef.ylat,griddef.xlon)
  # add longitude field
  if lreplace or not dataset.hasVariable('lon2D'):
    lon2D = Variable('lon2D', units='deg E', axes=axes, data=griddef.lon2D)
    if dataset.hasVariable('lon2D'): dataset.replaceVariable(lon2D, deepcopy=True, asNC=asNC)
    else: dataset.addVariable(lon2D, deepcopy=True, asNC=asNC)
  elif lcheck: raise DatasetError
  # add latitude field
  if lreplace or not dataset.hasVariable('lat2D'):
    lat2D = Variable('lat2D', units='deg N', axes=axes, data=griddef.lat2D)
    if dataset.hasVariable('lat2D'): dataset.replaceVariable(lat2D, deepcopy=True, asNC=asNC)
    else: dataset.addVariable(lat2D, deepcopy=True)
  elif lcheck: raise DatasetError
  # rerun GDAL initialization, so that the new arrays are GDAL enabled    
  if lgdal: dataset = addGDALtoDataset(dataset, projection=dataset.projection, geotransform=dataset.geotransform)
  # return dataset
  return dataset


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
    else: xlon = None; ylat = None; isProjected = None
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
        raise AxisError, 'Horizontal axes for non-projected GDAL variables have to be \'lon\' and \'lat\''
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
  xlon = Axis(length=xlen, coord=np.arange(x0 + dx/2., x0+xlen*dx, dx), atts=xatts)
  ylat = Axis(length=ylen, coord=np.arange(y0 + dy/2., y0+ylen*dy, dy), atts=yatts)
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

def addGDALtoVar(var, griddef=None, projection=None, geotransform=None, gridfolder=None):
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
      griddef = None # grid definition object  
      gridfolder = None # default search folder for shapefiles/masks and for GridDef, if passed by name
  '''
  # check some special conditions
  if not isinstance(var, Variable): 
    raise TypeError, 'This function can only be used to add GDAL functionality to \'Variable\' instances!'
  # only for 2D variables!
  if var.ndim >= 2:  # map-type: GDAL potential
    # infer or check projection and related parameters       
    if griddef is None:
      lgdal, projection, isProjected, xlon, ylat = getProjection(var, projection=projection)
    else:
      # use GridDefinition object 
      if isinstance(griddef,basestring): # load from pickle file
        griddef = loadPickledGridDef(grid=griddef, res=None, filename=None, folder=gridfolder)
      elif not isinstance(griddef,GridDefinition): pass 
      else: raise TypeError
      projection, isProjected, xlon, ylat = griddef.getProjection
      lgdal = xlon is not None and ylat is not None # need non-None xlon & ylat
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
    var.__dict__['griddef'] = griddef
    var.__dict__['gridfolder'] = gridfolder
    
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
    
    # extension to getMask
    def getMapMask(self, nomask=False):
      ''' A specialized version of the getMask method that gets a 2D map mask. '''
      return self.getMask(nomask=nomask, axes=(self.xlon, self.ylat), strict=True)      
    # add new method to object
    var.getMapMask = types.MethodType(getMapMask, var)   
    
    # extension to mean
    def mapMean(self, mask=None, **kwargs):
      ''' Compute mean over the horizontal axes, optionally applying a 2D shape or mask. '''
      if not self.data: raise DataError
      # determine relevant axes
      axes = {self.xlon.name:None, self.xlon.name:None} # the relevant map axes; entire coordinate
      # temporarily mask 
      if self.masked: oldmask = self.getMask() # save old mask
      else: oldmask = None
      self.mask(mask=mask, merge=True) # new mask on top of old mask
      # compute average
      kwargs.update(axes)# update dictionary
      newvar = self.mean(**kwargs)
      # lift mask
      if oldmask is not None: self.mask(mask=oldmask, merge=False)
      # return new variable
      return newvar
    # add new method to object
    var.mapMean = types.MethodType(mapMean, var)       
  
  # # the return value is actually not necessary, since the object is modified immediately
  return var

def addGDALtoDataset(dataset, griddef=None, projection=None, geotransform=None, gridfolder=None, geolocator=False):
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
      griddef = None # grid definition object  
      gridfolder = None # default search folder for shapefiles/masks and for GridDef, if passed by name
  '''
  # check some special conditions
  assert isinstance(dataset, Dataset), 'This function can only be used to add GDAL functionality to a \'Dataset\' instance!'
  # only for 2D variables!
  if len(dataset.axes) >= 2:  # else not a map-type
    # infer or check projection and related parameters       
    if griddef is None:
      lgdal, projection, isProjected, xlon, ylat = getProjection(dataset, projection=projection)
    else:
      # use GridDefinition object 
      if isinstance(griddef,basestring): # load from pickle file
        griddef = loadPickledGridDef(grid=griddef, res=None, filename=None, folder=gridfolder)
      elif isinstance(griddef,GridDefinition): pass 
      else: raise TypeError
      lgdal, projection, isProjected, xlon, ylat = getProjection(dataset, projection=griddef.projection)
#       projection, isProjected, xlon, ylat = griddef.getProjection()
#       lgdal = xlon is not None and ylat is not None # need non-None xlon & ylat        
  else: lgdal = False
  # modify instance attributes
  dataset.__dict__['gdal'] = lgdal  # all variables have this after going through this process
  
  if lgdal:  
    # infer or check geotransform
    geotransform = getGeotransform(xlon, ylat, geotransform=geotransform)
    # decide if addign a geolocator
    # add grid definition object (for convenience; recreate to match axes)
    griddef = GridDefinition(dataset.name, projection=projection, geotransform=geotransform, 
                             size=(len(xlon),len(ylat)), xlon=xlon, ylat=ylat, geolocator=geolocator)
    if geolocator:
      addGeoLocator(dataset, griddef=griddef, lgdal=False, lreplace=True, lcheck=True, asNC=False)
    # add new instance attributes (projection parameters)
    dataset.__dict__['isProjected'] = isProjected    
    dataset.__dict__['projection'] = projection
    dataset.__dict__['geotransform'] = geotransform
    dataset.__dict__['xlon'] = xlon
    dataset.__dict__['ylat'] = ylat
    dataset.__dict__['mapSize'] = (len(xlon),len(ylat))
    dataset.__dict__['griddef'] = griddef
    dataset.__dict__['gridfolder'] = gridfolder
    
    # add GDAL functionality to all variables!
    for var in dataset.variables.values():
      # call variable 'constructor' for all variables
      var = addGDALtoVar(var, projection=dataset.projection, geotransform=dataset.geotransform)
      # check result
      if not (var.ndim >= 2 and var.hasAxis(dataset.xlon) and var.hasAxis(dataset.ylat)) == var.gdal:
        raise GDALError, 'Variable {0:s} violates GDAL status (gdal={1:s})'.format(var.name, str(var.gdal))    
    
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
    
    def maskShape(self, name=None, filename=None, invert=False, **kwargs):
      ''' A method that generates a raster mask from a shape file and applies it to all GDAL variables. '''
      if name is not None and not isinstance(name,basestring): raise TypeError
      if filename is not None and not isinstance(filename,basestring): raise TypeError
      # get mask from shapefile
      shpfolder = self.gridfolder if filename is None else None
      shape = Shape(name=name, folder=shpfolder, hsapefile=filename) # load shape file
      mask = shape.rasterize(griddef=self.griddef, invert=invert, asVar=True) # extract mask
      # apply mask to dataset 
      self.mask(mask=mask, invert=False) # kwargs: merge=True, varlist=None, skiplist=None
      # return mask variable
      return mask
    # add new method to object
    dataset.maskShape = types.MethodType(maskShape, dataset)
            
  # # the return value is actually not necessary, since the object is modified immediately
  return dataset
  

## shapefile contianer class
class Shape(object):
  ''' A wrapper class for shapefiles, with some added functionality and raster itnerface '''
  
  def __init__(self, name=None, shapefile=None, folder=None, load=False, ldebug=False):
    ''' load shapefile '''
    if name is not None and not isinstance(name,basestring): raise TypeError
    if folder is not None and not isinstance(folder,basestring): raise TypeError
    if shapefile is not None and not isinstance(shapefile,basestring): raise TypeError
    # resolve file name and open file
    if ldebug: print(' - loading shapefile')
    if shapefile is None: shapefile = name + '.shp'
    if name is None: name = os.path.basename(shapefile)
    if folder is not None: shapefile = folder + '/' + shapefile
    else: folder = os.path.dirname(shapefile)
    if not os.path.exists(shapefile): raise IOError, 'File \'{}\' not found!'.format(shapefile)
    if ldebug: print(' - using shapefile \'{}\''.format(shapefile))
    # instance attributes
    self.name = name
    self.folder = folder
    self.shapefile = shapefile
    # load shapefile (or not)
    self._ogr = ogr.Open(shapefile) if load else None
  
  @property
  def OGR(self):
    ''' access to OGR dataset '''
    if self._ogr is None: self._ogr = ogr.Open(self.shapefile) # load data, if not already done 
    return self._ogr
  
  def getLayer(self, layer):
    ''' return a layer from the shapefile '''
    return self.OGR.GetLayer(layer) # get shape layer
    
  # rasterize shapefiles
  def rasterize(self, griddef=None, layer=0, invert=False, asVar=False, ldebug=False):
    ''' "burn" shapefile on a 2D raster; returns a 2D boolean array '''
    if griddef.__class__.__name__ != GridDefinition.__name__: raise TypeError 
    #if not isinstance(griddef,GridDefinition): raise TypeError # this is always False. probably due to pickling
    if not isinstance(invert,(bool,np.bool)): raise TypeError
    # fill values
    if invert: inside, outside = 1,0
    else: inside, outside = 0,1
    shp_lyr = self.getLayer(layer) # get shape layer
    # create raster to burn shape onto
    if ldebug: print(' - creating raster')
    msk_ds = ramdrv.Create(self.name, griddef.size[0], griddef.size[1], 1, gdal.GDT_Byte)
    # N.B.: this is a special case: only one band (1) and always boolean (gdal.GDT_Byte)
    # set projection parameters
    msk_ds.SetGeoTransform(griddef.geotransform)  # does the order matter?
    msk_ds.SetProjection(griddef.projection.ExportToWkt())  # is .ExportToWkt() necessary?
    # initialize raster band        
    msk_rst = msk_ds.GetRasterBand(1) # only one anyway...
    msk_rst.Fill(outside); msk_rst.SetNoDataValue(outside) # fill with zeros
    # burn shape layer onto raster band
    if ldebug: print(' - burning layer to raster')
    err = gdal.RasterizeLayer(msk_ds, [1], shp_lyr, burn_values = [inside]) # None, None, [1] # burn_value = 1
    # use argument ['ALL_TOUCHED=TRUE'] like so: None, None, [1], ['ALL_TOUCHED=TRUE']
    if err != 0: raise GDALError, 'ERROR CODE %i'%err
    #msk_ds.FlushCash()  
    # retrieve mask array from raster band
    if ldebug: print(' - retrieving mask')
    mask = msk_ds.GetRasterBand(1).ReadAsArray()
    # convert to Variable object, is desired
    if asVar: 
      mask = Variable(name=self.name, units='mask', axes=(griddef.ylat,griddef.xlon), data=mask, 
                      dtype=np.bool, mask=None, fillValue=outside, atts=None, plot=None) 
    # return mask array
    return mask  
   

# # run a test    
if __name__ == '__main__':

  ## test reading shapefile
  from datasets.common import grid_folder, shape_folder
  # load shapefile
  folder = shape_folder+'ARB_Aquanty'; shapefile='ARB_Basins_Outline_WGS84.shp' 
  shape = Shape(folder=folder, shapefile=shapefile)  
  # get mask from shape file
  griddef = loadPickledGridDef('arb2', res='d02', folder=grid_folder)
  shp_mask = shape.rasterize(griddef=griddef, invert=False, ldebug=True)
  # display
  import pylab as pyl
  pyl.imshow(np.flipud(shp_mask[:,:])); pyl.colorbar(); pyl.show(block=True)
