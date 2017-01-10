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
from collections import OrderedDict
import types  # needed to bind functions to objects
import pickle
# gdal imports
from osgeo import gdal, osr, ogr
from utils.misc import flip
# register RAM driver
ramdrv = gdal.GetDriverByName('MEM')
# use exceptions (off by default)
gdal.UseExceptions()
osr.UseExceptions()
ogr.UseExceptions()
# set default environment variable to prevent problems in IPython Notebooks
import os
os.environ.setdefault('GDAL_DATA','/usr/local/share/gdal')

# import all base functionality from PyGeoDat
from geodata.base import Variable, Axis, Dataset
from geodata.misc import printList, isEqual, isInt, isFloat, isNumber , ArgumentError
from geodata.misc import DataError, AxisError, GDALError, DatasetError


## utility functions and classes to handle projection information and related meta data

# utility function to check if longitude runs from 0 to 360, instead of -180 - 180
def checkWrap360(lwrap360, xlon):
  if lwrap360 is None:
    lwrap360 = False # until proven otherwise
    if xlon is not None:
      if any( xlon.getArray() > 180 ):
        lwrap360 = True # need to wrap around
        assert all( xlon.getArray() >= 0 )
        assert all( xlon.getArray() <= 360 ) 
      else:
        assert all( xlon.getArray() >= -180 )
        assert all( xlon.getArray() <= 180 )
  elif not isinstance(lwrap360,bool): raise TypeError
  # return boolean
  return lwrap360


class GridDefinition(object):
  ''' 
    A class that encapsulates all necessary information to fully define a grid.
    That includes GDAL spatial references and map-Axis instances with coordinates.
  '''
  name = '' # a name for the grid...
  scale = None # approximate resolution of the grid in degrees at the domain center
  projection = None  # GDAL SpatialReference instance
  isProjected = None  # logical; indicates whether the coordiante system is projected or geographic
  wrap360 = False # logical; whether longitude runs from 0 to 360 (True) or -180 to 180 (False)  
  xlon = None  # Axis instance; the west-east axis
  ylat = None  # Axis instance; the south-north axis
  geotransform = None  # 6-element vector defining a GDAL GeoTransform
  size = None  # tuple, defining the size of the x/lon and y/lat axes
  geolocator = False # whether or not geolocator arrays are available
  lon2D = None # 2D field of longitude at each grid point
  lat2D = None # 2D field of latitude at each grid point
      
  def __init__(self, name='', projection=None, geotransform=None, size=None, xlon=None, ylat=None, 
               lwrap360=None, geolocator=True, convention=None):
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
      if not gdalsr.IsProjected(): lwrap360 = checkWrap360(lwrap360, xlon)
    else:
      gdalsr = osr.SpatialReference() 
      gdalsr.SetWellKnownGeogCS('WGS84')                 
      if projection is None: # default: normal lat/lon projection
        lwrap360 = checkWrap360(lwrap360, xlon)
        if lwrap360: # longitude runs from 0 to 360 degrees, i.e. wraps at 360/0
          projection = dict(proj='longlat',lon_0=180,lat_0=0,x_0=0,y_0=0,lon_wrap=0) # lon = [0,360]
          #projection = dict(proj='longlat',lon_0=0,lat_0=0,x_0=0,y_0=0) # lon = [0,360]
        else: projection = dict(proj='longlat',lon_0=0,lat_0=0,x_0=0,y_0=0) # wraps at dateline          
      if isinstance(projection, dict): 
        if convention is None: convention = 'Proj4'
        gdalsr = getProjFromDict(projdict=projection, name='', GeoCS='WGS84', convention=convention)  # get projection from dictionary
      elif isinstance(projection, basestring):
        if convention is None: convention = 'Wkt'
        if convention.lower() == 'wkt': gdalsr.ImportFromWkt(projection)  # from Well-Known-Text
        elif convention.lower() == 'proj4': gdalsr.ImportFromProj4(projection)  # from Proj4 convention
      elif isinstance(projection, np.number):
        gdalsr.ImportFromEpsg(projection)  # from EPSG code    
      else: 
        raise TypeError, '\'projection\' has to be a GDAL SpatialReference object.'              
    # set projection attributes
    self.projection = gdalsr
    self.isProjected = gdalsr.IsProjected()
    self.wrap360 = lwrap360
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
      (llx,lly,llz) = tx.TransformPoint(xlon.coord[xs].astype(np.float64),ylat.coord[ys].astype(np.float64)); del llz
      (urx,ury,urz) = tx.TransformPoint(xlon.coord[xe].astype(np.float64),ylat.coord[ye].astype(np.float64)); del urz
      # N.B.: for some reason GDAL is very sensitive to type and does not understand numpy types
      dlon = ( urx - llx ) / ( xe - xs ); dlat = ( ury - lly ) / ( ye - ys )       
      self.scale = ( dlon + dlat ) / 2
      # add geolocator arrays
      if geolocator:
        x2D, y2D = np.meshgrid(xlon.coord, ylat.coord) # if we have x/y arrays
        xx = x2D.flatten().astype(np.float64); yy = y2D.flatten().astype(np.float64)
        lon2D = np.zeros_like(xx); lat2D = np.zeros_like(yy) 
        #for i in xrange(xx.size):
        #  (lon2D[i],lat2D[i],tmp) = tx.TransformPoint(xx[i],yy[i])
        # N.B.: apparently TransformPoints is not much faster than a simple loop... 
        point_array = np.concatenate((x2D.reshape((x2D.size,1)),y2D.reshape((y2D.size,1))), axis=1)
        #print point_array.shape; print point_array[:3,:]
        point_array = np.asarray(tx.TransformPoints(point_array.astype(np.float64)), dtype=np.float32)
        #print tmp.shape; print tmp[:3,:] # (lon2D,lat2D,zzz)
        lon2D = point_array[:,0]; lat2D = point_array[:,1] 
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
def loadPickledGridDef(grid=None, res=None, filename=None, folder=None, check=True, lfilepath=False):
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
    filepath = '{0:s}/{1:s}'.format(folder,filename)
  # load pickle
  if os.path.exists(filepath):
    filehandle = open(filepath, 'r')
    griddef = pickle.load(filehandle)
    filehandle.close()
  elif check: 
    raise IOError, "GridDefinition pickle file '{0:s}' not found!".format(filepath) 
  else:
    griddef = None
  # add path of pickle file, if desired
  if griddef and lfilepath: 
    griddef.filepath = filepath # monkey-patch...
  # return
  return griddef
# save GridDef to pickle
def pickleGridDef(griddef=None, folder=None, filename=None, lfeedback=True):
  ''' function to pickle griddefs in a standardized way '''
  if not isinstance(griddef,GridDefinition): raise TypeError
  if filename is not None and not isinstance(filename,basestring): raise TypeError
  if folder is not None and not isinstance(folder,basestring): raise TypeError
  # construct name
  filename = griddef_pickle.format(griddef.name) if filename is None else filename
  filepath = '{0:s}/{1:s}'.format(grid_folder if folder is None else folder,filename)
  # open file and save pickle
  filehandle = open(filepath, 'w')
  pickle.dump(griddef, filehandle)
  filehandle.close()
  # print some feedback
  if not os.path.exists(filepath):
    raise IOError, "Error while saving Pickle to '{0:s}'".format(filepath)
  elif lfeedback: print("   Saved Pickle to '{0:s}'".format(filepath))
  # return filename
  return filepath


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
    proj = projdict['proj']
    if proj == 'longlat' or proj == 'latlong':
      projstr = '+proj=latlong'; lproj = False
    else: 
      projstr = '+proj={0:s}'.format(proj); lproj = True 
    # loop over entries
    for key, value in projdict.iteritems():
      if key is not 'proj':
        if not isinstance(key, str): raise TypeError
        if not isinstance(value, (float,np.inexact,int,np.integer)): raise TypeError
        # translate dict entries to string
        projstr = '{0:s} +{1:s}={2:f}'.format(projstr, key, value)
    # load projection from proj4 string
    projection.ImportFromProj4(projstr)
  else:
    raise NotImplementedError
  # more meta data
  if lproj: projection.SetProjCS(name)  # establish that this is a projected system
  # N.B.: note that the seemingly analogous method SetGeogCS() has a different function (and is not necessary)
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
#       if not var.hasAxis('x') and var.hasAxis('y'): 
#         raise AxisError, 'Horizontal axes for projected GDAL variables have to \'x\' and \'y\'.'
      if var.hasAxis('x') and var.hasAxis('y'):
        xlon = var.x; ylat = var.y
      else: xlon = None; ylat = None
      # N.B.: staggered variables are usually only staggered in one dimension, but these variables can not
      #       be treated as a GDAL variable, because their geotransform would be different
    else: 
#       if not var.hasAxis('lon') and var.hasAxis('lat'):
#         raise AxisError, 'Horizontal axes for non-projected GDAL variables have to be \'lon\' and \'lat\''
      if var.hasAxis('lon') and var.hasAxis('lat'):
        xlon = var.lon; ylat = var.lat
      else: xlon = None; ylat = None    
      # N.B.: staggered variables are usually only staggered in one dimension, but these variables can not
      #       be treated as a GDAL variable, because their geotransform would be different
  # if the variable is map-like, add GDAL properties
  if xlon is not None and ylat is not None:
    lgdal = True
    # check axes
    axstr = "'x' and 'y'" if isProjected else "'lon' and 'lat'"
    if not isinstance(xlon, Axis) and not isinstance(ylat, Axis): 
      raise AxisError, "Error: attributes {:s} have to be axes.".format(axstr)
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
  ''' Function to check or infer GDAL geotransform from coordinate axes. 
      Note that due to machine-precision errors, recomputing the geotransform from coordinates can cause problems. '''
  if geotransform is None:  # infer geotransform from axes
    if not isinstance(ylat, Axis) or not isinstance(xlon, Axis): raise TypeError     
    if xlon.data and ylat.data:
      # infer GDAL geotransform vector from  coordinate vectors (axes)
      dx = xlon[1] - xlon[0]; dy = ylat[1] - ylat[0]
      # assert (np.diff(xlon).mean() == dx).all() and (np.diff(ylat).mean() == dy).all(), 'Coordinate vectors have to be uniform!'
      ulx = xlon[0] - dx / 2.; uly = ylat[0] - dy / 2.  # coordinates of upper left corner (same for source and sink)
      # GT(2) & GT(4) are zero for North-up; GT(1) & GT(5) are pixel width and height; (GT(0),GT(3)) is the top left corner
      geotransform = (float(ulx), float(dx), 0., float(uly), 0., float(dy))
    else: raise DataError, "Coordinate vectors are required to infer GDAL geotransform vector."
  else:  # check given geotransform
    geotransform = tuple(float(f) for f in geotransform)
    if xlon.data or ylat.data:
      # check if GDAL geotransform vector is consistent with coordinate vectors
      if not len(geotransform) == 6:
        raise GDALError('\'geotransform\' has to be a vector or list with 6 elements.')
      dx = geotransform[1]; dy = geotransform[5]; ulx = geotransform[0]; uly = geotransform[3] 
      # assert isZero(np.diff(xlon)-dx) and isZero(np.diff(ylat)-dy), 'Coordinate vectors have to be compatible with geotransform!'
      #print geotransform
      #print ulx + dx / 2., xlon[0], uly + dy / 2., ylat[0]
      # coordinates of upper left corner (same for source and sink)       
      if not isEqual(ulx + dx / 2., float(xlon[0])): raise GDALError('{} != {}'.format(ulx + dx / 2., float(xlon[0])))
      if not isEqual(uly + dy / 2., float(ylat[0])): raise GDALError('{} != {}'.format(uly + dy / 2., float(ylat[0])))
    else: 
      if not ( len(geotransform) == 6 and all(isFloat(geotransform)) ):
        raise GDALError('\'geotransform\' has to be a vector or list of 6 floating-point numbers.')
  # return results
  return geotransform


## functions to add GDAL functionality to existing Variable and Dataset instances

def addGDALtoVar(var, griddef=None, projection=None, geotransform=None, gridfolder=None, loverride=False):
  ''' 
    A function that adds GDAL-based geographic projection features to an existing Variable instance.
    
    New Instance Attributes: 
      gdal = False # whether or not this instance possesses any GDAL functionality and is map-like
      isProjected = False # whether lat/lon spherical (False) or a geographic projection (True)
      projection = None # a GDAL spatial reference object
      geotransform = None # a GDAL geotransform vector (can e inferred from coordinate vectors)
      mapSize = None # size of horizontal dimensions (y/lat,x/lon)
      bands = None # all dimensions except, the map coordinates 
      xlon = None # West-East axis
      ylat = None # South-North axis  
      griddef = None # grid definition object  
      gridfolder = None # default search folder for shapefiles/masks and for GridDef, if passed by name
  '''
  # check some special conditions
  if not isinstance(var, Variable): 
    raise TypeError, 'This function can only be used to add GDAL functionality to \'Variable\' instances!'
  
  ## skip, if already GDAL-ensembled (and not in override mode)
  if not loverride and 'gdal' in var.__dict__: 
      return var# return immediately
    
  # only for 2D variables!
  if var.ndim >= 2:  # map-type: GDAL potential
    # infer or check projection and related parameters       
    if griddef is None:
      lgdal, projection, isProjected, xlon, ylat = getProjection(var, projection=projection)
      if lgdal and xlon is not None and ylat is not None:
        griddef = GridDefinition(projection=projection, xlon=xlon, ylat=ylat, geotransform=geotransform)
    else:
      # use GridDefinition object 
      if isinstance(griddef,basestring): # load from pickle file
        griddef = loadPickledGridDef(grid=griddef, res=None, filename=None, folder=gridfolder)
      elif not isinstance(griddef,GridDefinition): raise TypeError
      projection, isProjected, xlon, ylat = griddef.getProjection()
      lgdal = ( ( xlon is not None and ylat is not None ) and # need non-None xlon & ylat
                ( var.hasAxis(ylat.name) and var.hasAxis(xlon.name)) ) # and need them in the Variable
  else: lgdal = False
  # add result to Variable instance
  var.__dict__['gdal'] = lgdal  # all variables have this after going through this process

  if lgdal:    
    # determine gdal-relevant shape parameters
    mapSize = var.shape[-2:]
    if mapSize != (len(ylat),len(xlon)):
      raise GDALError, (mapSize, (len(ylat),len(xlon)))
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
      string = self.__class__.prettyPrint(self, short=short)
      if not short:
        if var.projection is not None:
          string += '\nProjection: {0:s}'.format(self.projection.ExportToWkt())
      return string
    # add new method to object
    var.prettyPrint = types.MethodType(prettyPrint, var)
    
    # overload slicing
    def slicing(self, lslices=False, **kwargs):
      ''' This method implements access to slices via coordinate values and returns Variable objects. 
          Default behavior for different argument types: 
            - index by coordinate value, not array index, except if argument is a Slice object
            - interprete tuples of length 2 or 3 as ranges
            - treat lists and arrays as coordinate lists (can specify new list axis)
            - for backwards compatibility, None values are accepted and indicate the entire range 
          Type-based defaults are ignored if appropriate keyword arguments are specified. 
          Additionally, this method has been patched to propagate GDAL features.
      '''
      # slice and get new variable
      if lslices: newvar, slcs = self.__class__.slicing(self, lslices=True, **kwargs)
      else: newvar = self.__class__.slicing(self, lslices=False, **kwargs)      
      # propagate GDAL features
      if ( len(newvar.shape) >= 2 and ( self.ylat and self.xlon ) and 
           ( newvar.hasAxis(self.ylat.name) and newvar.hasAxis(self.xlon.name) ) ):
        if self.xlon.name in kwargs or self.ylat.name in kwargs:
          geotransform = None
        else: geotransform = self.geotransform
        newvar = addGDALtoVar(newvar, projection=self.projection, geotransform=geotransform)  # add GDAL functionality      
      else:
        newvar.__dict__['gdal'] = False # mark as negative
      # return results and slices, if requested
      if lslices: return newvar, slcs
      else: return newvar
    # add new method to object
    var.slicing = types.MethodType(slicing, var)      
    
    # overload copy method to propagate GDAL features
    def copy(self, projection=None, geotransform=None, **newargs):
      ''' A method to copy the Variable with just a link to the data. '''
      var = self.__class__.copy(self, **newargs)  # use class copy() function
      # handle geotransform
      if not geotransform and not projection:
        if 'axes' in newargs:  # if axes were changed, geotransform can change!
          var = addGDALtoVar(var, projection=self.projection) # infer from new axes
        else:
          var = addGDALtoVar(var, griddef=self.griddef) # just copy old grid
      else:
        # handle projection
        if projection is None: projection = self.projection
        if geotransform is None:
          if 'axes' in newargs: geotransform = None # infer from axes
          else: geotransform = self.geotransform # use old (should be unchanged) 
        var = addGDALtoVar(var, projection=projection, geotransform=geotransform) # add GDAL functionality
      return var
    # add new method to object
    var.copy = types.MethodType(copy, var)
        
    # define GDAL-related 'class methods'  
    def getGDAL(self, load=True, allocate=True, wrap360=False, fillValue=None, lupperleft=False):
      ''' Method that returns a gdal dataset, ready for use with GDAL routines. '''
      lperi = False
      if self.gdal and self.projection is not None:
        axstr = "'x' and 'y'" if isProjected else "'lon' and 'lat'"
        if (var.axisIndex(xlon) != var.ndim-1) or (var.axisIndex(ylat) != var.ndim-2):
          raise NotImplementedError, "Horizontal axes ({:s}) have to be the last indices.".format(axstr)
        if load:
          if not self.data: self.load()
          if not self.data: raise DataError, 'Need data in Variable instance in order to load data into GDAL dataset!'
          data = self.getArray(unmask=True)  # get unmasked data
          data = data.reshape(self.bands, self.mapSize[0], self.mapSize[1])  # reshape to fit bands
          if lperi: 
            tmp = np.zeros((self.bands, self.mapSize[0], self.mapSize[1]+1))
            tmp[:,:,0:-1] = data; tmp[:,:,-1] = data[:,:,0]
            data = tmp
        elif allocate: 
          if fillValue is None:
            if self.fillValue is not None: fillValue = self.fillValue  # use default 
            elif self.dtype is not None: fillValue = ma.default_fill_value(self.dtype)
            else: raise GDALError, "Need Variable with valid dtype to pre-allocate GDAL array!"
          data = np.zeros((self.bands,) + self.mapSize, dtype=self.dtype) + fillValue
        # to insure correct wrapping, geographic coordinate systems with longitudes reanging 
        # from 0 to 360 can optionally be shifted back by 180, to conform to GDAL conventions 
        # (the shift will only affect the GDAL Dataset, not the actual Variable) 
        if wrap360:
          geotransform = list(self.geotransform)
          shift = int( 180. / geotransform[1] )
          assert len(self.xlon) == data.shape[2], "Make sure the X-Axis is the last one!"
          # N.B.: GDAL enforces the following shape: (band, lat, lon)
          if load: data = np.roll(data, shift, axis=2) # shift data along the x-axis
          geotransform[0] = geotransform[0] - shift*geotransform[1] # record shift in geotransform 
        else: geotransform = self.geotransform
        # enforce orientation
        if lupperleft and geotransform[5] > 0:
          # use upper-left corner as reference; default in GDAL applications and requires dy < 0
          geotransform = (geotransform[0],geotransform[1],geotransform[2],
                          geotransform[3] + self.mapSize[0]*geotransform[5], # shift North
                          geotransform[4], -1*geotransform[5]) # make dy < 0
          data = flip(data, axis=-2) # flip y-axis
        elif not lupperleft and geotransform[5] < 0:
          # use lower-left corner as reference; default in GeoPy and works, if dy > 0
          geotransform = (geotransform[0],geotransform[1],geotransform[2],
                          geotransform[3] + self.mapSize[0]*geotransform[5], # shift South, dy < 0 !!!
                          geotransform[4], -1*geotransform[5]) # make dy > 0
          data = flip(data, axis=-2) # flip y-axis
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
        dataset.SetGeoTransform(geotransform)  # does the order matter?
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
    
    def loadGDAL(self, dataset, mask=True, wrap360=False, fillValue=None, lyflip=True):
      ''' Load data from the bands of a GDAL dataset into the variable. '''
      # check input
      if not isinstance(dataset, gdal.Dataset): raise TypeError
      if self.gdal:
        axstr = "'x' and 'y'" if isProjected else "'lon' and 'lat'"
        if (var.axisIndex(xlon) != var.ndim-1) or (var.axisIndex(ylat) != var.ndim-2):
          raise NotImplementedError, "Horizontal axes ({:s}) have to be the last indices.".format(axstr)        
        # check that GDAL and GeoPy datsets have the same coordinate system and grid
        projection = dataset.GetProjection()
        if self.projection.ExportToWkt() != projection: 
          raise GDALError, "Projection of Variable ({:s}) differs from projection of GDAL dataset ({:s}).".format(self.projection.ExportToWkt(),projection)        
        geotransform = dataset.GetGeoTransform()
        if wrap360: # is we need to wrap/shift by 180 degrees, there is an offset
          geotransform = list(geotransform); geotransform[0] += 180.
        lyf = False # whether or not y-flip is necessary 
        if self.geotransform != tuple(geotransform):
          # check if upper/lower corner flipped
          geotransform = (geotransform[0],geotransform[1],geotransform[2],
                          geotransform[3] + self.mapSize[0]*geotransform[5], # shift North
                          geotransform[4], -1*geotransform[5]) # make dy < 0
          if self.geotransform == geotransform: lyf = lyflip # flip data array and continue
          else: 
            raise GDALError, "Geotransform of Variable ({:s}) differs from geotransform of GDAL dataset ({:s}).".format(self.geotransform,geotransform)
        # get data field
        if self.bands == 1: data = dataset.ReadAsArray()[:, :]  # for 2D fields
        else: data = dataset.ReadAsArray()[0:self.bands, :, :]  # ReadAsArray(0,0,xe,ye)
        # fix upper/lower corner issue
        if lyf: data = flip(data, axis=-2) # flip y-axis
        # to insure correct wrapping, geographic coordinate systems with longitudes ranging 
        # from 0 to 360 can optionally be shifted back by 180, to conform to GDAL conventions 
        # (the shift will only affect the GDAL Dataset, not the actual Variable) 
        if wrap360:
          shift = -1 * int( 180. / self.geotransform[1] ) # shift in opposite direction
          xax = 1 if self.bands == 1 else 2
          assert len(self.xlon) == data.shape[xax], "Make sure the X-Axis is the last one!"
          # N.B.: GDAL enforces the following shape: ([band,] lat, lon)
          data = np.roll(data, shift, axis=xax) # shift data along the x-axis 
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
        # load data
        self.load(data=data)
      # return verification
      return self.data
    # add new method to object
    var.loadGDAL = types.MethodType(loadGDAL, var)    
    
    # update new instance attributes
    def load(self, data=None, mask=None):
      ''' Load new data array. '''
      var.__class__.load(self, data=data, mask=mask)    
      if ( len(self.shape) >= 2 and ( self.ylat and self.xlon ) and 
           ( self.hasAxis(self.ylat.name) and self.hasAxis(self.xlon.name) ) ):  
        # 2D (or more) with y/lat and x/lon axes keep GDAL status
        self.__dict__['mapSize'] = self.shape[-2:]  # need to update
        if self.mapSize != (len(self.ylat),len(self.xlon)):
          raise GDALError, (self.mapSize, (len(self.xlon),len(self.ylat)))
      else:  
        # if less than 2D or y/lat or x/lon axes missing, strip GDAL status
        self.__dict__['gdal'] = False
        self.__dict__['mapSize'] = None
        self.__dict__['projection'] = None
        self.__dict__['geotransform'] = None
        self.__dict__['bands'] = None
        self.__dict__['xlon'] = None
        self.__dict__['ylat'] = None
        # keep griddef - might be useful
      # for convenience
      return self
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
    def mapMean(self, mask=None, integral=False, invert=False, squeeze=True, **kwargs):
      ''' Compute mean over the horizontal axes, optionally applying a 2D shape or mask. '''
      if not self.data: raise DataError
      # if mask is a shape object, create the mask
      if isinstance(mask,Shape):
        shape = mask 
        mask = shape.rasterize(griddef=self.griddef, invert=invert, asVar=False)
      else: shape = None      
      # determine relevant axes
      axes = {self.xlon.name:None, self.ylat.name:None} # the relevant map axes; entire coordinate
      # temporarily mask 
      if self.masked: oldmask = ma.getmask(self.data_array) # save old mask
      else: oldmask = None
      self.mask(mask=mask, invert=invert, merge=True) # new mask on top of old mask
      # compute average
      kwargs.update(axes)# update dictionary
      newvar = self.mean(**kwargs)
      if squeeze: newvar.squeeze()
      # if integrating
      if integral:
        if not self.isProjected: raise NotImplementedError
        dx = self.geotransform[1]; dy = self.geotransform[5] 
        area = (1-mask).sum()*dx*dy
        newvar *= area # in-place scaling
        if self.xlon.units == self.ylat.units: newvar.units = '{} {}^2'.format(newvar.units,self.ylat.units) 
        else: newvar.units ='{} {} {}'.format(newvar.units,self.xlon.units,self.ylat.units)
      # lift mask
      if oldmask is not None: 
        self.data_array.mask = oldmask # change back to old mask
      else: 
        self.data_array.mask = ma.nomask # remove mask
        if not self.masked: self.data_array = np.asarray(self.data_array) # and change class to ndarray
      # return new variable
      return newvar
    # add new method to object
    var.mapMean = types.MethodType(mapMean, var) 
    
    # save variable as Arc/Info ASCII Grid / ASCII raster file using GDAL
    def ASCII_raster(self, prefix=None, folder=None, ext='.asc', filepath=None, wrap360=False, 
                     fillValue=None, lcoord=False, lfortran=True, formatter=None):
      ''' Export data to  Arc/Info ASCII Grid (ASCII raster format); if no filename is given, the filename will 
          be constructed from the variable name and the slice; note that each file can only contain a single 
          horizontal slice. 
          N.B.: The implementation is recursive, i.e. variables with more than two dimensions are sliced and 
          each a number of speperate calls to this function equal to the length of the dimension is issued; this
          is repeated for every dimension (over two), until the input is two-dimensional.
      '''
      # figure out filepath
      if filepath:
        # N.B.: This is basically a special option to export 2D fields to a custom path; if the dataset
        #       is not 2D, the path is disassembled into folder, prefix and extension to allow output
        #       to multiple files with coordinate indices.
        if self.ndim != 2: # only 2D fields
          # seperate folder and filename
          fp = filepath.split('/') # currently only works for Linux paths
          folder = '/'.join(fp[:-1]); filename = fp[-1]
          if '.' in filename: 
            fn = filename.split('.')
            if len(fn) != 2: raise NotImplementedError
            prefix = fn[0]; ext = '.{:s}'.format(fn[1]) # add dot back in
          else: 
            prefix = filename; ext = None
      else:
        if folder is None: 
          raise IOError, "Need to specify a folder or absolute path to export to ASCII raster file."
        prefix = prefix or self.name
      # handle different cases with recursion
      if self.ndim == 2: 
        # N.B.: GDAL can only write 2D datasets to ASCII raster; multi-dimensional datasets are 
        #       sliced recursively until they are 2D; at this point the recursion ends and the 
        #       sliced dataset/Variable can be exported to ASCII raster format (one per file).
        # get GDAL datast
        dataset = getGDAL(self, load=True, allocate=True, wrap360=wrap360, fillValue=fillValue, lupperleft=True)
        # N.B.: apparently the raster driver always assumes that the geotransform reference point is the upper left corner
        # get ASCII raster file driver
        ascii = gdal.GetDriverByName('AAIGrid')
        # construct filepath for 2D fields
        if not filepath: 
          filepath = '{:s}/{:s}'.format(folder,prefix)
          if ext: filepath = '{:s}{:s}'.format(filepath,ext)
        # easy: just write ASCII raster file
        ascii.CreateCopy(filepath, dataset)
        filelist = filepath # this will be returned
        # for good form, indirectly close the dataset
        dataset = None; ascii = None
      elif self.ndim > 2: 
        # for ND fields, a new file for each band is necessary, hence filepath changes for every band
        fax = self.axes[0] # take first axis to iterate over
        lenax = len(fax); axname = fax.name
        if not lcoord: one = 1 if lfortran else 0 # Fortran or C indexing
        # figure out formatter
        if formatter and axname in formatter:
          fmt = formatter[axname]
          if isinstance(fmt, (list,tuple)):
            axtag = fmt[0]; fmt = fmt[1]
          else: axtag = None # assign below           
        else:
          axtag = None # assign below
          if lcoord: fmt = '{}' # just a default... usually user-specified
          else: fmt = '{{:0{:d}d}}'.format(int(np.ceil(np.log10(lenax+one)))) # number of digits
          # N.B.: for Fortran convetion, start counting at 1, hence +1
        if axtag is None: axtag = axname if lcoord else 'i{:s}'.format(axname.title())
        prefix = '{:s}_{:s}_{:s}'.format(prefix,axtag,fmt)
        # loop over bands
        filelist = []
        for i in xrange(lenax):
          # work on each slice individually
          slcvar = self.slicing(lidx=True, **{axname:i})
          # assemble simplified file path
          if lcoord: pf = prefix.format(fax[i]) # use actual coordinate value
          else: pf = prefix.format(i+one) # start index at 1 --- Fortran convention
          # now call this function recursively for every slice, until input is 2D
          filepath = ASCII_raster(slcvar, prefix=pf, folder=folder, ext=ext, filepath=None, 
                                  wrap360=wrap360, fillValue=fillValue)
          if isinstance(filepath, basestring): filelist.append(filepath)
          else: filelist.extend(filepath)
          # N.B.: the function basically returns the last filepath
      else: raise NotImplementedError, self
      # return full path to file
      return filelist
    # add new method to object
    var.ASCII_raster = types.MethodType(ASCII_raster, var) 
  
  # # the return value is actually not necessary, since the object is modified immediately
  return var

def addGDALtoDataset(dataset, griddef=None, projection=None, geotransform=None, gridfolder=None, 
                     lwrap360=None, geolocator=False, lforce=False, loverride=False):
  ''' 
    A function that adds GDAL-based geographic projection features to an existing Dataset instance
    and all its Variables.
    
    New Instance Attributes: 
      gdal = False # whether or not this instance possesses any GDAL functionality and is map-like
      projection = None # a GDAL spatial reference object
      isProjected = False # whether lat/lon spherical (False) or a geographic projection (True)
      wrap360 = None # whether or not longitudes run from 0 to 360, instead of -180 to 180
      xlon = None # West-East axis
      ylat = None # South-North axis
      geotransform = None # a GDAL geotransform vector (can e inferred from coordinate vectors)
      mapSize = None # length of the lat/y and lon/x axes (in that order)
      griddef = None # grid definition object  
      gridfolder = None # default search folder for shapefiles/masks and for GridDef, if passed by name
  '''
  # check some special conditions
  assert isinstance(dataset, Dataset), 'This function can only be used to add GDAL functionality to a \'Dataset\' instance!'
  
  ## skip, if already GDAL-ensembled (and not in override mode)
  if not loverride and 'gdal' in dataset.__dict__: 
      return dataset # return immediately
    
  # only for 2D variables!
  if griddef:
    # infer or check projection and related parameters       
    # use GridDefinition object 
    if isinstance(griddef,basestring): # load from pickle file
      griddef = loadPickledGridDef(grid=griddef, res=None, filename=None, folder=gridfolder)
    elif isinstance(griddef,GridDefinition): pass 
    else: raise TypeError
    if lforce and griddef.xlon.name not in dataset.axes: dataset.addAxis(griddef.xlon)
    if lforce and griddef.ylat.name not in dataset.axes: dataset.addAxis(griddef.ylat)
    lgdal, projection, isProjected, xlon, ylat = getProjection(dataset, projection=griddef.projection)
    # safety checks
    xlon_name,ylat_name = ('x','y') if isProjected else ('lon','lat')
    if xlon_name in dataset.axes:
      if dataset.axes[xlon_name].units != griddef.xlon.units:
        raise AxisError("Units of Dataset and GridDef x-axes do not match: {} != {}".format(dataset.axes[xlon_name].units,griddef.xlon.units))
      if not isEqual(dataset.axes[xlon_name][:], griddef.xlon[:]):
        bias = np.mean(dataset.axes[xlon_name][:] - griddef.xlon[:])
        cc = np.corrcoef(dataset.axes[xlon_name][:], griddef.xlon[:])[0,-1]
        raise AxisError("Coordinates of Dataset and GridDef x-axes are inconsistent! Bias: {} Correltation: {}".format(bias,cc))
      if any([dataset.axes[xlon_name] != var.getAxis(xlon_name) for var in dataset.variables.values() if var.hasAxis(xlon_name)]):
        raise AxisError("X-Axes of Variables in Dataset are inconsistent!")
    if ylat_name in dataset.axes:
      if dataset.axes[ylat_name].units != griddef.ylat.units:
        raise AxisError("Units of Dataset and GridDef y-axes do not match: {} != {}".format(dataset.axes[ylat_name].units,griddef.ylat.units))
      if not isEqual(dataset.axes[ylat_name][:], griddef.ylat[:]):
        bias = np.mean(dataset.axes[ylat_name][:] - griddef.ylat[:])
        cc = np.corrcoef(dataset.axes[ylat_name][:], griddef.ylat[:])[0,-1]
        raise AxisError("Coordinates of Dataset and GridDef y-axes are inconsistent! Bias: {} Correltation: {}".format(bias,cc))
      if any([dataset.axes[ylat_name] != var.getAxis(ylat_name) for var in dataset.variables.values() if var.hasAxis(ylat_name)]):
        raise AxisError("Y-Axes of Variables in Dataset are inconsistent!")
#       assert dataset.axes[ylat_name].units == griddef.ylat.units and np.all(dataset.axes[ylat_name][:] == griddef.ylat[:])
#       assert all([dataset.axes[ylat_name] == var.getAxis(ylat_name) for var in dataset.variables.values() if var.hasAxis(ylat_name)])
#       projection, isProjected, xlon, ylat = griddef.getProjection()
#       lgdal = xlon is not None and ylat is not None # need non-None xlon & ylat        
  elif griddef is None and len(dataset.axes) >= 2:
    lgdal, projection, isProjected, xlon, ylat = getProjection(dataset, projection=projection)
  else: lgdal = False
  # modify instance attributes
  dataset.__dict__['gdal'] = lgdal  # all variables have this after going through this process
  
  if lgdal:  
    # infer or check geotransform
    geotransform = getGeotransform(xlon, ylat, geotransform=geotransform)
    # decide if adding a geolocator
    # add grid definition object (for convenience; recreate to match axes)
    griddef = GridDefinition(dataset.name, projection=projection, geotransform=geotransform, lwrap360=lwrap360, 
                             size=(len(xlon),len(ylat)), xlon=xlon, ylat=ylat, geolocator=geolocator)
    lwrap360 = griddef.wrap360 # whether or not longitudes run from 0 to 360, instead of -180 to 180
    if geolocator:
      addGeoLocator(dataset, griddef=griddef, lgdal=False, lreplace=False, lcheck=False, asNC=False)
    # add new instance attributes (projection parameters)
    dataset.__dict__['projection'] = projection
    dataset.__dict__['isProjected'] = isProjected
    dataset.__dict__['wrap360'] = lwrap360    
    dataset.__dict__['xlon'] = xlon
    dataset.__dict__['ylat'] = ylat
    dataset.__dict__['geotransform'] = geotransform
    dataset.__dict__['mapSize'] = (len(ylat),len(xlon))
    dataset.__dict__['griddef'] = griddef
    dataset.__dict__['gridfolder'] = gridfolder
    
    # add GDAL functionality to all variables!
    for var in dataset.variables.values():
      # call variable 'constructor' for all variables
      var = addGDALtoVar(var, griddef=griddef)
      # check result
      if var.ndim >= 2 and var.hasAxis(dataset.xlon) and var.hasAxis(dataset.ylat):
        if not var.gdal:  
          raise GDALError, "Variable '{:s}' violates GDAL status (gdal={:s})".format(var.name, str(var.gdal))    
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
    
    # overload slicing
    def slicing(self, **kwargs):
      ''' This method implements access to slices via coordinate values and returns a Dataset object; the 
          method relies on the Variable method for actual slicing but preserves the dataset integrity.
          Default behavior for different argument types: 
            - index by coordinate value, not array index, except if argument is a Slice object
            - interprete tuples of length 2 or 3 as ranges
            - treat lists and arrays as coordinate lists (can specify new list axis)
            - for backwards compatibility, None values are accepted and indicate the entire range 
          Type-based defaults are ignored if appropriate keyword arguments are specified.
          Additionally, this method has been patched to propagate GDAL features. '''
      # slice and get new variable
      newds = self.__class__.slicing(self, **kwargs)      
      # propagate GDAL features
      if newds.hasAxis(self.xlon.name) and newds.hasAxis(self.ylat.name):
        if self.xlon.name in kwargs or self.ylat.name in kwargs:
          geotransform = None
        else: geotransform = self.geotransform
        newds = addGDALtoDataset(newds, projection=self.projection, geotransform=geotransform)  # add GDAL functionality      
      else:
        newds.__dict__['gdal'] = False # mark as negative
      # return results and slices, if requested
      return newds
    # add new method to object
    dataset.slicing = types.MethodType(slicing, dataset)      

    def copy(self, griddef=None, projection=None, geotransform=None, **newargs):
      ''' A method to copy the Dataset with just a link to the data. Also supports new projections. '''
      # griddef supersedes all other arguments
      if griddef is not None:
        projection = griddef.projection
        #geotransform = griddef.geotransform
        if not 'axes' in newargs: newargs['axes'] = dict()
        newargs['axes'][self.xlon.name] = griddef.xlon
        newargs['axes'][self.ylat.name] = griddef.ylat
      # invoke class copy() function to copy dataset
      dataset = self.__class__.copy(self, **newargs)
      # handle geotransform
      #if geotransform is None:
      #  if 'axes' in newargs:  # if axes were changed, geotransform can change!
      #    geotransform = None  # infer from new axes
      #  else: geotransform = self.geotransform
      ## N.B.: geotransform should be inferred from axes - more robust when slicing!
      # handle projection
      if projection is None: projection = self.projection
      dataset = addGDALtoDataset(dataset, projection=projection, geotransform=None)  # add GDAL functionality
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
    
    def mapMean(self, mask=None, integral=False, invert=False, squeeze=True, checkAxis=True, coordIndex=True):
      ''' Average entire dataset over horizontal map coordinates; optionally apply 2D mask. '''
      newset = Dataset(name=self.name, varlist=[], atts=self.atts.copy()) 
      # N.B.: the returned dataset will not be GDAL enabled, because the map dimensions will be gone! 
      # if mask is a shape object, create the mask
      if isinstance(mask,Shape):
        shape = mask 
        mask = shape.rasterize(griddef=self.griddef, invert=invert, asVar=False)
      else: shape = None
      # relevant axes
      axes = {self.xlon.name:None, self.ylat.name:None} # the relevant map axes; entire coordinate
      # loop over variables
      for var in self.variables.values():
        # figure out, which axes apply
        tmpax = {key:value for key,value in axes.iteritems() if var.hasAxis(key)}
        # get averaged variable
        if len(tmpax) == 2:
          newset.addVariable(var.mapMean(mask=mask, integral=integral, invert=invert, coordIndex=coordIndex, 
                                       squeeze=True, checkAxis=True), copy=False) # new variable/values anyway
        elif len(tmpax) == 1:
          newset.addVariable(var.mean(coordIndex=coordIndex, squeeze=True, checkAxis=True, asVar=True, 
                                      **tmpax), copy=False) # new variable/values anyway        
        elif len(tmpax) == 0: 
          newset.addVariable(var, copy=True, deepcopy=True) # copy values
#         else: raise GDALError
      # add some record
      for key,value in axes.iteritems():
        if isinstance(value,(list,tuple)): newset.atts[key] = printList(value)
        elif isinstance(value,np.number): newset.atts[key] = str(value)      
        else: newset.atts[key] = 'n/a' 
      # add reference to shape object
      if shape is not None:
        newset.area = shape         
        if invert: newset.atts['integral'] = 'area outside of {:s}'.format(shape.name)
      else: 
        if invert: newset.atts['integral'] = 'area outside of mask'
      # return new dataset
      return newset
    # add new method to object
    dataset.mapMean = types.MethodType(mapMean, dataset)
    
    # save variable as Arc/Info ASCII Grid / ASCII raster file using GDAL
    def ASCII_raster(self, varlist=None, prefix=None, folder=None, ext='.asc', wrap360=False, 
                     fillValue=None, lcoord=False, lfortran=True, formatter=None):
      ''' Export data to  Arc/Info ASCII Grid (ASCII raster format); the filename will be constructed 
          from a prefix, the variable name and the slice; note that each file can only contain a single 
          horizontal slice (2D).  
      '''
      # check arguments
      if varlist is None: varlist = self.variables.keys()
      if not isinstance(varlist, (dict,tuple,list)): raise TypeError, varlist
      if isinstance(varlist, (tuple,list)): varlist = {var:None for var in varlist}
      # N.B.: the keys of a varlist are the variables that are to be exported and the values are the
      #       corresponding variable prefixes (instead of the variable names, which is the default)
      #if prefix is None: prefix=dataset.name
      if formatter is not None and not isinstance(formatter, (dict)): raise TypeError, formatter
      # N.B.: formatter keys are axes and values are either index formatting strings or tuples
      #       consisting of a new axis name and an index formatter
      if not folder: 
        raise ArgumentError, "A valid folder is necessary to export a dataset to ASCII raster format."
      if not os.path.exists(folder): os.makedirs(folder) # make sure folder exists
      # loop over variables
      filedict = dict()
      for varname,vartag in varlist.iteritems():
        var = self.variables[varname] # variable isntance
        if vartag is None: vartag = var.name
        # skip variables that are not gdal enabled
        if var.gdal: 
          # add prefix to variable name
          pf = '{:s}_{:s}'.format(prefix,vartag) if prefix else vartag
          # call export function on each variable
          filelist = var.ASCII_raster(prefix=pf, folder=folder, ext=ext, filepath=None, wrap360=wrap360, 
                                      fillValue=fillValue, lcoord=lcoord, lfortran=lfortran, 
                                      formatter=formatter)
          if isinstance(filelist,basestring): filelist = [filelist]
          filedict[vartag] = filelist
      return filedict
    # add new method to object
    dataset.ASCII_raster = types.MethodType(ASCII_raster, dataset)    
      
  ## the return value is actually not necessary, since the object is modified immediately
  return dataset
  

## shapefile contianer class
class Shape(object):
  ''' A wrapper class for shapefiles, with some added functionality and raster interface '''
  
  def __init__(self, name=None, long_name=None, shapefile=None, folder=None, data_source=None, 
               load=False, ldebug=False, shapetype=None):
    ''' load shapefile '''
    if name is not None and not isinstance(name,basestring): raise TypeError
    if folder is not None and not isinstance(folder,basestring): raise TypeError
    if shapefile is not None and not isinstance(shapefile,basestring): raise TypeError
    # resolve file name and open file
    if ldebug: print(' - loading shapefile')
    if shapefile is None: shapefile = name + '.shp' # will be absolute path
    elif shapefile[-4:] != '.shp': shapefile = shapefile + '.shp' # and need extension
    if folder is not None: shapefile = folder + '/' + shapefile
    if name is None: name = os.path.splitext(os.path.basename(shapefile))[0]
    if long_name is None: long_name = name
    else: folder = os.path.dirname(shapefile)
    if not os.path.exists(shapefile): raise IOError, 'File \'{}\' not found!'.format(shapefile)
    if ldebug: print(' - using shapefile \'{}\''.format(shapefile))
    # instance attributes
    self.name = name
    self.long_name = long_name
    self.folder = folder
    self.shapefile = shapefile
    self.data_source = data_source # source documentation...
    shapetype = self.__class__.__name__ if shapetype is None else shapetype
    self.shapetype = shapetype # for specific types of shapes, e.g. Basin, Lake, Prov, Natl
    # load shapefile (or not)
    self._ogr = ogr.Open(shapefile) if load else None
  
  @property
  def OGR(self):
    ''' access to OGR dataset '''
    if self._ogr is None: self._ogr = ogr.Open(self.shapefile) # load data, if not already done 
    return self._ogr
  
  def load(self):
    ''' load data and return self '''
    if self._ogr is None: self._ogr = ogr.Open(self.shapefile) # load data, if not already done 
    return self
  
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

# a container class that also acts as a shape for the outline
class ShapeSet(Shape): 
  ''' a container class for a set of shapes within a common outline and with common meta data; also acts as a Shape '''
  _ShapeClass = Shape # the class that is used to initialize the shape collection
  
  def __init__(self, name=None, long_name=None, shapefiles=None, folder=None, load=False, ldebug=False,
               data_source=None, outline=None, shapetype=None, **kwargs):
    ''' initialize shapes '''
    # sort shapes into ordered dictionary
    if isinstance(shapefiles,dict):
      names = shapefiles.keys(); shapefiles = shapefiles.values()
    else: names = [None]*len(shapefiles)
    shape_list = []
    # add to list, if already an instance, otherwise create new shape instance
    for shapename,shapefile in zip(names,shapefiles):
      if isinstance(shapefile, self._ShapeClass): shape = shapefile # trivial
      else: shape = self._ShapeClass(name=shapename, shapefile=shapefile, folder=folder, ldebug=ldebug, load=load, 
                                     data_source=data_source, shapetype=shapetype, **kwargs)
      shape_list.append(shape)
    shapes = OrderedDict(); shapefiles = OrderedDict()
    for shape in shape_list:
      shapes[shape.name] = shape; shapefiles[shape.name] = shape.shapefile
    # determine outline
    if outline is None: outline = shapes.keys()[0]    
    # N.B.: the shapefile of the outline will be added by the Shape constructor
    shapefile = os.path.basename(shapefiles[outline]) # use relaive path and folder
    folder = os.path.dirname(shapefiles[outline])
    # call Shape constructor
    super(ShapeSet,self).__init__(name=name, long_name=long_name, shapefile=shapefile, folder=folder, 
                                  data_source=data_source, shapetype=shapetype, load=load, ldebug=ldebug, **kwargs)
    # add remaining attributes
    self.outline = outline # name of the main shape which traces the outline
    self.shapefiles = shapefiles # absolute path to actual shapefiles
    self.shapes = shapes # OrderedDict of Shape instances
    
  def getShape(self, name):
    ''' wrapper method to return requested shape or None '''
    return self.shapes.get(name,None)

# # container class for known shapes with meta data
# class NamedShape(Shape):
#   ''' Just a container for shapes with additional meta information '''
#   def __init__(self, area=None, subarea=None, folder=None, shapefile=None, shapetype=None, shapes_dict=None, load=False, ldebug=False):
#     ''' save meta information; should be initialized from a BasinInfo instance '''
#     # resolve input
#     if isinstance(area,(basestring,ShapeInfo)):
#       if isinstance(area,basestring):
#         if area in shapes_dict: area = shapes_dict[area]
#         else: raise ValueError, 'Unknown area: {}'.format(area)
#       folder = area.folder
#       if subarea is None: 
#         subarea = area.outline
#         name = area.name      
#         long_name = area.long_name
#       elif isinstance(subarea,basestring):
#         name = subarea 
#         long_name = separateCamelCase(subarea, **{area.name:area.long_name})
#       else: raise TypeError
#       if subarea not in area.shapefiles: raise ValueError, 'Unknown subarea: {}'.format(subarea)
#       shapefile = area.shapefiles[subarea]
#       shapetype = area.shapetype            
#     elif isinstance(shapefile,basestring):
#       if folder is not None and isinstance(folder,basestring): shapefile = folder+'/'+shapefile
#       name = area 
#       long_name = None       
#     else: raise TypeError, 'Specify either area & station or folder & shapefile.'
#     # call Shape constructor
#     super(NamedShape,self).__init__(name=name, long_name=long_name, shapefile=shapefile, load=load, ldebug=ldebug)
#     # add info
#     self.info = area
#     self.shapetype = shapetype 


## run a test    
if __name__ == '__main__':

  from datasets.common import grid_folder, data_root

  mode = 'read_shape'
  
  ## test reading shapefile
  if mode == 'read_shape':
    
    # load shapefile
#     folder = data_root+'/shapes/Provinces/Manitoba/'; shapefile='Manitoba.shp'
    folder = data_root+'/shapes/Basins/Athabasca River Basin/'; shapefile='WholeARB.shp'
#     folder = data_root+'/shapes/Basins/South Sasketchewan River/'; shapefile='WholeSSR.shp'
    shape = Shape(folder=folder, shapefile=shapefile)  
    # get mask from shape file
    griddef = loadPickledGridDef('arb2_d02', res=None, folder=grid_folder)
    shp_mask = shape.rasterize(griddef=griddef, invert=False, ldebug=True)
    #   assert np.all( shp_mask[[0,-1],:] == True ) and np.all( shp_mask[:,[0,-1]] == True )
    # display
    import pylab as pyl
    pyl.imshow(np.flipud(shp_mask[:,:])); pyl.colorbar(); pyl.show(block=True)
