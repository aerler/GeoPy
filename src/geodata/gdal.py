'''
Created on 2013-08-23

A module that provides GDAL functionality to GeoData datasets and variables, 
and exposes some more GDAL functionality, such as regriding.

The GDAL functionality for Variables is implemented as a decorator that adds GDAL attributes 
and methods to an existing object/class instance.    

@author: Andre R. Erler, GPL v3
'''

# gdal imports
from osgeo import gdal, osr
# register RAM driver
ramdrv = gdal.GetDriverByName('MEM')

# import all base functionality from PyGeoDat
from geodata.base import *

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
    else: dset = None
    # return dataset
    return dset
    
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
