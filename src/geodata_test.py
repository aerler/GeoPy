'''
Created on 2013-08-24 

Unittest for the PyGeoDat main package geodata.

@author: Andre R. Erler, GPL v3
'''

import unittest
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

# import modules to be tested
from geodata.misc import isZero, isOne, isEqual
from geodata import Variable, Axis

class BaseTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)  
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Axis and a Variable instance for testing '''
    # some setting that will be saved for comparison
    self.size = (3,3,3) # size of the data array and axes
    te, ye, xe = self.size
    self.atts = dict(name = 'test',units = 'n/a',FillValue=-9999)
    self.data = np.random.random(self.size)   
    # create axis instances
    t = Axis(name='t', units='none', coord=(1,te,te))
    y = Axis(name='y', units='none', coord=(1,ye,ye))
    x = Axis(name='x', units='none', coord=(1,xe,xe))
    self.axes = (t,y,x)
    # create axis and variable instances (make *copies* of data and attributes!)
    self.var = Variable(name=self.atts['name'],units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    self.rav = Variable(name=self.atts['name'],units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    # check if data is loaded (future subclasses may initialize without loading data by default)
    if not self.var.data: self.var.load(self.data.copy()) # again, use copy!
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    
  ## basic tests every variable class should pass

  def testLoad(self):
    ''' test data loading and unloading '''
    # get test objects
    var = self.var
    # unload and load test
    var.unload()
    var.load(self.data.copy())
    assert self.size == var.shape
    assert isEqual(self.data, var.data_array)
    
  def testAttributes(self):
    ''' test handling of attributes '''
    # get test objects
    var = self.var; atts = self.atts
    # test getattr
    assert (atts['name'],atts['units']) == (var.name,var.units)
    # test setattr
    var.Comments = 'test'; var.plotComments = 'test' 
    #     print 'Name: %s, Units: %s, Missing Values: %s'%(var.name, var.units, var._FillValue)
    #     print 'Comments: %s, Plot Comments: %s'%(var.Comments,var.plotatts['plotComments'])

  def testIndexing(self):
    ''' test indexing and slicing '''
    # get test objects
    var = self.var
    # indexing (getitem) test  
    if len(var) == 3:  
      assert isEqual(self.data[1,1,1], var[1,1,1], masked_equal=True)
      assert isEqual(self.data[1,:,1:-1], var[1,:,1:-1], masked_equal=True)
      
  def testUnaryArithmetic(self):
    ''' test unary arithmetic functions '''
    # get test objects
    var = self.var
    # arithmetic test
    var += 2.
    var -= 2.
    var *= 2.
    var /= 2.
    # test results
    #     print (self.data.filled() - var.data_array.filled()).max()
    assert isEqual(self.data, var.data_array)
    
  def testAxis(self):
    ''' test stuff related to axes '''
    # get test objects
    var = self.var
    # test contains 
    for ax,n in zip(self.axes,self.size):
      assert ax in var
      assert len(ax) == n
#       if ax in var: print '%s is the %i. axis and has length %i'%(ax.name,var[ax]+1,len(ax))

  def testMask(self):
    ''' test masking and unmasking of data '''
    # get test objects
    var = self.var
    masked = var.masked
    mask = var.getMask()
    data = var.get(unmask=True, fillValue=-9999)
    # test unmasking and masking again
    var.unmask(fillValue=-9999)
    assert isEqual(data, var[:])
    var.mask(mask=mask)
    if masked:
      assert isEqual(self.data, var[:])
    else:
      assert isEqual(self.data, var.get(unmask=True))
    
  def testBinaryArithmetic(self):
    ''' test binary arithmetic functions '''
    # get test objects
    var = self.var
    rav = self.rav
    # arithmetic test
    a = var + rav
    assert isEqual(self.data*2, a.data_array)
    s = var - rav
    assert isZero(s.data_array)
    m = var * rav
    assert isEqual(self.data**2, m.data_array)
    if (rav.data_array == 0).any(): # can't divide by zero!
      if (rav.data_array != 0).any():  # test masking: mask zeros
        rav.mask(np.logical_not(rav.data_array), fillValue=rav.fillValue, merge=True)
      else: raise TypeError, 'Cannot divide by all-zero field!' 
    d = var / rav
    assert isOne(d.data_array)
    # test results
    #     print (self.data.filled() - var.data_array.filled()).max()
#     assert isEqual(np.ones_like(self.data), d.data_array)
#     assert isOne(d.data_array)  


# import modules to be tested
from geodata.netcdf import VarNC, AxisNC

class NetCDFTest(BaseTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset = 'GPCC' # dataset to use (also the folder name)
  variable = 'landmask' # variable to test
  RAM = True # base folder for file operations
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    if self.RAM: folder = '/media/tmp/'
    else: folder = '/home/DATA/DATA/%s/'%self.dataset # dataset name is also in folder name
    if self.dataset == 'GPCC':
      # load a netcdf dataset, so that we have something to play with
      self.ncdata = nc.Dataset(folder+'gpccavg/gpcc_25_clim_1979-1988.nc',mode='r')
    # load variable
    ncvar = self.ncdata.variables[self.variable]      
    # get dimensions and coordinate variables
    size = [len(self.ncdata.dimensions[dim]) for dim in ncvar.dimensions] 
    axes = [AxisNC(self.ncdata.variables[dim], length=le) for dim,le in zip(ncvar.dimensions,size)] 
    # initialize netcdf variable 
    self.ncvar = ncvar; self.axes = axes
    self.var = VarNC(ncvar, axes=axes, load=True)    
    self.rav = VarNC(ncvar, axes=axes, load=True) # second variable for binary operations    
    # save the original netcdf data
    self.data = self.ncdata.variables[self.variable][:].copy() #.filled(0)
    self.size = tuple([len(ax) for ax in axes])
    # construct attributes dictionary from netcdf attributes
    self.atts = { key : self.ncvar.getncattr(key) for key in self.ncvar.ncattrs() }
    self.atts['name'] = self.ncvar._name
    if 'units' not in self.atts: self.atts['units'] = '' 
      
  def tearDown(self):  
    self.var.unload()   
    self.ncdata.close()
  
  ## specific NetCDF test cases

  def testScaling(self):
    ''' test scale and offset operations '''
    # get test objects
    var = self.var
    # unload and change scale factors    
    var.unload()
    var.scale_factor = 2.
    var.add_offset = 100.
    # load data with new scaling
    var.load(scale=True)
    assert self.size == var.shape
    assert isEqual(self.data*2+100., var.data_array)
  
  def testLoadSlice(self):
    ''' test loading of slices '''
    # get test objects
    var = self.var
    var.unload()
    # load slice
    if len(var) == 3:
      sl = (slice(0,12,1),slice(20,50,5),slice(70,140,15))
      var.load(sl)
      assert (12,6,5) == var.shape
      assert isEqual(self.data.__getitem__(sl), var.data_array)
  
  def testIndexing(self):
    ''' test indexing and slicing '''
    # get test objects
    var = self.var
    # indexing (getitem) test    
    if len(var) == 3:
      assert isEqual(self.data[1,1,1], var[1,1,1])
      assert isEqual(self.data[1,:,1:-1], var[1,:,1:-1])
    # test axes


# import modules to be tested
from geodata.gdal import addGDAL 

class GDALTest(NetCDFTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset = 'GPCC' # dataset to use (also the folder name)
  variable = 'rain'
  RAM = True # base folder for file operations
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  # some projection settings for tests
  projection = ''
  
  def setUp(self):
    super(GDALTest,self).setUp() 
      
  def tearDown(self):  
    super(GDALTest,self).tearDown()
  
  ## specific NetCDF test cases

  def testAddProjection(self):
    ''' test function that adds projection features '''
    # get test objects
    var = self.var # NCVar object
    # add GDAL functionality to variable
    addGDAL(var)
    # trivial tests
    assert var.gdal
    assert var.isProjected == False
    assert var.getGDAL()
    data = var.getGDAL()
    assert data.ReadAsArray()[:,:,:].shape == (var.bands,)+var.mapSize 
    
    
if __name__ == "__main__":

    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.iteritems():
      if key[-4:] == 'Test':
        test_classes[key[:-4]] = val

    # list of tests to be performed
    tests = ['Base'] 
    tests = ['NetCDF']
    tests = ['GDAL']
    
    # run tests
    for test in tests:
      s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      unittest.TextTestRunner(verbosity=2).run(s)