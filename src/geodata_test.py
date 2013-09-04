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
from geodata.base import Variable, Axis, Dataset

class BaseVarTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)  
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Axis and a Variable instance for testing '''
    # some setting that will be saved for comparison
    self.size = (2,3,4) # size of the data array and axes
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
    if not self.rav.data: self.rav.load(self.data.copy()) # again, use copy!
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
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
    var.atts.comments = 'test'; var.plot['comments'] = 'test'
    assert var.plot.comments == var.atts['comments']      
    #     print 'Name: %s, Units: %s, Missing Values: %s'%(var.name, var.units, var._FillValue)
    #     print 'Comments: %s, Plot Comments: %s'%(var.Comments,var.plotatts['plotComments'])

  def testIndexing(self):
    ''' test indexing, slicing, and broadcasting '''
    # get test objects
    var = self.var
    # indexing (getitem) test  
    if len(var) == 3:  
      assert isEqual(self.data[1,1,1], var[1,1,1], masked_equal=True)
      assert isEqual(self.data[1,:,1:-1], var[1,:,1:-1], masked_equal=True)
    # test getArray and broadcasting (reordering and reshaping/extending) 
    z = Axis(name='z', units='none', coord=(1,5,5)) # new axis
    t,x,y = self.axes 
    new_axes = (t,z,y,x)
    data = var.getArray(axes=new_axes, copy=True)
    #     print var.shape # this is what it was
    #     print data.shape # this is what it is
    #     print tuple([len(ax) if ax in var.axes else 1 for ax in new_axes]) # this is what it should be
    assert data.shape == tuple([len(ax) if ax in var.axes else 1 for ax in new_axes])
      
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
    var = self.var; rav = self.rav
    masked = var.masked
    mask = var.getMask()
    data = var.getArray(unmask=True, fillValue=-9999)
    # test unmasking and masking again
    var.unmask(fillValue=-9999)
    assert isEqual(data, var[:]) # trivial
    var.mask(mask=mask)
    assert isEqual(self.data, var.getArray(unmask=(not masked)))
    # test masking with a variable
    var.unmask(fillValue=-9999)
    assert isEqual(data, var[:]) # trivial
    var.mask(mask=rav)
    assert isEqual(ma.array(self.data,mask=(rav.data_array>0)), var.getArray(unmask=False))
    
    
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


class BaseDatasetTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)  
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Dataset with Axes and a Variables for testing '''
    # some setting that will be saved for comparison
    self.size = (3,3,3) # size of the data array and axes
    te, ye, xe = self.size
    self.atts = dict(name = 'var',units = 'n/a',FillValue=-9999)
    self.data = np.random.random(self.size)   
    # create axis instances
    t = Axis(name='t', units='none', coord=(1,te,te))
    y = Axis(name='y', units='none', coord=(1,ye,ye))
    x = Axis(name='x', units='none', coord=(1,xe,xe))
    self.axes = (t,y,x)
    # create axis and variable instances (make *copies* of data and attributes!)
    var = Variable(name='var',units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    rav = Variable(name='rav',units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    self.var = var; self.rav = rav 
    # make dataset
    self.dataset = Dataset(varlist=[var, rav])
    # check if data is loaded (future subclasses may initialize without loading data by default)
    if not self.var.data: self.var.load(self.data.copy()) # again, use copy!
    if not self.rav.data: self.rav.load(self.data.copy()) # again, use copy!
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
  ## basic tests every variable class should pass
  
  def testAddRemove(self):
    ''' test adding and removing variables '''
    # test objects: var and ax
    name='test'
    ax = Axis(name='ax', units='none')
    var = Variable(name=name,units='none',axes=(ax,))
    dataset = self.dataset
    le = len(dataset)
    # add/remove axes
    dataset.addVariable(var)
    assert dataset.hasVariable(var)
    assert dataset.hasAxis(ax)
    assert len(dataset) == le + 1
    dataset.removeAxis(ax) # should not work now
    assert dataset.hasAxis(ax)    
    dataset.removeVariable(var)
    assert dataset.hasVariable(name) == False
    assert len(dataset) == le
    dataset.removeAxis(ax)
    assert dataset.hasAxis(ax) == False
    
  def testContainer(self):
    ''' test basic container functionality '''
    # test objects: vars and axes
    dataset = self.dataset
    # check container properties 
    assert len(dataset.variables) == len(dataset)
    for varname,varobj in dataset.variables.iteritems():
      assert varname in dataset
      assert varobj in dataset
    for axname,axobj in dataset.axes.iteritems():
      assert axname in dataset
      assert axobj in dataset


# import modules to be tested
from geodata.netcdf import VarNC, AxisNC

class NetCDFVarTest(BaseVarTest):  
  
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
    var.atts.scale_factor = 2.
    var.atts.add_offset = 100.
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

class GDALVarTest(NetCDFVarTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset = 'GPCC' # dataset to use (also the folder name)
  variable = 'rain'
  RAM = True # base folder for file operations
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  # some projection settings for tests
  projection = ''
  
  def setUp(self):
    super(GDALVarTest,self).setUp() 
      
  def tearDown(self):  
    super(GDALVarTest,self).tearDown()
  
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

    # tests to be performed
    # list of variable tests
    tests = ['BaseVar'] 
#     tests = ['NetCDFVar']
#     tests = ['GDALVar']
    # list of dataset tests
#     tests = ['BaseDataset'] 
#     tests = ['NetCDFDataset']
#     tests = ['GDALDataset']    
    
    # run tests
    for test in tests:
      s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      unittest.TextTestRunner(verbosity=2).run(s)