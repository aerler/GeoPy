'''
Created on 2013-08-24 

Unittest for the PyGeoDat main package geodata.

@author: Andre R. Erler, GPL v3
'''

import unittest
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import os, sys

# import geodata modules
from geodata.nctools import writeNetCDF
from geodata.misc import isZero, isOne, isEqual
from geodata.base import Variable, Axis, Dataset
from datasets.common import data_root
# import modules to be tested
from plotting.lineplots import linePlot 
from plotting.mapplots import srfcPlot
from plotting.utils import getFigAx, loadMPL


# use common MPL instance
mpl,pyl = loadMPL(linewidth=1.)

class LinePlotTest(unittest.TestCase):  
   
  def setUp(self):
    ''' create two test variables '''
    # create axis and variable instances (make *copies* of data and attributes!)
    x1 = np.linspace(0,10,11); xax1 = Axis(name='X1-Axis', units='X Units', coord=x1) 
    var1 = Variable(axes=(xax1,), data=x1.copy(), atts=dict(name='red', units='units'))
    self.var1 = var1; self.xax1 = xax1
    x2 = np.linspace(2,8,13); xax2 = Axis(name='X2-Axis', units='X Units', coord=x2)
    var2 = Variable(name='green',units='units',axes=(xax2,), data=(x2**2)/5.)
    self.var2 = var2; self.xax2 = xax2
    # add to list
    self.vars = [var1, var2]
    self.axes = [xax1, xax2]
        
  def tearDown(self):
    ''' clean up '''
    for var in self.vars:     
      var.unload() # just to do something... free memory
    for ax in self.axes:
      ax.unload()
    
  ## basic tests every variable class should pass

  def testBasicLinePlot(self):
    ''' test a simple line plot with two lines '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], mpl=mpl) # use test method name as title
    assert fig.__class__.__name__ == 'Figure'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var1 = self.var1; var2 = self.var2
    # create plot
    plts = linePlot([var1, var2], ax=ax, ylabel='custom label [{}]', ylim=var1.limits(), legend=2)
    assert len(plts) == 2
  
  def testAdvancedLinePlot(self):
    ''' test more advanced options of the line plot function '''    
    fig,ax = getFigAx(1, title='Fancy Plot Styles', name=sys._getframe().f_code.co_name[4:], mpl=mpl) # use test method name as title
    assert fig.__class__.__name__ == 'Figure'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var1 = self.var1; var2 = self.var2
    # define fancy attributes
    varatts = dict()
    for var in var1,var2: varatts[var.name] = dict(color=var.name, marker='*')       
    # define fancy legend
    legend = dict(loc=2, labelspacing=0.125, handlelength=2.5, handletextpad=0.5, fancybox=True)
    # create plot
    plts = linePlot([var1, var2], ax=ax, linestyles=('--','-.'), varatts=varatts, legend=legend)
    assert len(plts) == 2
        
  def testCombinedLinePlot(self):
    ''' test a two panel line plot with combined legend '''    
    fig,axes = getFigAx(2, name=sys._getframe().f_code.co_name[4:], mpl=mpl) # use test method name as title
    assert fig.__class__.__name__ == 'Figure'
    assert isinstance(axes,(list,tuple)) # should return a list of axes
    var1 = self.var1; var2 = self.var2
    # create plot
    for i,ax in enumerate(axes):
      plts = linePlot([var1, var2], ax=ax, ylim=var1.limits(), legend=0, title='Panel {:d}'.format(i+1))
      assert len(plts) == 2
    
if __name__ == "__main__":

    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.iteritems():
      if key[-4:] == 'Test':
        test_classes[key[:-4]] = val

    # list of tests to be performed
    tests = [] 
    # list of variable tests
    tests += ['LinePlot'] 
    
    # RAM disk settings ("global" variable)
    RAM = False # whether or not to use a RAM disk
    ramdisk = '/media/tmp/' # folder where RAM disk is mounted
    
    # run tests
    for test in tests:
      s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      unittest.TextTestRunner(verbosity=2).run(s)
    # show plots
    pyl.show()