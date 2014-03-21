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
    x = np.linspace(0,10,11)
    xaxis = Axis(name='X-Axis', units='X Units', coord=x)
    atts = dict(name='red', units='units') 
    self.var = Variable(axes=(xaxis,), data=x.copy(), atts=atts)
    self.rav = Variable(name='green',units='units',axes=(xaxis,), data=(x**2)/5.)
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
  ## basic tests every variable class should pass

  def testSimpleLinePlot(self):
    ''' test a simple line plot with two lines '''    
    fig,ax = getFigAx(1, fig=1, title=sys._getframe().f_code.co_name, mpl=mpl) # use test method name as title
    print fig.axes
    assert not isinstance(ax,(list,tuple))
    var = self.var; rav = self.rav
    # create plot
#     print var
#     print min(var),max(var)
#     for x in rav: print x
    plts = linePlot([var, rav], ax=ax, ylabel='test', ylim=(var.min(),var.max())) 
                    #linestyles, varatts, legend, xline, yline, title, xlabel, ylabel, xlim, ylim)
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