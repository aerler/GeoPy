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

import matplotlib as mpl
import matplotlib.pylab as pyl
# mpl.use('Agg') # enforce QT4

# import geodata modules
from utils.nctools import writeNetCDF
from geodata.misc import isZero, isOne, isEqual
from geodata.base import Variable, Axis, Dataset
from datasets.common import data_root
# import modules to be tested
from plotting.figure import getFigAx
# use common MPL instance
# from plotting.misc import loadMPL
# mpl,pyl = loadMPL(linewidth=1.)
from geodata.stats import mwtest, kstest, wrstest, VarRV


# RAM disk settings ("global" variable)
RAM = True # whether or not to use a RAM disk
ramdisk = '/media/tmp/' # folder where RAM disk is mounted
# stylesheet = None
figargs = dict(stylesheet='myggplot', lpresentation=True, lpublication=False)


class LinePlotTest(unittest.TestCase):  
   
  def setUp(self):
    ''' create two test variables '''
    # create axis and variable instances (make *copies* of data and attributes!)
    x1 = np.linspace(0,10,11); xax1 = Axis(name='X1-Axis', units='X Units', coord=x1) 
    var0 = Variable(axes=(xax1,), data=np.sin(x1), atts=dict(name='relative', units=''))
    var1 = Variable(axes=(xax1,), data=x1.copy(), atts=dict(name='blue', units='units'))
    self.var0 = var0; self.var1 = var1; self.xax1 = xax1
    x2 = np.linspace(2,8,13); xax2 = Axis(name='X2-Axis', units='X Units', coord=x2)
    var2 = Variable(name='purple',units='units',axes=(xax2,), data=(x2**2)/5.)
    self.var2 = var2; self.xax2 = xax2
    # create error variables with random noise
    noise1 = np.random.rand(len(xax1))*var1.data_array.std()/2.
    err1 = Variable(axes=(xax1,), data=noise1, atts=dict(name='blue_std', units='units'))
    noise2 = np.random.rand(len(xax2))*var2.data_array.std()/2.
    err2 = Variable(name='purple',units='units',axes=(xax2,), data=noise2)
    self.err1 = err1; self.err2 = err2
    # add to list
    self.vars = [var1, var2]
    self.errs = [err1, err2]
    self.axes = [xax1, xax2]
        
  def tearDown(self):
    ''' clean up '''
    for var in self.vars:     
      var.unload() # just to do something... free memory
    for ax in self.axes:
      ax.unload()
    
  ## basic plotting tests

  def testBasicLinePlot(self):
    ''' test a simple line plot with two lines '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0; var1 = self.var1; var2 = self.var2
    # create plot
    plts = ax.linePlot([var1, var2], ylabel='custom label [{1:s}]', llabel=True, 
                       ylim=var1.limits(), legend=2, hline=2., vline=(2,3))
    assert len(plts) == 2
    # add rescaled plot
    plts = ax.linePlot(var0, lrescale=True, scalefactor=2, offset=-1, llabel=True, legend=2, linestyle=':')
    assert len(plts) == 1    
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)
    
  def testBasicErrorPlot(self):
    ''' test a simple errorbar plot with two lines and their standard deviations '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var1 = self.var1; var2 = self.var2
    err1 = self.err1; err2 = self.err2
    # create plot
    plts = ax.linePlot([var1, var2], errorbar=[err1, err2], 
                       errorevery=[1, 3,], expand_list=['errorevery'], # expand skip interval
                       ylabel='Variables with Errors [{1:s}]', llabel=True, 
                       ylim=var1.limits(), legend=2, hline=2., vline=(2,3))
    assert len(plts) == 2
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)
    
  def testMeanAxisPlot(self):
    ''' test a simple errorbar plot with a parasitic axes showing the means '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var1 = self.var1; var2 = self.var2
    err1 = self.err1; err2 = self.err2
    # create plot
    plts = ax.linePlot([var1, var2], errorbar=[err1, err2], lparasiteMeans=True,
                       errorevery=[1, 3,], expand_list=['errorevery'], # expand skip interval
                       ylabel='Variables with Errors [{1:s}]', llabel=True, 
                       ylim=var1.limits(), legend=2, hline=2., vline=(2,3))
    assert len(plts) == 2
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)
    
  def testFancyErrorPlot(self):
    ''' test a plot with error bands with separate upper and lower limits '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var1 = self.var1; var2 = self.var2
    err1 = self.err1; err2 = self.err2
    # create plot
    plts = ax.linePlot([var1, var2], errorband=[err1, err2], 
                       edgecolor=('k',1), expand_list=('edgecolor',),
                       ylabel='Variables with Error Bands [{1:s}]', llabel=True, 
                       ylim=var1.limits(), legend=2, hline=2., vline=(2,3))
    assert len(plts) == 2
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)
  
  def testFancyBandPlot(self):
    ''' test a fancy error plot with two lines and their errors in transparent bands '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var1 = self.var1; var2 = self.var2
    lowvar1 = self.var1 - 1.; lowvar2 = None # self.var2 / 2. 
    upvar1 = self.var1 * 2.; upvar2 = self.var2  # + 1.
    # create plot
    bnds = ax.bandPlot(upper=[upvar1, upvar2], lower=[lowvar1, lowvar2], edgecolor=0.1)
    assert len(bnds) == 2
    plts = ax.linePlot([var1, var2], ylabel='Variables with Difference Bands [{1:s}]', 
                       llabel=True, ylim=var1.limits(), legend=2,)
    assert len(plts) == 2
    # add label
    ax.addLabel(label='Fancy\nError\nBands', loc=4, lstroke=False, size=None, prop=None)
  
  def testAdvancedLinePlot(self):
    ''' test more advanced options of the line plot function '''    
    var1 = self.var1; var2 = self.var2
    varatts = dict() # set some default values
    for var in var1,var2: varatts[var.name] = dict(color=var.name, marker='1', markersize=15)    
    fig,ax = getFigAx(1, title='Fancy Plot Styles', name=sys._getframe().f_code.co_name[4:], 
                      variable_plotargs=varatts, **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple,np.ndarray)) # should return a "naked" axes
    assert isinstance(ax.variable_plotargs, dict)
    # define fancy attributes
    plotatts = dict() # override some defaults
    plotatts[var1.name] = dict(color='red', marker='*', markersize=5)        
    # define fancy legend
    legend = dict(loc=2, labelspacing=0.125, handlelength=2.5, handletextpad=0.5, fancybox=True)
    # create plot
    plts = ax.linePlot([var1, var2], linestyles=('--','-.'), plotatts=plotatts, legend=legend)
    assert len(plts) == 2

  def testAxesGridLinePlot(self):
    ''' test a two panel line plot with combined legend '''        
    fig,axes = getFigAx(4, AxesGrid=True, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    #assert grid.__class__.__name__ == 'ImageGrid'
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyLocatableAxes'
    assert isinstance(axes,np.ndarray) # should return a list of axes
    var1 = self.var1; var2 = self.var2
    # create plot
    for ax in axes.ravel():
        plts = ax.linePlot([var1, var2], ylim=var1.limits(), legend=0)
        assert len(plts) == 2   
        
  def testCombinedLinePlot(self):
    ''' test a two panel line plot with combined legend '''    
    fig,axes = getFigAx(4, sharey=True, sharex=True, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert isinstance(axes,np.ndarray) # should return a list of axes
    var1 = self.var1; var2 = self.var2
    # create plot
    for i,ax in enumerate(axes.ravel()):
      plts = ax.linePlot([var1, var2], ylim=var1.limits(), legend=0,)
      ax.addTitle('Panel {:d}'.format(i+1))
      assert len(plts) == 2
    # add common legend
    fig.addSharedLegend(plots=plts)
    # add labels
    fig.addLabels(labels=None, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)
    # add a line
    ax.addHline(3)
    
    
class DistPlotTest(unittest.TestCase):  
   
  def setUp(self):
    ''' create two test variables '''
    # create axis and variable instances (make *copies* of data and attributes!)
    x1 = np.random.randn(150); xax1 = Axis(name='X1-Axis', units='X Units', length=len(x1)) 
    var1 = Variable(axes=(xax1,), data=x1.copy(), atts=dict(name='blue', units='units'))
    self.var1 = var1; self.xax1 = xax1
    x2 = np.random.randn(150); xax2 = Axis(name='X2-Axis', units='X Units', length=len(x2))
    var2 = Variable(name='purple',units='units',axes=(xax2,), data=x2)
    self.var2 = var2; self.xax2 = xax2
    # actual normal distribution
    self.dist = 'norm'
    distvar = VarRV(name=self.dist,units='units', dist=self.dist, params=(0,1))
    self.distVar = distvar
    # add to list
    self.vars = [var1, var2]
    self.axes = [xax1, xax2]
        
  def tearDown(self):
    ''' clean up '''
    for var in self.vars:     
      var.unload() # just to do something... free memory
    for ax in self.axes:
      ax.unload()
    
  ## plots for random variables and distributions

  def testBasicHistogram(self):
    ''' a simple bar plot of two normally distributed samples '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    # settings
    varlist = [self.var1, self.var2]
    nbins = 10
    # create regular histogram
    bins, ptchs = ax.histogram(varlist, bins=nbins, legend=2, alpha=0.5, rwidth=0.8, histtype='bar')
    # histtype = 'bar' | 'barstacked' | 'step' | 'stepfilled'
    assert len(ptchs) == 2
    assert len(bins) == nbins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
    #print bins[0], vmin; print bins[-1], vmax
    assert bins[0] == vmin and bins[-1] == vmax
    # add a KDE plot
    support = np.linspace(vmin, vmax, 100)
    kdevars = [var.kde(lflatten=True, lbootstrap=True, nbs=10) for var in varlist]
    # N.B.: the bootstrapping is just to test bootstrap tolerance in regular plotting methods
    ax.linePlot(kdevars, support=support, linewidth=2)
    # add label
    pval = kstest(varlist[0], varlist[1], lflatten=True) 
    pstr = 'p-value = {:3.2f}'.format(pval)
    ax.addLabel(label=pstr, loc=1, lstroke=False, lalphabet=True, size=None, prop=None)


  def testBootstrapCI(self):
    ''' test a simple line plot with two lines '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    # settings
    varlist = [self.var1, self.var2]
    nbins = 10
    # add regular histogram for comparison
    bins, ptchs = ax.histogram(varlist, bins=nbins, legend=2, alpha=0.5, rwidth=0.8, histtype='bar')
    # histtype = 'bar' | 'barstacked' | 'step' | 'stepfilled'
    assert len(ptchs) == 2
    assert len(bins) == nbins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
    #print bins[0], vmin; print bins[-1], vmax
    assert bins[0] == vmin and bins[-1] == vmax
#     generate errorbars based on bootstrap variance
#     kdeorig = []; kdeerrs = []
#     for var in varlist:
#       kdevar = var.kde(lflatten=True, lbootstrap=True, nbs=100) # generate KDE with bootstrapping
#       kdevars.append(kdevar) # regular KDE variable (DistVar)
#       kde = kdevar.pdf(support=support) # evaluate KDE and generate distribution      
#       kdeorig.append(kde(bootstrap=0)) # the first element, the actual distribution
#       kdeerrs.append(kde.std(axis='bootstrap')) # the variance of the bootstrapped sample
#     # add line plot with error bands     
#     ax.linePlot(kdeorig, errorband=kdeerrs, errorscale=0.5, linestyle='-', linewidth=2)
    # add bootstrap plot with errorbars    
    support = np.linspace(vmin, vmax, 100)
    fitvars = [var.fitDist(dist=self.dist,lflatten=True, lbootstrap=True, nbs=1000) for var in varlist] 
    ax.bootPlot(fitvars[0], support=support, errorscale=0.5, linewidth=2, lsmooth=False,
                percentiles=None, lvar=True, lvarBand=False, lmean=True, reset_color=True)
    ax.bootPlot(fitvars[1], support=support, errorscale=0.5, linewidth=2, lsmooth=False,
                percentiles=(0.25,0.75), lvar=False, lvarBand=False, lmedian=True, reset_color=False)  
    # add the actual distribution
    ax.linePlot(self.distVar, support=support, linewidth=1, marker='^')
    # add label
    pstr = "p-values for '{:s}':\n".format(self.dist)
    for var in varlist:
      pstr += '   {:<9s}   {:3.2f}\n'.format(var.name, var.kstest(dist=self.dist, asVar=False))
    pstr += '   2-samples   {:3.2f}\n'.format(kstest(varlist[0], varlist[1], lflatten=True))
    ax.addLabel(label=pstr, loc=1, lstroke=False, lalphabet=True, size=None, prop=None)

    
if __name__ == "__main__":

    
    specific_tests = []
    # LinePlot
#     specific_tests += ['BasicLinePlot']
#     specific_tests += ['BasicErrorPlot']
#     specific_tests += ['FancyErrorPlot']
#     specific_tests += ['FancyBandPlot']
#     specific_tests += ['AdvancedLinePlot']
#     specific_tests += ['CombinedLinePlot']
#     specific_tests += ['AxesGridLinePlot']    
#     specific_tests += ['MeanAxisPlot']
    # DistPlot
#     specific_tests += ['BasicHistogram']
    specific_tests += ['BootstrapCI']
    
    # list of tests to be performed
    tests = [] 
    # list of variable tests
#     tests += ['LinePlot'] 
    tests += ['DistPlot']
    

    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.iteritems():
      if key[-4:] == 'Test':
        test_classes[key[:-4]] = val


    # run tests
    report = []
    for test in tests: # test+'.test'+specific_test
      if specific_tests: 
        test_names = ['plotting_test.'+test+'Test.test'+s_t for s_t in specific_tests]
        s = unittest.TestLoader().loadTestsFromNames(test_names)
      else: s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      report.append(unittest.TextTestRunner(verbosity=2).run(s))
      
    # print summary
    runs = 0; errs = 0; fails = 0
    for name,test in zip(tests,report):
      #print test, dir(test)
      runs += test.testsRun
      e = len(test.errors)
      errs += e
      f = len(test.failures)
      fails += f
      if e+ f != 0: print("\nErrors in '{:s}' Tests: {:s}".format(name,str(test)))
    if errs + fails == 0:
      print("\n   ***   All {:d} Test(s) successfull!!!   ***   \n".format(runs))
    else:
      print("\n   ###     Test Summary:      ###   \n" + 
            "   ###     Ran {:2d} Test(s)     ###   \n".format(runs) + 
            "   ###      {:2d} Failure(s)     ###   \n".format(fails)+ 
            "   ###      {:2d} Error(s)       ###   \n".format(errs))
    
    # show plots
    pyl.show()