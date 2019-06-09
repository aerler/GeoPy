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
import pandas as pd
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


# work directory settings ("global" variable)
# the environment variable RAMDISK contains the path to the RAM disk
RAM = bool(os.getenv('RAMDISK', '')) # whether or not to use a RAM disk
# either RAM disk or data directory
workdir = os.getenv('RAMDISK', '') if RAM else '{:s}/test/'.format(os.getenv('DATA_ROOT', '')) 
if not os.path.isdir(workdir): raise IOError(workdir)
figargs = dict(stylesheet='myggplot', lpresentation=True, lpublication=False)


class SurfacePlotTest(unittest.TestCase):  
  
  def setUp(self):
    ''' create a 2D test variable '''
    # create axis and variable instances (make *copies* of data and attributes!)
    xax = Axis(name='X-Axis', units='X Units', coord=np.linspace(0,10,15))
    yax = Axis(name='Y-Axis', units='Y Units', coord=np.linspace(2,8,18))
    xx,yy = np.meshgrid(yax[:],xax[:],) # create mesh (transposed w.r.t. values)
    var0 = Variable(axes=(xax,yax), data=np.sin(xx)*np.cos(yy), atts=dict(name='Color', units='Color Units'))
    var1 = Variable(axes=(xax,yax), data=np.cos(xx)*np.sin(yy), atts=dict(name='Contour', units='Contour Units'))
    self.var0 = var0; self.var1 = var1; self.xax = xax; self.yax = yax
    # add to list
    self.axes = [xax, yax]
    self.vars = [var0, var1]
        
  def tearDown(self):
    ''' clean up '''
    for var in self.vars:     
      var.unload() # just to do something... free memory
    for ax in self.axes:
      ax.unload()
    
  ## basic plotting tests

  def testBasicSurfacePlot(self):
    ''' test a simple color/surface plot '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0
    # create plot
    vline = (2,3)
    plt = ax.surfacePlot(var0, ylabel='custom label [{UNITS:s}]', llabel=True, lprint=True,
                         ylim=self.yax.limits(), clim=var0.limits(), hline=2., vline=vline)
    assert plt
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)

  def testBasicContourPlot(self):
    ''' test a simple color/surface plot '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
#     assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0
    # create plot
    plt = ax.surfacePlot(var0, ylabel='custom label [{UNITS:s}]', llabel=True, lprint=True, 
                         aspect=1, lcontour=True, clevs=3,
                         ylim=self.xax.limits(), hline=(2,8), vline=None)
    assert plt
    plt = ax.surfacePlot(var0, lcontour=True, clevs=10, lfilled=False, colors='k', linewidth=2)
    assert plt
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)

  def testIrregularSurfacePlot(self):
    ''' test a color/surface plot with irregular coordiante variables '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0
    # create coordiante variables
    xax,yax = var0.axes
    xx,yy = np.meshgrid(xax[:],yax[:], indexing='ij')
    xax = Variable(name='X Coordinate', units='X Units', data=xx, axes=var0.axes)
    yax = Variable(name='Y Coordinate', units='Y Units', data=yy, axes=var0.axes)
    # create plot
    plt = ax.surfacePlot(var0, flipxy=False, clog=False, xax=xax, yax=yax,
                         llabel=True, lprint=True, clim=var0.limits(),)
    assert plt
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)

  def testSharedColorbar(self):
    ''' test a simple shared colorbar between to surface plots '''
    name = sys._getframe().f_code.co_name[4:]    
    fig,axes = getFigAx(2, name=name, title=name, **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert isinstance(axes,(np.ndarray,list,tuple)) # should return a "naked" axes
    # create plot
    for ax,var in zip(axes,self.vars):
        ax.surfacePlot(var, lprint=True, clim=var.limits(), centroids=dict(color='black'))        
    # add labels
    fig.addLabels(loc=4, lstroke=False, lalphabet=True,)
    fig.updateSubplots(left=0.02, bottom=0.03)
    # add shared colorbar
    cbar = fig.addSharedColorbar(ax=axes[1], location='bottom', clevs=3, lunits=True, length=0.8)
    assert cbar

  def testLogSurfacePlot(self):
    ''' test a surface plot with one logarithmic and one linear panel and a shared colorbar '''
    name = sys._getframe().f_code.co_name[4:]    
    fig,axes = getFigAx(2, name=name, title=name, **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert isinstance(axes,(np.ndarray,list,tuple)) # should return a "naked" axes
    # create plot
    expvar = ( self.var0 * 10 ).exp() # large exponents for large effects
    axes[0].surfacePlot(expvar, lprint=True, clog=False, )        
    axes[1].surfacePlot(expvar, lprint=True, clog=True, )        
    # add labels
    fig.addLabels(loc=4, lstroke=False, lalphabet=True,)
    fig.updateSubplots(left=0.02, bottom=0.03)
    # add shared colorbar
    cbar = fig.addSharedColorbar(ax=axes[0], location='bottom', clevs=3, lunits=True, length=0.8)
    assert cbar


class LinePlotTest(unittest.TestCase):  
  
  ldatetime = False # does not work with all plots...
   
  def setUp(self):
    ''' create two test variables '''
    # create axis and variable instances (make *copies* of data and attributes!)
    x1 = np.linspace(0,10,15); 
    x2 = np.linspace(2,8,18);
    if self.ldatetime:
        start_datetime, end_datetime = pd.to_datetime('1981-05-01'), pd.to_datetime('1981-05-16')
        t1 = np.arange(start_datetime, end_datetime, dtype='datetime64[D]') 
        xax1 = Axis(name='Time1-Axis', units='X Time', coord=t1) 
        t2 = np.arange(start_datetime, end_datetime+np.timedelta64(3, 'D'), dtype='datetime64[D]')
        xax2 = Axis(name='Time2-Axis', units='X Time', coord=t2)
    else:
        xax1 = Axis(name='X1-Axis', units='X Units', coord=x1)
        xax2 = Axis(name='X2-Axis', units='X Units', coord=x2)
    var0 = Variable(axes=(xax1,), data=np.sin(x1), atts=dict(name='relative', units=''))
    var1 = Variable(axes=(xax1,), data=x1.copy(), atts=dict(name='blue', units='units'))
    self.var0 = var0; self.var1 = var1; self.xax1 = xax1
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
    vline = np.datetime64('1981-05-16') if self.ldatetime else (2,3)
    plts = ax.linePlot([var1, var2], ylabel='custom label [{UNITS:s}]', llabel=True, lprint=True,
                       ylim=var1.limits(), legend=2, hline=2., vline=vline)
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
                       ylabel='Variables with Errors [{UNITS:s}]', llabel=True, 
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
                       ylabel='Variables with Errors [{UNITS:s}]', llabel=True, 
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
    errup1 = self.var1+self.err1*2.; errup2 = self.var2+self.err2*2.
    errdn1 = self.var1-self.err1/2.; errdn2 = self.var2-self.err2/2.
    # create plot
    plts = ax.linePlot([var1, var2], errorband=([errdn1, errdn2],[errup1, errup2]), 
                       edgecolor=('k',1), expand_list=('edgecolor',),
                       ylabel='Variables with Error Bands [{UNITS:s}]', llabel=True, 
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
    plts = ax.linePlot([var1, var2], ylabel='Variables with Difference Bands [{UNITS:s}]', 
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
    fig,axes = getFigAx(4, lAxesGrid=True, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
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
    x1 = np.random.randn(180); xax1 = Axis(name='X1-Axis', units='X Units', length=len(x1)) 
    var1 = Variable(axes=(xax1,), data=x1.copy(), atts=dict(name='blue', units='units'))
    self.var1 = var1; self.xax1 = xax1
    x2 = np.random.randn(180); xax2 = Axis(name='X2-Axis', units='X Units', length=len(x2))
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
    nbins = 15
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


  def testWeightedHistogram(self):
    ''' a weighted historgram '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    # settings
    nleg = 1 # for regular Normal distributions
    varlist = [self.var1, self.var2]
    for var in varlist:
        nleg = 5 
        var.data_array=np.linspace(-1, 1, num=var.shape[0])
    weights = [np.arange(0,self.var1.shape[0]),np.arange(self.var2.shape[0],0,-1)]
    nbins = 15
    # create regular histogram
    bins, ptchs = ax.histogram(varlist, bins=nbins, weights=weights, legend=nleg, alpha=0.5, rwidth=0.8, 
                               histtype='bar', flipxy=True)
    # histtype = 'bar' | 'barstacked' | 'step' | 'stepfilled'
    assert len(ptchs) == 2
    assert len(bins) == nbins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
    #print bins[0], vmin; print bins[-1], vmax
    assert bins[0] == vmin and bins[-1] == vmax
    # add a KDE plot
    support = np.linspace(vmin, vmax, 100)
    kdevars = [var.kde(weights=w, lflatten=True, lbootstrap=False, nbs=10) for var,w in zip(varlist,weights)]
    # N.B.: the bootstrapping is just to test bootstrap tolerance in regular plotting methods
    ax.linePlot(kdevars, support=support, linewidth=2, flipxy=True)


  def testBootstrapCI(self):
    ''' test a line plot with confidence intervals from bootstrapping '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    # settings
    varlist = [self.var1, self.var2]
    nbins = 15
    # add regular histogram for comparison
    bins, ptchs = ax.histogram(varlist, bins=nbins, legend=2, alpha=0.5, rwidth=0.8, histtype='bar')
    # histtype = 'bar' | 'barstacked' | 'step' | 'stepfilled'
    assert len(ptchs) == 2
    assert len(bins) == nbins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
    #print bins[0], vmin; print bins[-1], vmax
    assert bins[0] == vmin and bins[-1] == vmax
    # add bootstrap plot with errorbars    
    support = np.linspace(vmin, vmax, 100)
    fitvars = [var.fitDist(dist=self.dist,lflatten=True, lbootstrap=True, nbs=1000) for var in varlist] 
    ax.bootPlot(fitvars[0], support=support, errorscale=0.5, linewidth=2, lsmooth=False,
                percentiles=None, lvar=True, lvarBand=False, lmean=True, reset_color=True)
    ax.bootPlot(fitvars[1], support=support, errorscale=0.5, linewidth=2, lsmooth=False,
                percentiles=(0.25,0.75), lvar=False, lvarBand=False, lmedian=True, reset_color=False)  
    # add the actual distribution
    ax.linePlot(self.distVar, support=support, linewidth=1, marker='^')
    # add statistical info
    pstr = "p-values for '{:s}':\n".format(self.dist)
    for var in varlist:
      pstr += '   {:<9s}   {:3.2f}\n'.format(var.name, var.kstest(dist=self.dist, asVar=False))
    pstr += '   2-samples   {:3.2f}\n'.format(kstest(varlist[0], varlist[1], lflatten=True))
    ax.addLabel(label=pstr, loc=1, lstroke=False, lalphabet=True, size=None, prop=None)


  def testSamplePlot(self):
    ''' test a line and band plot showing the mean/median and given percentiles '''    
    fig,ax = getFigAx(1, name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    # settings
    varlist = [self.var1, self.var2]
    nbins = 15
    # add regular histogram for comparison
    bins, ptchs = ax.histogram(varlist, bins=nbins, legend=2, alpha=0.5, rwidth=0.8, histtype='bar')
    # histtype = 'bar' | 'barstacked' | 'step' | 'stepfilled'
    assert len(ptchs) == 2
    assert len(bins) == nbins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
    #print bins[0], vmin; print bins[-1], vmax
    assert bins[0] == vmin and bins[-1] == vmax
    # add bootstrap plot with errorbars    
    support = np.linspace(vmin, vmax, 100)
    fitvars = [var.fitDist(dist=self.dist,lflatten=True, lbootstrap=True, nbs=1000) for var in varlist]
    # add the actual distribution
    ax.linePlot(self.distVar, support=support, linewidth=1, marker='^')
    # add simple sample (using bootstrap axis as sample 
    ax.samplePlot(fitvars[0], support=support, linewidth=2, lsmooth=False, 
                  sample_axis='bootstrap', percentiles=(0.05,0.95), lmedian=True, reset_color=True)
    # replicate axis and add random noise to mean
    rndfit = fitvars[1].insertAxis(axis='sample',iaxis=0,length=100)
    rndfit.data_array += np.random.randn(*rndfit.shape)/100. 
    ax.samplePlot(rndfit, support=support, linewidth=2, lsmooth=False, 
                  bootstrap_axis='bootstrap', sample_axis=('no_axis','sample'),
                  percentiles=(0.25,0.75), lmean=True, reset_color=False)  
    # add statistical info
    pstr = "p-values for '{:s}':\n".format(self.dist)
    for var in varlist:
      pstr += '   {:<9s}   {:3.2f}\n'.format(var.name, var.kstest(dist=self.dist, asVar=False))
    pstr += '   2-samples   {:3.2f}\n'.format(kstest(varlist[0], varlist[1], lflatten=True))
    ax.addLabel(label=pstr, loc=1, lstroke=False, lalphabet=True, size=None, prop=None)



class PolarPlotTest(unittest.TestCase):  

  def setUp(self):
    ''' create two test variables '''
    # define plot ranges
    self.thetamin = 0.; self.Rmin = 0.; self.thetamax = 2*np.pi; self.Rmax = 2.
    # create theta axis and variable instances (values are radius values, I believe)
    theta1 = np.linspace(self.thetamin,self.thetamax,361)
    thax1 = Axis(atts=dict(name='$\\theta$-Axis', units='Radians'), coord=theta1) 
    var0 = Variable(axes=(thax1,), data=np.sin(theta1), atts=dict(name='Blue', units='units'))
    tmp = theta1.copy()*(self.Rmax-self.Rmin)/(self.thetamax-self.thetamin)
    var1 = Variable(axes=(thax1,), data=tmp, atts=dict(name='Red', units='units'))
    self.var0 = var0; self.var1 = var1; self.xax1 = theta1
    # create error variables with random noise
    noise0 = np.random.rand(len(thax1))*var0.data_array.std()/2.
    err0 = Variable(axes=(thax1,), data=noise0, atts=dict(name='Blue Noise', units='units'))
    noise1 = np.random.rand(len(thax1))*var1.data_array.std()/2.
    err1 = Variable(axes=(thax1,), data=noise1, atts=dict(name='Red Noise', units='units'))
    self.err1 = err1; self.err0 = err0
    # add to list
    self.vars = [var0, var1]
    self.errs = [err0, err1]
    self.axes = [thax1,]

  def tearDown(self):
    ''' clean up '''
    for var in self.vars:     
      var.unload() # just to do something... free memory
    for ax in self.axes:
      ax.unload()
    
  ## basic plotting tests

  def testBasicLinePlot(self):
    ''' test a simple line plot with two lines '''    
    fig,ax = getFigAx(1, lPolarAxes=True, 
                      name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyPolarAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0; var1 = self.var1
    # create plot
    plts = ax.linePlot([var0, var1], ylabel='custom label [{UNITS:s}]', llabel=True,
                       ylim=(self.Rmin,self.Rmax), legend=2, hline=1., vline=(np.pi/4.,np.pi/2.,np.pi))
    assert len(plts) == 2
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)
      
  def testAdvancedLinePlot(self):
    ''' test more advanced options of the line plot function '''    
    var1 = self.var1; var0 = self.var0
    varatts = dict() # set some default values
    fig,ax = getFigAx(1, title='Fancy Plot Styles', name=sys._getframe().f_code.co_name[4:], 
                      lPolarAxes=True, variable_plotargs=varatts, **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'MyPolarAxes'
    assert not isinstance(ax,(list,tuple,np.ndarray)) # should return a "naked" axes
    assert isinstance(ax.variable_plotargs, dict)
    # define fancy attributes
    plotatts = dict() # override some defaults
    plotatts[var0.name] = dict(color='blue', marker='*', markersize=5, markevery=20)        
    plotatts[var1.name] = dict(color='red', marker='*', markersize=5, markevery=20)        
    # define fancy legend
    legend = dict(loc=2, labelspacing=0.125, handlelength=2.5, handletextpad=0.5, fancybox=True)
    # create plot
    plts = ax.linePlot([var0, var1], linestyles=('--','-.'), plotatts=plotatts, legend=legend, ylim=(self.Rmin,self.Rmax))
    assert len(plts) == 2
       
  def testCombinedLinePlot(self):
    ''' test a two panel line plot with combined legend '''    
    fig,axes = getFigAx(4, lPolarAxes=[True,False]*2,
                        name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ in ('MyPolarAxes','MyAxes')
    assert isinstance(axes,np.ndarray) # should return a list of axes
    var1 = self.var1; var0 = self.var0
    # create plot
    for i,ax in enumerate(axes.ravel()):
      plts = ax.linePlot([var0, var1], legend=False, ylim=(self.Rmin,self.Rmax))
      ax.addTitle('Panel {:d}'.format(i+1))
      assert len(plts) == 2
    # add common legend
    fig.addSharedLegend(plots=plts)
    # add labels
    fig.addLabels(labels=None, loc='upper left', lstroke=False, lalphabet=True, size=None, prop=None)
    # add a line
    ax.addHline(3)
    

class TaylorPlotTest(PolarPlotTest):  

  def setUp(self):
    ''' create a reference and two test variables for Taylor plot'''
    self.thetamin = 0.; self.Rmin = 0.; self.thetamax = np.pi/2.; self.Rmax = 2.
    # create axis and variable instances (make *copies* of data and attributes!)
    self.x1 = np.linspace(0,10,11); self.xax1 = Axis(name='X1-Axis', units='X Units', coord=self.x1)
    self.data0 = np.sin(self.x1)
    self.var0 = Variable(axes=(self.xax1,), data=self.data0, atts=dict(name='Reference', units='units'))
    # create error variables with random noise
    self.data1 = self.data0 + ( np.random.rand(len(self.xax1))-0.5 )*0.5
    self.var1 = Variable(axes=(self.xax1,), data=self.data1, atts=dict(name='Blue', units='units'))
    self.data2 = self.data0 + ( np.random.rand(len(self.xax1))-0.5 )*1.5
    self.var2 = Variable(axes=(self.xax1,), data=self.data2, atts=dict(name='Red', units='units'))
    self.data3 = 1. + np.random.rand(len(self.xax1))*1.5
    self.var3 = Variable(axes=(self.xax1,), data=self.data3, atts=dict(name='Random', units='units'))
    # add to list
    self.vars = [self.var0, self.var1, self.var2, self.var3]
    self.data = [self.data0, self.data1, self.data2, self.data3]
    self.axes = [self.xax1,]
    
  ## basic plotting tests

  def testBasicScatterPlot(self):
    ''' test a simple scatter plot with two variables '''    
    fig,ax = getFigAx(1, lTaylor=True, axes_args=dict(std=1.5),
                      name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
#     assert fig.axes_class.__name__ == 'TaylorAxes' # in this case, just regular rectilinear axes
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0; var1 = self.var1; # var2 = self.var2
    # create plot
    plts = ax.scatterPlot(xvars=var0, yvars=var1, llabel=True, legend=0, lprint=True)
    assert len(plts) == 1
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)

  def testBasicTaylorPlot(self):
    ''' test a simple Taylor plot with two variables/timeseries and a reference '''    
    fig,ax = getFigAx(1, lTaylor=True, axes_args=dict(std=1.2, leps=True),
                       name=sys._getframe().f_code.co_name[4:], **figargs) # use test method name as title
    assert fig.__class__.__name__ == 'MyFigure'
    assert fig.axes_class.__name__ == 'TaylorAxes'
    assert not isinstance(ax,(list,tuple)) # should return a "naked" axes
    var0 = self.var0; var1 = self.var1; var2 = self.var2
    # set up reference
#     print(ax.setReference(var0))
#     ax.showRefLines()
    # add some dots...
    plts, = ax.taylorPlot([var1, var2], reference=var0, rmse_lines=6)
    assert len(plts) == 2
    plts, = ax.taylorPlot([var1*1.5, var2/2., var0], reference='Reference', loverride=True)
    assert len(plts) == 2
    # add a negative correlation
    negvar1 = var1 * -1
    plts, = ax.taylorPlot(negvar1, legend=1, lprint=True, label_ext=' (neg.)')
    assert len(plts) == 1
    # add a random variable (should have no correlation
    plts, = ax.taylorPlot(self.var3, legend=1, pval=0.01, lprint=True, linsig=True)
    assert len(plts) == 1
    # add label
    ax.addLabel(label=0, loc=4, lstroke=False, lalphabet=True, size=None, prop=None)


if __name__ == "__main__":

    
    specific_tests = []
    # SurfacePlot
#     specific_tests += ['BasicSurfacePlot']
#     specific_tests += ['BasicContourPlot']
#     specific_tests += ['IrregularSurfacePlot']
#     specific_tests += ['SharedColorbar']
#     specific_tests += ['LogSurfacePlot']
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
    specific_tests += ['WeightedHistogram']
#     specific_tests += ['BootstrapCI']
#     specific_tests += ['SamplePlot']
    # PolarPlot
#     specific_tests += ['BasicLinePlot']
#     specific_tests += ['AdvancedLinePlot']
#     specific_tests += ['CombinedLinePlot']
    # TaylorPlot
#     specific_tests += ['BasicScatterPlot']
#     specific_tests += ['BasicTaylorPlot']
        
    # list of tests to be performed
    tests = [] 
    # list of variable tests
#     tests += ['SurfacePlot'] 
#     tests += ['LinePlot'] 
    tests += ['DistPlot']
#     tests += ['PolarPlot']
#     tests += ['TaylorPlot']
    

    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.items():
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
      if e+ f != 0: print(("\nErrors in '{:s}' Tests: {:s}".format(name,str(test))))
    if errs + fails == 0:
      print(("\n   ***   All {:d} Test(s) successfull!!!   ***   \n".format(runs)))
    else:
      print(("\n   ###     Test Summary:      ###   \n" + 
            "   ###     Ran {:2d} Test(s)     ###   \n".format(runs) + 
            "   ###      {:2d} Failure(s)     ###   \n".format(fails)+ 
            "   ###      {:2d} Error(s)       ###   \n".format(errs)))
    
    # show plots
    pyl.show()