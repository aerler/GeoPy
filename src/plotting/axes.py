'''
Created on Dec 11, 2014

A custom Axes class that provides some specialized plotting functions and retains variable information.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid.axes_divider import LocatableAxes
from types import NoneType
# internal imports
from geodata.base import Variable
from geodata.misc import ListError, ArgumentError, isEqual, VariableError,\
  AxisError
from plotting.misc import smooth, checkVarlist, getPlotValues, errorPercentile, checkSample
from collections import OrderedDict
from utils.misc import binedges, expandArgumentList

# list of plot arguments that apply only to lines
line_args = ('lineformats','linestyles','markers','lineformat','linestyle','marker')

## new axes class
class MyAxes(Axes): 
  ''' 
    A custom Axes class that provides some specialized plotting functions and retains variable 
    information. The custom Figure uses this Axes class by default.
  '''
  variables          = None
  plots              = None
  variable_plotargs  = None
  dataset_plotargs   = None
  plot_labels        = None
  title_height       = 0.025 # fraction of figure height
  title_size         = 'medium'
  legend_handle      = None
  flipxy             = False
  xname              = None
  xunits             = None
  xpad               = 2    # pixels
  xtop               = False
  yname              = None
  yunits             = None
  ypad               = 0
  yright             = False
  figure             = None
  parasite_axes      = None
  axes_shift         = None
  
  def __init__(self, *args, **kwargs):
    ''' constructor to initialize some variables / counters '''  
    # call parent constructor
    super(MyAxes,self).__init__(*args, **kwargs)
    self.variables = OrderedDict() # save variables by label
    self.plots = OrderedDict() # save plot objects by label
    self.axes_shift = np.zeros(4, dtype=np.float32)
    self.updateAxes(mode='shift') # initialize
    
    
  def linePlot(self, varlist, varname=None, bins=None, support=None, errorbar=None, errorband=None,  
               legend=None, llabel=True, labels=None, hline=None, vline=None, title=None, lignore=False,        
               flipxy=None, xlabel=True, ylabel=True, xticks=True, yticks=True, reset_color=None, 
               lparasiteMeans=False, lparasiteErrors=False, parasite_axes=None, lrescale=False, 
               scalefactor=1., offset=0., bootstrap_axis='bootstrap',
               xlog=False, ylog=False, xlim=None, ylim=None, lsmooth=False, lperi=False, lprint=False,
               expand_list=None, lproduct='inner', method='pdf', plotatts=None, **plotargs):
    ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on 
        variable properties; extra keyword arguments (plotargs) are passed through expandArgumentList,
        before being passed to Axes.plot(). '''
    ## figure out variables
    varlist = checkVarlist(varlist, varname=varname, ndim=1, bins=bins, support=support, 
                           method=method, lignore=lignore, bootstrap_axis=bootstrap_axis)
    if errorbar: errlist = checkVarlist(errorbar, varname=varname, ndim=1, lignore=lignore, 
                                        bins=bins, support=support, method=method, bootstrap_axis=bootstrap_axis)
    else: errlist = [None]*len(varlist) # no error bars
    if errorband: bndlist = checkVarlist(errorband, varname=varname, ndim=1, lignore=lignore, 
                                         bins=bins, support=support, method=method, bootstrap_axis=bootstrap_axis)
    else: bndlist = [None]*len(varlist) # no error bands
    assert len(varlist) == len(errlist) == len(bndlist)
    # initialize axes names and units
    self.flipxy = flipxy
    if self.flipxy: varname,varunits,axname,axunits = self.xname,self.xunits,self.yname,self.yunits
    else: axname,axunits,varname,varunits = self.xname,self.xunits,self.yname,self.yunits
    ## figure out plot arguments
    # reset color cycle
    if reset_color is False: pass
    elif reset_color is True: self.set_color_cycle(None) # reset
    else: self.set_color_cycle(reset_color)
    # figure out label list
    if labels is None: labels = self._getPlotLabels(varlist)           
    elif len(labels) < len(varlist): raise ArgumentError, "Incompatible length of varlist and labels."
    elif len(labels) > len(varlist): labels = labels[:len(varlist)] # truncate 
    label_list = labels if llabel else [None]*len(labels) # used for plot labels later
    assert len(labels) == len(varlist)
    # finally, expand keyword arguments
    plotargs = self._expandArgumentList(labels=label_list, expand_list=expand_list, 
                                        lproduct=lproduct, plotargs=plotargs)
    assert len(plotargs) == len(varlist)
    # initialize parasitic axis for means
    if lparasiteMeans and self.parasite_axes is None:
      if parasite_axes is None: self.addParasiteAxes()
      else: self.addParasiteAxes(**parasite_axes)
    ## generate individual line plots
    plts = [] # list of plot handles
    for label,var in zip(labels,varlist): self.variables[label] = var # save plot variables
    # loop over variables and plot arguments
    N = len(varlist); xlen = ylen = None
    for n,var,errvar,bndvar,plotarg,label in zip(xrange(N),varlist,errlist,bndlist,plotargs,labels):
      if var is not None:
        varax = var.axes[0]
        # scale axis and variable values 
        axe, axunits, axname = self._getPlotValues(varax, checkunits=axunits, laxis=True, lperi=lperi)
        val, varunits, varname = self._getPlotValues(var, lrescale=lrescale, scalefactor=scalefactor, offset=offset,
                                                     checkunits=varunits, lsmooth=lsmooth, lperi=lperi, lshift=True)
        if errvar is not None: # for error bars
          err, varunits, errname = self._getPlotValues(errvar, lrescale=lrescale, scalefactor=scalefactor, offset=offset, 
                                                       checkunits=varunits, lsmooth=lsmooth, lperi=lperi, lshift=False); del errname
        else: err = None
        if bndvar is not None: # semi-transparent error bands
          bnd, varunits, bndname = self._getPlotValues(bndvar, lrescale=lrescale, scalefactor=scalefactor, offset=offset,
                                                       checkunits=varunits, lsmooth=lsmooth, lperi=lperi, lshift=False); del bndname
        else: bnd = None
        # variable and axis scaling is not always independent...
        if var.plot is not None and varax.plot is not None: 
          if varax.units != axunits and var.plot.preserve == 'area':
            val /= varax.plot.scalefactor
        # N.B.: other scaling behavior could be added here
        if lprint: print varname, varunits, np.nanmean(val), np.nanstd(val)   
        # update plotargs from defaults
        plotarg = self._getPlotArgs(label=label, var=var, llabel=llabel, plotatts=plotatts, plotarg=plotarg)
        plotarg['fmt'] = plotarg.pop('lineformat','') # rename (I prefer a different name)
        # N.B.: '' (empty string) is the default, None means no line is plotted, only errors!
        # extract arguments for error band
        bndarg    = plotarg.pop('bandarg',dict())
        where     = plotarg.pop('where',None)
        bandalpha = plotarg.pop('bandalpha',0.5)
        edgecolor = plotarg.pop('edgecolor',0.5)
        facecolor = plotarg.pop('facecolor',None)
        errorscale = plotarg.pop('errorscale',None)
        errorevery = plotarg.pop('errorevery',None)
        if errorscale is not None: 
          errorscale = errorPercentile(errorscale)
          if err is not None: err *= errorscale
        if errorevery is None:
          errorevery = len(axe)//25 + 1
        # figure out orientation and call plot function
        if self.flipxy: # flipped axes
          xlen = len(val); ylen = len(axe) # used later
          plt = self.errorbar(val, axe, xerr=err, yerr=None, errorevery=errorevery, **plotarg)[0]
          if lparasiteMeans: raise NotImplementedError
        else: # default orientation
          xlen = len(axe); ylen = len(val) # used later
          plt = self.errorbar(axe, val, xerr=None, yerr=err, errorevery=errorevery, **plotarg)[0]
          if lparasiteMeans:
            perr = err if err is not None else bnd # wouldn't work with bands, so use normal errorbars
            self.addParasiteMean(val, errors=perr if lparasiteErrors else None, n=n, N=N, lperi=lperi, style='myerrorbar', **plotarg)
        # figure out parameters for error bands
        if bnd is not None: 
          if errorscale is not None: bnd *= errorscale
          self._drawBand(axe, val+bnd, val-bnd, where=where, color=(facecolor or plt.get_color()), 
                         alpha=bandalpha*plotarg.get('alpha',1.), edgecolor=edgecolor, **bndarg)  
        plts.append(plt); self.plots[label] = plt
      else: plts.append(None)
    ## format axes and add annotation
    # set axes labels  
    if not lrescale: # don't reset name/units when variables were rescaled to existing axes
      if self.flipxy: self.xname,self.xunits,self.yname,self.yunits = varname,varunits,axname,axunits
      else: self.xname,self.xunits,self.yname,self.yunits = axname,axunits,varname,varunits
    # apply standard formatting and annotation
    self.formatAxesAndAnnotation(title=title, legend=legend, xlabel=xlabel, ylabel=ylabel, 
                                 hline=hline, vline=vline, xlim=xlim, xlog=xlog, xticks=xticks, 
                                 ylim=ylim, ylog=ylog, yticks=yticks, xlen=xlen, ylen=ylen)
    # return handles to line objects
    return plts


  def bandPlot(self, upper=None, lower=None, varname=None, bins=None, support=None, lignore=False,   
               legend=None, llabel=False, labels=None, hline=None, vline=None, title=None,   
               lrescale=False, scalefactor=1., offset=0., bootstrap_axis='bootstrap',     
               flipxy=None, xlabel=True, ylabel=True, xticks=True, yticks=True, reset_color=None, 
               xlog=None, ylog=None, xlim=None, ylim=None, lsmooth=False, lperi=False, lprint=False, 
               expand_list=None, lproduct='inner', method='pdf', plotatts=None, **plotargs):
    ''' A function to draw a colored bands between two lists of 1D variables representing the upper
        and lower limits of the bands; extra keyword arguments (plotargs) are passed through 
        expandArgumentList, before being passed on to Axes.fill_between() (used to draw bands). '''
    ## figure out variables
    upper = checkVarlist(upper, varname=varname, ndim=1, bins=bins, bootstrap_axis=bootstrap_axis, 
                               support=support, method=method, lignore=lignore)
    lower = checkVarlist(lower, varname=varname, ndim=1, bins=bins, bootstrap_axis=bootstrap_axis, 
                               support=support, method=method, lignore=lignore)
    assert len(upper) == len(lower)
    # initialize axes names and units
    self.flipxy = flipxy
    if self.flipxy: varname,varunits,axname,axunits = self.xname,self.xunits,self.yname,self.yunits
    else: axname,axunits,varname,varunits = self.xname,self.xunits,self.yname,self.yunits
    ## figure out plot arguments
    # reset color cycle
    if reset_color is False: pass
    elif reset_color is True: self.set_color_cycle(None) # reset
    else: self.set_color_cycle(reset_color)
    # figure out label list
    if labels is None: labels = self._getPlotLabels(upper)           
    elif len(labels) != len(upper): raise ArgumentError, "Incompatible length of varlist and labels."
    label_list = labels if llabel else [None]*len(labels) # used for plot labels later
    assert len(labels) == len(lower)
    # finally, expand keyword arguments
    plotargs = self._expandArgumentList(labels=label_list, expand_list=expand_list, 
                                        lproduct=lproduct, plotargs=plotargs)
    assert len(plotargs) == len(lower)
    ## generate individual line plots
    bnds = [] # list of plot handles
    xlen = ylen = None # initialize, incase list is empty
    for label,upvar,lowvar in zip(labels,upper,lower): 
      self.variables[str(label)+'_bnd'] = (upvar,lowvar) # save band variables under special name
    # loop over variables and plot arguments
    for upvar,lowvar,plotarg,label in zip(upper,lower,plotargs,labels):
      if upvar or lowvar:
        if upvar:
          varax = upvar.axes[0]
          assert lowvar is None or ( lowvar.hasAxis(varax.name) and lowvar.ndim == 1 )
        if lowvar:
          varax = lowvar.axes[0]
          assert upvar is None or ( upvar.hasAxis(varax.name) and upvar.ndim == 1 )          
        # scale axis and variable values 
        axe, axunits, axname = self._getPlotValues(varax, checkunits=axunits, laxis=True, lperi=lperi)
        if upvar: up, varunits, varname = self._getPlotValues(upvar, lrescale=lrescale, scalefactor=scalefactor, offset=offset, 
                                                              checkunits=varunits, lsmooth=lsmooth, lperi=lperi) 
        else: up = np.zeros_like(axe)
        if lowvar: low, varunits, varname = self._getPlotValues(lowvar, lrescale=lrescale, scalefactor=scalefactor, offset=offset, 
                                                                checkunits=varunits, lsmooth=lsmooth, lperi=lperi)
        else: low = np.zeros_like(axe)
        # variable and axis scaling is not always independent...
        if upvar.plot is not None and varax.plot is not None: 
          if varax.units != axunits and upvar.plot.preserve == 'area':
            up /= varax.plot.scalefactor; low /= varax.plot.scalefactor
        # N.B.: other scaling behavior could be added here
        if lprint: print varname, varunits, np.nanmean(up), np.nanmean(low)           
        if lsmooth: up = smooth(up); low = smooth(low)
        # update plotargs from defaults
        plotarg = self._getPlotArgs(label=label, var=upvar, llabel=llabel, plotatts=plotatts, plotarg=plotarg)
        ## draw actual bands 
        bnd = self._drawBand(axe, low, up, **plotarg)
        # book keeping
        if self.flipxy: xlen, ylen = len(low), len(axe) 
        else: xlen, ylen = len(axe), len(low)
        bnds.append(bnd); self.plots[label] = bnd
      else: bnds.append(None)
    ## format axes and add annotation
    # set axes labels  
    if self.flipxy: self.xname,self.xunits,self.yname,self.yunits = varname,varunits,axname,axunits
    else: self.xname,self.xunits,self.yname,self.yunits = axname,axunits,varname,varunits
    # apply standard formatting and annotation
    self.formatAxesAndAnnotation(title=title, legend=legend, xlabel=xlabel, ylabel=ylabel, 
                                 hline=hline, vline=vline, xlim=xlim, xlog=xlog, xticks=xticks, 
                                 ylim=ylim, ylog=ylog, yticks=yticks, xlen=xlen, ylen=ylen)
    # return handles to line objects
    return bnds 

  def _drawBand(self, axes, upper, lower, where=None, color=None, alpha=0.5, edgecolor=None, **bndarg):  
    ''' function to add an error band to a plot '''
    # get color from line object        
    CC = mpl.colors.ColorConverter()
    if color is None: color = self._get_lines.color_cycle.next()
    color = CC.to_rgb(color)
    # make darker edges
    if edgecolor is None: edgecolor = 0.5
    if alpha is None: alpha = 0.5
    elif isinstance(edgecolor,(int,np.int)): edgecolor = float(edgecolor)
    if isinstance(edgecolor,(float,np.float)): 
      edgecolor = tuple(c*edgecolor for c in color) # slightly darker edges
    # construct keyword arguments to fill_between  
    bndarg['edgecolor'] = edgecolor
    bndarg['facecolor'] = color
    bndarg['where'] = where
    bndarg['alpha'] = alpha
    # clean up plot arguments
    band_args = {key:value for key,value in bndarg.iteritems() if key not in line_args}
    if self.flipxy: self.fill_betweenx(y=axes, x1=lower, x2=upper, **band_args)
    else: self.fill_between(x=axes, y1=lower, y2=upper, interpolate=True, **band_args) # interpolate=True
  
  def samplePlot(self, varlist, varname=None, bins=None, support=None, percentiles=(0.25,0.75),   
                 sample_axis=None, lmedian=None, median_fmt='', lmean=True, mean_fmt='', 
                 bootstrap_axis=None, lrescale=False, scalefactor=1., offset=0., colors=None,
                 legend=None, llabel=True, labels=None, hline=None, vline=None, title=None,        
                 flipxy=None, xlabel=True, ylabel=False, xticks=True, yticks=True, reset_color=None, 
                 xlog=False, ylog=False, xlim=None, ylim=None, lsmooth=False, lprint=False,
                 lignore=False, expand_list=None, lproduct='inner', plotatts=None, method='pdf',
                 where=None, bandalpha=None, edgecolor=None, facecolor=None, bandarg=None, **plotargs):
    ''' A function to draw moments of a distribution/sample using line-styles and bands '''
    # plot mean
    if lmean: 
      # don't overwrite varlist and sample_axis (yet)
      meanlist, mean_axis = checkSample(varlist, varname=varname, bins=bins, support=support, 
                                        method=method, lignore=lignore, sample_axis=sample_axis, 
                                        temporary_sample_axis='temporary_sample_axis',
                                        bootstrap_axis=bootstrap_axis, lmergeBootstrap=False)
      means = [] # compute the means over the sample_axis; variables without sample_axis are used as is (and removed later)
      for var in meanlist:
        if var is None: means.append(None)
        elif var.hasAxis(mean_axis): means.append(var.mean(axis=mean_axis)) 
        else: 
          var.name += '_mean'; means.append(var) # don't confuse the naming scheme...
      plts = self.linePlot(varlist=means, llabel=llabel, labels=labels, lineformat=mean_fmt, colors=colors,
                           flipxy=flipxy, reset_color=reset_color, lsmooth=lsmooth, lprint=lprint,
                           legend=legend, hline=hline, vline=vline, title=title, xlog=xlog, ylog=ylog,
                           xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim,
                           lrescale=lrescale, scalefactor=scalefactor, offset=offset,
                           plotatts=plotatts, expand_list=expand_list, lproduct=lproduct, **plotargs)
      # these switches are not needed anymore (prevent duplication)
      legend=False; llabel=False; labels=None; hline=None; vline=None 
      title=None; reset_color=False; lprint=False
      # get line colors to use in all subsequent plots 
      colors = ['' if plt is None else plt.get_color() for plt in plts] # color argument has to be string
    # check and preprocess again, this time merge sample_axis with bootstrap_axis 
    varlist, sample_axis = checkSample(varlist, varname=varname, bins=bins, support=support, 
                                       method=method, lignore=lignore, sample_axis=sample_axis, 
                                       temporary_sample_axis='temporary_sample_axis',
                                       bootstrap_axis=bootstrap_axis, lmergeBootstrap=True)
    # remove variables that don't have the sample axis (replace with None)
    varlist = [None if var is None or not var.hasAxis(sample_axis) else var for var in varlist]
    # determine percentiles along bootstrap axis
    if percentiles is not None or lmedian:
      lmedian = lmedian is None or lmedian # default is to plot the median if percentiles are calculated
      if lmedian:
        if percentiles is None: percentiles = (0.5,)
#           raise ArgumentError, "Median only works with percentiles."
        elif len(percentiles) == 2: 
          percentiles = (percentiles[0], 0.5, percentiles[1]) # add median to percentiles
      # compute percentiles
      assert 1 <= len(percentiles) <= 3
      qvars = [None if var is None else var.percentile(q=percentiles, axis=sample_axis) for var in varlist]
      # add median plot
      if lmedian:
        mdslc = dict(percentile=1 if len(percentiles) == 3 else 0, lidx=True)  
        meadians = [None if var is None else var(**mdslc) for var in qvars]
        if median_fmt == '' and lmean: median_fmt = '--'
        tmpplts = self.linePlot(varlist=meadians, lineformat=median_fmt, llabel=llabel, labels=labels, 
                                legend=legend, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks,
                                xlim=xlim, ylim=ylim, lrescale=lrescale, scalefactor=scalefactor, 
                                offset=offset, flipxy=flipxy, reset_color=reset_color, lsmooth=lsmooth, 
                                lprint=lprint, colors=colors, title=title,
                                plotatts=plotatts, expand_list=expand_list, lproduct=lproduct, **plotargs)
        if not lmean:
          plts = tmpplts
          colors = ['' if plt is None else plt.get_color() for plt in plts] # color argument has to be string
      # percentile band
      if len(percentiles) == 3:
        upslc = dict(percentile=2 if lmedian else 1, lidx=True)
        uppers = [None if var is None else var(**upslc) for var in qvars]
        loslc = dict(percentile=0, lidx=True) 
        lowers = [None if var is None else var(**loslc) for var in qvars]
        # plot percentiles as error bands
        facecolor = facecolor or colors
        lsmoothBand = True if lsmooth or lsmooth is None else False
        # clean up plot arguments (check against a list of "known suspects"
        band_args = {key:value for key,value in plotargs.iteritems() if key not in line_args}
        # draw band plot between upper and lower percentile
        if bandalpha is None: bandalpha = 0.3 
        tmpplts = self.bandPlot(upper=uppers, lower=lowers, lignore=lignore, llabel=False, labels=None,
                                xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, 
                                xlim=xlim, ylim=ylim, scalefactor=scalefactor, offset=offset,
                                lrescale=lrescale, legend=False, 
                                flipxy=flipxy, reset_color=False, lsmooth=lsmoothBand, lprint=False, 
                                where=where, alpha=bandalpha, edgecolor=edgecolor, colors=facecolor,
                                expand_list=expand_list, lproduct=lproduct, plotatts=plotatts, **band_args)
        if not lmean and not lmedian: plts = tmpplts
    # done! 
    return plts
  
  def bootPlot(self, varlist, varname=None, bins=None, support=None, method='pdf', percentiles=(0.25,0.75),   
               bootstrap_axis='bootstrap', lmedian=None, median_fmt='', lmean=False, mean_fmt='', 
               lvar=False, lvarBand=False, lrescale=False, scalefactor=1., offset=0.,
               legend=None, llabel=True, labels=None, hline=None, vline=None, title=None,        
               flipxy=None, xlabel=True, ylabel=False, xticks=True, yticks=False, reset_color=None, 
               xlog=False, ylog=False, xlim=None, ylim=None, lsmooth=False, lprint=False,
               lignore=False, expand_list=None, lproduct='inner', plotatts=None,
               where=None, bandalpha=None, edgecolor=None, facecolor=None, bandarg=None,  
               errorscale=None, errorevery=None, **plotargs):
    ''' A function to draw the distribution of a random variable on a given support, including confidence 
        intervals derived from percentiles along a bootstrap axes '''
    # check input and evaluate distribution variables
    varlist = checkVarlist(varlist, varname=varname, ndim=2, bins=bins, support=support, 
                                 method=method, lignore=lignore, bootstrap_axis=None) # don't remove bootstrap
    # if bootstrap_axis is a list of axes, merge them
    if isinstance(bootstrap_axis,(list,tuple)):
      varlist = [var.mergeAxes(axes=bootstrap_axis, new_axis='temporary_bootstrap_axis', asVar=True, 
                               lcheckAxis=True, lvarall=True, ldsall=False) for var in varlist]
      bootstrap_axis = 'temporary_bootstrap_axis' # avoid name collisions
    # N.B.: two-dmensional: bootstrap axis and plot axis (bootstrap axis is not required anymore)
    if not any(var.hasAxis(bootstrap_axis) for var in varlist if var is not None):
      raise AxisError, "None of the Variables has a '{:s}'-axis!".format(bootstrap_axis)
    # simple error bars using the bootstrap variance
    errorbars = None; errorband = None
    if lvar:
      errorbars = [var.std(axis=bootstrap_axis) if var.hasAxis(bootstrap_axis) else None for var in varlist]
      if lvarBand: errorband = errorbars; errorbars = None # switch
    # plot the original distribution
    slc = {bootstrap_axis:0}
    original = [None if var is None else var(**slc) for var in varlist]
    plts = self.linePlot(varlist=original, errorbar=errorbars, errorband=errorband, 
                         errorevery=errorevery, errorscale=errorscale,
                         lrescale=lrescale, scalefactor=scalefactor, offset=offset, 
                         legend=legend, llabel=llabel, labels=labels, hline=hline, vline=vline, 
                         title=title, flipxy=flipxy, xlabel=xlabel, ylabel=ylabel, xticks=xticks, 
                         yticks=yticks, reset_color=reset_color, xlog=xlog, ylog=ylog, xlim=xlim, 
                         ylim=ylim, lsmooth=lsmooth, lprint=lprint, plotatts=plotatts,
                         expand_list=expand_list, lproduct=lproduct, **plotargs)
    assert len(plts) == len(varlist)    
    # get line colors to use in all subsequent plots 
    colors = ['' if plt is None else plt.get_color() for plt in plts] # color argument has to be string
    # remove variables that don't have the sample axis (replace with None)
    varlist = [None if var is None or not var.hasAxis(bootstrap_axis) else var for var in varlist]
    if mean_fmt == '': mean_fmt = '--'
    if median_fmt == '': median_fmt = '-.' if lmean else '--' 
    # add sample moments along bootstrap axis
    self.samplePlot(varlist, percentiles=percentiles, sample_axis=bootstrap_axis, lmedian=lmedian, 
                    median_fmt=median_fmt, lmean=lmean, mean_fmt=mean_fmt, lrescale=lrescale, 
                    scalefactor=scalefactor, offset=offset, colors=colors, legend=None, llabel=False, 
                    labels=None, hline=None, vline=None, title=None, flipxy=flipxy, xlabel=False, 
                    ylabel=False, xticks=xticks, yticks=yticks, reset_color=False, xlog=xlog, ylog=ylog, 
                    xlim=xlim, ylim=ylim, lsmooth=lsmooth, lprint=False, lignore=lignore, plotatts=plotatts, 
                    expand_list=expand_list, lproduct=lproduct, bandarg=bandarg, where=where, 
                    bandalpha=bandalpha, edgecolor=edgecolor, facecolor=facecolor, **plotargs)
    # done! 
    return plts
  
  def histogram(self, varlist, varname=None, bins=None, binedgs=None, histtype='bar', lstacked=False, 
                lnormalize=True, lcumulative=0, legend=None, llabel=True, labels=None, colors=None, 
                align='mid', rwidth=None, bottom=None, weights=None, xlabel=True, ylabel=True, lignore=False,  
                xticks=True, yticks=True, hline=None, vline=None, title=None, reset_color=True, lflatten=True,
                flipxy=None, log=False, xlim=None, ylim=None, lprint=False, plotatts=None, **histargs):
    ''' A function to draw histograms of a list of 1D variables into an axes, 
        and annotate the plot based on variable properties. '''
    ## check input
    varlist = checkVarlist(varlist, varname=varname, ndim=1, bins=bins, lflatten=lflatten,
                                 support=None, method='sample', lignore=lignore)
    # initialize axes names and units
    self.flipxy = flipxy
    # N.B.: histogram has opposite convention for axes
    if not self.flipxy: varname,varunits,axname,axunits = self.xname,self.xunits,self.yname,self.yunits
    else: axname,axunits,varname,varunits = self.xname,self.xunits,self.yname,self.yunits
    ## process arguuments
    # figure out bins
    vmin = np.min([var.min() for var in varlist if var is not None])
    vmax = np.max([var.max() for var in varlist if var is not None])
    bins, binedgs = binedges(bins=bins, binedgs=binedgs, limits=(vmin,vmax), lcheckVar=True)
    # reset color cycle
    if reset_color is False: pass
    elif reset_color is True: self.set_color_cycle(None) # reset
    else: self.set_color_cycle(reset_color)
    # figure out label list
    if labels is None: labels = self._getPlotLabels(varlist)           
    elif len(labels) != len(varlist): raise ArgumentError, "Incompatible length of varlist and labels."
    assert len(labels) == len(varlist)
    # loop over variables
    for label,var in zip(labels,varlist): self.variables[label] = var # save plot variables
    # generate a list from userdefined colors
    if isinstance(colors,(tuple,list)): 
      if not all([isinstance(color,(basestring,NoneType)) for color in colors]): raise TypeError
      if len(varlist) != len(colors): raise ListError, "Failed to match linestyles to varlist!"
    elif isinstance(colors,(basestring,NoneType)): colors = [colors]*len(varlist)
    else: raise TypeError    
    ## generate list of values for histogram
    values = []; color_list = []; label_list = []; vlen = None # list of plot handles
    for label,var,color in zip(labels,varlist, colors):
      if var is not None:
        # scale variable values(axes are irrelevant)
        val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
        val = val.ravel() # flatten array
        if not varname.endswith('_bins'): varname += '_bins'
        if lprint: print varname, varunits, np.nanmean(val), np.nanstd(val)
        # get default plotargs consistent with linePlot (but only color will be used)  
        plotarg = self._getPlotArgs(label, var, llabel=llabel, plotatts=plotatts, plotarg=None)
        # extract color
        if color is None and 'color' in plotarg: color = plotarg['color']
        if color is not None: color_list.append(color)
        # add label
        label_list.append(label)
        # save values 
        vlen = len(val)
        values.append(val)
    ## construct histogram
    # figure out orientation
    if self.flipxy: orientation = 'horizontal' 
    else: orientation = 'vertical'
    # clean plot arguments (some of these cause histogram to crash)
    for arg in ('linestyle','fmt','lineformat'):
      if arg in histargs: histargs.pop(arg)
      if arg+'s' in histargs: histargs.pop(arg+'s') # also check plural forms
    # call histogram method of Axis
    if llabel: 
      if self.plot_labels is None: label_list = label_list
      else: label_list = [self.plot_labels.get(label,label) for label in label_list] 
    else: label_list = None     
    colors = color_list or None 
    hdata, bin_edges, patches = self.hist(values, bins=binedgs, color=colors, label=label_list, 
                                          normed=lnormalize, weights=weights, cumulative=lcumulative,  
                                          stacked=lstacked, bottom=bottom, histtype=histtype, log=log,
                                          align=align, orientation=orientation, rwidth=rwidth, 
                                          **histargs)
    del hdata; assert isEqual(bin_edges, binedgs)
    # N.B.: generally we don't need to keep the histogram results - there are other functions for that
    ## format axes and add annotation
    # set axes labels  
    if not self.flipxy: self.xname,self.xunits,self.yname,self.yunits = varname,varunits,axname,axunits
    else: self.xname,self.xunits,self.yname,self.yunits = axname,axunits,varname,varunits
    # apply standard formatting and annotation
    self.formatAxesAndAnnotation(title=title, legend=legend, xlabel=xlabel, ylabel=ylabel, 
                                 hline=hline, vline=vline, xlim=xlim, xlog=None, xticks=xticks, 
                                 ylim=ylim, ylog=None, yticks=yticks, 
                                 xlen=None if self.flipxy else vlen, 
                                 ylen=vlen if self.flipxy else None)
    # return handle
    return bins, patches # bins can be used as support for distributions
  
  
  def addHline(self, hline, **kwargs):
    ''' add one or more horizontal lines to the plot '''
    if 'color' not in kwargs: kwargs['color'] = 'black'
    if not isinstance(hline,(list,tuple,np.ndarray)): hline = (hline,)
    lines = []
    for hl in list(hline):
      if isinstance(hl,(int,np.integer,float,np.inexact)): 
        lines.append(self.axhline(y=hl, **kwargs))
        if self.parasite_axes: 
          self.parasite_axes.axhline(y=hl, **kwargs)
      else: raise TypeError, hl
    return lines
  
  def addVline(self, vline, **kwargs):
    ''' add one or more horizontal lines to the plot '''
    if 'color' not in kwargs: kwargs['color'] = 'black'
    if not isinstance(vline,(list,tuple,np.ndarray)): vline = (vline,)
    lines = []
    for hl in list(vline):
      if isinstance(hl,(int,np.integer,float,np.inexact)): 
        lines.append(self.axvline(x=hl, **kwargs))
      else: raise TypeError, hl.__class__
    return lines    
  
  def addTitle(self, title, title_height=None, **kwargs):
    ''' add title and adjust margins '''
    if 'fontsize' not in kwargs: kwargs['fontsize'] = self.title_size
    title_height =  self.title_height if title_height is None else title_height
    if not self.get_title(loc='center'): self.updateAxes(height=-1*title_height)
    return self.set_title(title, **kwargs)
  
  def addLegend(self, loc=0, **kwargs):
    ''' add a legend to the axes '''
#       if 'fontsize' not in kwargs and self.get_yaxis().get_label():
#         kwargs['fontsize'] = self.get_yaxis().get_label().get_fontsize()
    if 'fontsize' not in kwargs:
      if min(self.get_position().bounds[2:4]) < 0.3: kwargs['fontsize'] = 'small'
      elif min(self.get_position().bounds[2:4]) < 0.6: kwargs['fontsize'] = 'medium'
      else: kwargs['fontsize'] = 'large'      
    kwargs['loc'] = loc
    # convert handles and labels to positional arguments (this is a bug in mpl)
    args = []
    if 'handles' in kwargs: args.append(kwargs.pop('handles'))
    if 'labels' in kwargs: args.append(kwargs.pop('labels'))
    self.legend_handle = self.legend(*args, **kwargs)
  
  def _positionParasiteAxes(self):
    ''' helper routine to put parasite axes in place '''
    pax = self.parasite_axes; wd = pax.wd; pad = pax.pad # saved with parasite axes
    pos = self.get_position()
    owd = pos.width*(1.-wd); pwd = pos.width*wd*pad; nwd = pos.width*wd*(1.-pad)
    # parent axes position
    pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=owd, height=pos.height)    
    self.set_position(pos)
    # parasite axes position
    paxpos = pax.get_position()
    paxpos = paxpos.from_bounds(x0=pos.x0+owd+pwd, y0=pos.y0, width=nwd, height=pos.height)
    pax.set_position(paxpos)
  
  def addParasiteAxes(self, wd=0.075, pad=0.2, offset=0., **kwargs):
    ''' add a parasitic axes on the right margin, similar to a colorbar, and hide axes grid etc. '''
    pos = self.get_position() # position will change later
    pax = self.figure.add_axes((pos.x0,pos.y0,pos.width,pos.height), label='parasite_axes', **kwargs)
    pax.wd = wd; pax.pad = pad
    # position parasite axis
    self.parasite_axes = pax
    self._positionParasiteAxes()
    # configure axes
    pax.grid(b=False, which='both', axis='both')
    pax.set_xlim((-0.5,0.5)); pax.set_ylim(self.get_ylim())
    for ax in pax.xaxis,pax.yaxis: 
      for tick in ax.get_ticklabels(): tick.set_visible(False)
    pax.set_xticks([])
    # copy some settings
    
    # add positioning parameters
    pax.n = 0; pax.N = 0; pax.offset = offset
    # return parasite axes
    return pax
    
  def addParasiteMean(self, values, errors=None, n=0, N=1, lperi=False, style='errorbar', **kwargs):
    ''' add a maker at the mean of the given values to the parasitic axes '''
    pax = self.parasite_axes
    # adjust offset (each new batch)
    if n == 0 or pax.N == 0: # this will work most of the time
      if pax.n > 0: pax.offset += 0.5/(pax.N+1.)
      pax.N = N # new N for new batch
    # generate mock line plot values
    xax = np.arange(-1,2, dtype=np.float) # x-axis
    # allow for slight offset, to disentangle
    if n != 0 or N != 1: 
      shift = float(n+1.)/float(N+1.) - 0.5
      xax[1] += pax.offset + shift
    # average values (three points, to avoid cut-off)
    if lperi: 
      vals = values[1:-1].copy() # cut off boundaries
      if errors is not None: errs = errors[1:-1].copy() # cut off boundaries
    else:
      vals = values.copy()
      if errors is not None: errs = errors.copy()
    # average means
    val = vals.mean().repeat(3)
    # average variances (not std directly)
    if errors is not None:
      if lperi: err = ((errs**2).mean(keepdims=True)**0.5)
      else: err = (((errs**2).mean(keepdims=True) + vals.var(keepdims=True))**0.5)
      err = err.repeat(3)
      # N.B.: std = sqrt( mean of variances + variance of means )
    else: err = None
    # remove some style parameters that don't apply
    kwargs.pop('errorevery', None)
    # draw mean
    if style.lower() == 'errorbar':
      pnt = pax.errorbar(xax,val, xerr=None, yerr=err, **kwargs)
    elif style.lower() == 'myerrorbar':
      # increase line thickness if no marker
      if 'marker' not in kwargs:
        kwargs['linewidth'] = kwargs.get('linewidth',mpl.rcParams['lines.linewidth']) * 1.5      
      pnt = pax.errorbar(xax,val, xerr=None, yerr=err, **kwargs)
    else: raise NotImplementedError
    # increase counter
    pax.n = n+1
    return pnt
    
  def _translateArguments(self, labels=None, expand_list=None, plotargs=None):
    # loop over special arguments that allow plural form
    for name in ('lineformats','linestyles','colors','markers'):
      if name in plotargs:
        args = plotargs.pop(name)  
        if isinstance(args,(tuple,list)): 
          if not all([isinstance(arg,basestring) for arg in args]): raise TypeError
          # adjust length
          if len(labels) > len(args): args += args[-1]*(len(labels)-len(args)) # extend last item
          elif len(labels) < len(args): args = args[:len(labels)] # cut off rest
        elif isinstance(args,basestring):
          args = [args]*len(labels)
        elif args is not None: raise TypeError, args
        if args is not None:
          plotargs[name[:-1]] = args # save list under singular name
          expand_list.append(name[:-1]) # cut off trailing 's' (i.e. proper singular form)
    return expand_list, plotargs
  
  def _expandArgumentList(self, labels=None, expand_list=None, lproduct='inner', plotargs=None):
    ''' function to expand arguments while applying some default treatments; plural forms of some
        plot arguments are automatically converted and expanded for all plotargs '''
    # line style parameters is just a list of line styles for each plot
    if expand_list is None: expand_list = []
    else: expand_list = list(expand_list)
    if lproduct == 'inner':
      expand_list.append('label')
      expand_list, plotargs = self._translateArguments(labels=labels, expand_list=expand_list, plotargs=plotargs)
      # actually expand list 
      plotargs = expandArgumentList(label=labels, expand_list=expand_list, lproduct=lproduct, **plotargs)
    else: raise NotImplementedError, lproduct
    # return cleaned-up and expanded plot arguments
    return plotargs
    
  def _getPlotValues(self, var, checkunits=None, lsmooth=False, lperi=False, lshift=False, 
                     laxis=False, lrescale=False, scalefactor=1., offset=0.):
    ''' retrieve plot values and apply optional scaling and offset (user-defined) '''
    if lrescale: checkunits = None
    val, varunits, varname = getPlotValues(var, checkunits=checkunits, checkname=None, lsmooth=lsmooth, 
                                           lperi=lperi, laxis=laxis)
    if lrescale:
      if self.flipxy: vlim,varunits = self.get_xlim(),self.xunits
      else: vlim,varunits = self.get_ylim(),self.yunits
      if offset != 0: val -= offset
      if scalefactor != 1: val /= scalefactor 
      val *= ( vlim[1] - vlim[0] )
      if lshift: val += vlim[0]  
    return val, varunits, varname
  
  def _getPlotLabels(self, varlist):
    ''' figure out reasonable plot labels based variable and dataset names '''
    # make list without None's for checking uniqueness
    nonone_list = [var for var in varlist if var is not None]
    # figure out line/plot label policy
    if not any(var.name == nonone_list[0].name for var in nonone_list[1:]):
      # if variable names are different
      labels = [None if var is None else var.name for var in varlist]
    elif ( all(var.dataset_name is not None for var in nonone_list) and
           not any(var.dataset_name == nonone_list[0].dataset_name for var in nonone_list[1:]) ):
      # if dataset names are different
      labels = [None if var is None else var.dataset_name for var in varlist]
    else: 
      # if no names are unique, just number
      labels = range(len(varlist))
    return labels
  
  def _getPlotArgs(self, label, var, llabel=False, plotatts=None, plotarg=None, plot_labels=None):
    ''' function to return plotting arguments/styles based on defaults and explicit arguments '''
    if not isinstance(label, (basestring,int,np.integer)): raise TypeError, label
    if not isinstance(var, Variable): raise TypeError
    if plotatts is not None and not isinstance(plotatts, dict): raise TypeError
    if plotarg is not None and not isinstance(plotarg, dict): raise TypeError
    args = dict()
    # apply figure/project defaults
    if label == var.name: # variable name has precedence
      if var.dataset_name is not None and self.dataset_plotargs is not None: 
        args.update(self.dataset_plotargs.get(var.dataset_name,{}))
      if self.variable_plotargs is not None: args.update(self.variable_plotargs.get(var.name,{}))
    else: # dataset name has precedence
      if self.variable_plotargs is not None: args.update(self.variable_plotargs.get(var.name,{}))
      if var.dataset_name is not None and self.dataset_plotargs is not None: 
        args.update(self.dataset_plotargs.get(var.dataset_name,{}))
    # apply axes/local defaults
    if plotatts is not None: args.update(plotatts.get(label,{}))
    if plotarg is not None: args.update(plotarg)
    # relabel (simple name mapping)
    if plot_labels is None: plot_labels = self.plot_labels
    if plot_labels and label in plot_labels:
      if llabel: args['label'] = plot_labels[label] # only, if we actually want labels!
      label = plot_labels[label]
    # return dictionary with keyword argument for plotting function
    return args

  def formatAxesAndAnnotation(self, title=None, legend=None, xlabel=None, ylabel=None, 
                              hline=None, vline=None, xlim=None, ylim=None, xlog=None, ylog=None,                                
                              xticks=None, xlen=None, yticks=None, ylen=None):
    ''' apply standard formatting and labeling to axes '''
    ## format axes
    # set plot scale (log/linear)
    if xlog is not None: self.set_xscale('log' if xlog else 'linear')
    if ylog is not None: self.set_yscale('log' if ylog else 'linear')
    # set axes limits
    if isinstance(xlim,(list,tuple)) and len(xlim)==2: self.set_xlim(*xlim)
    elif xlim is not None: raise TypeError
    if isinstance(ylim,(list,tuple)) and len(ylim)==2: self.set_ylim(*ylim)
    elif ylim is not None: raise TypeError 
    if self.parasite_axes: # mirror y-limits on parasite axes
      self.parasite_axes.set_ylim(self.get_ylim())
    # set title
    if title is not None: self.addTitle(title)
    # format axes ticks
    self.xTickLabels(xticks, n=xlen, loverlap=False) # False means overlaps will be prevented
    self.yTickLabels(yticks, n=ylen, loverlap=False)
    ## add axes labels and annotation
    # format axes labels
    self.xLabel(xlabel)
    self.yLabel(ylabel)    
    # N.B.: a typical custom label that makes use of the units would look like this: 'custom label [{1:s}]', 
    # where {} will be replaced by the appropriate default units (which have to be the same anyway)
    # add legend    
    if legend is False or legend is None: pass # no legend
    elif legend is True: self.addLegend(loc=0) # legend at default/optimal location
    # N.B.: apparently True and False test positive as integers...
    elif isinstance(legend,(int,np.integer,float,np.inexact)): self.addLegend(loc=legend)
    elif isinstance(legend,dict): self.addLegend(**legend) 
    # add orientation lines
    if hline is not None: self.addHline(hline)
    if vline is not None: self.addVline(vline)
  
  def xLabel(self, xlabel, name=None, units=None):
    ''' format x-axis label '''
    name = self.xname if name is None else name
    units = self.xunits if units is None else units
    # figure out label 
    if isinstance(xlabel,basestring):
      xlabel = xlabel.format(name,units)
    elif xlabel is not None and xlabel is not False:
      # only apply label, if ticks are also present
      xticks = self.xaxis.get_ticklabels()
      # len(xticks) > 0 is necessary to avoid errors with AxesGrid, which removes invisible tick labels      
      if len(xticks) > 0 and xticks[-1].get_visible(): xlabel = self._axLabel(xlabel, name, units)
    if isinstance(xlabel,basestring):
      # N.B.: labelpad is ignored by AxesGrid
      self.set_xlabel(xlabel, labelpad=self.xpad)
      # label position
      self.xaxis.set_label_position('top' if self.xtop else 'bottom')
    return xlabel    
  def yLabel(self, ylabel, name=None, units=None):
    ''' format y-axis label '''
    # apply Y-label to appropriate axes
    ax = self.parasite_axes if self.parasite_axes and self.yright else self
    name = self.yname if name is None else name
    units = self.yunits if units is None else units
    # figure out label 
    if isinstance(ylabel,basestring):
      ylabel = ylabel.format(name,units)
    elif ylabel is not None and ylabel is not False:
      # only apply label, if ticks are also present
      yticks = ax.yaxis.get_ticklabels()
      if len(yticks) > 0 and yticks[-1].get_visible(): ylabel = self._axLabel(ylabel, name, units)
    if isinstance(ylabel,basestring):
      # set Y-label
      ax.set_ylabel(ylabel, labelpad=self.ypad) # labelpad is ignored by AxesGrid
      # label position
      ax.yaxis.set_label_position('right' if self.yright else 'left')
    return ylabel    
  def _axLabel(self, label, name, units):
    ''' helper method to format axes lables '''
    if label is True: 
      if not name and not units: label = ''
      elif not units: label = '{0:s}'.format(name)
      elif not name: label = '[{:s}]'.format(units)
      else: label = '{0:s} [{1:s}]'.format(name,units)
    elif label is False or label is None: label = ''
    elif isinstance(label,basestring): label = label.format(name,units)
    else: raise ValueError, label
    return label
    
  def xTickLabels(self, xticks, n=None, loverlap=False):
    ''' format x-tick labels '''
    xaxis = self.xaxis
    xticks = self._tickLabels(xticks, xaxis)
    yticks = self.yaxis.get_ticklabels()
    # tick label position
    if self.xtop: 
      self.xaxis.set_tick_params(labeltop=True, labelbottom=False)
    # minor ticks
    if n is not None: self._minorTicks(xticks, n, xaxis)
    # tick label visibility
    if not loverlap and len(xticks) > 0 and (
        len(yticks) == 0 or not yticks[-1].get_visible() ):
        xticks[0].set_visible(False)
    return xticks
  def yTickLabels(self, yticks, n=None, loverlap=False):
    ''' format y-tick labels '''
    yaxis = self.parasite_axes.yaxis if self.parasite_axes and self.yright else self.yaxis 
    xticks = self.xaxis.get_ticklabels()
    yticks = self._tickLabels(yticks, yaxis)
    # tick label visibility
    if not loverlap and len(yticks) > 0 and (
        len(xticks) == 0 or not xticks[-1].get_visible() ):
        yticks[0].set_visible(False)
    # tick label position
    if self.yright:
      self.yaxis.set_tick_params(labelleft=False) # always need to switch off on master yaxis
      yaxis.set_tick_params(labelright=True, labelleft=False) # switch on on active yaxis
    # minor ticks (apply to major and parasite axes)
    for ax in self,self.parasite_axes:      
      if ax and n is not None: self._minorTicks(yticks, n, ax.yaxis)
    return yticks
  def _minorTicks(self, ticks, n, axis):
    ''' helper method to format axes ticks '''
    nmaj = len(ticks)
    if nmaj > 0:
      nmin = min(30//nmaj,n//nmaj+1) # number of sub-divisions == number of ticks +1
      axis.set_minor_locator(mpl.ticker.AutoMinorLocator(nmin))
  def _tickLabels(self, ticks, axis):
    ''' helper method to format axes ticks '''
    if ticks is True: 
      ticklist = axis.get_ticklabels()
    elif ticks is False: 
      ticklist = axis.get_ticklabels()
      for tick in ticklist: tick.set_visible(False)
      ticklist = []
    elif isinstance(ticks,list,tuple): 
      axis.set_ticklabels(ticks)
      ticklist = axis.get_ticklabels()
    else: raise ValueError, ticks
    return ticklist
      
  # add subplot/axes label (alphabetical indexing, byt default)
  def addLabel(self, label, loc=1, lstroke=False, lalphabet=True, size='large', font='monospace', prop=None, **kwargs):
    from string import lowercase # lowercase letters
    from matplotlib.offsetbox import AnchoredText 
    from matplotlib.patheffects import withStroke    
    # settings
    if prop is None: prop = dict(size=size, fontname=font)
    args = dict(pad=0., borderpad=1.5, frameon=False)
    args.update(kwargs)
    # create label    
    if lalphabet and isinstance(label,int):
      label = '('+lowercase[label]+')'    
    at = AnchoredText(label, loc=loc, prop=prop, **args)
    self.add_artist(at) # add to axes
    if lstroke: 
      at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
      
  def updateAxes(self, x0=0., y0=0., width=0., height=0., mode='shift'):
    ''' shift position of axes or reapply shift after subplot adjustment '''
    # format input
    if mode == 'shift':
      shift = np.asarray((x0, y0, width, height), dtype=np.float32)
      # update axes margins
      pos = self.get_position()
      oldpos = np.asarray((pos.x0, pos.y0, pos.width, pos.height), dtype=np.float32)
      newpos = oldpos + shift # shift the position
      self.set_position(pos.from_bounds(x0=newpos[0], y0=newpos[1], width=newpos[2], height=newpos[3]))
      # save shift
      self.axes_shift += shift
    elif mode == 'adjust':
      assert x0 == 0. and y0 == 0. and width == 0. and height == 0.
      # update axes margins
      pos = self.get_position()
      oldpos = np.asarray((pos.x0, pos.y0, pos.width, pos.height), dtype=np.float32)
      newpos = oldpos + self.axes_shift # apply recorded shift
      self.set_position(pos.from_bounds(x0=newpos[0], y0=newpos[1], width=newpos[2], height=newpos[3]))
    else: raise NotImplementedError
    # readjust parasite axes
    if self.parasite_axes: self._positionParasiteAxes() 

    
# a new class that combines the new axes with LocatableAxes for use with AxesGrid 
class MyLocatableAxes(LocatableAxes,MyAxes):
  ''' A new Axes class that adds functionality from MyAxes to a LocatableAxes for use in AxesGrid '''


if __name__ == '__main__':
    pass