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
# internal imports
from geodata.base import Variable, Dataset, Ensemble
from geodata.misc import ListError, AxisError, ArgumentError, isEqual
from plotting.misc import smooth, getPlotValues, errorPercentile
from collections import OrderedDict
from utils.misc import binedges, expandArgumentList, evalDistVars
from types import NoneType

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
  plots              = None
  title_height       = 0.0 # fraction of axes height
  flipxy             = False
  xname              = None
  xunits             = None
  xpad               = 2    # pixels
  yname              = None
  yunits             = None
  ypad               = 0
  figure             = None
  pax                = None
  
  def __init__(self, *args, **kwargs):
    ''' constructor to initialize some variables / counters '''  
    # call parent constructor
    super(MyAxes,self).__init__(*args, **kwargs)
    self.variables = OrderedDict() # save variables by label
    self.plots = OrderedDict() # save plot objects by label
    
    
  def linePlot(self, varlist, varname=None, bins=None, support=None, errorbar=None, errorband=None,  
               legend=None, llabel=True, labels=None, hline=None, vline=None, title=None, lignore=False,        
               flipxy=None, xlabel=True, ylabel=True, xticks=True, yticks=True, reset_color=None, 
               lparasiteMeans=False, parasite_axes=None, 
               xlog=False, ylog=False, xlim=None, ylim=None, lsmooth=False, lperi=False, lprint=False,
               expand_list=None, lproduct='inner', method='pdf', plotatts=None, **plotargs):
    ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on 
        variable properties; extra keyword arguments (plotargs) are passed through expandArgumentList,
        before being passed to Axes.plot(). '''
    ## figure out variables
    varlist = self._checkVarlist(varlist, varname=varname, ndim=1, bins=bins, support=support, method=method, lignore=lignore)
    if errorbar: errlist = self._checkVarlist(errorbar, varname=varname, ndim=1, 
                                              bins=bins, support=support, method=method, lignore=lignore)
    else: errlist = [None]*len(varlist) # no error bars
    if errorband: bndlist = self._checkVarlist(errorband, varname=varname, ndim=1, 
                                               bins=bins, support=support, method=method, lignore=lignore)
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
    elif len(labels) != len(varlist): raise ArgumentError, "Incompatible length of varlist and labels."
    label_list = labels if llabel else [None]*len(labels) # used for plot labels later
    assert len(labels) == len(varlist)
    # finally, expand keyword arguments
    plotargs = self._expandArgumentList(labels=label_list, expand_list=expand_list, 
                                        lproduct=lproduct, plotargs=plotargs)
    assert len(plotargs) == len(varlist)
    # initialize parasitic axis for means
    if lparasiteMeans and self.pax is None:
      if parasite_axes is None: parasite_axes = dict() 
      self.pax = self.addParasiticAxes(**parasite_axes)  
    ## generate individual line plots
    plts = [] # list of plot handles
    for label,var in zip(labels,varlist): self.variables[label] = var # save plot variables
    # loop over variables and plot arguments
    N = len(varlist)
    for n,var,errvar,bndvar,plotarg,label in zip(xrange(N),varlist,errlist,bndlist,plotargs,labels):
      if var is not None:
        varax = var.axes[0]
        # scale axis and variable values 
        axe, axunits, axname = getPlotValues(varax, checkunits=axunits, checkname=None, laxis=True, lperi=lperi)
        val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None, lsmooth=lsmooth, lperi=lperi)
        if errvar is not None: # for error bars
          err, varunits, errname = getPlotValues(errvar, checkunits=varunits, checkname=None, lsmooth=lsmooth, lperi=lperi); del errname
        else: err = None
        if bndvar is not None: # semi-transparent error bands
          bnd, varunits, bndname = getPlotValues(bndvar, checkunits=varunits, checkname=None, lsmooth=lsmooth, lperi=lperi); del bndname
        else: bnd = None      
        # variable and axis scaling is not always independent...
        if var.plot is not None and varax.plot is not None: 
          if varax.units != axunits and var.plot.preserve == 'area':
            val /= varax.plot.scalefactor
        # N.B.: other scaling behavior could be added here
        if lprint: print varname, varunits, np.nanmean(val), np.nanstd(val)   
        # update plotargs from defaults
        plotarg = self._getPlotArgs(label=label, var=var, plotatts=plotatts, plotarg=plotarg)
        plotarg['fmt'] = plotarg.pop('lineformat','') # rename (I prefer a different name)
        # N.B.: '' (empty string) is the default, None means no line is plotted, only errors!
        # extract arguments for error band
        bndarg    = plotarg.pop('bandarg',dict())
        where     = plotarg.pop('where',None)
        bandalpha = plotarg.pop('bandalpha',0.5)
        edgecolor = plotarg.pop('edgecolor',0.5)
        facecolor = plotarg.pop('facecolor',None)
        errorscale = plotarg.pop('errorscale',None)
        if errorscale is not None: errorscale = errorPercentile(errorscale)
        # figure out orientation and call plot function
        if self.flipxy: # flipped axes
          xlen = len(var); ylen = len(axe) # used later
          plt = self.errorbar(val, axe, xerr=err, yerr=None, **plotarg)[0]
          if lparasiteMeans: raise NotImplementedError
        else:# default orientation
          xlen = len(axe); ylen = len(val) # used later
          plt = self.errorbar(axe, val, xerr=None, yerr=err, **plotarg)[0]
          if lparasiteMeans: self.addParasiteMean(val, errors=err if err is not None else bnd, n=n, N=N, **plotarg)
        # figure out parameters for error bands
        if bnd is not None: 
          self._drawBand(axe, val+bnd*errorscale, val-bnd*errorscale, where=where, color=(facecolor or plt.get_color()), 
                         alpha=bandalpha*plotarg.get('alpha',1.), edgecolor=edgecolor, **bndarg)  
        plts.append(plt); self.plots[label] = plt
    ## format axes and add annotation
    # set axes labels  
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
               flipxy=None, xlabel=True, ylabel=True, xticks=True, yticks=True, reset_color=None, 
               xlog=None, ylog=None, xlim=None, ylim=None, lsmooth=False, lperi=False, lprint=False, 
               expand_list=None, lproduct='inner', method='pdf', plotatts=None, **plotargs):
    ''' A function to draw a colored bands between two lists of 1D variables representing the upper
        and lower limits of the bands; extra keyword arguments (plotargs) are passed through 
        expandArgumentList, before being passed on to Axes.fill_between() (used to draw bands). '''
    ## figure out variables
    upper = self._checkVarlist(upper, varname=varname, ndim=1, bins=bins, 
                               support=support, method=method, lignore=lignore)
    lower = self._checkVarlist(lower, varname=varname, ndim=1, bins=bins, 
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
    for label,upvar,lowvar in zip(labels,upper,lower): 
      self.variables[label+'_bnd'] = (upvar,lowvar) # save band variables under special name
    # loop over variables and plot arguments
    for upvar,lowvar,plotarg,label in zip(upper,lower,plotargs,labels):
      if upvar or lowvar:
        if upvar:
          varax = upvar.axes[0]
          assert lowvar is None or ( lowvar.hasAxis(varax) and lowvar.ndim == 1 )
        if lowvar:
          varax = lowvar.axes[0]
          assert upvar is None or ( upvar.hasAxis(varax) and upvar.ndim == 1 )          
        # scale axis and variable values 
        axe, axunits, axname = getPlotValues(varax, checkunits=axunits, checkname=None, laxis=True, lperi=lperi)
        if upvar: up, varunits, varname = getPlotValues(upvar, checkunits=varunits, checkname=None, lsmooth=lsmooth, lperi=lperi) 
        else: up = np.zeros_like(axe)
        if lowvar: low, varunits, varname = getPlotValues(lowvar, checkunits=varunits, checkname=None, lsmooth=lsmooth, lperi=lperi)
        else: low = np.zeros_like(axe)
        # variable and axis scaling is not always independent...
        if upvar.plot is not None and varax.plot is not None: 
          if varax.units != axunits and upvar.plot.preserve == 'area':
            up /= varax.plot.scalefactor; low /= varax.plot.scalefactor
        # N.B.: other scaling behavior could be added here
        if lprint: print varname, varunits, np.nanmean(up), np.nanmean(low)           
        if lsmooth: up = smooth(up); low = smooth(low)
        # update plotargs from defaults
        plotarg = self._getPlotArgs(label=label, var=upvar, plotatts=plotatts, plotarg=plotarg)
        ## draw actual bands 
        bnd = self._drawBand(axe, low, up, **plotarg)
        # book keeping
        if self.flipxy: xlen, ylen = len(low), len(axe) 
        else: xlen, ylen = len(axe), len(low)
        bnds.append(bnd); self.plots[label] = bnd
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
    elif isinstance(edgecolor,(int,np.int)): edgecolor = float(edgecolor)
    if isinstance(edgecolor,(float,np.float)): 
      edgecolor = tuple(c*edgecolor for c in color) # slightly darker edges
    # construct keyword arguments to fill_between  
    bndarg['edgecolor'] = edgecolor
    bndarg['facecolor'] = color
    bndarg['where'] = where
    bndarg['alpha'] = alpha
    if self.flipxy: self.fill_betweenx(y=axes, x1=lower, x2=upper, **bndarg)
    else: self.fill_between(x=axes, y1=lower, y2=upper, interpolate=True, **bndarg) # interpolate=True
  
  
  def histogram(self, varlist, varname=None, bins=None, binedgs=None, histtype='bar', lstacked=False, 
                lnormalize=True, lcumulative=0, legend=None, llabel=True, labels=None, colors=None, 
                align='mid', rwidth=None, bottom=None, weights=None, xlabel=True, ylabel=True, lignore=False,  
                xticks=True, yticks=True, hline=None, vline=None, title=None, reset_color=True, 
                flipxy=None, log=False, xlim=None, ylim=None, lprint=False, plotatts=None, **histargs):
    ''' A function to draw histograms of a list of 1D variables into an axes, 
        and annotate the plot based on variable properties. '''
    ## check input
    varlist = self._checkVarlist(varlist, varname=varname, ndim=1, bins=bins, 
                                 support=None, method='sample', lignore=lignore)
    # initialize axes names and units
    self.flipxy = flipxy
    # N.B.: histogram has opposite convention for axes
    if not self.flipxy: varname,varunits,axname,axunits = self.xname,self.xunits,self.yname,self.yunits
    else: axname,axunits,varname,varunits = self.xname,self.xunits,self.yname,self.yunits
    ## process arguuments
    # figure out bins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
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
    values = []; color_list = []; label_list = [] # list of plot handles
    for label,var,color in zip(labels,varlist, colors):
      if var is not None:
        # scale variable values(axes are irrelevant)
        val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
        val = val.ravel() # flatten array
        if not varname.endswith('_bins'): varname += '_bins'
        if lprint: print varname, varunits, np.nanmean(val), np.nanstd(val)
        # get default plotargs consistent with linePlot      
        plotarg = self._getPlotArgs(label, var, plotatts=plotatts, plotarg=None)
        # extract color
        if color is None and color in plotarg: color = plotarg['color']        
        if color is not None: color_list.append(color)
        # add label
        label_list.append(label)
        # save values 
        values.append(val)
    ## construct histogram
    # figure out orientation
    if self.flipxy: orientation = 'horizontal' 
    else: orientation = 'vertical'
    # call histogram method of Axis
    label_list = label_list if llabel else None
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
                                 xlen=None if self.flipxy else len(val), 
                                 ylen=len(val) if self.flipxy else None)
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
    if 'fontsize' not in kwargs: kwargs['fontsize'] = 'medium'
    title_height =  self.title_height if title_height is None else title_height
    pos = self.get_position()
    pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-title_height)    
    self.set_position(pos)
    return self.set_title(title, kwargs)
  
  def addLegend(self, loc=0, **kwargs):
      ''' add a legend to the axes '''
      if 'fontsize' not in kwargs and self.get_yaxis().get_label():
        kwargs['fontsize'] = self.get_yaxis().get_label().get_fontsize()
      kwargs['loc'] = loc
      self.legend(**kwargs)
  
  def _checkVarlist(self, varlist, varname=None, ndim=1, bins=None, support=None, method='pdf', lignore=None):
    ''' helper function to pre-process the variable list '''
    # varlist is the list of variable objects that are to be plotted
    if isinstance(varlist,Variable): varlist = [varlist]
    elif isinstance(varlist,Dataset): 
      if isinstance(varname,basestring): varlist = [varlist[varname]]
      elif isinstance(varname,(tuple,list)):
        varlist = [varlist[name] if name in varlist else None for name in varname]
      else: raise TypeError
    elif isinstance(varlist,(tuple,list,Ensemble)):
      if varname is not None:
        tmplist = []
        for var in varlist:
          if isinstance(var,Dataset): varlist.append(var)
          elif hasattr(var, varname): varlist.append(getattr(var,varname))
          else: varlist.append(None)
        varlist = tmplist; del tmplist
    else: raise TypeError
    if not all([isinstance(var,(Variable, NoneType)) for var in varlist]): raise TypeError
    for var in varlist: 
      if var is not None: var.squeeze() # remove singleton dimensions
    # evaluate distribution variables on support/bins
    if bins is not None or support is not None:
      varlist = evalDistVars(varlist, bins=bins, support=support, method=method, ldatasetLink=True) 
    # check axis: they need to have only one axes, which has to be the same for all!
    for var in varlist: 
      if var is None: pass
      elif var.ndim > ndim: 
        raise AxisError, "Variable '{:s}' has more than {:d} dimension(s); consider squeezing.".format(var.name,ndim)
      elif var.ndim < ndim: 
        raise AxisError, "Variable '{:s}' has less than {:d} dimension(s); consider display as a line.".format(var.name,ndim)
    # return cleaned-up and checkd variable list
    return varlist    
  
  def addParasiticAxes(self, wd=0.075, pad=0.2, offset=0., **kwargs):
    ''' add a parasitic axes on the right margin, similar to a colorbar, and hide axes grid etc. '''
    # position parasite axis
    pos = self.get_position()
    owd = pos.width*(1.-wd); bwd = pos.width*wd*pad; nwd = pos.width*wd*(1.-pad)
    pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=owd, height=pos.height)    
    self.set_position(pos)
    pax = self.figure.add_axes((pos.x0+owd+bwd, pos.y0, nwd, pos.height), **kwargs)
    # configure axes
    pax.grid(b=False, which='both', axis='both')
    pax.set_xlim((-0.5,0.5)); pax.set_ylim(self.get_ylim())
    for ax in pax.xaxis,pax.yaxis: 
      for tick in ax.get_ticklabels(): tick.set_visible(False)
    pax.set_xticks([])
    # add positioning parameters
    pax.n = 0; pax.N = 0; pax.offset = offset
    # return parasite axes
    return pax
    
  def addParasiteMean(self, values, errors=None, n=0, N=1, style='errorbar', **kwargs):
    ''' add a maker at the mean of the given values to the parasitic axes '''
    pax = self.pax
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
    val = values.mean().repeat(3)
    # average variances (not std directly)
    if errors is not None:
      err = ((errors**2).mean(keepdims=True)**0.5).repeat(3)
    else: err = None
    # remove some style parameters that don't apply
    kwargs.pop('errorevery', None)
    # draw mean
    if style.lower() == 'errorbar':
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
          if len(labels) != len(args): raise ListError, "Failed to match linestyles to varlist!"
        elif isinstance(args,basestring):
          args = [args]*len(labels)
        else: raise TypeError
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
      expand_list, plotargs = self._translateArguments(labels=labels, 
                                                       expand_list=expand_list, plotargs=plotargs)
      # actually expand list 
      plotargs = expandArgumentList(label=labels, expand_list=expand_list, lproduct=lproduct, **plotargs)
    else: raise NotImplementedError, lproduct
    # return cleaned-up and expanded plot arguments
    return plotargs
    
  def _getPlotLabels(self, varlist):
    ''' figure out reasonable plot labels based variable and dataset names '''
    # make list without None's for checking uniqueness
    nonone_list = [var for var in varlist if var is not None]
    # figure out line/plot label policy
    if not any(var.name == nonone_list[0].name for var in nonone_list[1:]):
      # if variable names are different
      labels = [None if var is None else var.name for var in varlist]
    elif ( all(var.dataset is not None for var in nonone_list) and
           not any(var.dataset.name == nonone_list[0].dataset.name for var in nonone_list[1:]) ):
      # if dataset names are different
      labels = [None if var is None else var.dataset.name for var in nonone_list]
    else: 
      # if no names are unique, just number
      labels = range(len(varlist))
    return labels
  
  def _getPlotArgs(self, label, var, plotatts=None, plotarg=None):
    ''' function to return plotting arguments/styles based on defaults and explicit arguments '''
    if not isinstance(label, basestring): raise TypeError
    if not isinstance(var, Variable): raise TypeError
    if plotatts is not None and not isinstance(plotatts, dict): raise TypeError
    if plotarg is not None and not isinstance(plotarg, dict): raise TypeError
    args = dict()
    # apply figure/project defaults
    if label == var.name: # variable name has precedence
      if var.dataset is not None and self.dataset_plotargs is not None: 
        args.update(self.dataset_plotargs.get(var.dataset.name,{}))
      if self.variable_plotargs is not None: args.update(self.variable_plotargs.get(var.name,{}))
    else: # dataset name has precedence
      if self.variable_plotargs is not None: args.update(self.variable_plotargs.get(var.name,{}))
      if var.dataset is not None and self.dataset_plotargs is not None: 
        args.update(self.dataset_plotargs.get(var.dataset.name,{}))
    # apply axes/local defaults
    if plotatts is not None: args.update(plotatts.get(label,{}))
    if plotarg is not None: args.update(plotarg)
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
    if isinstance(ylim,(list,tuple)) and len(ylim)==2: 
      self.set_ylim(*ylim)
      if self.pax is not None: self.pax.set_ylim(*ylim)
    elif ylim is not None: raise TypeError 
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
    if isinstance(legend,dict): self.addLegend(**legend) 
    elif isinstance(legend,(int,np.integer,float,np.inexact)): self.addLegend(loc=legend)
    # add orientation lines
    if hline is not None: self.addHline(hline)
    if vline is not None: self.addVline(vline)
  
  def xLabel(self, xlabel, name=None, units=None):
    ''' format x-axis label '''
    if xlabel is not None:
      xticks = self.get_xaxis().get_ticklabels()
      # len(xticks) > 0 is necessary to avoid errors with AxesGrid, which removes invisible tick labels      
      if len(xticks) > 0 and xticks[-1].get_visible(): 
        name = self.xname if name is None else name
        units = self.xunits if units is None else units
        xlabel = self._axLabel(xlabel, name, units)
        # N.B.: labelpad is ignored by AxesGrid
        self.set_xlabel(xlabel, labelpad=self.xpad)
    return xlabel    
  def yLabel(self, ylabel, name=None, units=None):
    ''' format y-axis label '''
    if ylabel is not None:
      yticks = self.get_yaxis().get_ticklabels()
      if len(yticks) > 0 and yticks[-1].get_visible(): 
        name = self.yname if name is None else name
        units = self.yunits if units is None else units
        ylabel = self._axLabel(ylabel, name, units)
        # N.B.: labelpad is ignored by AxesGrid
        self.set_ylabel(ylabel, labelpad=self.ypad)
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
    xticks = self._tickLabels(xticks, self.get_xaxis())
    yticks = self.get_yaxis().get_ticklabels()
    if not loverlap and len(xticks) > 0 and (
        len(yticks) == 0 or not yticks[-1].get_visible() ):
        xticks[0].set_visible(False)
    if n is not None: self._minorTickLabels(xticks, n, self.xaxis)
    return xticks
  def yTickLabels(self, yticks, n=None, loverlap=False):
    ''' format y-tick labels '''
    xticks = self.get_xaxis().get_ticklabels()
    yticks = self._tickLabels(yticks, self.get_yaxis())
    if not loverlap and len(yticks) > 0 and (
        len(xticks) == 0 or not xticks[-1].get_visible() ):
        yticks[0].set_visible(False)
    if n is not None: self._minorTickLabels(yticks, n, self.yaxis)      
    return yticks
  def _minorTickLabels(self, ticks, n, axis):
    ''' helper method to format axes ticks '''
    nmaj = len(ticks)
    if n%nmaj == 0: 
      nmin = n//nmaj
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
  def addLabel(self, label, loc=1, lstroke=False, lalphabet=True, size=None, prop=None, **kwargs):
    from string import lowercase # lowercase letters
    from matplotlib.offsetbox import AnchoredText 
    from matplotlib.patheffects import withStroke    
    # settings
    if prop is None: prop = dict()
    if not size: prop['size'] = 'large'
    args = dict(pad=0., borderpad=1.5, frameon=False)
    args.update(kwargs)
    # create label    
    if lalphabet and isinstance(label,int):
      label = '('+lowercase[label]+')'    
    at = AnchoredText(label, loc=loc, prop=prop, **args)
    self.add_artist(at) # add to axes
    if lstroke: 
      at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
        


# a new class that combines the new axes with LocatableAxes for use with AxesGrid 
class MyLocatableAxes(LocatableAxes,MyAxes):
  ''' A new Axes class that adds functionality from MyAxes to a LocatableAxes for use in AxesGrid '''


if __name__ == '__main__':
    pass