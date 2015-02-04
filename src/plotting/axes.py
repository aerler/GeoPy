'''
Created on Dec 11, 2014

A custom Axes class that provides some specialized plotting functions and retains variable information.

@author: Andre R. Erler, GPL v3
'''

# external imports
from types import NoneType
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid.axes_divider import LocatableAxes
# internal imports
from geodata.base import Variable, Dataset, Ensemble
from geodata.stats import DistVar, VarKDE, VarRV  
from geodata.misc import ListError, AxisError, ArgumentError, isEqual
from plotting.misc import smooth, getPlotValues
from collections import OrderedDict
from utils.misc import binedges

## new axes class
class MyAxes(Axes): 
  ''' 
    A custom Axes class that provides some specialized plotting functions and retains variable 
    information. The custom Figure uses this Axes class by default.
  '''
  variables    = None
  plots        = None
  title_height = 2
  flipxy       = False
  xname        = None
  xunits       = None
  xpad         = 2 
  yname        = None
  yunits       = None
  ypad         = 0
  
  def linePlot(self, varlist, varname=None, bins=None, support=None, linestyles=None, varatts=None, legend=None, llabel=True, labels=None, 
               xticks=True, yticks=True, hline=None, vline=None, title=None, reset_color=None, flipxy=None, xlabel=True, ylabel=True,
               xlog=False, ylog=False, xlim=None, ylim=None, lsmooth=False, lprint=False, **kwargs):
    ''' A function to draw a list of 1D variables into an axes, 
        and annotate the plot based on variable properties. '''
    # varlist is the list of variable objects that are to be plotted
    #print varlist
    if isinstance(varlist,Variable): varlist = [varlist]
    elif not isinstance(varlist,(tuple,list,Ensemble)):raise TypeError
    if varname is not None:
      varlist = [getattr(var,varname) if isinstance(var,Dataset) else var for var in varlist]
    if not all([isinstance(var,Variable) for var in varlist]): raise TypeError
    for var in varlist: var.squeeze() # remove singleton dimensions
    # evaluate distribution variables on support/bins
    if support is not None or bins is not None:
      if support is not None and bins is not None: raise ArgumentError
      if support is None and bins is not None: support = bins
      for var in varlist:
        if not isinstance(var,(DistVar,VarKDE,VarRV)): raise TypeError, "{:s} ({:s})".format(var.name, var.__class__.__name__)
      newlist = [var.pdf(support=support) for var in varlist]
      for new,var in zip(newlist,varlist): new.dataset= var.dataset # preserve dataset links to construct references
      varlist = newlist
    # linestyles is just a list of line styles for each plot
    if isinstance(linestyles,(basestring,NoneType)): linestyles = [linestyles]*len(varlist)
    elif not isinstance(linestyles,(tuple,list)): 
      if not all([isinstance(linestyles,basestring) for var in varlist]): raise TypeError
      if len(varlist) != len(linestyles): raise ListError, "Failed to match linestyles to varlist!"
    # varatts are variable-specific attributes that are parsed for special keywords and then passed on to the
    if varatts is None: varatts = [dict()]*len(varlist)  
    elif isinstance(varatts,dict):
      tmp = [varatts[var.name] if var.name in varatts else dict() for var in varlist]
      if any(tmp): varatts = tmp # if any variable names were found
      else: varatts = [varatts]*len(varlist) # assume it is one varatts dict, which will be used for all variables
    elif not isinstance(varatts,(tuple,list)): raise TypeError
    if not all([isinstance(atts,dict) for atts in varatts]): raise TypeError
    # check axis: they need to have only one axes, which has to be the same for all!
    if len(varatts) != len(varlist): raise ListError, "Failed to match varatts to varlist!"  
    for var in varlist: 
      if var.ndim > 1: raise AxisError, "Variable '{}' has more than one dimension; consider squeezing.".format(var.name)
      elif var.ndim == 0: raise AxisError, "Variable '{}' is a scalar; consider display as a line.".format(var.name)
    # line/plot label policy
    lname = not any(var.name == varlist[0].name for var in varlist[1:])
    if lname or not all(var.dataset is not None for var in varlist): ldataset = False
    elif not any(var.dataset.name == varlist[0].dataset.name for var in varlist[1:]): ldataset = True
    else: ldataset = False
    # initialize axes names and units
    self.flipxy = flipxy
    if self.flipxy:
      varname,varunits,axname,axunits = self.xname,self.xunits,self.yname,self.yunits
    else:
      axname,axunits,varname,varunits = self.xname,self.xunits,self.yname,self.yunits
    # reset color cycle
    if reset_color is False: pass
    elif reset_color is True: self.set_color_cycle(None) # reset
    else: self.set_color_cycle(reset_color)
    # prepare label list
    if labels is None: labels = []; lmklblb = True
    elif len(labels) == len(varlist): lmklblb = False
    else: raise ArgumentError, "Incompatible length of label list."
    # loop over variables
    plts = [] # list of plot handles
    if self.plots is None: self.plots = []
    if self.variables is None: self.variables = OrderedDict()
    for n,var,linestyle,varatt in zip(xrange(len(varlist)),varlist,linestyles,varatts):
      varax = var.axes[0]
      # scale axis and variable values 
      axe, axunits, axname = getPlotValues(varax, checkunits=axunits, checkname=None)
      val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
      # variable and axis scaling is not always independent...
      if var.plot is not None and varax.plot is not None: 
        if varax.units != axunits and var.plot.preserve == 'area':
          val /= varax.plot.scalefactor
      # save variable 
      if lmklblb: 
        if lname: label = var.name # default label: variable name
        elif ldataset: label = var.dataset.name
        else: label = n
        labels.append(label)
      else: label = labels[n]
      self.variables[label] = var
      # figure out keyword options
      kwatts = kwargs.copy(); kwatts.update(varatt) # join individual and common attributes     
      kwatts['label'] = label if llabel else None
      # N.B.: other scaling behavior could be added here
      if lprint: print varname, varunits, np.nanmean(val), np.nanstd(val)   
      if lsmooth: val = smooth(val)
      # figure out orientation
      if self.flipxy: xx,yy = val, axe 
      else: xx,yy = axe, val
      # call plot function
      if linestyle is None: plt = self.plot(xx, yy, **kwatts)[0]
      else: plt = self.plot(xx, yy, linestyle, **kwatts)[0]
      plts.append(plt); self.plots.append(plt)
    # set plot scale (log/linear)
    if xlog: self.set_xscale('log')
    else: self.set_xscale('linear')
    if ylog: self.set_yscale('log')
    else: self.set_yscale('linear')
    # set axes limits
    if isinstance(xlim,(list,tuple)) and len(xlim)==2: self.set_xlim(*xlim)
    elif xlim is not None: raise TypeError
    if isinstance(ylim,(list,tuple)) and len(ylim)==2: self.set_ylim(*ylim)
    elif ylim is not None: raise TypeError 
    # set title
    if title is not None: self.addTitle(title)
    # set axes labels  
    if self.flipxy: 
      self.xname,self.xunits,self.yname,self.yunits = varname,varunits,axname,axunits
    else: 
      self.xname,self.xunits,self.yname,self.yunits = axname,axunits,varname,varunits
    # format axes ticks
    self.xTickLabels(xticks, loverlap=False)
    self.yTickLabels(yticks, loverlap=False)
    # format axes labels
    self.xLabel(xlabel)
    self.yLabel(ylabel)    
    # N.B.: a typical custom label that makes use of the units would look like this: 'custom label [{1:s}]', 
    # where {} will be replaced by the appropriate default units (which have to be the same anyway)
    # make monthly ticks
    if self.xname == 'time' and self.xunits == 'month':
      if len(xticks) == 12 or len(xticks) == 13:
        self.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2)) # self.minorticks_on()
    # add legend
    if isinstance(legend,dict): self.addLegend(**legend) 
    elif isinstance(legend,(int,np.integer,float,np.inexact)): self.addLegend(loc=legend)
    # add orientation lines
    if hline is not None: self.addHline(hline)
    if vline is not None: self.addVline(vline)
    # return handle
    return plts

  def histogram(self, varlist, varname=None, bins=None, binedgs=None, histtype='bar', lstacked=False, lnormalize=True,
                lcumulative=0, varatts=None, legend=None, colors=None, llabel=True, labels=None, align='mid', rwidth=None, 
                bottom=None, weights=None, xticks=True, yticks=True, hline=None, vline=None, title=None, reset_color=True, 
                flipxy=None, xlabel=True, ylabel=True, log=False, xlim=None, ylim=None, lprint=False, **kwargs):
    ''' A function to draw histograms of a list of 1D variables into an axes, 
        and annotate the plot based on variable properties. '''
    # varlist is the list of variable objects that are to be plotted
    if isinstance(varlist,Variable): varlist = [varlist]
    elif isinstance(varlist,(tuple,list,Ensemble)): pass
    elif isinstance(varlist,Dataset): pass
    else: raise TypeError
    if varname is not None:
      varlist = [getattr(var,varname) if isinstance(var,Dataset) else var for var in varlist]
    if not all([isinstance(var,Variable) for var in varlist]): raise TypeError
    for var in varlist: var.squeeze() # remove singleton dimensions
    # varatts are variable-specific attributes that are parsed for special keywords and then passed on to the
    if varatts is None: varatts = [dict()]*len(varlist)  
    elif isinstance(varatts,dict):
      tmp = [varatts[var.name] if var.name in varatts else dict() for var in varlist]
      if any(tmp): varatts = tmp # if any variable names were found
      else: varatts = [varatts]*len(varlist) # assume it is one varatts dict, which will be used for all variables
    elif not isinstance(varatts,(tuple,list)): raise TypeError
    if not all([isinstance(atts,dict) for atts in varatts]): raise TypeError
    # check axis: they need to have only one axes, which has to be the same for all!
    if len(varatts) != len(varlist): raise ListError, "Failed to match varatts to varlist!"  
    # line/plot label policy
    lname = not any(var.name == varlist[0].name for var in varlist[1:])
    if lname or not all(var.dataset is not None for var in varlist): ldataset = False
    elif not any(var.dataset.name == varlist[0].dataset.name for var in varlist[1:]): ldataset = True
    else: ldataset = False
    # initialize axes names and units
    self.flipxy = flipxy
    if not self.flipxy: # histogram has opposite convention
      varname,varunits,axname,axunits = self.xname,self.xunits,self.yname,self.yunits
    else:
      axname,axunits,varname,varunits = self.xname,self.xunits,self.yname,self.yunits
    # reset color cycle
    if reset_color: self.set_color_cycle(None)
    # figure out bins
    vmin = np.min([var.min() for var in varlist])
    vmax = np.max([var.max() for var in varlist])
    bins, binedgs = binedges(bins=bins, binedgs=binedgs, limits=(vmin,vmax), lcheckVar=True)
    # prepare label list
    if labels is None: labels = []; lmklblb = True
    elif len(labels) == len(varlist): lmklblb = False
    else: raise ArgumentError, "Incompatible length of label list."
    if self.variables is None: self.variables = OrderedDict()
    # loop over variables
    values = [] # list of plot handles
    for n,var in zip(xrange(len(varlist)),varlist):
      # scale variable values(axes are irrelevant)
      val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
      val = val.ravel() # flatten array
      if not varname.endswith('_bins'): varname += '_bins'
      # figure out label
      if lmklblb:
        if lname: label = var.name # default label: variable name
        elif ldataset: label = var.dataset.name
        else: label = n
        labels.append(label)
      else: label = labels[n]
      # save variable  
      self.variables[label] = var
      if lprint: print varname, varunits, np.nanmean(val), np.nanstd(val)  
      # save values
      values.append(val)
    # figure out orientation
    if self.flipxy: orientation = 'horizontal' 
    else: orientation = 'vertical'
    # call histogram method of Axis
    if not llabel: labels = None 
    hdata, bin_edges, patches = self.hist(values, bins=binedgs, normed=lnormalize, weights=weights, cumulative=lcumulative, 
                                          bottom=bottom, histtype=histtype, align=align, orientation=orientation, 
                                          rwidth=rwidth, log=log, color=colors, label=labels, stacked=lstacked, **kwargs)
    del hdata; assert isEqual(bin_edges, binedgs)
    # N.B.: generally we don't need to keep the histogram results - there are other functions for that
    # set axes limits
    if isinstance(xlim,(list,tuple)) and len(xlim)==2: self.set_xlim(*xlim)
    elif xlim is not None: raise TypeError
    if isinstance(ylim,(list,tuple)) and len(ylim)==2: self.set_ylim(*ylim)
    elif ylim is not None: raise TypeError 
    # set title
    if title is not None: self.addTitle(title)
    # set axes labels  
    if not self.flipxy: 
      self.xname,self.xunits,self.yname,self.yunits = varname,varunits,axname,axunits
    else: 
      self.xname,self.xunits,self.yname,self.yunits = axname,axunits,varname,varunits
    # format axes ticks
    self.xTickLabels(xticks, loverlap=False)
    self.yTickLabels(yticks, loverlap=False)
    # format axes labels
    self.xLabel(xlabel)
    self.yLabel(ylabel)    
    # N.B.: a typical custom label that makes use of the units would look like this: 'custom label [{1:s}]', 
    # where {} will be replaced by the appropriate default units (which have to be the same anyway)
    # make monthly ticks
    if self.xname == 'time' and self.xunits == 'month':
      if len(xticks) == 12 or len(xticks) == 13:
        self.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2)) # self.minorticks_on()
    # add legend
    if isinstance(legend,dict): self.addLegend(**legend) 
    elif isinstance(legend,(int,np.integer,float,np.inexact)): self.addLegend(loc=legend)
    # add orientation lines
    if hline is not None: self.addHline(hline)
    if vline is not None: self.addVline(vline)
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
  
  def addTitle(self, title, **kwargs):
    ''' add title and adjust margins '''
    if 'fontsize' not in kwargs: kwargs['fontsize'] = 'medium'
    title_height = kwargs.pop('title_height', self.title_height)
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
    
  def xTickLabels(self, xticks, loverlap=False):
    ''' format x-tick labels '''
    xticks = self._tickLabels(xticks, self.get_xaxis())
    yticks = self.get_yaxis().get_ticklabels()
    if not loverlap and len(xticks) > 0 and (
        len(yticks) == 0 or not yticks[-1].get_visible() ):
        xticks[0].set_visible(False)
    return xticks
  def yTickLabels(self, yticks, loverlap=False):
    ''' format y-tick labels '''
    xticks = self.get_xaxis().get_ticklabels()
    yticks = self._tickLabels(yticks, self.get_yaxis())
    if not loverlap and len(yticks) > 0 and (
        len(xticks) == 0 or not xticks[-1].get_visible() ):
        yticks[0].set_visible(False)
    return yticks
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