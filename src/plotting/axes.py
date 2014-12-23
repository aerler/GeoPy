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
from geodata.base import Variable, Ensemble
from geodata.misc import ListError, AxisError
from plotting.misc import smooth, getPlotValues


## new axes class
class MyAxes(Axes): 
  ''' 
    A custom Axes class that provides some specialized plotting functions and retains variable 
    information. The custom Figure uses this Axes class by default.
  '''
  variables = None
  
  def linePlot(self, varlist, linestyles=None, varatts=None, legend=None,
               xline=None, yline=None, title=None, flipxy=None, xlabel=None, ylabel=None, xlim=None,
               ylim=None, lsmooth=False, lprint=False, **kwargs):
    ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on variable properties. '''
    # varlist is the list of variable objects that are to be plotted
    #print varlist
    if isinstance(varlist,Variable): varlist = [varlist]
    elif not isinstance(varlist,(tuple,list,Ensemble)) or not all([isinstance(var,Variable) for var in varlist]): raise TypeError
    for var in varlist: var.squeeze() # remove singleton dimensions
    self.variables = varlist # save references to variables
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
    # loop over variables
    plts = []; varname = None; varunits = None; axname = None; axunits = None # list of plot handles
    for var,linestyle,varatt in zip(varlist,linestyles,varatts):
      varax = var.axes[0]
      # scale axis and variable values 
      axe, axunits, axname = getPlotValues(varax, checkunits=axunits, checkname=None)
      val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
      # variable and axis scaling is not always independent...
      if var.plot is not None and varax.plot is not None: 
        if 'preserve' in var.plot and 'scalefactor' in varax.plot:
          if varax.units != axunits and var.plot.preserve == 'area':
            val /= varax.plot.scalefactor  
      # figure out keyword options
      kwatts = kwargs.copy(); kwatts.update(varatt) # join individual and common attributes     
      if 'label' not in kwatts: kwatts['label'] = var.name # default label: variable name
      # N.B.: other scaling behavior could be added here
      if lprint: print varname, varunits, val.mean()    
      if lsmooth: val = smooth(val)
      # figure out orientation
      if flipxy: xx,yy = val, axe 
      else: xx,yy = axe, val
      # call plot function
      if linestyle is None: plts.append(self.plot(xx, yy, **kwatts)[0])
      else: plts.append(self.plot(xx, yy, linestyle, **kwatts)[0])
    # set axes limits
    if isinstance(xlim,(list,tuple)) and len(xlim)==2: self.set_xlim(*xlim)
    elif xlim is not None: raise TypeError
    if isinstance(ylim,(list,tuple)) and len(ylim)==2: self.set_ylim(*ylim)
    elif ylim is not None: raise TypeError 
    # set title
    if title is not None:
      self.set_title(title, dict(fontsize='medium'))
      pos = self.get_position()
      pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-0.03)    
      self.set_position(pos)
    # set axes labels  
    if flipxy: xname,xunits,yname,yunits = varname,varunits,axname,axunits
    else: xname,xunits,yname,yunits = axname,axunits,varname,varunits
    if not xlabel: xlabel = '{0:s} [{1:s}]'.format(xname,xunits) if xunits else '{0:s}'.format(xname)
    else: xlabel = xlabel.format(xname,xunits)
    if not ylabel: ylabel = '{0:s} [{1:s}]'.format(yname,yunits) if yunits else '{0:s}'.format(yname)
    else: ylabel = ylabel.format(yname,yunits)
    # a typical custom label that makes use of the units would look like this: 'custom label [{1:s}]', 
    # where {} will be replaced by the appropriate default units (which have to be the same anyway)
    xpad =  2; xticks = self.get_xaxis().get_ticklabels()
    ypad = -2; yticks = self.get_yaxis().get_ticklabels()
    # len(xticks) > 0 is necessary to avoid errors with AxesGrid, which removes invisible tick labels 
    if len(xticks) > 0 and xticks[-1].get_visible(): self.set_xlabel(xlabel, labelpad=xpad)
    elif len(yticks) > 0 and not title: yticks[0].set_visible(False) # avoid overlap
    if len(yticks) > 0 and yticks[-1].get_visible(): self.set_ylabel(ylabel, labelpad=ypad)
    elif len(xticks) > 0: xticks[0].set_visible(False) # avoid overlap
    # make monthly ticks
    if axname == 'time' and axunits == 'month':
      self.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2)) # self.minorticks_on()
    # add legend
    if legend:
      legatts = dict()
      if self.get_yaxis().get_label():
        legatts['fontsize'] = self.get_yaxis().get_label().get_fontsize()
      if isinstance(legend,dict): legatts.update(legend) 
      elif isinstance(legend,(int,np.integer,float,np.inexact)): legatts['loc'] = legend
      self.legend(**legatts)
    # add orientation lines
    if isinstance(xline,(int,np.integer,float,np.inexact)): self.axhline(y=xline, color='black')
    elif isinstance(xline,dict): self.axhline(**xline)
    if isinstance(yline,(int,np.integer,float,np.inexact)): self.axvline(x=yline, color='black')
    elif isinstance(xline,dict): self.axvline(**yline)
    # return handle
    return plts
  
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