'''
Created on 2014-03-16

some useful plotting functions that take advantage of variable meta data

@author: Andre R. Erler, GPL v3
'''

# external imports
from types import NoneType
import numpy as np
import matplotlib as mpl
# import matplotlib.pylab as pyl
# #from mpl_toolkits.axes_grid1 import ImageGrid
# linewidth = .75
# mpl.rc('lines', linewidth=linewidth)
# if linewidth == 1.5: mpl.rc('font', size=12)
# elif linewidth == .75: mpl.rc('font', size=8)
# else: mpl.rc('font', size=10)
# # prevent figures from closing: don't run in interactive mode, or plt.show() will not block
# pyl.ioff()
# internal imports
from misc.signalsmooth import smooth
from utils import getPlotValues, getFigAx, updateSubplots
from geodata.base import Variable
from geodata.misc import AxisError, ListError, VariableError

#import pdb
#pdb.set_trace()


def linePlot(varlist, ax=None, fig=None, linestyles=None, varatts=None, legend=None,
	   				 xline=None, yline=None, title=None, flipxy=None, xlabel=None, ylabel=None, xlim=None,
	  	   		 ylim=None, lsmooth=False, lprint=False, **kwargs):
  ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on variable properties. '''
  # create axes, if necessary
  if ax is None: 
    if fig is None: fig,ax = getFigAx(1) # single panel
    else: ax = fig.axes[0]
  # varlist is the list of variable objects that are to be plotted
  #print varlist
  if isinstance(varlist,Variable): varlist = [varlist]
  elif not isinstance(varlist,(tuple,list)) or not all([isinstance(var,Variable) for var in varlist]): raise TypeError
  for var in varlist: var.squeeze() # remove singleton dimensions
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
    if linestyle is None: plts.append(ax.plot(xx, yy, **kwatts)[0])
    else: plts.append(ax.plot(xx, yy, linestyle, **kwatts)[0])
  # set axes limits
  if isinstance(xlim,(list,tuple)) and len(xlim)==2: ax.set_xlim(*xlim)
  elif xlim is not None: raise TypeError
  if isinstance(ylim,(list,tuple)) and len(ylim)==2: ax.set_ylim(*ylim)
  elif ylim is not None: raise TypeError 
  # set title
  if title is not None:
    ax.set_title(title, dict(fontsize='medium'))
    pos = ax.get_position()
    pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-0.03)    
    ax.set_position(pos)
  # set axes labels  
  if flipxy: xname,xunits,yname,yunits = varname,varunits,axname,axunits
  else: xname,xunits,yname,yunits = axname,axunits,varname,varunits
  if not xlabel: xlabel = '{0:s} [{1:s}]'.format(xname,xunits) if xunits else '{0:s}'.format(xname)
  else: xlabel = xlabel.format(xname,xunits)
  if not ylabel: ylabel = '{0:s} [{1:s}]'.format(yname,yunits) if yunits else '{0:s}'.format(yname)
  else: ylabel = ylabel.format(yname,yunits)
  # a typical custom label that makes use of the units would look like this: 'custom label [{1:s}]', 
  # where {} will be replaced by the appropriate default units (which have to be the same anyway)
  xpad =  2; xticks = ax.get_xaxis().get_ticklabels()
  ypad = -2; yticks = ax.get_yaxis().get_ticklabels()
  # len(xticks) > 0 is necessary to avoid errors with AxesGrid, which removes invisible tick labels 
  if len(xticks) > 0 and xticks[-1].get_visible(): ax.set_xlabel(xlabel, labelpad=xpad)
  elif len(yticks) > 0 and not title: yticks[0].set_visible(False) # avoid overlap
  if len(yticks) > 0 and yticks[-1].get_visible(): ax.set_ylabel(ylabel, labelpad=ypad)
  elif len(xticks) > 0: xticks[0].set_visible(False) # avoid overlap
  # make monthly ticks
  if axname == 'time' and axunits == 'month':
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2)) # ax.minorticks_on()
  # add legend
  if legend:
    legatts = dict()
    if ax.get_yaxis().get_label():
      legatts['fontsize'] = ax.get_yaxis().get_label().get_fontsize()
    if isinstance(legend,dict): legatts.update(legend) 
    elif isinstance(legend,(int,np.integer,float,np.inexact)): legatts['loc'] = legend
    ax.legend(**legatts)
  # add orientation lines
  if isinstance(xline,(int,np.integer,float,np.inexact)): ax.axhline(y=xline, color='black')
  elif isinstance(xline,dict): ax.axhline(**xline)
  if isinstance(yline,(int,np.integer,float,np.inexact)): ax.axvline(x=yline, color='black')
  elif isinstance(xline,dict): ax.axvline(**yline)
  # return handle
  return plts      


# add common/shared legend to a multi-panel plot
def addSharedLegend(fig, plts=None, legs=None, fontsize=None, **kwargs):
  ''' add a common/shared legend to a multi-panel plot '''
  # complete input
  if legs is None: legs = [plt.get_label() for plt in plts]
  elif not isinstance(legs, (list,tuple)): raise TypeError
  if not isinstance(plts, (list,tuple,NoneType)): raise TypeError
  # figure out fontsize and row numbers  
  fontsize = fontsize or fig.axes[0].get_yaxis().get_label().get_fontsize() # or fig._suptitle.get_fontsize()
  nlen = len(plts) if plts else len(legs)
  if fontsize > 11: ncols = 2 if nlen == 4 else 3
  else: ncols = 3 if nlen == 6 else 4              
  # make room for legend
  leghgt = np.ceil(nlen/ncols) * fontsize + 0.055
  ax = fig.add_axes([0, 0, 1,leghgt]) # new axes to hold legend, with some attributes
  ax.set_frame_on(False); ax.axes.get_yaxis().set_visible(False); ax.axes.get_xaxis().set_visible(False)
  fig = updateSubplots(fig, mode='shift', bottom=leghgt) # shift bottom upwards
  # define legend parameters
  legargs = dict(loc=10, ncol=ncols, borderaxespad=0., fontsize=fontsize, frameon=True,
                 labelspacing=0.1, handlelength=1.3, handletextpad=0.3, fancybox=True)
  legargs.update(kwargs)
  # create legend and return handle
  if plts: legend = ax.legend(plts, legs, **legargs)
  else: legend = ax.legend(legs, **legargs)
  return legend
  
  

# plots with error shading 
def addErrorPatch(ax, var, err, color=None, axis=None, xerr=True, alpha=0.25, check=False, cap=-1):
  from numpy import append, where, isnan
  from matplotlib.patches import Polygon 
  if isinstance(var,Variable):    
    if axis is None and var.ndim > 1: raise AxisError
    x = var.getAxis(axis).getArray()
    y = var.getArray(); 
    if isinstance(err,Variable): e = err.getArray()
    else: e = err
  else:
    if axis is None: raise ValueError
    y = axis; x = var; e = err
  if check:
    e = where(isnan(e),0,e)
    if cap > 0: e = where(e>cap,0,e)
  if xerr: 
    ix = append(x-e,(x+e)[::-1])
    iy = append(y,y[::-1])
  else:
    ix = append(y,y[::-1])
    iy = append(x-e,(x+e)[::-1])
  if color is None: raise NotImplementedError # should take color from plot line (variable)
  patch = Polygon(zip(ix,iy), alpha=alpha, facecolor=color, edgecolor=color)
  ax.add_patch(patch)
  return patch 


if __name__ == '__main__':
  pass