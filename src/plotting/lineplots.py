'''
Created on 2014-03-16

some useful plotting functions that take advantage of variable meta data

@author: Andre R. Erler, GPL v3
'''

# external imports
from types import NoneType
import numpy as np
# import matplotlib.pylab as pyl
# import matplotlib as mpl
# #from mpl_toolkits.axes_grid1 import ImageGrid
# linewidth = .75
# mpl.rc('lines', linewidth=linewidth)
# if linewidth == 1.5: mpl.rc('font', size=12)
# elif linewidth == .75: mpl.rc('font', size=8)
# else: mpl.rc('font', size=10)
# # prevent figures from closing: don't run in interactive mode, or plt.show() will not block
# pyl.ioff()
# internal imports
from utils import getPlotValues, getFigAx
from geodata.base import Variable
from geodata.misc import AxisError, ListError, VariableError


def linePlot(varlist, ax=None, fig=None, linestyles=None, varatts=None, legend=None, xline=None, yline=None, 
             title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):
  ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on variable properties. '''
  # create axes, if necessary
  if ax is None: 
    if fig is None: fig,ax = getFigAx(1) # single panel
    else: ax = fig.axes[0]
  # varlist is the list of variable objects that are to be plotted
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
  axname = varlist[0].axes[0].name
  for var in varlist:
    if not var.ndim: raise AxisError, "Variable '{}' has more than one dimension.".format(var.name)
    if not var.hasAxis(axname): raise AxisError, "Variable {} does not have a '{}' axis.".format(var.name,axname)
  # loop over variables
  flipxy = kwargs.pop('flipxy',False)
  plts = []; varname = None; varunits = None; axname = None; axunits = None # list of plot handles
  for var,linestyle,varatt in zip(varlist,linestyles,varatts):
    axe, axunits, axname = getPlotValues(var.axes[0], checkunits=axunits, checkname=axname)
    val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
    # figure out keyword options
    kwatts = kwargs.copy(); kwatts.update(varatt) # join individual and common attributes     
    if 'label' not in kwatts: kwatts['label'] = var.name # default label: variable name
    # N.B.: other scaling behavior could be added here
    print varname, varunits, val.mean()
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
  if title is not None: ax.set_title(title)
  # set axes labels
  xpad = 2; ypad = -2
  xlabel = xlabel or ('{} [{}]'.format(varname,varunits) if flipxy else '{} [{}]'.format(axname,axunits)) 
  ylabel = ylabel or ('{} [{}]'.format(axname,axunits) if flipxy else '{} [{}]'.format(varname,varunits)) 
  ax.set_xlabel(xlabel, labelpad=xpad)
  ax.set_ylabel(ylabel, labelpad=ypad)
  # make monthly ticks
  if axname == 'time' and axunits == 'month':
    #ax.minorticks_on()
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
  # add legend
  if isinstance(legend,dict): ax.legend(**legend)
  elif isinstance(legend,(int,np.integer,float,np.inexact)): ax.legend(loc=legend)
  # add orientation lines
  if isinstance(xline,(int,np.integer,float,np.inexact)): ax.axhline(y=xline, color='black')
  elif isinstance(xline,dict): ax.axhline(**xline)
  if isinstance(yline,(int,np.integer,float,np.inexact)): ax.axvline(x=yline, color='black')
  elif isinstance(xline,dict): ax.axvline(**yline)
  # return handle
  return plts      


# plots with error shading 
def addErrorPatch(ax, var, err, color, axis=None, xerr=True, alpha=0.25, check=False, cap=-1):
  from numpy import append, where, isnan
  from matplotlib.patches import Polygon 
  if isinstance(var,Variable):    
    if axis is None and var.ndim > 1: raise AxisError
    y = var.getAxis(axis).getArray()
    x = var.getArray(); 
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
  patch = Polygon(zip(ix,iy), alpha=alpha, facecolor=color, edgecolor=color)
  ax.add_patch(patch)
  return patch 


if __name__ == '__main__':
  pass