'''
Created on 2011-02-28

utility functions, mostly for plotting, that are not called directly

@author: Andre R. Erler
'''

# external imports
from types import ModuleType
import numpy as np
# internal imports
from geodata.base import Variable
from geodata.misc import VariableError, AxisError, isInt

# import matplotlib as mpl
# import matplotlib.pylab as pyl

# load matplotlib with some custom defaults
def loadMPL(linewidth=None, mplrc=None):
  import matplotlib as mpl
  mpl.use('QT4Agg') # enforce QT4
  import matplotlib.pylab as pyl
  # some custom defaults  
  if linewidth is not None:
    mpl.rc('lines', linewidth=linewidth)
    if linewidth == 1.5: mpl.rc('font', size=12)
    elif linewidth == .75: mpl.rc('font', size=8)
    else: mpl.rc('font', size=10)
  # apply rc-parameters from dictionary (override custom defaults)
  if (mplrc is not None) and isinstance(mplrc,dict):
    # loop over parameter groups
    for (key,value) in mplrc.iteritems():
      mpl.rc(key,**value)  # apply parameters
  # prevent figures from closing: don't run in interactive mode, or pyl.show() will not block
  pyl.ioff()
  # return matplotlib instance with new parameters
  return mpl, pyl


# method to return a figure and an array of ImageGrid axes
def getFigAx(subplot, name=None, title=None, figsize=None,  mpl=None, margins=None,
	     sharex=None, sharey=None, AxesGrid=False, ngrids=None, direction='row',
	     axes_pad = None, add_all=True, share_all=None, aspect=False,
	     label_mode='L', cbar_mode=None, cbar_location='right',
	     cbar_pad=None, cbar_size='5%', axes_class=None, lreduce=True): 
  # configure matplotlib
  if mpl is None: import matplotlib as mpl
  elif isinstance(mpl,dict): mpl = loadMPL(**mpl) # there can be a mplrc, but also others
  elif not isinstance(mpl,ModuleType): raise TypeError
  # figure out subplots
  if isinstance(subplot,(np.integer,int)):
    if subplot == 1: subplot = (1,1)
    elif subplot == 2: subplot = (1,2)
    elif subplot == 3: subplot = (1,3)
    elif subplot == 4: subplot = (2,2)
    elif subplot == 6: subplot = (2,3)
    elif subplot == 9: subplot = (3,3)
    else: raise NotImplementedError
  elif not (isinstance(subplot,(tuple,list)) and len(subplot) == 2) and all(isInt(subplot)): raise TypeError    
  # create figure
  if figsize is None: 
    if subplot == (1,1): figsize = (3.75,3.75)
    elif subplot == (1,2) or subplot == (1,3): figsize = (6.25,3.75)
    elif subplot == (2,1) or subplot == (3,1): figsize = (3.75,6.25)
    else: figsize = (6.25,6.25)
    #elif subplot == (2,2) or subplot == (3,3): figsize = (6.25,6.25)
    #else: raise NotImplementedError
  # figure out margins
  if margins is None:
    # N.B.: the rectangle definition is presumably left, bottom, width, height
    if subplot == (1,1): margins = (0.09,0.09,0.88,0.88)
    elif subplot == (1,2) or subplot == (1,3): margins = (0.06,0.1,0.92,0.87)
    elif subplot == (2,1) or subplot == (3,1): margins = (0.09,0.11,0.88,0.82)
    elif subplot == (2,2) or subplot == (3,3): margins = (0.055,0.055,0.925,0.925)
    else: margins = (0.09,0.11,0.88,0.82)
    #elif subplot == (2,2) or subplot == (3,3): margins = (0.09,0.11,0.88,0.82)
    #else: raise NotImplementedError    
    if title is not None: margins = margins[:3]+(margins[3]-0.03,) # make room for title
  if AxesGrid:
    if share_all is None: share_all = True
    if axes_pad is None: axes_pad = 0.05
    # create axes using the Axes Grid package
    fig = mpl.pylab.figure(facecolor='white', figsize=figsize)
    from mpl_toolkits.axes_grid1 import ImageGrid
    # AxesGrid: http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
    grid = ImageGrid(fig, margins, nrows_ncols = subplot, ngrids=ngrids, direction=direction, 
            		     axes_pad=axes_pad, add_all=add_all, share_all=share_all, aspect=aspect, 
                     label_mode=label_mode, cbar_mode=cbar_mode, cbar_location=cbar_location, 
                     cbar_pad=cbar_pad, cbar_size=cbar_size, axes_class=axes_class)
    # return figure and axes
    axes = tuple([ax for ax in grid]) # this is already flattened
    if lreduce and len(axes) == 1: axes = axes[0] # return a bare axes instance, if there is only one axes    
  else:
    # create axes using normal subplot routine
    if axes_pad is None: axes_pad = 0.03
    wspace = hspace = axes_pad
    if share_all: 
      sharex='all'; sharey='all'
    if sharex is True or sharex is None: sharex = 'col' # default
    if sharey is True or sharey is None: sharey = 'row'
    if sharex: hspace -= 0.015
    if sharey: wspace -= 0.015
    # create figure
    from matplotlib.pyplot import subplots
    # GridSpec: http://matplotlib.org/users/gridspec.html 
    fig, axes = subplots(subplot[0], subplot[1], sharex=sharex, sharey=sharey,
			                   squeeze=lreduce, facecolor='white', figsize=figsize)    
    # there is also a subplot_kw=dict() and fig_kw=dict()
    # just adjust margins
    margin_dict = dict(left=margins[0], bottom=margins[1], right=margins[0]+margins[2], 
                       top=margins[1]+margins[3], wspace=wspace, hspace=hspace)
    fig.subplots_adjust(**margin_dict)
  # add figure title
  if name is not None: fig.canvas.set_window_title(name) # window title
  if title is not None: fig.suptitle(title) # title on figure (printable)
  # return Figure/ImageGrid and tuple of axes
  #if AxesGrid: fig = grid # return ImageGrid instead of figure
  return fig, axes

# function to adjust subplot parameters
def updateSubplots(fig, mode='shift', **kwargs):
  ''' simple helper function to move (relocate), shift, or scale subplot margins '''
  pos = fig.subplotpars
  margins = dict() # original plot margins
  margins['left'] = pos.left; margins['right'] = pos.right 
  margins['top'] = pos.top; margins['bottom'] = pos.bottom
  margins['wspace'] = pos.wspace; margins['hspace'] = pos.hspace
  # update subplot margins
  if mode == 'move': margins.update(kwargs)
  else: 
    for key,val in kwargs.iteritems():
      if key in margins:
        if mode == 'shift': margins[key] += val
        elif mode == 'scale': margins[key] *= val
  # finally, actually update figure
  fig.subplots_adjust(**margins)
  # and now repair damage: restore axes
  for ax in fig.axes:
    if ax.get_title():
      pos = ax.get_position()
      pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-0.03)    
      ax.set_position(pos)

# method to check units and name, and return scaled plot value (primarily and internal helper function)
def getPlotValues(var, checkunits=None, checkname=None):
  ''' Helper function to check variable/axis, get (scaled) values for plot, and return appropriate units. '''
  if var.plot is not None and 'name' in var.plot: 
    varname = var.plot.name 
    if checkname is not None and varname != checkname: # only check plotname! 
      raise VariableError, "Expected variable name '{}', found '{}'.".format(checkname,varname)
  else: varname = var.atts['name']
  val = var.getArray(unmask=True) # the data to plot
  if var.plot is not None and 'scalefactor' in var.plot: 
    if var.atts['units'] != var.plot.units: 
      val = val *  var.plot.scalefactor
    varunits = var.plot.units
  else: varunits = var.atts['units']
  if var.plot is not None and 'offset' in var.plot: val += var.plot.offset    
  if checkunits is not None and  varunits != checkunits: 
    raise VariableError, "Units for variable '{}': expected {}, found {}.".format(var.name,checkunits,varunits) 
  # return values, units, name
  return val, varunits, varname     

  
# Log-axis ticks
def logTicks(ticks, base=None, power=0):
  ''' function to generate ticks for a given power of 10 based on a template '''
  if not isinstance(ticks, (list,tuple)): raise TypeError
  # translate base into power
  if base is not None: 
    if not isinstance(base,(int,np.number,float,np.inexact)): raise TypeError
    power = int(np.round(np.log(base)/np.log(10)))
  if not isinstance(power,(int,np.integer)): raise TypeError
  print power
  # generate ticks and apply template
  strtck = ['']*8
  for i in ticks:
    if not isinstance(i,(int,np.integer)) or i >= 8: raise ValueError
    idx = i-2
    if i in ticks: strtck[idx] = str(i)
    # adjust order of magnitude
    if power > 0: strtck[idx] += '0'*power
    elif power < 0: strtck[idx] = '0.' + '0'*(-1-power) + strtck[idx]
  # return ticks
  return strtck
# special versions
# N, returns ['2','','4','','6','','','']
def nTicks(**kwargs): return logTicks([2,4,6],**kwargs) 
# p, returns ['2','3','','5','','7','','']
def pTicks(**kwargs): return logTicks([2,3,5,7],**kwargs)


# function to smooth a vector (numpy array): moving mean, nothing fancy
def smoothVector(x,i):
  xs = x.copy() # smoothed output vector
  i = 2*i
  d = i+1 # denominator  for later
  while i>0:    
    t = x.copy(); t[i:] = t[:-i];  xs += t
    t = x.copy(); t[:-i] = t[i:];  xs += t
    i-=2
  return xs/d


# function to traverse nested lists recursively and perform the operation fct on the end members
def traverseList(lsl, fct):
  # traverse nested lists recursively
  if isinstance(lsl, list):
    return [traverseList(lsl[i], fct) for i in range(len(lsl))]
  # break recursion and apply function using the list element as argument 
  else: return fct(lsl)

  
# function to expand level lists and colorbar ticks
def expandLevelList(arg, vec=None):
  from numpy import asarray, ndarray, linspace, min, max
  ## figure out level list and return numpy array of levels
  # trivial case: already numpy array
  if isinstance(arg,ndarray):
    return arg 
  # list: recast as array
  elif isinstance(arg,list):
    return asarray(arg)
  # tuple with three or two elements: use as argument to linspace 
  elif isinstance(arg,tuple) and (len(arg)==3 or len(arg)==2):
    return linspace(*arg)
  # use additional info in vec to determine limits
  else:
    # figure out vec limits
    # use first two elements, third is number of levels
    if isinstance(vec,(tuple,list)) and len(vec)==3:  
      minVec = min(vec[:2]); maxVec = max(vec[:2])
    # just treat as level list
    else: 
      minVec = min(vec); maxVec = max(vec)
    # interpret arg as number of levels in given interval
    # only one element: just number of levels
    if isinstance(arg,(tuple,list,ndarray)) and len(arg)==1: 
      return linspace(minVec,maxVec,arg[0])
    # numerical value: use as number of levels
    elif isinstance(arg,(int,float)):
      return linspace(minVec,maxVec,arg)        


## add subplot/axes label
def addLabel(ax, label=None, loc=1, stroke=False, size=None, prop=None, **kwargs):
  from matplotlib.offsetbox import AnchoredText 
  from matplotlib.patheffects import withStroke
  from string import lowercase    
  # expand list
  if not isinstance(ax,(list,tuple)): ax = [ax] 
  l = len(ax)
  if not isinstance(label,(list,tuple)): label = [label]*l
  if not isinstance(loc,(list,tuple)): loc = [loc]*l
  if not isinstance(stroke,(list,tuple)): stroke = [stroke]*l
  # settings
  if prop is None:
    prop = dict()
  if not size: prop['size'] = 18
  args = dict(pad=0., borderpad=1.5, frameon=False)
  args.update(kwargs)
  # cycle over axes
  at = [] # list of texts
  for i in xrange(l):
    if label[i] is None:
      label[i] = '('+lowercase[i]+')'
    elif isinstance(label[i],int):
      label[i] = '('+lowercase[label[i]]+')'
    # create label    
    at.append(AnchoredText(label[i], loc=loc[i], prop=prop, **args))
    ax[i].add_artist(at[i]) # add to axes
    if stroke[i]: 
      at[i].txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
  return at
