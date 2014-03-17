'''
Created on 2011-02-28

utility functions, mostly for plotting, that are not called directly

@author: Andre R. Erler
'''

# external imports
import numpy as np
# internal imports
from geodata.base import Variable
from geodata.misc import AxisError

# load matplotlib (default)
def loadMPL(mplrc=None):
  import matplotlib as mpl
  # apply rc-parameters from dictionary
  if (mplrc is not None) and isinstance(mplrc,dict):
    # loop over parameter groups
    for (key,value) in mplrc.iteritems():
      mpl.rc(key,**value)  # apply parameters
  # return matplotlib instance with new parameters
  return mpl

  
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

  
# function to place (shared) colorbars at a specified figure margins
def sharedColorbar(fig, cf, clevs, colorbar, cbls, subplot, margins):
  loc = colorbar.pop('location','bottom')      
  # determine size and spacing
  if loc=='top' or loc=='bottom':
    orient = colorbar.pop('orientation','horizontal') # colorbar orientation
    je = subplot[1] # number of colorbars: number of rows
    ie = subplot[0] # number of plots per colorbar: number of columns
    cbwd = colorbar.pop('cbwd',0.025) # colorbar height
    sp = margins['wspace']
    wd = (margins['right']-margins['left'] - sp*(je-1))/je # width of each colorbar axis 
  else:
    orient = colorbar.pop('orientation','vertical') # colorbar orientation
    je = subplot[0] # number of colorbars: number of columns
    ie = subplot[1] # number of plots per colorbar: number of rows
    cbwd = colorbar.pop('cbwd',0.025) # colorbar width
    sp = margins['hspace']
    wd = (margins['top']-margins['bottom'] - sp*(je-1))/je # width of each colorbar axis
  shrink = colorbar.pop('shrinkFactor',1)
  # shift existing subplots
  if loc=='top': newMargin = margins['top']-margins['hspace'] -cbwd
  elif loc=='right': newMargin = margins['right']-margins['left']/2 -cbwd
  else: newMargin = 2*margins[loc] + cbwd    
  fig.subplots_adjust(**{loc:newMargin})
  # loop over variables (one colorbar for each)
  for i in range(je):
    if dir=='vertical': ii = je-i-1
    else: ii = i
    offset = (wd+sp)*float(ii) + wd*(1-shrink)/2 # offset due to previous colorbars
    # horizontal colorbar(s) at the top
    if loc == 'top': ci = i; cax = [margins['left']+offset, newMargin+margins['hspace'], shrink*wd, cbwd]             
    # horizontal colorbar(s) at the bottom
    elif loc == 'bottom': ci = i; cax = [margins['left']+offset, margins[loc], shrink*wd, cbwd]        
    # vertical colorbar(s) to the left (get axes reference right!)
    elif loc == 'left': ci = i*ie; cax = [margins[loc], margins['bottom']+offset, cbwd, shrink*wd]        
    # vertical colorbar(s) to the right (get axes reference right!)
    elif loc == 'right': ci = i*ie; cax = [newMargin+margins['wspace'], margins['bottom']+offset, cbwd, shrink*wd]
    # make colorbar 
    fig.colorbar(mappable=cf[ci],cax=fig.add_axes(cax),ticks=expandLevelList(cbls[i],clevs[i]),
                 orientation=orient,**colorbar)
  # return figure with colorbar (just for the sake of returning something) 
  return fig
