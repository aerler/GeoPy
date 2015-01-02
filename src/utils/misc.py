'''
Created on 2014-07-30

Random utility functions...

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
from utils.signalsmooth import smooth
# internal imports
from geodata.misc import ArgumentError, isEqual


def binedges(bins=None, binedgs=None, limits=None, lcheckVar=True):
  ''' utility function to generate and validate bins and binegdes from either one '''
  # check input
  if bins is None and binedgs is None: raise ArgumentError
  elif bins is not None and binedgs is not None:
    if len(bins)+1 != len(binedgs): raise ArgumentError
  if bins is not None:
    if limits is not None: vmin, vmax = limits
    else: raise ArgumentError
    # expand bins (values refer to center of bins)
    if isinstance(bins,(int,np.integer)):
      if bins == 1: bins = np.asarray(( (vmin+vmax)/2. ,)) 
      else: bins = np.linspace(vmin,vmax,bins)  
    elif isinstance(bins,(tuple,list)) and  0 < len(bins) < 4: 
      bins = np.linspace(*bins)
    elif not isinstance(bins,(list,np.ndarray)): raise TypeError
    if len(bins) == 1: 
      tmpbinedgs = np.asarray((vmin,vmax))
    else:
      hbd = np.diff(bins) / 2. # make sure this is a float!
      tmpbinedgs = np.hstack((bins[0]-hbd[0],bins[1:]-hbd,bins[-1]+hbd[-1])) # assuming even spacing
    if binedgs is None: binedgs = tmpbinedgs # computed from bins
    elif lcheckVar: assert isEqual(binedgs, np.asarray(tmpbinedgs, dtype=binedgs.dtype))
  if binedgs is not None:
    # expand bin edges
    if not isinstance(binedgs,(tuple,list)): binedgs = np.asarray(binedgs)
    elif not isinstance(binedgs,np.ndarray): raise TypeError  
    tmpbins = binedgs[1:] - ( np.diff(binedgs) / 2. ) # make sure this is a float!
    if bins is None: bins = tmpbins # compute from binedgs
    elif lcheckVar: assert isEqual(bins, np.asarray(tmpbins, dtype=bins.dtype))
  # return bins and binegdes
  return bins, binedgs


# histogram wrapper that suppresses additional output
def histogram(a, bins=10, range=None, weights=None, density=None): 
  ''' histogram wrapper that suppresses bin edge output, but is otherwise the same '''
  return np.histogram(a, bins=bins, range=range, weights=weights, density=density)[0]


# function to subtract the mean and divide by the standard deviation, i.e. standardize
def standardize(var, axis=None, lcopy=True, lsmooth=False, **kwargs):
  ''' subtract mean, divide by standard deviation, and optionally smooth time series; key word arguments are passed on to smoothing function '''
  if lcopy: 
    if not isinstance(var,np.ndarray): raise NotImplementedError # too many checks
    var = var.copy() # make copy - not in-place!
  var -= var.mean(axis=axis, keepdims=True)
  var /= var.std(axis=axis, keepdims=True)
  if lsmooth:
    var = smooth(var, **kwargs)
  return var

# function to detrend a time-series
def detrend(var, **kwargs): raise NotImplementedError, "Detrending is not implemented yet..."

# function to smooth a vector (numpy array): moving mean, nothing fancy
def movingMean(x,i):
  ''' smooth a vector (x, numpy array) using a moving mean of window width 2*i+1 '''
  if x.ndim > 1: raise ValueError
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
  ''' traverse nested lists recursively and perform the operation fct on the end members '''
  # traverse nested lists recursively
  if isinstance(lsl, list):
    return [traverseList(lsl[i], fct) for i in range(len(lsl))]
  # break recursion and apply function using the list element as argument 
  else: return fct(lsl)

  
