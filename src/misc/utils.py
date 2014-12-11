'''
Created on 2014-07-30

Random utility functions...

@author: Andre R. Erler, GPL v3
'''

import numpy as np
from misc.signalsmooth import smooth

# function to subtract the mean and divide by the standard deviation, i.e. standardize
def standardize(var, lsmooth=False, **kwargs):
  ''' subtract mean, divide by standard deviation, and optionally smooth time series; key word arguments are passed on to smoothing function '''
  if not isinstance(var,np.ndarray): raise NotImplementedError
  var = var.copy() # make copy - not in-place!
  var -= var.mean()
  var /= var.std()
  if lsmooth:
    var = smooth(var, **kwargs)
  return var


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

  
