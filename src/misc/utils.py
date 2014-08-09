'''
Created on 2014-07-30

Random utility functions...

@author: Andre R. Erler, GPL v3
'''

import numpy as np
from misc.signalsmooth import smooth


def standardize(var, lsmooth=False, **kwargs):
  ''' subtract mean, divide by standard deviation, and optionally smooth time series; key word arguments are passed on to smoothing function '''
  if not isinstance(var,np.ndarray): raise NotImplementedError
  var = var.copy() # make copy - not in-place!
  var -= var.mean()
  var /= var.std()
  if lsmooth:
    var = smooth(var, **kwargs)
  return var