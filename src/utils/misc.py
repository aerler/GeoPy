'''
Created on 2014-07-30

Random utility functions...

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import scipy.linalg as la
from utils.signalsmooth import smooth
from collections import namedtuple
# internal imports
from geodata.misc import ArgumentError, isEqual


# create a named tuple instance on the fly from dictionary
def namedTuple(typename=None, field_names=None, verbose=False, rename=False, **kwargs):
  ''' a wrapper for namedtuple that can create the class on the fly from a dict '''
  if typename is None: typename = 'NamedTuple'
  if field_names is None: field_names = kwargs.keys()
  # create namedtuple class
  NT = namedtuple(typename, field_names, verbose=verbose, rename=rename)
  # create namedtuple instance and populate with values from kwargs
  nt = NT(**kwargs) 
  # return tuple instance 
  return nt

# convert Python types to Numpy scalar types
def toNumpyScalar(num, dtype=None):
  ''' convert a Python number to an equivalent Numpy scalar type '''
  if isinstance(dtype,np.dtype): 
    num = dtype.type(num)
  else:  
    if isinstance(num, float): num = np.float64(num)
    elif isinstance(num, int): num = np.int64(num)
    elif isinstance(num, bool): num = np.bool8(num)
    else: raise NotImplementedError, num
  return num

# helper function to form inner and outer product of multiple lists
def expandArgumentList(expand_list=None, lproduct='outer', **kwargs):
  ''' A function that generates a list of complete argument dict's, based on given kwargs and certain 
      expansion rules: kwargs listed in expand_list are expanded and distributed element-wise, 
      either as inner or outer product, while other kwargs are repeated in every argument dict. '''
  # get load_list arguments
  expand_list = [el for el in expand_list if el in kwargs] # remove missing entries
  expand_dict = {el:kwargs[el] for el in expand_list}
  for el in expand_list: del kwargs[el]
  for el in expand_list: # check types 
    if not isinstance(expand_dict[el], (list,tuple)): 
      raise TypeError    
  ## identify expansion arguments
  if lproduct.lower() == 'inner':
    # inner product: essentially no expansion
    lst0 = expand_dict[expand_list[0]]; lstlen = len(lst0) 
    for el in expand_list: # check length
      if len(expand_dict[el]) == 1: 
        expand_dict[el] = expand_dict[el]*lstlen # broadcast singleton list
      elif len(expand_dict[el]) != lstlen: 
        raise TypeError, 'Lists have to be of same length to form inner product!'
    list_dict = expand_dict
  elif lproduct.lower() == 'outer':
    lstlen = 1
    for el in expand_list:
      lstlen *= len(expand_dict[el])
    ## define function for recursion 
    # basically, loop over each list independently
    def loop_recursion(*args, **kwargs):
      ''' handle any number of loop variables recursively '''
      # interpete arguments
      if len(args) == 1:
        # initialize dictionary of lists (only first recursion level)
        loop_list = args[0][:] # use copy, since it will be decimated 
        list_dict = {key:list() for key in kwargs.iterkeys()}
      elif len(args) == 2:
        loop_list = args[0][:] # use copy of list, to avoid interference with other branches
        list_dict = args[1] # this is not a copy: all branches append to the same lists!
      # handle loops
      if len(loop_list) > 0:
        # initiate a new recursion layer and a new loop
        arg_name = loop_list.pop(0)
        for arg in kwargs[arg_name]:
          kwargs[arg_name] = arg # just overwrite
          # new recursion branch
          list_dict = loop_recursion(loop_list, list_dict, **kwargs)
      else:
        # terminate recursion branch
        for key,value in kwargs.iteritems():
          list_dict[key].append(value)
      # return results 
      return list_dict
    # execute recursive function    
    list_dict = loop_recursion(expand_list, **expand_dict) # use copy of 
    assert all(key in expand_dict for key in list_dict.iterkeys()) 
    assert all(len(list_dict[el])==lstlen for el in expand_list) # check length    
    assert all(len(ld)==lstlen for ld in list_dict.itervalues()) # check length     
  else: raise ArgumentError
  ## generate list of argument dicts
  arg_dicts = []
  for n in xrange(lstlen):
    # assemble arguments
    lstargs = {key:lst[n] for key,lst in list_dict.iteritems()}
    arg_dict = kwargs.copy(); arg_dict.update(lstargs)
    arg_dicts.append(arg_dict)    
  # return list of arguments
  return arg_dicts


# convenience function to evaluate a list of DistVar's
def evalDistVars(varlist, bins=None, support=None, method='pdf', ldatasetLink=True, bootstrap_axis='bootstrap'):
  ''' Convenience function to evaluate a list of DistVars on a given support/bins;
      leaves other Variables untouched. '''
  from geodata.stats import DistVar, VarKDE, VarRV # avoid circular import
  # evaluate distribution variables on support/bins
  if support is not None or bins is not None:
    # find support/bins
    if support is not None and bins is not None: raise ArgumentError
    if support is None and bins is not None: support = bins
    # check variables and evaluate
    if bootstrap_axis is not None: slc = {bootstrap_axis:0}
    newlist = []
    for var in varlist:
      if var is None: new = None
      else: 
        # remove bootstrap axis
        if bootstrap_axis is not None and var.hasAxis(bootstrap_axis): var = var(**slc)
        # evluate distributions
        if isinstance(var,(DistVar,VarKDE,VarRV)): 
          new = getattr(var,method)(support=support) # evaluate DistVar
          #if ldatasetLink: new.dataset= var.dataset # preserve dataset links to construct references
        else: new = var
      newlist.append(new)
    assert not any(isinstance(var,(DistVar,VarKDE,VarRV)) for var in newlist)
  else: newlist = varlist # do nothing
  # return list of variables (with not DistVars)
  return newlist
  

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


# function to perform PCA
def PCA(data, degree=None, lprewhiten=False, lpostwhiten=False, lEOF=False, lfeedback=False):
  ''' A function to perform principal component analysis and return the time-series of the leading EOF's. '''
  data = np.asarray(data)
  if not data.ndim == 2: raise ArgumentError
  # pre-whiten features
  if lprewhiten:
    data -= data.mean(axis=0, keepdims=True)
    data /= data.std(axis=0, keepdims=True)
  # compute PCA
  R = np.cov(data.transpose()) # covariance matrix
  eig, eof = la.eigh(R) # eigenvalues, eigenvectors (of symmetric matrix)
  ieig = np.argsort(eig,)[::-1] # sort in descending order
  eig = eig[ieig]; eof = eof[:,ieig]
  eig /= eig.sum() # normalize by total variance
  # truncate EOF's
  if degree is not None:
      eig = eig[:degree]; eof = eof[:,:degree]
  # generate report/feedback
  if lfeedback:
    string = "Variance explained by {:s} PCA's: {:s}; total variance explained: {:2.0f}%"
    eiglist = ', '.join('{:.0f}%'.format(e*100.) for e in eig)
    dgrstr = 'all' if degree is None else "{:d} leading".format(degree)
    print(string.format(dgrstr, eiglist, eig.sum()*100.))
  # project data onto (leading) EOF's
  pca = np.dot(data,eof) # inverse order, because the are transposed
  # post-whiten features
  if lpostwhiten:
    pca -= pca.mean(axis=0, keepdims=True)
    pca /= pca.std(axis=0, keepdims=True)
  # return results
  if lEOF: return pca, eig, eof
  else: return pca, eig  

# histogram wrapper that suppresses additional output
def histogram(a, bins=10, range=None, weights=None, density=None): 
  ''' histogram wrapper that suppresses bin edge output, but is otherwise the same '''
  return np.histogram(a, bins=bins, range=range, weights=weights, density=density)[0]

# percentile wrapper that casts the output into a single array
def percentile(a, q, axis=None, interpolation='linear', keepdims=False): 
  ''' percentile wrapper that casts the output into a single array, but is otherwise the same '''
  # in this version 'interpolation' and 'keepdims' are not yet supported
  parr = np.asarray(np.percentile(a, q, axis=axis, out=None, overwrite_input=False))
  parr = np.rollaxis(parr, axis=0, start=parr.ndim) # move percentile axis to the back
  return parr

# function to subtract the mean and divide by the standard deviation, i.e. standardize
def standardize(var, axis=None, lcopy=True, **kwargs):
  ''' subtract mean, divide by standard deviation, and optionally smooth time series; key word arguments are passed on to smoothing function '''
  if not isinstance(var,np.ndarray): raise NotImplementedError # too many checks
  if lcopy: var = var.copy() # make copy - not in-place!
  # compute standardized variable
  var -= var.mean(axis=axis, keepdims=True)
  var /= var.std(axis=axis, keepdims=True)
  return var

# function to detrend a time-series
def detrend(var, ax=None, lcopy=True, ldetrend=True, degree=1, rcond=None, w=None,  lsmooth=False, window_len=11, window='hanning'): 
  ''' subtract a linear trend from a time-series array '''
  # check input
  if not isinstance(var,np.ndarray): raise NotImplementedError # too many checks
  if lcopy: var = var.copy() # make copy - not in-place!
  # fit over entire array (usually not what we want...)
  if ax is None and ldetrend: ax = np.arange(var.size) # make dummy axis, if necessary
  if var.ndim != 1:
    shape = var.shape 
    var = var.ravel() # flatten array, if necessary
  else: shape = None
  # apply optional detrending
  if ldetrend:
    # fit linear trend
    trend = np.polyfit(ax, var, deg=degree, rcond=rcond, w=w, full=False, cov=False)
    # evaluate and subtract linear trend
    var -= np.polyval(trend, ax) # residuals
  # apply optional smoothing
  if lsmooth: var = smooth(var, window_len=window_len, window=window)  
  # return detrended and/or smoothed time-series
  if shape is not None: var = var.reshape(shape)
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

  
