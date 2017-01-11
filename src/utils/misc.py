'''
Created on 2014-07-30

Random utility functions...

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import scipy.linalg as la
from utils.signalsmooth import smooth
import collections as col
# internal imports
from geodata.misc import ArgumentError, isEqual, AxisError

## a method to tabulate variables (adapted from Variable) 
def tabulate(data, row_idx=0, col_idx=1, header=None, labels=None, cell_str='{}', cell_idx=None, cell_fct=None, 
             lflatten=False, mode='mylatex', filename=None, folder=None, lfeedback=True, **kwargs):
  ''' Create a nicely formatted table in the selected format ('mylatex' or call tabulate); 
      cell_str controls formatting of each cell, and also supports multiple arguments along 
      an axis. lflatten skips cell axis checking and lumps all remaining axes together. '''
  # check input
  if not isinstance(data, np.ndarray): raise TypeError()
  if cell_idx is not None:
    if not data.ndim == 3: raise AxisError()
  elif lflatten:
    if not data.ndim >= 2: raise AxisError()
  elif not data.ndim == 2: raise AxisError()
  if not isinstance(cell_str,basestring): raise TypeError(cell_str)
  if cell_fct: 
    if not callable(cell_fct): raise TypeError(cell_fct)
    lcellfct = True
  else: lcellfct = False
  if cell_idx >= data.ndim: raise AxisError(cell_idx)
  collen = data.shape[col_idx]; rowlen = data.shape[row_idx] 
  if row_idx < col_idx: col_idx -= 1 # this is a shortcut for later (data gets sliced by row)
  llabel = False; lheader = False
  if labels: 
    if len(labels) != rowlen: raise AxisError(data.shape)
    llabel = True 
  if header: 
    if llabel:
      if len(header) == collen: header = ('',) + tuple(header)
      elif not len(header) == collen+1: raise AxisError(header)
    elif not len(header) == collen: raise AxisError(header)
    lheader = True
  ## assemble table in nested list
  table = [] # list of rows
  if lheader: table.append(header) # first row
  # loop over rows
  for i in xrange(rowlen):
    row = [labels[i]] if labels else []
    rowdata = data.take(i, axis=row_idx)
    # loop over columns
    for j in xrange(collen):
      celldata = rowdata.take(j, axis=col_idx)
      # pass data to string of function
      if isinstance(celldata, np.ndarray):
        if lflatten: celldata = celldata.ravel() 
        elif celldata.ndim > 1: raise AxisError(celldata.shape)
        cell = cell_fct(celldata) if lcellfct else cell_str.format(*celldata) 
      else: 
        cell = cell_fct(celldata) if lcellfct else cell_str.format(celldata)
      # N.B.: cell_fct also has to return a string
      row.append(cell)
    table.append(row)
  ## now make table   
  if mode.lower() == 'mylatex':
    # extract settings 
    lhline = kwargs.pop('lhline', True)
    lheaderhline = kwargs.pop('lheaderhline', True)
    cell_del = kwargs.pop('cell_del','  &  ') # regular column delimiter      
    line_brk = kwargs.pop('line_break',' \\\\ \\hline' if lhline else ' \\\\') # escape backslash
    tab_begin = kwargs.pop('tab_begin','') # by default, no tab environment
    tab_end = kwargs.pop('tab_end','') # by default, no tab environment
    extra_hline = kwargs.pop('extra_hline', []) # row_idx or label with extra \hline command
    # align cells
    nrow = rowlen+1 if lheader else rowlen
    ncol = collen+1 if llabel else collen
    col_fmts = [] # column width
    for j in xrange(ncol):
      wd = 0
      for i in xrange(nrow): wd = max(wd,len(table[i][j]))
      col_fmts.append('{{:^{:d}s}}'.format(wd))
    # assemble table string
    string = tab_begin + '\n' if tab_begin else '' # initialize
    for i,row in enumerate(table):
      row = [fmt_str.format(cell) for fmt_str,cell in zip(col_fmts,row)]
      string += (' '+row[0]) # first cell
      for cell in row[1:]: string += (cell_del+cell)
      string += line_brk # add latex line break
      if i in extra_hline or (llabel and row[0] in extra_hline): string += ' \\hline'
      if lheaderhline and i == 0: string += ' \\hline' # always put one behind the header
      string += '\n' # add actual line break 
    if tab_end: string += (tab_end+'\n') 
  else:
    # use the tabulate module (it's not standard, so import only when needed)
    from tabulate import tabulate 
    string = tabulate(table, tablefmt=mode, **kwargs)
    # headers, floatfmt, numalign, stralign, missingval
  ## write to file
  if filename:
    if folder: filename = folder+'/'+filename
    f = open(filename, mode='w')
    f.write(string) # write entire string and nothing else
    f.close()
    if lfeedback: print(filename)
  # return string for printing
  return string


# wrapper for namedtuple that supports defaults
def defaultNamedtuple(typename, field_names, defaults=None):
  ''' wrapper for namedtuple that supports defaults; adapted from stackoverflow:
      https://stackoverflow.com/questions/11351032/named-tuple-and-optional-keyword-arguments ''' 
  T = col.namedtuple(typename, field_names) # make named tuple
  T.__new__.__defaults__ = (None,) * len(T._fields) # set defaults to None
  # add custom defaults
  if defaults is not None:
    if isinstance(defaults, col.Mapping):
        prototype = T(**defaults)
    elif isinstance(defaults, col.Iterable):
        prototype = T(*defaults)
    else: raise ArgumentError(str(defaults))
    T.__new__.__defaults__ = tuple(prototype)
#     # add self-referenc defaults
#     if ref_prefix:
#       l = len(ref_prefix)
#       for field,value in T._asdict().iteritems():
#         if isinstance(value,basestring) and value[:l] == ref_prefix:
#           T.__dict__[field] = T.__dict__[value[l:]]
#     # N.B.: this would have to go into the constructor in order to work...
  # return namedtuple with defaults
  return T
  
# create a named tuple instance on the fly from dictionary
def namedTuple(typename=None, field_names=None, verbose=False, rename=False, **kwargs):
  ''' a wrapper for namedtuple that can create the class on the fly from a dict '''
  if typename is None: typename = 'NamedTuple'
  if field_names is None: field_names = kwargs.keys()
  # create namedtuple class
  NT = col.namedtuple(typename, field_names, verbose=verbose, rename=rename)
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
    else: raise NotImplementedError(num)
  return num

# transform an n-dim array to a 2-dim array by collapsing all but the last/innermost dimension
def collapseOuterDims(ndarray, axis=None, laddOuter=True):
  ''' transform an n-dim array to a 2-dim array by collapsing all but the last/innermost dimension '''
  if not isinstance(ndarray, np.ndarray): raise TypeError(ndarray)
  if ndarray.ndim <2:
    if laddOuter: ndarray.reshape((1,ndarray.size))
    else: raise AxisError(ndarray.shape)
  if axis is not None and not (axis == -1 or axis == ndarray.ndim-1):
    if not isinstance(axis,(int,np.integer)): raise TypeError(axis)
    ndarray = np.rollaxis(ndarray, axis=axis, start=ndarray.ndim) # make desired axis innermost axis
  shape = (np.prod(ndarray.shape[:-1]), ndarray.shape[-1]) # new 2D shape
  ndarray = np.reshape(ndarray, shape) # just a new view
  return ndarray # return reshaped (and reordered) array

# apply an operation on a list of 1D arrays over a selected axis and loop over all others (in all arrays)
def apply_over_arrays(fct, *arrays, **kwargs):
  ''' similar to apply_along_axis, but operates on a list of ndarray's and is not parallelized '''
  axis = kwargs.pop('axis',-1)
  lexitcode = kwargs.pop('lexitcode',False)
  lout = 'out' in kwargs # output array (for some numpy functions)
  # pre-process input (get reshaped views)
  arrays = [collapseOuterDims(array, axis=axis, laddOuter=True) for array in arrays]
  ie = arrays[0].shape[0]
  if not all(array.shape[0]==ie for array in arrays): 
    raise AxisError("Cannot coerce input arrays into compatible shapes.")
  # special handling of output arrays
  if lout: 
    out = collapseOuterDims(kwargs['out'], axis=axis, laddOuter=True)
    if out.shape[0] != ie: raise AxisError("Output array has incompatible shape.")
  # loop over outer dimension and apply function
  if lexitcode: ecs = [] # exit code
  for i in xrange(ie):
    arrslc = [array[i,:] for array in arrays]
    if lout: kwargs['out'] = out[i,:]
    ec = fct(*arrslc, **kwargs)
    if lexitcode: ecs.append(ec)
  #if lexitcode and not all(ecs): raise AssertionError("Some function executions were not successful!")
  # return output list (or None's if fct has no exit code)
  return ecs if lexitcode else None

## define function for recursion 
# basically, loop over each list independently
def _loop_recursion(*args, **kwargs):
  ''' handle any number of loop variables recursively '''
  # interpete arguments (kw-expansion is necessary)
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
      list_dict = _loop_recursion(loop_list, list_dict, **kwargs)
  else:
    # terminate recursion branch
    for key,value in kwargs.iteritems():
      list_dict[key].append(value)
  # return results 
  return list_dict

# helper function to check lists
def _prepareList(exp_list, kwargs):
  ''' helper function to clean list elements '''
  # get exp_list arguments
  exp_list = [el for el in exp_list if el in kwargs] # remove missing entries
  exp_dict = {el:kwargs[el] for el in exp_list}
  for el in exp_list: del kwargs[el]
  for el in exp_list: # check types 
    if not isinstance(exp_dict[el], (list,tuple)): raise TypeError(el)
  return exp_list, exp_dict

# helper function to form inner and outer product of multiple lists
def expandArgumentList(inner_list=None, outer_list=None, expand_list=None, lproduct='outer', **kwargs):
  ''' A function that generates a list of complete argument dict's, based on given kwargs and certain 
      expansion rules: kwargs listed in expand_list are expanded and distributed element-wise, 
      either as inner ('inner_list') or outer ('outer_list') product, while other kwargs are repeated 
      in every argument dict. 
      Arguments can be expanded simultaneously (in parallel) within an outer product by specifying
      them as a tuple within the outer product argument list ('outer_list'). '''
  if not (expand_list or inner_list or outer_list): 
    arg_dicts = [kwargs] # return immediately - nothing to do
  else:
      
    # handle legacy arguments
    if expand_list is not None:
      if inner_list is not None or outer_list is not None: raise ArgumentError("Can not mix input modes!")
      if lproduct.lower() == 'inner': inner_list = expand_list
      elif lproduct.lower() == 'outer': outer_list = expand_list
      else: raise ArgumentError(lproduct)
    outer_list = outer_list or []; inner_list = inner_list or []
      
    # handle outer product expansion first
    if len(outer_list) > 0:
      kwtmp = {key:value for key,value in kwargs.iteritems() if key not in inner_list}
      
      # detect variables for parallel expansion
      # N.B.: parallel outer expansion is handled by replacing the arguments in each parallel expansion group
      #       with a single (fake) argument that is a tuple of the original argument values; the tuple is then,
      #       after expansion, disassembled into its former constituent arguments
      par_dict = dict()
      for kw in outer_list:
        if isinstance(kw,(tuple,list)):
          # retrieve parallel expansion group 
          par_args = [kwtmp.pop(name) for name in kw]
          if not all([len(args) == len(par_args[0]) for args in par_args]): 
            raise ArgumentError("Lists for parallel expansion arguments have to be of same length!")
          # introduce fake argument and save record
          fake = 'TMP_'+'_'.join(kw)+'_{:d}'.format(len(kw)) # long name that is unlikely to interfere...
          par_dict[fake] = kw # store record of parallel expansion for reassembly later
          kwtmp[fake] = zip(*par_args) # transpose lists to get a list of tuples                      
        elif not isinstance(kw,basestring): raise TypeError(kw)
      # replace entries in outer list
      if len(par_dict)>0:
        outer_list = outer_list[:] # copy list
        for fake,names in par_dict.iteritems():
          if names in outer_list:
            outer_list[outer_list.index(names)] = fake
      assert all([ isinstance(arg,basestring) for arg in outer_list])
      
      outer_list, outer_dict = _prepareList(outer_list, kwtmp)
      lstlen = 1
      for el in outer_list:
        lstlen *= len(outer_dict[el])
      # execute recursive function for outer product expansion    
      list_dict = _loop_recursion(outer_list, **outer_dict) # use copy of
      # N.B.: returns a dictionary where all kwargs have been expanded to lists of appropriate length
      assert all(key in outer_dict for key in list_dict.iterkeys()) 
      assert all(len(list_dict[el])==lstlen for el in outer_list) # check length    
      assert all(len(ld)==lstlen for ld in list_dict.itervalues()) # check length  
      
      # disassemble parallel expansion tuple and reassemble as individual arguments
      if len(par_dict)>0:
        for fake,names in par_dict.iteritems():
          assert fake in list_dict
          par_args = zip(*list_dict.pop(fake)) # transpose, to get an expanded tuple for each argument
          assert len(par_args) == len(names) 
          for name,args in zip(names,par_args): list_dict[name] = args
         
    # handle inner product expansion last
    if len(inner_list) > 0:
      kwtmp = kwargs.copy()
      if len(outer_list) > 0: 
        kwtmp.update(list_dict)
        inner_list = outer_list + inner_list
      # N.B.: this replaces all outer expansion arguments with lists of appropriate length for inner expansion
      inner_list, inner_dict = _prepareList(inner_list, kwtmp)
      # inner product: essentially no expansion
      lst0 = inner_dict[inner_list[0]]; lstlen = len(lst0) 
      for el in inner_list: # check length
        if len(inner_dict[el]) == 1: 
          inner_dict[el] = inner_dict[el]*lstlen # broadcast singleton list
        elif len(inner_dict[el]) != lstlen: 
          raise TypeError('Lists have to be of same length to form inner product!')
      list_dict = inner_dict
      
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
    if len(bins)+1 != len(binedgs): raise ArgumentError(len(bins))
  if bins is not None:
    if limits is not None: vmin, vmax = limits
    else: raise ArgumentError(bins)
    # expand bins (values refer to center of bins)
    if isinstance(bins,(int,np.integer)):
      if bins == 1: bins = np.asarray(( (vmin+vmax)/2. ,)) 
      else: bins = np.linspace(vmin,vmax,bins)  
    elif isinstance(bins,(tuple,list)) and  0 < len(bins) < 4: 
      bins = np.linspace(*bins)
    elif not isinstance(bins,(list,np.ndarray)): raise TypeError(bins)
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
    elif not isinstance(binedgs,np.ndarray): raise TypeError(binedgs)
    tmpbins = binedgs[1:] - ( np.diff(binedgs) / 2. ) # make sure this is a float!
    if bins is None: bins = tmpbins # compute from binedgs
    elif lcheckVar: assert isEqual(bins, np.asarray(tmpbins, dtype=bins.dtype))
  # return bins and binegdes
  return bins, binedgs


# function to perform PCA
def PCA(data, degree=None, lprewhiten=False, lpostwhiten=False, lEOF=False, lfeedback=False):
  ''' A function to perform principal component analysis and return the time-series of the leading EOF's. '''
  data = np.asarray(data)
  if not data.ndim == 2: raise ArgumentError(data.ndim)
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
  ''' histogram wrapper that suppresses bin edge output and handles NaN inputs'''
  # make sure NaNs don't cause range errors
  # if all is NaN, we need to handle the return array manually
  if np.all(np.isnan(a)):
    # determin number of bins
    if isinstance(bins, (list,tuple,np.ndarray)):l = len(bins)-1 # array of bin *edges*
    elif isinstance(bins,(int,np.integer)): l = bins
    else: raise TypeError(bins)
    # make histogram for invalid data... 
    h = np.zeros(l) # counts per bin: nothing
    if density: h *= np.NaN # normalized by integral of nothing...
  else:
    # in all other cases, use standard fct.
    if range is None: range = (np.nanmin(a),np.nanmax(a))
    h = np.histogram(a, bins=bins, range=range, weights=weights, density=density)[0]
  return h

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

# function to reverse or flip along a particular axis
def flip(a, axis=0):
  ''' function to reverse or flip along a particular axis ''' 
  idx = [slice(None)]*a.ndim # construct list of indexing slices
  idx[axis] = slice(None, None, -1) # this one reverses the order
  return a[idx] # apply abd return
  
# function to detrend a time-series
def detrend(var, ax=None, lcopy=True, ldetrend=True, ltrend=False, degree=1, rcond=None, w=None,  
            lsmooth=False, lresidual=False, window_len=11, window='hanning'): 
  ''' subtract a linear trend from a time-series array (operation is in-place) '''
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
  if ldetrend or ltrend:
    # fit linear trend
    trend = np.polyfit(ax, var, deg=degree, rcond=rcond, w=w, full=False, cov=False)
    # evaluate and subtract linear trend
    if ldetrend and ltrend: raise ArgumentError("Can either return trend/polyfit or residuals, not both.")
    elif ldetrend and not ltrend: var -= np.polyval(trend, ax) # residuals
    elif ltrend and not ldetrend: var = np.polyval(trend, ax) # residuals
  # apply optional smoothing
  if lsmooth and lresidual: raise ArgumentError("Can either return smoothed array or residuals, not both.")
  elif lsmooth: var = smooth(var, window_len=window_len, window=window)  
  elif lresidual: var -= smooth(var, window_len=window_len, window=window)
  # return detrended and/or smoothed time-series
  if shape is not None: var = var.reshape(shape)
  return var

# function to smooth a vector (numpy array): moving mean, nothing fancy
def movingMean(x,i):
  ''' smooth a vector (x, numpy array) using a moving mean of window width 2*i+1 '''
  if x.ndim > 1: raise ValueError(x.ndim)
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


# funcion that returns the last n lines of a file, like tail
# adapted from Stack Overflow:
# 'https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-with-python-similar-to-tail' 
def tail(f, n=20):
    ''' Returns the last 'n' lines of file 'f' as a list. '''
    if n == 0:
        return []
    BUFSIZ = 1024
    f.seek(0, 2)
    bytes = f.tell()
    size = n + 1
    block = -1
    data = []
    while size > 0 and bytes > 0:
        if bytes - BUFSIZ > 0:
            # Seek back one whole BUFSIZ
            f.seek(block * BUFSIZ, 2)
            # read BUFFER
            data.insert(0, f.read(BUFSIZ))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            data.insert(0, f.read(bytes))
        linesFound = data[0].count('\n')
        size -= linesFound
        bytes -= BUFSIZ
        block -= 1
    return ''.join(data).splitlines()[-n:]