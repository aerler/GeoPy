'''
Created on Dec 12, 2014

Some Variable subclasses for handling distribution functions over grid points. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import scipy.stats as ss
# internal imports
from geodata.base import Variable, Axis
from geodata.misc import DataError, ArgumentError, VariableError, AxisError, DistVarError
from plotting.properties import getPlotAtts
from numpy.linalg.linalg import LinAlgError
from processing.multiprocess import apply_along_axis
import functools


# convenience function to generate a DistVar from another Variable object
def asDistVar(var, axis='time', dist='KDE', **kwargs):
  ''' generate a DistVar of type 'dist' from a Variable 'var'; use dimension 'axis' as sample axis '''
  if not isinstance(var,Variable): raise VariableError
  if not var.data: var.load() 
  # prepare input
  iaxis=var.axisIndex(axis)
  axes = var.axes[:iaxis]+var.axes[iaxis+1:] # i.e. without axis/iaxis
  units = var.units # these will be the sample units, not the distribution units 
  # choose distribution
  if dist.lower() == 'kde': dvar = VarKDE(samples=var.data_array, units=units, axis=iaxis, axes=axes, **kwargs)
  else: dvar = VarRV(dist=dist, samples=var.data_array, units=units, axis=iaxis, axes=axes, **kwargs)
  return dvar

# base class for distributions 
class DistVar(Variable):
  '''
  A base class for variables that represent a distribution at each grid point. This class implements
  access to (re-)samplings of the distribution, as well as histogram- and CDF-type grid-based 
  representations of the distributions. All operations add an additional (innermost) dimension;
  representations of the distribution require a support vector, sampling only a sample size.. 
  '''
  paramAxis = None # axis for distribution parameters (None is distribution objects are stored)

  def __init__(self, name=None, units=None, axes=None, samples=None, params=None, axis=None, dtype=None, 
               masked=None, mask=None, fillValue=None, atts=None, ic=None, ldebug=False, **kwargs):
    '''
    This method creates a new DisVar instance from data and parameters. If data is provided, a sample
    axis has to be specified or the last (innermost) axis is assumed to be the sample axis.
    An estimation/fit will be performed at every grid point and stored in an array.
    Note that 'dtype' and 'units' refer to the sample data, not the distribution.
    '''
    # if parameters are provided
    if params is not None:
      if samples is not None: raise ArgumentError 
      if isinstance(params,np.ndarray):
        params = np.asarray(params, dtype=dtype, order='C')
      if axis is not None and not axis == params.ndim-1:
        params = np.rollaxis(params, axis=axis, start=params.ndim) # roll sample axis to last (innermost) position
      if dtype is None: dtype = np.dtype('float') # default sample dtype
      if len(axes) != params.ndim: raise AxisError
      if masked is None:
        if isinstance(samples, ma.MaskedArray): masked = True
        else: masked = False  
    # if samples are provided
    if samples is not None:
      if params is not None: raise ArgumentError 
      # ensure data type
      if dtype is None: dtype = samples.dtype
      elif not np.issubdtype(samples.dtype,dtype): raise DataError
      # handle masks
      if isinstance(samples, ma.MaskedArray):
        if masked is None: masked = True
        if fillValue is None: fillValue = samples.fill_value 
        if np.issubdtype(samples.dtype,np.integer): # recast as floats, so we can use np.NaN 
          samples = np.asarray(samples, dtype='float') 
        samples = samples.filled(np.NaN)
      else: masked = False
      # make sure sample axis is last
      if isinstance(samples,np.ndarray): 
        samples = np.asarray(samples, dtype=dtype, order='C')
      if axis is not None and not axis == samples.ndim-1:
        samples = np.rollaxis(samples, axis=axis, start=samples.ndim) # roll sample axis to last (innermost) position
        # N.B.: we may also need some NaN/masked-value handling here...
      # check axes
      if len(axes) == samples.ndim: axes = axes[:axis]+axes[axis+1:]
      elif len(axes) != samples.ndim-1: raise AxisError
      # estimate distribution parameters
      # N.B.: the method estimate() should be implemented by specific child classes
      params = self._estimate_distribution(samples, ic=None, ldebug=ldebug, **kwargs)
      # N.B.: 'ic' are initial guesses for parameter values; 'kwargs' are for the estimator algorithm 
    # sample fillValue
    if fillValue is None: fillValue = np.NaN
    # mask invalid or None parameters
#     if masked: 
#       if np.issubdtype(params.dtype,np.inexact): invalid = np.NaN
#       else: invalid = None # "null object"
#       params = ma.masked_equal(params, invalid) # mask invalid values 
#       params.set_fill_value = invalid # fill value to use later       
    # generate new parameter axis
    if params.ndim == len(axes)+1:
      patts = dict(name='params',units='',long_name='Distribution Parameters')
      paramAxis = Axis(coord=np.arange(params.shape[-1]), atts=patts)
      axes = axes + (paramAxis,)
    else: paramAxis = None
    # check diemnsions, but leave dtype open to be inferred from data
    assert params.ndim == len(axes)
    assert all(ps==len(ax) for ps,ax in zip(params.shape,axes))
    # create variable object using parent constructor
    super(DistVar,self).__init__(name=name, units=units, axes=axes, data=params, dtype=None, mask=mask, 
                                 fillValue=fillValue, atts=atts, plot=None)
    # reset dtype to sample dtype (not parameter dtype!)
    self.masked = masked
    self.fillValue = fillValue # was overwritten by parameter array fill_value
    assert self.masked == masked
    self.dtype = dtype # property is overloaded in DistVar
    # N.B.: in this variable dtype and units refer to the sample data, not the distribution!
    self.paramAxis = paramAxis 
    
  @property
  def dtype(self):
    ''' The data type of the samlple data (inferred from initialization data). '''
    return self._dtype   
  @dtype.setter
  def dtype(self, dtype):
    self._dtype = dtype
  
  @property
  def masked(self):
    ''' A flag indicating if the data is masked. '''    
    return self._masked
  @masked.setter
  def masked(self, masked):
    self._masked = masked
  
  @property
  def fillValue(self):
    ''' The fillValue for masks (stored in the atts dictionary). '''
    fillValue = self.atts.get('fillValue',None)
    return fillValue
  @fillValue.setter
  def fillValue(self, fillValue):
    self.atts['fillValue'] = fillValue
  
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, ic=None, ldebug=False, **kwargs):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    raise NotImplementedError
  # distribution-specific method; should be overloaded by subclass
  def _sample_distribution(self, n):
    ''' draw n samples from the distribution for each grid point and return as ndarray '''
    raise NotImplementedError
  # distribution-specific method; should be overloaded by subclass
  def _density_distribution(self, support):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    raise NotImplementedError
  # distribution-specific method; should be overloaded by subclass
  def _cumulative_distribution(self, support):
    ''' compute CDF at given support points for each grid point and return as ndarray '''
    raise NotImplementedError

  # overload histogram and return a PDF (like a histogram)
  def _get_dist(self, dist_op, dist_type, support=None, moments=None, support_axis=None, axis_idx=None, 
                asVar=True, name=None, axatts=None, varatts=None, **kwargs):
    ''' return a distribution function as a Variable object or ndarray '''
    if not asVar and support_axis is not None: raise ArgumentError
    if axis_idx is None: 
      if self.paramAxis is None: axis_idx = self.ndim 
      else: axis_idx = self.ndim-1
    elif not axis_idx <= self.ndim: raise ArgumentError
    # expand support
    if support is not None:
      if support_axis is not None: raise ArgumentError
      if isinstance(support,(int,np.integer)): raise DistVarError
      elif isinstance(support,(tuple,list)):
        if 0 < len(support) < 4: support = np.linspace(*support)
        else: support = np.asarray(support)
    elif support_axis is not None:
      if not isinstance(support_axis, Axis): raise TypeError
      support = support_axis.coord 
    # get distribution/stats values
    if support is not None:
      if not isinstance(support,np.ndarray): raise TypeError
      dist_data = dist_op(support, **kwargs)
    elif moments is not None:
      if isinstance(moments, (basestring)):
        dist_data = dist_op(moments=moments, **kwargs)
      elif isinstance(moments, (int,np.integer)):
        dist_data = dist_op(n=moments, **kwargs)
      else: raise TypeError
    else: 
      dist_data = dist_op(**kwargs)
    # test if squeeze works
    if dist_data.shape[-1] == 1:
      dist_data = dist_data.squeeze()
      lsqueezed = True
    else: lsqueezed = False
    # handle masked values
    if self.masked: 
      dist_data = ma.masked_invalid(dist_data, copy=False) 
      # N.B.: comparisons with NaN always evaluate to False!
      if self.fillValue is not None:
        dist_data = ma.masked_equal(dist_data, self.fillValue)
      ma.set_fill_value(dist_data,self.fillValue)      
    # arrange axes
    if axis_idx != self.ndim: 
      dist_data = np.rollaxis(dist_data, axis=dist_data.ndim-1,start=axis_idx)
    # setup histogram axis and variable attributes (special case)
    if asVar:
      basename = self.name.split('_')[0]
      if not lsqueezed:
        if support_axis is None:
          # generate new histogram axis
          daxatts = self.atts.copy() # variable values become axis
          daxatts['name'] = '{:s}_bins'.format(basename) # remove suffixes (like _dist)
          daxatts['units'] = self.units # the histogram axis has the same units as the variable
          daxatts['long_name'] = '{:s} Axis'.format(self.atts.get('long_name',self.name.title()))    
          if axatts is not None: daxatts.update(axatts)
          support_axis = Axis(coord=support, atts=daxatts)
        else:
          if axatts is not None: support_axis.atts.update(axatts) # just update attributes
      # generate new histogram variable
      plotatts = getPlotAtts(name=dist_type, units='') # infer meta data from plot attributes
      dvaratts = self.atts.copy() # this is either density or frequency
      dvaratts['name'] = name or '{:s}_{:s}'.format(basename,dist_type)
      dvaratts['long_name'] = '{:s} of {:s}'.format(plotatts.name,self.atts.get('long_name',basename.title()))
      dvaratts['units'] = plotatts.units
      if varatts is not None: dvaratts.update(varatts)
      # create new variable
      if self.paramAxis is None: axes = self.axes
      else:
        assert self.axisIndex(self.paramAxis, lcheck=True) == self.ndim-1 
        axes = self.axes[:-1]
      if lsqueezed: axes = axes[:axis_idx]+axes[axis_idx:] # remove support if singleton
      else: axes = axes[:axis_idx]+(support_axis,)+axes[axis_idx:] # skip last
      dvar = Variable(data=dist_data, axes=axes, atts=dvaratts, plot=plotatts)
    else:
      dvar = dist_data
    # return results
    return dvar
  
  # overload histogram and return a PDF (like a histogram)
  def histogram(self, bins=None, asVar=True, name=None, support=None, support_axis=None, axis_idx=None, 
                haxatts=None, hvaratts=None, ldensity=True, lflatten=False, fillValue=None, **kwargs):
    ''' return the probability density function '''
    if not ldensity: raise DistVarError
    if lflatten: raise DistVarError
    if not asVar and support_axis is not None: raise ArgumentError
    if axis_idx is None: axis_idx = self.ndim
    elif not axis_idx <= self.ndim: raise ArgumentError
    # expand bins (values refer to center of bins)
    if bins is not None:
      if support is not None: raise ArgumentError
      support = bins
    # get distribution values
    hvar = self._get_dist(self._density_distribution, 'pdf', support=support, support_axis=support_axis, 
                          axis_idx=axis_idx, asVar=asVar, name=name, axatts=haxatts, varatts=hvaratts)
    # return results
    return hvar
  
  # basically an alias for histogram
  def PDF(self, asVar=True, name=None, support=None, support_axis=None, axis_idx=None, 
                axatts=None, varatts=None, lflatten=False, fillValue=None, **kwargs):
    ''' return the probability density function '''
    if lflatten: raise DistVarError
    if not asVar and support_axis is not None: raise ArgumentError
    if axis_idx is None: axis_idx = self.ndim
    elif not axis_idx <= self.ndim: raise ArgumentError
    # get distribution values
    hvar = self._get_dist(self._density_distribution, 'pdf', support=support, support_axis=support_axis, 
                          axis_idx=axis_idx, asVar=asVar, name=name, axatts=axatts, varatts=varatts)
    # return results
    return hvar
  
  # overload histogram and return a PDF (like a histogram)
  def CDF(self, bins=None, asVar=True, name=None, support=None, support_axis=None, axis_idx=None, 
                caxatts=None, cvaratts=None, axatts=None, varatts=None, lnormalize=True, 
                lflatten=False, fillValue=None, **kwargs):
    ''' return the probability density function '''
    if not lnormalize: raise DistVarError
    if lflatten: raise DistVarError
    if not asVar and support_axis is not None: raise ArgumentError
    if axis_idx is None: axis_idx = self.ndim
    elif not axis_idx <= self.ndim: raise ArgumentError
    # expand bins (values refer to center of bins)
    if bins is not None:
      if support is not None: raise ArgumentError
      support = bins
    # expand c*atts
    if axatts is not None and caxatts is not None: raise ArgumentError
    axatts = axatts or caxatts
    if varatts is not None and cvaratts is not None: raise ArgumentError 
    varatts = varatts or cvaratts
    # get distribution values
    hvar = self._get_dist(self._cumulative_distribution, 'cdf', support=support, support_axis=support_axis, 
                          axis_idx=axis_idx, asVar=asVar, name=name, axatts=axatts, varatts=varatts)
    # return results
    return hvar
  
  # a new resampling function
  def resample(self, N=0, asVar=True, name=None, support=None, sample_axis=None, axis_idx=None, 
               axatts=None, varatts=None, fillValue=None, **kwargs):
    ''' draw a new sample of length 'N', over 'support', or along a 'sample_axis' and return as new variable '''
    # create dummy support, if necessary 
    if N > 0:
      if support is not None or sample_axis is not None: raise ArgumentError
      support = np.arange(N)
    elif support is None and sample_axis is None: raise ArgumentError  
    # override default variable attributes
    if asVar:
      if sample_axis is None:
        # generate new histogram axis
        saxatts = self.atts.copy() # variable values become axis
        saxatts['name'] = 'sample_axis'.format(self.name) # we don't know the real name
        saxatts['units'] = '' # we don't know the sampling interval...
        saxatts['long_name'] = 'Sampling Axis for '.format(self.atts.get('long_name',self.name.title()))    
        if axatts is not None: saxatts.update(axatts)
        sample_axis = Axis(coord=support, atts=saxatts)
        support = None # avoid conflict with sample axis
      else:
        if support is not None: raise ArgumentError
        sample_axis.atts.update(axatts) # just update attributes
      # generate new histogram variable
      varname = name or self.name.split('_')[0] # use first part of dist varname
      plotatts = getPlotAtts(varname, units=self.units) # infer meta data from plot attributes
      svaratts = self.atts.copy() # this is either density or frequency
      svaratts['name'] = varname
      svaratts['long_name'] = plotatts.name
      svaratts['units'] = self.units
      if varatts is not None: svaratts.update(varatts)
    elif sample_axis is not None: raise ArgumentError
    # get distribution values
    var = self._get_dist(self._resample_distribution, 'resample', support=support, support_axis=sample_axis, 
                         axis_idx=axis_idx, asVar=asVar, name=name, axatts=saxatts, varatts=svaratts)
    # return new variable
    return var


## VarKDE subclass and helper functions

# N.B.: need to define helper functions outside of class definition, otherwise pickle / multiprocessing 
#       will fail... need to fix arguments with functools.partial

# estimate KDE from sample vector
def kde_estimate(sample, ldebug=False, **kwargs):
  if ldebug: 
    if np.all(np.isnan(sample)):
      print('NaN'); res = None
    elif np.all(sample == sample[0]):
      print('equal'); res = None
    elif isinstance(sample,ma.MaskedArray) and np.all(sample.mask):
      print('masked'); res = None
    else:
      try: 
        res = ss.kde.gaussian_kde(sample, **kwargs)
      except LinAlgError:
        print('linalgerr'); res = None
  else:
    if np.all(np.isnan(sample)): res = None
    else: res = ss.kde.gaussian_kde(sample, **kwargs)
  return (res,) # need to return an iterable

# evaluate KDE over a given support
def kde_eval(kde, support=None, n=0, fillValue=np.NaN):
  if kde[0] is None: res = np.zeros(n)+fillValue 
  else: res = kde[0].evaluate(support)
  return res

# resample KDE over a given support
def kde_resample(kde, support=None, n=0, fillValue=np.NaN, dtype=np.float):
  if kde[0] is None: res = np.zeros(n)+fillValue 
  else: res = np.asarray(kde[0].resample(size=n), dtype=dtype).ravel()
  return res

# integrate KDE over a given support (from -inf)
def kde_cdf(kde, support=None, n=0, fillValue=np.NaN):
    if kde[0] is None: res = np.zeros(n)+fillValue
    else: 
      fct = lambda s: kde[0].integrate_box_1d(-1*np.inf,s)
      res = np.asarray([fct(s) for s in support])
    return res
  
# Subclass of DistVar implementing Kernel Density Estimation
class VarKDE(DistVar):
  ''' A subclass of DistVar implementing Kernel Density Estimation (scipy.stats.kde) ''' 
  
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, ic=None, ldebug=False, **kwargs):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    fct = functools.partial(kde_estimate, ldebug=ldebug, **kwargs)
    kernels = apply_along_axis(fct, samples.ndim-1, samples).squeeze()
    assert samples.shape[:-1] == kernels.shape
    # return an array of kernels
    return kernels

  # distribution-specific method; should be overloaded by subclass
  def _density_distribution(self, support):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    n = len(support); fillValue = self.fillValue or np.NaN
    data = self.data_array.reshape(self.data_array.shape+(1,)) # expand
    fct = functools.partial(kde_eval, support=support, n=n, fillValue=fillValue)
    pdf = apply_along_axis(fct, self.ndim, data)
    assert pdf.shape == self.shape + (len(support),)
    return pdf
  
  # distribution-specific method; should be overloaded by subclass
  def _resample_distribution(self, support):
    ''' draw n samples from the distribution for each grid point and return as ndarray '''
    n = len(support) # in order to use _get_dist(), we have to pass a dummy support
    fillValue = self.fillValue or np.NaN # for masked values
    data = self.data_array.reshape(self.data_array.shape+(1,)) # expand
    fct = functools.partial(kde_resample, support=support, n=n, fillValue=fillValue, dtype=self.dtype)
    samples = apply_along_axis(fct, self.ndim, data)
    assert samples.shape == self.shape + (n,)
    assert np.issubdtype(samples.dtype, self.dtype)
    return samples
  
  # distribution-specific method; should be overloaded by subclass
  def _cumulative_distribution(self, support):
    ''' integrate PDF over given support to produce a CDF and return as ndarray '''
    n = len(support); fillValue = self.fillValue or np.NaN
    data = self.data_array.reshape(self.data_array.shape+(1,))
    fct = functools.partial(kde_cdf, support=support, n=n, fillValue=fillValue)
    cdf = apply_along_axis(fct, self.ndim, data)
    assert cdf.shape == self.shape + (len(support),)
    return cdf
  

## VarRV subclass and helper functions

# N.B.: need to define helper functions outside of class definition, otherwise pickle / multiprocessing 
#       will fail... need to fix arguments with functools.partial

icres = None # persistent variable with most recent successful fit parameters
# N.B.: globals are only visible within the process, so this works nicely with multiprocessing
# estimate RV from sample vector
def rv_fit(sample, dist_type=None, ic=None, plen=None, lpersist=True, ldebug=False, **kwargs):
  global icres
  if ldebug: 
    if np.all(np.isnan(sample)):
      print('NaN'); res = (np.NaN,)*plen
    elif np.all(sample == sample[0]):
      print('equal'); res = (np.NaN,)*plen
    elif isinstance(sample,ma.MaskedArray) and np.all(sample.mask):
      print('masked'); res = (np.NaN,)*plen
    else:
      try: 
        if lpersist: 
          if icres is None: 
            res = getattr(ss,dist_type).fit(sample)
            print("Setting first guess: {:s}".format(str(res)))
          else: res = getattr(ss,dist_type).fit(sample, *icres[:-2], loc=icres[-2], scale=icres[-1], **kwargs)
          icres = res # update first guess
        elif ic is None: res = getattr(ss,dist_type).fit(sample)
        else: res = getattr(ss,dist_type).fit(sample, *ic[:-2], loc=ic[-2], scale=ic[-1], **kwargs)
      except LinAlgError:
        print('linalgerr'); res = (np.NaN,)*plen
  else:
    if np.all(np.isnan(sample)): res = (np.NaN,)*plen
    elif lpersist:
      if icres is None: res = getattr(ss,dist_type).fit(sample)
      else: res = getattr(ss,dist_type).fit(sample, *icres[:-2], loc=icres[-2], scale=icres[-1], **kwargs)
      icres = res # update first guess
    elif ic is None: res = getattr(ss,dist_type).fit(sample)
    else: res = getattr(ss,dist_type).fit(sample, *ic[:-2], loc=ic[-2], scale=ic[-1], **kwargs)
  return res # already is a tuple

# evaluate a RV distribution type over a given support with given parameters
def rv_eval(params, dist_type=None, fct_type=None, support=None, n=None, fillValue=np.NaN):
  if np.any(np.isnan(params)): res = np.zeros(len(support))+fillValue 
  else: res = getattr(getattr(ss,dist_type), fct_type)(support, *params[:-2], loc=params[-2], scale=params[-1])
  return res

# compute certain stats moments of a RV distribution type with given parameters
def rv_stats(params, dist_type=None, fct_type=None, moments=None, n=None, fillValue=np.NaN):
  if moments is not None:
    if np.any(np.isnan(params)): res = np.zeros(len(moments))+fillValue 
    else: res = getattr(getattr(ss,dist_type), fct_type)(*params[:-2], loc=params[-2], scale=params[-1], moments=moments)
  elif n is not None:
    if np.any(np.isnan(params)): res = fillValue 
    else: res = getattr(getattr(ss,dist_type), fct_type)(n,*params[:-2], loc=params[-2], scale=params[-1])
    res = (res,)
  else:
    if np.any(np.isnan(params)): res = fillValue 
    else: res = getattr(getattr(ss,dist_type), fct_type)(*params[:-2], loc=params[-2], scale=params[-1])
    res = (res,)
  return res

# draw n random samples from the RV distribution
def rv_resample(params, dist_type=None, n=None, fillValue=np.NaN, dtype=np.float):
  if np.any(np.isnan(params)): res = np.zeros(n)+fillValue 
  else: res = np.asarray(getattr(ss,dist_type).rvs(*params[:-2], loc=params[-2], scale=params[-1], size=n), dtype=dtype)
  return res

# Subclass of DistVar implementing various random variable distributions
class VarRV(DistVar):
  ''' A subclass of DistVar implementing Random Variable distributions (scipy.stats.rv_continuous) '''
  dist_func = None # the scipy RV distribution object
  dist_type = ''   # name of the distribution
  
  def __init__(self, dist='', **kwargs):
    ''' initialize a random variable distribution of type 'dist' '''
    # some aliases
    if dist.lower() in ('genextreme', 'gev'): dist = 'genextreme' 
    elif dist.lower() in ('genpareto', 'gpd'): dist = 'genpareto' # 'pareto' is something else!
    # look up module & set distribution
    if dist in ss.__dict__: 
      self.dist_func = ss.__dict__[dist]
      self.dist_type = dist
    else: 
      raise ArgumentError, "No distribution '{:s}' in module scipy.stats!".format(dist)
    # initialize distribution variable
    super(VarRV,self).__init__(**kwargs) 
    
  def __getattr__(self, attr):
    ''' use methods of from RV distribution through _get_dist wrapper '''
    if hasattr(self.dist_func, attr):
      attr = functools.partial(self._get_dist, self._compute_distribution, attr, rv_fct=attr)
    return attr
  
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, ic=None, plen=None, lpersist=True, ldebug=False, **kwargs):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    global icres
    if ic is None:
      if plen is None: plen = 3 # default for most distributions
      fct = functools.partial(rv_fit, dist_type=self.dist_type, plen=plen, ldebug=ldebug, **kwargs)
    else:
      if lpersist: icres = ic # set initial conditions
      if plen is None: plen = len(ic)
      elif plen != len(ic): raise ArgumentError
      fct = functools.partial(rv_fit, *ic[0:-2], loc=ic[-2], scale=ic[-1], plen=plen, 
                              dist_type=self.dist_type, ldebug=ldebug, **kwargs)
    params = apply_along_axis(fct, samples.ndim-1, samples)
    if lpersist: icres = None # reset; in parallel mode actually not necessary
    assert samples.shape[:-1]+(plen,) == params.shape
    # return an array of kernels
    return params

  # universal RV method applicator for distributions
  def _compute_distribution(self, *args, **kwargs):
    ''' compute a given distribution type over the given support points for each grid point and return as ndarray '''
    if 'rv_fct' not in kwargs: raise ArgumentError
    else: rv_fct = kwargs.pop('rv_fct')
    if  len(args) == 0:
      fillValue = self.fillValue or np.NaN
      fct = functools.partial(rv_stats, dist_type=self.dist_type, fct_type=rv_fct, fillValue=fillValue, **kwargs)
      dist = apply_along_axis(fct, self.ndim-1, self.data_array)
      assert dist.shape[:-1] == self.shape[:-1]
    elif  len(args) == 1 and rv_fct == 'moment':
      raise NotImplementedError
      fillValue = self.fillValue or np.NaN
      fct = functools.partial(rv_stats, dist_type=self.dist_type, fct_type=rv_fct, fillValue=fillValue, **kwargs)
      dist = apply_along_axis(fct, self.ndim-1, self.data_array)
      assert dist.shape[:-1] == self.shape[:-1]
    elif len(args) == 1:
      support = args[0]
      assert isinstance(support, np.ndarray)
      n = len(support); fillValue = self.fillValue or np.NaN
      fct = functools.partial(rv_eval, dist_type=self.dist_type, fct_type=rv_fct, 
                              support=support, n=n, fillValue=fillValue, **kwargs)
      dist = apply_along_axis(fct, self.ndim-1, self.data_array)
      assert dist.shape == self.shape[:-1] + (len(support),)
    else: raise ArgumentError
    return dist
  
#   # universal RV method applicator for statistical moments
#   def _compute_moments(self, rv_fct=None, **kwargs):
#     ''' compute a given distribution type over the given support points for each grid point and return as ndarray '''
#     fillValue = self.fillValue or np.NaN
#     fct = functools.partial(rv_eval, dist_type=self.dist_type, fct_type=rv_fct, fillValue=fillValue, **kwargs)
#     stats = apply_along_axis(fct, self.ndim-1, self.data_array)
#     assert stats.shape[:-1] == self.shape[:-1]
#     return stats

  # distribution-specific method; should be overloaded by subclass
  def _density_distribution(self, support, **kwargs):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    return self._compute_distribution(support, rv_fct='pdf', **kwargs)

  # distribution-specific method; should be overloaded by subclass
  def _cumulative_distribution(self, support, **kwargs):
    ''' compute CDF at given support points for each grid point and return as ndarray '''
    return self._compute_distribution(support, rv_fct='cdf', **kwargs)
  
  # distribution-specific method; should be overloaded by subclass
  def _resample_distribution(self, support):
    ''' draw n samples from the distribution for each grid point and return as ndarray '''
    n = len(support) # in order to use _get_dist(), we have to pass a dummy support
    fillValue = self.fillValue or np.NaN # for masked values
    fct = functools.partial(rv_resample, dist_type=self.dist_type, n=n, fillValue=fillValue, dtype=self.dtype)
    samples = apply_along_axis(fct, self.ndim-1, self.data_array)
    assert samples.shape == self.shape[:-1] + (n,)
    assert np.issubdtype(samples.dtype, self.dtype)
    return samples
