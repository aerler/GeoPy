'''
Created on Dec 12, 2014

Some Variable subclasses for handling distribution functions over grid points. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import scipy.stats as ss
from numpy.linalg.linalg import LinAlgError
from processing.multiprocess import apply_along_axis
import functools
# internal imports
from geodata.base import Variable, Axis
from geodata.misc import DataError, ArgumentError, VariableError, AxisError, DistVarError
from plotting.properties import getPlotAtts


## statistical tests and utility functions

## univariate statistical tests and functions

# wrapper for Anderson-Darling Test
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
def anderson_wrapper(data, dist='norm', ignoreNaN=True):
  ''' Anderson-Darling Test, to test whether or not the data is from a given distribution. The 
      returned p-value indicates the probability that the data is from the given distribution, 
      i.e. a low p-value means the data are likely not from the tested distribution. Note that 
      the maximum returned p-value is 15% for the normal & exponential and 25% for the 
      logistic & Gumbel distributions. '''
  if ignoreNaN: 
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans) < 3: return np.NaN # return, if less than 3 non-NaN's
    data = data[nonans] # remove NaN's
  A2, crit, sig = ss.anderson(data, dist=dist)
  return sig[max(0,np.searchsorted(crit,A2)-1)]/100.

# wrapper for single-sample Kolmogorov-Smirnov Test
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
def kstest_wrapper(data, dist='norm', ignoreNaN=True, args=None, N=20, alternative='two-sided', mode='approx'):
  ''' Kolmogorov-Smirnov Test, to test whether or not the data is from a given distribution. The 
      returned p-value indicates the probability that the data is from the given distribution, 
      i.e. a low p-value means the data are likely not from the tested distribution.
      Note that, for this test, it is necessary to specify shape, location, and scale parameters,
      to obtain meaningful results (c,loc,scale). '''
  if ignoreNaN: 
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans) < 3: return np.NaN # return, if less than 3 non-NaN's
    data = data[nonans] # remove NaN's
  D, pval = ss.kstest(data, dist, args=args, N=N, alternative=alternative, mode=mode)
  return pval

# wrapper for normaltest, a SciPy function to test normality
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
def normaltest_wrapper(data, axis=None, ignoreNaN=True):
  ''' SciPy test, to test whether or not the data is from a normal distribution. The 
      returned p-value indicates the probability that the data is from a normal distribution, 
      i.e. a low p-value means the data are likely not from a normal distribution.
      This is a combination of the skewtest and the kurtosistest and can be applied along a 
      specified axis of a multi-dimensional arrays (using the 'axis' keyword), or over the 
      flattened array (axis=None). '''
  if axis is None and ignoreNaN: 
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans) < 3: return np.NaN # return, if less than 3 non-NaN's
    data = data[nonans] # remove NaN's
  k2, pval = ss.normaltest(data, axis=axis)
  return pval

# global variable that is used to retain parameters for Shapiro-wilk test
shapiro_a = None
# wrapper for Shapiro-Wilk test of normality
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
def shapiro_wrapper(data, reta=False, ignoreNaN=True):
  ''' Shapiro-Wilk Test, to test whether or not the data is from a normal distribution. The 
      returned p-value indicates the probability that the data is from a normal distribution, 
      i.e. a low p-value means the data are likely not from a normal distribution. '''
  if ignoreNaN: 
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans) < 3: return np.NaN # return, if less than 3 non-NaN's
    data = data[nonans] # remove NaN's
  if reta:
    global shapiro_a
    if  shapiro_a is None or len(shapiro_a) != len(data)//2:
      W, pval, a = ss.shapiro(data, a=None, reta=True)
      shapiro_a = a # save parameters
    else:
      W, pval = ss.shapiro(data, a=shapiro_a, reta=False)
  else:
    W, pval = ss.shapiro(data, a=None, reta=False)
  return pval
  # N.B.: a only depends on the length of data, so it can be easily reused in array operation


## bivariate statistical tests and functions
    
# Kolmogorov-Smirnov Test on 2 samples
def ks_2samp(sample1, sample2, lstatistic=False, ignoreNaN=True, **kwargs):
  ''' Apply the Kolmogorov-Smirnov Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution; a high p-value means, the two samples are likely
      drawn from the same distribution. 
      The Kolmogorov-Smirnov Test is a non-parametric test that works well for all types of 
      distributions (normal and non-normal). '''
  if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
  testfct = functools.partial(ks_2samp_wrapper, ignoreNaN=ignoreNaN)
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=True, **kwargs)
  return pvar
kstest = ks_2samp # alias

# apply-along-axis wrapper for the Kolmogorov-Smirnov Test on 2 samples
def ks_2samp_wrapper(data, size1=None, ignoreNaN=True):
  ''' Apply the Kolmogorov-Smirnov Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's, allows application over a field, and only returns the p-value. '''
  if ignoreNaN:
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans[:size1]) < 3 or np.sum(nonans[size1:]) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data[nonans[:size1]]; data2 = data[nonans[size1:]] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # apply test
  D, pval = ss.ks_2samp(data1, data2)
  return pval  


# Stundent's T-test for two independent samples
def ttest_ind(sample1, sample2, equal_var=True, lstatistic=False, ignoreNaN=True, **kwargs):
  ''' Apply the Stundent's T-test for two independent samples, to test whether the samples 
      are drawn from the same underlying (continuous) distribution; a high p-value means, 
      the two samples are likely drawn from the same distribution. 
      The T-test implementation is vectoriezed (unlike all other tests).'''
  if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
  testfct = functools.partial(ttest_ind_wrapper, ignoreNaN=ignoreNaN, equal_var=equal_var)
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=False, **kwargs)
  return pvar
ttest = ttest_ind # alias

# apply-along-axis wrapper for the Kolmogorov-Smirnov Test on 2 samples
def ttest_ind_wrapper(data, size1=None, axis=None, ignoreNaN=True, equal_var=True):
  ''' Apply the Stundent's T-test for two independent samples, to test whether the samples 
      are drawn from the same underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's and only returns the p-value (t-test is already vectorized). '''
  if axis is None and ignoreNaN:
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans[:size1]) < 3 or np.sum(nonans[size1:]) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data[nonans[:size1]]; data2 = data[nonans[size1:]] # remove NaN's
  elif axis is None:
    data1 = data[:size1]; data2 = data[size1:]
  else:
    data1, data2 = np.split(data, [size1], axis=axis)
  # apply test
  D, pval = ss.ttest_ind(data1, data2, axis=axis, equal_var=equal_var)
  return pval  

# Mann-Whitney Rank Test on 2 samples
def mannwhitneyu(sample1, sample2, ignoreNaN=True, lonesided=False, lstatistic=False, 
                 use_continuity=True, **kwargs):
  ''' Apply the Mann-Whitney Rank Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution; a high p-value means, the two samples are likely
      drawn from the same distribution.
      The Mann-Whitney Test has very high efficiency for non-normal distributions and is almost as
      reliable as the T-test for normal distributions. It is more sophisticated than the 
      Wilcoxon Ranksum Test and also handles ties between ranks. 
      One-sided p-values test the hypothesis that one distribution is larger than the other;
      the two-sided test just tests, if the distributions are different. '''
  if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
  testfct = functools.partial(mannwhitneyu_wrapper, ignoreNaN=ignoreNaN, 
                              use_continuity=use_continuity)
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=True, **kwargs)
  if not lonesided: # transform to twosided (multiply p-value by 2)
    if isinstance(pvar,Variable): pvar.data_array *= 2.
    else : pvar *= 2.
  # N.B.: for some reason Mann-Whitney returns the one-sided p-value... The one-sided p-value 
  #       is for the hypothesis that one sample is larger than the other.
  #       It is better to correct here, so we can use vector multiplication.
  return pvar
mwtest = mannwhitneyu # alias

# apply-along-axis wrapper for the Mann-Whitney Rank Test on 2 samples
def mannwhitneyu_wrapper(data, size1=None, ignoreNaN=True, use_continuity=True, loneside=False):
  ''' Apply the Mann-Whitney Rank Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's, allows application over a field, and only returns the p-value. '''
  if ignoreNaN:
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans[:size1]) < 3 or np.sum(nonans[size1:]) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data[nonans[:size1]]; data2 = data[nonans[size1:]] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # apply test
  D, pval = ss.mannwhitneyu(data1, data2, use_continuity=use_continuity)
  return pval  


# Wilcoxon Ranksum Test on 2 samples
def ranksums(sample1, sample2, lstatistic=False, ignoreNaN=True, **kwargs):
  ''' Apply the Wilcoxon Ranksum Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution; a high p-value means, the two samples are likely
      drawn from the same distribution. 
      The Ranksum Test has higher efficiency for non-normal distributions and is almost as
      reliable as the T-test for normal distributions. It is less sophisticated than the 
      Mann-Whitney Test and does not handle ties between ranks. '''
  if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
  testfct = functools.partial(ranksums_wrapper, ignoreNaN=ignoreNaN)
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=True, **kwargs)
  return pvar
wrstest = ranksums # alias

# apply-along-axis wrapper for the Wilcoxon Ranksum Test on 2 samples
def ranksums_wrapper(data, size1=None, ignoreNaN=True):
  ''' Apply the Wilcoxon Ranksum Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's, allows application over a field, and only returns the p-value. '''
  if ignoreNaN:
    nonans = np.invert(np.isnan(data)) # test for NaN's
    if np.sum(nonans[:size1]) < 3 or np.sum(nonans[size1:]) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data[nonans[:size1]]; data2 = data[nonans[size1:]] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # apply test
  D, pval = ss.ranksums(data1, data2)
  return pval  


# generic applicator function for 2 sample statistical tests
def apply_stat_test_2samp(sample1, sample2, fct=None, axis=None, axis_idx=None, name=None, 
                          lflatten=False, fillValue=None, lpval=True, lcorr=False, asVar=None,
                          lcheckVar=True, lcheckAxis=True, pvaratts=None, cvaratts=None, **kwargs):
  ''' Apply a bivariate statistical test to two sample Variables and return the result as a Variable object;
      the function will be applied along the specified axis or over flattened arrays. '''
  # some input checking
  if lflatten and axis is not None: raise ArgumentError
  if not lflatten and axis is None: raise ArgumentError
  if asVar is None: asVar = not lflatten
  # check sample vars
  lvar1 = isinstance(sample1, Variable)
  lvar2 = isinstance(sample2, Variable)
  # figure out axes
  if axis_idx is not None and axis is None: 
    if lvar1: axis = sample1.axes[axis_idx].name
    elif lvar2: axis = sample2.axes[axis_idx].name
    if lvar1 and lvar2 and not (axis != sample2.axes[axis_idx].name): 
      raise AxisError, "Axis index '{:d}' does not refer to the same axis in the two samples.".format(axis_idx)
    axis_idx1 = axis_idx2 = axis_idx
  elif axis_idx is None and axis is not None: 
    if not lvar1 and not lvar2: ArgumentError, "Keyword 'axis' requires at least one sample to be a Variable instance."
    if lvar1: axis_idx1 = sample1.axisIndex(axis)
    if lvar2: axis_idx2 = sample2.axisIndex(axis)
    if not lvar1: axis_idx1 = axis_idx2
    if not lvar2: axis_idx2 = axis_idx1
  elif lflatten: axis_idx1 = axis_idx2 = 0
  else: raise ArgumentError
  del axis_idx # should not be used any longer    
  # check sample variables
  for sample in sample1,sample2:
    if sample.dtype.kind in ('S',):
      if lcheckVar: raise VariableError, "Statistical tests does not work with string Variables!"
      else: return None
    if isinstance(sample,Variable):
      if not lflatten and not sample.hasAxis(axis):
        if lcheckAxis: raise AxisError, "Variable '{:s}' has no axis '{:s}'.".format(sample.name, axis)
        else: return None
    elif not isinstance(sample, np.ndarray):
      raise TypeError, "Samples have to be Variable instances or Numpy 'ndarray'."
    # choose a fillValue (triggers only once)
    if fillValue is None and sample.masked:
      if np.issubdtype(sample.dtype,np.integer): fillValue = 0
      elif np.issubdtype(sample.dtype,np.inexact): fillValue = np.NaN
      else: raise NotImplementedError
  # check that dtype and dimensions are equal
  if sample1.dtype != sample2.dtype: raise TypeError, "Samples need to have same dtype."
  if not lflatten:
    if sample1.ndim != sample2.ndim: raise AxisError, "Samples need to have same number of dimensions."
    rshape = sample1.shape[:axis_idx1] + sample1.shape[axis_idx1+1:]
    if rshape != sample2.shape[:axis_idx2] + sample2.shape[axis_idx2+1:]: 
      raise AxisError, "Samples need to have same shape (except sample axis)."
    if lvar1 and lvar2:
      axes1 = tuple(ax.name for ax in sample1.axes if ax.name != axis)
      axes2 = tuple(ax.name for ax in sample2.axes if ax.name != axis)
      if axes1 != axes2: raise AxisError, "Axes of samples are inconsistent."
  # create attributes for new p-values variable object
  if asVar and lpval:
    #if not name and not varatts: raise ArgumentError, 'Need a name or variable attributes to create a Variable.'
    varatts = dict()
    if lvar1 and lvar2:
      varatts['name'] = name or '{:s}_{:s}_pval'.format(sample1.name,sample2.name)
      varatts['long_name'] = 'p-value of {:s} and {:s}'.format(sample1.name.title(),sample1.name.title())
    else:
      if not name: varatts['name'] = 'pval'; varatts['long_name'] = 'p-value'
      else: varatts['name'] = name; varatts['long_name'] = 'p-value ({:s})'.format(name)
    varatts['units'] = '' # p-value / probability
    if pvaratts is not None: varatts.update(pvaratts)
    pvaratts = varatts.copy()
    pvarplot = getPlotAtts(name=name or pvaratts['name'], units='') # infer meta data from plot attributes
  # prepare data
  def preprocess(sample, axis_idx):
    ''' helper function to pre-process each sample '''
    # get data
    if isinstance(sample,Variable): 
      data = sample.getArray(unmask=True, fillValue=fillValue, copy=True)
    elif isinstance(sample,ma.MaskedArray): data = sample.filled(np.NaN)
    elif isinstance(sample,np.ndarray): data = sample.copy()
    else: raise TypeError
    # roll sampel axis to end (or flatten)
    if lflatten: data = data.ravel()
    else: data = np.rollaxis(data, axis=axis_idx, start=sample.ndim)
    return data
  data1 = preprocess(sample1, axis_idx1)
  data2 = preprocess(sample2, axis_idx2)
  assert lflatten or data1.shape[:-1] == data2.shape[:-1]
  # apply test (serial)
  if lflatten:
    axis_idx = data1.ndim-1; size1 = data1.shape[-1]; laax=True # shorcuts
    # merge sample arrays, save dividing index 'size1' (only one argument array per point along axis)
    data_array = np.concatenate((data1, data2), axis=axis_idx) 
    pval = fct(data_array, size1=size1)
    # create new Axis and Variable objects (1-D)
    if asVar: 
      raise NotImplementedError, "Cannot return a single scalar as a Variable object."
    else: pvar = pval
  # apply test (parallel)
  else: 
    axis_idx = data1.ndim-1; size1 = data1.shape[-1]; laax=True # shorcuts
    # merge sample arrays, save dividing index 'size1' (only one argument array per point along axis)
    data_array = np.concatenate((data1, data2), axis=axis_idx) 
    # select test and set parameters
    pval = apply_along_axis(fct, axis_idx, data_array, laax=laax) # apply test in parallel, distributing the data
    assert pval.ndim == sample1.ndim-1
    assert pval.shape == rshape
    # handle masks etc.
    if (lvar1 and sample1.masked) or (lvar2 and sample1.masked): 
      pval = ma.masked_invalid(pval, copy=False) 
      # N.B.: comparisons with NaN always evaluate to False!
      if fillValue is not None:
        pval = ma.masked_equal(pval, fillValue)
      ma.set_fill_value(pval,fillValue)          
    if asVar:
      if lpval:
        axes = sample1.axes[:axis_idx1] + sample1.axes[axis_idx1+1:]
        pvar = Variable(data=pval, axes=axes, atts=pvaratts, plot=pvarplot)
    else: pvar = pval
  # return results
  return pvar

## distribution variable classes 

# convenience function to generate a DistVar from another Variable object
def asDistVar(var, axis='time', dist='KDE', lflatten=False, name=None, atts=None, **kwargs):
  ''' generate a DistVar of type 'dist' from a Variable 'var'; use dimension 'axis' as sample axis '''
  if not isinstance(var,Variable): raise VariableError
  if not var.data: var.load() 
  # create some sensible default attributes
  varatts = dict()
  varatts['name'] = name or '{:s}_{:s}'.format(var.name,dist)
  varatts['long_name'] = "'{:s}'-distribution of {:s}".format(dist,var.atts.get('long_name',var.name.title()))
  varatts['units'] = var.units # these will be the sample units, not the distribution units
  if atts is not None: varatts.update(atts)
  # figure out axes
  if lflatten:
    iaxis = None; axes = None
  else:
    iaxis=var.axisIndex(axis)
    axes = var.axes[:iaxis]+var.axes[iaxis+1:] # i.e. without axis/iaxis 
  # choose distribution
  dist = dist.lower()
  if dist.lower() == 'kde': dvar = VarKDE(samples=var.data_array, axis=iaxis, axes=axes, lflatten=lflatten, atts=varatts, **kwargs)
  elif hasattr(ss,dist):
    dvar = VarRV(dist=dist, samples=var.data_array, axis=iaxis, axes=axes, lflatten=lflatten, atts=varatts, **kwargs)
  else:
    raise AttributeError, "Distribution '{:s}' not found in scipy.stats.".format(dist)
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
               lflatten=False, masked=None, mask=None, fillValue=None, atts=None, ic=None, ldebug=False, **kwargs):
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
      # flatten, if required
      if lflatten: 
        samples = samples.ravel()
        if axis is not None: raise AxisError, "'axis' keyword can not be used with lflatten=True"
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
      if lflatten:
        if axes is not None: raise AxisError, "'axes' keyword can not be used with lflatten=True"
        axes = tuple() # empty tuple
      else:
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

  # generic function to retrieve distribution values
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
    nonans = np.invert(np.isnan(sample)) # test for NaN's
    if not np.any(nonans):
      print('all NaN'); res = (np.NaN,)*plen
    elif np.sum(nonans) < plen: 
      print('NaN'); res = (np.NaN,)*plen # require at least plen non-NaN points 
    else:
      sample = sample[nonans] # remove NaN's
      if np.all(sample == sample[0]):
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
    nonans = np.invert(np.isnan(sample)) # test for NaN's
    if np.sum(nonans) < plen: 
      res = (np.NaN,)*plen # require at least plen non-NaN points 
    else:
      sample = sample[nonans] # remove NaN's
      if lpersist:
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

# perform a Kolmogorov-Smirnov Test of goodness of fit
def rv_kstest(data_array, nparams=0, dist_type=None, ignoreNaN=True, N=20, alternative='two-sided', mode='approx'):
  if np.any(np.isnan(data_array[:nparams])): pval = np.NaN 
  else: pval = kstest_wrapper(data_array[nparams:], dist=dist_type, args=data_array[:nparams], ignoreNaN=ignoreNaN, N=N, alternative=alternative, mode=mode)
  return pval

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
      if plen is None: 
        if self.dist_type == 'norm': plen = 2 # mean and std. dev.
        else: plen = 3 # default for most distributions
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

  # 
  def kstest(self, sample, name=None, axis_idx=None, lstatistic=False, 
             fillValue=None, ignoreNaN=True, N=20, alternative='two-sided', mode='approx', 
             asVar=True, lcheckVar=True, lcheckAxis=True, pvaratts=None, **kwargs):
    ''' apply a Kolmogorov-Smirnov Test to the sample data, based on this distribution '''
    # check input
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Statistical tests does not work with string Variables!"
      else: return None
    if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
    # choose a fillValue, because np.histogram does not ignore masked values but does ignore NaNs
    if fillValue is None and self.masked:
      if np.issubdtype(self.dtype,np.integer): fillValue = 0
      elif np.issubdtype(self.dtype,np.inexact): fillValue = np.NaN
      else: raise NotImplementedError
    # if sample is a variable, check and figure out sample axes   
    sax = self.ndim-1
    if isinstance(sample,Variable):
      assert self.axisIndex('params') == sax
      for ax in self.axes[:-1]:
        if not sample.hasAxis(ax): # last is parameter axis
          raise AxisError, "Sample Variable needs to have a '{:s}' axis.".format(ax.name)
        if len(sample.getAxis(ax)) != len(ax): # last is parameter axis
          raise AxisError, "Axis '{:s}' in Sample and DistVar have different length!".format(ax.name)
      sample_data = sample.getArray(unmask=True, fillValue=fillValue, copy=True) # actual data (will be reordered)
      # move extra dimensions to the back
      exax = [ax for ax in sample.axes if not self.hasAxis(ax)] # extra dimensions
      for ax in exax[::-1]:
        sample_data = np.rollaxis(sample_data, axis=sample.axisIndex(ax), start=sample.ndim)
      # N.B.: the order of these dimensions will be reversed, but order doesn't matter here
    else:
      if isinstance(sample,ma.MaskedArray): sample_data = sample.filled(np.NaN)
      else: sample_data = sample.copy()
      if axis_idx is not None and axis_idx != sax:
        sample_data = np.rollaxis(sample_data, axis=axis_idx, start=sample.ndim)
    # check dimensions of sample_data and reshape
    if sample_data.shape[:sax] != self.shape[:-1]: 
      if lcheckAxis: raise AxisError, "Sample has incompatible shape."
      else: return None
    # collapse all remaining dimensions at the end and use as sample dimension
    sample_data = sample_data.reshape(sample_data.shape[:sax]+(np.prod(sample_data.shape[sax:]),))
    assert sample_data.ndim == self.ndim
    # apply test function (parallel)
    fct = functools.partial(rv_kstest, nparams=len(self.paramAxis), dist_type=self.dist_type, ignoreNaN=ignoreNaN, N=N, alternative=alternative, mode=mode)
    data_array = np.concatenate((self.data_array, sample_data), axis=sax) # merge params and sample arrays (only one argument array per point along axis) 
    pval = apply_along_axis(fct, sax, data_array) # apply test in parallel, distributing the data
    assert pval.ndim == sax
    assert pval.shape == self.shape[:-1]
    # handle masked values
    if self.masked: 
      pval = ma.masked_invalid(pval, copy=False) 
      # N.B.: comparisons with NaN always evaluate to False!
      if self.fillValue is not None:
        pval = ma.masked_equal(pval, self.fillValue)
      ma.set_fill_value(pval,self.fillValue)          
    # create new variable object
    if asVar:
      plotatts = getPlotAtts(name='pval', units='') # infer meta data from plot attributes
      varatts = self.atts.copy()
      varatts['name'] = name or '{:s}_pval'.format(self.name)
      varatts['long_name'] = "p-value of {:s}".format(self.atts.get('long_name',self.name.title()))
      varatts['units'] = '' # p-value / probability
      if pvaratts is not None: varatts.update(pvaratts)
      assert self.axisIndex('params') == sax
      pvar = Variable(data=pval, axes=self.axes[:-1], atts=varatts, plot=plotatts)
    else: pvar = pval
    # return results
    return pvar
    