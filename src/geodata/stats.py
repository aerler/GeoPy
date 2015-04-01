'''
Created on Dec 12, 2014

A statistics module that makes some of the functionality of the SciPy's scipy.stats module available 
with Variables. It includes some common statistical functions and tests, as well assome Variable 
subclasses for handling distribution functions over grid points. 

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
from utils.misc import standardize, smooth, detrend
import utils.stats as myss # modified stats fucntions from scipy 
from plotting.properties import getPlotAtts


## statistical tests and utility functions

## univariate statistical tests 
# N.B.: application functions are implemented as Variable class methods (see geodata.base module)

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
  if args is None: args = ()
  D, pval = ss.kstest(data, dist, args=args, N=N, alternative=alternative, mode=mode); del D
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
  k2, pval = ss.normaltest(data, axis=axis); del k2
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
      W, pval, a = ss.shapiro(data, a=None, reta=True); del W
      shapiro_a = a # save parameters
    else:
      W, pval = ss.shapiro(data, a=shapiro_a, reta=False); del W
  else:
    W, pval = ss.shapiro(data, a=None, reta=False); del W
  return pval
  # N.B.: a only depends on the length of data, so it can be easily reused in array operation


## bivariate statistical tests
    
# Kolmogorov-Smirnov Test on 2 samples
def ks_2samp(sample1, sample2, lstatistic=False, ignoreNaN=True, **kwargs):
  ''' Apply the Kolmogorov-Smirnov Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution; a high p-value means, the two samples are likely
      drawn from the same distribution. 
      The Kolmogorov-Smirnov Test is a non-parametric test that works well for all types of 
      distributions (normal and non-normal). '''
  if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
  testfct = functools.partial(ks_2samp_wrapper, ignoreNaN=ignoreNaN)
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=True, 
                               lpval=True, lrho=False, **kwargs)
  return pvar
kstest = ks_2samp # alias

# apply-along-axis wrapper for the Kolmogorov-Smirnov Test on 2 samples
def ks_2samp_wrapper(data, size1=None, ignoreNaN=True):
  ''' Apply the Kolmogorov-Smirnov Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's, allows application over a field, and only returns the p-value. '''
  if ignoreNaN:
    data1 = data[:size1]; data2 = data[size1:]
    nonans1 = np.invert(np.isnan(data1)) # test for NaN's
    nonans2 = np.invert(np.isnan(data2))
    if np.sum(nonans1) < 3 or np.sum(nonans2) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data1[nonans1]; data2 = data2[nonans2] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # apply test
  D, pval = ss.ks_2samp(data1, data2); del D
  return pval  


# Stundent's T-test for two independent samples
def ttest_ind(sample1, sample2, equal_var=True, lstatistic=False, ignoreNaN=True, **kwargs):
  ''' Apply the Stundent's T-test for two independent samples, to test whether the samples 
      are drawn from the same underlying (continuous) distribution; a high p-value means, 
      the two samples are likely drawn from the same distribution. 
      The T-test implementation is vectoriezed (unlike all other tests).'''
  if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
  testfct = functools.partial(ttest_ind_wrapper, ignoreNaN=ignoreNaN, equal_var=equal_var)
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=False, 
                               lpval=True, lrho=False, **kwargs)
  return pvar
ttest = ttest_ind # alias

# apply-along-axis wrapper for the Kolmogorov-Smirnov Test on 2 samples
def ttest_ind_wrapper(data, size1=None, axis=None, ignoreNaN=True, equal_var=True):
  ''' Apply the Stundent's T-test for two independent samples, to test whether the samples 
      are drawn from the same underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's and only returns the p-value (t-test is already vectorized). '''
  if axis is None and ignoreNaN:
    data1 = data[:size1]; data2 = data[size1:]
    nonans1 = np.invert(np.isnan(data1)) # test for NaN's
    nonans2 = np.invert(np.isnan(data2))
    if np.sum(nonans1) < 3 or np.sum(nonans2) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data1[nonans1]; data2 = data2[nonans2] # remove NaN's
  elif axis is None:
    data1 = data[:size1]; data2 = data[size1:]
  else:
    data1, data2 = np.split(data, [size1], axis=axis)
  # apply test
  D, pval = ss.ttest_ind(data1, data2, axis=axis, equal_var=equal_var); del D
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
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=True, 
                               lpval=True, lrho=False, **kwargs)
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
    data1 = data[:size1]; data2 = data[size1:]
    nonans1 = np.invert(np.isnan(data1)) # test for NaN's
    nonans2 = np.invert(np.isnan(data2))
    if np.sum(nonans1) < 3 or np.sum(nonans2) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data1[nonans1]; data2 = data2[nonans2] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # apply test
  D, pval = ss.mannwhitneyu(data1, data2, use_continuity=use_continuity); del D
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
  pvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, laax=True, 
                               lpval=True, lrho=False, **kwargs)
  return pvar
wrstest = ranksums # alias

# apply-along-axis wrapper for the Wilcoxon Ranksum Test on 2 samples
def ranksums_wrapper(data, size1=None, ignoreNaN=True):
  ''' Apply the Wilcoxon Ranksum Test, to test whether two samples are drawn from the same
      underlying (continuous) distribution. This is a wrapper for the SciPy function that 
      removes NaN's, allows application over a field, and only returns the p-value. '''
  if ignoreNaN:
    data1 = data[:size1]; data2 = data[size1:]
    nonans1 = np.invert(np.isnan(data1)) # test for NaN's
    nonans2 = np.invert(np.isnan(data2))
    if np.sum(nonans1) < 3 or np.sum(nonans2) < 3: return np.NaN # return, if less than 3 non-NaN's
    data1 = data1[nonans1]; data2 = data2[nonans2] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # apply test
  D, pval = ss.ranksums(data1, data2); del D
  return pval  


## bivariate statistical functions

# Pearson's Correlation Coefficient between two samples
def pearsonr(sample1, sample2, lpval=False, lrho=True, ignoreNaN=True, lstandardize=False, 
             lsmooth=False, window_len=11, window='hanning', ldetrend=False, dof=None, **kwargs):
  ''' Compute and return the linear correlation coefficient and/or the p-value
      of Pearson's correlation. 
      Pearson's Correlation Coefficient measures the linear relationship between
      the two sample variables (this is the ordinary correlation coefficient);
      the p-values assume that the samples are normally distributed. 
      Standardization and smoothing is also supported; detrending is not implemented yet. '''
  testfct = functools.partial(pearsonr_wrapper, lpval=lpval, lrho=lrho, ignoreNaN=ignoreNaN,
                              lstandardize=lstandardize, ldetrend=ldetrend, dof=dof,
                              lsmooth=lsmooth, window_len=window_len, window=window)
  rvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, 
                               lpval=lpval, lrho=lrho, laax=True, **kwargs)
  return rvar
corrcoef = pearsonr

# apply-along-axis wrapper for the Pearson's Correlation Coefficient on 2 samples
def pearsonr_wrapper(data, size1=None, lpval=False, lrho=True, ignoreNaN=True, lstandardize=False, 
                     lsmooth=False, window_len=11, window='hanning', ldetrend=False, dof=None):
  ''' Compute the Pearson's Correlation Coefficient of two samples. This is a wrapper 
      for the SciPy function allows application over a field, and returns 
      the correlation coefficient and/or the p-value. '''
  # N.B.: the Numpy corrcoef function also only operates on flat arrays 
  if ignoreNaN:
    data1 = data[:size1]; data2 = data[size1:] # find NaN's
    nans1 = np.isnan(data1); nans2 = np.isnan(data2) # remove in both arrays
    nonans = np.invert(np.logical_or(nans1,nans2))
    if np.sum(nonans) < 3: # return, if too many NaN's 
      if lrho and lpval: return np.zeros(2)+np.NaN
      else: return np.NaN # need to conform to output size
    data1 = data1[nonans]; data2 = data2[nonans] # remove NaN's
  else:
    data1 = data[:size1]; data2 = data[size1:]
  # pre-process data
  if lstandardize: 
    data1 = standardize(data1, axis=None, lcopy=False) # apply_stat_test_2samp alread 
    data2 = standardize(data2, axis=None, lcopy=False) #   makes a copy, no need here
  if lsmooth:
    window_len = min(data1.size,window_len) # automatically shring window
    data1 = smooth(data1,  window_len=window_len, window=window)
    data2 = smooth(data2,  window_len=window_len, window=window)
  if ldetrend:
    data1 = detrend(data1); data2 = detrend(data2)
  # apply test
  rho, pval = myss.pearsonr(data1, data2, dof=dof)
  # select output
  if lrho and lpval: return np.asarray((rho,pval))
  elif lrho: return rho
  elif lpval: return pval
  else: raise ArgumentError  


# Spearman's Rank-order Correlation Coefficient between two samples
def spearmanr(sample1, sample2, lpval=False, lrho=True, ignoreNaN=True, lstandardize=False, 
              lsmooth=False, window_len=11, window='hanning', ldetrend=False, dof=None, **kwargs):
  ''' Compute and return the linear correlation coefficient and/or the p-value
      of Spearman's Rank-order Correlation Coefficient. 
      Spearman's Rank-order Correlation Coefficient measures the monotonic 
      relationship between the two samples; it is more robust for non-linear
      and non-normally distributed samples than the ordinary correlation
      coefficient.  
      Standardization and smoothing is also supported; detrending is not implemented yet. '''
  testfct = functools.partial(spearmanr_wrapper, lpval=lpval, lrho=lrho, ignoreNaN=ignoreNaN,
                              lstandardize=lstandardize, ldetrend=ldetrend, dof=dof,
                              lsmooth=lsmooth, window_len=window_len, window=window)
  laax = lsmooth or ldetrend # true, if any of these, false otherwise
  rvar = apply_stat_test_2samp(sample1, sample2, fct=testfct, 
                               lpval=lpval, lrho=lrho, laax=laax, **kwargs)
  return rvar
spearmancc = spearmanr

# apply-along-axis wrapper for the Spearman's Rank-order Correlation Coefficient on 2 samples
def spearmanr_wrapper(data, size1=None, axis=None, lpval=False, lrho=True, ignoreNaN=True, lstandardize=False, 
                      lsmooth=False, window_len=11, window='hanning', ldetrend=False, dof=None):
  ''' Compute the Spearman's Rank-order Correlation Coefficient of two samples. This is a wrapper 
      for the SciPy function allows application over a field, and returns 
      the correlation coefficient and/or the p-value. 
      The SciPy implementation works along multiple axes (using Numpy's
      apply_along_axis), but it is not truly vectorized. '''
  if axis is None and ignoreNaN:
    data1 = data[:size1]; data2 = data[size1:] # find NaN's
    nans1 = np.isnan(data1); nans2 = np.isnan(data2) # remove in both arrays
    nonans = np.invert(np.logical_or(nans1,nans2))
    if np.sum(nonans) < 3: # return, if too many NaN's 
      if lrho and lpval: return np.zeros(2)+np.NaN
      else: return np.NaN # need to conform to output size
    data1 = data1[nonans]; data2 = data2[nonans] # remove NaN's
  elif axis is None:
    data1 = data[:size1]; data2 = data[size1:]
  else:
    data1, data2 = np.split(data, [size1], axis=axis)
  # pre-process data
  if lstandardize: 
    data1 = standardize(data1, axis=axis, lcopy=False) # apply_stat_test_2samp alread
    data2 = standardize(data2, axis=axis, lcopy=False) #   makes a copy, no need here
  if lsmooth:
    window_len = min(data1.size,window_len) # automatically shring window
    data1 = smooth(data1, window_len=window_len, window=window)
    data2 = smooth(data2, window_len=window_len, window=window)
  if ldetrend:
    data1 = detrend(data1); data2 = detrend(data2)
  # apply test
  rho, pval = myss.spearmanr(data1, data2, axis=axis, dof=dof)
  # select output
  if lrho and lpval: 
    return np.concatenate((rho.reshape(rho.shape+(1,)),pval.reshape(pval.shape+(1,))), axis=pval.ndim)
  elif lrho: return rho
  elif lpval: return pval
  else: raise ArgumentError  


# generic applicator function for 2 sample statistical tests
def apply_stat_test_2samp(sample1, sample2, fct=None, axis=None, axis_idx=None, name=None, laax=True, 
                          lflatten=False, fillValue=None, lpval=True, lrho=False, asVar=None,
                          lcheckVar=True, lcheckAxis=True, pvaratts=None, rvaratts=None, **kwargs):
  ''' Apply a bivariate statistical test or function to two sample Variables and return the result 
      as a Variable object; the function will be applied along the specified axis or over flattened arrays. 
      This function can return both, the p-value and the function result (other than the p-value). '''
  # some input checking
  if lflatten and axis is not None: raise ArgumentError
  if not lflatten and axis is None: 
    if sample2.ndim > 1 and sample2.ndim > 1: raise ArgumentError
    else: lflatten = True # treat both as flat arrays...
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
  for sample,lvar in ((sample1,lvar1),(sample2,lvar2)):
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
    lmasked = ( lvar and sample.masked ) or ( not lvar and isinstance(sample, np.ma.MaskedArray) )
    if lmasked and fillValue is None: fillValue = np.NaN
#       if np.issubdtype(sample.dtype,np.integer): fillValue = 0
#       elif np.issubdtype(sample.dtype,np.inexact): fillValue = np.NaN
#       else: raise NotImplementedError
  # check that dtype and dimensions are equal
  if sample1.dtype.kind != sample2.dtype.kind: raise TypeError, "Samples need to have same dtype."
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
  if asVar:
    if lpval:
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
    if lrho:
      varatts = dict()
      if lvar1 and lvar2:
        varatts['name'] = name or '{:s}_{:s}_rho'.format(sample1.name,sample2.name)
        varatts['long_name'] = 'Correlation Coefficient of {:s} and {:s}'.format(sample1.name.title(),sample1.name.title())
      else:
        if not name: varatts['name'] = 'rho'; varatts['long_name'] = 'Correlation Coefficient'
        else: varatts['name'] = name; varatts['long_name'] = 'Correlation Coefficient ({:s})'.format(name)
      varatts['units'] = '' # p-value / probability
      if rvaratts is not None: varatts.update(rvaratts)
      rvaratts = varatts.copy()
      rvarplot = getPlotAtts(name=name or rvaratts['name'], units='') # infer meta data from plot attributes
  # prepare data
  def preprocess(sample, axis_idx):
    ''' helper function to pre-process each sample '''
    # get data
    if isinstance(sample,Variable): 
      sample = sample.getArray(unmask=False, fillValue=None, copy=True)
    if isinstance(sample,ma.MaskedArray): 
      if np.issubdtype(sample.dtype,np.integer): data = sample.astype(np.float_)
      else: data = sample.copy() 
      data = data.filled(fillValue)
    elif isinstance(sample,np.ndarray): data = sample.copy() 
    else: raise TypeError, sample.__class__
    # roll sampel axis to end (or flatten)
    if lflatten: data = data.ravel()
    else: data = np.rollaxis(data, axis=axis_idx, start=sample.ndim)
    return data
  data1 = preprocess(sample1, axis_idx1)
  data2 = preprocess(sample2, axis_idx2)
  assert lflatten or data1.shape[:-1] == data2.shape[:-1]
  # apply test (serial)
  if lflatten:
    assert data1.ndim == 1 and data2.ndim == 1
    # merge sample arrays, save dividing index 'size1' (only one argument array per point along axis)
    data_array = np.concatenate((data1, data2), axis=0) 
    res = fct(data_array, size1=data1.size) # evaluate function
    # disentagle results
    if lrho and lpval:
      rvar, pvar = res[0],res[1]
    elif lpval: pvar = res
    elif lrho: rvar = res
    if asVar: # doesn't work here
      raise NotImplementedError, "Cannot return a single scalar as a Variable object."
  # apply test (parallel)
  else: 
    axis_idx = data1.ndim-1; size1 = data1.shape[-1] # shorcuts
    # merge sample arrays, save dividing index 'size1' (only one argument array per point along axis)
    data_array = np.concatenate((data1, data2), axis=axis_idx) 
    # select test and set parameters
    fct = functools.partial(fct, size1=size1)
    res = apply_along_axis(fct, axis_idx, data_array, chunksize=500000//len(data_array), laax=laax) # apply test in parallel, distributing the data
    # handle masks etc.
    if (lvar1 and sample1.masked) or (lvar2 and sample1.masked): 
      res = ma.masked_invalid(res, copy=False) 
      # N.B.: comparisons with NaN always evaluate to False!
      if fillValue is not None:
        res = ma.masked_equal(res, fillValue)
      ma.set_fill_value(res,fillValue)
    if lrho and lpval:
      assert res.ndim == sample1.ndim
      assert res.shape == rshape+(2,)
      res = np.rollaxis(res, axis=res.ndim-1, start=0)
      rvar = res[0,:]; pvar = res[1,:]
    else:
      assert res.ndim == sample1.ndim-1
      assert res.shape == rshape
      if lpval: pvar = res
      elif lrho: rvar = res
    if asVar:
      axes = sample1.axes[:axis_idx1] + sample1.axes[axis_idx1+1:]
      if lpval: pvar = Variable(data=pvar, axes=axes, atts=pvaratts, plot=pvarplot)
      if lrho: rvar = Variable(data=rvar, axes=axes, atts=rvaratts, plot=rvarplot)
  # return results
  if lrho and lpval: return rvar, pvar
  elif lpval: return pvar
  elif lrho: return rvar
  else: ArgumentError

## distribution variable classes 

# dictionary with distribution definitions for common variables  
var_dists = dict() # tuple( dist_name, kwargs)
# var_dists['CDD'] = ('gumbel_r', dict())
# var_dists['CWD'] = ('gumbel_r', dict())
var_dists['MaxWaterflx_7d'] = ('gumbel_r', dict())
# var_dists['CDD'] = ('genextreme', dict())
# var_dists['CWD'] = ('genextreme', dict())
variable_distributions = var_dists # alias for imports

# function to return an appropriate distribution for a variable
def defVarDist(var, var_dists=None):
  ''' return an appropriate distribution for a variable, based on some heuristics '''
  if not isinstance(var, Variable): raise TypeError
  elif isinstance(var, DistVar): raise TypeError, "Variable is already a DistVar!"
  varname, units = var.name, var.units
  # check explicit definition first
  if var_dists is None: var_dists = variable_distributions
  if varname in var_dists: 
    dist, dist_args = var_dists[varname]
  # now, apply heuristics
  elif varname[:3] in ('Min','Max'):
    dist, dist_args = 'genextreme', dict(ic_shape=0)
  elif units in ('mm/month','mm/day','mm/s','kg/m^2/s'):
    #dist, dist_args = ('gumbel_r', dict())
    dist, dist_args = 'genextreme', dict(ic_shape=0)
  elif units in ('C','K','Celsius','Kelvin'):
    dist, dist_args = 'norm', dict()
  elif units == 'days': # primarily consecutive dry/wet days
    dist, dist_args = 'kde', dict()
  else: # fallback
    dist, dist_args = 'norm', dict()
  # return distribution definition
  return dist, dist_args

# convenience function to generate a DistVar from another Variable object
def asDistVar(var, axis='time', dist=None, lflatten=False, name=None, atts=None, var_dists=None,
              lsuffix=False, asVar=True, lcheckVar=True, lcheckAxis=True, **kwargs):
  ''' generate a DistVar of type 'dist' from a Variable 'var'; use dimension 'axis' as sample axis '''
  if not isinstance(var,Variable): raise VariableError
  # this really only works for numeric types
  if var.dtype.kind in ('S',): 
    if lcheckVar: raise VariableError, "Distributions don't work with string Variables!"
    else: return None
  if not var.data: var.load()
  # select appropriate distribution, if not specified 
  if dist is None or dist.lower() == 'default':
    dist, dist_args = defVarDist(var, var_dists= variable_distributions if var_dists is None else var_dists)
    dist_args.update(kwargs) # kwargs have precedence
  else: dist_args = kwargs
  dist = dist.lower()
  # create some sensible default attributes
  varatts = dict()
  varatts['name'] = name or ( '{:s}_{:s}'.format(var.name,dist) if lsuffix else var.name)
  varatts['long_name'] = "'{:s}'-distribution of {:s}".format(dist,var.atts.get('long_name',var.name.title()))
  varatts['units'] = var.units # these will be the sample units, not the distribution units
  if atts is not None: varatts.update(atts)
  # figure out axes
  if var.ndim == 1: lflatten = True
  if lflatten:
    iaxis = None; axes = None
  else:
    if not var.hasAxis(axis):
      if lcheckAxis: raise AxisError
      else: return None
    iaxis=var.axisIndex(axis)
    axes = var.axes[:iaxis]+var.axes[iaxis+1:] # i.e. without axis/iaxis 
  # create DistVar instance
  if dist.lower() == 'kde': dvar = VarKDE(samples=var.data_array, axis=iaxis, axes=axes, 
                                          lflatten=lflatten, atts=varatts, **dist_args)
  elif hasattr(ss,dist):
    dvar = VarRV(dist=dist, samples=var.data_array, axis=iaxis, axes=axes, 
                 lflatten=lflatten, atts=varatts, **dist_args)
  else:
    raise AttributeError, "Distribution '{:s}' not found in scipy.stats.".format(dist)
  # return DistVar instance
  return dvar

# base class for distributions 
class DistVar(Variable):
  '''
    A base class for variables that represent a distribution at each grid point. This class implements
    access to (re-)samplings of the distribution, as well as histogram- and CDF-type grid-based 
    representations of the distributions. All operations add an additional (innermost) dimension;
    representations of the distribution require a support vector, sampling only a sample size.. 
  '''
  dist_type = '' # name of the distribution type
  paramAxis = None # axis for distribution parameters (None is distribution objects are stored)

  def __init__(self, name=None, units=None, axes=None, samples=None, params=None, axis=None, dtype=None,
               lflatten=False, masked=None, mask=None, fillValue=None, atts=None, ldebug=False, 
               lbootstrap=False, nbs=1000, **kwargs):
    '''
      This method creates a new DisVar instance from data and parameters. If data is provided, a sample
      axis has to be specified or the last (innermost) axis is assumed to be the sample axis.
      An estimation/fit will be performed at every grid point and stored in an array.
      Note that 'dtype' and 'units' refer to the sample data, not the distribution.
    '''
    # if parameters are provided
    if params is not None:
      if samples is not None: raise ArgumentError 
      if isinstance(params,(np.ndarray, list, tuple)):
        params = np.asanyarray(params, order='C')
      else: TypeError
      if axis is not None and not axis == params.ndim-1:
        params = np.rollaxis(params, axis=axis, start=params.ndim) # roll sample axis to last (innermost) position
      if dtype is None: dtype = np.dtype('float') # default sample dtype
      if axes is None: axes = ()
      if len(axes) == params.ndim-1:
        if any(ax.name.startswith('params_') for ax in axes): raise AxisError
      elif len(axes) != params.ndim: raise AxisError
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
          samples = np.where(samples.mask, np.NaN, np.asarray(samples, dtype=np.float))
        else: 
          samples = samples.filled(np.NaN)
      else: masked = False
      # make sure sample axis is last
      if isinstance(samples,np.ndarray): 
        samples = np.asarray(samples, dtype=np.float, order='C')
      if axis is not None and not axis == samples.ndim-1:
        samples = np.rollaxis(samples, axis=axis, start=samples.ndim) # roll sample axis to last (innermost) position
        # N.B.: we may also need some NaN/masked-value handling here...
      # check axes
      if lflatten:
        if axes is not None: raise AxisError, "'axes' keyword can not be used with lflatten=True"
        samples = samples.ravel()
        axes = tuple() # empty tuple
      else:
        if len(axes) == samples.ndim: axes = axes[:axis]+axes[axis+1:]
        elif len(axes) != samples.ndim-1: raise AxisError
      ## add bootstrap axis and generate bootstrap samples
      if lbootstrap:
        # create and add bootstrap axis
        bsatts = dict(name='bootstrap',units='',long_name='Bootstrap Samples')
        bsax = Axis(coord=np.arange(nbs), atts=bsatts)
        axes = (bsax,) + axes # add this axis as outer-most
        shape = (nbs,) + samples.shape
        # resample the samples (nbs times)
        bootstrap = np.zeros(shape, dtype=samples.dtype) # allocate memory
        bootstrap[0,:] = samples # first element is the real sample data
        sz = samples.size; sshp = samples.shape  
        for i in xrange(1,nbs):
          idx = np.random.randint(sz, size=sshp) # select random indices
          bootstrap[i,:] = samples[idx] # write random sample into array
        samples = bootstrap
        # N.B.: from here one everything should proceed normally, with the extra bootstrap axis in the 
        #       resulting DistVar object; obtain confidence intervalls as percentiles along this axis
        # N.B.: In order to save memory, this could be inplemented more efficiently within the fit-       
        #       functions, so that the resampling only occurs immediately before fitting.
        #       Essentially, there would be two separate estimations, one with the original sample 
        #       vector, and a given number of estimations with prior resampling; these would then be 
        #       stacked along the bootstrap axis.
        # N.B.: Performing resampling and estimation iteratively, one by one, would also save memory, 
        #       but prevent parallelization of the bootstrap process.
      # estimate distribution parameters
      params = self._estimate_distribution(samples, ldebug=ldebug, **kwargs)
      # N.B.: the method estimate() should be implemented by specific child classes      
      # N.B.: 'ic' are initial guesses for parameter values; 'kwargs' are for the estimator algorithm 
    # sample fillValue
    if fillValue is None: fillValue = np.NaN
    # generate new parameter axis
    if params.ndim > 0: params_name = 'params_#{:d}'.format(params.shape[-1]) # can happen with single KDE
    if params.ndim == len(axes)+1:
      patts = dict(name=params_name,units='',long_name='Distribution Parameters')
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
    self._masked = masked
    self.fillValue = fillValue # was overwritten by parameter array fill_value
    assert self.masked == masked
    self.dtype = dtype # property is overloaded in DistVar
    # N.B.: in this variable dtype and units refer to the sample data, not the distribution!
    if params.ndim > 0 and self.hasAxis(params_name):
      self.paramAxis = self.getAxis(params_name) 
    # aliases
    self.pdf = self.PDF
    self.cdf = self.CDF
    
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
    
  def replaceAxis(self, oldaxis, newaxis=None):
    ''' Replace axis and handle parameter axis as well '''
    # check if parameter axis is being replaced
    if isinstance(oldaxis, Axis): 
      lparamAxis = oldaxis == self.paramAxis
    else:
      lparamAxis = oldaxis == self.paramAxis.name
    # call parent method and replace axis
    ec = super(DistVar,self).replaceAxis(oldaxis, newaxis=newaxis)
    # assign new paramter axis
    if lparamAxis: self.paramAxis = newaxis
    return ec
  
  def copy(self, deepcopy=False, **newargs): # this methods will have to be overloaded, if class-specific behavior is desired
    ''' A method to copy the Variable with just a link to the data. '''
    if deepcopy:
      var = self.deepcopy( **newargs)
    else:
      # N.B.: don't pass name and units as they just link to atts anyway, and if passed directly, they overwrite user atts
      assert len(self.axes) == self.data_array.ndim
      args = dict(axes=self.axes, params=self.data_array, dtype=self.dtype, fillValue=self.fillValue,
                  masked=self.masked, atts=self.atts.copy(), plot=self.plot.copy())
      if 'data' in newargs:
        newargs['params'] = newargs.pop('data') # different name for the "data"
      args.update(newargs) # apply custom arguments (also arguments related to subclasses)      
      var = self.__class__(**args) # create a new basic Variable instance
    # N.B.: this function will be called, in a way, recursively, and collect all necessary arguments along the way
    return var

  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, ldebug=False, **kwargs):
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
      if self.name.endswith('_'+self.dist_type):
        basename = self.name[:self.name.find('_'+self.dist_type)] # remove _dist_name
      else: basename = self.name 
      #basename = str('_').join(self.name.split('_')[:-1],)
      if dist_type in ('ppf', 'isf'):
        if not lsqueezed:
          if support_axis is None:
            # generate new histogram axis
            axplotatts = getPlotAtts(name='quant', units='') # infer meta data from plot attributes
            daxatts = self.paramAxis.atts.copy() # original Axis also has dataset reference/name
            daxatts['name'] = '{:s}_quants'.format(basename) # remove suffixes (like _dist)
            daxatts['units'] = '' # quantiles have no units
            daxatts['long_name'] = '{:s} Quantile Axis'.format(self.name.title())
            if axatts is not None: daxatts.update(axatts)
            support_axis = Axis(coord=support, atts=daxatts, plot=axplotatts)
          else:
            if axatts is not None: support_axis.atts.update(axatts) # just update attributes
        # generate new histogram variable
        plotatts = getPlotAtts(name=basename, units=self.units) # infer meta data from plot attributes
        dvaratts = self.atts.copy() # this is either density or frequency
        dvaratts['name'] = name or '{:s}'.format(basename)
        dvaratts['long_name'] = '{:s}'.format(plotatts.name)
        dvaratts['units'] = self.units or plotatts.units
      else:
        if not lsqueezed:
          if support_axis is None:
            # generate new histogram axis
            axplotatts = getPlotAtts(name=basename, units=self.units) # infer meta data from plot attributes
            daxatts = self.atts.copy() # variable values become axis
            daxatts['name'] = '{:s}_bins'.format(basename) # remove suffixes (like _dist)
            daxatts['units'] = self.units # the histogram axis has the same units as the variable
            daxatts['long_name'] = '{:s} Axis'.format(self.name.title())
            if axatts is not None: daxatts.update(axatts)
            support_axis = Axis(coord=support, atts=daxatts, plot=axplotatts)
          else:
            if axatts is not None: support_axis.atts.update(axatts) # just update attributes
        # generate new histogram variable
        plotatts = getPlotAtts(name=dist_type, units='') # infer meta data from plot attributes
        dvaratts = self.paramAxis.atts.copy() if self.paramAxis else self.atts.copy() 
        # N.B.: we need tp transfer some dataset information; the original Axis instance also has the
        #       dataset reference/name, but VarKDE has no parameter axis...
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
    nonans = np.invert(np.isnan(sample)) # test for NaN's
    if not np.any(nonans):
      print('all NaN'); res = None
    elif np.sum(nonans) < 3: 
      print('NaN'); res = None # require at least plen non-NaN points 
    else:
      sample = sample[nonans] # remove NaN's
      if np.all(sample == sample[0]):
        print('equal'); res = None
      elif isinstance(sample,ma.MaskedArray) and np.all(sample.mask):
        print('masked'); res = None
      else:
        try: 
          res = ss.kde.gaussian_kde(sample, **kwargs)
        except LinAlgError:
          print('linalgerr'); res = None
  else:
    nonans = np.invert(np.isnan(sample)) # test for NaN's
    if not np.any(nonans): 
      res = None # require at least plen non-NaN points 
    else:
      sample = sample[nonans] # remove NaN's
      res = ss.kde.gaussian_kde(sample, **kwargs)
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
  dist_type = 'kde'
  
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, ic_shape=None, ic_args=None, ic_loc=None, ic_scale=None, ldebug=False, **kwargs):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    fct = functools.partial(kde_estimate, ldebug=ldebug, **kwargs)
    kernels = apply_along_axis(fct, samples.ndim-1, samples, chunksize=100000//len(samples)).squeeze()
    assert samples.shape[:-1] == kernels.shape
    # return an array of kernels
    return kernels

  # distribution-specific method; should be overloaded by subclass
  def _density_distribution(self, support):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    n = len(support); fillValue = self.fillValue or np.NaN
    data = self.data_array.reshape(self.data_array.shape+(1,)) # expand
    fct = functools.partial(kde_eval, support=support, n=n, fillValue=fillValue)
    pdf = apply_along_axis(fct, self.ndim, data, chunksize=100000//n)
    assert pdf.shape == self.shape + (len(support),)
    return pdf
  
  # distribution-specific method; should be overloaded by subclass
  def _resample_distribution(self, support):
    ''' draw n samples from the distribution for each grid point and return as ndarray '''
    n = len(support) # in order to use _get_dist(), we have to pass a dummy support
    fillValue = self.fillValue or np.NaN # for masked values
    data = self.data_array.reshape(self.data_array.shape+(1,)) # expand
    fct = functools.partial(kde_resample, support=support, n=n, fillValue=fillValue, dtype=self.dtype)
    samples = apply_along_axis(fct, self.ndim, data, chunksize=100000//n)
    assert samples.shape == self.shape + (n,)
    assert np.issubdtype(samples.dtype, self.dtype)
    return samples
  
  # distribution-specific method; should be overloaded by subclass
  def _cumulative_distribution(self, support):
    ''' integrate PDF over given support to produce a CDF and return as ndarray '''
    n = len(support); fillValue = self.fillValue or np.NaN
    data = self.data_array.reshape(self.data_array.shape+(1,))
    fct = functools.partial(kde_cdf, support=support, n=n, fillValue=fillValue)
    cdf = apply_along_axis(fct, self.ndim, data, chunksize=500000//n)
    assert cdf.shape == self.shape + (len(support),)
    return cdf
  
## VarRV subclass and helper functions

# N.B.: need to define helper functions outside of class definition, otherwise pickle / multiprocessing 
#       will fail... need to fix arguments with functools.partial

# persistent variables with most recent successful fit parameters
global_loc   = None # location parameter ("mean")
global_scale = None # scale parameter ("standard deviation")
global_shape = None # single shape parameter
global_args  = None # multiple shape parameters
# N.B.: globals are only visible within the process, so this works nicely with multiprocessing
# estimate RV from sample vector
def rv_fit(sample, dist_type=None, ic_shape=None, ic_args=None, ic_loc=None, ic_scale=None, plen=None, lpersist=False, ldebug=False, **kwargs):
  nonans = np.invert(np.isnan(sample)) # test for NaN's
  if np.sum(nonans) < plen: 
    res = (np.NaN,)*plen # require at least plen non-NaN points 
    if ldebug:
      if not np.any(nonans):
        print('all NaN'); res = (np.NaN,)*plen
      elif np.sum(nonans) < plen: 
        print('NaN'); res = (np.NaN,)*plen # require at least plen non-NaN points 
  else:
    sample = sample[nonans] # remove NaN's
    if ldebug:
      # fit will fail under these conditions, but they should not occur at this point
      if np.all(sample == sample[0]):
        print('equal'); res = (np.NaN,)*plen
      elif isinstance(sample,ma.MaskedArray) and np.all(sample.mask):
        print('masked'); res = (np.NaN,)*plen
    # begin actual computation
    try:
      if lpersist:
        global global_loc, global_scale, global_shape, global_args # load globals
        if global_loc is not None:   ic_loc   = global_loc
        if global_scale is not None: ic_scale = global_scale
      # estimate location and scale, if not specified
      if ic_loc is None: ic_loc = sample.mean()
      if ic_scale is None: ic_scale = sample.std()
      if ic_shape is None: # loc + scale / shape == 0 for GEV family
        ic_shape = -1. * ic_loc / ic_scale if ic_scale else 0.  
      # N.B.: for the GEV dist., ( loc + scale / shape ) is the left end (begin) of the support
      # start parameter estimation
      if plen == 2: # only location and shape (e.g. normal or Gumbel distributions)
        res = getattr(ss,dist_type).fit(sample, loc=ic_loc, scale=ic_scale, **kwargs)
        if lpersist: global_loc = res[0]; global_scale = res[1] # update first guess
      elif plen == 3: # additional shape parameter (e.g. Generalized Extreme Value and Pareto distributions)
        if lpersist and global_shape is not None: ic_shape = global_shape
        res = getattr(ss,dist_type).fit(sample, shape=ic_shape, loc=ic_loc, scale=ic_scale, **kwargs)
        if lpersist: global_shape = res[0]; global_loc = res[1]; global_scale = res[2] # update first guess
      elif plen > 3: # everything with more than one shape parameter...
        if lpersist and global_args is not None:  ic_args  = global_args
        res = getattr(ss,dist_type).fit(sample, *ic_args, loc=ic_loc, scale=ic_scale, **kwargs)
        if lpersist: global_args = res[:-2]; global_loc = res[-2]; global_scale = res[-1] # update first guess
      else: raise NotImplementedError
    except LinAlgError:
      if ldebug: print('linalgerr')
      res = (np.NaN,)*plen
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
  dist_class = None # the scipy RV distribution object
  dist_type = ''   # name of the distribution
  
  # initial guesses for shape parameter
  def __init__(self, dist='', ic_shape=None, ic_args=None, ic_loc=None, ic_scale=None, **kwargs):
    ''' initialize a random variable distribution of type 'dist' '''
    # some aliases
    if dist.lower() in ('genextreme', 'gev'): dist = 'genextreme' 
    elif dist.lower() in ('genpareto', 'gpd'): dist = 'genpareto' # 'pareto' is something else!
    # look up module & set distribution
    if dist == '':
      raise ArgumentError, "No distribution 'dist' specified!"
    elif dist in ss.__dict__: 
      self.dist_class = getattr(ss,dist) # distribution class (from SciPy)
      self.dist_type = dist # name of distribution (in SciPy)
    else: 
      raise ArgumentError, "No distribution '{:s}' in module scipy.stats!".format(dist)
    # N.B.: the distribution info will be available to the _estimate_distribution-method
    # initialize distribution variable
    super(VarRV,self).__init__(ic_shape=ic_shape, ic_args=ic_args, ic_loc=ic_loc, ic_scale=ic_scale, **kwargs)
    # N.B.: ic-parameters and kwargs are passed one to _estimate_distribution-method
  
  def copy(self, deepcopy=False, **newargs): # this methods will have to be overloaded, if class-specific behavior is desired
    ''' A method to copy the Variable with just a link to the data. '''
    #if 'dist' in newargs: raise ArgumentError # can happen through deepcopy
    newargs['dist'] = self.dist_type
    return super(VarRV,self).copy(deepcopy=deepcopy, **newargs)
  
  def __getattr__(self, attr):
    ''' use methods of from RV distribution through _get_dist wrapper '''
    if hasattr(self.dist_class, attr):
      attr = functools.partial(self._get_dist, self._compute_distribution, attr, rv_fct=attr)
    return attr
  
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, ic_shape=None, ic_args=None, ic_loc=None, ic_scale=None, lpersist=False, ldebug=False, **kwargs):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    if lpersist: # reset global parameters
      global_loc   = None # location parameter ("mean")
      global_scale = None # scale parameter ("standard deviation")
      global_shape = None # single shape parameter
      global_args  = None # multiple shape parameters
    plen = self.dist_class.numargs + 2 # infer number of parameters
    fct = functools.partial(rv_fit, ic_shape=ic_shape, ic_args=ic_args, ic_loc=ic_loc, ic_scale=ic_scale, plen=plen, 
                            dist_type=self.dist_type, lpersist=lpersist, ldebug=ldebug, **kwargs)
    params = apply_along_axis(fct, samples.ndim-1, samples, chunksize=int(300//plen//len(samples)))
    if lpersist: # reset global parameters 
      global_loc   = None # location parameter ("mean")
      global_scale = None # scale parameter ("standard deviation")
      global_shape = None # single shape parameter
      global_args  = None # multiple shape parameters
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
      dist = apply_along_axis(fct, self.ndim-1, self.data_array, chunksize=100)
      assert dist.shape[:-1] == self.shape[:-1]
    elif  len(args) == 1 and rv_fct == 'moment':
      raise NotImplementedError
      fillValue = self.fillValue or np.NaN
      fct = functools.partial(rv_stats, dist_type=self.dist_type, fct_type=rv_fct, fillValue=fillValue, **kwargs)
      dist = apply_along_axis(fct, self.ndim-1, self.data_array, chunksize=100)
      assert dist.shape[:-1] == self.shape[:-1]
    elif len(args) == 1:
      support = args[0]
      assert isinstance(support, np.ndarray)
      n = len(support); fillValue = self.fillValue or np.NaN
      fct = functools.partial(rv_eval, dist_type=self.dist_type, fct_type=rv_fct, 
                              support=support, n=n, fillValue=fillValue, **kwargs)
      dist = apply_along_axis(fct, self.ndim-1, self.data_array, chunksize=10000//n)
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
    samples = apply_along_axis(fct, self.ndim-1, self.data_array, chunksize=100000//n)
    assert samples.shape == self.shape[:-1] + (n,)
    assert np.issubdtype(samples.dtype, self.dtype)
    return samples
  
  # rescale the distribution (change the parameters)
  def rescale(self, reference=None, lflatten=True, shape=None, loc=None, scale=None, axis_idx=None, 
              fillValue=None, linplace=False, asVar=True, lcheckVar=True, lcheckAxis=True):
    ''' rescale the distribution parameters with given values '''
    if linplace and asVar: raise ArgumentError
    # figure out array order 
    sax = self.ndim-1
    if len(self.paramAxis) == 2: iloc=0; iscale=1; ishape=None
    elif len(self.paramAxis) == 3: ishape=0; iloc=1; iscale=2
    else: raise NotImplementedError
    if shape is not None and ishape is None: raise ArgumentError
    # roll parameter axis to the front
    if linplace: data_array = np.rollaxis(self.data_array, axis=sax, start=0)
    else: data_array = np.rollaxis(self.data_array.copy(), axis=sax, start=0)
    # pre-process input
    if reference is not None:
      # get properly formatted sample data
      sample_data = self._extractSampleData(reference, axis_idx=axis_idx, fillValue=fillValue, lcheckVar=True, lcheckAxis=True)
      if lflatten: 
        sample_data = sample_data.ravel() 
        sax = None # compute moments over flat array, not along the sample axis (would fail anyway)
      # N.B.: if the DistVar is already "flat", lflatten is unneccessary, since _extractSampleData will flatten the sample
      # estimate location and scale from a given reference
      if loc is None: 
        norm = np.nanmean(data_array[iloc,:]).ravel() if lflatten else data_array[iloc]
        loc = np.nanmean(sample_data, axis=sax) / norm
      if scale is None: 
        norm = np.nanmean(data_array[iscale,:]).ravel() if lflatten else data_array[iscale]
        scale = np.nanstd(sample_data, axis=sax) / norm
    # apply scaling
    lone = data_array.ndim == 1 
    if loc is not None: 
      if lone: data_array[iloc] *= loc; data_array[iscale] *= loc # need to scale variance as well!
      else: data_array[iloc,:] *= loc; data_array[iscale,:] *= loc # need to scale variance as well!
    if scale is not None: 
      if lone: data_array[iscale] *= scale
      else: data_array[iscale,:] *= scale
    if shape is not None: 
      if lone: data_array[ishape] *= shape 
      else: data_array[ishape,:] *= shape 
    # N.B.: rolling back axes should not be necessary, since np.rollaxis only returns a view and all operations are in-place
    # create variable, if desired
    if asVar:
      data_array = np.rollaxis(data_array, axis=0, start=self.ndim) # roll parameter axis to the back
      rsvar = self.copy(data=data_array)
      # add record of scale factors
      rsvar.atts['loc_factor'] = loc.ravel()[0] if isinstance(loc,np.ndarray) else loc
      rsvar.atts['scale_factor'] = scale.ravel()[0] if isinstance(scale,np.ndarray) else scale
      rsvar.atts['shape_factor'] = shape.ravel()[0] if isinstance(shape,np.ndarray) else shape 
    else:
      # alternatively, return scale factors for further usage
      if ishape is None: rsvar = loc, scale
      else: rsvar = shape, loc, scale
    # return
    return rsvar

  # convenience function to get properly formatted sample data from a sample argument
  def _extractSampleData(self, sample, axis_idx=None, fillValue=None, lcheckVar=True, lcheckAxis=True):
    ''' Check if a sample is consistent with the distribution attributes and return a properly 
        formatted sample array with the same shape as the distribution variable and the sample 
        dimension shifted to the back (all remaining sample dimensions are flattened). '''
    # check input
    if sample.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Statistical tests do not work with string Variables!"
      else: return None
    # choose a fillValue, because np.histogram does not ignore masked values but does ignore NaNs
    if fillValue is None:
      if np.issubdtype(self.dtype,np.integer): fillValue = 0
      elif np.issubdtype(self.dtype,np.inexact): fillValue = np.NaN
      else: raise NotImplementedError
    # if sample is a variable, check and figure out sample axes   
    sax = self.ndim-1
    if isinstance(sample,Variable):
      assert self.axisIndex(self.paramAxis) == sax
      for ax in self.axes[:-1]:
        if not sample.hasAxis(ax.name): # last is parameter axis
          if lcheckAxis: raise AxisError, "Sample Variable needs to have a '{:s}' axis.".format(ax.name)
          else: return None
        if len(sample.getAxis(ax.name)) != len(ax): # last is parameter axis
          if lcheckAxis: raise AxisError, "Axis '{:s}' in Sample and DistVar have different length!".format(ax.name)
          else: return None
      sample_data = sample.getArray(unmask=True, fillValue=fillValue, copy=True) # actual data (will be reordered)
      # reorder axes so that shape is compatible
      z = self.ndim-1; iaxes = [-1]*sample.ndim 
      for iax,ax in enumerate(sample.axes):
        if self.hasAxis(ax.name): # adopt order from self
          iaxes[self.axisIndex(ax.name)] = iax
        else: # append at the end
          iaxes[z] = iax; z += 1 # increment
      assert z == sample.ndim
      assert all(iax >= 0 for iax in iaxes)
      sample_data = np.transpose(sample_data, axes=iaxes)
    else:
      if isinstance(sample,ma.MaskedArray): sample_data = sample.filled(fillValue)
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
    # return properly formatted sample data
    return sample_data

  # Kolmogorov-Smirnov Test for goodness-of-fit
  def kstest(self, sample, name=None, axis_idx=None, lstatistic=False, 
             fillValue=None, ignoreNaN=True, N=20, alternative='two-sided', mode='approx', 
             asVar=True, lcheckVar=True, lcheckAxis=True, pvaratts=None, **kwargs):
    ''' apply a Kolmogorov-Smirnov Test to the sample data, based on this distribution '''
    # check input
    if self.dtype.kind in ('S',): 
      if lcheckVar: raise VariableError, "Statistical tests does not work with string Variables!"
      else: return None
    if lstatistic: raise NotImplementedError, "Return of test statistic is not yet implemented; only p-values are returned."
    # get properly formatted sample data
    sax = self.ndim-1
    sample_data = self._extractSampleData(sample, axis_idx=axis_idx, fillValue=fillValue, lcheckVar=True, lcheckAxis=True)
    # apply test function (parallel)
    fct = functools.partial(rv_kstest, nparams=len(self.paramAxis), dist_type=self.dist_type, ignoreNaN=ignoreNaN, N=N, alternative=alternative, mode=mode)
    data_array = np.concatenate((self.data_array, sample_data), axis=sax) # merge params and sample arrays (only one argument array per point along axis) 
    pval = apply_along_axis(fct, sax, data_array, chunksize=100000//len(data_array)) # apply test in parallel, distributing the data
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
      assert self.axisIndex(self.paramAxis) == sax
      pvar = Variable(data=pval, axes=self.axes[:-1], atts=varatts, plot=plotatts)
    else: pvar = pval
    # return results
    return pvar
    