'''
Created on Dec 12, 2014

Some Variable subclasses for handling distribution functions over grid points. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import scipy.stats as ss
# internal imports
from geodata.base import Variable, Axis
from geodata.misc import DataError, ArgumentError, VariableError, AxisError, DistVarError
from plotting.properties import getPlotAtts
from numpy.linalg.linalg import LinAlgError


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

  def __init__(self, name=None, units=None, axes=None, samples=None, params=None, axis=None, dtype=None, 
               mask=None, fillValue=None, atts=None):
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
    # if samples are provided
    if samples is not None:
      if params is not None: raise ArgumentError 
      if isinstance(samples,np.ndarray): 
        samples = np.asarray(samples, dtype=dtype, order='C')
      if axis is not None and not axis == samples.ndim-1:
        samples = np.rollaxis(samples, axis=axis, start=samples.ndim) # roll sample axis to last (innermost) position
      if dtype is None: dtype = samples.dtype
      elif not np.issubdtype(samples.dtype,dtype): raise DataError 
        # N.B.: we may also need some NaN/masked-value handling here...
      # check axes
      if len(axes) != samples.ndim-1: raise AxisError
      # estimate distribution parameters
      # N.B.: the method estimate() should be implemented by specific child classes
      params = self._estimate_distribution(samples) # this is used as "data"
    # generate new parameter axis
    if params.ndim == len(axes)+1:
      patts = dict(name='params',units='',long_name='Distribution Parameters')
      paraAxis = Axis(coord=np.arange(params.shape[-1]), atts=patts)
      axes = axes + (paraAxis,)
    # check diemnsions, but leave dtype open to be inferred from data
    assert params.ndim == len(axes)
    assert all(ps==len(ax) for ps,ax in zip(params.shape,axes))
    # create variable object using parent constructor
    super(DistVar,self).__init__(name=name, units=units, axes=axes, data=params, dtype=None, mask=mask, 
                                 fillValue=fillValue, atts=atts, plot=None)
    # reset dtype to sample dtype (not parameter dtype!)
    self.dtype = dtype # property is overloaded in DistVar
    # N.B.: in this variable dtype and units refer to the sample data, not the distribution! 
    
  @property
  def dtype(self):
    ''' The data type of the samlple data (inferred from initialization data). '''
    return self._dtype   
  @dtype.setter
  def dtype(self, dtype):
    self._dtype = dtype
    
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    raise NotImplementedError
  # distribution-specific method; should be overloaded by subclass
  def _sample_distribution(self, n):
    ''' draw n samples from the distribution for each grid point and return as ndarray '''
    raise NotImplementedError
  # distribution-specific method; should be overloaded by subclass
  def _compute_distribution(self, support):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    raise NotImplementedError

  # overload histogram and return a PDF (like a histogram)
  def _get_dist(self, dist_op, dist_type, support=None, support_axis=None, axis_idx=None, 
                asVar=True, name=None, axatts=None, varatts=None, **kwargs):
    ''' return a distribution function as a Variable object or ndarray '''
    if not asVar and support_axis is not None: raise ArgumentError
    if axis_idx is None: axis_idx = self.ndim
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
    if not isinstance(support,np.ndarray): raise TypeError
    # get histogram/PDF values
    dist_data = dist_op(support)
    # arrange axes
    if axis_idx != self.ndim: 
      dist_data = np.rollaxis(dist_data, axis=dist_data.ndim-1,start=axis_idx)
    # setup histogram axis and variable attributes (special case)
    if asVar:
      basename = self.name.split('_')[0]
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
      axes = self.axes[:axis_idx]+(support_axis,)+self.axes[axis_idx:]
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
    hvar = self._get_dist(self._compute_distribution, 'pdf', support=support, support_axis=support_axis, 
                          axis_idx=axis_idx, asVar=asVar, name=name, axatts=haxatts, varatts=hvaratts)
    # return results
    return hvar
  
  # overload histogram and return a PDF (like a histogram)
  def CDF(self, bins=None, asVar=True, name=None, support=None, support_axis=None, axis_idx=None, 
                caxatts=None, cvaratts=None, lnormalize=True, lflatten=False, fillValue=None, **kwargs):
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
    # get distribution values
    hvar = self._get_dist(self._cumulative_distribution, 'cdf', support=support, support_axis=support_axis, 
                          axis_idx=axis_idx, asVar=asVar, name=name, axatts=caxatts, varatts=cvaratts)
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

# Subclass of DistVar implementing Kernel Density Estimation
class VarKDE(DistVar):
  ''' A subclass of DistVar implementing Kernel Density Estimation ''' 
  
  # distribution-specific method; should be overloaded by subclass
  def _estimate_distribution(self, samples, **kwargs):
    ''' esimtate/fit distribution from sample array for each grid point and return parameters as ndarray  '''
    lmask = isinstance(samples,np.ma.MaskedArray)
    def kde_estimate(sample): 
      if lmask and np.any(samples.mask): res = None
      else:
        try: res = ss.kde.gaussian_kde(sample, **kwargs)
        except LinAlgError: res = None
      return (res,)
    kernels = np.apply_along_axis(kde_estimate, samples.ndim-1, samples).squeeze()
    assert samples.shape[:-1] == kernels.shape
    # return an array of kernels
    return kernels

  # distribution-specific method; should be overloaded by subclass
  def _compute_distribution(self, support, **kwargs):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    n = len(support); fillValue = self.fillValue or 0
    def kde_eval(kde):
      if kde[0] is None: res = np.zeros(n)+fillValue 
      else: res = kde[0].evaluate(support, **kwargs)
      return res
    data = self.data_array.reshape(self.data_array.shape+(1,))
    pdf = np.apply_along_axis(kde_eval, self.ndim, data)
    assert pdf.shape == self.shape + (len(support),)
    return pdf
  
  # distribution-specific method; should be overloaded by subclass
  def _resample_distribution(self, support, **kwargs):
    ''' draw n samples from the distribution for each grid point and return as ndarray '''
    n = len(support) # in order to use _get_dist(), we have to pass a dummy support
    fillValue = self.fillValue or 0 # for masked values
    def kde_resample(kde):
      if kde[0] is None: res = np.zeros(n)+fillValue 
      else: res = np.asarray(kde[0].resample(size=n, **kwargs), dtype=self.dtype).ravel()
      return res
    data = self.data_array.reshape(self.data_array.shape+(1,))
    samples = np.apply_along_axis(kde_resample, self.ndim, data)
    assert samples.shape == self.shape + (n,)
    assert np.issubdtype(samples.dtype, self.dtype)
    return samples
  
  # distribution-specific method; should be overloaded by subclass
  def _cumulative_distribution(self, support, **kwargs):
    ''' compute PDF at given support points for each grid point and return as ndarray '''
    n = len(support); fillValue = self.fillValue or 0
    def kde_cdf(kde):
      if kde[0] is None: res = np.zeros(n)+fillValue
      else: 
        fct = lambda s: kde[0].integrate_box_1d(-1*np.inf,s, **kwargs)
        res = np.asarray([fct(s) for s in support])
      return res
    data = self.data_array.reshape(self.data_array.shape+(1,))
    pdf = np.apply_along_axis(kde_cdf, self.ndim, data)
    assert pdf.shape == self.shape + (len(support),)
    return pdf
  

# Subclass of DistVar implementing various random variable distributions
class VarRV(DistVar): pass