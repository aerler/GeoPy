'''
Created on Feb 9, 2017

The definition of BiasCorrection classes, which actually implement the bias-correction methods. We have to keep the class
definition in a separate module, so it can be pickled and imported in other modules.

@author: Andre R. Erler, GPL v3
'''

# external imports
import collections as col
import numpy as np
# internal imports
from geodata.misc import isEqual, DataError
from geodata.netcdf import DatasetNetCDF, VarNC


# some helper stuff for validation
eps = np.finfo(np.float32).eps # single precision rounding error is good enough
Stats = col.namedtuple('Stats', ('Bias','RMSE','Corr'))

def getBCmethods(method, **bcargs):
    ''' function that returns an instance of a specific BiasCorrection child class specified as method; 
        other kwargs are passed on to constructor of BiasCorrection '''
    # decide based on method name; instantiate object
    if method.lower() == 'test':
          return BiasCorrection(**bcargs)
    elif method.lower() == 'mybc':
        return MyBC(**bcargs)
    elif method.lower() == 'delta':
        return Delta(**bcargs)
    elif method.upper() == 'SMBC':
        return SMBC(**bcargs)
    elif method.upper() == 'AABC':
        return AABC(**bcargs)
    else:
        raise NotImplementedError(method)
  
def getPickleFileName(method=None, obs_name=None, mode=None, periodstr=None, gridstr=None, domain=None, tag=None, pattern=None):
    ''' generate a name for a bias-correction pickle file, based on parameters '''
    if pattern is None: pattern = 'bias_{:s}.pickle'
    # abbreviation for data mode
    if mode == 'climatology': aggregation = 'clim'
    elif mode == 'time-series': aggregation = 'monthly'
    elif mode[-5:] == '-mean': aggregation = mode[:-5]
    else: aggregation = mode
    # string together different parameter arguments
    name = method if method else '' # initialize
    if obs_name: name += '_{:s}'.format(obs_name)
    if mode: name += '_{:s}'.format(aggregation)
    if periodstr: name += '_{:s}'.format(periodstr)
    if gridstr: name += '_{:s}'.format(gridstr)
    if domain: name += '_d{:02d}'.format(domain)
    if tag: name += '_{:s}'.format(tag)
    picklefile = pattern.format(name) # insert name into fixed pattern
    return picklefile

## classes that implement bias correction 

class BiasCorrection(object):
    ''' A parent class from which specific bias correction classes will be derived; the purpose is to provide a 
        unified interface for all different methods, similar to the train/predict paradigm in SciKitLearn. '''
    name = 'generic' # name used in file names
    long_name = 'Generic Bias-Correction' # name for printing
    varlist = None # variables that a being corrected
    _picklefile = None # name of the pickle file where the object will be stored
    
    def __init__(self, varlist=None, **bcargs):
        ''' take arguments that have been passed from caller and initialize parameters '''
        self.varlist = varlist
    
    def train(self, dataset, observations, **kwargs):
        ''' loop over variables that need to be corrected and call method-specific training function '''
        # figure out varlist
        if self.varlist is None: 
            self._getVarlist(dataset, observations) # process all that are present in both datasets        
        # loop over variables that will be corrected
        self._correction = dict()
        for varname in self.varlist:
            # get variable object
            var = dataset[varname]
            if not var.data: var.load() # assume it is a VarNC, if there is no data
            obsvar = observations[varname] # should be loaded
            if not obsvar.data: obsvar.load() # assume it is a VarNC, if there is no data
            assert var.data and obsvar.data, obsvar.data      
            # check if they are actually equal
            if isEqual(var.data_array, obsvar.data_array, eps=eps, masked_equal=True):
                correction = None
            else: 
                correction = self._trainVar(var, obsvar, **kwargs)
            # save correction parameters
            self._correction[varname] = correction
  
    def _trainVar(self, var, obsvar, **kwargs):
        ''' optimize parameters for best fit of dataset to observations and save parameters;
            this method should be implemented for each method '''
        return None # do nothing
  
    def correct(self, dataset, asNC=False, **kwargs):
        ''' loop over variables and apply correction function based on specific method using stored parameters '''
        # NetCDF datasets get special handling, so we only replace the variables we need to replace
        if isinstance(dataset,DatasetNetCDF):
            if not asNC: dataset.load() # otherwise we loose data
            bcds = dataset.copy(axesdeep=True, varsdeep=False, asNC=asNC) # make a copy, but don't duplicate data
        else: 
            asNC=False
            bcds = dataset.copy(axesdeep=True, varsdeep=False) # make a copy, but don't duplicate data
        # loop over variables that will be corrected
        for varname in self.varlist:
            if varname in dataset:
                # get variable object
                oldvar = dataset[varname].load()
                newvar = bcds[varname] # should be loaded
                if isinstance(newvar,VarNC): # the corrected variable needs to load data, hence can't be VarNC          
                    newvar = newvar.copy(axesdeep=False, varsdeep=False, asNC=False) 
                assert varname in self._correction, self._correction
                # bias-correct data and load in new variable 
                if self._correction[varname] is not None:
                    newvar.load(self._correctVar(oldvar))
                if newvar is not bcds[varname]: 
                    bcds[varname] = newvar # attach new (non-NC) var
        # return bias-corrected dataset
        return bcds
    
    def _correctVar(self, var, **kwargs):
        ''' apply bias correction to new variable and return bias-corrected data;
            this method should be implemented for each method '''
        return var.data_array # do nothing, just return input
    
    def _getVarlist(self, dataset, observations):
        ''' find all valid candidate variables for bias correction present in both input datasets '''
        varlist = []
        # loop over variables
        for varname in observations.variables.keys():
            if varname in dataset.variables and not dataset[varname].strvar:
                #if np.issubdtype(dataset[varname].dtype,np.inexact) or np.issubdtype(dataset[varname].dtype,np.integer):
                varlist.append(varname) # now we have a valid variable
        # save and return varlist
        self.varlist = varlist
        return varlist        
  
    def validate(self, dataset, observations, lprint=True, **kwargs):
        ''' apply correction to dataset and return statistics of fit to observations '''
        # apply correction    
        bcds = self.correct(dataset, **kwargs)
        validation = dict()
        # evaluate fit by variable
        for varname in self.varlist:
            # get variable object
            bcvar = bcds[varname].load()
            obsvar = observations[varname] # should be loaded
            assert bcvar.data and obsvar.data, obsvar.data
            # compute statistics if bias correction was actually performed
            if self._correction[varname] is None:
                stats = None
            else:
                delta = bcvar.data_array - obsvar.data_array
                bias = delta.mean()
                if bias < eps: bias = 0.
                rmse = np.asscalar(np.sqrt(np.mean(delta**2)))
                if rmse < eps: rmse = 0.
                corr = np.corrcoef(bcvar.data_array.flatten(),obsvar.data_array.flatten())[0,1]
                stats = Stats(Bias=bias,RMSE=rmse,Corr=corr)
            if lprint: print(varname,stats)
            validation[varname] = stats
        # return fit statistics
        self._validation = validation # also store
        return validation
    
    def picklefile(self, obs_name=None, mode=None, periodstr=None, gridstr=None, domain=None, tag=None):
        ''' generate a standardized name for the pickle file, based on arguments '''
        if self._picklefile is None:      
            self._picklefile = getPickleFileName(method=self.name, obs_name=obs_name, mode=mode, periodstr=periodstr, 
                                                 gridstr=gridstr, domain=domain, tag=tag) 
        return self._picklefile
    
    def __str__(self):
        ''' a string representation of the method and parameters '''
        text = '{:s} Object'.format(self.long_name)
        if self._picklefile is not None:
            text += '\n  Picklefile: {:s}'.format(self._picklefile) 
        return text
    
#   def __getstate__(self):
#     ''' support pickling '''
#     pickle = self.__dict__.copy()
#     # handle attributes that don't pickle
#     pass
#     # return instance dict to pickle
#     return pickle
#   
#   def __setstate__(self, pickle):
#     ''' support pickling '''
#     # handle attirbutes that don't pickle
#     pass
#     # update instance dict with pickle dict
#     self.__dict__.update(pickle)
    
    
class Delta(BiasCorrection):
    ''' A class that implements a simple grid point-based Delta-Method bias correction. '''
    name = 'Delta' # name used in file names
    long_name = 'Simple Delta-Method' # name for printing
    _ratio_units = ('mm/day','kg/m^2/s','J/m^2/s','W/m^2') # variable units that indicate ratio
      
    def _trainVar(self, var, obsvar, **kwargs):
        ''' take difference (or ratio) between observations and simulation and use as correction '''
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            delta = obsvar.data_array / var.data_array 
        else: # default behavior is differences
            delta = obsvar.data_array - var.data_array
        # return correction parameters, i.e. delta
        return delta
          
    def _correctVar(self, var, **kwargs):
        ''' use stored ratios to bias-correct the input dataset and return a new copy '''
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            data = var.data_array * self._correction[var.name]
        else: # default behavior is differences
            data = var.data_array + self._correction[var.name]    
        # return bias-corrected data (copy)
        return data


class SMBC(Delta):
    name = 'SMBC' # name used in file names
    long_name = 'Simple Monthly Mean Adjustment' # name for printing
      
    def _trainVar(self, var, obsvar, **kwargs):
        ''' take difference (or ratio) between spatially averaged observations and simulation and use as correction '''
        obsdata = obsvar.data_array; vardata = var.data_array
        if obsdata.ndim == 3:
            assert obsdata.shape[0] == 12, obsvar
            # average spatially but keep seasonal cycle (but need to keep singleton dimensions)
            for n in range(1,obsdata.ndim):
                obsdata = obsdata.mean(axis=n, keepdims=True)
                vardata = vardata.mean(axis=n, keepdims=True)
        else:
            # average spatially, assuming not time dimension - very simple!
            obsdata = obsdata.mean(); vardata = vardata.mean()
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            r = obsdata / vardata 
        else: # default behavior is differences
            r = obsdata - vardata
        # consistency check
        if obsdata.ndim == 3 and r.size != 12: 
            raise DataError(r.shape)
        elif obsdata.ndim != 3 and not np.isscalar(r):
            raise DataError(r)
        # return correction parameters (12-element vector)
        return r


class AABC(Delta):
    name = 'AABC' # name used in file names
    long_name = 'Simple Annual Average Correction' # name for printing
      
    def _trainVar(self, var, obsvar, **kwargs):
        ''' take difference (or ratio) between averaged observations and simulation and use as correction '''
        obsdata = obsvar.data_array; vardata = var.data_array
        # average spatially and temporally (over seasonal cycle) - very simple!
        obsdata = obsdata.mean(); vardata = vardata.mean()
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            r = obsdata / vardata 
        else: # default behavior is differences
            r = obsdata - vardata
        if not np.isscalar(r): raise DataError(r)
        # return correction parameter
        return r
        
        
class MyBC(BiasCorrection):
    ''' A BiasCorrection class that implements snowmelt shift and utilizes different (unobserved) precipitation types '''
    
    def _trainVar(self, var, obsvar, **kwargs):
        ''' optimize parameters for best fit of dataset to observations and save parameters;
            this method should be implemented for each method '''
        raise NotImplementedError
  
    def _correctVar(self, var, **kwargs):
        ''' apply bias correction to new variable and return bias-corrected data;
            this method should be implemented for each method '''
        raise NotImplementedError
