'''
Created on Feb 9, 2017

The definition of BiasCorrection classes, which actually implement the bias-correction methods. We have to keep the class
definition in a separate module, so it can be pickled and imported in other modules.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os, gzip, pickle
import collections as col
import numpy as np
import pandas as pd
from warnings import warn
# internal imports
from geodata.misc import isEqual, DataError, VariableError
from geodata.netcdf import DatasetNetCDF, VarNC


# some helper stuff for validation
eps = np.finfo(np.float32).eps # single precision rounding error is good enough
Stats = col.namedtuple('Stats', ('Correction','RMSE','Correlation','Bias',))

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
    if mode is None: pass
    elif mode == 'climatology': aggregation = 'clim'
    elif mode == 'time-series': aggregation = 'monthly'
    elif mode[-5:] == '-mean': aggregation = mode[:-5]
    else: aggregation = mode
    # string together different parameter arguments
    name = method if method else '' # initialize
    if obs_name: name += '_{:s}'.format(obs_name)
    if mode: name += '_{:s}'.format(aggregation)
    if domain: name += '_d{:02d}'.format(domain)
    if periodstr: name += '_{:s}'.format(periodstr)
    if gridstr: name += '_{:s}'.format(gridstr)
    if tag: name += '_{:s}'.format(tag)
    picklefile = pattern.format(name) # insert name into fixed pattern
    return picklefile
  
def findPicklePath(method=None, obs_name=None, gridstr=None, domain=None, tag=None, pattern=None, 
                   folder=None, lgzip=None):
    ''' construct pickle file name, add folder, and autodetect gzip '''
    picklefile = getPickleFileName(method=method, obs_name=obs_name, gridstr=gridstr, domain=domain, 
                                   tag=tag, pattern=pattern)
    picklepath = '{:s}/{:s}'.format(folder,picklefile)
    # autodetect gzip
    if lgzip:
        picklepath += '.gz' # add extension
        if not os.path.exists(picklepath): raise IOError(picklepath)
    elif lgzip is None:
        lgzip = False
        if not os.path.exists(picklepath):
            lgzip = True # assume gzipped file
            picklepath += '.gz' # try with extension...
            if not os.path.exists(picklepath): raise IOError(picklepath)
    elif not os.path.exists(picklepath): raise IOError(picklepath)
    # return validated path
    return picklepath

def loadBCpickle(method=None, obs_name=None, gridstr=None, domain=None, tag=None, pattern=None, 
                folder=None, lgzip=None):
    ''' construct pickle path, autodetect gzip, and load pickle - return BiasCorrection object '''
    # get pickle path (autodetect gzip)
    picklepath = findPicklePath(method=method, obs_name=obs_name, gridstr=gridstr, domain=domain, tag=tag, 
                                pattern=pattern, folder=folder, lgzip=lgzip)
    # load pickle from file
    op = gzip.open if lgzip else open
    with op(picklepath, 'rb') as filehandle:
        BC = pickle.load(filehandle) 
    ## N.B.: for some reason this code in a function can cause errors, but if it is take out ouf the function
    ##       and copied directly into the calling code, it works just fine... no idea why...
    return BC
  

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
  
    def correct(self, dataset, asNC=False, varlist=None, varmap=None, **kwargs):
        ''' loop over variables and apply correction function based on specific method using stored parameters '''
        # NetCDF datasets get special handling, so we only replace the variables we need to replace
        if isinstance(dataset,DatasetNetCDF):
            if not asNC: dataset.load() # otherwise we loose data
            bcds = dataset.copy(axesdeep=True, varsdeep=False, asNC=asNC) # make a copy, but don't duplicate data
        else: 
            asNC=False
            bcds = dataset.copy(axesdeep=True, varsdeep=False) # make a copy, but don't duplicate data
        # prepare variable map, so we can iterate easily
        itermap = dict() # the map we are going to iterate over
        varlist = self.varlist if varlist is None else varlist
        for varname in varlist:
            if varmap and varname in varmap:
                maplist = varmap[varname]
                if isinstance(maplist, (list,tuple)): itermap[varname] = maplist
                elif isinstance(maplist, str): itermap[varname] = (maplist,)
                else: raise TypeError(maplist)
            else:
                itermap[varname] = (varname,)
        # loop over variables that will be corrected
        for srcvar,maplist in list(itermap.items()):
            for tgtvar in maplist:
                if tgtvar in dataset:
                    # get variable object
                    oldvar = dataset[tgtvar].load()
                    newvar = bcds[tgtvar] # should be loaded
                    if isinstance(newvar,VarNC): # the corrected variable needs to load data, hence can't be VarNC          
                        newvar = newvar.copy(axesdeep=False, varsdeep=False, asNC=False) 
                    assert varname in self._correction, self._correction
                    # bias-correct data and load in new variable 
                    if self._correction[srcvar] is not None:
                        newvar.load(self._correctVar(oldvar, srcvar))
                    if newvar is not bcds[tgtvar]: 
                        bcds[tgtvar] = newvar # attach new (non-NC) var
        # return bias-corrected dataset
        return bcds
    
    def _correctVar(self, var, varname=None, **kwargs):
        ''' apply bias correction to new variable and return bias-corrected data;
            this method should be implemented for each method '''
        if varname is None: varname = var.name # allow for variable mapping
        return var.data_array # do nothing, just return input
    
    def _getVarlist(self, dataset, observations):
        ''' find all valid candidate variables for bias correction present in both input datasets '''
        varlist = []
        # loop over variables
        for varname in list(observations.variables.keys()):
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
                correction = np.mean(self._correction[varname]) # can be scalar 
                bias = delta.mean()
                if bias < eps: bias = 0.
                rmse = np.asscalar(np.sqrt(np.mean(delta**2)))
                if rmse < eps: rmse = 0.
                correlation = np.corrcoef(bcvar.data_array.flatten(),obsvar.data_array.flatten())[0,1]
                stats = Stats(Correction=correction,RMSE=rmse,Correlation=correlation,Bias=bias,)
            if lprint: print((varname,stats))
            validation[varname] = stats
        # return fit statistics
        self._validation = validation # also store
        return validation
    
    def picklefile(self, obs_name=None, mode=None, periodstr=None, gridstr=None, domain=None, tag=None):
        ''' generate a standardized name for the pickle file, based on arguments '''
        if self._picklefile is None:      
            self._picklefile = getPickleFileName(method=self.name, obs_name=obs_name, periodstr=periodstr, 
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
    _ratio_units = None # variable units that indicate ratio
    _operation = None
    
    def __init__(self, ratio_units=None, **kwargs):
        super(Delta,self).__init__(**kwargs)
        if ratio_units is None:
            self._ratio_units = ('mm/day','kg/m^2/s','J/m^2/s','W/m^2')
        else: self._ratio_units = ratio_units 
        self._operation = dict()
    
    def _checkClim(self, time_axis, modvar, obsvar):
        # check if we are dealing with a monthly climatology
        for var,typ in zip([modvar,obsvar],['model','ref/obs']):
            tax = var.getAxis(time_axis)
            if len(tax) != 12 and 'month' not in tax.units.lower(): 
                raise VariableError("The Variable '{}' ({}) does notappear to be a monthly climatology:\n{}".format(var.name,typ,str(var)))
    
    def _extendClim(self, correction, var, time_axis):
        # extend monthly normal correction factors to multiple years
        tidx = var.axisIndex(time_axis)
        assert var.ndim == correction.ndim
        assert correction.shape[tidx] == 12
        assert var.shape[tidx]%12 == 0
        return np.repeat(correction, var.shape[tidx]//12, axis=tidx)
        
    def _trainVar(self, var, obsvar, time_axis='time', **kwargs):
        ''' take difference (or ratio) between observations and simulation and use as correction '''
        self._checkClim(time_axis, var, obsvar) # check time axis
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            delta = obsvar.data_array / var.data_array 
            self._operation[var.name] = 'ratio'
        else: # default behavior is differences
            delta = obsvar.data_array - var.data_array
            self._operation[var.name] = 'diff'
        # return correction parameters, i.e. delta
        return delta
          
    def _correctVar(self, var, varname=None, time_axis='time', **kwargs):
        ''' use stored ratios to bias-correct the input dataset and return a new copy '''
        if varname is None: varname = var.name # allow for variable mapping
        # check some input
        if varname not in self._correction:
            raise ValueError("No bias-correction values were found for variable '{}'.".format(varname))
        if '_operation' not in self.__dict__:
            warn("The '_operation' dictionary was not found in the BiasCorrection instance '{}';\n it is likely an older version of the Class and should be recomputed.")
        # format correction
        correction = self._extendClim(self._correction[varname], var, time_axis)
        # decide between difference or ratio based on variable type
        if self._operation[varname] == 'ratio': # ratio for fluxes
            data = var.data_array * correction
        elif self._operation[varname] == 'diff': # default behavior is differences
            data = var.data_array + correction
        else:
            raise ValueError(self._operation[varname])
        # return bias-corrected data (copy)
        return data
      
    def correctionByTime(self, varname, time, ldt=True, time_idx=0, **kwargs):
        ''' return formatted correction arrays based on an array or list of datetime objects '''
        # interprete datetime: convert to monthly indices
        if ldt:
            # if the time array/list is datetime-like, we have to extract the month
            if isinstance(time,(list,tuple)):
                time = [pd.to_datetime(t).month-1 for t in time]
            elif isinstance(time,np.ndarray):
                if np.issubdtype(time.dtype, np.datetime64):
                    time = time.astype('datetime64[M]').astype('int16') % 12
            else:
                raise TypeError(time)
        # ... but in the end we need an array of indices
        time = np.asarray(time, dtype='int16')
        # construct view into correction array based on indices
        correction = np.take(self._correction[varname], time, axis=time_idx,)
        # return correction factors for requested time indices
        return correction

    def correctArray(self, data_array, varname, time, ldt=True, time_idx=0, **kwargs):
        ''' apply a correction to a simple numpy array '''
        # check dimensions
        assert data_array.shape[time_idx] == len(time), (data_array.shape, len(time))
        # get correction array based on list of requested time stamps/indices
        correction = self.correctionByTime(varname, time, ldt=ldt, time_idx=time_idx, **kwargs)
        # decide between difference or ratio based on variable type
        if self._operation[varname] == 'ratio': # ratio for fluxes
            data = data_array * correction
        elif self._operation[varname] == 'diff': # default behavior is differences
            data = data_array + correction
        else:
            raise ValueError(self._operation[varname])
        # return corrected data array
        return data
        

class SMBC(Delta):
    name = 'SMBC' # name used in file names
    long_name = 'Simple Monthly Mean Adjustment' # name for printing
      
    def _trainVar(self, var, obsvar, time_axis='time', **kwargs):
        ''' take difference (or ratio) between spatially averaged observations and simulation and use as correction '''
        self._checkClim(time_axis, var, obsvar) # check time axis
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
            self._operation[var.name] = 'ratio'
        else: # default behavior is differences
            r = obsdata - vardata
            self._operation[var.name] = 'diff'
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
      
    def _trainVar(self, var, obsvar, time_axis='time', **kwargs):
        ''' take difference (or ratio) between averaged observations and simulation and use as correction '''
        obsdata = obsvar.data_array; vardata = var.data_array
        # average spatially and temporally (over seasonal cycle) - very simple!
        obsdata = obsdata.mean(); vardata = vardata.mean()
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            r = obsdata / vardata 
            self._operation[var.name] = 'ratio'            
        else: # default behavior is differences
            r = obsdata - vardata
            self._operation[var.name] = 'diff'            
        if not np.isscalar(r): raise DataError(r)
        # return correction parameter
        return r
        
        
class MyBC(BiasCorrection):
    ''' A BiasCorrection class that implements snowmelt shift and utilizes different (unobserved) precipitation types '''
    
    def _trainVar(self, var, obsvar, time_axis='time', **kwargs):
        ''' optimize parameters for best fit of dataset to observations and save parameters;
            this method should be implemented for each method '''
        raise NotImplementedError
  
    def _correctVar(self, var, varname=None, time_axis='time', **kwargs):
        ''' apply bias correction to new variable and return bias-corrected data;
            this method should be implemented for each method '''
        if varname is None: varname = var.name # allow for variable mapping
        raise NotImplementedError
