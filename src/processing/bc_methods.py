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
    # meta data for dataset record
    version = 1
    note = ''
    var_notes = None # variable-specific notes
    obs_name = None
    obs_title = None
    
    def __init__(self, varlist=None, **bcargs):
        ''' take arguments that have been passed from caller and initialize parameters '''
        self.varlist = varlist
        self.var_notes = dict() # this should be not be a class attribute

    def train(self, dataset, observations, **kwargs):
        ''' loop over variables that need to be corrected and call method-specific training function '''
        # save meta data
        self.obs_name = observations.name
        self.obs_title = observations.title
        # figure out varlist
        if self.varlist is None: 
            self._getVarlist(dataset, observations) # process all that are present in both datasets        
        # loop over variables that will be corrected
        self._correction = dict()
        for varname in self.varlist:
            # check for special treatment
            fctname = '_train_'+varname
            if  hasattr(self, fctname):
                # call custom training method for this variable
                getattr(self, fctname)(varname, dataset, observations, **kwargs)
                # assignment of _correction item is done internally
            else:
                # otherwise get variable object
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
        itermap = dict()
        for varname in bcds.variables.keys():
            if hasattr(self,'_correct_'+varname): 
                itermap[varname] = '_correct_by_built-in_method'
            elif varname in self._correction: 
                itermap[varname] = varname
            elif varmap is not None and varname in varmap:
                if varmap[varname] in self._correction: 
                  itermap[varname] = varmap[varname]
            

        # loop over variables in soon-to-be bias-corrected dataset
        for varname,bcvar in itermap.items():
            # get variable object
            oldvar = dataset[varname].load()
            newvar = bcds[varname] # should be loaded
            if isinstance(newvar,VarNC): # the corrected variable needs to load data, hence can't be VarNC          
                newvar = newvar.copy(axesdeep=False, varsdeep=False, asNC=False) 
            # bias-correct data and load in new variable 
            if varlist is None or varname in varlist:
                # figure out method
                if bcvar == '_correct_by_built-in_method':
                    # call custom correction method for this variable
                    fctname = '_correct_'+varname
                    corrected_array = getattr(self, fctname)(varname=varname, dataset=dataset, **kwargs)
                    # load corrected data
                    newvar.load(corrected_array)
                else:
                    if self._correction[bcvar] is not None:
                        # use default correction (scale of shift based on units)
                        corrected_array = self._correctVar(oldvar, varname=bcvar, **kwargs)
                        # load corrected data
                        newvar.load(corrected_array)
                # save meta data about bias correction
                newvar.atts['bc_method']    = self.name
                newvar.atts['bc_long_name'] = self.long_name
                newvar.atts['bc_version']   = self.version
                newvar.atts['bc_variable']  = bcvar
                if varname in self.var_notes: 
                    newvar.atts['bc_note']  = self.var_notes[varname]
                elif bcvar in self.var_notes: 
                    newvar.atts['bc_note']  = self.var_notes[bcvar]
                else:
                    newvar.atts['bc_note']  = ''
                newvar.atts['bc_obs_name']  = self.obs_name
                newvar.atts['bc_obs_title'] = self.obs_title
                if newvar is not bcds[varname]: 
                    bcds[varname] = newvar # attach new (non-NC) var
        # save meta data about bias correction
        bcds.atts['bc_method']    = self.name
        bcds.atts['bc_long_name'] = self.long_name
        bcds.atts['bc_version']   = self.version
        bcds.atts['bc_note']      = self.note
        bcds.atts['bc_obs_name']  = self.obs_name
        bcds.atts['bc_obs_title'] = self.obs_title
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
            if varname in dataset.variables:
                if np.issubdtype(dataset[varname].dtype, np.inexact)  and not dataset[varname].strvar:
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
        assert var.ndim == correction.ndim, var
        assert correction.shape[tidx] == 12
        assert var.shape[tidx]%12 == 0
        tiling = [1,]*var.ndim
        tiling[tidx] = var.shape[tidx]//12
        return np.tile(correction, tiling)
        
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
        correction = self._correction[varname]
        if isinstance(correction,np.ndarray):
            correction = self._extendClim(correction, var, time_axis)
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
        

# helper function
def spatialAverage(var, time_axis='time', keepdims=True):
    ''' helper function to compute spatial averages and preserve the tiem axis ''' 
    data = var.load().data_array
    if var.hasAxis(time_axis):
        # handle variables with time axis
        if var.ndim > 1:
            tax = var.axisIndex(time_axis)
            for i in range(var.ndim-1,-1,-1):
                # average spatial dimensions but not time
                if i != tax:
                    data = data.mean(axis=i, keepdims=keepdims)
    else:
        # without time axis, this is simple
        data = data.mean()
    return data

class SMBC(Delta):
    name = 'SMBC' # name used in file names
    long_name = 'Simple Monthly Mean Adjustment' # name for printing
      
    def _trainVar(self, var, obsvar, time_axis='time', **kwargs):
        ''' take difference (or ratio) between spatially averaged observations and simulation and use as correction '''
        self._checkClim(time_axis, var, obsvar) # check time axis
        vardata = spatialAverage(var, time_axis=time_axis, keepdims=True)
        obsdata = spatialAverage(obsvar, time_axis=time_axis, keepdims=True)
        # decide between difference or ratio based on variable type
        if var.units in self._ratio_units: # ratio for fluxes
            r = obsdata / vardata 
            self._operation[var.name] = 'ratio'
        else: # default behavior is differences
            r = obsdata - vardata
            self._operation[var.name] = 'diff'
        # consistency check
        if var.hasAxis(time_axis):
            if r.size != 12: 
                raise DataError(r.shape)
        else:
            if not np.isscalar(r):
                raise DataError(r)
        # make sure values are finite and replace invalid values with mean
        if isinstance(r,np.ma.masked_array):
            r = r.filled(np.NaN)
        r = np.where(np.isfinite(r), r, np.NaN)
        rm = np.nanmean(r)
        r = np.where(np.isnan(r), rm, r)
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
        
        
class MyBC(AABC):
    ''' A BiasCorrection class that implements snowmelt shift and estimates convective and 
        non-convective precipitation via best fit of seasonal cycle to total precipitation '''
    name = 'MyBC' # name used in file names
    long_name = 'Custom Bias Correction' # name for printing
    version = 1.0
    note = 'distinguishes between convective and grid-scale precip'
    # for temporary internal storage
    _precip = None
    _preccu = None
    _precnc = None 
    _liqprec = None
    _solprec = None
    _snwmlt = None
    _liqwatflx = None 

    
    def _getVarlist(self, dataset, observations):
        ''' do usual matching, but add precip types and remove others '''
        # do matching
        varlist = super(MyBC,self)._getVarlist(dataset, observations)
        # remove some precip types
        for varname in ('solprec','liqprec','precip'):
            if varname in varlist: varlist.remove(varname)
        # add convective and grid-scale precip (not in obs) and snowmelt
        for varname in ('preccu','precnc','snwmlt'):
            if varname not in varlist: varlist.append(varname)
        # save and return varlist
        self.varlist = varlist
        return varlist        
   
    # helper functions
    
    def _objective(self, x, obs_precip=None, preccu=None, precnc=None):
        ''' objective function to minimize '''
        delta = obs_precip - x[0]*preccu - x[1]*precnc
        return np.sum(delta**2)
      
    def _first_guess(self, obs_precip=None, preccu=None, precnc=None, time_idx=0):
        ''' estimate reasonable first guess '''
        assert obs_precip.shape == preccu.shape == precnc.shape
        if obs_precip.shape[time_idx] == 12:
            # assume winter is non-convective and deduce convective from sum
            winter = (0,1,2,10,11,)
            b = ( obs_precip.take(winter,axis=time_idx).sum(axis=time_idx,keepdims=False) /
                                          precnc.take(winter,axis=time_idx).sum(axis=time_idx,keepdims=False) )
            a = ( ( obs_precip.sum(axis=time_idx,keepdims=False) - b*precnc.sum(axis=time_idx,keepdims=False) ) /
                                                                  preccu.sum(axis=time_idx,keepdims=False) )
        else:
            # otherwise start by assuming all is non-convective
            a = 0
            b = obs_precip.sum(axis=time_idx, keepdims=False)/precnc.sum(axis=time_idx, keepdims=False)
        # return first-guess parameters as 1d array
        return np.asarray((a, b))

    # convective and grid-scale precipitation correction
  
    def _fitPrecip(self, varname, dataset, observations, time_axis='time', **kwargs):
        ''' optimize parameters for best fit of dataset to observations and save parameters;
            this method should be implemented for each method '''
        assert 'precip' in observations, observations
        obs_precip = spatialAverage(observations['precip'], time_axis=time_axis, keepdims=False)
        assert 'precnc' in dataset and 'preccu' in dataset, dataset
        preccu = spatialAverage(dataset['preccu'], time_axis=time_axis, keepdims=False)
        precnc = spatialAverage(dataset['precnc'], time_axis=time_axis, keepdims=False)
        # calculate first guess
        time_idx = observations['precip'].axisIndex(time_axis)
        x0 = self._first_guess(obs_precip, preccu, precnc, time_idx=time_idx)
#         self._correction['preccu'] = x0[0]
#         self._correction['precnc'] = x0[1]
        # optimize estimate
        from scipy.optimize import minimize
        res = minimize(self._objective, x0, (obs_precip, preccu, precnc),
                       method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        # assign values
        self._correction['preccu'] = res.x[0]
        self._correction['precnc'] = res.x[1]
        # add note
        precip_note = 'convective and grid-scale precip are corrected simultaneously so as to match total observed precip'
        for precip_type in ('precip','preccu','precnc','liqprec','solprec'):
            self.var_notes[precip_type] = precip_note
    
    def _train_preccu(self, varname, dataset, observations, time_axis='time', **kwargs):
        ''' call fit function if not done already (can be called through either preccu or precnc) '''
        assert 'preccu' == varname and varname in dataset, dataset
        if varname not in self._correction:
            self._fitPrecip(varname, dataset, observations, time_axis, **kwargs)
            
    def _train_precnc(self, varname, dataset, observations, time_axis='time', **kwargs):
        ''' call fit function if not done already (can be called through either preccu or precnc) '''
        assert 'precnc' == varname and varname in dataset, dataset
        if varname not in self._correction:
            self._fitPrecip(varname, dataset, observations, time_axis, **kwargs)

    def _correct_preccu(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' correct convective precipitation and save for later use '''
        assert 'preccu' in dataset, dataset
        assert 'preccu' in self._correction, self._correction
        if self._preccu is None:
            self._preccu = self._correction['preccu'] * dataset.variables['preccu'].load().data_array
        return self._preccu

    def _correct_precnc(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' correct non-convective/grid-scale precipitation and save for later use '''
        assert 'precnc' in dataset, dataset
        assert 'precnc' in self._correction, self._correction
        if self._precnc is None:
            self._precnc = self._correction['precnc'] * dataset.variables['precnc'].load().data_array
        return self._precnc
            
    # other precipitation types (liquid, solid, total)
    
    def _correct_liqprec(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' correct liquid precipitation based on precip types, assuming convective precip is always liquid '''
        assert 'liqprec' in dataset and 'preccu' in dataset, dataset
        assert 'precnc' in self._correction and 'preccu' in self._correction, self._correction
        if self._liqprec is None:
            b = self._correction['precnc']
            liqprec = b * dataset.variables['liqprec'].load().data_array 
            liqprec += ( self._correction['preccu'] - b ) * dataset.variables['preccu'].load().data_array
            self._liqprec = liqprec
        return self._liqprec
      
    def _correct_solprec(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' correct solid precipitation as non-convective/grid-scale precip '''
        assert 'solprec' in dataset, dataset
        assert 'precnc' in self._correction and 'preccu' in self._correction, self._correction
        if self._solprec is None:
            self._solprec = self._correction['precnc'] * dataset.variables['solprec'].load().data_array
        return self._solprec

    def _correct_precip(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' compute bias-corrected precip from convective and grid-scale precip '''
        assert 'preccu' in dataset and 'precnc' in dataset, dataset
        if self._precip is None:
            if self._preccu is None:
                self._correct_preccu(varname=varname, dataset=dataset, time_axis=time_axis, **kwargs)
            if self._precnc is None:
                self._correct_precnc(varname=varname, dataset=dataset, time_axis=time_axis, **kwargs)
            self._precip = self._preccu + self._precnc
        return self._precip
       
    # snowmelt correction
  
    def _train_snwmlt(self, varname, dataset, observations, time_axis='time', **kwargs):
        ''' estimate temporal offset to shift snowmelt variable in time, on top of bias-correction'''
        # just use arbitrary one month shift, until we have something better
        self._correction[varname] = 0
        self.var_notes['snwmlt'] = 'snowmelt is currently corrected like grid-scale precip (no temporal shift)'
        self.var_notes['liqwatflx'] = 'liquid water flux is the sum of snowmelt and liquid precip, which are bias-corrected independently'
    
    def _correct_snwmlt(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' apply temporal shift and scale date; return bias-corrected, shifted data '''
        assert 'precnc' in self._correction, self._correction
        if self._snwmlt is None:
            var = dataset.variables['snwmlt'].load()
            # temporal shift 
            shift = self._correction['snwmlt']
            if shift:
                tax = var.axisIndex(time_axis)
                assert var.shape[tax]%12 == 0, var # this is really only applicable to periodic monthly data...
                assert 'month' in var.axes[tax].units.lower(), var.axes[tax]
                data = np.roll(var.data_array, shift=shift, axis=tax) 
            else:
                data = var.data_array
            # scale to grid-scale precip
            data *= self._correction['precnc']
            self._snwmlt = data
        # return shifted and scaled data array       
        return self._snwmlt

    # liquid water flux
  
    def _correct_liqwatflx(self, varname=None, dataset=None, time_axis='time', **kwargs):
        ''' compute bias-corrected liquid water flux from liquid precip and snowmelt '''
        if self._liqwatflx is None:
            if self._liqprec is None:
                self._correct_liqprec(varname=varname, dataset=dataset, time_axis=time_axis, **kwargs)
            if self._snwmlt is None:
                self._correct_snwmlt(varname=varname, dataset=dataset, time_axis=time_axis, **kwargs)
            self._liqwatflx = self._liqprec + self._snwmlt
        return self._liqwatflx