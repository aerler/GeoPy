'''
Created on 2017-02-01

A script to generate bias correction objects for WRF or other experiments; includes the definition of the 
BiasCorrection object; the actual correction is carried out inthe export module.

@author: Andre R. Erler, GPL v3
'''

# external imports
import pickle, os # check if files are present etc.
import numpy as np
import collections as col
from datetime import datetime 
from importlib import import_module
import logging     
# internal imports
from geodata.misc import DateError, printList, isEqual
from datasets import gridded_datasets
from processing.multiprocess import asyncPoolEC
from processing.misc import getMetaData,  getExperimentList, loadYAML
from datasets.common import loadDataset
from geodata.netcdf import DatasetNetCDF

# some helper stuff for validation
eps = np.finfo(np.float32).eps # single precision rounding error is good enough
Stats = col.namedtuple('Stats', ('Bias','RMSE','Corr'))

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
    if not asNC and isinstance(dataset,DatasetNetCDF): dataset.load() # otherwise we loose data
    bcds = dataset.copy(axesdeep=True, varsdeep=False, asNC=asNC) # make a copy, but don't duplicate data
    # loop over variables that will be corrected
    for varname in self.varlist:
        # get variable object
        oldvar = dataset[varname].load()
        newvar = bcds[varname] # should be loaded
        assert varname in self._correction, self._correction
        # bias-correct data and load in new variable 
        if self._correction[varname] is not None:
            newvar.load(self._correctVar(oldvar))
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
  
  def picklefile(self, obs_name=None, grid_name=None):
    ''' generate a name for the pickle file, based on methd and options '''
    if self._picklefile is None:
      name = self.name
      if obs_name: name += '_{:s}'.format(obs_name)
      if grid_name: name += '_{:s}'.format(grid_name)
      self._picklefile = 'bias_{:s}.pickle'.format(name)
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
  name = 'delta' # name used in file names
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

class MyBC(BiasCorrection):
  pass

class SMBC(BiasCorrection):
  pass


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
  elif method == 'SMBC':
    return SMBC(**bcargs)
  else:
    raise NotImplementedError(method)
  

# worker function that is to be passed to asyncPool for parallel execution; use of TrialNError decorator is assumed
def generateBiasCorrection(dataset, mode, dataargs, obs_dataset, bc_method, bc_args, loverwrite=False, 
                           ldebug=False, lparallel=False, pidstr='', logger=None):
  ''' worker function to generate a bias correction objects for a given dataset '''
  # input checking
  if not isinstance(dataset,basestring): raise TypeError
  if not isinstance(dataargs,dict): raise TypeError # all dataset arguments are kwargs 
  
  # logging
  if logger is None: # make new logger     
    logger = logging.getLogger() # new logger
    logger.addHandler(logging.StreamHandler())
  else:
    if isinstance(logger,basestring): 
      logger = logging.getLogger(name=logger) # connect to existing one
    elif not isinstance(logger,logging.Logger): 
      raise TypeError, 'Expected logger ID/handle in logger KW; got {}'.format(str(logger))

  ## extract meta data from arguments
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs, lone=False)
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr; avgfolder = dataargs.avgfolder
  
  # parse export options
  bc_args = bc_args.copy() # first copy, then modify...
  # initialize BiasCorrection class instance
  BC = getBCmethods(bc_method, **bc_args)
  # get folder for target dataset and do some checks
  picklefile = BC.picklefile(obs_name=obs_dataset.name, grid_name=dataargs.grid)
  if ldebug: picklefile = 'test_' + picklefile 
  picklepath = '{:s}/{:s}'.format(avgfolder,picklefile)
  
  # check if we are overwriting an existing file
  if not os.path.exists(avgfolder): raise IOError, "Dataset folder '{:s}' does not exist!".format(avgfolder)
  lskip = False # else just go ahead
  if os.path.exists(picklepath): 
    if not loverwrite: 
      age = datetime.fromtimestamp(os.path.getmtime(picklepath))
      # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
      if age > srcage: 
        lskip = True
        if hasattr(obs_dataset, 'filepath') and obs_dataset.filepath is not None:
          obsage = datetime.fromtimestamp(os.path.getmtime(obs_dataset.filepath))
          if age < obsage: lskip = False

  
  # depending on last modification time of file or overwrite setting, start computation, or skip
  if lskip:        
    # print message
    skipmsg =  "\n{:s}   >>>   Skipping: Bias-correction '{:s} for dataset '{:s}' already exists and is newer than source file.".format(pidstr,BC.long_name,dataset_name)
    skipmsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,picklepath)
    logger.info(skipmsg) 
    del BC             
  else:
          
    ## actually load datasets
    dataset = loadfct() # load source data
    # check period
    if 'period' in dataset.atts and dataargs.periodstr != dataset.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,dataset.atts.period)

    # print message
    if mode == 'climatology': opmsgstr = 'Exporting Climatology ({:s}) to {:s} Format'.format(periodstr, BC.long_name)
    elif mode == 'time-series': opmsgstr = 'Exporting Time-series to {:s} Format'.format(BC.long_name)
    elif mode[-5:] == '-mean': opmsgstr = 'Exporting {:s}-Mean ({:s}) to {:s} Format'.format(mode[:-5], periodstr, BC.long_name)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)        
    # print feedback to logger
    logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
    if not lparallel and ldebug: logger.info('\n'+str(dataset)+'\n')
    
    # N.B.: data are not loaded immediately but on demand; this way I/O and computing are further
    #       disentangled and not all variables are always needed
    
    
    # "train", i.e. optimize fit parameters
    BC.train(dataset, obs_dataset)
    
    # print bias-correction
    if not lparallel and ldebug:
      logger.info('\n'+str(BC)+'\n')
      print("Bias-correction Statistics:")
      BC.validate(dataset, obs_dataset, lprint=True)    
      print('')  
      
    ## pickle bias-correction object with trained parameters
    # open file and save pickle
    if os.path.exists(picklepath): os.remove(picklepath)
    filehandle = open(picklepath, 'wb')
    pickle.dump(BC, filehandle)
    filehandle.close()
    if not os.path.exists(picklepath):
      raise IOError, "Error while saving Pickle to '{0:s}'".format(picklepath)

      
    # write results to file
    writemsg =  "\n{:s}   >>>   Generation of BiasCorrection '{:s}' for Dataset '{:s}' complete.".format(pidstr,bc_method, dataset_name,)
    writemsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,picklepath)
    logger.info(writemsg)      
       
    # clean up and return
    dataset.unload(); del dataset, BC
    return 0 # "exit code"
    # N.B.: garbage is collected in multi-processing wrapper


if __name__ == '__main__':
  
  ## read environment variables
  # number of processes NP 
  if os.environ.has_key('PYAVG_THREADS'): 
    NP = int(os.environ['PYAVG_THREADS'])
  else: NP = None
  # run script in debug mode
  if os.environ.has_key('PYAVG_DEBUG'): 
    ldebug =  os.environ['PYAVG_DEBUG'] == 'DEBUG' 
  else: ldebug = False
  # run script in batch or interactive mode
  if os.environ.has_key('PYAVG_BATCH'): 
    lbatch =  os.environ['PYAVG_BATCH'] == 'BATCH' 
  else: lbatch = False # for debugging
  # re-compute everything or just update 
  if os.environ.has_key('PYAVG_OVERWRITE'): 
    loverwrite =  os.environ['PYAVG_OVERWRITE'] == 'OVERWRITE' 
  else: loverwrite = ldebug # False means only update old files
  
  ## define settings
  if lbatch:
    # load YAML configuration
    config = loadYAML('biascorrection.yaml', lfeedback=True)
    # read config object
    NP = NP or config['NP']
    loverwrite = config['loverwrite']
    # source data specs
    modes = config['modes']
    load_list = config['load_list']
    periods = config['periods']
    # Datasets
    datasets = config['datasets']
    resolutions = config['resolutions']
    unity_grid = config.get('unity_grid',None)
    lLTM = config['lLTM']
    # CESM
    CESM_project = config['CESM_project']
    CESM_experiments = config['CESM_experiments']
    CESM_filetypes = config['CESM_filetypes']
    load3D = config['load3D']
    # WRF
    WRF_project = config['WRF_project']
    WRF_experiments = config['WRF_experiments']
    WRF_filetypes = config['WRF_filetypes']
    domains = config['WRF_domains']
    grids = config['grids']
    # target data specs
    export_arguments = config['export_parameters'] # this is actually a larger data structure
    lm3 = export_arguments['lm3'] # convert water flux from kg/m^2/s to m^3/m^2/s    
  else:
    # settings for testing and debugging
#     NP = 2 ; ldebug = False # for quick computations
    NP = 1 ; ldebug = True # just for tests
    modes = ('climatology',) # 'climatology','time-series'
    loverwrite = True
    ## datasets to be bias-corrected
    periods = []
    periods += [15]
#     periods += [30]
    # Observations/Reanalysis
    resolutions = {'CRU':'','GPCC':['025','05','10','25'],'NARR':'','CFSR':['05','031'],'NRCan':'NA12'}
    lLTM = True # also regrid the long-term mean climatologies 
    datasets = []
#     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
    # CESM experiments (short or long name) 
    CESM_project = None # all available experiments
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['Ens']
    CESM_filetypes = ['atm','lnd']
    # WRF experiments (short or long name)
    WRF_project = 'GreatLakes' # only GreatLakes experiments
#     WRF_project = 'WesternCanada' # only WesternCanada experiments
    WRF_experiments = [] # use None to process all WRF experiments
#     WRF_experiments += ['g3-ensemble','t3-ensemble',]
#     WRF_experiments += ['erai-g3','erai-t3']
#     WRF_experiments += ['erai-g3','erai-g']
#     WRF_experiments += ['g-ensemble']
    WRF_experiments += ['g-ctrl']
    # other WRF parameters 
    domains = 1 # domains to be processed
#     domains = None # process all domains
    WRF_filetypes = ('hydro','srfc','xtrm','lsm','rad') # available input files
#     WRF_filetypes = ('aux',) # only preprocessed auxiliary files
    ## observations (i.e. the reference dataset; arguments passed to loadDataset)
    obs_name = 'NRCan'
    obs_mode = 'climatology'
    obs_args = dict(resolution='na12')
    ## remaining parameters
    load_list = None # variables that need to be loaded
    varlist = None # variables that should be bias-corrected
    grid = 'grw2' # need a common grid for all datasets
    bc_method = 'Delta' # bias correction method
    bc_args = dict() # paramters for bias correction
  
  ## process arguments
  if isinstance(periods, (np.integer,int)): periods = [periods]
  # check and expand WRF experiment list
  WRF_experiments = getExperimentList(WRF_experiments, WRF_project, 'WRF')
  if isinstance(domains, (np.integer,int)): domains = [domains]
  # check and expand CESM experiment list
  CESM_experiments = getExperimentList(CESM_experiments, CESM_project, 'CESM')
  # expand datasets and resolutions
  if datasets is None: datasets = gridded_datasets  
  # update some dependencies
  unity_grid = grid # trivial in this case
  obs_args['grid'] = grid
  obs_args['varlist'] = load_list   

  ## load observations/reference dataset
  obs_dataset = loadDataset(name=obs_name, mode=obs_mode, **obs_args).load()
  
  # print an announcement
  if len(WRF_experiments) > 0:
    print('\n Bias-correcting WRF Datasets:')
    print([exp.name for exp in WRF_experiments])
  if len(CESM_experiments) > 0:
    print('\n Bias-correcting CESM Datasets:')
    print([exp.name for exp in CESM_experiments])
  if len(datasets) > 0:
    print('\n Bias-correcting Other Datasets:')
    print(datasets)
  print('\n On Grid/Resolution:\n   {:s}'.format(grid))
  print('\n Variable List: {:s}'.format('All' if varlist is None else printList(varlist)))
  print('\n Bias-Correction Method: {:s}'.format(bc_method))
  print('\n Observationa/Reference Dataset:\n   {:s}'.format(obs_dataset if ldebug else obs_dataset.name))
  print('\n OVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
    
  ## construct argument list
  args = []  # list of job packages
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode[-5:] == '-mean': periodlist = periods
    elif mode == 'climatology': periodlist = periods
    elif mode == 'time-series': periodlist = (None,)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)

      
    # observational datasets (grid depends on dataset!)
    for dataset in datasets:
      mod = import_module('datasets.{0:s}'.format(dataset))
      if isinstance(resolutions,dict): 
        if dataset not in resolutions: resolutions[dataset] = ('',)
        elif not isinstance(resolutions[dataset],(list,tuple)): resolutions[dataset] = (resolutions[dataset],)                
      elif resolutions is not None: raise TypeError                                
      if mode[-5:] == '-mean' or mode == 'climatology':
        # some datasets come with a climatology 
        if lLTM:
          if resolutions is None: dsreses = mod.LTM_grids
          elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.LTM_grids]  
          for dsres in dsreses: 
            args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=None, resolution=dsres, unity_grid=unity_grid)) ) # append to list
        # climatologies derived from time-series
        if resolutions is None: dsreses = mod.TS_grids
        elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
        for dsres in dsreses:
          for period in periodlist:
            args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=period, resolution=dsres, unity_grid=unity_grid)) ) # append to list            
      elif mode == 'time-series': 
        # regrid the entire time-series
        if resolutions is None: dsreses = mod.TS_grids
        elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
        for dsres in dsreses:
          args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=None, resolution=dsres, unity_grid=unity_grid)) ) # append to list            
    
    # CESM datasets
    for experiment in CESM_experiments:
      for period in periodlist:
        # arguments for worker function: dataset and dataargs       
        args.append( ('CESM', mode, dict(experiment=experiment, filetypes=CESM_filetypes, grid=grid, 
                                         varlist=load_list, period=period, load3D=load3D)) )
    # WRF datasets
    for experiment in WRF_experiments:
      # effectively, loop over domains
      if domains is None:
        tmpdom = range(1,experiment.domains+1)
      else: tmpdom = domains
      for domain in tmpdom:
        for period in periodlist:
          # arguments for worker function: dataset and dataargs       
          args.append( ('WRF', mode, dict(experiment=experiment, filetypes=WRF_filetypes, grid=grid, 
                                          varlist=load_list, domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(obs_dataset=obs_dataset, bc_method=bc_method, bc_args=bc_args, loverwrite=loverwrite)
  # N.B.: formats will be iterated over inside export function
  
  ## call parallel execution function
  ec = asyncPoolEC(generateBiasCorrection, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+int(10.*ec/len(args))) if ec > 0 else 0)
