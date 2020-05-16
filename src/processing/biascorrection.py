'''
Created on 2017-02-01

A script to generate bias correction objects for WRF or other experiments; BiasCorrection classes are defined in the module
'bc_methods'; the actual correction is carried out in the export module.

@author: Andre R. Erler, GPL v3
'''

# external imports
import pickle
import os, gzip # check if files are present etc.
import numpy as np
from datetime import datetime 
from importlib import import_module
import logging     
# internal imports
from geodata.misc import DateError, printList
from datasets import gridded_datasets
from processing.multiprocess import asyncPoolEC
from processing.misc import getMetaData,  getExperimentList, loadYAML
from datasets.common import loadDataset
from processing.bc_methods import getBCmethods


# worker function that is to be passed to asyncPool for parallel execution; use of TrialNError decorator is assumed
def generateBiasCorrection(dataset, mode, dataargs, obs_dataset, bc_method, bc_args, loverwrite=False, lgzip=None, tag=None, 
                           ldebug=False, lparallel=False, pidstr='', logger=None):
  ''' worker function to generate a bias correction objects for a given dataset '''
  # input checking
  if not isinstance(dataset,str): raise TypeError
  if not isinstance(dataargs,dict): raise TypeError # all dataset arguments are kwargs 
  
  # logging
  if logger is None: # make new logger     
    logger = logging.getLogger() # new logger
    logger.addHandler(logging.StreamHandler())
  else:
    if isinstance(logger,str): 
      logger = logging.getLogger(name=logger) # connect to existing one
    elif not isinstance(logger,logging.Logger): 
      raise TypeError('Expected logger ID/handle in logger KW; got {}'.format(str(logger)))

  ## extract meta data from arguments
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs, lone=False)
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr; avgfolder = dataargs.avgfolder
  
  # parse export options
  bc_args = bc_args.copy() # first copy, then modify...
  # initialize BiasCorrection class instance
  BC = getBCmethods(bc_method, **bc_args)
  # get folder for target dataset and do some checks
  picklefile = BC.picklefile(obs_name=obs_dataset.name, gridstr=dataargs.grid, domain=dataargs.domain, tag=tag)
  if ldebug: picklefile = 'test_' + picklefile 
  picklepath = '{:s}/{:s}'.format(avgfolder,picklefile)
  
  # check if we are overwriting an existing file
  if not os.path.exists(avgfolder): raise IOError("Dataset folder '{:s}' does not exist!".format(avgfolder))
  lskip = False # else just go ahead
  if os.path.exists(picklepath) and not loverwrite: 
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
      raise DateError("Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,dataset.atts.period))

    # print message
    if mode == 'climatology': opmsgstr = 'Bias-correcting Climatology ({:s}) using {:s}'.format(periodstr, BC.long_name)
    elif mode == 'time-series': opmsgstr = 'Bias-correcting Time-series using {:s}'.format(BC.long_name)
    elif mode[-5:] == '-mean': opmsgstr = 'Bias-correcting {:s}-Mean ({:s}) using {:s}'.format(mode[:-5], periodstr, BC.long_name)
    else: raise NotImplementedError("Unrecognized Mode: '{:s}'".format(mode))        
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
    if lgzip:
      op = gzip.open 
      picklepath += '.gz'
    else: op = open
    with op(picklepath, 'wb') as filehandle:
      pickle.dump(BC, filehandle, protocol=2) # should be new binary protocol
    if not os.path.exists(picklepath):
      raise IOError("Error while saving Pickle to '{0:s}'".format(picklepath))

      
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
  if 'PYAVG_THREADS' in os.environ: 
    NP = int(os.environ['PYAVG_THREADS'])
  else: NP = None
  # run script in debug mode
  if 'PYAVG_DEBUG' in os.environ: 
    ldebug =  os.environ['PYAVG_DEBUG'] == 'DEBUG' 
  else: ldebug = False
  # run script in batch or interactive mode
  if 'PYAVG_BATCH' in os.environ: 
    lbatch =  os.environ['PYAVG_BATCH'] == 'BATCH' 
  else: lbatch = False # for debugging
  # re-compute everything or just update 
  if 'PYAVG_OVERWRITE' in os.environ: 
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
    WRF_domains = config['WRF_domains']
    grids = config['grids']
    # target data specs
    export_arguments = config['export_parameters'] # this is actually a larger data structure
    lm3 = export_arguments['lm3'] # convert water flux from kg/m^2/s to m^3/m^2/s    
  else:
    # settings for testing and debugging
    NP = 1 ; ldebug = False # for quick computations
#     NP = 1 ; ldebug = True # just for tests
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
#     datasets += ['SnoDAS']; periods = [(2010,2019)]
#     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
    # CESM experiments (short or long name) 
    CESM_project = None # all available experiments
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['Ens']
    CESM_filetypes = ['atm','lnd']
    # WRF experiments (short or long name)
    WRF_experiments = [] # use None to process all WRF experiments
#     WRF_project = 'GreatLakes' # only GreatLakes experiments
#     WRF_experiments += ['erai-g3','erai-t3']
#     WRF_experiments += ['erai-g','erai-t']
#     WRF_experiments += ['g-ensemble','t-ensemble']
#     WRF_experiments += ['g3-ensemble','t3-ensemble']
#     WRF_experiments += ['g-ensemble']
    WRF_project = 'WesternCanada' # only WesternCanada experiments
#     WRF_experiments += ['max-ensemble']
#     WRF_experiments += ['ctrl-ensemble']
#     WRF_experiments += ['erai-max','erai-ctrl']
    WRF_experiments += ['max-ensemble','ctrl-ensemble']
    # other WRF parameters 
#     WRF_domains = 1 # domains to be processed (None=all)
    WRF_domains = None # process all domains
    WRF_filetypes = ('hydro','srfc','xtrm','lsm',) # available input files
#     WRF_filetypes = ('aux',) # only preprocessed auxiliary files
#     WRF_filetypes = ('hydro',) # only preprocessed auxiliary files
    ## observations (i.e. the reference dataset; arguments passed to loadDataset)
    obs_mode = 'climatology'
    obs_name = 'CRU'
    obs_args = dict(period=(1979,1994))
#     obs_args = dict(varatts=dict(pet=dict(name='pet_wrf')), period=(1979,1994)) # pet_wrf is deprecated now
#     obs_name = 'NRCan'
#     obs_args = dict(resolution='na12', period=(1970,2000))
#     obs_args = dict(resolution='na12', period=(1980,2010)) # deprecated: varatts=dict(pet=dict(name='pet_wrf')
    # renaming NRCan pet to pet_wrf is necessary to bias-correct WRF PET
    ## remaining parameters
    lgzip = True # compress pickles
    tag = None # an additional tag string for pickle name
    load_list = None # variables that need to be loaded
    varlist = None # variables that should be bias-corrected
    grid = 'arb3' # need a common grid for all datasets    
#     grid = 'grw1' # need a common grid for all datasets
#     grid = 'grw2' # need a common grid for all datasets
#     grid = 'grw3' # need a common grid for all datasets
#     grid = 'uph1' # grid for Elisha
#     grid = 'asb1' # need a common grid for all datasets
#     grid = 'brd1' # need a common grid for all datasets
#     grid = 'snw1' # need a common grid for all datasets
#     grid = 'son1' # 5km southern Ontario
#     grid = 'son2' # 1km southern Ontario
    bc_method = 'AABC' # annual average bias correction method
#     bc_method = 'SMBC' # annual average bias correction method
#     bc_method = 'Delta' # grid-point-wise monthly bias correction method
#     bc_method = 'MyBC' # BC methid with some custom functions
    bc_args = dict() # paramters for bias correction
  
  ## process arguments
  if isinstance(periods, (np.integer,int)): periods = [periods]
  # check and expand WRF experiment list
  WRF_experiments = getExperimentList(WRF_experiments, WRF_project, 'WRF')
  if isinstance(WRF_domains, (np.integer,int)): WRF_domains = [WRF_domains]
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
  print(('\n On Grid/Resolution:\n   {:s}'.format(grid)))
  print(('\n Variable List: {:s}'.format('All' if varlist is None else printList(varlist))))
  print(('\n Bias-Correction Method: {:s}'.format(bc_method)))
  print(('\n Observationa/Reference Dataset:\n   {:s}'.format(str(obs_dataset) if ldebug else obs_dataset.name)))
  print(('\n OVERWRITE: {0:s}\n'.format(str(loverwrite))))
  
    
  ## construct argument list
  args = []  # list of job packages
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode[-5:] == '-mean': periodlist = periods
    elif mode == 'climatology': periodlist = periods
    elif mode == 'time-series': periodlist = (None,)
    else: raise NotImplementedError("Unrecognized Mode: '{:s}'".format(mode))

      
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
      if WRF_domains is None:
        tmpdom = list(range(1,experiment.domains+1))
      else: tmpdom = WRF_domains
      for domain in tmpdom:
        for period in periodlist:
          # arguments for worker function: dataset and dataargs       
          args.append( ('WRF', mode, dict(experiment=experiment, filetypes=WRF_filetypes, grid=grid, 
                                          varlist=load_list, domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(obs_dataset=obs_dataset, bc_method=bc_method, bc_args=bc_args, loverwrite=loverwrite, lgzip=lgzip, tag=tag)
  # N.B.: formats will be iterated over inside export function
  
  ## call parallel execution function
  ec = asyncPoolEC(generateBiasCorrection, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+int(10.*ec/len(args))) if ec > 0 else 0)
