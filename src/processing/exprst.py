'''
Created on 2016-04-21

A script to convert datasets to raster format using GDAL. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import os, shutil # check if files are present etc.
import numpy as np
from importlib import import_module
from datetime import datetime
import logging     
# internal imports
from geodata.misc import DateError, printList, ArgumentError, VariableError
from geodata.netcdf import DatasetNetCDF
from geodata.base import Dataset
from geodata.gdal import addGeoLocator
from datasets import gridded_datasets
from datasets.common import addLengthAndNamesOfMonth
from processing.multiprocess import asyncPoolEC
from processing.process import CentralProcessingUnit
from processing.misc import getMetaData,  getExperimentList, loadYAML


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performExport(dataset, mode, dataargs, expargs, loverwrite=False, 
                  ldebug=False, lparallel=False, pidstr='', logger=None):
  ''' worker function to perform regridding for a given dataset and target grid '''
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
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs)
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr

  # parse export options
  project = expargs.pop('project')
  varlist = expargs.pop('varlist')
  expfolder = expargs.pop('folder')
  expformat = expargs.pop('format')
  lm3 = expargs.pop('lm3') # convert kg/m^2 to m^3 (water flux)
  # get folder for target dataset and do some checks
  expfolder = expfolder.format(project, dataset_name, grid)
    
  # prepare target dataset (which is mainly just a folder)
  if ldebug: expfolder = expfolder + 'test/' # test in subfolder
  if not os.path.exists(expfolder): 
    # create new folder
    os.makedirs(expfolder)
    lskip = False # actually do export
  elif loverwrite:
    shutil.rmtree(expfolder) # remove old folder and contents
    os.makedirs(expfolder) # create new folder
    lskip = False # actually do export
  else:
    age = datetime.fromtimestamp(os.path.getmtime(expfolder))
    # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
    lskip = ( age > srcage ) # skip if newer than source    
  assert os.path.exists(expfolder), expfolder
  
  # depending on last modification time of file or overwrite setting, start computation, or skip
  if lskip:        
    # print message
    skipmsg =  "\n{:s}   >>>   Skipping: Format '{:s} for dataset '{:s}' already exists and is newer than source file.".format(pidstr,expformat,dataset_name)
    skipmsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,expfolder)
    logger.info(skipmsg)              
  else:
          
    ## actually load datasets
    dataset = loadfct() # load source data
    # check period
    if 'period' in dataset.atts and dataargs.periodstr != dataset.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,dataset.atts.period)

    # print message
    if mode == 'climatology': opmsgstr = 'Exporting Climatology ({:s}) to {:s} Format'.format(periodstr, expformat)
    elif mode == 'time-series': opmsgstr = 'Exporting Time-series to {:s} Format'.format(expformat)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)        
    # print feedback to logger
    logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
    if not lparallel and ldebug: logger.info('\n'+str(dataset)+'\n')
    
    # Compute intermediate variables, if necessary
    for var in varlist:
      if var in dataset:
        dataset[var].load() # load data (may not have to load all)
      else:
        if var == 'waterflx':
          if all(v in dataset for v in ('liqprec','evap','snwmlt')):
            pass
          else: raise VariableError, "Prerequisites for Variable '{:s}' not found.".format(var)
        else: raise VariableError, "Unsupported Variable '{:s}'.".format(var)
            
    # convert units for water flux
    if lm3:
      for varname in varlist:
        var = dataset[varname] # this is just a reference
        if var.units == 'kg/m^2/s':
          var /= 1000. # divide to get m^3/s
          var.units = 'm^3/s' # update units
          assert dataset[varname].units == 'm^3/s'
    
    # print dataset
    if not lparallel and ldebug:
      logger.info('\n'+str(dataset)+'\n')
      
    # export to selected format (by variable)
    if expformat == 'ASCII_raster':
      for var in varlist:
        # call export function for format from Variable
        filepath = getattr(dataset[var],expformat)(folder=expfolder, **expargs)
        if not os.path.exists(filepath): raise IOError, filepath # independent check
    elif expformat == 'NetCDF':
      raise NotImplementedError
      
      
    # write results to file
    writemsg =  "\n{:s}   >>>   Export of Dataset '{:s}' to Format '{:s}' complete.".format(pidstr,dataset_name, expformat)
    writemsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,expfolder)
    logger.info(writemsg)      
       
    # clean up and return
    dataset.unload(); #del dataset
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
    config = loadYAML('regrid.yaml', lfeedback=True)
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
    domains = config['domains']
    grids = config['grids']
    # target data specs
    export_arguments = config['export_arguments'] # this is actually a larger data structure
  else:
    # settings for testing and debugging
#     NP = 2 ; ldebug = False # for quick computations
    NP = 1 ; ldebug = True # just for tests
    modes = ('climatology',) # 'climatology','time-series'
#     modes = ('time-series',) # 'climatology','time-series'
    loverwrite = True
#     varlist = None
    load_list = ['waterflx','liqprec','evap','snwmlt']
    periods = []
    periods += [15]
#     periods += [30]
    # Observations/Reanalysis
    resolutions = {'CRU':'','GPCC':'25','NARR':'','CFSR':'05'}
    datasets = []
    lLTM = True # also regrid the long-term mean climatologies 
#     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
    # CESM experiments (short or long name) 
    CESM_project = None # all available experiments
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['CESM','CESM-2050']
#     CESM_experiments += ['Ctrl', 'Ens-A', 'Ens-B', 'Ens-C']
#     CESM_filetypes = ['atm','lnd']
    CESM_filetypes = ['atm']
    # WRF experiments (short or long name)
    WRF_project = 'GreatLakes' # only GreatLakes experiments
#     WRF_project = 'WesternCanada' # only WesternCanada experiments
    WRF_experiments = [] # use None to process all WRF experiments
    WRF_experiments += ['g-ctrl']
#     WRF_experiments += ['new-v361-ctrl', 'new-v361-ctrl-2050', 'new-v361-ctrl-2100']
#     WRF_experiments += ['erai-3km','max-3km']
#     WRF_experiments += ['max-ctrl','max-ctrl-2050','max-ctrl-2100']
#     WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]    
#     WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',]
    # other WRF parameters 
    domains = 2 # domains to be processed
#     domains = None # process all domains
#     WRF_filetypes = ('hydro','xtrm','srfc','lsm') # filetypes to be processed
    WRF_filetypes = ('hydro',) # filetypes to be processed # ,'rad'
    # typically a specific grid is required
    grids = [] # list of grids to process
#     grids += [None] # special keyword for native grid
    grids += ['grw2']# small grid for HGS GRW project
    ## export parameters
    project = 'GRW' # project designation    
    varlist = ['waterflx'] # varlist for export    
    expfolder = '{0:s}/HGS/{{0:s}}/{{1:s}}/{{2:s}}/'.format(os.getenv('DATA_ROOT', None)) # project/experiment/grid 
    expformat = 'ASCII_raster' # formats to export to
    lm3 = True # convert water flux from kg/m^2/s to m^3/s
    # assemble export arguments
    export_arguments = dict(project=project, varlist=varlist, format=expformat, folder=expfolder, lm3=lm3)
  
  ## process arguments    
  if isinstance(periods, (np.integer,int)): periods = [periods]
  # check and expand WRF experiment list
  WRF_experiments = getExperimentList(WRF_experiments, WRF_project, 'WRF')
  if isinstance(domains, (np.integer,int)): domains = [domains]
  # check and expand CESM experiment list
  CESM_experiments = getExperimentList(CESM_experiments, CESM_project, 'CESM')
  # expand datasets and resolutions
  if datasets is None: datasets = gridded_datasets  
  
  # print an announcement
  if len(WRF_experiments) > 0:
    print('\n Exporting WRF Datasets:')
    print([exp.name for exp in WRF_experiments])
  if len(CESM_experiments) > 0:
    print('\n Exporting CESM Datasets:')
    print([exp.name for exp in CESM_experiments])
  if len(datasets) > 0:
    print('\n And Observational Datasets:')
    print(datasets)
  print('\n From Grid/Resolution:\n   {:s}'.format(printList(grids)))
  print('\n To File Format {:s}'.format(expformat))
  print('   ({:s})'.format(expfolder))
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
  # check formats (will be iterated over in export function, hence not part of task list)
  if expformat.lower() not in ('ascii_raster','netcdf'):
    raise ArgumentError, "Unsupported file format: '{:s}'".format(expformat)
    
  ## construct argument list
  args = []  # list of job packages
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode == 'climatology': periodlist = periods
    elif mode == 'time-series': periodlist = (None,)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)

    # loop over target grids ...
    for grid in grids:
      
        # observational datasets (grid depends on dataset!)
        for dataset in datasets:
          mod = import_module('datasets.{0:s}'.format(dataset))
          if isinstance(resolutions,dict): 
            if dataset not in resolutions: resolutions[dataset] = ('',)
            elif not isinstance(resolutions[dataset],(list,tuple)): resolutions[dataset] = (resolutions[dataset],)                
          elif resolutions is not None: raise TypeError                                
          if mode == 'climatology':
            # some datasets come with a climatology 
            if lLTM:
              if resolutions is None: dsreses = mod.LTM_grids
              elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.LTM_grids]  
              for dsres in dsreses: 
                args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=None, resolution=dsres)) ) # append to list
            # climatologies derived from time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              for period in periodlist:
                args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=period, resolution=dsres)) ) # append to list            
          elif mode == 'time-series': 
            # regrid the entire time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=None, resolution=dsres)) ) # append to list            
        
        # CESM datasets
        for experiment in CESM_experiments:
          for filetype in CESM_filetypes:
            for period in periodlist:
              # arguments for worker function: dataset and dataargs       
              args.append( ('CESM', mode, dict(experiment=experiment, filetypes=[filetype], grid=grid, varlist=load_list, 
                                               period=period, load3D=load3D)) )
        # WRF datasets
        for experiment in WRF_experiments:
          for filetype in WRF_filetypes:
            # effectively, loop over domains
            if domains is None:
              tmpdom = range(1,experiment.domains+1)
            else: tmpdom = domains
            for domain in tmpdom:
              for period in periodlist:
                # arguments for worker function: dataset and dataargs       
                args.append( ('WRF', mode, dict(experiment=experiment, filetypes=[filetype], grid=grid, varlist=load_list, 
                                                domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(expargs=export_arguments, loverwrite=loverwrite)
  # N.B.: formats will be iterated over inside export function
  
  ## call parallel execution function
  ec = asyncPoolEC(performExport, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+np.ceil(10.*ec/len(args))) if ec > 0 else 0)
