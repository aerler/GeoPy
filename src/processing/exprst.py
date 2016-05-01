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
from geodata.base import Dataset
from geodata.gdal import addGDALtoDataset
from geodata.misc import DateError, printList, ArgumentError, VariableError
from datasets import gridded_datasets
from processing.multiprocess import asyncPoolEC
from processing.misc import getMetaData,  getExperimentList, loadYAML

## functions to compute relevant variables

# compute surface water flux
def computeWaterFlux(dataset):
  ''' function to compute the net water flux at the surface '''
  # check prerequisites
  if 'liqprec' in dataset: # this is the preferred computation
    for pv in ('evap','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for net water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['liqprec'] + dataset['snwmlt'] - dataset['evap']
  elif 'solprec' in dataset: # alternative computation, mainly for CESM
    for pv in ('precip','evap','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for net water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['precip'] - dataset['solprec'] + dataset['snwmlt'] - dataset['evap']
  else: 
    raise VariableError, "No liquid or solid precip found to compute net water flux."
  var.name = 'waterflx' # give correct name (units should be correct)
  assert var.units == dataset['evap'].units, var
  # return new variable
  return var

# compute downward/liquid component of surface water flux
def computeLiquidWaterFlux(dataset):
  ''' function to compute the downward/liquid component of water flux at the surface '''
  # check prerequisites
  if 'liqprec' in dataset: # this is the preferred computation
    for pv in ('liqprec','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for liquid water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['liqprec'] + dataset['snwmlt']
  elif 'solprec' in dataset: # alternative computation, mainly for CESM
    for pv in ('precip','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for net water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['precip'] - dataset['solprec'] + dataset['snwmlt']
  else: 
    raise VariableError, "No liquid or solid precip found to compute net water flux.".format(pv)
  var.name = 'liqwatflx' # give correct name (units should be correct)
  assert var.units == dataset['snwmlt'].units, var
  # return new variable
  return var

# compute potential evapo-transpiration
def computePotEvapPM(dataset):
  ''' function to compute potential evapotranspiration (according to Penman-Monteith method:
      https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation,
      http://www.fao.org/docrep/x0490e/x0490e06.htm#formulation%20of%20the%20penman%20monteith%20equation)
  '''
  raise NotImplementedError
  # check prerequisites
  for pv in (): 
    if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for potential evapo-transpiration not found.".format(pv)
    else: dataset[pv].load() # load data for computation
  # compute waterflux (returns a Variable instance)
  var = dataset['']
  var.name = 'pet' # give correct name (units should be correct)
  assert var.units == dataset[''].units, var
  # return new variable
  return var

# compute potential evapo-transpiration
def computePotEvapTh(dataset):
  ''' function to compute potential evapotranspiration (according to Thornthwaite method:
      https://en.wikipedia.org/wiki/Potential_evaporation) '''
  raise NotImplementedError
  # check prerequisites
  for pv in (): 
    if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for potential evapo-transpiration not found.".format(pv)
    else: dataset[pv].load() # load data for computation
  # compute waterflux (returns a Variable instance)
  var = dataset['']
  var.name = 'pet' # give correct name (units should be correct)
  assert var.units == dataset[''].units, var
  # return new variable
  return var


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
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs, lone=False)
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr

  # parse export options
  expargs = expargs.copy() # first copy, then modify...
  project = expargs.pop('project')
  varlist = expargs.pop('varlist')
  expfolder = expargs.pop('folder')
  expprefix = expargs.pop('prefix')
  expformat = expargs.pop('format')
  lm3 = expargs.pop('lm3') # convert kg/m^2 to m^3 (water flux)
  # get folder for target dataset and do some checks
  expname = '{:s}_d{:02d}'.format(dataset_name,dataargs.domain) if dataargs.domain else dataset_name
  expprd = 'clim_{:s}'.format(periodstr) if periodstr else 'timeseries' 
  expfolder = expfolder.format(project, grid, expname, expprd)
  expprefix = expprefix.format(project, grid, expname, expprd)
    
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
    source = loadfct() # load source data
    # check period
    if 'period' in source.atts and dataargs.periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)

    # print message
    if mode == 'climatology': opmsgstr = 'Exporting Climatology ({:s}) to {:s} Format'.format(periodstr, expformat)
    elif mode == 'time-series': opmsgstr = 'Exporting Time-series to {:s} Format'.format(expformat)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)        
    # print feedback to logger
    logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
    if not lparallel and ldebug: logger.info('\n'+str(source)+'\n')
    
    # create GDAL-enabled target dataset
    sink = Dataset(axes=(source.xlon,source.ylat), name=expname)
    addGDALtoDataset(dataset=sink, griddef=source.griddef)
    assert sink.gdal, sink
    
    # Compute intermediate variables, if necessary
    for varname in varlist:
      if varname in source:
        var = source[varname].load() # load data (may not have to load all)
      else:
        if varname == 'waterflx': var = computeWaterFlux(source)
        elif varname == 'liqwatflx': var = computeLiquidWaterFlux(source)
        elif varname == 'pet' or varname == 'pet_pm': var = None # skip for now
          #var = computePotEvapPM(source) # default
        elif varname == 'pet_th': var = None # skip for now
          #var = computePotEvapTh(source) # simplified formula (less prerequisites)
        else: raise VariableError, "Unsupported Variable '{:s}'.".format(var)
      # for now, skip variables that are None
      if var:
        # convert units
        if lm3:
          if var.units == 'kg/m^2/s':
            var /= 1000. # divide to get m^3/s
            var.units = 'm^3/s' # update units
        # add to new dataset
        sink += var
      
    
    # print dataset
    if not lparallel and ldebug:
      logger.info('\n'+str(sink)+'\n')
      
    # export to selected format (by variable)
    if expformat == 'ASCII_raster':
      # export dataset to raster format
      folder = sink.ASCII_raster(prefix=expprefix, varlist=None, folder=expfolder, **expargs)
      if not os.path.exists(folder): raise IOError, folder # independent check
    elif expformat == 'NetCDF':
      raise NotImplementedError # not sure if this will ever be useful...
      
      
    # write results to file
    writemsg =  "\n{:s}   >>>   Export of Dataset '{:s}' to Format '{:s}' complete.".format(pidstr,expname, expformat)
    writemsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,expfolder)
    logger.info(writemsg)      
       
    # clean up and return
    source.unload(); #del source
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
  
  lbatch = True
  ## define settings
  if lbatch:
    # load YAML configuration
    config = loadYAML('exprst.yaml', lfeedback=True)
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
    domains = config['WRF_domains']
    grids = config['grids']
    # target data specs
    export_arguments = config['export_parameters'] # this is actually a larger data structure
    lm3 = export_arguments['lm3'] # convert water flux from kg/m^2/s to m^3/s    
  else:
    # settings for testing and debugging
    NP = 2 ; ldebug = False # for quick computations
#     NP = 1 ; ldebug = True # just for tests
    modes = ('climatology',) # 'climatology','time-series'
#     modes = ('time-series',) # 'climatology','time-series'
    loverwrite = True
#     varlist = None
    load_list = ['waterflx','liqprec','solprec','precip','evap','snwmlt','pet']
    periods = []
    periods += [15]
#     periods += [30]
    # Observations/Reanalysis
    resolutions = {'CRU':'','GPCC':'25','NARR':'','CFSR':'05'}
    datasets = [] # this will generally not work, because we don't have snow/-melt...
    lLTM = False # also regrid the long-term mean climatologies 
#     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
    # CESM experiments (short or long name) 
    CESM_project = None # all available experiments
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['Ens']
    CESM_experiments += ['Ctrl-1', 'Ctrl-A', 'Ctrl-B', 'Ctrl-C']
    CESM_filetypes = ['atm','lnd']
    # WRF experiments (short or long name)
    WRF_project = 'GreatLakes' # only GreatLakes experiments
#     WRF_project = 'WesternCanada' # only WesternCanada experiments
    WRF_experiments = [] # use None to process all WRF experiments
    WRF_experiments += ['g-ctrl','g-ctrl-2050','g-ctrl-2100']
#     WRF_experiments += ['new-v361-ctrl', 'new-v361-ctrl-2050', 'new-v361-ctrl-2100']
#     WRF_experiments += ['erai-3km','max-3km']
#     WRF_experiments += ['max-ctrl','max-ctrl-2050','max-ctrl-2100']
#     WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]    
#     WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',]
    # other WRF parameters 
    domains = None # domains to be processed
#     domains = None # process all domains
#     WRF_filetypes = ('hydro','xtrm','srfc','lsm') # filetypes to be processed
    WRF_filetypes = ('hydro',) # filetypes to be processed # ,'rad'
    # typically a specific grid is required
    grids = [] # list of grids to process
#     grids += [None] # special keyword for native grid
    grids += ['grw2']# small grid for HGS GRW project
    ## export parameters
    export_arguments = dict(
        project = 'GRW', # project designation    
        varlist = ['waterflx','liqwatflx','pet'], # varlist for export    
        folder = '{0:s}/HGS/{{0:s}}/{{1:s}}/{{2:s}}/{{3:s}}/'.format(os.getenv('DATA_ROOT', None)),
        prefix = '{0:s}_{1:s}_{2:s}_{3:s}', # argument order: project/grid/experiment/period/
        format = 'ASCII_raster', # formats to export to
        lm3 = True) # convert water flux from kg/m^2/s to m^3/s
  
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
  print('To File Format {:s}'.format(export_arguments['format']))
  print('\n Project Designation: {:s}'.format(export_arguments['project']))
  print('Export Folder: {:s}'.format(export_arguments['folder']))
  print('File Prefix: {:s}'.format(export_arguments['prefix']))
  print('Export Variable List: {:s}'.format(printList(export_arguments['varlist'])))
  if export_arguments['lm3']: '\n Converting kg/m^2/s (mm/s) into m^3/s'
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
  # check formats (will be iterated over in export function, hence not part of task list)
  if export_arguments['format'] not in ('ASCII_raster',):
    raise ArgumentError, "Unsupported file format: '{:s}'".format(export_arguments['format'])
    
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
  kwargs = dict(expargs=export_arguments, loverwrite=loverwrite)
  # N.B.: formats will be iterated over inside export function
  
  ## call parallel execution function
  ec = asyncPoolEC(performExport, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+np.ceil(10.*ec/len(args))) if ec > 0 else 0)
