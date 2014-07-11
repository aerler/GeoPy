'''
Created on 2013-09-23

A script to reproject and resample datasets in this package onto a given grid.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os # check if files are present
import numpy as np
from importlib import import_module
from datetime import datetime
import logging     
# internal imports
from geodata.misc import DatasetError, DateError, isInt, printList
from geodata.netcdf import DatasetNetCDF
from geodata.base import Dataset
from geodata.gdal import GDALError, GridDefinition, addGeoLocator, loadPickledGridDef
from datasets import dataset_list
from datasets.common import addLengthAndNamesOfMonth, getFileName, getCommonGrid, grid_folder
from processing.multiprocess import asyncPoolEC
from processing.process import CentralProcessingUnit
# WRF specific
from datasets.WRF import getWRFgrid, loadWRF
from datasets.WRF import fileclasses as WRF_fileclasses
from projects.WRF_experiments import WRF_exps
# CESM specific
from datasets.CESM import loadCESM, CESM_exps


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performRegridding(dataset, griddef, dataargs, loverwrite=False, varlist=None, lwrite=True, lreturn=False,
                      ldebug=False, lparallel=False, pidstr='', logger=None):
  ''' worker function to perform regridding for a given dataset and target grid '''
  # input checking
  if not isinstance(dataset,basestring): raise TypeError
  if not isinstance(dataargs,dict): raise TypeError # all dataset arguments are kwargs 
  if not isinstance(griddef,GridDefinition): raise TypeError
  if lparallel: 
    if not lwrite: raise IOError, 'Can only write to disk in parallel mode (i.e. lwrite = True).'
    if lreturn: raise IOError, 'Can not return datasets in parallel mode (i.e. lreturn = False).'
  
  # logging
  if logger is None: # make new logger     
    logger = logging.getLogger() # new logger
    logger.addHandler(logging.StreamHandler())
  else:
    if isinstance(logger,basestring): 
      logger = logging.getLogger(name=logger) # connect to existing one
    elif not isinstance(logger,logging.Logger): 
      raise TypeError, 'Expected logger ID/handle in logger KW; got {}'.format(str(logger))

  # load source
  if dataset == 'WRF': 
    # WRF datasets
    module = import_module('datasets.WRF')
    exp = dataargs['experiment']    
    dataset_name = exp.name
    domain = dataargs['domain']
    # figure out period
    period = dataargs['period']
    if isinstance(period,(int,np.integer)):
      beginyear = int(exp.begindate[0:4])
      period = (beginyear, beginyear+period)
    elif len(period) != 2 and all(isInt(period)): raise DateError
    # identify file and domain
    if len(dataargs['filetypes']) > 1: raise DatasetError # process only one file at a time
    filetype = dataargs['filetypes'][0]
    if isinstance(domain,(list,tuple)): domain = domain[0]
    if not isinstance(domain, (np.integer,int)): raise DatasetError
    # load source data 
    source = loadWRF(experiment=dataset_name, name=None, domains=domain, grid=None, period=period, 
                     filetypes=[filetype], varlist=None, varatts=None, lconst=True) # still want topography...
    # source = loadWRF(experiment, name, domains, grid, period, filetypes, varlist, varatts)
    periodstr = '{0:4d}-{1:4d}'.format(*period)
    if 'period' in source.atts and periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)
    datamsgstr = 'Processing WRF Experiment \'{0:s}\' from {1:s}'.format(dataset_name, periodstr)
  elif dataset == 'CESM': 
    # WRF datasets
    module = import_module('datasets.CESM')
    exp = dataargs['experiment']    
    dataset_name = exp.name
    # figure out period
    period = dataargs['period']
    if isinstance(period,(int,np.integer)):
      beginyear = int(exp.begindate[0:4])
      period = (beginyear, beginyear+period)
    elif len(period) != 2 and all(isInt(period)): raise DateError
    # identify file
    if len(dataargs['filetypes']) > 1: raise DatasetError # process only one file at a time
    filetype = dataargs['filetypes'][0]    
    # load source data 
    source = loadCESM(experiment=dataset_name, name=None, grid=None, period=period, filetypes=[filetype],  
                      varlist=None, varatts=None, loadAll=True, translateVars=None)
    periodstr = '{0:4d}-{1:4d}'.format(*period)
    if 'period' in source.atts and periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)
    datamsgstr = 'Processing CESM Experiment \'{0:s}\' from {1:s}'.format(dataset_name, periodstr)  
  elif dataset == dataset.upper():
    # observational datasets
    module = import_module('datasets.{0:s}'.format(dataset))      
    dataset_name = module.dataset_name
    resolution = dataargs['resolution']
    if resolution: grid_name = '{0:s}_{1:s}'.format(dataset_name,resolution)
    else: grid_name = dataset_name   
    # figure out period
    period = dataargs['period']
    if isinstance(period,(int,np.integer)):
      period = (1979, 1979+period) # they all begin in 1979
    elif period is None: pass
    elif len(period) != 2 and not all(isInt(period)): raise DateError
    # load pre-processed climatology
    source = module.loadClimatology(name=dataset_name, period=period, grid=None, resolution=resolution,  
                                    varlist=None, varatts=None, folder=module.avgfolder, filelist=None)
    # loadClimatology(name, period, grid, varlist, varatts, folder, filelist)
    if period is None: periodstr = 'Climatology' 
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
    datamsgstr = 'Processing Dataset {0:s} from {1:s}'.format(dataset_name, periodstr)
  else:
    raise DatasetError, 'Dataset \'{0:s}\' not found!'.format(dataset)
  opmsgstr = 'Reprojecting and Resampling to {0:s} Grid'.format(griddef.name)      
  # print feedback to logger
  # source.load() # not really necessary
  logger.info('\n{0:s}   ***   {1:^50s}   ***   \n{0:s}   ***   {2:^50s}    ***   \n'.format(pidstr,datamsgstr,opmsgstr))
  if not lparallel:
    logger.info('\n'+str(source)+'\n')
  # determine age of oldest source file
  if not loverwrite:
    sourceage = datetime.today()
    for filename in source.filelist:
      age = datetime.fromtimestamp(os.path.getmtime(filename))
      sourceage = age if age < sourceage else sourceage    
          
  # prepare target dataset
  if dataset == 'WRF':
    gridstr = '_{}'.format(griddef.name.lower()) if griddef.name.lower() else ''
    periodstr = '_{}'.format(periodstr) if periodstr else ''# I know, this is pointless at the moment...
    if lwrite:
      filename = module.clim_file_pattern.format(filetype,domain,gridstr,periodstr)
      avgfolder = '{0:s}/{1:s}/'.format(module.avgfolder,dataset_name)    
  elif dataset == 'CESM':
    gridstr = '_{}'.format(griddef.name.lower()) if griddef.name.lower() else ''
    periodstr = '_{}'.format(periodstr) if periodstr else ''# I know, this is pointless at the moment...
    if lwrite:
      filename = module.clim_file_pattern.format(filetype,gridstr,periodstr)
      avgfolder = '{0:s}/{1:s}/'.format(module.avgfolder,dataset_name)    
  elif dataset == dataset.upper(): # observational datasets
    if lwrite:
      filename = getFileName(grid=griddef.name, period=period, name=grid_name, filepattern=None)
      avgfolder = module.avgfolder
  else: raise DatasetError
  if ldebug: filename = 'test_' + filename
  if not os.path.exists(avgfolder): raise IOError, 'Dataset folder \'{0:s}\' does not exist!'.format(avgfolder)
  lskip = False # else just go ahead
  if lwrite:
    filepath = avgfolder+filename
    if os.path.exists(filepath): 
      if not loverwrite: 
        age = datetime.fromtimestamp(os.path.getmtime(filepath))
        # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
        if age > sourceage and os.path.getsize(filepath) > 1e6: lskip = True
        # N.B.: NetCDF files smaller than 1MB are usually incomplete header fragments from a previous crashed
      if not lskip: os.remove(filepath) # recompute
  
  # depending on last modification time of file or overwrite setting, start computation, or skip
  if lskip:        
    # print message
    logger.info('{0:s}   >>>   Skipping: File \'{1:s}\' already exists and is newer than source file.   <<<   \n'.format(pidstr,filename))              
  else:
          
    ## create new sink/target file
    # set attributes   
    atts=source.atts
    atts['period'] = periodstr; atts['name'] = dataset_name; atts['grid'] = griddef.name
    atts['title'] = '{0:s} Climatology on {1:s} Grid'.format(dataset_name, griddef.name)
    # make new dataset
    if lwrite: # write to NetCDF file 
      sink = DatasetNetCDF(folder=avgfolder, filelist=[filename], atts=atts, mode='w')
    else: sink = Dataset(atts=atts) # ony create dataset in memory
    
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=True)
  
    # perform regridding (if target grid is different from native grid!)
    if griddef.name != dataset:
      # reproject and resample (regrid) dataset
      CPU.Regrid(griddef=griddef, flush=False)

    # get results
    CPU.sync(flush=True)
    
    # add geolocators
    sink = addGeoLocator(sink, griddef=griddef, lgdal=True, lreplace=True, lcheck=True)
    # N.B.: WRF datasets come with their own geolocator arrays - we need to replace those!

    # add length and names of month
    if not sink.hasVariable('length_of_month') and sink.hasVariable('time'): 
      addLengthAndNamesOfMonth(sink, noleap=False) 
    
    # print dataset
    if not lparallel:
      logger.info('\n'+str(sink)+'\n')   
    # write results to file
    if lwrite:
      sink.sync()
      logger.info('\n{0:s} Writing to: \'{1:s}\'\n'.format(pidstr,filename))
      if not lreturn: sink.close()
    # return dataset
    if lreturn:
      # return dataset for further use
      return sink
    else:
      return 0 # "exit code"


if __name__ == '__main__':
  
  ## read arguments
  # number of processes NP 
  if os.environ.has_key('PYAVG_THREADS'): 
    NP = int(os.environ['PYAVG_THREADS'])
  else: NP = None
  # run script in debug mode
  if os.environ.has_key('PYAVG_DEBUG'): 
    ldebug =  os.environ['PYAVG_DEBUG'] == 'DEBUG' 
  else: ldebug = False # i.e. append
  # run script in interactive mode
  if os.environ.has_key('PYAVG_INTERACT'): 
    linteract =  os.environ['PYAVG_INTERACT'] == 'INTERACT' 
  else: linteract = False # i.e. append  
  # re-compute everything or just update 
  if os.environ.has_key('PYAVG_OVERWRITE'): 
    loverwrite =  os.environ['PYAVG_OVERWRITE'] == 'OVERWRITE' 
  else: loverwrite = ldebug # False means only update old files
  
  # default settings
  if linteract:
    ldebug = False
    NP = NP or 4
    loverwrite = True
    varlist = None
#     varlist = ['precip',]
    periods = []
#     periods += [1]
#     periods += [3]
#     periods += [5]
    periods += [10]
#     periods += [15]
#     periods += [30]
#     periods += [(1984,1994)]
#     periods += [(1989,1994)]
#     periods += [(1949,2009)]
#     periods += [(1997,1998)]
    # Observations/Reanalysis
    datasets = []
#     datasets += ['PRISM','GPCC']; periods = None
    datasets += ['PCIC']; periods = None
#     datasets += ['CFSR', 'NARR']
#     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
    resolutions = None
    # CESM
    CESM_experiments = []
#     CESM_experiments += ['CESM','CESM-2050']
#     CESM_experiments += ['Ctrl', 'Ens-A', 'Ens-B', 'Ens-C']
#     CESM_experiments += ['Ctrl-2050', 'Ens-A-2050', 'Ens-B-2050', 'Ens-C-2050']
    CESM_filetypes = ['atm','lnd']
    # WRF
    WRF_experiments = []
#     WRF_experiments += ['max']
#     WRF_experiments += ['max-1deg']
#     WRF_experiments += ['ctrl-1-arb1', 'ctrl-2-arb1']
#     WRF_experiments += ['max','max-lowres','max-nmp','max-nosub']
#     WRF_experiments += ['max','max-A','max-nofdda','max-fdda']
#     WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]    
#     WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',]
#     WRF_experiments += ['max-ens','max-ens-2050']
#     WRF_experiments += ['new','grell','gulf','cfsr-new']
#     WRF_experiments = ['new-grell-old'] # WRF experiment names (passed through WRFname)
    domains = [1,2] # domains to be processed
#     WRF_filetypes = ['hydro','xtrm','srfc','lsm'] # filetypes to be processed
    WRF_filetypes = ['srfc','xtrm','plev3d','hydro','lsm','rad'] # filetypes to be processed
#     WRF_filetypes = ['srfc']
    # grid to project onto
    lpickle = True
    grids = dict()
    grids['col1'] = ['d03','d02','d01'] # innermost WRF Columbia domain
#     grids['col2'] = ['d03','d02','d01'] # innermost WRF Columbia domain
#     grids['grb2'] = ['d02'] # Marc's standard GRB inner domain
#     grids['arb2'] = ['d01','d02'] # WRF standard ARB inner domain
#     grids['arb3'] = ['d02'] # WRF new ARB inner domain
#     grids['ARB_small'] = ['025','05'] # small custom geographic grids
#     grids['ARB_large'] = ['025','05'] # large custom geographic grids
#     grids['cesm1x1'] = [None] # CESM grid
#     grids = dict(NARR=[None]) # CESM grid
  else:
    NP = NP or 4
    #loverwrite = False
    varlist = None # process all variables
    datasets = None # process all applicable
    periods = [5,10,15] # climatology periods to process
#     periods = [(1979,1984),(1979,1989)] # climatology periods to process 
#     periods = None # process only overall climatologies 
    resolutions = None
    # CESM
    CESM_experiments = None
    CESM_filetypes = ['atm','lnd']    
    # WRF
    WRF_experiments = [] # process all WRF experiments
    WRF_experiments += ['max','gulf','new','noah'] # WRF experiment names (passed through WRFname) 
    domains = [1,2] # domains to be processed
    WRF_filetypes = WRF_fileclasses.keys() # process all filetypes 
    # grid to project onto
    lpickle = True
    #d12 = ['d01','d02']
    #grids = dict(arb1=d12, arb2=d12, arb3=d12) # dict with list of resolutions
    grids = dict(arb2=['d02']) # dict with list of resolutions  
    
  
  ## process arguments    
  # expand experiments
  if WRF_experiments is None: WRF_experiments = WRF_exps.values() # do all 
  else: WRF_experiments = [WRF_exps[exp] for exp in WRF_experiments]
  if CESM_experiments is None: CESM_experiments = CESM_exps.values() # do all 
  else: CESM_experiments = [CESM_exps[exp] for exp in CESM_experiments]
  
  # expand datasets and resolutions
  if datasets is None: datasets = dataset_list  
  if resolutions is None: resolutions = dict()
  elif not isinstance(resolutions,dict): raise TypeError 
  new_ds = []
  for dataset in datasets:
    mod = import_module('datasets.{0:s}'.format(dataset))    
    if periods is None:
      if len(mod.LTM_grids) > 0: 
        new_ds.append(dataset)
        if dataset not in resolutions or resolutions[dataset] is None: resolutions[dataset] = mod.LTM_grids
    else:
      if len(mod.TS_grids) > 0: 
        new_ds.append(dataset)
        if dataset not in resolutions or resolutions[dataset] is None: resolutions[dataset] = mod.TS_grids
  if periods is None: periods = [None]
  datasets = new_ds      
  
  # print an announcement
  if len(WRF_experiments) > 0:
    print('\n Regridding WRF Datasets:')
    print([exp.name for exp in WRF_experiments])
  if len(CESM_experiments) > 0:
    print('\n Regridding CESM Datasets:')
    print([exp.name for exp in CESM_experiments])
  if len(datasets) > 0:
    print(' And Observational Datasets:')
    print(datasets)
  print('\n To Grid and Resolution:')
  for grid,reses in grids.iteritems():
    print('   {0:s} {1:s}'.format(grid,printList(reses)))
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
    
  ## construct argument list
  args = []  # list of job packages
  # loop over target grids ...
  for grid,reses in grids.iteritems():
    # ... and resolutions
    for res in reses:
      
      # load target grid definition
      if lpickle:
        griddef = loadPickledGridDef(grid=grid, res=res, folder=grid_folder)
      else:
        griddef = getCommonGrid(grid) # try this first (common grids)
        # else, determine new grid from existing dataset
        if griddef is None:
          if grid == grid.lower(): # WRF grid      
            griddef = getWRFgrid(experiment=grid, domains=[1])
          elif grid == grid.upper(): # observations
            griddef = import_module(grid[0:4]).__dict__[grid+'_grid']
          else: pass # we could try CESM grids here, at a later stage
      # check if grid was defined properly
      if not isinstance(griddef,GridDefinition): 
        raise GDALError, 'No valid grid defined! (grid={0:s})'.format(grid)        
      
      # observational datasets
      for dataset in datasets:
        for period in periods:
          for resolution in resolutions[dataset]:
            # arguments for worker function: dataset and dataargs       
            args.append( (dataset, griddef, dict(period=period, resolution=resolution)) ) # append to list               
      # CESM datasets
      for experiment in CESM_experiments:
        for filetype in CESM_filetypes:
          for period in periods:
            # arguments for worker function: dataset and dataargs       
            args.append( ('CESM', griddef, dict(experiment=experiment, filetypes=[filetype], period=period)) )
      # WRF datasets
      for experiment in WRF_experiments:
        for filetype in WRF_filetypes:
          for domain in domains:
            for period in periods:
              # arguments for worker function: dataset and dataargs       
              args.append( ('WRF', griddef, dict(experiment=experiment, filetypes=[filetype], domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(loverwrite=loverwrite, varlist=varlist)
          
  ## call parallel execution function
  asyncPoolEC(performRegridding, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
