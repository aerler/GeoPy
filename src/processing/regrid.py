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
from datasets.WRF import getWRFgrid, loadWRF, loadWRF_TS
from projects.WRF_experiments import WRF_exps
# CESM specific
from datasets.CESM import loadCESM, loadCESM_TS, CESM_exps


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performRegridding(dataset, mode, griddef, dataargs, loverwrite=False, varlist=None, lwrite=True, lreturn=False,
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
    if period is None: pass
    elif isinstance(period,(int,np.integer)):
      beginyear = int(exp.begindate[0:4])
      period = (beginyear, beginyear+period)
    elif len(period) != 2 and all(isInt(period)): raise DateError
    del exp
    # identify file and domain
    if len(dataargs['filetypes']) > 1: raise DatasetError # process only one file at a time
    filetype = dataargs['filetypes'][0]
    if isinstance(domain,(list,tuple)): domain = domain[0]
    if not isinstance(domain, (np.integer,int)): raise DatasetError
    # load source data
    if mode == 'climatology':
      source = loadWRF(experiment=dataset_name, name=None, domains=domain, grid=None, period=period, 
                       filetypes=[filetype], varlist=None, varatts=None, lconst=True) # still want topography...
    elif mode == 'time-series':
      source = loadWRF_TS(experiment=dataset_name, name=None, domains=domain, grid=None, 
                       filetypes=[filetype], varlist=None, varatts=None, lconst=True) # still want topography...
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)
    # check period
    if period is None: periodstr = '' 
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
    if 'period' in source.atts and periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)
    datamsgstr = "Processing WRF '{:s}'-file from Experiment '{:s}' (d{:02d})".format(filetype, dataset_name, domain)
  elif dataset == 'CESM': 
    # WRF datasets
    module = import_module('datasets.CESM')
    exp = dataargs['experiment']    
    dataset_name = exp.name
    # figure out period
    period = dataargs['period']
    if period is None: pass
    elif isinstance(period,(int,np.integer)):
      beginyear = int(exp.begindate[0:4])
      period = (beginyear, beginyear+period)
    elif len(period) != 2 and all(isInt(period)): raise DateError
    del exp
    # identify file
    if len(dataargs['filetypes']) > 1: raise DatasetError # process only one file at a time
    filetype = dataargs['filetypes'][0]        
    # load source data 
    load3D = dataargs.pop('load3D',None) # if 3D fields should be loaded (default: False)
    if mode == 'climatology':
      source = loadCESM(experiment=dataset_name, name=None, grid=None, period=period, filetypes=[filetype],  
                        varlist=None, varatts=None, load3D=load3D, translateVars=None)
    elif mode == 'time-series':
      source = loadCESM_TS(experiment=dataset_name, name=None, grid=None, filetypes=[filetype],  
                           varlist=None, varatts=None, load3D=load3D, translateVars=None)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)
    # check period
    if period is None: periodstr = ''
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
    if 'period' in source.atts and periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)
    datamsgstr = "Processing CESM '{:s}'-file from Experiment '{:s}'".format(filetype, dataset_name)  
  elif dataset == dataset.upper():
    # observational datasets
    module = import_module('datasets.{0:s}'.format(dataset))      
    dataset_name = module.dataset_name
    resolution = dataargs['resolution']
    if resolution: grid_name = '{0:s}_{1:s}'.format(dataset_name,resolution)
    else: grid_name = dataset_name   
    # figure out period
    period = dataargs['period']    
    if period is None: pass
    elif isinstance(period,(int,np.integer)):
      period = (1979, 1979+period) # they all begin in 1979
    elif len(period) != 2 and not all(isInt(period)): raise DateError
    # load pre-processed climatology
    if mode == 'climatology':
      source = module.loadClimatology(name=dataset_name, period=period, grid=None, resolution=resolution,  
                                      varlist=None, varatts=None, folder=module.avgfolder, filelist=None)
    elif mode == 'time-series':
      source = module.loadTimeSeries(name=dataset_name, grid=None, resolution=resolution,  
                                      varlist=None, varatts=None, folder=None, filelist=None)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)
    datamsgstr = "Processing Dataset '{:s}'".format(dataset_name)
    # check period
    if period is None: 
      if mode == 'climatology': periodstr = 'Long-Term Mean'
      else: periodstr = ''
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
  else:
    raise DatasetError, "Dataset '{:s}' not found!".format(dataset)
  del dataargs
  # common message
  if mode == 'climatology': opmsgstr = 'Regridding Climatology ({:s}) to {:s} Grid'.format(periodstr, griddef.name)
  elif mode == 'time-series': opmsgstr = 'Regridding Time-series to {:s} Grid'.format(griddef.name)
  else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)        
  # print feedback to logger
  logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
  if not lparallel and ldebug:
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
    periodstr = '_{}'.format(periodstr) if periodstr else ''
    if lwrite:
      if mode == 'climatology': filename = module.clim_file_pattern.format(filetype,domain,gridstr,periodstr)
      elif mode == 'time-series': filename = module.ts_file_pattern.format(filetype,domain,gridstr)
      else: raise NotImplementedError
      avgfolder = '{0:s}/{1:s}/'.format(module.avgfolder,dataset_name)    
  elif dataset == 'CESM':
    gridstr = '_{}'.format(griddef.name.lower()) if griddef.name.lower() else ''
    periodstr = '_{}'.format(periodstr) if periodstr else ''
    if lwrite:
      if mode == 'climatology': filename = module.clim_file_pattern.format(filetype,gridstr,periodstr)
      elif mode == 'time-series': filename = module.ts_file_pattern.format(filetype,gridstr)
      else: raise NotImplementedError
      avgfolder = '{0:s}/{1:s}/'.format(module.avgfolder,dataset_name)    
  elif dataset == dataset.upper(): # observational datasets
    if lwrite:
      avgfolder = module.avgfolder
      filename = getFileName(grid=griddef.name, period=period, name=grid_name, filetype=mode)      
  else: raise DatasetError
  if ldebug: filename = 'test_' + filename
  if not os.path.exists(avgfolder): raise IOError, "Dataset folder '{:s}' does not exist!".format(avgfolder)
  lskip = False # else just go ahead
  if lwrite:
    if lreturn: tmpfilename = filename # no temporary file if dataset is passed on (can't rename the file while it is open!)
    else: 
      if lparallel: tmppfx = 'tmp_wrfavg_{:s}_'.format(pidstr[1:-1])
      else: tmppfx = 'tmp_wrfavg_'.format(pidstr[1:-1])
      tmpfilename = tmppfx + filename      
    filepath = avgfolder + filename
    tmpfilepath = avgfolder + tmpfilename
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
    skipmsg =  "\n{:s}   >>>   Skipping: file '{:s}' in dataset '{:s}' already exists and is newer than source file.".format(pidstr,filename,dataset_name)
    skipmsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,filepath)
    logger.info(skipmsg)              
  else:
          
    ## create new sink/target file
    # set attributes   
    atts=source.atts.copy()
    atts['period'] = periodstr; atts['name'] = dataset_name; atts['grid'] = griddef.name
    if mode == 'climatology': atts['title'] = '{:s} Climatology on {:s} Grid'.format(dataset_name, griddef.name)
    elif mode == 'time-series':  atts['title'] = '{:s} Time-series on {:s} Grid'.format(dataset_name, griddef.name)
      
    # make new dataset
    if lwrite: # write to NetCDF file 
      if os.path.exists(tmpfilepath): os.remove(tmpfilepath) # remove old temp files 
      sink = DatasetNetCDF(folder=avgfolder, filelist=[tmpfilename], atts=atts, mode='w')
    else: sink = Dataset(atts=atts) # ony create dataset in memory
    
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=False, feedback=ldebug)
  
    # perform regridding (if target grid is different from native grid!)
    if griddef.name != dataset:
      # reproject and resample (regrid) dataset
      CPU.Regrid(griddef=griddef, flush=True)

    # get results    
    CPU.sync(flush=True)
    
    # add geolocators
    sink = addGeoLocator(sink, griddef=griddef, lgdal=True, lreplace=True, lcheck=True)
    # N.B.: WRF datasets come with their own geolocator arrays - we need to replace those!
    
    # add length and names of month
    if mode == 'climatology' and not sink.hasVariable('length_of_month') and sink.hasVariable('time'): 
      addLengthAndNamesOfMonth(sink, noleap=True if dataset.upper() in ('WRF','CESM') else False) 
    
    # print dataset
    if not lparallel and ldebug:
      logger.info('\n'+str(sink)+'\n')   
    # write results to file
    if lwrite:
      sink.sync()
      writemsg =  "\n{:s}   >>>   Writing to file '{:s}' in dataset {:s}".format(pidstr,filename,dataset_name)
      writemsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,filepath)
      logger.info(writemsg)      
      
      # rename file to proper name
      if not lreturn:
        sink.unload(); sink.close(); del sink # destroy all references 
        os.rename(tmpfilepath,filepath)
      # N.B.: there is no temporary file if the dataset is returned, because an open file can't be renamed
        
    # clean up and return
    source.unload(); del source, CPU
    if lreturn:      
      return sink # return dataset for further use (netcdf file still open!)
    else:            
      return 0 # "exit code"
    # N.B.: garbage is collected in multi-processing wrapper


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
  # run script in batch or interactive mode
  if os.environ.has_key('PYAVG_BATCH'): 
    lbatch =  os.environ['PYAVG_BATCH'] == 'BATCH' 
  else: lbatch = False # i.e. append  
  # re-compute everything or just update 
  if os.environ.has_key('PYAVG_OVERWRITE'): 
    loverwrite =  os.environ['PYAVG_OVERWRITE'] == 'OVERWRITE' 
  else: loverwrite = ldebug # False means only update old files
  
  # default settings
  if not lbatch:
    ldebug = False
    NP = 4 or NP # to avoid memory issues...
    modes = ('climatology',) # 'climatology','time-series'
#     modes = ('time-series',) # 'climatology','time-series'
    loverwrite = True
    varlist = None
#     varlist = ['lat2D',]
    periods = []
#     periods += [1]
#     periods += [3]
    periods += [5]
#     periods += [10]
#     periods += [15]
#     periods += [30]
#     periods += [(1984,1994)]
#     periods += [(1989,1994)]
#     periods += [(1949,2009)]
#     periods += [(1997,1998)]
    # Observations/Reanalysis
    resolutions = {'CRU':'','GPCC':'25','NARR':'','CFSR':'05'}
    datasets = []
    lLTM = True # also regrid the long-term mean climatologies 
#     datasets += ['PRISM','GPCC']; periods = None
#     datasets += ['PCIC']; periods = None
#     datasets += ['CFSR', 'NARR']
#     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
    # CESM experiments (short or long name) 
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['CESM','CESM-2050']
#     CESM_experiments += ['Ctrl', 'Ens-A', 'Ens-B', 'Ens-C']
#     CESM_experiments += ['Ctrl-2050', 'Ens-A-2050', 'Ens-B-2050', 'Ens-C-2050']
    CESM_filetypes = ['atm','lnd']
    # WRF experiments (short or long name)
    WRF_experiments = [] # use None to process all CESM experiments
    WRF_experiments += ['max-1deg-2100']
#     WRF_experiments += ['max-1deg','max-1deg-2050','max-1deg-2100']
#     WRF_experiments += ['max','max-clm','max-nmp','max-nosub']
#     WRF_experiments += ['max-ctrl','max-ctrl-2050','max-ctrl-2100']
#     WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]    
#     WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',]
#     WRF_experiments += ['max-ens','max-ens-2050']
#     WRF_experiments += ['ctrl-1-arb1', 'new-ctrl', 'max-ctrl'] #  old ctrl simulations (arb1)
#     WRF_experiments += ['new-ctrl', 'new-ctrl-2050', 'cfsr-new', 'new-grell',] # new standard runs (arb3) 
#     WRF_experiments += ['new-grell-old', 'new-noah', 'v35-noah'] # new sensitivity tests (arb3)
#     WRF_experiments += ['cam-ctrl', 'cam-ctrl-1-2050', 'cam-ctrl-2-2050', 'cam-ctrl-2-2100'] # old cam simulations (arb1) 
#     WRF_experiments += ['ctrl-1-arb1', 'ctrl-2-arb1', 'ctrl-arb1-2050'] #  old ctrl simulations (arb1)
#     WRF_experiments += ['cfsr-cam', 'cam-ens-A', 'cam-ens-B', 'cam-ens-C'] # old ensemble simulations (arb1)
    # other WRF parameters 
    domains = (2,) # domains to be processed
#     WRF_filetypes = ('hydro','xtrm','srfc','lsm') # filetypes to be processed
#     WRF_filetypes = ('plev3d',) # filetypes to be processed # ,'rad'
    WRF_filetypes = ('srfc','xtrm','plev3d','hydro','lsm') # filetypes to be processed # ,'rad'
#     WRF_filetypes = ('const',); periods = None
    # grid to project onto
    lpickle = True
    grids = dict()
#     grids['col2'] = ('d03','d02','d01') # innermost WRF Columbia domain
#     grids['grb2'] = ('d02',) # Marc's standard GRB inner domain
#     grids['arb2'] = ('d02',) # WRF standard ARB inner domain
#     grids['arb3'] = ('d02',) # WRF new ARB inner domain
#     grids['ARB_small'] = ('025','05') # small custom geographic grids
#     grids['ARB_large'] = ('025','05') # large custom geographic grids
    grids['cesm1x1'] = (None,) # CESM grid
#     grids['NARR'] = (None,) # NARR grid
#     grids['CRU'] = (None,) # CRU grid
  else:
    NP = NP or 4 # time-series might take more memory!
    modes = ('climatology','time-series')
    #modes = ('time-series',)
    #modes = ('climatology',)
    loverwrite = False
    varlist = None # process all variables
    periods = (5,10,15,) # climatology periods to process
    #periods = (15,) # for tests
    # Datasets
    datasets = None # process all applicable
    resolutions = None # process all applicable
    lLTM = True 
    # CESM
    load3D = False
    CESM_experiments = None
    CESM_filetypes = ('atm','lnd')    
    # WRF
    WRF_experiments = [] # process WRF experiments on different grids
    WRF_experiments += ['new-v36-nmp', 'new-v36-noah', 'erai-v36-noah', 'new-v36-clm',]
    WRF_experiments += ['new-ctrl', 'new-ctrl-2050', 'new-ctrl-2100', 'cfsr-new', ] # new standard runs (arb3) 
    WRF_experiments += ['new-grell', 'new-grell-old', 'new-noah', 'v35-noah'] # new sensitivity tests (arb3)
    WRF_experiments += ['cam-ctrl', 'cam-ctrl-1-2050', 'cam-ctrl-2-2050', 'cam-ctrl-2-2100'] # old cam simulations (arb1) 
    WRF_experiments += ['ctrl-1-arb1', 'ctrl-2-arb1', 'ctrl-arb1-2050'] #  old ctrl simulations (arb1)
    WRF_experiments += ['cfsr-cam', 'cam-ens-A', 'cam-ens-B', 'cam-ens-C'] # old ensemble simulations (arb1)    
    domains = (1,2,) # domains to be processed
    #domains = (2,) # for tests
    WRF_filetypes = ('srfc','xtrm','plev3d','hydro','lsm') # process all filetypes except 'rad'
    #WRF_filetypes = WRF_filetypes = ('hydro',) # for tests
    # grid to project onto
    lpickle = True
    #d12 = ('d01','d02')
    #grids = dict(arb1=d12, arb2=d12, arb3=d12) # dict with list of resolutions
    #grids = dict(arb2=('d02',),cesm1x1=(None,)) # dict with list of resolutions
    grids = dict(arb2=('d02',)) # dict with list of resolutions  
    
  
  ## process arguments    
  if periods is None: periods = [None]
  # expand experiments
  if WRF_experiments is None: WRF_experiments = WRF_exps.values() # do all 
  else: WRF_experiments = [WRF_exps[exp] for exp in WRF_experiments]
  if CESM_experiments is None: CESM_experiments = CESM_exps.values() # do all 
  else: CESM_experiments = [CESM_exps[exp] for exp in CESM_experiments]  
  # expand datasets and resolutions
  if datasets is None: datasets = dataset_list  
#   if resolutions is None: resolutions = dict()
#   elif not isinstance(resolutions,dict): raise TypeError 
  
  # print an announcement
  if len(WRF_experiments) > 0:
    print('\n Regridding WRF Datasets:')
    print([exp.name for exp in WRF_experiments])
  if len(CESM_experiments) > 0:
    print('\n Regridding CESM Datasets:')
    print([exp.name for exp in CESM_experiments])
  if len(datasets) > 0:
    print('\n And Observational Datasets:')
    print(datasets)
  print('\n To Grid and Resolution:')
  for grid,reses in grids.iteritems():
    print('   {0:s} {1:s}'.format(grid,printList(reses)))
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
    
  ## construct argument list
  args = []  # list of job packages
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode == 'climatology': periodlist = periods
    elif mode == 'time-series': periodlist = (None,)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)

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
                args.append( (dataset, mode, griddef, dict(period=None, resolution=dsres)) ) # append to list
            # climatologies derived from time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              for period in periodlist:
                args.append( (dataset, mode, griddef, dict(period=period, resolution=dsres)) ) # append to list            
          elif mode == 'time-series': 
            # regrid the entire time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              args.append( (dataset, mode, griddef, dict(period=None, resolution=dsres)) ) # append to list            
        
        # CESM datasets
        for experiment in CESM_experiments:
          for filetype in CESM_filetypes:
            for period in periodlist:
              # arguments for worker function: dataset and dataargs       
              args.append( ('CESM', mode, griddef, dict(experiment=experiment, filetypes=[filetype], period=period, load3D=load3D)) )
        # WRF datasets
        for experiment in WRF_experiments:
          for filetype in WRF_filetypes:
            for domain in domains:
              for period in periodlist:
                # arguments for worker function: dataset and dataargs       
                args.append( ('WRF', mode, griddef, dict(experiment=experiment, filetypes=[filetype], domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(loverwrite=loverwrite, varlist=varlist)
          
  ## call parallel execution function
  ec = asyncPoolEC(performRegridding, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(np.ceil(10*ec/len(args))))