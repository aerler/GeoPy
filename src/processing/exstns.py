'''
Created on 2014-09-03

A script to extract point/station time-series from gridded monthly datasets.
Adapted from regrid.py.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os # check if files are present
import numpy as np
from importlib import import_module
from datetime import datetime
import logging   
import functools  
# internal imports
from geodata.misc import DateError, printList
from geodata.netcdf import DatasetNetCDF
from geodata.base import Dataset
from datasets import gridded_datasets
from processing.multiprocess import asyncPoolEC
from processing.process import CentralProcessingUnit
from processing.misc import getMetaData, getTargetFile, getExperimentList, loadYAML


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performExtraction(dataset, mode, stnfct, dataargs, loverwrite=False, varlist=None, lwrite=True, lreturn=False,
                      ldebug=False, lparallel=False, pidstr='', logger=None):
  ''' worker function to extract point data from gridded dataset '''  
  # input checking
  if not isinstance(dataset,basestring): raise TypeError
  if not isinstance(dataargs,dict): raise TypeError # all dataset arguments are kwargs 
  if not callable(stnfct): raise TypeError # function to load station dataset
  if lparallel: 
    if not lwrite: raise IOError, 'In parallel mode we can only write to disk (i.e. lwrite = True).'
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

  lclim = False; lts = False
  if mode == 'climatology': lclim = True
  elif mode == 'time-series': lts = True
  else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)
  
  ## extract meta data from arguments
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs)
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr; avgfolder = dataargs.avgfolder

  # load template dataset
  stndata = stnfct() # load station dataset from function
  if not isinstance(stndata, Dataset): raise TypeError
  # N.B.: the loading function is necessary, because DataseNetCDF instances do not pickle well 
            
  # get filename for target dataset and do some checks
  filename = getTargetFile(dataset=dataset, mode=mode, dataargs=dataargs, lwrite=lwrite, 
                           grid=stndata.name, period=None, filetype=None) 
  
  if ldebug: filename = 'test_' + filename
  if not os.path.exists(avgfolder): raise IOError, "Dataset folder '{:s}' does not exist!".format(avgfolder)
  lskip = False # else just go ahead
  if lwrite:
    if lreturn: 
      tmpfilename = filename # no temporary file if dataset is passed on (can't rename the file while it is open!)
    else: 
      if lparallel: tmppfx = 'tmp_exstns_{:s}_'.format(pidstr[1:-1])
      else: tmppfx = 'tmp_exstns_'.format(pidstr[1:-1])
      tmpfilename = tmppfx + filename      
    filepath = avgfolder + filename
    tmpfilepath = avgfolder + tmpfilename
    if os.path.exists(filepath): 
      if not loverwrite: 
        age = datetime.fromtimestamp(os.path.getmtime(filepath))
        # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
        if age > srcage and os.path.getsize(filepath) > 1e5: lskip = True
        # N.B.: NetCDF files smaller than 100kB are usually incomplete header fragments from a previous crashed
      if not lskip: os.remove(filepath) # recompute
  
  # depending on last modification time of file or overwrite setting, start computation, or skip
  if lskip:        
    # print message
    skipmsg =  "\n{:s}   >>>   Skipping: file '{:s}' in dataset '{:s}' already exists and is newer than source file.".format(pidstr,filename,dataset_name)
    skipmsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,filepath)
    logger.info(skipmsg)              
  else:
          
    ## actually load datasets
    source = loadfct() # load source 
    # check period
    if 'period' in source.atts and dataargs.periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)
  
    # print message
    if lclim: opmsgstr = "Extracting '{:s}'-type Point Data from Climatology ({:s})".format(stndata.name, periodstr)
    elif lts: opmsgstr = "Extracting '{:s}'-type Point Data from Time-series".format(stndata.name)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)
    # print feedback to logger
    logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
    if not lparallel and ldebug: logger.info('\n'+str(source)+'\n')  
    
    ## create new sink/target file
    # set attributes   
    atts=source.atts.copy()
    atts['period'] = dataargs.periodstr if dataargs.periodstr else 'time-series' 
    atts['name'] = dataset_name; atts['station'] = stndata.name
    atts['title'] = '{:s} (Stations) from {:s} {:s}'.format(stndata.title,dataset_name,mode.title())
    # make new dataset
    if lwrite: # write to NetCDF file 
      if os.path.exists(tmpfilepath): os.remove(tmpfilepath) # remove old temp files 
      sink = DatasetNetCDF(folder=avgfolder, filelist=[tmpfilename], atts=atts, mode='w')
    else: sink = Dataset(atts=atts) # ony create dataset in memory
    
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=False, feedback=ldebug)
  
    # extract data at station locations
    CPU.Extract(template=stndata, flush=True)
    # get results    
    CPU.sync(flush=True)
    
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
        if os.path.exists(filepath): os.remove(filepath) # remove old file
        os.rename(tmpfilepath,filepath)
      # N.B.: there is no temporary file if the dataset is returned, because an open file can't be renamed
        
    # clean up and return
    source.unload(); del source#, CPU
    if lreturn:      
      return sink # return dataset for further use (netcdf file still open!)
    else:            
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
    config = loadYAML('exstns.yaml', lfeedback=True)
    # read config object
    NP = NP or config['NP']
    loverwrite = config['loverwrite']
    # source data specs
    modes = config['modes']
    varlist = config['varlist']
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
    # target data specs
    stations = config['stations']
  else:
    NP = 2 ; ldebug = False # for quick computations
    modes = ('time-series',) # 'climatology','time-series'
    loverwrite = True
    varlist = None
    periods = []
#     periods += [1]
#     periods += [3]
#     periods += [5]
#     periods += [10]
    periods += [15]
    # Observations/Reanalysis
    datasets = []; resolutions = None
    lLTM = True # also regrid the long-term mean climatologies 
#     datasets += ['PRISM','GPCC']; periods = None
#     datasets += ['PCIC']; periods = None
#     datasets += ['CFSR']; resolutions = {'CFSR':'031'}
#     datasets += ['NARR']
#     datasets += ['GPCC']; resolutions = {'GPCC':['025','05','10','25']}
#     datasets += ['CRU']
#     datasets += ['Unity']    
    # CESM experiments (short or long name) 
    CESM_project = None # all available experiments
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['Ctrl-1']
#     CESM_experiments += ['Ctrl-1', 'Ctrl-A', 'Ctrl-B', 'Ctrl-C']
    CESM_filetypes = ['atm'] # ,'lnd'
    # WRF experiments (short or long name)
    WRF_project = 'GreatLakes' # only WesternCanada experiments
    WRF_experiments = [] # use None to process all CESM experiments
#     WRF_experiments += ['marc-g','marc-gg','marc-g-2050','marc-gg-2050']
#     WRF_experiments += ['marc-m','marc-mm', 'marc-t','marc-m-2050','marc-mm-2050', 'marc-t-2050']
#     WRF_experiments += ['erai-g','erai-t']
    WRF_experiments += ['g-ctrl','g-ens-A','g-ens-B','g-ens-C',]
    WRF_experiments += ['g-ctrl-2050','g-ens-A-2050','g-ens-B-2050','g-ens-C-2050',]
    WRF_experiments += ['g-ctrl-2100','g-ens-A-2100','g-ens-B-2100','g-ens-C-2100',]
#     WRF_experiments += ['ctrl-1','ctrl-2050','ctrl-2100',]
#     WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',][:1]
#     WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]
#     WRF_experiments += ['max-ctrl-2100','max-ens-A-2100','max-ens-B-2100','max-ens-C-2100',]        
    # other WRF parameters 
    domains = None # domains to be processed
#     WRF_filetypes = ('hydro','xtrm','srfc','lsm') # filetypes to be processed
    WRF_filetypes = ('aux',)
#     WRF_filetypes = ('const',); periods = None
    # station datasets to match    
    stations = dict(EC=('precip',)) # currently there is only one type: the EC weather stations
  
  
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
    print('\n Extracting from WRF Datasets:')
    print([exp.name for exp in WRF_experiments])
  if len(CESM_experiments) > 0:
    print('\n Extracting from CESM Datasets:')
    print([exp.name for exp in CESM_experiments])
  if len(datasets) > 0:
    print('\n Extracting from Observational Datasets:')
    print(datasets)
  print('\n According to Station Datasets:')
  for stntype,datatypes in stations.iteritems():
    print('   {0:s} {1:s}'.format(stntype,printList(datatypes)))
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
    
  ## construct argument list
  args = []  # list of job packages (commands)
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode == 'climatology': periodlist = periods
    elif mode == 'time-series': periodlist = (None,)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)

    # loop over station dataset types ...
    for stntype,datatypes in stations.iteritems():
      # ... and their filetypes
      for datatype in datatypes:
        
        # assemble function to load station data (with arguments)
        station_module = import_module('datasets.{0:s}'.format(stntype)) # load station data module
        # load station dataset into memory, so that it can be shared by all workers
        stnname = stntype.lower()+datatype.lower() 
        stnfct = functools.partial(station_module.loadStationTimeSeries, name=stnname, filetype=datatype)
               
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
                args.append( (dataset, mode, stnfct, dict(varlist=varlist, period=None, resolution=dsres)) ) # append to list
            # climatologies derived from time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              for period in periodlist:
                args.append( (dataset, mode, stnfct, dict(varlist=varlist, period=period, resolution=dsres)) ) # append to list            
          elif mode == 'time-series': 
            # regrid the entire time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              args.append( (dataset, mode, stnfct, dict(varlist=varlist, period=None, resolution=dsres)) ) # append to list            
        
        # CESM datasets
        for experiment in CESM_experiments:
          for filetype in CESM_filetypes:
            for period in periodlist:
              # arguments for worker function: dataset and dataargs       
              args.append( ('CESM', mode, stnfct, dict(experiment=experiment, varlist=varlist, filetypes=[filetype], 
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
                args.append( ('WRF', mode, stnfct, dict(experiment=experiment, varlist=varlist, filetypes=[filetype], 
                                                        domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(loverwrite=loverwrite, varlist=varlist)
          
  ## call parallel execution function
  ec = asyncPoolEC(performExtraction, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+int(10.*ec/len(args))) if ec > 0 else 0)
