'''
Created on 2014-01-26

A script to extract time-series averaged over the area defined by a collection of shapes (shapefiles).
Adapted from regrid.py.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os # check if files are present
import numpy as np
from importlib import import_module
from datetime import datetime
import logging   
from collections import OrderedDict
# internal imports
from geodata.misc import DateError, DatasetError, printList
from geodata.netcdf import DatasetNetCDF
from geodata.base import Dataset
from datasets import gridded_datasets
from processing.misc import getMetaData, getTargetFile, getExperimentList, loadYAML,\
  getProjectVars
from processing.multiprocess import asyncPoolEC
from processing.process import CentralProcessingUnit


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performShapeAverage(dataset, mode, shape_name, shape_dict, dataargs, loverwrite=False, varlist=None, 
                        lwrite=True, lreturn=False, lappend=False,
                        ldebug=False, lparallel=False, pidstr='', logger=None):
  ''' worker function to extract point data from gridded dataset '''  
  # input checking
  if not isinstance(dataset,basestring): raise TypeError
  if not isinstance(dataargs,dict): raise TypeError # all dataset arguments are kwargs 
  if not isinstance(shape_dict, OrderedDict): raise TypeError
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

  ## extract meta data from arguments
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs)  
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr; avgfolder = dataargs.avgfolder
            
  # get filename for target dataset and do some checks
  filename = getTargetFile(dataset=dataset, mode=mode, dataargs=dataargs, lwrite=lwrite, shape=shape_name) 
    
  if ldebug: filename = 'test_' + filename  
  if not os.path.exists(avgfolder): raise IOError, "Dataset folder '{:s}' does not exist!".format(avgfolder)  
  lskip = False # else just go ahead
  if lwrite:
    if lreturn: 
      tmpfilename = filename # no temporary file if dataset is passed on (can't rename the file while it is open!)
    else: 
      if lparallel: tmppfx = 'tmp_{:s}_{:s}_'.format(shape_name,pidstr[1:-1])
      else: tmppfx = 'tmp_{:s}_'.format(shape_name)
      tmpfilename = tmppfx + filename      
    filepath = avgfolder + filename
    tmpfilepath = avgfolder + tmpfilename
    if os.path.exists(filepath): 
      if not loverwrite: 
        age = datetime.fromtimestamp(os.path.getmtime(filepath))
        # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
        if age > srcage and os.path.getsize(filepath) > 1e4: lskip = True
        # N.B.: NetCDF files smaller than 10kB are usually incomplete header fragments from a previous crashed

  
  # depending on last modification time of file or overwrite setting, start computation, or skip
  if lskip:        
    # print message
    skipmsg =  "\n{:s}   >>>   Skipping: file '{:s}' in dataset '{:s}' already exists and is newer than source file.".format(pidstr,filename,dataset_name)
    skipmsg += "\n{:s}   >>>   ('{:s}')\n".format(pidstr,filepath)
    logger.info(skipmsg)              
  else:
              
    if lappend: raise NotImplementedError
    
    ## actually load datasets
    source = loadfct() # load source 
    # check period
    if 'period' in source.atts and dataargs.periodstr != source.atts.period: # a NetCDF attribute
      raise DateError, "Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period)

    # common message
    if mode == 'climatology': opmsgstr = "Computing Area Averages from Climatology ({:s})".format(periodstr)
    elif mode == 'time-series': opmsgstr = "Computing Area Averages from Time-series"
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)        
    # print feedback to logger
    logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
    if not lparallel and ldebug: logger.info('\n'+str(source)+'\n')
          
    ## create new sink/target file
    # set attributes   
    atts=source.atts.copy()
    atts['period'] = periodstr[1:] if periodstr else 'time-series' 
    atts['name'] = dataset_name; atts['shapes'] = shape_name
    atts['title'] = 'Area Averages from {:s} {:s}'.format(dataset_name,mode.title())
    # make new dataset
    if lwrite: # write to NetCDF file 
      if os.path.exists(tmpfilepath): os.remove(tmpfilepath) # remove old temp files 
      sink = DatasetNetCDF(folder=avgfolder, filelist=[tmpfilename], atts=atts, mode='w')
    else: sink = Dataset(atts=atts) # ony create dataset in memory
    
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=False, feedback=ldebug)
  
    # extract data at station locations
    CPU.ShapeAverage(shape_dict=shape_dict, shape_name=shape_name, flush=True)
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
      if not lreturn and not lappend:
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
  else: ldebug = False # i.e. append
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
    config = loadYAML('shpavg.yaml', lfeedback=True)
    # read config object
    NP = NP or config['NP']
    loverwrite = config['loverwrite']
    lappend = config['lappend']
    # source data specs
    modes = config['modes']
    varlist = config['varlist']
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
    domains = config['domains']
    grid = config.get('grid',None)
    # target data specs
    shape_name = config['shape_name']
    shapes = config['shapes']
  else:
#     NP = 1 ; ldebug = True # for quick computations
    NP = 3 ; ldebug = False # for quick computations
#     modes = ('time-series','climatology')
    modes = ('climatology',) 
    loverwrite = False
    varlist = None # ['T2']
    periods = []
#     periods += [1]
#     periods += [3]
#    periods += [5]
#    periods += [10]
    periods += [15]
#     periods += [30]
    grid = None # just a default value - native grid
    # Observations/Reanalysis
    lLTM = True 
    datasets = []; resolutions = None; unity_grid = None #'arb2_d02'
    resolutions = {'CRU':'','GPCC':['025','05','10','25'],'NARR':'','CFSR':['031','05'],'NRCan':'NA12'}
    datasets = []
    datasets += ['NRCan']; periods = [(1970,2000),(1980,2010)]; lLTM = False
#     datasets += ['PRISM']; periods = None; lLTM = True
#     datasets += ['PCIC','PRISM']; periods = None; lLTM = True
#     datasets += ['CFSR']; resolutions = {'CFSR':['031','05']}
#     datasets += ['Unity']    
    # CESM experiments (short or long name) 
    CESM_project = None # use all experiments in project module
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['Ctrl-1']
    CESM_filetypes = ['atm'] # ,'lnd'
#     CESM_filetypes = ['lnd']
    # WRF experiments (short or long name)
#     WRF_project = None
#     WRF_project = 'WesternCanada' # only use WesternCanada experiments
    WRF_project = 'GreatLakes' # only use GreatLakes experiments
    WRF_experiments = []
#     WRF_experiments += ['erai-t', 'erai-g','erai-t3', 'erai-g3']
#     WRF_experiments += ['g3-ensemble','g3-ensemble-2050','g3-ensemble-2050']
#     WRF_experiments += ['t3-ensemble','t3-ensemble-2050','t3-ensemble-2050']
#     WRF_experiments += ['g-ensemble','g-ensemble-2050','g-ensemble-2100']
#     WRF_experiments += ['t-ensemble','t-ensemble-2050','t-ensemble-2100']
#     WRF_experiments += ['g-ensemble','t-ensemble']
#     WRF_experiments += ['g-ctrl', 'g-ctrl-2050', 'g-ctrl-2100']
#     WRF_experiments += ['g-ctrl','g-ens-A','g-ens-B','g-ens-C',]
#     WRF_experiments += ['g-ctrl-2050','g-ens-A-2050','g-ens-B-2050','g-ens-C-2050',]
#     WRF_experiments += ['g-ctrl-2100','g-ens-A-2100','g-ens-B-2100','g-ens-C-2100',]
#     WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',]
#     WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]
#     WRF_experiments += ['max-ctrl-2100','max-ens-A-2100','max-ens-B-2100','max-ens-C-2100',]
#     WRF_experiments += ['max-ens','max-ens-2050','max-ens-2100'] # requires different implementation...
    # other WRF parameters 
    domains = None # domains to be processed
#     domains = (2,) # domains to be processed
#     WRF_filetypes = ('hydro','lsm','xtrm','rad','srfc')
#     WRF_filetypes = ('srfc','xtrm','plev3d','hydro','lsm') # filetypes to be processed # ,'rad'
#     WRF_filetypes = ('xtrm','lsm') # filetypes to be processed    
#     WRF_filetypes = ('const',); periods = None
#     WRF_filetypes = ('aux','aabc')
    WRF_filetypes = ('aux',)
#     grid = 'grw2' # grid parameter to load datasets
    # define shape data  
    shapes = OrderedDict()
#     shape_name = 'shpavg' # all Canadian shapes
#     shapes['provinces'] = None # Canadian provinces from EC module
#     shapes['basins'] = None # river basins (in Canada) from WSC module
#     shape_name = 'wcavg' # Western Canadian shapes
#     shapes['provinces'] = ['BC','AB'] # Canadian provinces from EC module
#     shapes['basins'] = ['PSB','NorthernPSB','SouthernPSB','FRB','UpperFRB','LowerFRB','CRB',
#                         'ARB','UpperARB','LowerARB','SSR','NRB',] # river basins (in Canada) from WSC module
    shape_name = 'glbshp' # only Canadian river basins
    shapes['provinces'] = ['MB','ON','QC'] # Canadian provinces from EC module
    shapes['basins'] = ['LandGLB','GLB','GRW','UpperGRW','LowerGRW','NorthernGRW','SouthernGRW','WesternGRW'] # river basins (in Canada) from WSC module
#     shape_name = 'glakes' # Great Lakes
#     shapes['great_lakes'] = None # the Great Lakes of North America
     
 
  ## process arguments    
  if isinstance(periods, (np.integer,int)): periods = [periods]
  # check and expand WRF experiment list
  WRF_experiments = getExperimentList(WRF_experiments, WRF_project, 'WRF')
  if isinstance(domains, (np.integer,int)): domains = [domains]
  # check and expand CESM experiment list
  CESM_experiments = getExperimentList(CESM_experiments, CESM_project, 'CESM')
  # expand datasets and resolutions
  if datasets is None: datasets = gridded_datasets
  if unity_grid is None and 'Unity' in datasets:
    if WRF_project: unity_grid = import_module('projects.{:s}'.format(WRF_project)).unity_grid
    else: raise DatasetError("Dataset 'Unity' has no native grid - please set 'unity_grid'.") 
  

  # import shapes from project
  proj_dict = getProjectVars(shapes.keys(), project=WRF_project, module=None)
  # assemble shape dictionary
  shape_dict = OrderedDict()
  shapenames = shapes.keys(); 
  shapenames.sort(); shapenames.reverse() # for backwards compatibility
  for shapename in shapenames:
    proj_shapes = proj_dict[shapename]
    if not isinstance(proj_shapes, dict): raise TypeError(proj_shapes)
    shapelist = shapes[shapename]
    if shapelist is None: 
      shapelist = proj_shapes.keys()
      if not isinstance(proj_shapes, OrderedDict): shapelist.sort() # sort names in-place
      shapes[shapename] = shapelist # update shapes for report
    try:
      for key in shapelist: shape_dict[key] = proj_shapes[key]
    except KeyError: 
      raise KeyError("Name '{:s}' not found in shape dictionary '{:s}'.".format(key,shapename))

    
  # print an announcement
  if len(WRF_experiments) > 0:
    print('\n Averaging from WRF Datasets ({:s}):'.format(WRF_project))
    print([exp.name for exp in WRF_experiments])
  if len(CESM_experiments) > 0:
    print('\n Averaging from CESM Datasets ({:s}):'.format(CESM_project))
    print([exp.name for exp in CESM_experiments])
  if len(datasets) > 0:
    print('\n Averaging from Observational Datasets:')
    print(datasets)
  print('\n Using Shapefiles:')
  for shptype,shplst in shapes.iteritems():
    print('   {0:s} {1:s}'.format(shptype,printList(shplst)))
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))

  
  ## construct argument list
  args = []  # list of job packages (commands)
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode == 'climatology': periodlist = [] if periods is None else periods
    elif mode == 'time-series': periodlist = (None,)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)

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
            args.append( (dataset, mode, shape_name, shape_dict, dict(varlist=varlist, period=None, resolution=dsres, 
                                                                      grid=grid, unity_grid=unity_grid)) ) # append to list
        # climatologies derived from time-series
        if resolutions is None: dsreses = mod.TS_grids
        elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
        for dsres in dsreses:
          for period in periodlist:
            args.append( (dataset, mode, shape_name, shape_dict, dict(varlist=varlist, period=period, resolution=dsres, 
                                                                      grid=grid, unity_grid=unity_grid)) ) # append to list            
      elif mode == 'time-series': 
        # regrid the entire time-series
        if resolutions is None: dsreses = mod.TS_grids
        elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
        for dsres in dsreses:
          args.append( (dataset, mode, shape_name, shape_dict, dict(varlist=varlist, period=None, resolution=dsres, 
                                                                    grid=grid, unity_grid=unity_grid)) ) # append to list            
    
    # CESM datasets
    for experiment in CESM_experiments:
      for filetype in CESM_filetypes:
        for period in periodlist:
          # arguments for worker function: dataset and dataargs       
          args.append( ('CESM', mode, shape_name, shape_dict, dict(experiment=experiment, varlist=varlist, filetypes=[filetype], 
                                                                   grid=grid, period=period, load3D=load3D)) )
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
            args.append( ('WRF', mode, shape_name, shape_dict, dict(experiment=experiment, varlist=varlist, filetypes=[filetype], 
                                                                    grid=grid, domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(loverwrite=loverwrite, varlist=varlist)
          
  ## call parallel execution function
  ec = asyncPoolEC(performShapeAverage, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+int(10.*ec/len(args))) if ec > 0 else 0)
