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
from geodata.misc import DateError, DatasetError, printList
from geodata.netcdf import DatasetNetCDF
from geodata.base import Dataset
from geodata.gdal import GDALError, GridDefinition, addGeoLocator
from datasets import gridded_datasets
from datasets.common import addLengthAndNamesOfMonth, getCommonGrid
from processing.multiprocess import asyncPoolEC
from processing.process import CentralProcessingUnit
from processing.misc import getMetaData, getTargetFile, getExperimentList, loadYAML


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performRegridding(dataset, mode, griddef, dataargs, loverwrite=False, varlist=None, lwrite=True, 
                      lreturn=False, ldebug=False, lparallel=False, pidstr='', logger=None):
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

  ## extract meta data from arguments
  dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs)
  dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr; avgfolder = dataargs.avgfolder

  # get filename for target dataset and do some checks
  filename = getTargetFile(dataset=dataset, mode=mode, dataargs=dataargs, lwrite=lwrite, grid=griddef.name.lower(),) 
    
  # prepare target dataset
  if ldebug: filename = 'test_' + filename
  if not os.path.exists(avgfolder): raise IOError, "Dataset folder '{:s}' does not exist!".format(avgfolder)
  lskip = False # else just go ahead
  if lwrite:
    if lreturn: tmpfilename = filename # no temporary file if dataset is passed on (can't rename the file while it is open!)
    else: 
      if lparallel: tmppfx = 'tmp_regrid_{:s}_'.format(pidstr[1:-1])
      else: tmppfx = 'tmp_regrid_'.format(pidstr[1:-1])
      tmpfilename = tmppfx + filename      
    filepath = avgfolder + filename
    tmpfilepath = avgfolder + tmpfilename
    if os.path.exists(filepath): 
      if not loverwrite: 
        age = datetime.fromtimestamp(os.path.getmtime(filepath))
        # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
        if age > srcage and os.path.getsize(filepath) > 1e6: 
          lskip = True
          if hasattr(griddef, 'filepath') and griddef.filepath is not None:
            gridage = datetime.fromtimestamp(os.path.getmtime(griddef.filepath))
            if age < gridage: lskip = False
        # N.B.: NetCDF files smaller than 1MB are usually incomplete header fragments from a previous crashed
  
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
    if mode == 'climatology': opmsgstr = 'Regridding Climatology ({:s}) to {:s} Grid'.format(periodstr, griddef.name)
    elif mode == 'time-series': opmsgstr = 'Regridding Time-series to {:s} Grid'.format(griddef.name)
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)        
    # print feedback to logger
    logger.info('\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr))
    if not lparallel and ldebug: logger.info('\n'+str(source)+'\n')
    
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
        if os.path.exists(filepath): os.remove(filepath) # remove old file
        os.rename(tmpfilepath,filepath) # this would also overwrite the old file...
      # N.B.: there is no temporary file if the dataset is returned, because an open file can't be renamed
        
    # clean up and return
    source.unload(); del source, CPU
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
    config = loadYAML('regrid.yaml', lfeedback=True)
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
    # target data specs
    grids = config['grids']
  else:
    # settings for testing and debugging
#     NP = 1 ; ldebug = True # for quick computations
    NP = 3 ; ldebug = False # just for tests
#     modes = ('climatology','time-series') # 'climatology','time-series'
#     modes = ('climatology',) # 'climatology','time-series'
    modes = ('time-series',) # 'climatology','time-series'
    loverwrite = True
    varlist = None
#     varlist = ['LU_INDEX',]
    periods = []
#     periods += [1]
#     periods += [3]
#     periods += [5]
#     periods += [10]
    periods += [15]
#     periods += [30]
    # Observations/Reanalysis
    resolutions = {'CRU':'','GPCC':['025','05','10','25'],'NARR':'','CFSR':['05','031'],'NRCan':'NA12'}; unity_grid = 'arb2_d02'
    datasets = []
    lLTM = True # also regrid the long-term mean climatologies 
#     datasets += ['NRCan']; lLTM = False; periods = [(1970,2000),(1980,2010)] # NRCan normals period
#     resolutions = {'NRCan': ['na12_ephemeral','na12_maritime','na12_prairies'][2:]}
#     datasets += ['PRISM','GPCC','PCIC']; #periods = None
#     datasets += ['CFSR', ] # CFSR_05 does not have precip
#     datasets += ['GPCC']; resolutions = {'GPCC':['025','05']}
#     datasets += ['GPCC']; # resolutions = {'GPCC':['05']}
#     datasets += ['CRU']
    # CESM experiments (short or long name) 
    CESM_project = None # all available experiments
    load3D = False
    CESM_experiments = [] # use None to process all CESM experiments
#     CESM_experiments += ['Ens','Ens-2050','Ens-2100']
#     CESM_experiments += ['CESM','CESM-2050']
#     CESM_experiments += ['Ctrl', 'Ens-A', 'Ens-B', 'Ens-C']
#     CESM_experiments += ['Ctrl-2050', 'Ens-A-2050', 'Ens-B-2050', 'Ens-C-2050']
#     CESM_filetypes = ['atm','lnd']
    CESM_filetypes = ['atm']
    # WRF experiments (short or long name)
    WRF_project = 'GreatLakes' # only GreatLakes experiments
#     WRF_project = 'WesternCanada' # only WesternCanada experiments
#     WRF_experiments = None # use None to process all WRF experiments
    WRF_experiments = []
#     WRF_experiments += ['g-ensemble','g-ensemble-2050','g-ensemble-2100']
#     WRF_experiments += ['t-ensemble','t-ensemble-2050','t-ensemble-2100']
#     WRF_experiments += ['g3-ensemble','g3-ensemble-2050','g3-ensemble-2100',]
#     WRF_experiments += ['t3-ensemble','t3-ensemble-2050','t3-ensemble-2100']
#     WRF_experiments += ['erai-g','erai-t']
#     WRF_experiments += ['erai-g3','erai-t3']
#     WRF_experiments += ['t3-ensemble-2100','g3-ensemble-2100']
#     WRF_experiments += ['g-ctrl',     'g-ens-A',     'g-ens-B',     'g-ens-C',]
#     WRF_experiments += ['g-ctrl-2050','g-ens-A-2050','g-ens-B-2050','g-ens-C-2050',]
#     WRF_experiments += ['g-ctrl-2100','g-ens-A-2100','g-ens-B-2100','g-ens-C-2100',]
#     WRF_experiments += ['t-ctrl',     't-ens-A',     't-ens-B',     't-ens-C',]
#     WRF_experiments += ['t-ctrl-2050','t-ens-A-2050','t-ens-B-2050','t-ens-C-2050',]
#     WRF_experiments += ['t-ctrl-2100','t-ens-A-2100','t-ens-B-2100','t-ens-C-2100',]
#     WRF_experiments += ['g-ensemble','t-ensemble']
    WRF_experiments += ['g-ensemble']
#     WRF_experiments += ['new-v361-ctrl', 'new-v361-ctrl-2050', 'new-v361-ctrl-2100']
#     WRF_experiments += ['erai-v361-noah', 'new-v361-ctrl', 'new-v36-clm',]
#     WRF_experiments += ['erai-wc2-bugaboo','erai-wc2-rocks']
#     WRF_experiments += ['max-ens-2050','max-ens-2100']
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
    domains = 1 # domains to be processed
#     domains = None # process all domains
    WRF_filetypes = ('hydro','xtrm','srfc','lsm','rad',) # filetypes to be processed
#     WRF_filetypes = ('hydro',) # filetypes to be processed
#     WRF_filetypes = ('srfc','xtrm','plev3d','hydro','lsm') # filetypes to be processed # ,'rad'
    WRF_filetypes = ('const',); periods = None
    # grid to project onto
    grids = dict()
    grids['asb1'] = None # small grid for Assiniboine river basin, 5km
    grids['brd1'] = None # small grid for Assiniboine subbasin, 5km
    grids['grw1'] = None # high-res grid for GRW, 1km
    grids['grw2'] = None # small grid for GRW, 5km
    grids['snw1'] = None # large grid for whole Canada
    grids['can1'] = None # large grid for whole Canada
#     grids['wc2'] = ('d02','d01') # new Brian's Columbia domain (Western Canada 2)
#     grids['glb1'] = ('d01','d02',) # Marc's/Jon's standard Great Lakes domain
# #     grids['glb1'] = ('d02',) # Marc's standard GLB inner domain
#     grids['glb1-90km'] = ('d01',) # 90km GLB domain
#     grids['arb2'] = ('d01','d02') # WRF standard ARB inner domain
#     grids['arb3'] = ('d01','d02','d03',) # WRF standard ARB inner domain
# #     grids['arb3'] = ('d03',) # WRF standard ARB inner domain
# #     grids['ARB_small'] = ('025','05') # small custom geographic grids
# #     grids['ARB_large'] = ('025','05') # large custom geographic grids
#     grids['cesm1x1'] = None # CESM 1-deg. grid
#     grids['NARR'] = None # NARR grid
#     grids['CRU'] = None # CRU grid
#     grids['GPCC'] = ('025',) # GPCC LTM grid
#     grids['PRISM'] = None # larger PRISM grid
#     grids['PCIC'] = None # 1km PCIC PRISM grid
  
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
    print('   {0:s} {1:s}'.format(grid,printList(reses) if reses else ''))
  print('\nOVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
    
  ## construct argument list
  args = []  # list of job packages
  # loop over modes
  for mode in modes:
    # only climatology mode has periods    
    if mode == 'climatology': 
      periodlist = periods if isinstance(periods, (tuple,list)) else (periods,)
    elif mode == 'time-series': 
      periodlist = (None,) # ignore periods
    else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)

    # loop over target grids ...
    for grid,reses in grids.iteritems():
      # ... and resolutions
      if reses is None: reses = (None,)
      for res in reses:
        
        # load target grid definition
        griddef = getCommonGrid(grid, res=res, lfilepath=True)
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
                args.append( (dataset, mode, griddef, dict(varlist=varlist, period=None, resolution=dsres, unity_grid=unity_grid)) ) # append to list
            # climatologies derived from time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              for period in periodlist:
                args.append( (dataset, mode, griddef, dict(varlist=varlist, period=period, resolution=dsres, unity_grid=unity_grid)) ) # append to list            
          elif mode == 'time-series': 
            # regrid the entire time-series
            if resolutions is None: dsreses = mod.TS_grids
            elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
            for dsres in dsreses:
              args.append( (dataset, mode, griddef, dict(varlist=varlist, period=None, resolution=dsres, unity_grid=unity_grid)) ) # append to list            
        
        # CESM datasets
        for experiment in CESM_experiments:
          for filetype in CESM_filetypes:
            for period in periodlist:
              # arguments for worker function: dataset and dataargs       
              args.append( ('CESM', mode, griddef, dict(experiment=experiment, varlist=varlist, filetypes=[filetype], 
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
                args.append( ('WRF', mode, griddef, dict(experiment=experiment, varlist=varlist, filetypes=[filetype], 
                                                         domain=domain, period=period)) )
      
  # static keyword arguments
  kwargs = dict(loverwrite=loverwrite, varlist=varlist)
  
  ## call parallel execution function
  ec = asyncPoolEC(performRegridding, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
  # exit with fraction of failures (out of 10) as exit code
  exit(int(10+int(10.*ec/len(args))) if ec > 0 else 0)
