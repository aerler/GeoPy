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
# internal imports
from geodata.misc import DatasetError
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import GDALError, GridDefinition
from datasets.common import addLengthAndNamesOfMonth, getFileName, getCommonGrid, loadPickledGridDef
from processing.multiprocess import asyncPoolEC
from processing.process import CentralProcessingUnit
# WRF specific
from datasets.WRF import fileclasses, getWRFgrid, loadWRF
from datasets.WRF_experiments import exps


# worker function that is to be passed to asyncPool for parallel execution; use of the decorator is assumed
def performRegridding(dataset, dataargs, griddef=None, loverwrite=False, 
                      lparallel=False, pidstr='', logger=None):
  ''' worker function to perform regridding for a given dataset and target grid '''
  # input checking
  if not isinstance(dataset,basestring): raise TypeError
  if not isinstance(kwargs,dict): raise TypeError # all dataset arguments are kwargs 
  if griddef is not None and not isinstance(griddef,GridDefinition): raise TypeError
  
  #TODO: make better use of Exp class
  
  # load source
  if dataset == 'WRF': 
    module = import_module('datasets.WRF')
    exp = dataargs['experiment']    
    dataset_name = exp.name
    period = dataargs['period']
    #TODO: calculate period boundaries from begindate and period length
    if len(dataargs['filetypes']) > 1: raise DatasetError
    filetype = dataargs['filetypes'][0]
    if not isinstance(dataargs['domains'], (np.integer,int)): raise DatasetError
    domain = dataargs['domains']
    #TODO: resolve arguments transparently
    source = loadWRF(**dataargs) # the name of the experiment is in the dataargs
    periodstr = source.atts.period # a NetCDF attribute    
    datamsgstr = 'Processing WRF Experiment \'{0:s}\' from {1:s}'.format(dataset_name, periodstr) 
  elif dataset == dataset.upper():
    module = import_module('datasets.{0:s}'.format(dataset))      
    dataset_name = module.dataset_name
    source = module.loadClimatology(**dataargs) # load pre-processed climatology
    #TODO: calculate period boundaries from begindate and period length
    periodstr = '{0:4d}-{1:4d}'.format(*dataargs['period'])
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
    filename = module.file_pattern.format(filetype,domain,'_{}'.format(griddef.name),periodstr)
    avgfolder = '{0:s}/{1:s}/'.format(module.avgfolder,dataset_name)    
  elif dataset == dataset.upper(): # observational datasets
    filename = getFileName(grid=griddef.name, period=period, name=dataset_name, filepattern=module.avgfile)
    avgfolder = module.avgfolder
  if ldebug: filename = 'test_' + filename
  if not os.path.exists(avgfolder): raise IOError, 'Dataset folder \'{0:s}\' does not exist!'.format(avgfolder)
  lskip = False # else just go ahead
  if os.path.exists(avgfolder+filename): 
    if not loverwrite: 
      age = datetime.fromtimestamp(os.path.getmtime(avgfolder+filename))
      # if sink file is newer than source file, skip (do not recompute)
      if age > sourceage: lskip = True
      #print sourceage, age
    if not lskip: os.remove(avgfolder+filename) 
  # else: lskip = False (see above)
  
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
    sink = DatasetNetCDF(folder=avgfolder, filelist=[filename], atts=atts, mode='w')
    
    # initialize processing
#     CPU = CentralProcessingUnit(source, sink, varlist=None, tmp=True)
    CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=True)
  
    # perform regridding (if target grid is different from native grid!)
    if griddef.name != dataset:
      # reproject and resample (regrid) dataset
      CPU.Regrid(griddef=griddef, flush=False)

    # get results
    CPU.sync(flush=True)
      
    if 'convertPrecip' in module.__dict__:
      # convert precip data to SI units (mm/s) 
      module.__dict__['convertPrecip'](sink.precip) # convert in-place
#     # add landmask
#     if not sink.hasVariable('landmask'): addLandMask(sink) # create landmask from precip mask
#     linvert = True if dataset == 'CFSR' else False
#     sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=linvert, merge=False) # mask all fields using the new landmask
    # add length and names of month
    if not sink.hasVariable('length_of_month'): addLengthAndNamesOfMonth(sink, noleap=False) 
    
    # close... and write results to file
    logger.info('\n{0:s} Writing to: \'{1:s}\'\n'.format(pidstr,filename))
    sink.sync()
    sink.close()
    # print dataset
    if not lparallel:
      logger.info('\n'+str(sink)+'\n')   


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
  # re-compute everything or just update 
  if os.environ.has_key('PYAVG_OVERWRITE'): 
    loverwrite =  os.environ['PYAVG_OVERWRITE'] == 'OVERWRITE' 
  else: loverwrite = ldebug # False means only update old files
  
  # default settings
  if ldebug:
    ldebug = False
    NP = NP or 4
    #loverwrite = True
    varlist = None # ['',] # None
#     periods = [(1979,1989)]
    periods = [5,10]
    #TODO: use period length instead of boundaries
#     periods = [(1997,1998)]
    datasets = []
    # WRF
    experiments = ['max-ctrl'] # WRF experiment names (passed through WRFname)
#     experiments = ['coast-brian']
    domains = [1,2] # domains to be processed
    filetypes = ['xtrm',] # filetypes to be processed
    #filetypes = ['srfc','xtrm','plev3d','hydro','lsm','rad'] # filetypes to be processed
  else:
    NP = NP or 4
    #loverwrite = False
    varlist = None # process all variables
    periods = [(1979,1984),(1979,1989)] # climatology periods to process 
    datasets = ['NARR','CFSR','GPCC','CRU','PRISM'] # datasets to process
    # WRF
    experiments = [] # process all WRF experiments
    #experiments = ['max','gulf','new','noah'] # WRF experiment names (passed through WRFname) 
    domains = [1,2] # domains to be processed
    filetypes = fileclasses.keys() # process all filetypes 
    
  # grid to project onto
  lpickle = True
  grid = 'ARB_small'  
  res = '05' # qualifier to grid (not all grids)
      
  # expand experiments
  if experiments is None: experiments = [] 
  elif len(experiments) > 0: experiments = [exps[exp] for exp in experiments]
  elif not ldebug: experiments = exps # do all, but not in debug mode!    
   
  # print an announcement
  print('\n Regridding WRF Datasets:')
  print([exp.name for exp in experiments])
  print(' And Observational Datasets:')
  print(datasets)
  print('\n To {0:s} Grid:'.format(grid))
  print('OVERWRITE: {0:s}\n'.format(str(loverwrite)))
  
  # load target grid definition
  if lpickle:
    griddef = loadPickledGridDef(grid, res=res)
  else:
    griddef = getCommonGrid(grid) # try this first (common grids)
    # else, determine new grid from existing dataset
    if griddef is None:
      if grid == grid.lower(): # WRF grid      
        griddef = getWRFgrid(experiment=grid, domains=[1])
      elif grid == grid.upper(): # observations
        griddef = import_module(grid[0:4]).__dict__[grid+'_grid']
      else: pass # we could try CESM grids here, at a later stage
  # check is grid was defined properly
  if not isinstance(griddef,GridDefinition): 
    raise GDALError, 'No valid grid defined! (grid={0:s})'.format(grid)        
  
  # assemble job packages
  args = []
  # observational datasets
  for dataset in datasets:
    for period in periods:
      # arguments for worker function: dataset and dataargs       
      args.append( (dataset, dict(period=period)) ) # append to list               
  # WRF datasets
  for experiment in experiments:
    for filetype in filetypes:
      for domain in domains:
        for period in periods:
          # arguments for worker function: dataset and dataargs       
          args.append( ('WRF', dict(experiment=experiment, filetypes=[filetype], domains=domain, period=period)) )
  # static keyword arguments
  kwargs = dict(griddef=griddef, loverwrite=loverwrite)        
  # call parallel execution function
  asyncPoolEC(performRegridding, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
