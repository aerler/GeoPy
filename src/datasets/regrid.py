'''
Created on 2013-09-23

A script to reproject and resample datasets in this package onto a given grid.

@author: Andre R. Erler, GPL v3
'''

# external imports
from importlib import import_module
import os # check if files are present
import multiprocessing # parallelization
import logging # used to control error output of sub-processes
from geodata.multiprocess import asyncPoolEC
from datetime import datetime
# internal imports
from geodata.process import CentralProcessingUnit
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import GDALError, GridDefinition
from datasets.common import addLengthAndNamesOfMonth, getFileName, getCommonGrid
# WRF specific
from datasets.WRF import fileclasses, getWRFgrid, loadWRF
from plotting.ARB_settings import WRFname


def performRegridding(dataset, args, kwargs, griddef, lparallel=False, logger=None):
  ''' worker function to perform regridding for a given dataset and target grid '''
  # input checking
  if not isinstance(dataset,basestring): raise TypeError
  if not isinstance(args,list): raise TypeError
  if not isinstance(kwargs,dict): raise TypeError
  if not isinstance(griddef,GridDefinition): raise TypeError
  if not isinstance(lparallel,bool): raise TypeError

  # logging
  if logger is None: 
    logger = logging.getLogger() # new logger
    logger.addHandler(logging.StreamHandler())
  else: logger = logging.getLogger(name=logger) # connect to existing one 

  # parallelism
  if lparallel:
    pid = int(multiprocessing.current_process().name.split('-')[-1]) # start at 1
    pidstr = '[proc%02i]'%pid # pid for parallel mode output
    #multiprocessing.current_process().name = 'proc%02i'%pid; pidstr = ''
    # N.B.: the string log formatting is basically done manually here, 
    #       so that I have better control over line spacing  
  else:
    pidstr = '' # don't print process ID, sicne there is only one

  # handle errors below (don't necessarily crash)
  try:
    
    # import dataset routines
    ds = import_module(dataset)    

    periodstr = '{0:4d}-{1:4d}'.format(begindate,enddate)
    print('\n')
    print('   ***   Processing Dataset %s from %s   ***   '%(dataset, periodstr))
    print('             (regridding from %s to %s)   '%(dataset,grid))
    print('\n')        
    
    # load source
    if dataset == dataset.lower(): # WRF dataset      
      source = loadWRF(*args,**kwargs)
    elif grid == grid.upper(): # observations
      griddef = import_module(grid[0:4]).__dict__[grid+'_grid']
    else: 
      # we could try CESM grids here, at a later stage
      raise GDALError, 'No valid grid defined! (grid={0:s})'.format(grid)
    print griddef  

    source = ds.loadClimatology(*args, **kwargs) # load pre-processed climatology
    # source.load() # not really necessary
    print(source)
    print('\n')
          
    # prepare target dataset
    filename = getFileName(grid=grid, period=period, name=dataset, filepattern=ds.avgfile)
    if os.path.exists(ds.avgfolder+filename): os.remove(ds.avgfolder+filename)
    atts=source.atts
    atts['period']=periodstr; atts['name'] = dataset, atts['title'] = '%s Climatology'%dataset 
    sink = DatasetNetCDF(folder=ds.avgfolder, filelist=[filename], atts=atts, mode='w')
    
    # initialize processing
    CPU = CentralProcessingUnit(source, sink, tmp=True)

#     if period is not None and dataset != 'PRISM':
#       # determine averaging interval
#       offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
#       # start processing climatology
#       CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
    
    # perform regridding (if target grid is different from native grid!)
    if griddef.name != dataset:
      # reproject and resample (regrid) dataset
      CPU.Regrid(griddef=griddef, flush=False)

    # get results
    CPU.sync(flush=True)
      
    if 'convertPrecip' in ds.__dict__:
      # convert precip data to SI units (mm/s) 
      ds.__dict__['convertPrecip'](sink.precip) # convert in-place
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
      logger.info('\n'+sink+'\n')   
  
      # return exit code
    return 0 # everything OK
  except Exception: # , err
    # an error occurred
    logging.exception(pidstr) # print stack trace of last exception and current process ID 
    return 1 # indicate failure


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
    #ldebug = False
    NP = NP or 2
    #loverwrite = True
    varlist = ['precip',]
    periods = [(1979,1981)]
    datasets = ['CFSR']
    # WRF
    experiments = ['max'] # WRF experiment names (passed through WRFname)
    domains = [1,2] # domains to be processed
    filetypes = ['srfc',] # filetypes to be processed
    #filetypes = ['srfc','xtrm','plev3d','hydro',] # filetypes to be processed
  else:
    NP = NP or 4
    #loverwrite = False
    varlist = None # process all variables
    periods = [5,10] # climatology periods to process 
    datasets = ['NARR','CFSR','GPCC','CRU','PRISM'] # datasets to process
    # WRF
    experiments = [] # process all WRF experiments
    #experiments = ['max','gulf','new','noah'] # WRF experiment names (passed through WRFname) 
    domains = [1,2] # domains to be processed
    filetypes = fileclasses.keys() # process all filetypes 
    
  # grid to project onto
  grid = 'NARR'  
      
  # expand experiments 
  if len(experiments) > 0: experiments = [WRFname[exp] for exp in experiments]
  else: experiments = [exp for exp in WRFname.values()]    

  # load target grid definition
  griddef = getCommonGrid(grid) # try this first (common grids)
  # else, determine new grid from existing dataset
  if griddef is None:
    if grid == grid.lower(): # WRF grid      
      griddef = getWRFgrid(experiment=grid, domains=[1])
    elif grid == grid.upper(): # observations
      griddef = import_module(grid[0:4]).__dict__[grid+'_grid']
    else: 
      # we could try CESM grids here, at a later stage
      raise GDALError, 'No valid grid defined! (grid={0:s})'.format(grid)
  print griddef  
    
  # assemble job packages
  args = []
  for dataset in datasets:
    for period in periods:
      # make package (tuple,dict)
      tmp = ( (dataset,), dict(period=period) ) # arguments for worker function      
      args.append(tmp) # append to list
               
  # print an announcement
  print('\n Regridding WRF Datasets:\n')
  print(experiments)
  print('\n And Observational Datasets:\n')
  print(datasets)
  print('\n To {0:s} Grid.\n'.format(grid))
  print('\nOVERWRITE: {0:s}\n'.format(loverwrite))
  
  # static keyword arguments
  kwargs = dict(griddef=None, loverwrite=loverwrite)        
  # call parallel execution function
  asyncPoolEC(performRegridding, args, kwargs, NP=NP, ldebug=ldebug)
