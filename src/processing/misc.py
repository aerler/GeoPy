'''
Created on Feb 11, 2015

Some utility functions related to processing datasets (to avoid code repetition)

@author: Andre R. Erler, GPL v3
'''

# external imports
import os
import numpy as np
from importlib import import_module
import functools
import yaml,os
# internal imports
from geodata.misc import DatasetError, DateError, isInt
from utils.misc import namedTuple
from datasets.common import getFileName
# WRF specific
from datasets.WRF import loadWRF, loadWRF_TS
# CESM specific
from datasets.CESM import loadCESM, loadCESM_TS


# load YAML configuration file
def loadYAML(default, lfeedback=True):
  ''' load YAML configuration file and return config object '''
  # check for environment variable
  if os.environ.has_key('PYAVG_YAML'): 
    yamlfile = os.environ['PYAVG_YAML']
    # try to guess variations, if path is not valid
    if not os.path.exists(yamlfile): 
      if os.path.isdir(yamlfile): # use default filename in directory
        yamlfile = '{:s}/{:s}'.format(yamlfile,default)
      else: # use filename in current directory
        yamlfile = '{:s}/{:s}'.format(os.getcwd(),yamlfile)
    if lfeedback: print("\nLoading user-specified YAML configuration file:\n '{:s}'".format(yamlfile))
  else: # use default in current directory
    yamlfile = '{:s}/{:s}'.format(os.getcwd(),default)
    if lfeedback: print("\nLoading default YAML configuration file:\n '{:s}'".format(yamlfile))
  # check if file exists  
  if not os.path.exists(yamlfile): 
    raise IOError, "YAML configuration file not found!\n ('{:s}')".format(yamlfile)
  # read file
  with open(yamlfile) as f: 
    config = yaml.load(f, Loader=yaml.Loader)
  # return config object
  return config


## convenience function to load WRF experiment list and replace names with Exp objects
def getExperimentList(experiments, project, dataset):
  ''' load WRF or CESM experiment list and replace names with Exp objects '''
  # load WRF experiments list
  project = 'projects' if not project else 'projects.{:s}'.format(project)
  mod = import_module('{:s}.{:s}_experiments'.format(project,dataset))
  exps, enss = mod.exps, mod.enss; del mod
  # expand WRF experiments
  if experiments is None: # do all (except ensembles)
    experiments = [exp for exp in exps.itervalues() if exp.shortname not in enss] 
  else: 
    try: experiments = [exps[exp] for exp in experiments]
    except KeyError: # throw exception is experiment is not found
      raise KeyError, "{1:s} experiment '{0:s}' not found in {1:s} experiment list (loaded from '{2:s}').".format(exp,dataset,project)
  # return expanded list of experiments
  return experiments

## prepare target dataset
def getTargetFile(name, dataset, mode, module, dataargs, lwrite):
  ''' generate filename for target dataset '''
  # extract some variables
  periodstr = dataargs.periodstr; filetype = dataargs.filetype; domain = dataargs.domain
  sstr = '_{}'.format(name) # use name as "grid" designation for station data
  pstr = '_{}'.format(periodstr) if periodstr else ''
  # figure out filename
  if dataset == 'WRF' and lwrite:
    if mode == 'climatology': filename = module.clim_file_pattern.format(filetype,domain,sstr,pstr)
    elif mode == 'time-series': filename = module.ts_file_pattern.format(filetype,domain,sstr)
    else: raise NotImplementedError
  elif dataset == 'CESM' and lwrite:
    if mode == 'climatology': filename = module.clim_file_pattern.format(filetype,sstr,pstr)
    elif mode == 'time-series': filename = module.ts_file_pattern.format(filetype,sstr)
    else: raise NotImplementedError
  elif ( dataset == dataset.upper() or dataset == 'Unity' ) and lwrite: # observational datasets
    filename = getFileName(grid=name, period=dataargs.period, name=dataargs.obs_res, filetype=mode)      
  elif not lwrite: raise DatasetError
  if not os.path.exists(dataargs.avgfolder): 
    raise IOError, "Dataset folder '{:s}' does not exist!".format(dataargs.avgfolder)
  # return filename
  return filename


## determine dataset metadata
def getMetaData(dataset, mode, dataargs):
  ''' determine dataset type and meta data, as well as path to main source file '''
  # determine dataset mode
  lclim = False; lts = False
  if mode == 'climatology': lclim = True
  elif mode == 'time-series': lts = True
  else: raise NotImplementedError, "Unrecognized Mode: '{:s}'".format(mode)
  # defaults for specific variables
  obs_res = None; domain = None; filetype = None
  # determine meta data based on dataset type
  if dataset == 'WRF': 
    # WRF datasets
    module = import_module('datasets.WRF')
    exp = dataargs['experiment']    
    dataset_name = exp.name
    domain = dataargs['domain']
    grid = dataargs.get('grid',None)
    # figure out period
    period = dataargs['period']
    if period is None: pass
    elif isinstance(period,(int,np.integer)):
      beginyear = int(exp.begindate[0:4])
      period = (beginyear, beginyear+period)
    elif len(period) != 2 and all(isInt(period)): raise DateError
    if period is None: periodstr = '' 
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
    gridstr = grid if grid is not None else ''      
    # identify file and domain
    if len(dataargs['filetypes']) > 1: raise DatasetError # process only one file at a time
    filetype = dataargs['filetypes'][0]
    if isinstance(domain,(list,tuple)): domain = domain[0]
    if not isinstance(domain, (np.integer,int)): raise DatasetError    
    datamsgstr = "Processing WRF '{:s}'-file from Experiment '{:s}' (d{:02d})".format(filetype, dataset_name, domain)
    # assemble filename to check modification dates (should be only one file)    
    fileclass = module.fileclasses[filetype] # avoid WRF & CESM name collision
    pstr = '_'+periodstr if periodstr else ''
    gstr = '_'+gridstr if gridstr else ''
    if lclim: filename = fileclass.climfile.format(domain,gstr,pstr) # insert domain number, grid, and period
    elif lts: filename = fileclass.tsfile.format(domain,gstr) # insert domain number, and grid
    avgfolder = exp.avgfolder
    # load source data
    if lclim:
      loadfct = functools.partial(loadWRF, experiment=exp, name=None, domains=domain, grid=None, 
                                  period=period, filetypes=[filetype], varatts=None, lconst=True) # still want topography...
    elif lts:
      loadfct = functools.partial(loadWRF_TS, experiment=exp, name=None, domains=domain, grid=None, 
                                  filetypes=[filetype], varatts=None, lconst=True) # still want topography...
    filepath = '{:s}/{:s}'.format(avgfolder,filename)
  elif dataset == 'CESM': 
    # CESM datasets
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
    # identify file
    if len(dataargs['filetypes']) > 1: raise DatasetError # process only one file at a time
    filetype = dataargs['filetypes'][0]        
    # check period
    if period is None: periodstr = ''
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
    datamsgstr = "Processing CESM '{:s}'-file from Experiment '{:s}'".format(filetype, dataset_name) 
    # assemble filename to check modification dates (should be only one file)    
    fileclass = module.fileclasses[filetype] # avoid WRF & CESM name collision
    pstr = '_'+periodstr if periodstr else ''
    if lclim: filename = fileclass.climfile.format('',pstr) # insert domain number, grid, and period
    elif lts: filename = fileclass.tsfile.format('') # insert domain number, and grid
    avgfolder = exp.avgfolder
    # load source data 
    load3D = dataargs.pop('load3D',None) # if 3D fields should be loaded (default: False)
    if lclim:
      loadfct = functools.partial(loadCESM, experiment=exp, name=None, grid=None, period=period, 
                                  filetypes=[filetype], varatts=None, load3D=load3D, translateVars=None)
    elif lts:
      loadfct = functools.partial(loadCESM_TS, experiment=exp, name=None, grid=None, 
                                  filetypes=[filetype], varatts=None, load3D=load3D, translateVars=None)     
    filepath = '{:s}/{:s}'.format(avgfolder,filename)
  elif dataset == dataset.upper() or dataset == 'Unity':
    # observational datasets
    module = import_module('datasets.{0:s}'.format(dataset))      
    dataset_name = module.dataset_name
    resolution = dataargs['resolution']
    if resolution: obs_res = '{0:s}_{1:s}'.format(dataset_name,resolution)
    else: obs_res = dataset_name   
    # figure out period
    period = dataargs['period']    
    if period is None: pass
    elif isinstance(period,(int,np.integer)):
      period = (1979, 1979+period) # they all begin in 1979
    elif len(period) != 2 and not all(isInt(period)): raise DateError
    datamsgstr = "Processing Dataset '{:s}'".format(dataset_name)
    # check period
    if period is None: 
      if mode == 'climatology': periodstr = 'Long-Term Mean'
      else: periodstr = ''
    else: periodstr = '{0:4d}-{1:4d}'.format(*period)
    # assemble filename to check modification dates (should be only one file)    
    filename = getFileName(grid=None, period=period, name=obs_res, filetype=mode)
    avgfolder = module.avgfolder
    # load pre-processed climatology
    if lclim:
      loadfct = functools.partial(module.loadClimatology, name=dataset_name, period=period, grid=None, 
                                  resolution=resolution, varatts=None, folder=module.avgfolder, filelist=None)
    elif lts:
      loadfct = functools.partial(module.loadTimeSeries, name=dataset_name, grid=None, 
                                  resolution=resolution, varatts=None, folder=None, filelist=None)
    # check if the source file is actually correct
    filepath = '{:s}/{:s}'.format(avgfolder,filename)
    if not os.path.exists(filepath): 
      source = loadfct() # no varlist - obs don't have many variables anyways
      filepath = source.filelist[0]
      # N.B.: it would be nice to print a message, but then we would have to make the logger available,
      #       which would be too much trouble
  else:
    raise DatasetError, "Dataset '{:s}' not found!".format(dataset)
  ## assemble and return meta data
  if not os.path.exists(filepath): raise IOError, "Source file '{:s}' does not exist!".format(filepath)        
  dataargs = namedTuple(dataset_name=dataset_name, period=period, periodstr=periodstr, avgfolder=avgfolder, 
                        filetype=filetype, domain=domain, obs_res=obs_res) 
  # return meta data
  return module, dataargs, loadfct, filepath, datamsgstr
    


if __name__ == '__main__':
    pass