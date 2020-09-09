'''
Created on 2016-04-21

A script to convert datasets to raster format using GDAL. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import pickle, re
import os, shutil, gzip # check if files are present etc.
import numpy as np
from importlib import import_module
from datetime import datetime
import logging     
# internal imports
from geodata.base import Dataset, concatDatasets
from geodata.gdal import addGDALtoDataset, addGDALtoVar
from geodata.misc import DateError, DatasetError, printList, ArgumentError, VariableError, GDALError
from datasets import gridded_datasets
from processing.multiprocess import asyncPoolEC
from processing.misc import getMetaData,  getExperimentList, loadYAML, getTargetFile
from utils.nctools import writeNetCDF
# new variable functions and bias-correction 
import processing.newvars as newvars
from processing.bc_methods import findPicklePath

## helper classes to handle different file formats

class FileFormat(object):
  ''' A parent class from which specific format classes will be derived; the purpose is to provide a 
      unified interface for several different file formats. '''
  
  def __init__(self, **expargs):
    ''' take arguments that have been passed from caller and initialize parameters '''
    pass
  
  @property
  def destination(self):
    ''' access output destination '''
    return self.filepath
  
  def defineDataset(self, name=None, period=None, grid=None, ldebug=False, **kwargs):
    ''' a method to set external parameters about the Dataset, so that the export destination
        can be determined (and returned) '''
    self.filepath = None
    return self.filepath

  def prepareDestination(self, srcage=None, loverwrite=False):
    ''' create or clear the destination folder, as necessary, and check if source is newer (for skipping) '''
    pass

  def exportDataset(self, dataset):
    ''' method to write a Dataset instance to disk in the given format; this will be format specific '''
    pass
     
class NetCDF(object):
  ''' A class to handle exports to NetCDF format (v4 by default). '''
  
  def __init__(self, project=None, filetype='aux', folder=None, bc_method=None, **expargs):
    ''' take arguments that have been passed from caller and initialize parameters '''
    if bc_method:
      if not filetype: filetype = bc_method.lower()
      elif not filetype.startswith(bc_method.lower()):
          raise ArgumentError(filetype, bc_method)
    self.bc_method = bc_method
    self.filetype = filetype; self.folder_pattern = folder    
    self.export_arguments = expargs
  
  @property
  def destination(self):
    ''' access output destination '''
    return self.filepath
  
  def defineDataset(self, dataset=None, mode=None, bc_method=None, dataargs=None, lwrite=True, ldebug=False):
    ''' a method to set external parameters about the Dataset, so that the export destination
        can be determined (and returned) '''
    # get filename for target dataset and do some checks
    if self.folder_pattern is None: avgfolder = dataargs.avgfolder # regular source dataset location
    else: self.folder_pattern.format(dataset, self.project, dataargs.dataset_name,) # this could be expanded with dataargs 
    if not os.path.exists(avgfolder): raise IOError("Dataset folder '{:s}' does not exist!".format(avgfolder))
    filename = getTargetFile(dataset=dataset, mode=mode, dataargs=dataargs, lwrite=lwrite, filetype=self.filetype)
    if ldebug: filename = 'test_{:s}'.format(filename)
    self.filepath = '{:s}/{:s}'.format(avgfolder,filename)
    return self.filepath

  def prepareDestination(self, srcage=None, loverwrite=False, lparallel=True):
    ''' create or clear the destination folder, as necessary, and check if source is newer (for skipping) '''
    # prepare target dataset (which is a NetCDF file)
    filepath = self.filepath
    if os.path.exists(filepath):
      if loverwrite:
        os.remove(filepath) # remove old file
        lskip = False # actually do export
      else:
        age = datetime.fromtimestamp(os.path.getmtime(filepath))
        # if source file is newer than sink file or if sink file is a stub, recompute, otherwise skip
        lskip = ( age > srcage ) and os.path.getsize(filepath) > 1e4 # skip if newer than source
    else: lskip = False    
    # return with a decision on skipping
    return lskip 

  def exportDataset(self, dataset):
    ''' method to export a Dataset instance to NetCDF format and write to disk '''
    # create NetCDF file
    filepath = self.filepath
    writeNetCDF(dataset=dataset, ncfile=filepath, **self.export_arguments)
    # check first and last
    if not os.path.exists(filepath): raise IOError(filepath)
   
class ASCII_raster(FileFormat):
  ''' A class to handle exports to ASCII_raster format. '''
  
  def __init__(self, project=None, folder=None, prefix=None, bc_method=None, **expargs):
    ''' take arguments that have been passed from caller and initialize parameters '''
    self.project = project; self.folder_pattern = folder; self.prefix_pattern = prefix; self.bc_method = bc_method
    self.export_arguments = expargs
  
  @property
  def destination(self):
    ''' access output destination '''
    return self.folder
      
  def defineDataset(self, dataset=None, mode=None, dataargs=None, lwrite=True, ldebug=False):
    ''' a method to set exteral parameters about the Dataset, so that the export destination
        can be determined (and returned) '''
    # extract variables
    dataset_name = dataargs.dataset_name; domain = dataargs.domain; grid = dataargs.grid
    if dataargs.period is None: periodstr = None
    elif isinstance(dataargs.period,(tuple,list)):
      periodstr = '{0:02d}'.format(int(dataargs.period[1]-dataargs.period[0]))
    else: periodstr = '{0:02d}'.format(dataargs.period)
    lnkprdstr = dataargs.periodstr
    # assemble specific names
    expname = '{:s}_d{:02d}'.format(dataset_name,domain) if domain else dataset_name
    if mode == 'climatology': 
      expprd = 'clim' if periodstr is None else 'clim_{:s}'.format(periodstr) 
      lnkprd = 'clim' if lnkprdstr is None else 'clim_{:s}'.format(lnkprdstr)
    elif mode == 'time-series': 
      expprd = 'timeseries'; lnkprd = None
    elif mode[-5:] == '-mean': 
      expprd = mode[:-5] if periodstr is None else '{:s}_{:s}'.format(mode[:-5],periodstr)
      lnkprd = mode[:-5] if lnkprdstr is None else '{:s}_{:s}'.format(mode[:-5],lnkprdstr)
    else: raise NotImplementedError("Unrecognized Mode: '{:s}'".format(mode))        
    # insert into patterns 
    metadict = dict(PROJECT=self.project, GRID=grid, EXPERIMENT=expname, PERIOD=expprd, 
                    RESOLUTION=dataargs.resolution, BIAS=self.bc_method)
    self.folder = self.folder_pattern.format(**metadict)
    if ldebug: self.folder = self.folder + '/test/' # test in subfolder
    self.prefix = self.prefix_pattern.format(**metadict) if self.prefix_pattern else None
    # create link with alternate period designation
    self.altprdlnk = None
    if lnkprd is not None and hasattr(os,'symlink'): # Windows does not have symlinks, so this does not work
        # find folder that contains period folder
        folder_names = self.folder_pattern.split('/')
        i = -1; lprd = False
        for name in folder_names:
            i += 1    
            if '{PERIOD' in name:
                lprd = True; break
        if lprd:
            root_folder = '/'.join(folder_names[:i]).format(**metadict)
            period_pattern = folder_names[i] # the folder name containing the period string
            metadict.pop('PERIOD')
            link_name = period_pattern.format(PERIOD=lnkprd,**metadict)
            link_dest = period_pattern.format(PERIOD=expprd,**metadict)
            self.altprdlnk = (root_folder, link_dest, link_name)
    # return folder (no filename)
    return self.folder
  
  def prepareDestination(self, srcage=None, loverwrite=False):
    ''' create or clear the destination folder, as necessary, and check if source is newer (for skipping) '''
    ## prepare target dataset (which is mainly just a folder)
    if not os.path.exists(self.folder): 
      # create new folder
      os.makedirs(self.folder)
      lskip = False # actually do export
    elif loverwrite:
      shutil.rmtree(self.folder) # remove old folder and contents
      os.makedirs(self.folder) # create new folder
      lskip = False # actually do export
    else:
      age = datetime.fromtimestamp(os.path.getmtime(self.folder))
      # if source file is newer than target folder, recompute, otherwise skip
      lskip = ( age > srcage ) # skip if newer than source 
    if not os.path.exists(self.folder): raise IOError(self.folder)
    ## put in alternative symlink (relative path) for period section
    if self.altprdlnk:
      root_folder, link_dest, link_name = self.altprdlnk
      pwd = os.getcwd(); os.chdir(root_folder)
      if not os.path.exists(link_name): 
        os.symlink(link_dest, link_name) # create new symlink
      if os.path.islink(link_name): 
        os.remove(link_name) # remove old link before creating new one
        os.symlink(link_dest, link_name) # create new symlink      
      elif loverwrite:  
        shutil.rmtree(link_name) # remove old folder and contents
        os.symlink(link_dest, link_name) # create new symlink
      os.chdir(pwd) # return to original directory
    # return with a decision on skipping
    return lskip 
    
  def exportDataset(self, dataset):
    ''' method to write a Dataset instance to disk in the given format; this will be format specific '''
    # export dataset to raster format
    filedict = dataset.ASCII_raster(prefix=self.prefix, varlist=None, folder=self.folder, **self.export_arguments)
    # check first and last
    if not os.path.exists(list(filedict.values())[0][0]): raise IOError(list(filedict.values())[0][0]) # random check
    if not os.path.exists(list(filedict.values())[-1][-1]): raise IOError(list(filedict.values())[-1][-1]) # random check

  
def getFileFormat(fileformat, bc_method=None, **expargs):
  ''' function that returns an instance of a specific FileFormat child class specified in expformat; 
      other kwargs are passed on to constructor of FileFormat '''
  # decide based on expformat; instantiate object
  if fileformat == 'ASCII_raster':
    return ASCII_raster(bc_method=bc_method, **expargs)
  elif fileformat.lower() in ('netcdf','netcdf4'):
    return NetCDF(bc_method=bc_method, **expargs)
  else:
    raise NotImplementedError(fileformat)
  

# worker function that is to be passed to asyncPool for parallel execution; use of the TrialNError decorator is assumed
def performExport(dataset, mode, dataargs, expargs, bcargs, loverwrite=False, 
                  ldebug=False, lparallel=False, pidstr='', logger=None):
    ''' worker function to export ASCII rasters for a given dataset '''
    # input checking
    if not isinstance(dataset,str): raise TypeError
    if not isinstance(dataargs,dict): raise TypeError # all dataset arguments are kwargs 
    
    # logging
    if logger is None: # make new logger     
        logger = logging.getLogger() # new logger
        logger.addHandler(logging.StreamHandler())
    else:
        if isinstance(logger,str): 
            logger = logging.getLogger(name=logger) # connect to existing one
        elif not isinstance(logger,logging.Logger): 
            raise TypeError('Expected logger ID/handle in logger KW; got {}'.format(str(logger)))
  
    ## extract meta data from arguments
    dataargs, loadfct, srcage, datamsgstr = getMetaData(dataset, mode, dataargs, lone=False)
    dataset_name = dataargs.dataset_name; periodstr = dataargs.periodstr; domain = dataargs.domain
    
    # figure out bias correction parameters
    if bcargs:
        bcargs = bcargs.copy() # first copy, then modify...
        bc_method = bcargs.pop('method',None)
        if bc_method is None: raise ArgumentError("Need to specify bias-correction method to use bias correction!")
        bc_obs = bcargs.pop('obs_dataset',None)
        if bc_obs is None: raise ArgumentError("Need to specify observational dataset to use bias correction!")
        bc_reference = bcargs.pop('reference',None)
        if bc_reference is None: # infer from experiment name
            if dataset_name[-5:] in ('-2050','-2100'): bc_reference = dataset_name[:-5] # cut of period indicator and hope for the best 
            else: bc_reference = dataset_name 
        bc_grid = bcargs.pop('grid',None)
        if bc_grid is None: bc_grid = dataargs.grid
        bc_domain = bcargs.pop('domain',None)
        if bc_domain is None: bc_domain = domain
        bc_varlist = bcargs.pop('varlist',None)
        bc_varmap = bcargs.pop('varmap',None)       
        bc_tag = bcargs.pop('tag',None) # an optional name extension/tag
        bc_pattern = bcargs.pop('file_pattern',None) # usually default in getPickleFile
        lgzip = bcargs.pop('lgzip',None) # if pickle is gzipped (None: auto-detect based on file name extension)
        # get name of pickle file (and folder)
        picklefolder = dataargs.avgfolder.replace(dataset_name,bc_reference)
        picklepath = findPicklePath(method=bc_method, obs_name=bc_obs, gridstr=bc_grid, domain=bc_domain, 
                                    tag=bc_tag, pattern=bc_pattern, folder=picklefolder, lgzip=lgzip)
        # determine age of pickle file and compare against source age
        pickleage = datetime.fromtimestamp(os.path.getmtime(picklepath))
    else:
      bc_method = False 
      pickleage = srcage
    
    # parse export options
    expargs = expargs.copy() # first copy, then modify...
    lm3 = expargs.pop('lm3') # convert kg/m^2/s to m^3/m^2/s (water flux)
    expformat = expargs.pop('format') # needed to get FileFormat object
    exp_list= expargs.pop('exp_list') # this handled outside of export
    compute_list = expargs.pop('compute_list', []) # variables to be (re-)computed - by default all
    src_varmap = expargs.pop('src_varmap', dict()) # names of variables in soruce dataset
    # initialize FileFormat class instance
    fileFormat = getFileFormat(expformat, bc_method=bc_method, **expargs)
    # get folder for target dataset and do some checks
    expname = '{:s}_d{:02d}'.format(dataset_name,domain) if domain else dataset_name
    expfolder = fileFormat.defineDataset(dataset=dataset, mode=mode, dataargs=dataargs, lwrite=True, ldebug=ldebug)
  
    # prepare destination for new dataset
    lskip = fileFormat.prepareDestination(srcage=max(srcage,pickleage), loverwrite=loverwrite)
  
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
          raise DateError("Specifed period is inconsistent with netcdf records: '{:s}' != '{:s}'".format(periodstr,source.atts.period))
      
      # load BiasCorrection object from pickle
      if bc_method:     
          # some code to inspect pickles (for testing)
          #from processing.bc_methods import loadBCpickle
          #BC = loadBCpickle(method=bc_method, obs_name=bc_obs, gridstr=bc_grid, domain=bc_domain, 
          #                 tag=bc_tag, pattern=bc_pattern, folder=picklefolder, lgzip=lgzip)
          #times = np.arange(np.datetime64('2009-01-01'), np.datetime64('2010-01-01'))
          #test = BC.correctionByTime(varname='liqwatflx', time=times, )
          #print(test.shape)
          op = gzip.open if lgzip else open
          with op(picklepath, 'rb') as filehandle:
              BC = pickle.load(filehandle) 
          # assemble logger entry
          bcmsgstr = "(performing bias-correction using {:s} from {:s} towards {:s})".format(BC.long_name,bc_reference,bc_obs)
      
      # print message
      if mode == 'climatology': opmsgstr = 'Exporting Climatology ({:s}) to {:s} Format'.format(periodstr, expformat)
      elif mode == 'time-series': opmsgstr = 'Exporting Time-series to {:s} Format'.format(expformat)
      elif mode[-5:] == '-mean': opmsgstr = 'Exporting {:s}-Mean ({:s}) to {:s} Format'.format(mode[:-5], periodstr, expformat)
      else: raise NotImplementedError("Unrecognized Mode: '{:s}'".format(mode))        
      # print feedback to logger
      logmsg = '\n{0:s}   ***   {1:^65s}   ***   \n{0:s}   ***   {2:^65s}   ***   \n'.format(pidstr,datamsgstr,opmsgstr)
      if bc_method:
          logmsg += "{0:s}   ***   {1:^65s}   ***   \n".format(pidstr,bcmsgstr)
      logger.info(logmsg)
      if not lparallel and ldebug: logger.info('\n'+str(source)+'\n')
      
      # apply bias-correction (to source dataset, replacing NC vars with non-NC vars)
      if bc_method:
          source = BC.correct(source, asNC=False, varlist=bc_varlist, varmap=bc_varmap) # load bias-corrected variables into memory
      
      # create GDAL-enabled target dataset
      sink = Dataset(axes=(source.xlon,source.ylat), name=expname, title=source.title, atts=source.atts.copy())
      addGDALtoDataset(dataset=sink, griddef=source.griddef)
      assert sink.gdal, sink
        
      # N.B.: for variables that are not bias-corrected, data are not loaded immediately but on demand; this way 
      #       I/O and computing can be further disentangled and not all variables are always needed
      
      # check if we have radiation data
      lrad = ( 'rad' in dataargs.filetypes )
      
      # regex for generic shift pattern (below)
      regex1 = re.compile('.*[+-]\d$')
      regex2 = re.compile('.*[+-]\d\d$')
      
      # compute intermediate variables, if necessary
      for varname in exp_list:
          variables = None # variable list
          var = None
          # (re-)compute variable, if desired...
          if varname in compute_list:              
              if varname == 'precip': var = newvars.computeTotalPrecip(source)
              elif varname == 'waterflx': var = newvars.computeWaterFlux(source)
              elif varname == 'liqwatflx': var = newvars.computeLiquidWaterFlux(source)
              elif varname == 'netrad': var = newvars.computeNetRadiation(source, asVar=True)
              elif varname == 'netrad_bb': var = newvars.computeNetRadiation(source, asVar=True, lrad=False, name='netrad_bb')
              elif varname == 'netrad_bb0': var = newvars.computeNetRadiation(source, asVar=True, lrad=False, lA=False, name='netrad_bb0')
              elif varname == 'vapdef': var = newvars.computeVaporDeficit(source)
              elif varname in ('pet','pet_pm','petrad','petwnd') and 'pet' not in sink:
                  if 'petrad' in exp_list or 'petwnd' in exp_list:
                      variables = newvars.computePotEvapPM(source, lterms=True, lrad=lrad) # default; returns mutliple PET terms
                  else: var = newvars.computePotEvapPM(source, lterms=False) # returns only PET
              elif varname == 'pet_th': var = None # skip for now
                  #var = computePotEvapTh(source) # simplified formula (less prerequisites)
              else:
                  # determine generic shift pattern
                  r1 = regex1.search(varname)
                  r2 = regex2.search(varname)
                  if r1:
                      variable = r1.group()[:-2]
                      shift = int(r1.group()[-2:])
                  elif r2:
                      variable = r2.group()[:-3]
                      shift = int(r2.group()[-3:])/10.
                  else:
                      shift = None
                  if shift is not None:
                      if variable == 'liqwatflx': var = newvars.computeLiquidWaterFlux(source, shift=shift)
                      else: var = newvars.shiftVariable(source[variable], shift=shift)
                                                
          # ... otherwise load from source file
          if var is None and variables is None:
              if varname in source:
                  var = source[varname].load() # load data (may not have to load all)
              elif varname in src_varmap:
                  srcname = src_varmap[varname] 
                  if srcname in source:
                      var = source[srcname].load() # load data (may not have to load all)
                      var.name = varname # rename for output
          #else: raise VariableError, "Unsupported Variable '{:s}'.".format(varname)
          # for now, skip variables that are None
          if var or variables:
              # handle lists as well
              if var and variables: raise VariableError(var,variables)
              elif var: variables = (var,)
              for var in variables:
                  addGDALtoVar(var=var, griddef=sink.griddef)
                  if not var.gdal and isinstance(fileFormat,ASCII_raster):
                      raise GDALError("Exporting to ASCII_raster format requires GDAL-enabled variables.")
                  # add to new dataset
                  sink += var
      # convert units
      if lm3:
          for var in sink:
              if var.units == 'kg/m^2/s':
                  var /= 1000. # divide to get m^3/m^2/s
                  var.units = 'm^3/m^2/s' # update units
      
      # compute seasonal mean if we are in mean-mode
      if mode[-5:] == '-mean': 
          sink = sink.seasonalMean(season=mode[:-5], lclim=True)
          # N.B.: to remain consistent with other output modes, 
          #       we need to prevent renaming of the time axis
          sink = concatDatasets([sink,sink], axis='time', lensembleAxis=True)
          sink.squeeze() # we need the year-axis until now to distinguish constant fields; now remove
      
      # print dataset
      if not lparallel and ldebug:
          logger.info('\n'+str(sink)+'\n')
        
      # export new dataset to selected format
      fileFormat.exportDataset(sink)
        
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
    if 'PYAVG_THREADS' in os.environ: 
      NP = int(os.environ['PYAVG_THREADS'])
    else: NP = None
    # run script in debug mode
    if 'PYAVG_DEBUG' in os.environ: 
      ldebug =  os.environ['PYAVG_DEBUG'] == 'DEBUG' 
    else: ldebug = False
    # run script in batch or interactive mode
    if 'PYAVG_BATCH' in os.environ: 
      lbatch =  os.environ['PYAVG_BATCH'] == 'BATCH' 
    else: lbatch = False # for debugging
    # re-compute everything or just update 
    if 'PYAVG_OVERWRITE' in os.environ: 
      loverwrite =  os.environ['PYAVG_OVERWRITE'] == 'OVERWRITE' 
    else: loverwrite = ldebug # False means only update old files
    
    ## define settings
    if lbatch:
        # load YAML configuration
        config = loadYAML('export.yaml', lfeedback=True)
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
        domains = config['WRF_domains']
        grids = config['grids']
        # bias correction
        bc_args = config.get('bias_correction',None) # missing parameters are inferred from experiment
        if bc_args is not None:    
            bc_method = bc_args.pop('method') # bias-correction method
            obs_dataset = bc_args.pop('obs_dataset') # observations used for bias correction
            bc_reference = bc_args.pop('reference',None) # reference experiment (None: auto-detect based on name)
        else: bc_method = None # bc_method == None: no bias correction
        # target data specs
        export_arguments = config['export_parameters'] # this is actually a larger data structure
    else:
        # settings for testing and debugging
        NP = 1; ldebug = False # for quick computations
#         NP = 1 ; ldebug = True # just for tests
#         modes = ('time-series','climatology')
        modes = ('annual-mean','climatology', 'time-series')
#         modes = ('annual-mean','climatology',)
#         modes = ('annual-mean',)
#         modes = ('climatology',)  
#         modes = ('time-series',)  
        loverwrite = True
        exp_list= None
        load_list = []
        # obs variables
#         load_list = ['lat2D','lon2D',]
        load_list = ['liqwatflx','pet_pts','pet_har'] # Merged Forcing...
#         load_list = ['liqwatflx',] # SnoDAS...
#         load_list = ['liqwatflx_swe',] # corrected SnoDAS...
#         load_list = ['lat2D','lon2D','liqwatflx','pet']
#         CMC_adjusted = sum([['liqwatflx'+tag,'liqwatflx_CMC'+tag] for tag in ('','_adj30','_adj35')],[])
#         print(CMC_adjusted)
#         load_list = ['lat2D','lon2D','pet',]+CMC_adjusted # 'precip',
#         # WRF variables
#         load_list += ['lat2D','lon2D','zs']
#         ## for HGS/ASCII export
#         load_list = []
#         load_list += ['pet_wrf','pet','evap'] # ET
#         load_list += ['waterflx','liqwatflx','snwmlt','snow',] # water flux / snow
#         ## for NetCDF-4 export/analysis
#         load_list += ['liqprec','solprec','precip','preccu','precnc'] # precip types
# #         # PET variables (for WRF)
#         load_list += ['ps','u10','v10','Q2','Tmin','Tmax','T2','TSmin','TSmax',] # wind & temperature
#         #load_list += ['grdflx','A','SWDNB','e','LWDNB',] # radiation (short)
#         load_list += ['grdflx','A','SWD','e','GLW','SWDNB','SWUPB','LWDNB','LWUPB'] # radiation
# #         # WRF constants
#         load_list += ['lat2D','lon2D','zs','landuse','landmask','LANDUSEF','vegcat','SHDMAX','SHDMIN',
#                     'SOILHGT','soilcat','SOILCTOP','SOILCBOT','LAKE_DEPTH','SUNSHINE','MAPFAC_M'] # constants
        # period list
        periods = [] 
        periods += [15]
#         periods += [30]
        # Observations/Reanalysis
        resolutions = {'CRU':'','GPCC':['025','05','10','25'],'NARR':'','CFSR':['05','031'],'NRCan':'NA12'}
        lLTM = False # also regrid the long-term mean climatologies 
        datasets = []
#         datasets += ['NRCan']; periods = [(1970,2000),(1980,2010),] # this will generally not work, because we don't have snow/-melt...
#         resolutions = {'NRCan': ['na12_ephemeral','na12_maritime','na12_prairies'][1:2]}
#         datasets = ['NRCan']; periods = [(1970,2000),]; resolutions = {'NRCan': ['na12_taiga','na12_alpine',][:1]}
#         datasets = ['NRCan']; periods = [(1980,2010),]; resolutions = {'NRCan': ['na12_maritime',]}
#         datasets = ['NRCan']; periods = [(1970,2000),]; resolutions = {'NRCan': ['na12_maritime',]}
    #     datasets += ['GPCC','CRU']; #resolutions = {'GPCC':['05']}
#         datasets = ['SnoDAS']; periods = [(2011,2019)]; resolutions['SnoDAS']=('','rfbc')
        datasets = ['MergedForcing']; periods = [(2011,2018)]
        # CESM experiments (short or long name) 
        CESM_project = None # all available experiments
        load3D = False
        CESM_experiments = [] # use None to process all CESM experiments
    #     CESM_experiments += ['Ens']
    #     CESM_experiments += ['Ctrl-1', 'Ctrl-A', 'Ctrl-B', 'Ctrl-C']
        CESM_filetypes = ['atm','lnd']
        # WRF experiments (short or long name)
#         WRF_project = 'GreatLakes'; unity_grid = 'glb1_d02' # only GreatLakes experiments
        WRF_project = 'WesternCanada'; unity_grid = 'arb2_d02' # only WesternCanada experiments
        WRF_experiments = [] # use None to process all WRF experiments
#         WRF_experiments += ['erai-g','erai-t','erai-g3','erai-t3',]
#         WRF_experiments += ['g-ensemble','g-ensemble-2050','g-ensemble-2100']
#         WRF_experiments += ['t-ensemble','t-ensemble-2050','t-ensemble-2100']
#         WRF_experiments += ['g3-ensemble','g3-ensemble-2050','g3-ensemble-2100']
#         WRF_experiments += ['t3-ensemble','t3-ensemble-2050','t3-ensemble-2100']
#         WRF_experiments += ['t3-ensemble-2100','g3-ensemble-2100']
#         WRF_experiments += ['max-ensemble','ctrl-ensemble']
#         WRF_experiments += ['g-ctrl','g-ctrl-2050','g-ctrl-2100']
#         WRF_experiments += ['t-ctrl','t-ctrl-2050','t-ctrl-2100']
#         WRF_experiments += ['new-v361-ctrl', 'new-v361-ctrl-2050', 'new-v361-ctrl-2100']
#         WRF_experiments += ['erai-3km','max-3km']
#         WRF_experiments += ['max-ctrl','max-ctrl-2050','max-ctrl-2100']
#         WRF_experiments += ['max-ensemble']
#         WRF_experiments += ['ctrl-ensemble','ctrl-ensemble-2050','ctrl-ensemble-2100']
#         WRF_experiments += ['max-ensemble','max-ensemble-2050','max-ensemble-2100']
#         WRF_experiments += ['max-ctrl','max-ens-A','max-ens-B','max-ens-C',]
#         WRF_experiments += ['max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',]    
#         WRF_experiments += ['max-ctrl-2100','max-ens-A-2100','max-ens-B-2100','max-ens-C-2100',] 
#         WRF_experiments += ['ctrl-1',   'ctrl-ens-A',     'ctrl-ens-B',     'ctrl-ens-C',]
#         WRF_experiments += ['ctrl-2050','ctrl-ens-A-2050','ctrl-ens-B-2050','ctrl-ens-C-2050',]    
#         WRF_experiments += ['ctrl-2100','ctrl-ens-A-2100','ctrl-ens-B-2100','ctrl-ens-C-2100',]    
#         WRF_experiments += ['g3-ctrl',     'g3-ens-A',     'g3-ens-B',     'g3-ens-C',]
#         WRF_experiments += ['g3-ctrl-2050','g3-ens-A-2050','g3-ens-B-2050','g3-ens-C-2050',]
#         WRF_experiments += ['g3-ctrl-2100','g3-ens-A-2100','g3-ens-B-2100','g3-ens-C-2100',]
#         WRF_experiments += ['t3-ctrl',     't3-ens-A',     't3-ens-B',     't3-ens-C',]
#         WRF_experiments += ['t3-ctrl-2050','t3-ens-A-2050','t3-ens-B-2050','t3-ens-C-2050',]
#         WRF_experiments += ['t3-ctrl-2100','t3-ens-A-2100','t3-ens-B-2100','t3-ens-C-2100',]
#         WRF_experiments += ['g-ctrl',     'g-ens-A',     'g-ens-B',     'g-ens-C',]  # bc_reference = 'g-ensemble'
#         WRF_experiments += ['g-ctrl-2050','g-ens-A-2050','g-ens-B-2050','g-ens-C-2050',]
#         WRF_experiments += ['g-ctrl-2100','g-ens-A-2100','g-ens-B-2100','g-ens-C-2100',]
#         WRF_experiments += ['t-ctrl',     't-ens-A',     't-ens-B',     't-ens-C',] # bc_reference = 't-ensemble'
#         WRF_experiments += ['t-ctrl-2050','t-ens-A-2050','t-ens-B-2050','t-ens-C-2050',]
#         WRF_experiments += ['t-ctrl-2100','t-ens-A-2100','t-ens-B-2100','t-ens-C-2100',]
        # other WRF parameters 
#         domains = 2 # domains to be processed
#         domains = 1 # domains to be processed
        domains = None # process all domains
#         WRF_filetypes = ('hydro',) # available input files
#         WRF_filetypes = ('hydro','srfc','xtrm','lsm','rad') # available input files
        WRF_filetypes = ('hydro','srfc','xtrm','lsm',) # available input files
#         WRF_filetypes = ('const',) # with radiation files
        ## bias-correction paramter
        bc_method = None; bc_tag = '' # no bias correction
#         bc_method = 'SMBC' # bias correction method (None: no bias correction)        
#         bc_method = 'AABC' # bias correction method (None: no bias correction)        
#         bc_method = 'MyBC' # bias correction method (None: no bias correction)        
#         obs_dataset = 'CRU' # the observational dataset 
#         bc_tag = bc_method+'_'+obs_dataset+'_' 
#         bc_reference = 'ctrl-ensemble' # reference experiment (None: auto-detect based on name)
#         bc_reference = 'max-ensemble' # reference experiment (None: auto-detect based on name)
        bc_reference = None # auto-detect reference experiment based on name
        bc_varmap = dict(TSmin='Tmin', TSmax='Tmax',Tmean='T2', evap='pet', # pet='pet_wrf', pet_wrf='pet', 
                         SWUPB='SWDNB',SWD='SWDNB',SWDNB='SWD', LWDNB='GLW',GLW='LWDNB',)
        bc_args = dict(grid=None, domain=None, lgzip=True, varmap=bc_varmap) # missing/None parameters are inferred from experiment
        # typically a specific grid is required
        grids = [] # list of grids to process
#         grids += [None]; project = None # special keyword for native grid
#         grids += ['arb2']; project = 'ARB' # main grid for ARB project
#         grids += ['uph1']; project = 'Elisha' # grid for Elisha
#         grids += ['glb1']; project = 'GLB' # grid for Great Lakes Basin project
#         grids += ['grw1']; project = 'GRW' # finer 1km grid for GRW project
#         grids += ['grw2']; project = 'GRW' # small grid for GRW project
#         grids += ['grw3']; project = 'GRW' # fine grid for GRW project
#         grids += ['asb1']; project = 'ASB' # main grid for ASB project
#         grids += ['brd1']; project = 'ASB' # small grid for ASB project
#         grids += ['can1']; project = 'CAN' # large Canada-wide grid
#         grids += ['snw1']; project = 'SNW' # south nation watershed
#         grids += ['son1']; project = 'SON' # southern Ontario watersheds
        grids += ['son2']; project = 'SON' # southern Ontario watersheds
        ## export to ASCII raster
        hgs_root = os.getenv('HGS_ROOT', os.getenv('DATA_ROOT', None)+'/HGS/')
        export_arguments = dict(
            # NRCan
#             folder = '{0:s}/{{PROJECT}}/{{GRID}}/{{EXPERIMENT}}/{1:s}{{PERIOD}}/climate_forcing/'.format(os.getenv('HGS_ROOT', None),bc_tag),
            folder = '{0:s}/{{PROJECT}}/{{GRID}}/{{EXPERIMENT}}/{{PERIOD}}/climate_forcing/'.format(hgs_root),
#             compute_list = [], exp_list= ['lat2D','lon2D','pet']+CMC_adjusted,   # varlist for NRCan
#             compute_list = [], exp_list= ['lat2D','lon2D','pet','liqwatflx','liqwatflx_CMC'], # varlist for NRCan
#             exp_list= ['liqwatflx',], src_varmap=dict(liqwatflx='liqwatflx_swe'), # varlist for SnoDAS
            compute_list = [], exp_list= ['pet_pts','pet_har','liqwatflx'], # varlist for MergedForcing
            # WRF
# #             exp_list= ['landuse','landmask'],
# #             exp_list= ['lat2D','lon2D','zs','LU_MASK','LU_INDEX','LANDUSEF','VEGCAT','SHDMAX','SHDMIN',
# #                        'SOILHGT','SOILCAT','SOILCTOP','SOILCBOT','LAKE_DEPTH','SUNSHINE','MAPFAC_M'], # constants
# #             compute_list = ['waterflx','liqwatflx','pet'], # variables that should be (re-)computed
# #             exp_list= ['lat2D','lon2D','zs','waterflx','liqwatflx','pet','pet_wrf'], # varlist for export
# #             compute_list = ['liqwatflx','pet'], exp_list= ['lat2D','lon2D','zs','liqwatflx','pet'], # short varlist for quick export
# #             compute_list = ['waterflx','waterflx-1','liqwatflx','liqwatflx-1','snwmlt-1','pet-1','liqwatflx-05','snwmlt-05','pet-05'],
# #             exp_list= ['lat2D','lon2D','zs','pet','liqwatflx','liqwatflx-1','pet-1','liqwatflx-05','pet-05'], # varlist with shifts
# #             exp_list= ['lat2D','lon2D','zs','pet','liqwatflx'], # short varlist for quick export
#             compute_list = ['pet-1','pet-05'], exp_list= ['pet','pet-1','pet-05'], # varlist with shifts
#             folder = '{0:s}/{{PROJECT}}/{{GRID}}/{{EXPERIMENT}}/{1:s}{{PERIOD}}/climate_forcing/'.format(os.getenv('HGS_ROOT'),bc_tag),
# #             folder = '//aquanty-nas/share/temp_data_exchange/Erler/{PROJECT}/{EXPERIMENT}/{PERIOD}/',
# #             folder = '//aquanty-nas/share/temp_data_exchange/Erler/{{PROJECT}}/{{EXPERIMENT}}/{bc_tag:s}{{PERIOD}}/'.format(bc_tag=bc_tag),
# #             folder = '{0:s}/{{PROJECT}}/{{GRID}}/{{EXPERIMENT}}/land_data/'.format(os.getenv('HGS_ROOT')),
# #             folder = '//AQFS1/Data/temp_data_exchange/{PROJECT}/{GRID}/{EXPERIMENT}/land_data/',
            # common
            project = project, # project designation  
            prefix = '{GRID}', # based on keyword arguments
            format = 'ASCII_raster', # formats to export to
            fillValue = 0, noDataValue = -9999, # in case we interpolate across a missing value...
            lm3 = True) # convert water flux from kg/m^2/s to m^3/m^2/s
        ## export to NetCDF (aux-file)
#         exp_list = []
#         exp_list += ['waterflx','liqwatflx','liqprec','solprec','precip','snow','snowh','snwmlt',]
#         exp_list += ['netrad','netrad_bb0','netrad_bb','vapdef','pet','pet_wrf','petrad','petwnd']
# #         exp_list += ['Tmin','Tmax','T2','Tmean','TSmin','TSmax','SWDNB','LWDNB','zs','lat2D','lon2D',]
# #         exp_list += ['SWDNB','SWUPB',]
# #         exp_list += ['Tmin','Tmax','T2','Tmean','TSmin','TSmax','Q2','evap','waterflx','zs','lat2D','lon2D',]
# #         exp_list += ['liqwatflx','liqprec','solprec','preccu','precnc','precip','snwmlt','pet_wrf','pet']
# #         exp_list += ['liqwatflx-1','snwmlt-1','pet-1','liqwatflx-05','snwmlt-05','pet-05']
#         if bc_method:
#             filename = bc_method
#             if obs_dataset == 'NRCan': pass # for historical reasons, NRCan gets a free pass (default)
#             elif obs_dataset == 'Unity': filename += '1'
#             elif obs_dataset == 'CRU': filename += '2'
#             else:
#                 raise NotImplemented("Need to assign number/identifier to obs dataset '{}'".format(obs_dataset))
#         else: 
#             filename = 'AUX'
# #         compute_list = ['waterflx','waterflx-1','liqwatflx','liqwatflx-1','snwmlt-1','pet-1','liqwatflx-05','snwmlt-05','pet-05']
#         compute_list = ['waterflx','liqwatflx','pet'] # variables that should be (re-)computed
#         export_arguments = dict(format = 'NetCDF',
#                                 exp_list= exp_list, compute_list=compute_list, 
#                                 project = filename, filetype = filename.lower(),
#                                 lm3 = False) # do not convert water flux from kg/m^2/s to m^3/m^2/s
      
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
      print('\n Exporting WRF Datasets:')
      print([exp.name for exp in WRF_experiments])
    if len(CESM_experiments) > 0:
      print('\n Exporting CESM Datasets:')
      print([exp.name for exp in CESM_experiments])
    if len(datasets) > 0:
      print('\n And Observational Datasets:')
      print(datasets)
    print(('\n From Grid/Resolution:\n   {:s}'.format(printList(grids))))
    print(('To File Format {:s}'.format(export_arguments['format'])))
    print(('\n Project Designation: {:s}'.format(export_arguments['project'])))
    if bc_method:
      print('\n And Observational Datasets:')
      print(datasets)
    print(('Export Variable List: {:s}'.format(printList(export_arguments['exp_list']))))
    if export_arguments['lm3']: '\n Converting kg/m^2/s (mm/s) into m^3/m^2/s (m/s)'
    # check formats (will be iterated over in export function, hence not part of task list)
    if export_arguments['format'] == 'ASCII_raster':
      print(('Export Folder: {:s}'.format(export_arguments['folder'])))
      print(('File Prefix: {:s}'.format(export_arguments['prefix'])))
    elif export_arguments['format'].lower() in ('netcdf','netcdf4'):
      pass
    else:
      raise ArgumentError("Unsupported file format: '{:s}'".format(export_arguments['format']))
    print(('\nOVERWRITE: {0:s}'.format(str(loverwrite))))
    # bias-correction parameters (if used)
    print(('\nBias-Correction: {}'.format(bc_method)))
    if bc_method:
      print(('  Observational Dataset: {}'.format(obs_dataset)))
      print(('  Reference Dataset: {}'.format(bc_reference)))
      print(('  Parameters: {}'.format(bc_args)))
    print('\n') # separator space
      
    ## construct argument list
    args = []  # list of job packages
    # loop over modes
    for mode in modes:
      # only climatology mode has periods    
      if mode[-5:] == '-mean': periodlist = periods
      elif mode == 'climatology': periodlist = periods
      elif mode == 'time-series': periodlist = (None,)
      else: raise NotImplementedError("Unrecognized Mode: '{:s}'".format(mode))
  
      # loop over target grids ...
      for grid in grids:
        
          # observational datasets (grid depends on dataset!)
          for dataset in datasets:
            mod = import_module('datasets.{0:s}'.format(dataset))
            if isinstance(resolutions,dict): 
              if dataset not in resolutions: resolutions[dataset] = ('',)
              elif not isinstance(resolutions[dataset],(list,tuple)): resolutions[dataset] = (resolutions[dataset],)                
            elif resolutions is not None: raise TypeError                                
            if mode[-5:] == '-mean' or mode == 'climatology':
              # some datasets come with a climatology 
              if lLTM:
                if resolutions is None: dsreses = mod.LTM_grids
                elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.LTM_grids]  
                for dsres in dsreses: 
                  args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=None, resolution=dsres, unity_grid=unity_grid)) ) # append to list
              # climatologies derived from time-series
              if resolutions is None: dsreses = mod.TS_grids
              elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
              for dsres in dsreses:
                for period in periodlist:
                  args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=period, resolution=dsres, unity_grid=unity_grid)) ) # append to list            
            elif mode == 'time-series': 
              # regrid the entire time-series
              if resolutions is None: dsreses = mod.TS_grids
              elif isinstance(resolutions,dict): dsreses = [dsres for dsres in resolutions[dataset] if dsres in mod.TS_grids]  
              for dsres in dsreses:
                args.append( (dataset, mode, dict(grid=grid, varlist=load_list, period=None, resolution=dsres, unity_grid=unity_grid)) ) # append to list            
          
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
              tmpdom = list(range(1,experiment.domains+1))
            else: tmpdom = domains
            for domain in tmpdom:
              for period in periodlist:
                # arguments for worker function: dataset and dataargs       
                args.append( ('WRF', mode, dict(experiment=experiment, filetypes=WRF_filetypes, grid=grid, 
                                                varlist=load_list, domain=domain, period=period)) )
    
    # put bias correction arguments into a single dict
    if bc_method:
        bc_args['method'] = bc_method
        bc_args['obs_dataset'] = obs_dataset 
        bc_args['reference'] = bc_reference
    else:
        bc_args = None
    # static keyword arguments
    kwargs = dict(expargs=export_arguments, bcargs=bc_args, loverwrite=loverwrite, )
    # N.B.: formats will be iterated over inside export function
    
    ## call parallel execution function
    ec = asyncPoolEC(performExport, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
    # exit with fraction of failures (out of 10) as exit code
    exit(int(10+int(10.*ec/len(args))) if ec > 0 else 0)
