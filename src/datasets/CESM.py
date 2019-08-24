'''
Created on 2013-12-04

This module contains common meta data and access functions for CESM model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os
try: import pickle as pickle
except: import pickle

# from atmdyn.properties import variablePlotatts
from geodata.base import Variable, Axis, concatDatasets, monthlyUnitsList
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, GDALError
from geodata.misc import DatasetError, AxisError, DateError, ArgumentError, isNumber, isInt
from datasets.common import grid_folder, default_varatts, getRootFolder
from datasets.common import addLengthAndNamesOfMonth, selectElements, stn_params, shp_params
from geodata.gdal import loadPickledGridDef, pickleGridDef
from datasets.WRF import Exp as WRF_Exp
from processing.process import CentralProcessingUnit

# some meta data (needed for defaults)
dataset_name = 'CESM' # dataset name
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='WRF')
outfolder = root_folder + 'cesmout/' # WRF output folder
avgfolder = root_folder + 'cesmavg/' # monthly averages and climatologies
cvdpfolder = root_folder + 'cvdp/' # CVDP output (netcdf files and HTML tree)
diagfolder = root_folder + 'diag/' # output from AMWG diagnostic package (climatologies and HTML tree) 

## list of experiments
class Exp(WRF_Exp): 
  parameters = WRF_Exp.parameters.copy()
  defaults = WRF_Exp.defaults.copy()
  # special CESM parameters
  parameters['cvdpfolder'] = dict(type=str,req=True) # new parameters need to be registered
  parameters['diagfolder'] = dict(type=str,req=True) # new parameters need to be registered
  # and defaults
  defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/'.format(avgfolder,atts['name'])
  defaults['cvdpfolder'] = lambda atts: '{0:s}/{1:s}/'.format(cvdpfolder,atts['name'])
  defaults['diagfolder'] = lambda atts: '{0:s}/{1:s}/'.format(diagfolder,atts['name'])
  defaults['domains'] = None # not applicable here
  defaults['parent'] = None # not applicable here
  

# return name and folder
def getFolderName(name=None, experiment=None, folder=None, mode='avg', cvdp_mode=None, exps=None):
  ''' Convenience function to infer and type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  # figure out experiment name
  if exps is None or ( experiment is None and name not in exps ):
    if cvdp_mode is None: cvdp_mode = 'ensemble' # backwards-compatibility
    if not isinstance(folder,str):
      if mode == 'cvdp' and ( cvdp_mode == 'observations' or cvdp_mode == 'grand-ensemble' ): 
        folder = "{:s}/grand-ensemble/".format(cvdpfolder)              
      else: raise IOError("Need to specify an experiment folder in order to load data.")    
  else:
    # load experiment meta data
    if isinstance(experiment,Exp): pass # preferred option
    elif exps is None: raise DatasetError('No dictionary of Exp instances specified.')
    elif isinstance(experiment,str): experiment = exps[experiment] 
    elif isinstance(name,str) and name in exps: experiment = exps[name]
    else: raise DatasetError('Dataset of name \'{0:s}\' not found!'.format(name or experiment))
    if cvdp_mode is None:
      cvdp_mode = 'ensemble' if experiment.ensemble else ''  
    # root folder
    if folder is None: 
      if mode == 'avg': folder = experiment.avgfolder
      elif mode == 'cvdp': 
        if cvdp_mode == 'ensemble': 
          expfolder = experiment.ensemble or experiment.name 
          folder = "{:s}/{:s}/".format(cvdpfolder,expfolder)
        elif cvdp_mode == 'grand-ensemble': folder = "{:s}/grand-ensemble/".format(cvdpfolder)
        else: folder = experiment.cvdpfolder
      elif mode == 'diag': folder = experiment.diagfolder
      else: raise NotImplementedError("Unsupported mode: '{:s}'".format(mode))
    elif not isinstance(folder,str): raise TypeError
    # name
    if name is None: name = experiment.shortname
    if not isinstance(name,str): raise TypeError      
  # check if folder exists
  if not os.path.exists(folder): raise IOError('Dataset folder does not exist: {0:s}'.format(folder))
  # return name and folder
  return folder, experiment, name


# function to undo NCL's lonFlip
def flipLon(data, flip=144, lrev=False, var=None, slc=None):
  ''' shift longitude on the fly, so as to undo NCL's lonFlip; only works on entire array '''
  if var is not None: # ignore parameters
    if not isinstance(var,VarNC): raise TypeError
    ax = var.axisIndex('lon')
    flip = len(var.lon)/2
  if data.ndim < var.ndim: # some dimensions have been squeezed
    sd = 0 # squeezed dimensions before ax
    for sl in slc:
      if isinstance(sl,(int,np.integer)): sd += 1
    ax -= sd # remove squeezed dimensions
  if not ( data.ndim > ax and data.shape[ax] == flip*2 ): 
    raise NotImplementedError("Can only shift longitudes of the entire array!")
  # N.B.: this operation only makes sense with a full array!
  if lrev: flip *= -1 # reverse flip  
  data = np.roll(data, shift=flip, axis=ax) # shift values half way along longitude
  return data


## variable attributes and name
# a generic class for CESM file types
class FileType(object): 
  ''' A generic class that describes CESM file types. '''
  def __init__(self, *args, **kwargs):
    ''' generate generic attributes using a name argument '''
    if len(args) == 1: 
      name = args[0]
      self.name = name
      self.atts = dict() # should be properly formatted already
      #self.atts = dict(netrad = dict(name='netrad', units='W/m^2'))
      self.vars = []    
      self.climfile = 'cesm{:s}{{0:s}}_clim{{1:s}}.nc'.format(name) # generic climatology name 
      # final filename needs to be extended by ('_'+grid,'_'+period)
      self.tsfile = 'cesm{:s}{{0:s}}_monthly.nc'.format(name) # generic time-series name
      # final filename needs to be extended by ('_'+grid,)
    else: raise ArgumentError  
# surface variables
class ATM(FileType):
  ''' Variables and attributes of the surface files. '''
  def __init__(self):
    self.atts = dict(TREFHT   = dict(name='T2', units='K'), # 2m Temperature
                     TREFMXAV = dict(name='Tmax', units='K'),   # Daily Maximum Temperature (at surface)                     
                     TREFMNAV = dict(name='Tmin', units='K'),   # Daily Minimum Temperature (at surface)
                     TREFMX   = dict(name='MaxTmax', units='K'), # Monthly Maximum Temperature (at surface)                     
                     TREFMN   = dict(name='MinTmin', units='K'), # Monthly Minimum Temperature (at surface)
                     QREFHT   = dict(name='q2', units='kg/kg'), # 2m water vapor mass mixing ratio                     
                     TS       = dict(name='Ts', units='K'), # Skin Temperature (SST)
                     #TS       = dict(name='SST', units='K'), # Skin Temperature (SST)
                     TSMX     = dict(name='MaxTs', units='K'),   # Maximum Skin Temperature (SST)
                     TSMN     = dict(name='MinTs', units='K'),   # Minimum Skin Temperature (SST)                     
                     PRECT    = dict(name='precip', units='kg/m^2/s', scalefactor=1000.), # total precipitation rate (kg/m^2/s) 
                     PRECC    = dict(name='preccu', units='kg/m^2/s', scalefactor=1.), # convective precipitation rate (kg/m^2/s)
                     PRECSC   = dict(name='solpreccu', units='kg/m^2/s', scalefactor=1.), # solid convective precip rate (kg/m^2/s)
                     PRECL    = dict(name='precnc', units='kg/m^2/s', scalefactor=1.), # grid-scale precipitation rate (kg/m^2/s)
                     PRECSL   = dict(name='solprec', units='kg/m^2/s', scalefactor=1.), # solid precipitation rate
                     PRECSH   = dict(name='precsh', units='kg/m^2/s', scalefactor=1.), # shallow convection precip rate (kg/m^2/s)
                     PRECTMX  = dict(name='MaxPrecip', units='kg/m^2/s', scalefactor=1.), # maximum daily precip                    
                     MaxPRECT_1d  = dict(name='MaxPrecip_1d', units='kg/m^2/s', scalefactor=1000.), # maximum daily precip                    
                     MaxPRECC_1d  = dict(name='MaxPreccu_1d', units='kg/m^2/s', scalefactor=1.), # maximum daily precip                    
                     precip    = dict(name='precip', units='kg/m^2/s', scalefactor=1.), # total precipitation rate (kg/m^2/s) 
                     preccu    = dict(name='preccu', units='kg/m^2/s', scalefactor=1.), # convective precipitation rate (kg/m^2/s)
                     solpreccu   = dict(name='solpreccu', units='kg/m^2/s', scalefactor=1.), # solid convective precip rate (kg/m^2/s)
                     precnc    = dict(name='precnc', units='kg/m^2/s', scalefactor=1.), # grid-scale precipitation rate (kg/m^2/s)
                     solprec   = dict(name='solprec', units='kg/m^2/s', scalefactor=1.), # solid precipitation rate
                     precsh   = dict(name='precsh', units='kg/m^2/s', scalefactor=1.), # shallow convection precip rate (kg/m^2/s)
                     MaxPrecip  = dict(name='MaxPrecip', units='kg/m^2/s', scalefactor=1.), # maximum daily precip                    
                     MaxPrecip_1d  = dict(name='MaxPrecip_1d', units='kg/m^2/s', scalefactor=1.), # maximum daily precip                    
                     MaxPreccu_1d  = dict(name='MaxPreccu_1d', units='kg/m^2/s', scalefactor=1.), # maximum daily precip                    
                     SNOWHLND = dict(name='snow', units='kg/m^2', scalefactor=1000.), # actuall SWE, not snow depth
                     SNOWHICE = dict(name='snow_ice', units='kg/m^2', scalefactor=1000.), # actuall SWE, not snow depth
                     ICEFRAC  = dict(name='seaice', units=''), # seaice fraction
                     SHFLX    = dict(name='hfx', units='W/m^2'), # surface sensible heat flux
                     LHFLX    = dict(name='lhfx', units='W/m^2'), # surface latent heat flux
                     QFLX     = dict(name='evap', units='kg/m^2/s'), # surface evaporation
                     FLUT     = dict(name='LWUPT', units='W/m^2'), # Outgoing Longwave Radiation
                     FLDS     = dict(name='LWDNB', units='W/m^2'), # Ground Longwave Radiation
                     FSDS     = dict(name='SWDNB', units='W/m^2'), # Downwelling Shortwave Radiation                     
                     OLR      = dict(name='LWUPT', units='W/m^2'), # Outgoing Longwave Radiation
                     GLW      = dict(name='LWDNB', units='W/m^2'), # Ground Longwave Radiation
                     SWD      = dict(name='SWDNB', units='W/m^2'), # Downwelling Shortwave Radiation                     
                     PS       = dict(name='ps', units='Pa'), # surface pressure
                     PSL      = dict(name='pmsl', units='Pa'), # mean sea level pressure
                     PHIS     = dict(name='zs', units='m', scalefactor=1./9.81), # surface elevation
                     #LANDFRAC = dict(name='landfrac', units=''), # land fraction
                     )
    self.vars = list(self.atts.keys())    
    self.climfile = 'cesmatm{0:s}_clim{1:s}.nc' # the filename needs to be extended by ('_'+grid,'_'+period)
    self.tsfile = 'cesmatm{0:s}_monthly.nc' # the filename needs to be extended by ('_'+grid)
# CLM variables
class LND(FileType):
  ''' Variables and attributes of the land surface files. '''
  def __init__(self):
    self.atts = dict(topo     = dict(name='hgt', units='m'), # surface elevation
                     landmask = dict(name='landmask', units=''), # land mask
                     landfrac = dict(name='landfrac', units=''), # land fraction
                     FSNO     = dict(name='snwcvr', units=''), # snow cover (fractional)
                     QMELT    = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate
                     QOVER    = dict(name='sfroff', units='kg/m^2/s'), # surface run-off
                     QRUNOFF  = dict(name='runoff', units='kg/m^2/s'), # total surface and sub-surface run-off
                     QIRRIG   = dict(name='irrigation', units='kg/m^2/s'), # water flux through irrigation
                     H2OSOI   = dict(name='aSM', units='m^3/m^3'), # absolute soil moisture
                     )
    self.vars = list(self.atts.keys())    
    self.climfile = 'cesmlnd{0:s}_clim{1:s}.nc' # the filename needs to be extended by ('_'+grid,'_'+period)
    self.tsfile = 'cesmlnd{0:s}_monthly.nc' # the filename needs to be extended by ('_'+grid)
# CICE variables
class ICE(FileType):
  ''' Variables and attributes of the seaice files. '''
  def __init__(self):
    self.atts = dict() # currently not implemented...                     
    self.vars = list(self.atts.keys())
    self.climfile = 'cesmice{0:s}_clim{1:s}.nc' # the filename needs to be extended by ('_'+grid,'_'+period)
    self.tsfile = 'cesmice{0:s}_monthly.nc' # the filename needs to be extended by ('_'+grid)

# CVDP variables
class CVDP(FileType):
  ''' Variables and attributes of the CVDP netcdf files. '''
  def __init__(self):
    self.atts = dict(pdo_pattern_mon = dict(name='PDO_eof', units='', scalefactor=-1.), # PDO EOF
                     pdo_timeseries_mon = dict(name='PDO', units='', scalefactor=-1.), # PDO time-series
                     pna_mon = dict(name='PNA_eof', units=''), # PNA EOF
                     pna_pc_mon = dict(name='PNA', units=''), # PNA time-series
                     npo_mon = dict(name='NPO_eof', units=''), # NPO EOF
                     npo_pc_mon = dict(name='NPO', units=''), # NPO time-series
                     nao_mon = dict(name='NAO_eof', units=''), # PDO EOF
                     nao_pc_mon = dict(name='NAO', units=''), # PDO time-series
                     nam_mon = dict(name='NAM_eof', units=''), # NAM EOF
                     nam_pc_mon = dict(name='NAM', units=''), # NAM time-series
                     amo_pattern_mon = dict(name='AMO_eof', units='', # AMO EOF
                                            transform=flipLon), # undo shifted longitude (done by NCL)
                     amo_timeseries_mon = dict(name='AMO', units=''), # AMO time-series 
                     nino34 = dict(name='NINO34', units=''), # ENSO Nino34 index
                     npi_ndjfm = dict(name='NPI', units=''), # some North Pacific Index ???
                     )                    
    self.vars = list(self.atts.keys())
    self.indices = [var['name'] for var in list(self.atts.values()) if var['name'].upper() == var['name'] and var['name'] != 'NPI']
    self.eofs = [var['name'] for var in list(self.atts.values()) if var['name'][-4:] == '_eof']
    self.cvdpfile = '{:s}.cvdp_data.{:s}.nc' # filename needs to be extended with experiment name and period

# AMWG diagnostic variables
class Diag(FileType):
  ''' Variables and attributes of the AMWG diagnostic netcdf files. '''
  def __init__(self):
    self.atts = dict() # currently not implemented...                     
    self.vars = list(self.atts.keys())
    self.diagfile = NotImplemented # filename needs to be extended with experiment name and period

# axes (don't have their own file)
class Axes(FileType):
  ''' A mock-filetype for axes. '''
  def __init__(self):
    self.atts = dict(time        = dict(name='time', units='days', offset=-47116, atts=dict(long_name='Month since 1979')), # time coordinate (days since 1979-01-01)
                     TIME        = dict(name='year', units='year', atts=dict(long_name='Years since 1979')), # yearly time coordinate in CVDP files
                     # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                     #       the time offset is chose such that 1979 begins with the origin (time=0)
                     lon           = dict(name='lon', units='deg E'), # west-east coordinate
                     lat           = dict(name='lat', units='deg N'), # south-north coordinate
                     LON           = dict(name='lon', units='deg E'), # west-east coordinate (actually identical to lon!)
                     LAT           = dict(name='lat', units='deg N'), # south-north coordinate (actually identical to lat!)                     
                     levgrnd       = dict(name='s', units=''), # soil layers
                     lev = dict(name='lev', units='')) # hybrid pressure coordinate
    self.vars = list(self.atts.keys())
    self.climfile = None
    self.tsfile = None

# data source/location
fileclasses = dict(atm=ATM(), lnd=LND(), axes=Axes(), cvdp=CVDP()) # ice=ICE() is currently not supported because of the grid
# list of variables and dimensions that should be ignored
ignore_list_2D = ('nbnd', 'slat', 'slon', 'ilev', # atmosphere file
                  'levlak', 'latatm', 'hist_interval', 'latrof', 'lonrof', 'lonatm', # land file
                  ) # CVDP file (omit shifted longitude)
ignore_list_3D = ('lev', 'levgrnd',) # ignore all 3D variables (and vertical axes)

## Functions to load different types of CESM datasets

# CVDP diagnostics (monthly time-series, EOF pattern and correlations) 
def loadCVDP_Obs(name=None, grid=None, period=None, varlist=None, varatts=None, 
                 translateVars=None, lautoregrid=None, ignore_list=None, lindices=False, leofs=False):
  ''' Get a properly formatted monthly observational dataset as NetCDFDataset. '''
  if grid is not None: raise NotImplementedError
  # check datasets
  if name is None:
    if varlist is not None:
      if any(ocnvar in varlist for ocnvar in ('PDO','NINO34','AMO')): 
        name = 'HadISST'
      elif any(ocnvar in varlist for ocnvar in ('NAO','NPI','PNA', 'NPO')): 
        name = '20thC_ReanV2'
    else: raise ArgumentError("Need to provide either 'name' or 'varlist'!")
  name = name.lower() # ignore case
  if name in ('hadisst','sst','ts'):
    name = 'HadISST'; period = period or (1920,2012)
  elif name in ('mlost','t2','tas'):
    name = 'MLOST'; period = period or (1920,2012)
  elif name in ('20thc_reanv2','ps','psl'):
    name = '20thC_ReanV2'; period = period or (1920,2012)
  elif name in ('gpcp','precip','prect','ppt'):
    name = 'GPCP'; period = period or (1979,2014)
  else: raise NotImplementedError("The dataset '{:s}' is not available.".format(name))
  # load smaller selection
  if varlist is None and ( lindices or leofs ):
    varlist = []
  if lindices: varlist += fileclasses['cvdp'].indices
  if leofs: varlist += fileclasses['cvdp'].eofs
  return loadCESM_All(experiment=None, name=name, grid=grid, period=period, filetypes=('cvdp',), 
                      varlist=varlist, varatts=varatts, translateVars=translateVars, 
                      lautoregrid=lautoregrid, load3D=False, ignore_list=ignore_list, mode='CVDP', 
                      cvdp_mode='observations', lcheckExp=False)

# CVDP diagnostics (monthly time-series, EOF pattern and correlations) 
def loadCVDP(experiment=None, name=None, grid=None, period=None, varlist=None, varatts=None, 
             cvdp_mode=None, translateVars=None, lautoregrid=None, ignore_list=None, 
             lcheckExp=True, lindices=False, leofs=False, lreplaceTime=True, exps=None):
  ''' Get a properly formatted monthly CESM climatology as NetCDFDataset. '''
  if grid is not None: raise NotImplementedError
#   if period is None: period = 15
  # load smaller selection
  if varlist is None and ( lindices or leofs ):
    varlist = []
    if lindices: varlist += fileclasses['cvdp'].indices
    if leofs: varlist += fileclasses['cvdp'].eofs
  return loadCESM_All(experiment=experiment, name=name, grid=grid, period=period, filetypes=('cvdp',), 
                  varlist=varlist, varatts=varatts, translateVars=translateVars, lautoregrid=lautoregrid, 
                  load3D=True, ignore_list=ignore_list, mode='CVDP', cvdp_mode=cvdp_mode, 
                  lcheckExp=lcheckExp, lreplaceTime=lreplaceTime, exps=exps)

# Station Time-Series (monthly)
def loadCESM_StnTS(experiment=None, name=None, station=None, filetypes=None, varlist=None, 
                   varatts=None, translateVars=None, load3D=False, ignore_list=None, 
                   lcheckExp=True, lreplaceTime=True, lwrite=False, exps=None):
  ''' Get a properly formatted CESM dataset with a monthly time-series at station locations. '''
  return loadCESM_All(experiment=experiment, name=name, grid=None, period=None, station=station, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, lreplaceTime=lreplaceTime, 
                      translateVars=translateVars, lautoregrid=False, load3D=load3D, lwrite=lwrite, 
                      ignore_list=ignore_list, mode='time-series', lcheckExp=lcheckExp, 
                      check_vars='station_name', exps=exps)

# Station Time-Series (monthly)
def loadCESM_ShpTS(experiment=None, name=None, shape=None, filetypes=None, varlist=None, varatts=None,  
                   translateVars=None, load3D=False, ignore_list=None, lcheckExp=True, lreplaceTime=True,
                   lencl=False, lwrite=False, exps=None):
  ''' Get a properly formatted CESM dataset with a monthly time-series averaged over regions. '''
  return loadCESM_All(experiment=experiment, name=name, grid=None, period=None, shape=shape, lencl=lencl, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, lreplaceTime=lreplaceTime, 
                      translateVars=translateVars, lautoregrid=False, load3D=load3D, station=None, 
                      ignore_list=ignore_list, mode='time-series', lcheckExp=lcheckExp, 
                      lwrite=lwrite, check_vars='shape_name', exps=exps)

# Time-Series (monthly)
def loadCESM_TS(experiment=None, name=None, grid=None, filetypes=None, varlist=None, varatts=None,  
                translateVars=None, lautoregrid=None, load3D=False, ignore_list=None, lcheckExp=True,
                lreplaceTime=True, lwrite=False, exps=None):
  ''' Get a properly formatted CESM dataset with a monthly time-series. (wrapper for loadCESM)'''
  return loadCESM_All(experiment=experiment, name=name, grid=grid, period=None, station=None, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, translateVars=translateVars, 
                      lautoregrid=lautoregrid, load3D=load3D, ignore_list=ignore_list, mode='time-series', 
                      lcheckExp=lcheckExp, lreplaceTime=lreplaceTime, lwrite=lwrite, exps=exps)

# Station Climatologies (monthly)
def loadCESM_Stn(experiment=None, name=None, station=None, period=None, filetypes=None, varlist=None, 
                 varatts=None, translateVars=None, lautoregrid=None, load3D=False, ignore_list=None, 
                 lcheckExp=True, lwrite=False, exps=None):
  ''' Get a properly formatted CESM dataset with the monthly climatology at station locations. '''
  return loadCESM_All(experiment=experiment, name=name, grid=None, period=period, station=station, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, lreplaceTime=False, 
                      translateVars=translateVars, lautoregrid=lautoregrid, load3D=load3D, 
                      ignore_list=ignore_list, mode='climatology', lcheckExp=lcheckExp, 
                      lwrite=lwrite, check_vars='station_name', exps=exps)

# Regional Climatologies (monthly)
def loadCESM_Shp(experiment=None, name=None, shape=None, period=None, filetypes=None, varlist=None, 
                 varatts=None, translateVars=None, lautoregrid=None, load3D=False, ignore_list=None, 
                 lcheckExp=True, lencl=False, lwrite=False, exps=None):
  ''' Get a properly formatted CESM dataset with the monthly climatology averaged over regions. '''
  return loadCESM_All(experiment=experiment, name=name, grid=None, period=period, station=None, 
                      shape=shape, lencl=lencl, filetypes=filetypes, varlist=varlist, varatts=varatts, 
                      lreplaceTime=False, translateVars=translateVars, lautoregrid=lautoregrid, 
                      load3D=load3D, ignore_list=ignore_list, mode='climatology', exps=exps, 
                      lcheckExp=lcheckExp, lwrite=lwrite, check_vars='shape_name')

# load minimally pre-processed CESM climatology files 
def loadCESM(experiment=None, name=None, grid=None, period=None, filetypes=None, varlist=None, 
             varatts=None, translateVars=None, lautoregrid=None, load3D=False, ignore_list=None, 
             lcheckExp=True, lreplaceTime=True, lencl=False, lwrite=False, exps=None):
  ''' Get a properly formatted monthly CESM climatology as NetCDFDataset. '''
  return loadCESM_All(experiment=experiment, name=name, grid=grid, period=period, station=None, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, translateVars=translateVars, 
                      lautoregrid=lautoregrid, load3D=load3D, ignore_list=ignore_list, exps=exps, 
                      mode='climatology', lcheckExp=lcheckExp, lreplaceTime=lreplaceTime, lwrite=lwrite)


# load any of the various pre-processed CESM climatology and time-series files 
def loadCESM_All(experiment=None, name=None, grid=None, station=None, shape=None, period=None, 
                 varlist=None, varatts=None, translateVars=None, lautoregrid=None, load3D=False, 
                 ignore_list=None, mode='climatology', cvdp_mode=None, lcheckExp=True, exps=None,
                 lreplaceTime=True, filetypes=None, lencl=False, lwrite=False, check_vars=None):
  ''' Get any of the monthly CESM files as a properly formatted NetCDFDataset. '''
  # period
  if isinstance(period,(tuple,list)):
    if not all(isNumber(period)): raise ValueError
  elif isinstance(period,str): period = [int(prd) for prd in period.split('-')]
  elif isinstance(period,(int,np.integer)) or period is None : pass # handled later
  else: raise DateError("Illegal period definition: {:s}".format(str(period)))
  # prepare input  
  lclim = False; lts = False; lcvdp = False; ldiag = False # mode switches
  if mode.lower() == 'climatology': # post-processed climatology files
    lclim = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='avg', exps=exps)    
    if period is None: raise DateError('Currently CESM Climatologies have to be loaded with the period explicitly specified.')
  elif mode.lower() in ('time-series','timeseries'): # concatenated time-series files
    lts = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='avg', exps=exps)
    lclim = False; period = None; periodstr = None # to indicate time-series (but for safety, the input must be more explicit)
    if lautoregrid is None: lautoregrid = False # this can take very long!
  elif mode.lower() == 'cvdp': # concatenated time-series files
    lcvdp = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='cvdp', 
                                           cvdp_mode=cvdp_mode, exps=exps)
    if period is None:
      if not isinstance(experiment,Exp): raise DatasetError('Periods can only be inferred for registered datasets.')
      period = (experiment.beginyear, experiment.endyear)  
  elif mode.lower() == 'diag': # concatenated time-series files
    ldiag = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='diag', exps=exps)
    raise NotImplementedError("Loading AMWG diagnostic files is not supported yet.")
  else: raise NotImplementedError("Unsupported mode: '{:s}'".format(mode))  
  # cast/copy varlist
  if isinstance(varlist,str): varlist = [varlist] # cast as list
  elif varlist is not None: varlist = list(varlist) # make copy to avoid interference
  # handle stations and shapes
  if station and shape: raise ArgumentError
  elif station or shape: 
    if lcvdp: raise NotImplementedError('CVDP data is not available as station data.')
    if lautoregrid: raise GDALError('Station data can not be regridded, since it is not map data.')   
    lstation = bool(station); lshape = bool(shape)
    # add station/shape parameters
    if varlist:
      params = stn_params if lstation else shp_params
      for param in params:
        if param not in varlist: varlist.append(param)
  else:
    lstation = False; lshape = False
  # period  
  if isinstance(period,(int,np.integer)):
    if not isinstance(experiment,Exp): raise DatasetError('Integer periods are only supported for registered datasets.')
    period = (experiment.beginyear, experiment.beginyear+period)
  if lclim: periodstr = '_{0:4d}-{1:4d}'.format(*period)
  elif lcvdp: periodstr = '{0:4d}-{1:4d}'.format(period[0],period[1]-1)
  else: periodstr = ''
  # N.B.: the period convention in CVDP is that the end year is included
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = ['atm','lnd']
  elif isinstance(filetypes,(list,tuple,set,str)):
    if isinstance(filetypes,str): filetypes = [filetypes]
    else: filetypes = list(filetypes)
    # interprete/replace WRF filetypes (for convenience)
    tmp = []
    for ft in filetypes:
      if ft in ('const','drydyn3d','moist3d','rad','plev3d','srfc','xtrm','hydro'):
        if 'atm' not in tmp: tmp.append('atm')
      elif ft in ('lsm','snow'):
        if 'lnd' not in tmp: tmp.append('lnd')
      elif ft in ('aux'): pass # currently not supported
#       elif ft in (,):
#         if 'atm' not in tmp: tmp.append('atm')
#         if 'lnd' not in tmp: tmp.append('lnd')        
      else: tmp.append(ft)
    filetypes = tmp; del tmp
    if 'axes' not in filetypes: filetypes.append('axes')    
  else: raise TypeError  
  atts = dict(); filelist = []; typelist = []
  for filetype in filetypes:
    fileclass = fileclasses[filetype] if filetype in fileclasses else FileType(filetype)
    if lclim and fileclass.climfile is not None: filelist.append(fileclass.climfile)
    elif lts and fileclass.tsfile is not None: filelist.append(fileclass.tsfile)
    elif lcvdp and fileclass.cvdpfile is not None: filelist.append(fileclass.cvdpfile)
    elif ldiag and fileclass.diagfile is not None: filelist.append(fileclass.diagfile)
    typelist.append(filetype)
    atts.update(fileclass.atts) 
  # figure out ignore list  
  if ignore_list is None: ignore_list = set(ignore_list_2D)
  elif isinstance(ignore_list,(list,tuple)): ignore_list = set(ignore_list)
  elif not isinstance(ignore_list,set): raise TypeError
  if not load3D: ignore_list.update(ignore_list_3D)
  if lautoregrid is None: lautoregrid = not load3D # don't auto-regrid 3D variables - takes too long!
  # translate varlist
  if varatts is not None: atts.update(varatts)
  lSST = False
  if varlist is not None:
    varlist = list(varlist) 
    if 'SST' in varlist: # special handling of name SST variable, as it is part of Ts
      varlist.remove('SST')
      if not 'Ts' in varlist: varlist.append('Ts')
      lSST = True # Ts is renamed to SST below
    #if translateVars is None: varlist = list(varlist) + translateVarNames(varlist, atts) # also aff translations, just in case
    #elif translateVars is True: varlist = translateVarNames(varlist, atts) 
    # N.B.: DatasetNetCDF does never apply translation!
  # NetCDF file mode
  ncmode = 'rw' if lwrite else 'r'   
  # get grid or station-set name
  if lstation or lshape:
    # the station or shape name can be inserted as the grid name
    if lstation: gridstr = '_'+station.lower(); # only use lower case for filenames
    elif lshape: gridstr = '_'+shape.lower(); # only use lower case for filenames
    griddef = None
    if grid and grid != experiment.grid: gridstr += '_'+grid.lower(); # only use lower case for filenames
  else:
    if grid is None or grid == experiment.grid: 
      gridstr = ''; griddef = None
    else: 
      gridstr = '_'+grid.lower() # only use lower case for filenames
      griddef = loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder, check=True)
  # insert grid name and period
  filenames = []
  for filetype,fileformat in zip(typelist,filelist):
    if lclim: filename = fileformat.format(gridstr,periodstr) # put together specfic filename for climatology
    elif lts: filename = fileformat.format(gridstr) # or for time-series
    elif lcvdp: filename = fileformat.format(experiment.name if experiment else name,periodstr) # not implemented: gridstr
    elif ldiag: raise NotImplementedError
    else: raise DatasetError
    filenames.append(filename) # append to list (passed to DatasetNetCDF later)
    # check existance
    filepath = '{:s}/{:s}'.format(folder,filename)
    if not os.path.exists(filepath):
      nativename = fileformat.format('',periodstr) # original filename (before regridding)
      nativepath = '{:s}/{:s}'.format(folder,nativename)
      if os.path.exists(nativepath):
        if lautoregrid: 
          from processing.regrid import performRegridding # causes circular reference if imported earlier
          griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder)
          dataargs = dict(experiment=experiment, filetypes=[filetype], period=period)
          print(("The '{:s}' (CESM) dataset for the grid ('{:s}') is not available:\n Attempting regridding on-the-fly.".format(name,filename,grid)))
          if performRegridding('CESM','climatology' if lclim else 'time-series', griddef, dataargs): # default kwargs
            raise IOError("Automatic regridding failed!")
          print(("Output: '{:s}'".format(name,filename,grid,filepath)))            
        else: raise IOError("The '{:s}' (CESM) dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.".format(name,filename,grid)) 
      else: raise IOError("The '{:s}' (CESM) dataset file '{:s}' does not exits!\n({:s})".format(name,filename,folder))
   
  # load dataset
  #print varlist, filenames
  if experiment: title = experiment.title
  else: title = name
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=None, 
                          varatts=atts, title=title, multifile=False, ignore_list=ignore_list, 
                          ncformat='NETCDF4', squeeze=True, mode=ncmode, check_vars=check_vars)
  # replace time axis
  if lreplaceTime:
    if lts or lcvdp:
      # check time axis and center at 1979-01 (zero-based)
      if experiment is None: ys = period[0]; ms = 1
      else: ys,ms,ds = [int(t) for t in experiment.begindate.split('-')]; assert ds == 1
      if dataset.hasAxis('time'):
        ts = (ys-1979)*12 + (ms-1); te = ts+len(dataset.time) # month since 1979 (Jan 1979 = 0)
        atts = dict(long_name='Month since 1979-01')
        timeAxis = Axis(name='time', units='month', coord=np.arange(ts,te,1, dtype='int16'), atts=atts)
        dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
      if dataset.hasAxis('year'):
        ts = ys-1979; te = ts+len(dataset.year) # month since 1979 (Jan 1979 = 0)
        atts = dict(long_name='Years since 1979-01')
        yearAxis = Axis(name='year', units='year', coord=np.arange(ts,te,1, dtype='int16'), atts=atts)
        dataset.replaceAxis(dataset.year, yearAxis, asNC=False, deepcopy=False)
    elif lclim:
      if dataset.hasAxis('time') and not dataset.time.units.lower() in monthlyUnitsList:
        atts = dict(long_name='Month of the Year')
        timeAxis = Axis(name='time', units='month', coord=np.arange(1,13, dtype='int16'), atts=atts)
        assert len(dataset.time) == len(timeAxis), dataset.time
        dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
      elif dataset.hasAxis('year'): raise NotImplementedError(dataset)
  # rename SST
  if lSST: dataset['SST'] = dataset.Ts
  # correct ordinal number of shape (should start at 1, not 0)
  if lshape:
    # mask all shapes that are incomplete in dataset
    if lencl and 'shp_encl' in dataset: dataset.mask(mask='shp_encl', invert=True)   
    if dataset.hasAxis('shapes'): raise AxisError("Axis 'shapes' should be renamed to 'shape'!")
    if not dataset.hasAxis('shape'): raise AxisError
    if dataset.shape.coord[0] == 0: dataset.shape.coord += 1
  # check
  if len(dataset) == 0: raise DatasetError('Dataset is empty - check source file or variable list!')
  # add projection, if applicable
  if not ( lstation or lshape ):
    dataset = addGDALtoDataset(dataset, griddef=griddef, gridfolder=grid_folder, lwrap360=True, geolocator=True)
  # return formatted dataset
  return dataset

# load a pre-processed CESM ensemble and concatenate time-series (also for CVDP) 
def loadCESM_ShpEns(ensemble=None, name=None, shape=None, filetypes=None, years=None,
                    varlist=None, varatts=None, translateVars=None, load3D=False, lcheckVars=None, 
                    ignore_list=None, lcheckExp=True, lencl=False, axis=None, lensembleAxis=False,
                    exps=None, enses=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadCESM_Ensemble(ensemble=ensemble, name=name, grid=None, station=None, shape=shape, 
                           filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, 
                           translateVars=translateVars, lautoregrid=False, load3D=load3D, 
                           ignore_list=ignore_list, cvdp_mode='ensemble', lcheckExp=lcheckExp, 
                           mode='time-series', lreplaceTime=True, lencl=lencl, lcheckVars=lcheckVars,
                           axis=axis, lensembleAxis=lensembleAxis, check_vars='shape_name',
                           exps=exps, enses=enses)

# load a pre-processed CESM ensemble and concatenate time-series (also for CVDP) 
def loadCESM_StnEns(ensemble=None, name=None, station=None, filetypes=None, years=None,
                    varlist=None, varatts=None, translateVars=None, load3D=False, lcheckVars=None, 
                    ignore_list=None, lcheckExp=True, axis=None, lensembleAxis=False,
                    exps=None, enses=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  return loadCESM_Ensemble(ensemble=ensemble, name=name, grid=None, station=station, shape=None,
                           filetypes=filetypes, years=years, varlist=varlist, varatts=varatts, 
                           translateVars=translateVars, lautoregrid=False, load3D=load3D, 
                           ignore_list=ignore_list, cvdp_mode='ensemble', lcheckExp=lcheckExp, 
                           mode='time-series', lreplaceTime=True, lcheckVars=lcheckVars, axis=axis, 
                           lensembleAxis=lensembleAxis, check_vars='station_name',
                           exps=exps, enses=enses)

  
# load a pre-processed CESM ensemble and concatenate time-series (also for CVDP) 
def loadCESM_Ensemble(ensemble=None, name=None, title=None, grid=None, station=None, shape=None, 
                      years=None, varlist=None, varatts=None, translateVars=None, lautoregrid=None, 
                      load3D=False, ignore_list=None, cvdp_mode='ensemble', lcheckExp=True, lencl=False, 
                      mode='time-series', lindices=False, leofs=False, filetypes=None, lreplaceTime=True,
                      axis=None, lensembleAxis=False, lcheckVars=None, check_vars=None,
                      exps=None, enses=None):
  ''' A function to load all datasets in an ensemble and concatenate them along the time axis. '''
  # obviously this only works for modes that produce a time-axis
  if mode.lower() not in ('time-series','timeseries','cvdp'): 
    raise ArgumentError("Concatenated ensembles can not be constructed in mode '{:s}'".format(mode)) 
  # figure out ensemble
  if ensemble is None and name is not None: 
    ensemble = name; name = None # just switch
  if isinstance(ensemble,(tuple,list)):
    if not all([isinstance(exp,(str,Exp)) for exp in ensemble]): 
      raise TypeError    
    ensemble = [ens if isinstance(ens,Exp) else exps[ens] for ens in ensemble] # convert to Exp's    
    # annotation
    if name is None: name = ensemble[0].shortname
    if title is None: title = ensemble[0].title
  else:
    if isinstance(ensemble,Exp): 
      # annotation (while ensemble is an Exp instance)
      if name is None: name = ensemble.shortname
      if title is None: title = ensemble.title              
      ensemble = enses[ensemble.shortname]
    elif isinstance(ensemble,str): 
      if name is None: name = ensemble
      if title is None: title = ensemble                    
      # convert name to actual ensemble object
      if not isinstance(enses,dict): raise DatasetError('No dictionary of Exp instances specified.')
      if ensemble in enses: ensemble = enses[ensemble]
      else: raise TypeError(ensemble)
    else: raise TypeError
  # figure out time period
  if years is None: years =15; yrtpl = (0,15)
  elif isInt(years): yrtpl = (0,years)
  elif isinstance(years,(list,tuple)) and len(years)==2: raise NotImplementedError 
  else: raise TypeError  
  montpl = (0,years*12)
  # load datasets (and load!)
  datasets = []
  if mode.lower() in ('time-series','timeseries'): lts = True; lcvdp = False
  elif mode.lower() == 'cvdp': lts = False; lcvdp = True
  for exp in ensemble:
    if lts:
      ds = loadCESM_All(experiment=exp, name=None, grid=grid, station=station, shape=shape, 
                        varlist=varlist, varatts=varatts, translateVars=translateVars, period=None,
                        lautoregrid=lautoregrid, load3D=load3D, ignore_list=ignore_list, 
                        filetypes=filetypes, lencl=lencl, mode=mode, cvdp_mode='', 
                        lcheckExp=lcheckExp, lreplaceTime=lreplaceTime, check_vars=check_vars).load()
    elif lcvdp:
      ds = loadCVDP(experiment=exp, name=None, varlist=varlist, varatts=varatts, period=years, 
                    translateVars=translateVars, lautoregrid=lautoregrid, #lencl=lencl, check_vars=check_vars, 
                    ignore_list=ignore_list, cvdp_mode=cvdp_mode, lcheckExp=lcheckExp, leofs=leofs, 
                    lindices=lindices, lreplaceTime=lreplaceTime).load()
    else: raise NotImplementedError
    if montpl or yrtpl: ds = ds(year=yrtpl, time=montpl, lidx=True) # slice the time dimension to make things consistent
    datasets.append(ds)
  # harmonize axes (this will usually not be necessary for CESM, since the grids are all the same)
  for axname,ax in ds.axes.items():
    if not all([dataset.hasAxis(axname) for dataset in datasets]): 
      raise AxisError("Not all datasets have Axis '{:s}'.".format(axname))
    if not all([len(dataset.axes[axname]) == len(ax) for dataset in datasets]):
      datasets = selectElements(datasets, axis=axname, testFct=None, imaster=None, linplace=False, lall=True)
  # concatenate datasets (along 'time' and 'year' axis!)  
  if axis is None:
    if lensembleAxis: axis = 'ensemble' 
    elif lts: axis='time'
    elif lcvdp: axis=('time','year')
    else: raise NotImplementedError
  if lcheckVars is None: lcheckVars = bool(varlist)
  dataset = concatDatasets(datasets, axis=axis, coordlim=None, name=name, title=title, idxlim=None, 
                           lensembleAxis=lensembleAxis, offset=None, axatts=None, lcpOther=True, 
                           lcpAny=False, check_vars=check_vars, lcheckVars=lcheckVars)
  # return concatenated dataset
  return dataset

## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
avgfolder # root folder for monthly averages
outfolder # root folder for direct WRF output
ts_file_pattern = 'cesm{0:s}{1:s}_monthly.nc' # filename pattern: filetype, grid
clim_file_pattern = 'cesm{0:s}{1:s}_clim{2:s}.nc' # filename pattern: filetype, grid, period
data_folder = root_folder # folder for user data
grid_def = {'':None} # there are too many... 
grid_res = {'':1.} # approximate grid resolution at 45 degrees latitude
default_grid = None 
# functions to access specific datasets
loadLongTermMean = None # WRF doesn't have that...
loadClimatology = loadCESM # pre-processed, standardized climatology
loadTimeSeries = loadCESM_TS # time-series data
loadStationClimatology = loadCESM_Stn # pre-processed, standardized climatology at stations
loadStationTimeSeries = loadCESM_StnTS # time-series data at stations
loadShapeClimatology = loadCESM_Shp # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries = loadCESM_ShpTS # time-series without associated grid (e.g. provinces or basins)


## (ab)use main execution for quick test
if __name__ == '__main__':
  
  # set mode/parameters
#   mode = 'test_climatology'
#   mode = 'test_timeseries'
#   mode = 'test_ensemble'
#   mode = 'test_point_climatology'
#   mode = 'test_point_timeseries'
#   mode = 'test_point_ensemble'
#   mode = 'test_cvdp'
  mode = 'pickle_grid'
#     mode = 'shift_lon'
#   experiments = ['Ctrl-1', 'Ctrl-A', 'Ctrl-B', 'Ctrl-C']
#   experiments += ['Ctrl-2050', 'Ctrl-A-2050', 'Ctrl-B-2050', 'Ctrl-C-2050']
  experiments = ('Ctrl-1',)
  periods = (15,)
  filetypes = ('atm',) # ['atm','lnd','ice']
  grids = ('cesm1x1',)*len(experiments) # grb1_d01
#   pntset = 'shpavg'
  pntset = 'ecprecip'

  from projects.CESM_experiments import Exp, CESM_exps, ensembles
  # N.B.: importing Exp through CESM_experiments is necessary, otherwise some isinstance() calls fail

  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,experiment in zip(grids,experiments):
      
      print('')
      print(('   ***   Pickling Grid Definition for {0:s}   ***   '.format(grid)))
      print('')
      
      # load GridDefinition
      dataset = loadCESM(experiment=CESM_exps[experiment], grid=None, filetypes=['lnd'], period=(1979,1989))
      griddef = dataset.griddef
      #del griddef.xlon, griddef.ylat      
      print(griddef)
      griddef.name = grid
      print(('   Loading Definition from \'{0:s}\''.format(dataset.name)))
      # save pickle
      filepath = pickleGridDef(griddef, lfeedback=True, loverwrite=True, lgzip=True)
      
      print(('   Saving Pickle to \'{0:s}\''.format(filepath)))
      print('')
      
      # load pickle to make sure it is right
      del griddef
      griddef = loadPickledGridDef(grid, res=None, folder=grid_folder)
      print(griddef)
      print('')
      print(griddef.wrap360)
      
  # load ensemble "time-series"
  elif mode == 'test_ensemble':
    
    print('')
#     dataset = loadCESM_Ensemble(ensemble='Ens-2050', varlist=['precip'], filetypes=['atm'])
    dataset = loadCESM_Ensemble(ensemble='Ens-2050', mode='cvdp', exps=CESM_exps, enses=ensembles)
    print('')
    print(dataset)
    print('')
    print((dataset.year))
    print((dataset.year.coord))
  
  # load station climatology file
  elif mode == 'test_point_climatology':
    
    print('')
    if pntset in ('shpavg',):
      dataset = loadCESM_Shp(experiment='Ctrl-1', shape=pntset, filetypes=['atm'], period=(1979,1994),
                             exps=CESM_exps)
      print('')
      print(dataset)
      print('')
      print((dataset.shape))
      print((dataset.shape.coord))
      assert dataset.shape.coord[-1] == len(dataset.shape)  # this is a global model!    
    else:
      dataset = loadCESM_Stn(experiment='Ctrl-1', station=pntset, filetypes=['atm'], period=(1979,1994),
                             exps=CESM_exps)
      print('')
      print(dataset)
      print('')
      print((dataset.station))
      print((dataset.station.coord))
      assert dataset.station.coord[-1] == len(dataset.station)  # this is a global model!
    
  # load station time-series file
  elif mode == 'test_point_timeseries':    
    print('')
    if pntset in ('shpavg',):
      dataset = loadCESM_ShpTS(experiment='Ctrl-1', shape=pntset, filetypes=['atm'], exps=CESM_exps, 
                               lwrite=True)
    else:
      dataset = loadCESM_StnTS(experiment='Ctrl-1', station=pntset, filetypes=['atm'], lwrite=True,
                               exps=CESM_exps)
    print('')
    print(dataset)
    assert 'w' in dataset.mode
    dataset.sync()
    print('')
    print((dataset.time))
    print((dataset.time.coord))
    
  # load station ensemble "time-series"
  elif mode == 'test_point_ensemble':
    
    lensembleAxis = False
    variable = 'MaxPrecip_1d'
    print('')
    if pntset in ('shpavg',):
      dataset = loadCESM_ShpEns(ensemble='Ens', shape=pntset, filetypes=['atm'], 
                                lensembleAxis=lensembleAxis, varlist=[variable], 
                                exps=CESM_exps, enses=ensembles)
    else:
      dataset = loadCESM_StnEns(name='Ens', station=pntset, filetypes=['hydro'], 
                                lensembleAxis=lensembleAxis, varlist=[variable], 
                                exps=CESM_exps, enses=ensembles)
    print('')
    print(dataset)
    assert dataset.name == 'Ens'
    assert not lensembleAxis or dataset.hasAxis('ensemble') 
    print('')
    print((dataset.time))
    print((dataset.time.coord))
    print('')
    print((dataset[variable]))
    print((dataset[variable].mean()))
  
  # load averaged climatology file
  elif mode == 'test_climatology' or mode == 'test_timeseries':
    
    for grid,experiment in zip(grids,experiments):
      
      print('')
      if mode == 'test_timeseries':
        dataset = loadCESM_TS(experiment=experiment, varlist=None, grid=grid, filetypes=filetypes, 
                              exps=CESM_exps)
      else:
        period = periods[0] # just use first element, no need to loop
        dataset = loadCESM(experiment=experiment, varlist=['precip'], grid=grid, filetypes=filetypes, 
                           period=period, exps=CESM_exps)
      print(dataset)
      print('')
      print((dataset.geotransform))
      print('')
      print(dataset.precip)
      print(dataset.precip.mean()*86400, dataset.precip.std()*86400)

#       if dataset.isProjected:
#         print dataset.x
#         print dataset.x.coord
#       else:
#         print dataset.lon
#         print dataset.lon.coord
#       print('')      
#       print(dataset.time)
#       print(dataset.time.coord)
      # show some variables
#       if 'zs' in dataset: var = dataset.zs
#       elif 'hgt' in dataset: var = dataset.hgt
#       else: var = dataset.lon2D
#       var.load()
#       print var
#       var = var.mean(axis='time',checkAxis=False)
      # display
#       import pylab as pyl
#       pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), dataset.precip.getArray().mean(axis=0))
#       pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), dataset.runoff.getArray().mean(axis=0))
#       pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), var.getArray())
#       pyl.colorbar()
#       pyl.show(block=True)
  
  # load CVDP file
  elif mode == 'test_cvdp':
    
    for grid,experiment in zip(grids,experiments):
      
      print('')
      period = periods[0] # just use first element, no need to loop
      dataset = loadCVDP(experiment=experiment, period=period, cvdp_mode='ensemble', 
                         exps=CESM_exps) # lindices=True
      #dataset = loadCVDP_Obs(name='GPCP')
      print(dataset)
#       print(dataset.geotransform)
      print((dataset.year))
      print((dataset.year.coord))
      # print some variables
#       print('')
#       eof = dataset.pdo_pattern; eof.load()
# #       print eof
#       print('')
#       ts = dataset.pdo_timeseries; ts.load()
# #       print ts
#       print ts.mean()
      # display
#       import pylab as pyl
#       pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), dataset.precip.getArray().mean(axis=0))
#       pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), dataset.runoff.getArray().mean(axis=0))
#       pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), eof.getArray())
#       pyl.colorbar()
#       pyl.show(block=True)
      print('')
  
  # shift dataset from 0-360 to -180-180
  elif mode == 'shift_lon':
   
    # loop over periods
    for prdlen in periods: # (15,): # 
      # loop over experiments
      for experiment in experiments: # ('CESM',): #  
        # loop over filetypes
        for filetype in filetypes: # ('lnd',): #  
          fileclass = fileclasses[filetype]
          
          # load source
          exp = CESM_exps[experiment]
          period = (exp.beginyear, exp.beginyear+prdlen)
          periodstr = '{0:4d}-{1:4d}'.format(*period)
          print('\n')
          print(('   ***   Processing Experiment {0:s} for Period {1:s}   ***   '.format(exp.title,periodstr)))
          print('\n')
          # prepare file names
          filename = fileclass.climfile.format('','_'+periodstr)
          origname = 'orig'+filename[4:]; tmpname = 'tmp.nc'
          filepath = exp.avgfolder+filename; origpath = exp.avgfolder+origname; tmppath = exp.avgfolder+tmpname
          # load source
          if os.path.exists(origpath) and os.path.exists(filepath): 
            os.remove(filepath) # overwrite old file
            os.rename(origpath,filepath) # get original source
          source = loadCESM(experiment=exp, period=period, filetypes=[filetype])
          print(source)
          print('\n')
          # savety checks
          if os.path.exists(origpath): raise IOError
          if np.max(source.lon.getArray()) < 180.: raise AxisError
          if not os.path.exists(filepath): raise IOError
          # prepare sink
          if os.path.exists(tmppath): os.remove(tmppath)
          sink = DatasetNetCDF(name=None, folder=exp.avgfolder, filelist=[tmpname], atts=source.atts, mode='w')
          sink.atts.period = periodstr 
          sink.atts.name = exp.name
          
          # initialize processing
          CPU = CentralProcessingUnit(source, sink, tmp=False)
          
          # shift longitude axis by 180 degrees left (i.e. 0 - 360 -> -180 - 180)
          CPU.Shift(lon=-180, flush=True)
          
          # sync temporary storage with output
          CPU.sync(flush=True)
              
          # add new variables
          # liquid precip (atmosphere file)
          if sink.hasVariable('precip') and sink.hasVariable('solprec'):
            data = sink.precip.getArray() - sink.solprec.getArray()
            Var = Variable(axes=sink.precip.axes, name='liqprec', data=data, atts=default_varatts['liqprec'])            
            sink.addVariable(Var, asNC=True) # create variable and add to dataset
          # net precip (atmosphere file)
          if sink.hasVariable('precip') and sink.hasVariable('evap'):
            data = sink.precip.getArray() - sink.evap.getArray()
            Var = Variable(axes=sink.precip.axes, name='p-et', data=data, atts=default_varatts['p-et'])
            sink.addVariable(Var, asNC=True) # create variable and add to dataset      
          # underground runoff (land file)
          if sink.hasVariable('runoff') and sink.hasVariable('sfroff'):
            data = sink.runoff.getArray() - sink.sfroff.getArray()
            Var = Variable(axes=sink.runoff.axes, name='ugroff', data=data, atts=default_varatts['ugroff'])
            sink.addVariable(Var, asNC=True) # create variable and add to dataset    
    
          # add length and names of month
          if sink.hasAxis('time', strict=False):
            addLengthAndNamesOfMonth(sink, noleap=True)     
          # close...
          sink.sync()
          sink.close()
          
          # move files
          os.rename(filepath, origpath)
          os.rename(tmppath,filepath)
          
          # print dataset
          print('')
          print(sink)               
