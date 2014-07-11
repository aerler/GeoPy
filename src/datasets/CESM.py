'''
Created on 2013-12-04

This module contains common meta data and access functions for CESM model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os, pickle
# from atmdyn.properties import variablePlotatts
from geodata.base import Variable
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset
from geodata.misc import DatasetError, AxisError, DateError
from datasets.common import translateVarNames, data_root, grid_folder, default_varatts, addLengthAndNamesOfMonth 
from geodata.gdal import loadPickledGridDef, griddef_pickle
from projects.WRF_experiments import Exp
from processing.process import CentralProcessingUnit

# some meta data (needed for defaults)
root_folder = data_root + 'CESM/' # long-term mean folder
outfolder = root_folder + 'cesmout/' # WRF output folder
avgfolder = root_folder + 'cesmavg/' # long-term mean folder

## list of experiments
# N.B.: This is the reference list, with unambiguous, unique keys and no aliases/duplicate entries  
experiments = dict() # dictionary of experiments
Exp.defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/'.format(avgfolder,atts['name'])
Exp.defaults['parents'] = None # not applicable here
# list of experiments
# historical
experiments['ens20trcn1x1'] = Exp(shortname='CESM', name='ens20trcn1x1', title='CESM Ensemble Mean', begindate='1979-01-01', enddate='1995-01-01', grid='cesm1x1')
experiments['tb20trcn1x1'] = Exp(shortname='Ctrl', name='tb20trcn1x1', title='Ctrl (CESM)', begindate='1979-01-01', enddate='1995-01-01', grid='cesm1x1')
experiments['hab20trcn1x1'] = Exp(shortname='Ens-A', name='hab20trcn1x1', title='Ens-A (CESM)', begindate='1979-01-01', enddate='1995-01-01', grid='cesm1x1')
experiments['hbb20trcn1x1'] = Exp(shortname='Ens-B', name='hbb20trcn1x1', title='Ens-B (CESM)', begindate='1979-01-01', enddate='1995-01-01', grid='cesm1x1')
experiments['hcb20trcn1x1'] = Exp(shortname='Ens-C', name='hcb20trcn1x1', title='Ens-C (CESM)', begindate='1979-01-01', enddate='1995-01-01', grid='cesm1x1')
# mid-21st century
experiments['ensrcp85cn1x1'] = Exp(shortname='CESM-2050', name='ensrcp85cn1x1', title='CESM Ensemble Mean (2050)', begindate='2045-01-01', enddate='2060-01-01', grid='cesm1x1')
experiments['seaice-5r-hf'] = Exp(shortname='Seaice-2050', name='seaice-5r-hf', title='Seaice (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', grid='cesm1x1')
experiments['htbrcp85cn1x1'] = Exp(shortname='Ctrl-2050', name='htbrcp85cn1x1', title='Ctrl (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', grid='cesm1x1')
experiments['habrcp85cn1x1'] = Exp(shortname='Ens-A-2050', name='habrcp85cn1x1', title='Ens-A (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', grid='cesm1x1')
experiments['hbbrcp85cn1x1'] = Exp(shortname='Ens-B-2050', name='hbbrcp85cn1x1', title='Ens-B (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', grid='cesm1x1')
experiments['hcbrcp85cn1x1'] = Exp(shortname='Ens-C-2050', name='hcbrcp85cn1x1', title='Ens-C (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', grid='cesm1x1')
# mid-21st century
experiments['ensrcp85cn1x1d'] = Exp(shortname='CESM-2100', name='ensrcp85cn1x1d', title='CESM Ensemble Mean (2100)', begindate='2085-01-01', enddate='2100-01-01', grid='cesm1x1')
experiments['seaice-5r-hfd'] = Exp(shortname='Seaice-2100', name='seaice-5r-hf', title='Seaice (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', grid='cesm1x1')
experiments['htbrcp85cn1x1d'] = Exp(shortname='Ctrl-2100', name='htbrcp85cn1x1d', title='Ctrl (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', grid='cesm1x1')
experiments['habrcp85cn1x1d'] = Exp(shortname='Ens-A-2100', name='habrcp85cn1x1d', title='Ens-A (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', grid='cesm1x1')
experiments['hbbrcp85cn1x1d'] = Exp(shortname='Ens-B-2100', name='hbbrcp85cn1x1d', title='Ens-B (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', grid='cesm1x1')
experiments['hcbrcp85cn1x1d'] = Exp(shortname='Ens-C-2100', name='hcbrcp85cn1x1d', title='Ens-C (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', grid='cesm1x1')

## an alternate dictionary using short names and aliases for referencing
exps = dict()
# use short names where availalbe, normal names otherwise
for key,item in experiments.iteritems():
  exps[item.name] = item
  if item.shortname is not None: 
    exps[item.shortname] = item
  # both, short and long name are added to list
# add aliases here
CESM_exps = exps # alias for whole dict
CESM_experiments = experiments # alias for whole dict


# return name and folder
def getFolderName(name=None, experiment=None, folder=None):
  ''' Convenience function to infer and type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  # figure out experiment name
  if experiment is None:
    if not isinstance(folder,basestring): 
      raise IOError, "Need to specify an experiment folder in order to load data."    
    # load experiment meta data
    if name in exps: experiment = exps[name]
    else: raise DatasetError, 'Dataset of name \'{0:s}\' not found!'.format(name)
  else:
    if isinstance(experiment,(Exp,basestring)):
      if isinstance(experiment,basestring): experiment = exps[experiment] 
      # root folder
      if folder is None: folder = experiment.avgfolder
      elif not isinstance(folder,basestring): raise TypeError
      # name
      if name is None: name = experiment.name
    if not isinstance(name,basestring): raise TypeError      
  # check if folder exists
  if not os.path.exists(folder): raise IOError, 'Dataset folder does not exist: {0:s}'.format(folder)
  # return name and folder
  return folder, experiment, name


## variable attributes and name
class FileType(object): pass # ''' Container class for all attributes of of the constants files. '''
# surface variables
class ATM(FileType):
  ''' Variables and attributes of the surface files. '''
  def __init__(self):
    self.atts = dict(TREFHT   = dict(name='T2', units='K'), # 2m Temperature
                     QREFHT   = dict(name='q2', units='kg/kg'), # 2m water vapor mass mixing ratio                     
                     TS       = dict(name='Ts', units='K'), # Skin Temperature (SST)
                     TSMN     = dict(name='Tmin', units='K'),   # Minimum Temperature (at surface)
                     TSMX     = dict(name='Tmax', units='K'),   # Maximum Temperature (at surface)                     
                     PRECT    = dict(name='precip', units='kg/m^2/s', scalefactor=1000.), # total precipitation rate (kg/m^2/s)
                     PRECC    = dict(name='preccu', units='kg/m^2/s', scalefactor=1000.), # convective precipitation rate (kg/m^2/s)
                     PRECL    = dict(name='precnc', units='kg/m^2/s', scalefactor=1000.), # grid-scale precipitation rate (kg/m^2/s)
                     #NetPrecip    = dict(name='p-et', units='kg/m^2/s'), # net precipitation rate
                     #LiquidPrecip = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation rate
                     PRECSL   = dict(name='solprec', units='kg/m^2/s', scalefactor=1000.), # solid precipitation rate
                     #SNOWLND   = dict(name='snow', units='kg/m^2'), # snow water equivalent
                     SNOWHLND = dict(name='snowh', units='m'), # snow depth
                     SNOWHICE = dict(name='snowhice', units='m'), # snow depth
                     ICEFRAC  = dict(name='seaice', units=''), # seaice fraction
                     SHFLX    = dict(name='hfx', units='W/m^2'), # surface sensible heat flux
                     LHFLX    = dict(name='lhfx', units='W/m^2'), # surface latent heat flux
                     QFLX     = dict(name='evap', units='kg/m^2/s'), # surface evaporation
                     FLUT     = dict(name='OLR', units='W/m^2'), # Outgoing Longwave Radiation
                     FLDS     = dict(name='GLW', units='W/m^2'), # Ground Longwave Radiation
                     FSDS     = dict(name='SWD', units='W/m^2'), # Downwelling Shortwave Radiation                     
                     PS       = dict(name='ps', units='Pa'), # surface pressure
                     PSL      = dict(name='pmsl', units='Pa'), # mean sea level pressure
                     PHIS     = dict(name='zs', units='m', scalefactor=1./9.81), # surface elevation
                     #LANDFRAC = dict(name='landfrac', units=''), # land fraction
                     )
    self.vars = self.atts.keys()    
    self.climfile = 'cesmatm{0:s}_clim{1:s}.nc' # the filename needs to be extended by ('_'+grid,'_'+period)
    self.tsfile = NotImplemented # native CESM output
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
                     )
    self.vars = self.atts.keys()    
    self.climfile = 'cesmlnd{0:s}_clim{1:s}.nc' # the filename needs to be extended by ('_'+grid,'_'+period)
    self.tsfile = NotImplemented # native CESM output
# CICE variables
class ICE(FileType):
  ''' Variables and attributes of the seaice files. '''
  def __init__(self):
    self.atts = dict() # currently not implemented...                     
    self.vars = self.atts.keys()
    self.climfile = 'cesmice{0:s}_clim{1:s}.nc' # the filename needs to be extended by ('_'+grid,'_'+period)
    self.tsfile = NotImplemented # native CESM output

# axes (don't have their own file)
class Axes(FileType):
  ''' A mock-filetype for axes. '''
  def __init__(self):
    self.atts = dict(time        = dict(name='time', units='month'), # time coordinate
                     # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                     #       the time offset is chose such that 1979 begins with the origin (time=0)
                     lon           = dict(name='lon', units='deg E'), # west-east coordinate
                     lat           = dict(name='lat', units='deg N'), # south-north coordinate
                     levgrnd = dict(name='s', units=''), # soil layers
                     lev = dict(name='lev', units='')) # hybrid pressure coordinate
    self.vars = self.atts.keys()
    self.climfile = None
    self.tsfile = None

# data source/location
fileclasses = dict(atm=ATM(), lnd=LND(), axes=Axes()) # ice=ICE() is currently not supported because of the grid


## Functions to load different types of WRF datasets

# Time-Series (monthly)
def loadCESM_TS(experiment=None, name=None, filetypes=None, varlist=None, varatts=None):
  ''' Get a properly formatted CESM dataset with monthly time-series. '''
  raise NotImplementedError


# pre-processed climatology files (varatts etc. should not be necessary) 
def loadCESM(experiment=None, name=None, grid=None, period=None, filetypes=None, varlist=None, 
            varatts=None, loadAll=False, translateVars=None, lautoregrid=False):
  ''' Get a properly formatted monthly CESM climatology as NetCDFDataset. '''
  # prepare input  
  folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None)
  # N.B.: 'experiment' can be a string name or an Exp instance
  # period  
  if isinstance(period,(tuple,list)): pass
  elif isinstance(period,basestring): pass
  elif period is None: pass
  elif isinstance(period,(int,np.integer)) and isinstance(experiment,Exp):
    period = (experiment.beginyear, experiment.beginyear+period)
  else: raise DateError   
  if period is None or period == '': 
    raise DateError, 'Currently CESM Climatologies have to be loaded with the period explicitly specified.'
  elif isinstance(period,basestring): periodstr = '_{0:s}'.format(period)
  else: periodstr = '_{0:4d}-{1:4d}'.format(*period)  
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = fileclasses.keys()
  elif isinstance(filetypes,(list,tuple,set)):
    filetypes = list(filetypes)  
    if 'axes' not in filetypes: filetypes.append('axes')    
  else: raise TypeError  
  atts = dict(); filelist = []; typelist = []
  for filetype in filetypes:
    fileclass = fileclasses[filetype]
    if fileclass.climfile is not None: # this eliminates const files
      filelist.append(fileclass.climfile)
      typelist.append(filetype)
    atts.update(fileclass.atts) 
  if varatts is not None: atts.update(varatts)  
  # translate varlist
  if varlist is None and not loadAll: varlist = atts.keys() # default varlist
  elif varlist is not None:
    if translateVars is None: varlist = list(varlist) + translateVarNames(varlist, atts) # also aff translations, just in case
    elif translateVars is True: varlist = translateVarNames(varlist, atts) 
    # N.B.: DatasetNetCDF does never apply translation!
  # get grid name
  if grid is None or grid == experiment.grid: 
    gridstr = ''; griddef = None
  else: 
    gridstr = '_%s'%grid.lower() # only use lower case for filenames
    griddef = loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder, check=True)
  # insert grid name and period
  filenames = []
  for filetype,fileformat in zip(typelist,filelist):
    filename = fileformat.format(gridstr,periodstr) # put together specfic filename
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
          if performRegridding('CESM', griddef, dataargs): # default kwargs
            raise IOError, "Automatic regridding failed!"
        else: raise IOError, "The CESM dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.".format(filename,grid) 
      else: raise IOError, "The CESM dataset file '{:s}' does not exits!".format(filename)
   
  # load dataset
  #print varlist, filenames
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=None, 
                          varatts=atts, multifile=False, ncformat='NETCDF4', squeeze=True)
  # check
  if len(dataset) == 0: raise DatasetError, 'Dataset is empty - check source file or variable list!'
  # add projection
  dataset = addGDALtoDataset(dataset, griddef=griddef, gridfolder=grid_folder, geolocator=True)
  # return formatted dataset
  return dataset

## Dataset API

dataset_name = 'CESM' # dataset name
root_folder # root folder of the dataset
avgfolder # root folder for monthly averages
outfolder # root folder for direct WRF output
ts_file_pattern = 'cesm{0:s}{1:s}_monthly.nc' # filename pattern: filetype, grid, period
clim_file_pattern = 'cesm{0:s}{1:s}_clim{2:s}.nc' # filename pattern: filetype, grid, period
data_folder = root_folder # folder for user data
grid_def = {'':None} # there are too many... 
grid_res = {'':1.} # approximate grid resolution at 45 degrees latitude
default_grid = None 
# functions to access specific datasets
loadLongTermMean = None # WRF doesn't have that...
loadTimeSeries = loadCESM_TS # time-series data
loadClimatology = loadCESM # pre-processed, standardized climatology


## (ab)use main execution for quick test
if __name__ == '__main__':

  mode = 'test_climatology'
#   mode = 'pickle_grid'
#   mode = 'shift_lon'
  filetypes = ['atm','lnd',]
  grids = ['arb2_d02']; experiments = ['CESM'] # grb1_d01

  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,experiment in zip(grids,experiments):
      
      print('')
      print('   ***   Pickling Grid Definition for {0:s}   ***   '.format(grid))
      print('')
      
      # load GridDefinition
      dataset = loadCESM(experiment=experiment, grid=None, filetypes=['lnd'], period=(1979,1989))
      griddef = dataset.griddef
      #del griddef.xlon, griddef.ylat      
      print griddef
      griddef.name = grid
      print('   Loading Definition from \'{0:s}\''.format(dataset.name))
      # save pickle
      filename = '{0:s}/{1:s}'.format(grid_folder,griddef_pickle.format(grid))
      if os.path.exists(filename): os.remove(filename) # overwrite
      filehandle = open(filename, 'w')
      pickle.dump(griddef, filehandle)
      filehandle.close()
      
      print('   Saving Pickle to \'{0:s}\''.format(filename))
      print('')
      
      # load pickle to make sure it is right
      del griddef
      griddef = loadPickledGridDef(grid, res=None, folder=grid_folder)
      print(griddef)
      print('')
    
  # load averaged climatology file
  elif mode == 'test_climatology':
    
    for grid,experiment in zip(grids,experiments):
      
      print('')
      dataset = loadCESM(experiment=experiment, varlist=['zs'], grid=grid, filetypes=['atm',], 
                         period=(1979,1984),lautoregrid=True) # ['atm','lnd','ice']
      print(dataset)
      dataset.lon2D.load()
      #     # display
      import pylab as pyl
  #     pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), dataset.precip.getArray().mean(axis=0))
      pyl.pcolormesh(dataset.lon2D.getArray(), dataset.lat2D.getArray(), dataset.zs.getArray())
      pyl.colorbar()
      pyl.show(block=True)
      print('')
      print(dataset.geotransform)
  
  # shift dataset from 0-360 to -180-180
  elif mode == 'shift_lon':

    prdlen = 5    
    experiments = ['Ctrl', 'Ens-A', 'Ens-B', 'Ens-C']
#     experiments = CESM_experiments.keys()
    
    # loop over experiments
    for experiment in experiments:
      # loop over filetypes
      for filetype in filetypes: # ['lnd'] 
        fileclass = fileclasses[filetype]
        
        # load source
        exp = CESM_exps[experiment]
        period = (exp.beginyear, exp.beginyear+prdlen)
        periodstr = '{0:4d}-{1:4d}'.format(*period)
        print('\n')
        print('   ***   Processing Experiment {0:s} for Period {1:s}   ***   '.format(exp.title,periodstr))
        print('\n')
        # prepare file names
        filename = fileclass.climfile.format('','_'+periodstr)
        origname = 'orig'+filename[4:]; tmpname = 'tmp.nc'
        filepath = exp.avgfolder+filename; origpath = exp.avgfolder+origname; tmppath = exp.avgfolder+tmpname
        # load source
        if os.path.exists(origpath) and os.path.exists(filepath): 
          os.remove(filepath) # overwrite old file
          os.rename(origpath,filepath) # get original source
        source = loadCESM(experiment=exp, period=period, filetypes=[filetype], loadAll=False)
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
  
        # make new masks
  #       if sink.hasVariable('landfrac'):
  #         # create variable and add to dataset
  #         dataset.addVariable(Variable(axes=axes, name=maskname, data=mask, atts=atts), asNC=True)
          
        # add new variables
        # liquid precip
        if sink.hasVariable('precip') and sink.hasVariable('solprec'):
          data = sink.precip.getArray() - sink.solprec.getArray()
          Var = Variable(axes=sink.precip.axes, name='liqprec', data=data, atts=default_varatts['liqprec'])
          # create variable and add to dataset          
          sink.addVariable(Var, asNC=True)
        # net precip
        if sink.hasVariable('precip') and sink.hasVariable('evap'):
          data = sink.precip.getArray() - sink.evap.getArray()
          Var = Variable(axes=sink.precip.axes, name='p-et', data=data, atts=default_varatts['p-et'])
          # create variable and add to dataset          
          sink.addVariable(Var, asNC=True)
  
  #       # add names and length of months
  #       sink.axisAnnotation('name_of_month', name_of_month, 'time', 
  #                           atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
  #       #print '   ===   month   ===   '
  #       sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
  #                     atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
        
        # add length and names of month
        if sink.hasAxis('time', strict=False):
          addLengthAndNamesOfMonth(sink, noleap=True)     
  #       # add geolocators
  #       sink = addGeoLocator(sink)  
        # close...
        sink.sync()
        sink.close()
        
        # move files
        os.rename(filepath, origpath)
        os.rename(tmppath,filepath)
        
        # print dataset
        print('')
        print(sink)     
        
