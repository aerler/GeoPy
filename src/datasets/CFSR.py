'''
Created on 2013-09-12

This module contains meta data and access functions for the monthly CFSR time-series data.

@author: Andre R. Erler, GPL v3
'''

# external imports
import os
# internal imports
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.misc import DatasetError
from geodata.gdal import addGDALtoDataset, GridDefinition
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root, loadClim
from geodata.process import CentralProcessingUnit


## CRU Meta-data

dataset_name = 'CFSR'

# CFSR grid definition           
geotransform_031 = (-180.15625, 0.3125, 0.0, 89.915802001953125, 0.0, -0.30960083)
size_031 = (1152,576) # (x,y) map size
geotransform_05 = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
size_05 = (720,360) # (x,y) map size

# make GridDefinition instance
CFSR_031_grid = GridDefinition(name='CFSR_031', projection=None, geotransform=geotransform_031, size=size_031)
CFSR_05_grid = GridDefinition(name='CFSR_05', projection=None, geotransform=geotransform_05, size=size_05)
CFSR_grid = CFSR_031_grid # default

# variable attributes and name
varatts = dict(TMP_L103_Avg = dict(name='T2', units='K'), # 2m average temperature
               TMP_L1 = dict(name='Ts', units='K'), # average skin temperature
               PRATE_L1 = dict(name='precip', units='kg/m^2/s'), # total precipitation
               PRES_L1 = dict(name='ps', units='Pa'), # surface pressure
               WEASD_L1 = dict(name='snow', units='kg/m^2'), # snow water equivalent
               SNO_D_L1 = dict(name='snowh', units='m'), # snow depth
               # lower resolution
               PRMSL_L101 = dict(name='pmsl', units='Pa'), # sea-level pressure
               # static fields (full resolution)
               LAND_L1 = dict(name='landmask', units=''), # land mask
               HGT_L1 = dict(name='zs', units='m'), # surface elevation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='day', scalefactor=1/24.), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
# N.B.: the time-series begins in 1979 (time=0), so no offset is necessary

nofile = ('lat','lon','time') # variables that don't have their own files
# file names by variable
hiresfiles = dict(TMP_L103_Avg='TMP.2m', TMP_L1='TMP.SFC', PRATE_L1='PRATE.SFC', PRES_L1='PRES.SFC',
                 WEASD_L1='WEASD.SFC', SNO_D_L1='SNO_D.SFC')
hiresstatic = dict(LAND_L1='LAND.SFC', HGT_L1='HGT.SFC')
lowresfiles = dict(PRMSL_L101='PRMSL.MSL')
lowresstatic = dict() # currently none available
hiresfiles = {key:'flxf06.gdas.{0:s}.grb2.nc'.format(value) for key,value in hiresfiles.iteritems()}
hiresstatic = {key:'flxf06.gdas.{0:s}.grb2.nc'.format(value) for key,value in hiresstatic.iteritems()}
lowresfiles = {key:'pgbh06.gdas.{0:s}.grb2.nc'.format(value) for key,value in lowresfiles.iteritems()}
lowresstatic = {key:'pgbh06.gdas.{0:s}.grb2.nc'.format(value) for key,value in lowresstatic.iteritems()}
# list of variables to load
# varlist = ['precip','snowh'] + hiresstatic.keys() + list(nofile) # hires + coordinates
varlist_hires = hiresfiles.keys() + hiresstatic.keys() + list(nofile) # hires + coordinates    
varlist_lowres = lowresfiles.keys() + lowresstatic.keys() + list(nofile) # hires + coordinates

# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder


## Functions to load different types of CFSR datasets 

tsfolder = root_folder + 'Monthly/'
def loadCFSR_TS(name=dataset_name, varlist=None, varatts=varatts, resolution='hires', filelist=None, folder=tsfolder):
  ''' Get a properly formatted CFSR dataset with monthly mean time-series. '''
  # translate varlist
  if varlist is None:
    if resolution == 'hires' or resolution == '03': varlist = varlist_hires
    elif resolution == 'lowres' or resolution == '05': varlist = varlist_lowres     
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  if filelist is None: # generate default filelist
    if resolution == 'hires' or resolution == '03': 
      files = [hiresfiles[var] for var in varlist if var in hiresfiles]
    elif resolution == 'lowres' or resolution == '05': 
      files = [lowresfiles[var] for var in varlist if var in lowresfiles]
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=files, varlist=varlist, varatts=varatts, 
                          check_override=['time'], multifile=False, ncformat='NETCDF4_CLASSIC')
  # load static data
  if filelist is None: # generate default filelist
    if resolution == 'hires' or resolution == '03': 
      files = [hiresstatic[var] for var in varlist if var in hiresstatic]
    elif resolution == 'lowres' or resolution == '05': 
      files = [lowresstatic[var] for var in varlist if var in lowresstatic]
    # create singleton time axis
    staticdata = DatasetNetCDF(name=name, folder=folder, filelist=files, varlist=varlist, varatts=varatts, 
                               axes=dict(lon=dataset.lon, lat=dataset.lat), multifile=False, 
                               check_override=['time'], ncformat='NETCDF4_CLASSIC')
    # N.B.: need to override the axes, so that the datasets are consistent
  if len(staticdata.variables) > 0:
    for var in staticdata.variables.values(): 
      if not dataset.hasVariable(var.name):
        var.squeeze() # remove time dimension
        dataset.addVariable(var, copy=False) # no need to copy... but we can't write to the netcdf file!
  # add projection  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset


# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'cfsravg/' 
avgfile = 'cfsr%s_clim%s.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadCFSR(name=dataset_name, period=None, grid=None, resolution=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly CFSR climatology as a DatasetNetCDF. '''
  # prepare input
  if grid is not None and grid[0:5].lower() == 'cfsr_': 
    resolution = grid[5:]
    grid = None
  elif resolution is None: resolution = '03'
  # check resolution
  if grid is None:
    # check for valid resolution
    if resolution == 'hires': resolution = '03' 
    elif resolution == 'lowres': resolution = '05' 
    elif resolution not in ('03','05'): 
      raise DatasetError, "Selected resolution '%s' is not available!"%resolution  
    grid = resolution # grid supersedes resolution  
  # load standardized climatology dataset with GPCC-specific parameters
  dataset = loadClim(name=name, folder=folder, projection=None, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
grid_def = {0.31:CFSR_031_grid, 0.5:CFSR_05_grid}  # standardized grid dictionary, addressed by grid resolution
grid_tag = {0.31:'031', 0.5:'05'} # tag used in climatology files
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = loadCFSR_TS # time-series data
loadClimatology = loadCFSR # pre-processed, standardized climatology


## (ab)use main execution for quick test
if __name__ == '__main__':
  
  mode = 'test_climatology'
#   mode = 'average_timeseries'
  reses = ('05',) # for testing
  reses = ('hires', 'lowres')
  period = (1979,1981)
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_climatology':
      
      # load averaged climatology file
      print('')
      dataset = loadCFSR(resolution=res,period=period)
      print(dataset)
      print('')
      print(dataset.geotransform)
              
    elif mode == 'average_timeseries':
      
      # load source
      periodstr = '%4i-%4i'%period
      print('\n')
      print('   ***   Processing Resolution %s from %s   ***   '%(res,periodstr))
      print('\n')
      source = loadCFSR_TS(resolution=res)
      print(source)
      print('\n')
      # prepare sink
      filename = avgfile%('_'+res,'_'+periodstr)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      sink = DatasetNetCDF(name='CFSR Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
      sink.atts.period = periodstr 
      
      # determine averaging interval
      offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, tmp=True)
      
      # start processing climatology
      print('')
      print('   +++   processing climatology   +++   ') 
      CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
      print('\n')
      
      # shift longitude axis by 180 degrees left (i.e. 0 - 360 -> -180 - 180)
      print('')
      print('   +++   processing shift/roll   +++   ') 
      CPU.Shift(lon=-180, flush=False)
      print('\n')      
      
      # sync temporary storage with output
      CPU.sync(flush=True)

      # make new masks
      if sink.hasVariable('landmask'):
        sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=True, merge=False)

      # add names and length of months
      sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                          atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
      #print '   ===   month   ===   '
      sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                    atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
      
      # close...
      sink.sync()
      sink.close()
      # print dataset
      print('')
      print(sink)     
      