'''
Created on Jan 9, 2017

This module contains meta data and access functions for normals and monthly historical time-series data from the 
Canadian Forest Service (Natural Resources Canada, NRCan)

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc # netcdf python module
import os
# internal imports
from geodata.base import Variable, Axis
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition, addGDALtoVar
from datasets.common import translateVarNames, days_per_month, name_of_month, data_root 
from datasets.common import loadObservations, grid_folder, transformMonthly, transformDays, timeSlice
from geodata.misc import DatasetError, VariableError
from utils.nctools import writeNetCDF


## NRCan Meta-data

dataset_name = 'NRCan'
root_folder = '{:s}/{:s}/'.format(data_root,dataset_name) # the dataset root folder

# NRCan grid definitions           
geotransform_NA12 = (-168.0, 1./12., 0.0, 25.0, 0.0, 1./12.)
size_NA12 = (1392, 720) # (x,y) map size of NRCan grid
geotransform_CA12 = (-141.0, 1./12., 0.0, 41.0, 0.0, 1./12.)
size_CA12 = (1068, 510) # (x,y) map size of NRCan grid
geotransform_CA24 = (-141.0, 1./24., 0.0, 41.0, 0.0, 1./24.)
size_CA24 = (2136, 1008) # (x,y) map size of NRCan grid
# make GridDefinition instances
NRCan_NA12_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_NA12, size=size_NA12)
NRCan_CA12_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_CA12, size=size_CA12)
NRCan_CA24_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_CA24, size=size_CA24)
# default grid (NA12)
NRCan_grid = NRCan_NA12_grid; geotransform = geotransform_NA12; size = size_NA12


# variable attributes and names (only applied to original time-series!)
varatts = dict(maxt = dict(name='Tmax', units='K'), # 2m maximum temperature
               mint = dict(name='Tmin', units='K'), # 2m minimum temperature
               pcp  = dict(name='precip', units='kg/m^2/s'), # total precipitation
               pet  = dict(name='pet', units='kg/m^2/s'), # potential evapo-transpiration
               rain = dict(name='liqprec', units='kg/m^2/s'), # total precipitation
               snwd = dict(name='snowh', units='m'), # snow depth
               rrad = dict(name='SWD', units='J/m^2/s'), # solar radiation
               # diagnostic variables
               T2 = dict(name='T2', units='K'), # 2m average temperature
               solprec = dict(name='liqprec', units='kg/m^2/s'), # total precipitation
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='month', ), # time coordinate
               # N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field

tsvaratts = varatts
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    
# variable and file lists settings
nofile = ('T2','solprec','lat','lon','time') # variables that don't have their own files


## Functions to load different types of NRCan datasets 

def checkGridRes(grid, resolution, period=None, lclim=False):
  ''' helper function to verify grid/resoluton selection ''' 
  # prepare input
  if resolution is None: 
      resolution = '12' 
  elif grid[:2].upper() in ('NA','CA'): 
      resolution = grid[2:]; grid = None
  # check for valid resolution 
  if resolution not in ('12','24',): 
      raise DatasetError, "Selected resolution '%s' is not available!"%resolution  
  if resolution == '24' and period is not None: 
      raise DatasetError, "The highest resolution is only available for the long-term mean!"
  # return
  return grid, resolution

# pre-processed climatology and timeseries files (varatts etc. should not be necessary)
avgfolder = root_folder + 'nrcanavg/' 
avgfile = 'nrcan{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
tsfile = 'nrcan{0:s}_monthly.nc' # extend with grid type only

# function to load these files...
def loadNRCan(name=dataset_name, resolution=None, period=None, grid=None, varlist=None, varatts=None, 
              folder=avgfolder, filelist=None, lautoregrid=True):
    ''' Get the pre-processed monthly NRCan climatology as a DatasetNetCDF. '''
    grid, resolution = checkGridRes(grid, resolution, period=period, lclim=True)
    # load standardized climatology dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, folder=folder, projection=None, resolution=resolution, period=period, 
                               grid=grid, varlist=varlist, varatts=varatts, filepattern=avgfile, 
                               filelist=filelist, lautoregrid=lautoregrid, mode='climatology')
    # return formatted dataset
    return dataset

# function to load Time-series (monthly)
def loadNRCan_TS(name=dataset_name, grid=None, resolution=None, varlist=None, varatts=None, 
                 folder=avgfolder, filelist=None, lautoregrid=True):
    ''' Get the pre-processed monthly NRCan time-series as a DatasetNetCDF at station locations. '''
    grid, resolution = checkGridRes(grid, resolution, period=None, lclim=False)
    # load standardized time-series dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, folder=folder, projection=None, period=None, grid=grid, 
                               varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='time-series')
    # return formatted dataset
    return dataset

# function to load station climatologies
def loadNRCan_Stn(name=dataset_name, period=None, station=None, resolution=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=True):
    ''' Get the pre-processed monthly NRCan climatology as a DatasetNetCDF at station locations. '''
    grid, resolution = checkGridRes(None, resolution, period=period, lclim=True); del grid
    # load standardized climatology dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, folder=folder, projection=None, period=period, station=station, 
                               varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='climatology')
    # return formatted dataset
    return dataset

# function to load station time-series
def loadNRCan_StnTS(name=dataset_name, station=None, resolution=None, varlist=None, varatts=None, 
                    folder=avgfolder, filelist=None, lautoregrid=True):
    ''' Get the pre-processed monthly NRCan time-series as a DatasetNetCDF at station locations. '''
    grid, resolution = checkGridRes(None, resolution, period=None, lclim=False); del grid
    # load standardized time-series dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, folder=folder, projection=None, period=None, station=station, 
                               varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='time-series')
    # return formatted dataset
    return dataset

# function to load regionally averaged climatologies
def loadNRCan_Shp(name=dataset_name, period=None, shape=None, resolution=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=True, lencl=False):
    ''' Get the pre-processed monthly NRCan climatology as a DatasetNetCDF averaged over regions. '''
    grid, resolution = checkGridRes(None, resolution, period=period, lclim=True); del grid
    # load standardized climatology dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, folder=folder, projection=None, period=period, shape=shape, lencl=lencl,
                               station=None, varlist=varlist, varatts=varatts, filepattern=avgfile, 
                               filelist=filelist, resolution=resolution, lautoregrid=False, mode='climatology')
    # return formatted dataset
    return dataset

# function to load regional/shape time-series
def loadNRCan_ShpTS(name=dataset_name, shape=None, resolution=None, varlist=None, varatts=None, 
                    folder=avgfolder, filelist=None, lautoregrid=True, lencl=False):
    ''' Get the pre-processed monthly NRCan time-series as a DatasetNetCDF averaged over regions. '''
    grid, resolution = checkGridRes(None, resolution, period=None, lclim=False); del grid
    # load standardized time-series dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, folder=folder, projection=None, shape=shape, station=None, lencl=lencl, 
                               varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='time-series', period=None)
    # return formatted dataset
    return dataset


## functions to load ASCII data and generate complete GeoPy datasets
from utils.ascii import rasterDataset

# Normals (long-term means): ASCII data specifications
norm_defaults = dict(axes=('time',None,None), dtype=np.float32)
norm_vardefs = dict(maxt = dict(grid='NA12', name='Tmax', units='K', offset=273.15, **norm_defaults), # 2m maximum temperature
                    mint = dict(grid='NA12', name='Tmin', units='K', offset=273.15, **norm_defaults), # 2m minimum temperature
                    pcp  = dict(grid='NA12', name='precip', units='kg/m^2/month', transform=transformMonthly, **norm_defaults), # total precipitation
                    pet  = dict(grid='NA12', name='pet', units='kg/m^2/month', transform=transformMonthly, **norm_defaults), # potential evapo-transpiration
                    rrad = dict(grid='NA12', name='SWD', units='J/m^2/month', transform=transformMonthly, **norm_defaults), # solar radiation
                    rain = dict(grid='CA12', name='liqprec', units='kg/m^2/month', transform=transformMonthly, **norm_defaults), # total precipitation
                    snwd = dict(grid='CA12', name='snowh', units='m', scalefactor=1./100., **norm_defaults), ) # snow depth
norm_axdefs = dict(time = dict(name='time', units='month', coord=np.arange(1,13)),) # time coordinate
norm_grid_pattern = root_folder+'{GRID:s}_normals/' # dataset root folder
norm_var_pattern = '{VAR:s}/{VAR:s}_{time:02d}.asc.gz' # path to variables

# load normals (from different/unspecified periods... ), computer some derived variables, and combine NA and CA grids
def loadASCII_Normals(name=dataset_name, title='NRCan Gridded Normals', atts=None, 
                      NA_grid=None, CA_grid=None, resolution=12, grid_defs=None,
                      var_pattern=norm_var_pattern, grid_pattern=norm_grid_pattern, vardefs=norm_vardefs, axdefs=norm_axdefs):
    ''' load NRCan normals from ASCII files, merge CA and NA grids and compute some additional variables; return Dataset '''
    
    # determine grids / resolution
    if grid_defs is None: 
      grid_defs = grid_def # define in API; register for all pre-defined grids
    if resolution is not None:
      resolution = str(resolution)
      NA_grid = 'NA{:s}'.format(resolution) if NA_grid is None else NA_grid.upper()
      CA_grid = 'CA{:s}'.format(resolution) if CA_grid is None else CA_grid.upper()
      
    # seperate variables
    NA_vardefs = dict(); CA_vardefs = dict()
    for key,var in vardefs.items():
        var = var.copy(); grid = var.pop('grid',None).upper()
        if grid.upper() == NA_grid: NA_vardefs[key] = var
        elif grid.upper() == CA_grid: CA_vardefs[key] = var
        else: raise VariableError(grid)
        
    # load NA grid
    dataset = rasterDataset(name=name, title=title, vardefs=NA_vardefs, axdefs=axdefs, atts=atts, projection=None, 
                            griddef=grid_defs[NA_grid], lgzip=None, lgdal=True, lmask=True, fillValue=None, lskipMissing=True, 
                            lgeolocator=True, file_pattern=grid_pattern.format(GRID=NA_grid)+var_pattern )    
    # load CA grid
    ca_ds = rasterDataset(name=name, title=title, vardefs=CA_vardefs, axdefs=axdefs, atts=atts, projection=None, 
                          griddef=grid_defs[CA_grid], lgzip=None, lgdal=True, lmask=True, fillValue=None, lskipMissing=True, 
                          lgeolocator=False, file_pattern=grid_pattern.format(GRID=CA_grid)+var_pattern )
    
    # merge grids
    naaxes = dataset.axes
    nagt = dataset.geotransform; cagt = ca_ds.geotransform
    assert nagt[2] == nagt[4] == cagt[2] == cagt[4] == 0
    assert nagt[1] == cagt[1] and nagt[5] == cagt[5]
    ios = int( ( cagt[0] - nagt[0] ) / nagt[1] )
    jos = int( ( cagt[3] - nagt[3] ) / nagt[5] )
    nashp = dataset.mapSize # mapSize has the correct axis order (y,x)
    caje,caie = ca_ds.mapSize # axis order is (y,x)
    # create new variables
    for key,var in ca_ds.variables.items():
        # create new data array
        assert var.shape[-2:] == (caje,caie)
        data = np.ma.empty(var.shape[:-2]+nashp) # use the shape of the NA grid and other axes from the original
        data[:] = np.ma.masked # everything that is not explicitly assigned, shall be masked
        data[...,jos:jos+caje,ios:ios+caie] = var.data_array # assign partial data
        # figure out axes and create Variable
        axes = [naaxes[ax.name] for ax in var.axes]
        newvar = Variable(name=key, units=var.units, axes=axes, data=data, atts=var.atts, plot=var.plot)
        newvar = addGDALtoVar(newvar, griddef=dataset.griddef,)
        dataset.addVariable(newvar, copy=False)
    
    # return properly formatted dataset
    return dataset

# Historical time-series
hist_vardefs = NotImplemented
hist_axdefs = NotImplemented
# N.B.: the time-series time offset has to be chose such that 1979 begins with the origin (time=0)
hist_folder_pattern = data_root+'{GRID:s}_hist/'
hist_file_pattern = '{VAR:s}/{year:04d}/{VAR:s}_{month:02d}.asc.gz'

# load historical time-series
def loadASCII_Historical(**kwargs):
    ''' '''
    raise NotImplementedError

## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = norm_grid_pattern+norm_var_pattern # filename pattern: variable name and resolution
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: variable name and resolution
data_folder = avgfolder # folder for user data
grid_def = {'NA12':NRCan_NA12_grid, 'CA12':NRCan_CA12_grid, 'CA24':NRCan_CA24_grid} # standardized grid dictionary
LTM_grids = ['NA12','CA12','CA24'] # grids that have long-term mean data 
TS_grids = ['NA12','CA12'] # grids that have time-series data
grid_res = {'NA12':1./12.,'CA12':1./12.,'CA24':1./24.} # no special name, since there is only one...
default_grid = NRCan_NA12_grid
# functions to access specific datasets
loadLongTermMean = loadNRCan # climatology provided by publisher
loadTimeSeries = loadNRCan_TS # time-series data
loadClimatology = loadNRCan # pre-processed, standardized climatology
loadStationClimatology = loadNRCan_Stn # climatologies without associated grid (e.g. stations) 
loadStationTimeSeries = loadNRCan_StnTS # time-series without associated grid (e.g. stations)
loadShapeClimatology = loadNRCan_Shp # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries = loadNRCan_ShpTS # time-series without associated grid (e.g. provinces or basins)


if __name__ == '__main__':
  
    
    mode = 'convert_ASCII'
    
    
    if mode == 'convert_ASCII':
      
        # load ASCII dataset with default values
        dataset = loadASCII_Normals(name='NRCan', title='NRCan Test Dataset', atts=None, 
                                    NA_grid=None, CA_grid=None, resolution=12, grid_defs=grid_def,)        
        # test 
        print(dataset)