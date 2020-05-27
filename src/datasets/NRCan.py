'''
Created on Jan 9, 2017

This module contains meta data and access functions for normals and monthly historical time-series data from the 
Canadian Forest Service (Natural Resources Canada, NRCan)

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import os
# internal imports
from geodata.base import Variable, Axis
from geodata.gdal import GridDefinition, addGDALtoVar
from datasets.common import getRootFolder, loadObservations, transformMonthly, addLengthAndNamesOfMonth, monthlyTransform, addLandMask
from geodata.misc import DatasetError, VariableError, AxisError
from utils.nctools import writeNetCDF


## NRCan Meta-data

dataset_name = 'NRCan'
root_folder = getRootFolder(dataset_name=dataset_name) # get dataset root folder based on environment variables

# NRCan grid definitions           
# make GridDefinition instances
geotransform_NA12 = (-168.0, 1./12., 0.0, 25.0, 0.0, 1./12.); size_NA12 = (1392, 720) # (x,y) map size of NRCan grid
NRCan_NA12_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_NA12, size=size_NA12)
geotransform_NA60 = (-168.0, 1./60., 0.0, 25.0, 0.0, 1./60.); size_NA60 = (6960, 3600) # (x,y) map size of NRCan grid
NRCan_NA60_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_NA60, size=size_NA60)
geotransform_CA12 = (-141.0, 1./12., 0.0, 41.0, 0.0, 1./12.); size_CA12 = (1068, 510) # (x,y) map size of NRCan grid
NRCan_CA12_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_CA12, size=size_CA12)
geotransform_CA24 = (-141.0, 1./24., 0.0, 41.0, 0.0, 1./24.); size_CA24 = (2136, 1008) # (x,y) map size of NRCan grid
NRCan_CA24_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform_CA24, size=size_CA24)
NRCan_grids = ['NA12','NA60','CA12','CA24']
# default grid (NA12)
NRCan_grid = NRCan_NA12_grid; geotransform = geotransform_NA12; size = size_NA12


# variable attributes and names (only applied to original time-series!)
varatts = dict(Tmax    = dict(name='Tmax', units='K'), # 2m maximum temperature
               Tmin    = dict(name='Tmin', units='K'), # 2m minimum temperature
               precip  = dict(name='precip', units='kg/m^2/s'), # total precipitation
               pet     = dict(name='pet', units='kg/m^2/s'), # potential evapo-transpiration
               liqprec = dict(name='liqprec', units='kg/m^2/s'), # total precipitation
               snowh   = dict(name='snowh', units='m'), # snow depth
               SWD     = dict(name='SWDNB', units='W/m^2', scalefactor=30.4e6), # solar radiation, corrected (MJ/day->J/month)
               SWDNB   = dict(name='SWDNB', units='W/m^2'), # solar radiation
               # diagnostic variables
               T2        = dict(name='T2', units='K'), # 2m average temperature
               solprec   = dict(name='solprec', units='kg/m^2/s'), # total precipitation
               snow      = dict(name='snow', units='kg/m^2'), # snow water equivalent
               snwmlt    = dict(name='snwmlt', units='kg/m^2/s'), # snow melt (rate)
               snow_acc  = dict(name='snow_acc', units='kg/m^2/s'), # rat of change of snowpack - in lieu of actual snowmelt
               liqwatflx = dict(name='liqwatflx', units='kg/m^2/s'), # liquid water forcing (rate)
               landmask  = dict(name='landmask', units='N/A'), # the land mask...
               # axes (don't have their own file; listed in axes)
               time = dict(name='time', units='month',), # time coordinate
               # N.B.: the time-series time offset has to be chosen such that 1979 begins with the origin (time=0)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field

tsvaratts = varatts.copy()
# list of variables to load
varlist = list(varatts.keys()) # also includes coordinate fields    
# variable and file lists settings
nofile = ('T2','solprec','lat','lon','time') # variables that don't have their own files

## Functions to load different types of NRCan datasets 

def checkGridRes(grid, resolution, snow_density=None, period=None, lclim=False):
  ''' helper function to verify grid/resoluton selection '''
  # prepare input
  if grid is not None and grid.upper() in NRCan_grids:
      resolution = grid.lower()
      grid = None
  if resolution is None: resolution = 'na12' # default
  if not isinstance(resolution, str): raise TypeError(resolution) 
  # figure out clim/TS
  if period is not None: lclim=True
  # check for valid resolution 
  if lclim and resolution not in LTM_grids and resolution.upper() not in LTM_grids: 
      raise DatasetError("Selected resolution '{:s}' is not available for long-term means!".format(resolution))
  if not lclim and resolution not in TS_grids and resolution.upper() not in TS_grids: 
      raise DatasetError("Selected resolution '{:s}' is not available for historical time-series!".format(resolution))
  # handle special case of snow density parameter: append to resolution
  if snow_density:
      # check validity (just to raise error if invalid)
      tmp = getSnowDensity(snow_class=snow_density, lraise=True); del tmp
      # append to resolution
      if resolution is None: resolution = snow_density
      else: resolution = resolution + '_' + snow_density
  # return
  return grid, resolution

# pre-processed climatology and timeseries files (varatts etc. should not be necessary)
clim_period = (1970,2000) # default time period for long-term means
#clim_period = (1980,2010) # default time period for long-term means
avgfolder = root_folder + 'nrcanavg/' 
avgfile = 'nrcan{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
tsfile = 'nrcan{0:s}_monthly.nc' # extend with grid type only
# daily data
daily_folder    = root_folder + dataset_name.lower()+'_daily/' 
netcdf_filename = dataset_name.lower()+'_{RES:s}_{VAR:s}_daily.nc' # extend with variable name
netcdf_dtype    = np.dtype('<f4') # little-endian 32-bit float
netcdf_settings = dict(chunksizes=(8,256,256))

# function to load these files...
def loadNRCan(name=dataset_name, title=dataset_name, resolution=None, period=clim_period, grid=None, varlist=None, 
              snow_density=None, varatts=None, folder=avgfolder, filelist=None, lautoregrid=False, filemode='r'):
    ''' Get the pre-processed monthly NRCan climatology as a DatasetNetCDF. '''
    grid, resolution = checkGridRes(grid, resolution, snow_density=snow_density, period=period, lclim=True)
    # load standardized climatology dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, title=title, folder=folder, projection=None, resolution=resolution, period=period, 
                               grid=grid, varlist=varlist, varatts=varatts, filepattern=avgfile, griddef=NRCan_NA12_grid,
                               filelist=filelist, lautoregrid=lautoregrid, mode='climatology', filemode=filemode)
    # return formatted dataset
    return dataset

# function to load Time-series (monthly)
def loadNRCan_TS(name=dataset_name, title=dataset_name, grid=None, resolution=None, varlist=None, varatts=None, 
                 snow_density=None, folder=avgfolder, filelist=None, lautoregrid=False, filemode='r'):
    ''' Get the pre-processed monthly NRCan time-series as a DatasetNetCDF at station locations. '''
    grid, resolution = checkGridRes(grid, resolution, snow_density=snow_density, period=None, lclim=False)
    # load standardized time-series dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, title=title, folder=folder, projection=None, period=None, grid=grid, 
                               varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='time-series', filemode=filemode)
    # return formatted dataset
    return dataset

# function to load station climatologies
def loadNRCan_Stn(name=dataset_name, title=dataset_name, period=clim_period, station=None, resolution=None, varlist=None, 
                  snow_density=None, varatts=None, folder=avgfolder, filelist=None):
    ''' Get the pre-processed monthly NRCan climatology as a DatasetNetCDF at station locations. '''
    grid, resolution = checkGridRes(None, resolution, snow_density=snow_density, period=period, lclim=True); del grid
    # load standardized climatology dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, title=title, folder=folder, projection=None, period=period, station=station, 
                               varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='climatology')
    # return formatted dataset
    return dataset

# function to load station time-series
def loadNRCan_StnTS(name=dataset_name, title=dataset_name, station=None, resolution=None, varlist=None, varatts=None, 
                    snow_density=None, folder=avgfolder, filelist=None):
    ''' Get the pre-processed monthly NRCan time-series as a DatasetNetCDF at station locations. '''
    grid, resolution = checkGridRes(None, resolution, snow_density=snow_density, period=None, lclim=False); del grid
    # load standardized time-series dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, title=title, folder=folder, projection=None, period=None, station=station, 
                               varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='time-series')
    # return formatted dataset
    return dataset

# function to load regionally averaged climatologies
def loadNRCan_Shp(name=dataset_name, title=dataset_name, period=clim_period, shape=None, resolution=None, varlist=None, 
                  snow_density=None, varatts=None, folder=avgfolder, filelist=None, lencl=False):
    ''' Get the pre-processed monthly NRCan climatology as a DatasetNetCDF averaged over regions. '''
    grid, resolution = checkGridRes(None, resolution, snow_density=snow_density, period=period, lclim=True); del grid
    # load standardized climatology dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, title=title, folder=folder, projection=None, period=period, shape=shape, 
                               lencl=lencl, station=None, varlist=varlist, varatts=varatts, filepattern=avgfile, 
                               filelist=filelist, resolution=resolution, lautoregrid=False, mode='climatology')
    # return formatted dataset
    return dataset

# function to load regional/shape time-series
def loadNRCan_ShpTS(name=dataset_name, title=dataset_name, shape=None, resolution=None, varlist=None, varatts=None, 
                    snow_density=None, folder=avgfolder, filelist=None, lencl=False):
    ''' Get the pre-processed monthly NRCan time-series as a DatasetNetCDF averaged over regions. '''
    grid, resolution = checkGridRes(None, resolution, snow_density=snow_density, period=None, lclim=False); del grid
    # load standardized time-series dataset with NRCan-specific parameters
    dataset = loadObservations(name=name, title=title, folder=folder, projection=None, shape=shape, station=None, 
                               lencl=lencl, varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                               resolution=resolution, lautoregrid=False, mode='time-series', period=None)
    # return formatted dataset
    return dataset


## snow density estimates

def getSnowDensity(snow_class, lraise=True):
    ''' '''
    #       estimates from the Canadian Meteorological Centre for maritime climates (Table 3):
    #       https://nsidc.org/data/NSIDC-0447/versions/1
    #       a factor of 1000 has been applied, because snow depth is in m (and not mm)
    if snow_class.lower() == 'tundra':
      # Tundra snow cover
      density = np.asarray([0.2303, 0.2427, 0.2544, 0.2736, 0.3117, 0.3693, 0.3693, 0.3693, 0.2, 0.2, 0.2107, 0.2181], dtype=np.float32)*1000.
    elif snow_class.lower() == 'taiga':
      # Taiga snow cover
      density = np.asarray([0.1931, 0.2059, 0.2218, 0.2632, 0.3190, 0.3934, 0.3934, 0.3934, 0.16, 0.16, 0.1769, 0.1798], dtype=np.float32)*1000.
    elif snow_class.lower() == 'maritime':
      # Maritime snow cover
      density = np.asarray([0.2165, 0.2485, 0.2833, 0.332, 0.3963, 0.501, 0.501, 0.501, 0.16, 0.16, 0.1835, 0.1977], dtype=np.float32)*1000.
    elif snow_class.lower() == 'ephemeral':
      # Ephemeral snow cover
      density = np.asarray([0.3168, 0.3373, 0.3643, 0.4046, 0.4586, 0.5098, 0.5098, 0.5098, 0.25, 0.25, 0.3, 0.3351], dtype=np.float32)*1000.
    elif snow_class.lower() == 'prairies':
      # Prairie snow cover
      density = np.asarray([0.2137, 0.2416, 0.2610, 0.308, 0.3981, 0.4645, 0.4645, 0.4645, 0.14, 0.14, 0.1616, 0.1851], dtype=np.float32)*1000.
    elif snow_class.lower() == 'alpine':
      # Alpine snow cover
      density = np.asarray([0.2072, 0.2415, 0.2635, 0.312, 0.3996, 0.4889, 0.4889, 0.4889, 0.16, 0.16, 0.172, 0.1816], dtype=np.float32)*1000.
    elif lraise:
      raise ValueError("Value '{}' for snow denisty class not defined.".format(snow_class))
    return density

## functions to load ASCII data and generate complete GeoPy datasets

# a universal load function for normals and historical timeseries; also computes some derived variables, and combines NA and CA grids
def loadASCII_TS(name=None, title=None, atts=None, derived_vars=None, varatts=None, NA_grid=None, CA_grid=None, lskipNA=False,
                 merged_axis=None, time_axis='time', resolution=None, grid_defs=None, period=None, var_pattern=None, 
                 snow_density='maritime', grid_pattern=None, vardefs=None, axdefs=None, lfeedback=True):
    ''' load NRCan time-series data from ASCII files, merge CA and NA grids and compute some additional variables; return Dataset '''
    
    from utils.ascii import rasterDataset

    # determine grids / resolution
    if grid_defs is None: 
      grid_defs = grid_def # define in API; register for all pre-defined grids
    if resolution is not None:
      resolution = str(resolution)
      NA_grid = 'NA{:s}'.format(resolution) if NA_grid is None else NA_grid.upper()
      CA_grid = 'CA{:s}'.format(resolution) if CA_grid is None else CA_grid.upper()
      
    # seperate variables
    NA_vardefs = dict(); CA_vardefs = dict()
    for key,var in list(vardefs.items()):
        var = var.copy(); grid = var.pop('grid',None).upper()
        if grid.upper() not in grid_defs:
            # skip variable
            print("Warning: grid '{}' for variable '{}'('{}') not found - variable will be skipped!".format(grid,key,var['name']))
        elif grid.upper() == NA_grid: NA_vardefs[key] = var
        elif grid.upper() == CA_grid: CA_vardefs[key] = var
        else: raise VariableError(grid)
        
    # determine period extension
    prdstr = '_{0:04d}-{1:04d}'.format(period[0]+1, period[1]) if period is not None else ''
        
    # load NA grid
    if NA_vardefs:
        dataset = rasterDataset(name=name, title=title, vardefs=NA_vardefs, axdefs=axdefs, atts=atts, projection=None, 
                                griddef=grid_defs[NA_grid], lgzip=None, lgdal=True, lmask=True, fillValue=None, 
                                lskipMissing=True, lgeolocator=True, time_axis=time_axis, lfeedback=lfeedback,
                                file_pattern=grid_pattern.format(GRID=NA_grid,PRDSTR=prdstr)+var_pattern )    
    else:
        if lskipNA:
            dataset = None
        else:
            raise NotImplementedError("North America grid '{}' not defined; could either skip or construct from pickle.".format(NA_grid))
    # load CA grid
    if CA_vardefs:
        ca_ds = rasterDataset(name=name, title=title, vardefs=CA_vardefs, axdefs=axdefs, atts=atts, projection=None, 
                              griddef=grid_defs[CA_grid], lgzip=None, lgdal=True, lmask=True, fillValue=None, 
                              lskipMissing=True, lgeolocator=False, time_axis=time_axis, lfeedback=lfeedback,
                              file_pattern=grid_pattern.format(GRID=CA_grid,PRDSTR=prdstr)+var_pattern )
        if dataset is None:
            dataset = ca_ds
        else:
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
            for key,var in list(ca_ds.variables.items()):
                # create new data array
                assert var.shape[-2:] == (caje,caie)
                data = np.ma.empty(var.shape[:-2]+nashp, dtype=var.dtype) # use the shape of the NA grid and other axes from the original
                data[:] = np.ma.masked # everything that is not explicitly assigned, shall be masked
                data[...,jos:jos+caje,ios:ios+caie] = var.data_array # assign partial data
                # figure out axes and create Variable
                axes = [naaxes[ax.name] for ax in var.axes]
                newvar = Variable(name=key, units=var.units, axes=axes, data=data, atts=var.atts, plot=var.plot)
                newvar = addGDALtoVar(newvar, griddef=dataset.griddef,)
                dataset.addVariable(newvar, copy=False)
    else:
        pass # can be skipped - Canada doesn't matter ;-)
      
    # snow needs some special care: replace mask with mask from rain and set the rest to zero
    if 'snowh' in dataset:
        assert 'liqprec' in dataset
        assert dataset.snowh.shape == dataset.liqprec.shape, dataset
        snwd = ma.masked_where(condition=dataset.liqprec.data_array.mask, a=dataset.snowh.data_array.filled(0), copy=False)
        dataset.snowh.data_array = snwd # reassingment is necessary, because filled() creates a copy
        dataset.snowh.fillValue = dataset.liqprec.fillValue 
        assert np.all( dataset.snowh.data_array.mask == dataset.liqprec.data_array.mask ), dataset.snowh.data_array
        assert dataset.snowh.fillValue == dataset.liqprec.fillValue, dataset.snowh.data_array
    
    # merge time axes (for historical timeseries)
    if merged_axis:
        if merged_axis.name == 'time':
            if not 'merged_axes' in merged_axis.atts: 
                raise AxisError('No list/tuple of merge_axes specified in merged_axis atts!')
            merge_axes = merged_axis.atts['merged_axes']
            dataset = dataset.mergeAxes(axes=merge_axes, new_axis=merged_axis, axatts=None, asVar=True, linplace=True, 
                                        lcheckAxis=False, lcheckVar=None, lvarall=True, ldsall=True, lstrict=True)
    
    # compute some secondary/derived variables
    if derived_vars:
      for var in derived_vars:
          # don't overwrite existing variables
          if var in dataset: raise DatasetError(var)
          # 2m Temperature as mean of diurnal min/max temperature
          if var == 'T2':                 
              if not ( 'Tmin' in dataset and 'Tmax' in dataset ): # check prerequisites
                  raise VariableError("Prerequisites for '{:s}' not found.\n{}".format(var,dataset))
              # compute values and add to dataset
              dataset[var] = ( dataset.Tmax + dataset.Tmin ) / 2. # simple average
          # Solid Precipitation (snow) as difference of total and liquid precipitation (rain)
          elif var == 'solprec':                 
              if not ( 'precip' in dataset and 'liqprec' in dataset ): # check prerequisites
                  raise VariableError("Prerequisites for '{:s}' not found.\n{}".format(var,dataset))
              # compute values and add to dataset
              newvar = dataset.precip - dataset.liqprec # simple difference
              newvar.data_array.clip(min=0, out=newvar.data_array) # clip values smaller than zero (in-place)
              dataset[var] = newvar
          # Snowmelt as residual of snow fall and accumulation changes
          elif var == 'snow':
              if not 'snowh' in dataset: # check prerequisites
                  raise VariableError("Prerequisites for '{:s}' not found.\n{}".format(var,dataset))
              # before we can compute anything, we need estimates of snow density from a seasonal climatology
              density = getSnowDensity(snow_density)
              density_note = "Snow density extimates from CMC for {:s} snow cover (Tab. 3): https://nsidc.org/data/NSIDC-0447/versions/1#title15".format(snow_density.title())
              # compute values and add to dataset
              newvar = monthlyTransform(var=dataset.snowh.copy(deepcopy=True), lvar=True, linplace=True, scalefactor=density)
              newvar.atts['long_name'] = 'Snow Water Equivalent at the end of the month.'
              newvar.atts['note'] = density_note
              dataset[var] = newvar
          # Snowmelt as residual of snow fall and snow accumulation (water equivalent) changes
          elif var == 'snwmlt':
              if not ( 'solprec' in dataset and 'snow' in dataset ): # check prerequisites
                  raise VariableError("Prerequisites for '{:s}' not found.\n{}".format(var,dataset))
              snow = dataset.snow; tax = snow.axes[0]; swe = snow.data_array 
              if tax.name != 'time' and len(tax) == 12:
                  raise NotImplementedError("Computing differences is currently only implemented for climatologies.")             
              # compute central differences
              delta = ma.diff(swe, axis=0); dd = ( swe[0,:] - swe[-1,:] ).reshape((1,)+swe.shape[1:])
              assert dd.ndim == swe.ndim
              assert np.all( dd.mask[0,:] == swe.mask[0,:] ), dd
              #data = -1 * ( ma.concatenate((dd,delta), axis=0) + ma.concatenate((delta,dd), axis=0) ) / 2.
              data = -1 * ma.concatenate((dd,delta), axis=0) 
              # N.B.: snow values are already at the end of the month, so differences are average snowmelt over the month
              # create snowmelt variable and do some conversions
              newvar = addGDALtoVar(Variable(data=data, axes=snow.axes, name=var, units='kg/m^2/month'), griddef=dataset.griddef)
              newvar = transformMonthly(var=newvar, slc=None, l365=False, lvar=True, linplace=True)
              newvar += dataset.solprec # add that in-place as well, but after transforming monthly SWE change to SI rate
              newvar.data_array.clip(min=0, out=newvar.data_array) # clip values smaller than zero (in-place)
              newvar.atts['note'] = density_note
              dataset[var] = newvar
              ## normalize snowmelt so that it does not exceed snow fall
              r = dataset.snwmlt.mean(axis=0,keepdims=True,asVar=False)/dataset.solprec.mean(axis=0,keepdims=True,asVar=False)
              rm = r.mean()
              print(("\nSnowmelt to snowfall ratio: {}\n".format(rm)))            
              if rm > 1:
                #r0 = dataset.snwmlt.mean(axis=0,keepdims=True,asVar=False)/dataset.solprec.mean(axis=0,keepdims=True,asVar=False) 
                dataset.snwmlt.data_array /= r # normalize to total snow fall annually and grid point-wise
                assert np.ma.allclose(dataset.snwmlt.mean(axis=0,asVar=False), dataset.solprec.mean(axis=0,asVar=False)), dataset.snwmlt.mean()/dataset.solprec.mean()
              # add snow ratio as diagnostic
              atts = dict(name='ratio', units='', long_name='Ratio of Snowfall to Snowmelt')
              dataset += addGDALtoVar(Variable(data=r.squeeze(), axes=snow.axes[1:], atts=atts), griddef=dataset.griddef)    
          elif var == 'liqwatflx':
              # surface water forcing (not including ET)
              if not ( 'liqprec' in dataset and 'snwmlt' in dataset ): # check prerequisites
                  raise VariableError("Prerequisites for '{:s}' not found.\n{}".format(var,dataset))
              # create variable and compute data
              assert dataset.liqprec.units == 'kg/m^2/s', dataset.liqprec.units
              assert dataset.snwmlt.units == 'kg/m^2/s', dataset.snwmlt.units  
              data = dataset.liqprec[:] + dataset.snwmlt[:]
              newvar = addGDALtoVar(Variable(data=data, axes=dataset.liqprec.axes, name=var, units='kg/m^2/s'), griddef=dataset.griddef)
              newvar.data_array.clip(min=0, out=newvar.data_array) # clip values smaller than zero (in-place)
              newvar.atts['note'] = density_note
              dataset[var] = newvar
          else: raise VariableError(var)
          # for completeness, add attributes
          dataset[var].atts.update(varatts[var])
          dataset[var].data_array._fill_value = dataset[var].fillValue
    
    # add length and names of month
    if dataset.hasAxis('time') and len(dataset.time) == 12:
        addLengthAndNamesOfMonth(dataset) # basically only works for climatologies
    addLandMask(dataset, varname='precip', maskname='landmask', atts=None)
    
    # return properly formatted dataset
    return dataset

## Normals (long-term means): ASCII data specifications
# monthly normals at 1/12 degree resolution (~10 km)
norm12_period = (1970,2000)
norm12_defaults = dict(axes=('time',None,None), dtype=np.float32)
norm12_vardefs = dict(maxt = dict(grid='NA12', name='Tmax', units='K', offset=273.15, **norm12_defaults), # 2m maximum temperature, originally in degrees Celsius
                      mint = dict(grid='NA12', name='Tmin', units='K', offset=273.15, **norm12_defaults), # 2m minimum temperature
                      pcp  = dict(grid='NA12', name='precip', units='kg/m^2/month', transform=transformMonthly, **norm12_defaults), # total precipitation
                      pet  = dict(grid='NA12', name='pet', units='kg/m^2/month', transform=transformMonthly, **norm12_defaults), # potential evapo-transpiration
                      rrad = dict(grid='NA12', name='SWDNB', units='W/m^2', scalefactor=1e6/86400., **norm12_defaults), # solar radiation, originally in MJ/m^2/day
                      rain = dict(grid='CA12', name='liqprec', units='kg/m^2/month', transform=transformMonthly, **norm12_defaults), # total precipitation
                      snwd = dict(grid='CA12', name='snowh', units='m', scalefactor=1./100., **norm12_defaults), ) # snow depth
norm12_axdefs = dict(time = dict(name='time', units='month', coord=np.arange(1,13)),) # time coordinate
norm12_derived = ('T2','solprec','snow','snwmlt','liqwatflx')
norm12_grid_pattern = root_folder+'{GRID:s}_normals{PRDSTR:s}/' # dataset root folder
norm12_var_pattern = '{VAR:s}/{VAR:s}_{time:02d}.asc.gz' # path to variables
norm12_title = 'NRCan Gridded Normals'

def loadASCII_Normals(name=dataset_name, title=norm12_title, atts=None, derived_vars=norm12_derived, varatts=varatts, 
                      NA_grid=None, CA_grid=None, resolution=12, grid_defs=None, period=norm12_period, snow_density='maritime',
                      var_pattern=norm12_var_pattern, grid_pattern=norm12_grid_pattern, vardefs=norm12_vardefs, axdefs=norm12_axdefs):
    ''' load NRCan normals from ASCII files, merge CA and NA grids and compute some additional variables; return Dataset '''
    return loadASCII_TS(name=name, title=title, atts=atts, derived_vars=derived_vars, varatts=varatts, snow_density=snow_density,
                        NA_grid=NA_grid, CA_grid=CA_grid, merged_axis=None, resolution=resolution, grid_defs=grid_defs, 
                        period=period, var_pattern=var_pattern, grid_pattern=grid_pattern, vardefs=vardefs, axdefs=axdefs)


## Historical time-series: ASCII data specifications
# monthly transient at 1/12 degree resolution (~10 km)
# hist_period = (1866,2013) # precip and min/max T only
mons12_period = (1950,2010) # with rain, and snow from 1958 - 2010
mons12_defaults = dict(axes=('year','month',None,None), dtype=np.float32)
mons12_vardefs = dict(maxt = dict(grid='NA12', name='Tmax', units='K', offset=273.15, **mons12_defaults), # 2m maximum temperature, originally in degrees Celsius
                      mint = dict(grid='NA12', name='Tmin', units='K', offset=273.15, **mons12_defaults), # 2m minimum temperature
                      pcp  = dict(grid='NA12', name='precip', units='kg/m^2/month', transform=transformMonthly, **mons12_defaults), # total precipitation
                      rain = dict(grid='CA12', name='liqprec', units='kg/m^2/month', transform=transformMonthly, **mons12_defaults), # total precipitation
                      snwd = dict(grid='CA12', name='snowh', units='m', scalefactor=1./100., **mons12_defaults), ) # snow depth
mons12_axdefs = dict(year = dict(name='year', units='year', coord=None), # yearly coordinate; select coordinate based on period
                     month = dict(name='month', units='month', coord=np.arange(1,13)),) # monthly coordinate
# define merged time axis
mons12_matts = dict(name='time', units='month', long_name='Months since 1979-01', merged_axes = ('year','month'))
# N.B.: the time-series time offset has to be chose such that 1979 begins with the origin (time=0)
mons12_derived = norm12_derived # same as for normals
mons12_grid_pattern = root_folder+'{GRID:s}_hist/'
mons12_var_pattern = '{VAR:s}/{year:04d}/{VAR:s}_{month:02d}.asc.gz'
mons12_title = 'NRCan Historical Gridded Time-series'

# monthly transient at 1/60 degree resolution (~2 km)
mons60_period = (2011,2018) # SnoDAS period for southern Ontario
mons60_defaults = mons12_defaults
mons60_vardefs = dict(maxt = dict(grid='NA60', name='Tmax', units='K', offset=273.15, **mons60_defaults), # 2m maximum temperature, originally in degrees Celsius
                      mint = dict(grid='NA60', name='Tmin', units='K', offset=273.15, **mons60_defaults), # 2m minimum temperature
                      pcp  = dict(grid='NA60', name='precip', units='kg/m^2/month', transform=transformMonthly, **mons60_defaults),) # total precipitation
# define original split and merged time axes
mons60_axdefs = mons12_axdefs; mons60_matts = mons12_matts
# N.B.: the time-series time offset has to be chose such that 1979 begins with the origin (time=0)
mons60_derived = ('T2',) # no snow or rain yet
mons60_grid_pattern = root_folder+'{GRID:s}_mons/'
mons60_var_pattern = '{VAR:s}/{year:04d}/{VAR:s}60_{month:02d}.asc.gz'
mons60_title = 'NRCan Historical Gridded Time-series'


def loadASCII_Hist(name=dataset_name, title=mons12_title, atts=None, derived_vars=mons12_derived, varatts=varatts, snow_density='maritime',
                   NA_grid=None, CA_grid=None, resolution=12, grid_defs=None, period=mons12_period, merged_axis=mons12_matts,
                   var_pattern=mons12_var_pattern, grid_pattern=mons12_grid_pattern, vardefs=mons12_vardefs, axdefs=mons12_axdefs):
    ''' load historical NRCan timeseries from ASCII files, merge CA and NA grids and compute some additional variables; return Dataset '''
    # figure out time period for merged time axis
    for axname,axdef in list(axdefs.items()):
        if 'coord' not in axdef or axdef['coord'] is None:
            assert axdef['units'].lower() == 'year', axdef
            axdef['coord'] = np.arange(period[0],period[1]+1)
    if merged_axis:
        if isinstance(merged_axis,dict) and period:
            merged_axis = Axis(coord=np.arange((period[0]-1979)*12,(period[1]-1978)*12), atts=merged_axis)
            assert 'merged_axes' in merged_axis.atts
            nlen = np.prod([len(mons12_axdefs[axname]['coord']) for axname in merged_axis.atts['merged_axes']])
            assert len(merged_axis) == nlen, (nlen,merged_axis.prettyPrint(short=True)) 
        elif not isinstance(merged_axis,Axis):
            raise TypeError(merged_axis)
    # load ASCII data
    return loadASCII_TS(name=name, title=title, atts=atts, derived_vars=derived_vars, varatts=varatts, time_axis='month', 
                        snow_density=snow_density,
                        NA_grid=NA_grid, CA_grid=CA_grid, merged_axis=merged_axis, resolution=resolution, grid_defs=grid_defs, 
                        period=period, var_pattern=var_pattern, grid_pattern=grid_pattern, vardefs=vardefs, axdefs=axdefs)


# daily transient at 1/12 degree resolution
day12_period = (2011,2018) # SnoDAS period for southern Ontario
day12_defaults = dict(axes=('year','day',None,None), dtype=np.float32, fillValue=None)
day12_vardefs = dict(maxt = dict(grid='CA12', name='Tmax', units='K', offset=273.15, alt_name='max', **day12_defaults), # 2m maximum temperature, originally in degrees Celsius
                     mint = dict(grid='CA12', name='Tmin', units='K', offset=273.15, alt_name='min', **day12_defaults), # 2m minimum temperature
                     pcp  = dict(grid='CA12', name='precip', units='kg/m^2/day', **day12_defaults),) # total precipitation
# define original split and merged time axes
day12_axdefs = dict(time = dict(name='time', units='day', coord=np.arange(1,366)),) # time coordinate
day12_matts = dict(name='time', units='day', long_name='Days since 1979-01-01', merged_axes = ('year','day'))
# N.B.: the time-series time offset has to be chose such that 1979 begins with the origin (time=0)
day12_derived = ('T2',) # no snow or rain yet
day12_grid_pattern = root_folder+'{GRID:s}_Daily/'
day12_var_pattern = '{VAR:s}/{year:04d}/{VAR:s}{year:04d}_{day:d}.asc.gz'
day12_title = 'NRCan Daily Gridded Time-series'


def loadASCII_Daily(name=dataset_name, title=day12_title, atts=None, derived_vars=day12_derived, varatts=varatts, snow_density='maritime',
                   NA_grid=None, CA_grid=None, resolution=12, grid_defs=None, period=day12_period, merged_axis=day12_matts,
                   var_pattern=day12_var_pattern, grid_pattern=day12_grid_pattern, vardefs=day12_vardefs, axdefs=day12_axdefs):
    ''' load historical NRCan timeseries from ASCII files, merge CA and NA grids and compute some additional variables; return Dataset '''
    # figure out time period for merged time axis
    for axname,axdef in list(axdefs.items()):
        if 'coord' not in axdef or axdef['coord'] is None:
            assert axdef['units'].lower() == 'year', axdef
            axdef['coord'] = np.arange(period[0],period[1]+1)
    if merged_axis:
        if isinstance(merged_axis,dict) and period:
            nlen = np.prod([len(axdefs[axname]['coord']) for axname in merged_axis['merged_axes']])
            merged_axis = Axis(coord=np.arange(nlen), atts=merged_axis)
            assert 'merged_axes' in merged_axis.atts
            assert len(merged_axis) == nlen, (nlen,merged_axis.prettyPrint(short=True))
        elif not isinstance(merged_axis,Axis):
            raise TypeError(merged_axis)
    # load ASCII data
    return loadASCII_TS(name=name, title=title, atts=atts, derived_vars=derived_vars, varatts=varatts, time_axis='day', 
                        snow_density=snow_density, lskipNA=True,
                        NA_grid=NA_grid, CA_grid=CA_grid, merged_axis=merged_axis, resolution=resolution, grid_defs=grid_defs, 
                        period=period, var_pattern=var_pattern, grid_pattern=grid_pattern, vardefs=vardefs, axdefs=axdefs)


# Historical time-series
CMC_period = (1998,2015)
CMC_vardefs = dict(snowh = dict(grid='NA12', name='snowh', units='m', dtype=np.float32, scalefactor=0.01, # Snow depth, originally in cm
                               axes=('year','month',None,None),),) # this is the axes order in which the data are read                   
CMC_axdefs = dict(year = dict(name='year', units='year', coord=np.arange(CMC_period[0],CMC_period[1]+1)), # yearly coordinate
                  month = dict(name='month', units='month', coord=np.arange(1,13)),) # monthly coordinate - will be replaced
# N.B.: the time-series time offset has to be chose such that 1979 begins with the origin (time=0)
CMC_derived = ('snow','snow_acc',)
CMC_root = root_folder+'/CMC_hist/'
CMC_var_pattern = '{VAR:s}/ps_cmc_sdepth_analyses_{year:04d}_ascii/{year:04d}_{month:02d}_01.tif'
CMC_title = 'CMC Historical Gridded Snow Time-series'

# load normals (from different/unspecified periods... ), computer some derived variables, and combine NA and CA grids
def loadCMC_Hist(name='CMC', title=CMC_title, atts=None, derived_vars=CMC_derived, varatts=varatts, 
                 grid='NA12', resolution=12, grid_defs=None, period=CMC_period, lcheck=True, mask=None,
                 lmergeTime=False, # merge the year and month "axes" into a single monthly time axis 
                 snow_density=None,
                 var_pattern=CMC_var_pattern, data_root=CMC_root, vardefs=CMC_vardefs, axdefs=CMC_axdefs):
    ''' load CMC historical snow time-series from GeoTIFF files, merge with NRCan dataset and recompute snowmelt '''

    from utils.ascii import rasterDataset

    # determine grids / resolution
    if grid_defs is None: 
      grid_defs = grid_def # define in API; register for all pre-defined grids
    if resolution is not None:
      resolution = str(resolution)
      grid = 'NA{:s}'.format(resolution) if grid is None else grid.upper()            
    # update period
    if period is not None: # this is mainly for testing
      axdefs['year']['coord'] = np.arange(period[0],period[1]+1)
      
    # load NA grid
    dataset = rasterDataset(name=name, title=title, vardefs=vardefs, axdefs=axdefs, atts=atts, projection=None, 
                            griddef=grid_defs[grid], lgzip=None, lgdal=True, lmask=False, fillValue=0, lskipMissing=True, 
                            lgeolocator=False, file_pattern=data_root+var_pattern )    

    # merge year and month axes
    dataset = dataset.mergeAxes(axes=list(axdefs.keys()), axatts=varatts['time'], linplace=True)
    assert dataset.hasAxis('time'), dataset
    assert dataset.time[0] == 0, dataset.time.coord
    dataset.time.coord += 12 * ( axdefs['year']['coord'][0] - 1979 ) # set origin to Jan 1979! (convention)
    dataset.time.atts['long_name'] = 'Month since 1979-01'
    
    # apply mask
    if mask:
        if not isinstance(mask,Variable): raise TypeError(mask)
        dataset.mask(mask=mask) 
    
    # shift snow values by one month, since these values are for the 1st of the month
    snowh = dataset.snowh; tax = snowh.axisIndex('time'); tlen1 = snowh.shape[tax]-1
    assert lcheck is False or ( snowh.masked and np.all( snowh.data_array.mask.take([0], axis=tax) ) ), snowh.data_array.mask.take([0], axis=tax).sum()
    snowh.data_array = np.roll(snowh.data_array, -1, axis=tax) # there is no MA function, for some reason it works just fine... 
    assert lcheck is False or ( snowh.masked and np.all( snowh.data_array.mask.take([tlen1], axis=tax) ) ), snowh.data_array.mask.take([tlen1], axis=tax).sum()
    assert 'long_name' not in snowh.atts, snowh.atts['long_name']
    snowh.atts['long_name'] = "Snow Water Equivalent (end of month)"
    
    # compute derived variables
    for var in derived_vars:
        if var == 'snow':
            # compute snow water equivalent
            # before we can compute anything, we need estimates of snow density from a seasonal climatology
            density = getSnowDensity(snow_class=snow_density)
            density_note = "Snow density estimates from CMC for {:s} snow cover (Tab. 3): https://nsidc.org/data/NSIDC-0447/versions/1#title15".format(snow_density.title())
            # compute values and add to dataset
            newvar = monthlyTransform(var=dataset.snowh.copy(deepcopy=True), scalefactor=density, lvar=True, linplace=True)
            newvar.atts['long_name'] = 'Snow Water Equivalent at the end of the month.'
            newvar.atts['note'] = density_note
        elif var == 'snow_acc':
            # compute snow accumulation
            snow = dataset.snow; tax = snow.axisIndex('time'); data = snow[:]
            delta = ma.empty_like(data)
            assert tax == 0, snow            
            delta[1:,:] = ma.diff(data, axis=tax); delta[1,:] = ma.masked
            # N.B.: the snow/SWE date has already been shifted to the end of the month
            # create snow accumulation variable and divide by time
            newvar = Variable(data=delta, axes=snow.axes, name=var, units='kg/m^2/month')
            newvar = transformMonthly(var=newvar, slc=None, l365=False, lvar=True, linplace=True)
        # general stuff for all variables
        newvar = addGDALtoVar(newvar, griddef=dataset.griddef)
        dataset[var] = newvar
    # apply varatts
    for varname,var in list(dataset.variables.items()): 
        var.atts.update(varatts[varname]) # update in-place 
        # N.B.: 'long_name' and 'note' are not in varatts, and 'snow_acc

    # return dataset
    return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = norm12_grid_pattern+norm12_var_pattern # filename pattern: variable name and resolution
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: variable name and resolution
data_folder = avgfolder # folder for user data
grid_def = {'NA12':NRCan_NA12_grid, 'NA60':NRCan_NA60_grid, 'CA12':NRCan_CA12_grid, 'CA24':NRCan_CA24_grid} # standardized grid dictionary
LTM_grids = ['NA12','CA12','CA24'] # grids that have long-term mean data 
LTM_grids += ['na12_tundra','na12_taiga','na12_maritime','na12_ephemeral','na12_prairies','na12_alpine',] # some fake grids to accommodate different snow densities
TS_grids = ['NA12','NA60','CA12'] # grids that have time-series data
TS_grids += ['na60_'+var for var in varlist]
TS_grids += ['na12_tundra','na12_taiga','na12_maritime','na12_ephemeral','na12_prairies','na12_alpine',] # some fake grids to accommodate different snow densities
grid_res = {'NA12':1./12.,'NA60':1./60.,'CA12':1./12.,'CA24':1./24.} # no special name, since there is only one...
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
  
#     mode = 'test_climatology'
#     mode = 'test_timeseries'
#     mode = 'test_point_climatology'
#     mode = 'test_point_timeseries'
#     mode = 'convert_Normals'
#     mode = 'convert_Historical'
#     mode = 'convert_Daily'
    mode = 'convert_to_netcdf';
#     mode = 'add_CMC'
#     mode = 'test_CMC'
    pntset = 'glbshp' # 'ecprecip'
#     pntset = 'ecprecip'

    # period
#     period = (1970,2000)
    period = (1980,2010)
#     period = (2011,2019)
    # snow density/type
#     snow_density = 'ephemeral'
    snow_density = 'maritime'
#     snow_density = 'prairies'
#     snow_density = 'taiga'
#     snow_density = 'alpine'        

     
    res = None; grid = None
    
    if mode == 'convert_to_netcdf':

        from utils.ascii import convertRasterToNetCDF
        from time import time
        
        # parameters for daily ascii
        varlist = ['pcp',]
#         varlist = day12_vardefs.keys()
        grid_res = 'CA12'
        griddef = grid_def[grid_res]
        # parameters for rasters
        start_date = '2011-01-01'; end_date = '2011-02-01'; sampling = 'D'; loverwrite = True
#         start_date = '2011-01-01'; end_date = '2018-01-01'; sampling = 'D'; loverwrite = False
        raster_folder = root_folder + grid_res+'_Daily/'
        def raster_path_func(datetime, varname, **varatts):
            ''' determine path to appropriate raster for given datetime and variable'''
            day = datetime.dayofyear
            if not datetime.is_leap_year and day >= 60: day += 1
            altname = varatts.get('alt_name',varname)
            if varname in ('mint','maxt') and datetime.year in (2016,2017):
                path = '{VAR:s}/{YEAR:04d}/{ALT:s}/{YEAR:04d}_{DAY:d}.asc.gz'.format(YEAR=datetime.year, VAR=varname, ALT=altname, DAY=day)
            else:
                path = '{VAR:s}/{YEAR:04d}/{ALT:s}{YEAR:04d}_{DAY:d}.asc.gz'.format(YEAR=datetime.year, VAR=varname, ALT=altname, DAY=day)
            return path
        # NetCDF definitions
        ds_atts = dict(start_date=start_date, end_date=end_date, sampling=sampling)

        # start operation
        start = time()
        
        ## loop over variables (individual files)
        for varname in varlist:
            
            print("\n   ***   Reading rasters for variable '{}' ('{}')   ***   \n".format(varname,day12_vardefs[varname]['name']))
            
            nc_name = day12_vardefs[varname]['name']
            nc_filepath = daily_folder + netcdf_filename.format(VAR=nc_name, RES=grid_res).lower()
            vardef = {varname:day12_vardefs[varname]} # only one variable
            # read rasters and write to NetCDF file
            convertRasterToNetCDF(filepath=nc_filepath, raster_folder=raster_folder, raster_path_func=raster_path_func, vardefs=vardef, 
                                  start_date=start_date, end_date=end_date, sampling=sampling, ds_atts=ds_atts, griddef=griddef,
                                  loverwrite=loverwrite,)
            
            assert os.path.exists(nc_filepath), nc_filepath
            print('\nSaving to NetCDF-4 file:\n '+nc_filepath+'\n')
        
        # print timing
        end = time()
        print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))

        
        # inspect Dataset
        import xarray as xr
        xds = xr.open_dataset(nc_filepath, decode_cf=True, decode_times=True, decode_coords=True, use_cftime=True)
        print(xds)
        #print(ds.variables)
        #print(xds['time'])
    
    elif mode == 'convert_Daily':
        
        ## N.B.: this processes the entire dataset in-memory, which is not really feasible for daily data;
        #        use the convert_to_netcdf option above
        
        # parameters
#         snow_density = 'ephemeral'
        snow_density = 'maritime'
#         snow_density = 'prairies'
        if not os.path.exists(daily_folder): os.mkdir(daily_folder)
        # use actual, real values
        # NA12 grid
        title = day12_title; resolution = 12; grid_pattern  = day12_grid_pattern 
        vardefs = day12_vardefs; var_pattern = day12_var_pattern; derived_vars = day12_derived
        period = day12_period; split_axdefs = day12_axdefs; merged_atts = day12_matts         
        # test values
        varname = 'pcp'; period = (2014,2015); snow_density = None
        split_axdefs = dict(year= dict(name='year', units='year', coord=np.arange(2014,2015)),
                            day = dict(name='day', units='day', coord=np.arange(1,367)),) # time coordinate
        merged_atts = dict(name='time', units='day', long_name='Days since 2014-01-01', merged_axes = ('year','day'))
        vardefs = {varname:day12_vardefs[varname]}; derived_vars = None;
        file_tag = day12_vardefs[varname]['name'] # use common variable name as file tag
        file_tag += '_test'
        # test values
#         period = (1970,2000) # for production
#         period = (1981,2010) # for production
#         period = (1991,2000) # for testing
#         vardefs = dict(maxt = dict(grid='NA12', name='Tmax', units='K', offset=273.15, **hist_defaults), # 2m maximum temperature, originally in degrees Celsius
#                        mint = dict(grid='NA12', name='Tmin', units='K', offset=273.15, **hist_defaults), # 2m minimum temperature
#                        snwd = dict(grid='CA12', name='snowh', units='m', scalefactor=1./100., **hist_defaults), # snow depth
#                        pcp  = dict(grid='NA12', name='precip', units='kg/m^2/month', transform=transformMonthly, **hist_defaults),)
#         derived_vars = ('T2',)
        # load ASCII dataset with default values
        dataset = loadASCII_Daily(title=title, resolution=resolution, grid_pattern=grid_pattern, 
                                 vardefs=vardefs, var_pattern=var_pattern, derived_vars=derived_vars,                                  
                                 period=period, axdefs=split_axdefs, merged_axis=merged_atts,
                                 snow_density=snow_density, grid_defs=grid_def,)        
        # test 
        print(dataset)
        print('')
        print((dataset.precip))
        print(("\nVariable Size in memory: {:f} MB".format(dataset.precip.data_array.nbytes/1024./1024.)))
        # write to NetCDF
        grdstr = '_na{:d}_{:s}'.format(resolution, file_tag)
        ncfile = daily_folder + tsfile.format(grdstr)
        print('')
        writeNetCDF(dataset=dataset, ncfile=ncfile, ncformat='NETCDF4', zlib=True, writeData=True, overwrite=True, 
                    skipUnloaded=False, feedback=True, close=True)
        assert os.path.exists(ncfile), ncfile
        
    elif mode == 'test_climatology':
            
        # load averaged climatology file
        print('')
        dataset = loadNRCan(grid=grid,period=period,resolution=res, varatts=dict(pet=dict(name='pet_wrf')),
                            varlist=['liqwatflx_adj30'])
        print(dataset)
        print('')
        print((dataset.geotransform))
        print((dataset.liqwatflx.mean()))
        print((dataset.liqwatflx.masked))
        
        # print time coordinate
        print()
        print(dataset.time.atts)
        print()
        print(dataset.time.data_array)
          
    elif mode == 'test_timeseries':
      
        # load time-series file
        print('')
        dataset = loadNRCan_TS(grid=grid,resolution='na12_maritime')
        print(dataset)
        print('')
        print((dataset.time))
        print((dataset.time.coord))
        print((dataset.time.coord[29*12])) # Jan 1979
          
    if mode == 'test_point_climatology':
            
        # load averaged climatology file
        print('')
        if pntset in ('shpavg','glbshp'): 
            dataset = loadNRCan_Shp(shape=pntset, resolution=res, period=period)
            print((dataset.shp_area.mean()))
            print('')
        else: dataset = loadNRCan_Stn(station=pntset, resolution=res, period=period)
        dataset.load()
        print(dataset)
        print('')
        print((dataset['shape_name']))
        print('')
        print((dataset['shape_name'][:]))
        print('')
        print((dataset.filepath))

#         dataset = dataset(shape_name='GRW')
#         print(dataset)
#         print('')
#         print(dataset.atts.shp_area)
#         print(dataset.liqprec.mean()*86400)
#         print(dataset.precip.masked)
#         print(dataset.T2.mean())
#         print(dataset.atts.shp_empty,dataset.atts.shp_full,dataset.atts.shp_encl,)
        
#         # print time coordinate
#         print
#         print dataset.time.atts
#         print
#         print dataset.time.data_array

    elif mode == 'test_point_timeseries':
    
        # load station time-series file
        print('') 
        if pntset in ('shpavg',): dataset = loadNRCan_ShpTS(shape=pntset, resolution=res)
        else: dataset = loadNRCan_StnTS(station=pntset, resolution=res)
        print(dataset)
        print('')
        print((dataset.time))
        print((dataset.time.coord))
        assert dataset.time.coord[29*12] == 0 # Jan 1979
        assert dataset.shape[0] == 1
        
    elif mode == 'convert_Normals':
        
        # parameters
        prdstr = '_{}-{}'.format(*period)
        resolution = 12; grdstr = '_na{:d}_{:s}'.format(resolution, snow_density)
        ncfile = avgfolder + avgfile.format(grdstr,prdstr)
        if not os.path.exists(avgfolder): os.mkdir(avgfolder)
        # load ASCII dataset with default values
        dataset = loadASCII_Normals(period=period, resolution=resolution, snow_density=snow_density, grid_defs=grid_def,)        
        # test 
        print(dataset)
        print('')
        print((dataset.snow))
        # write to NetCDF
        print('')
        writeNetCDF(dataset=dataset, ncfile=ncfile, ncformat='NETCDF4', zlib=True, writeData=True, overwrite=True, 
                    skipUnloaded=False, feedback=True, close=True)
        assert os.path.exists(ncfile), ncfile
        
    elif mode == 'convert_Historical':
        
        # parameters
#         snow_density = 'ephemeral'
        snow_density = 'maritime'
#         snow_density = 'prairies'
        if not os.path.exists(avgfolder): os.mkdir(avgfolder)
        # use actual, real values
        # NA12 grid
        title = mons12_title; resolution = 12; grid_pattern  = mons12_grid_pattern 
        vardefs = mons12_vardefs; var_pattern = mons12_var_pattern; derived_vars = mons12_derived
        period = mons12_period; split_axdefs = mons12_axdefs; merged_atts = mons12_matts         
        file_tag = snow_density
        # NA60 grid
        varname = 'pcp'; period = (2011,2018); snow_density = None
        title = mons60_title; resolution = 60; grid_pattern  = mons60_grid_pattern 
        vardefs = {varname:mons60_vardefs[varname]} 
        var_pattern = mons60_var_pattern; derived_vars = None # mons60_derived
        split_axdefs = mons60_axdefs; merged_atts = mons60_matts         
        file_tag = mons60_vardefs[varname]['name'] # use common variable name as file tag
        # test values
#         period = (1970,2000) # for production
#         period = (1981,2010) # for production
#         period = (1991,2000) # for testing
#         vardefs = dict(maxt = dict(grid='NA12', name='Tmax', units='K', offset=273.15, **hist_defaults), # 2m maximum temperature, originally in degrees Celsius
#                        mint = dict(grid='NA12', name='Tmin', units='K', offset=273.15, **hist_defaults), # 2m minimum temperature
#                        snwd = dict(grid='CA12', name='snowh', units='m', scalefactor=1./100., **hist_defaults), # snow depth
#                        pcp  = dict(grid='NA12', name='precip', units='kg/m^2/month', transform=transformMonthly, **hist_defaults),)
#         derived_vars = ('T2',)
        # load ASCII dataset with default values
        dataset = loadASCII_Hist(title=title, resolution=resolution, grid_pattern=grid_pattern, 
                                 vardefs=vardefs, var_pattern=var_pattern, derived_vars=derived_vars,                                  
                                 period=period, axdefs=split_axdefs, merged_axis=merged_atts,
                                 snow_density=snow_density, grid_defs=grid_def,)        
        # test 
        print(dataset)
        print('')
        print((dataset.precip))
        # write to NetCDF
        grdstr = '_na{:d}_{:s}'.format(resolution, file_tag)
        ncfile = avgfolder + tsfile.format(grdstr)
        print('')
        writeNetCDF(dataset=dataset, ncfile=ncfile, ncformat='NETCDF4', zlib=True, writeData=True, overwrite=True, 
                    skipUnloaded=False, feedback=True, close=True)
        assert os.path.exists(ncfile), ncfile
        
    elif mode == 'add_CMC':
        
        ## SWE correction for CMC data
        scale_tag = ''
        scale_factor = 1.
        scale_note = None
#         scale_tag = '_adj30'
#         scale_factor = 3.
#         scale_note = 'CMC SWE data has been scaled by 3.0 to match NRCan SWE over Canada'
#         scale_tag = '_adj35'
#         scale_factor = 3.5
#         scale_note = 'CMC SWE data has been scaled by 3.5 to match NRCan SWE over Canada'
        
#         CMC_period = (1998,1999) # for tests
#         filelist = ['test_' + avgfile.format('_na{:d}'.format(12),'_1970-2000')]
        filelist = None
        
        # load NRCan dataset (for precip and to add variables)
        nrcan = loadNRCan(filelist=filelist, period=period, filemode='rw', snow_density=snow_density).load()

        # load ASCII dataset with default values
        cmc = loadCMC_Hist(period=CMC_period, mask=nrcan.landmask, snow_density=snow_density)        
        # test 
        print(cmc)
        # climatology
        print('')
        cmc = cmc.climMean()
#         print(cmc)
        # apply scale factor
        for varname,var in list(cmc.variables.items()):
            if varname.lower().startswith('snow'):
                if scale_factor != 1:
                    var *= scale_factor # scale snow/SWE variables
                    # N.B.: we are mainly using SWE differences, but this is all linear...
        # values
        print('')
        var = cmc.snow_acc.mean(axes=('lat','lon'))
        print((var[:]))
        print('')
        for varname,var in list(cmc.variables.items()):
            if var.masked:
                print((varname, float(var.data_array.mask.sum())/float(var.data_array.size)))
        # add liquid water flux, based on precip and snow accumulation/storage changes
        print('')
        lwf = 'liqwatflx'; data = ( nrcan.precip[:] - cmc.snow_acc[:] ).clip(min=0) # clip smaller than zero
        cmc[lwf] = addGDALtoVar(Variable(data=data, axes=cmc.snow_acc.axes, atts=varatts[lwf]), griddef=cmc.griddef)        
        print((cmc[lwf])) 
        # values
        print('')
        var = cmc[lwf].mean(axes=('lat','lon'))
        print((var[:]))
        
        # create merged lwf and add to NRCan
        for varname in (lwf,'snow','snowh'):
            if varname+'_NRCan' in nrcan:
                nrcan_var = nrcan[varname+'_NRCan'] 
            else:
                nrcan_var = nrcan[varname].load().copy(deepcopy=True) # load liqwatflx and rename
                nrcan[varname+'_NRCan'] = nrcan_var
            varname_tag = varname + scale_tag
            new_var = nrcan_var.copy(deepcopy=False) # replace old variable
            data = np.where(nrcan_var.data_array.mask,cmc[varname].data_array,nrcan_var.data_array)
            new_var.data_array = data
            new_var.atts['note'] = 'merged data from NRCan and CMC'
            if scale_note: new_var.atts['note'] = new_var.atts['note'] + '; ' + scale_note
            if varname == lwf: new_var.atts['long_name'] = 'Merged Liquid Water Flux'
            if varname == 'snow': new_var.atts['long_name'] = 'Merged Snow Water Equivalent'
            if varname == 'snowh': new_var.atts['long_name'] = 'Merged Snow Depth'
            new_var.fillValue = -999.
            # save variable in NRCan dataset
            if varname_tag in nrcan: del nrcan[varname_tag] # remove old variable
            nrcan[varname_tag] = new_var
        print((nrcan[lwf+scale_tag]))
        # add other CMC variables to NRCan datasets
        for varname,var in list(cmc.variables.items()):
            if varname in CMC_derived or varname in CMC_vardefs or varname == lwf:
                if scale_note: var.atts['note'] = scale_note
                cmc_var = varname+'_CMC'+scale_tag
                if cmc_var in nrcan: del nrcan[cmc_var] # overwrite existing
                nrcan[cmc_var] = var
        print('')
        print(nrcan)
        # save additional variables
        nrcan.close(); del nrcan # implies sync
        
        # now check
        print('')
        nrcan = loadNRCan(filelist=filelist, period=period, snow_density=snow_density)
        print(nrcan)
        print(("\nNetCDF file path:\n '{}'".format(nrcan.filelist[0])))
        print('')
        for varname,var in list(cmc.variables.items()):
            if varname in CMC_derived or varname in CMC_vardefs:
                assert varname+'_CMC'+scale_tag in nrcan, nrcan
#             print('')
#             print(nrcan[varname+'_CMC'])
        
    elif mode == 'test_CMC':
        
        # load ASCII dataset with default values
        period = (1998,2000)
        cmc = loadCMC_Hist(period=period, lcheck=True)        
        # test 
        print(cmc)
        assert cmc.time[0] == 12*(period[0]-1979), cmc.time[:] 
        # climatology
        print('')
        cmc = cmc.climMean()
#         print(cmc)
        # values
        print('')
        var = cmc.snow.mean(axes=('lat','lon'))
        print((var[:]))
        for varname,var in list(cmc.variables.items()):
            print((varname, var.masked, float(var.data_array.mask.sum())/float(var.data_array.size)))
        
