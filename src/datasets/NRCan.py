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
from geodata.base import Variable
from geodata.gdal import GridDefinition, addGDALtoVar
from datasets.common import data_root, loadObservations, transformMonthly, addLengthAndNamesOfMonth,\
  monthlyTransform
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
varatts = dict(Tmax    = dict(name='Tmax', units='K'), # 2m maximum temperature
               Tmin    = dict(name='Tmin', units='K'), # 2m minimum temperature
               precip  = dict(name='precip', units='kg/m^2/s'), # total precipitation
               pet     = dict(name='pet', units='kg/m^2/s'), # potential evapo-transpiration
               liqprec = dict(name='liqprec', units='kg/m^2/s'), # total precipitation
               snowh   = dict(name='snowh', units='m'), # snow depth
               SWD     = dict(name='SWD', units='J/m^2/s'), # solar radiation
               # diagnostic variables
               T2      = dict(name='T2', units='K'), # 2m average temperature
               solprec = dict(name='liqprec', units='kg/m^2/s'), # total precipitation
               snow    = dict(name='snow', units='kg/m^2'), # snow water equivalent
               snwmlt  = dict(name='snwmlt', units='kg/m^2/s'), # snow melt (rate)
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
  # figure out resolution and grid
  if resolution is None: 
      if grid is None:  
          resolution = 'na12'
      else: 
          resolution = grid
          grid = None
  if not isinstance(resolution, basestring): raise TypeError(resolution) 
  # figure out clim/TS
  if period is not None: lclim=True
  # check for valid resolution 
  if lclim and resolution.upper() not in LTM_grids: 
      raise DatasetError("Selected resolution '{:s}' is not available for long-term means!".format(resolution))
  if not lclim and resolution.upper() not in TS_grids: 
      raise DatasetError("Selected resolution '{:s}' is not available for historical time-series!".format(resolution))
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
                               grid=grid, varlist=varlist, varatts=varatts, filepattern=avgfile, griddef=NRCan_NA12_grid,
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
norm_derived = ('T2','solprec','snow','snwmlt')
norm_grid_pattern = root_folder+'{GRID:s}_normals/' # dataset root folder
norm_var_pattern = '{VAR:s}/{VAR:s}_{time:02d}.asc.gz' # path to variables
norm_title = 'NRCan Gridded Normals'

# load normals (from different/unspecified periods... ), computer some derived variables, and combine NA and CA grids
def loadASCII_Normals(name=dataset_name, title=norm_title, atts=None, derived_vars=norm_derived, varatts=varatts, 
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
        data = np.ma.empty(var.shape[:-2]+nashp, dtype=var.dtype) # use the shape of the NA grid and other axes from the original
        data[:] = np.ma.masked # everything that is not explicitly assigned, shall be masked
        data[...,jos:jos+caje,ios:ios+caie] = var.data_array # assign partial data
        # figure out axes and create Variable
        axes = [naaxes[ax.name] for ax in var.axes]
        newvar = Variable(name=key, units=var.units, axes=axes, data=data, atts=var.atts, plot=var.plot)
        newvar = addGDALtoVar(newvar, griddef=dataset.griddef,)
        dataset.addVariable(newvar, copy=False)
    # snow needs some special care: replace mask with mask from rain and set the rest to zero
    assert dataset.snowh.shape == dataset.liqprec.shape, dataset
    snwd = ma.masked_where(condition=dataset.liqprec.data_array.mask, a=dataset.snowh.data_array.filled(0), copy=False)
    dataset.snowh.data_array = snwd # reassingment is necessary, because filled() creates a copy
    dataset.snowh.fillValue = dataset.liqprec.fillValue 
    assert np.all( dataset.snowh.data_array.mask == dataset.liqprec.data_array.mask ), dataset.snowh.data_array
    assert dataset.snowh.fillValue == dataset.liqprec.fillValue, dataset.snowh.data_array
    
    # compute some secondary/derived variables
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
            # N.B.: before we can compute anything, we need estimates of snow density; the values below are seasonal
            #       estimates from the Canadian Meteorological Centre for maritime climates (Table 4):
            #       https://nsidc.org/data/docs/daac/nsidc0447_CMC_snow_depth/
            #       a factor of 1000 has been applied, because snow depth is in m (and not mm)
            # Maritime snow cover
#             density = np.asarray([0.2165, 0.2485, 0.2833, 0.332, 0.3963, 0.501, 0.501, 0.501, 0.16, 0.16, 0.1835, 0.1977], dtype=np.float32)*1000.
            # Ephemeral snow cover
            density = np.asarray([0.3168, 0.3373, 0.3643, 0.4046, 0.4586, 0.5098, 0.5098, 0.5098, 0.25, 0.25, 0.3, 0.3351], dtype=np.float32)*1000.
            # Prairie snow cover
#             density = np.asarray([0.2137, 0.2416, 0.2610, 0.308, 0.3981, 0.4645, 0.4645, 0.4645, 0.14, 0.14, 0.1616, 0.1851], dtype=np.float32)*1000.
            # Note: these snow density values are for maritime climates only! values for the Prairies and the North are 
            #       substantially different! this is for applications in southern Ontario
            # compute values and add to dataset
            dataset[var] = monthlyTransform(var=dataset.snowh.copy(deepcopy=True), lvar=True, linplace=True, scalefactor=density)
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
            data = -1 * ( ma.concatenate((dd,delta), axis=0) + ma.concatenate((delta,dd), axis=0) ) / 2.
            # create snowmelt variable and do some conversions
            newvar = addGDALtoVar(Variable(data=data, axes=snow.axes, name=var, units='kg/m^2/month'), griddef=dataset.griddef)
            newvar = transformMonthly(var=newvar, slc=None, l365=False, lvar=True, linplace=True)
            newvar += dataset.solprec # ad that in-place as well
            newvar.data_array.clip(min=0, out=newvar.data_array) # clip values smaller than zero (in-place)
            dataset[var] = newvar
            # normalize snowmelt so that it does not exceed snow fall
            r = dataset.snwmlt.mean(axis=0,keepdims=True,asVar=False)/dataset.solprec.mean(axis=0,keepdims=True,asVar=False)
            rm = r.mean()
            print("\nSnowmelt to snowfall ratio: {}\n".format(rm))            
            if rm > 1:
              #r0 = dataset.snwmlt.mean(axis=0,keepdims=True,asVar=False)/dataset.solprec.mean(axis=0,keepdims=True,asVar=False) 
              dataset.snwmlt.data_array /= r # normalize to total snow fall annually and grid point-wise
            assert np.ma.allclose(dataset.snwmlt.mean(axis=0,asVar=False), dataset.solprec.mean(axis=0,asVar=False)), dataset.snwmlt.mean()/dataset.solprec.mean()
            # add snow ratio as diagnostic
            atts = dict(name='ratio', units='', long_name='Ratio of Snowfall to Snowmelt')
            dataset += addGDALtoVar(Variable(data=r.squeeze(), axes=snow.axes[1:], atts=atts), griddef=dataset.griddef)    
        else: raise VariableError(var)
        # for completeness, add attributes
        dataset[var].atts = varatts[var]
    
    # add length and names of month
    addLengthAndNamesOfMonth(dataset)
    
    # return properly formatted dataset
    return dataset

# Historical time-series
hist_vardefs = NotImplemented
hist_axdefs = NotImplemented
# N.B.: the time-series time offset has to be chose such that 1979 begins with the origin (time=0)
hist_derived = NotImplemented
hist_grid_pattern = data_root+'{GRID:s}_hist/'
hist_var_pattern = '{VAR:s}/{year:04d}/{VAR:s}_{month:02d}.asc.gz'
hist_title = 'NRCan Historical Gridded Time-series'

# load normals (from different/unspecified periods... ), computer some derived variables, and combine NA and CA grids
def loadASCII_Hist(name=dataset_name, title=hist_title, atts=None, derived_vars=hist_derived, varatts=varatts, 
                   NA_grid=None, CA_grid=None, resolution=12, grid_defs=None,
                   lmergeTime=False, # merge the year and month "axes" into a single monthly time axis 
                   var_pattern=hist_var_pattern, grid_pattern=hist_grid_pattern, vardefs=hist_vardefs, axdefs=hist_axdefs):
    ''' load NRCan historical time-series from ASCII files, merge CA and NA grids and compute some additional variables; 
        merge year and month axes and return as Dataset '''
  
    # load exactly like normals and let it merge the grids
    dataset = loadASCII_Normals(name=dataset_name, title=hist_title, atts=atts, derived_vars=hist_derived, 
                                varatts=varatts, NA_grid=NA_grid, CA_grid=CA_grid, 
                                resolution=resolution, grid_defs=grid_defs, var_pattern=hist_var_pattern, 
                                grid_pattern=hist_grid_pattern, vardefs=hist_vardefs, axdefs=hist_axdefs)
    
    # merge different time axes
    if lmergeTime is not None:
        raise NotImplementedError
        # use dataset.mergeAxes() to merge year and month, but need to come up with something to recenter at 1979
    
    # return dataset
    return dataset


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
  
    mode = 'test_climatology'
#     mode = 'test_timeseries'
#     mode = 'test_point_climatology'
#     mode = 'test_point_timeseries'
#     mode = 'convert_Normals'
    pntset = 'shpavg' # 'ecprecip'
    period = None; res = 'na12'; grid = None
    
    if mode == 'test_climatology':
            
        # load averaged climatology file
        print('')
        dataset = loadNRCan(grid=grid,period=period,resolution=res)
        print(dataset)
        print('')
        print(dataset.geotransform)
        print(dataset.precip.getArray().mean())
        print(dataset.precip.masked)
        
        # print time coordinate
        print
        print dataset.time.atts
        print
        print dataset.time.data_array
          
    elif mode == 'test_timeseries':
      
        # load time-series file
        print('')
        dataset = loadNRCan_TS(grid=grid,resolution=res)
        print(dataset)
        print('')
        print(dataset.time)
        print(dataset.time.coord)
        print(dataset.time.coord[78*12]) # Jan 1979
          
    if mode == 'test_point_climatology':
            
        # load averaged climatology file
        print('')
        if pntset in ('shpavg',): 
            dataset = loadNRCan_Shp(shape=pntset, resolution=res, period=period)
            print(dataset.shp_area.mean())
            print('')
        else: dataset = loadNRCan_Stn(station=pntset, resolution=res, period=period)
        dataset.load()
        print(dataset)
        print('')
        print(dataset.precip.mean())
        print(dataset.precip.masked)
        
        # print time coordinate
        print
        print dataset.time.atts
        print
        print dataset.time.data_array

    elif mode == 'test_point_timeseries':
    
        # load station time-series file
        print('') 
        if pntset in ('shpavg',): dataset = loadNRCan_ShpTS(shape=pntset, resolution=res)
        else: dataset = loadNRCan_StnTS(station=pntset, resolution=res)
        print(dataset)
        print('')
        print(dataset.time)
        print(dataset.time.coord)
        assert dataset.time.coord[78*12] == 0 # Jan 1979
        assert dataset.shape[0] == 1
        
    elif mode == 'convert_Normals':
        
        # parameters
        resolution = 12; grdstr = '_na{:d}'.format(resolution); prdstr = ''
        ncfile = avgfolder + avgfile.format(grdstr,prdstr)
        if not os.path.exists(avgfolder): os.mkdir(avgfolder)
        # load ASCII dataset with default values
        dataset = loadASCII_Normals(name='NRCan', title='NRCan Test Dataset', atts=None, 
                                    NA_grid=None, CA_grid=None, resolution=resolution, grid_defs=grid_def,)        
        # test 
        print(dataset)
        # write to NetCDF
        print('')
        writeNetCDF(dataset=dataset, ncfile=ncfile, ncformat='NETCDF4', zlib=True, writeData=True, overwrite=True, 
                    skipUnloaded=False, feedback=True, close=True)
        assert os.path.exists(ncfile), ncfile