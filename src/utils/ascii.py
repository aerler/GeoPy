'''
Created on Jan 4, 2017

A module to load ASCII raster data into numpy arrays.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import gzip, shutil, tempfile
import os, gc
# internal imports
from geodata.base import Variable, Axis, Dataset
from geodata.gdal import addGDALtoDataset, addGDALtoVar, getAxes
from geodata.misc import AxisError, ArgumentError, NetCDFError
from utils.misc import flip, expandArgumentList
from utils.nctools import coerceAtts

# the environment variable RAMDISK contains the path to the RAM disk
ramdisk = os.getenv('RAMDISK', None)
if ramdisk and not os.path.exists(ramdisk): 
  raise IOError(ramdisk)


# helper function
def samplingUnits(sampling):
    ''' convert Pandas sampling letter to time units'''
    if sampling == 's': units = 'seconds'
    elif sampling == 'm': units = 'minutes'
    elif sampling == 'D': units = 'days'
    elif sampling == 'M': units = 'months'
    elif sampling == 'Y': units = 'years'
    else: 
        raise NotImplementedError(sampling)
    return units 

## functions to convert a raster dataset to NetCDF (time-step by time-step)

def convertRasterToNetCDF(filepath=None, raster_folder=None, raster_path_func=None, start_date=None, end_date=None, sampling='M', 
                          ds_atts=None, vardefs=None, projection=None, geotransform=None, size=None,  griddef=None, 
                          lgzip=None, lgdal=True, lmask=True, lskipMissing=True, lfeedback=True, var_start_idx=None,
                          loverwrite=False, use_netcdf_tools=True, lzlib=True, **ncargs):
    ''' function to load a set of raster variables that are stored in a systematic directory tree into a NetCDF dataset
        Variables are defined as follows:
          vardefs[varname] = dict(name=string, units=string, axes=tuple of strings, atts=dict, plot=dict, dtype=np.dtype, fillValue=value)
        Currently, only the horizontal raster axes and a datetime axis are supported; the former are inferred from griddef and 
        the latter is constructed from start_date, end_date and sampling and stored following CF convention. 
        The path to raster files is constructed as raster_folder+raster_path, where raster_path is the output of 
        raster_path_func(datetime, varname, **varatts), which has to be defined by the user.        
    '''
    import pandas as pd
    import netCDF4 as nc
    from utils.nctools import add_coord, add_var, coerceAtts
      
    ## open NetCDF dataset
    if isinstance(filepath,str) and ( loverwrite or not os.path.exists(filepath) ):
          
        # generate list of datetimes from end dates and frequency
        datetime64_array = np.arange(start_date,end_date, dtype='datetime64[{}]'.format(sampling))
        xlon, ylat = (griddef.xlon, griddef.ylat) # used for checking
    
        if use_netcdf_tools:
            from geospatial.netcdf_tools import createGeoNetCDF
            ncds = createGeoNetCDF(filepath, atts=ds_atts, time=datetime64_array, varatts=None, # default atts 
                                   crs=projection, geotrans=geotransform, size=size, griddef=griddef, # probably griddef...
                                   nc_format=ncargs.get('format','NETCDF4'), zlib=lzlib, loverwrite=loverwrite)
        else:
            if 'format' not in ncargs: ncargs['format'] = 'NETCDF4'
            ncargs['clobber'] = loverwrite
            if loverwrite: ncargs['mode'] = 'w' 
            ncds = nc.Dataset(filepath, **ncargs)
            # setup horizontal dimensions and coordinate variables
            for ax in (xlon, ylat):
                add_coord(ncds, name=ax.name, data=ax.coord, atts=ax.atts, zlib=lzlib)
            # create time dimension and coordinate variable
            start_datetime = datetime64_array[0]
            time_coord = ( ( datetime64_array - start_datetime ) / np.timedelta64(1,sampling) ).astype('int64')
            time_units_name = samplingUnits(sampling)
            time_units = time_units_name + ' since ' + str(start_datetime)
            tatts = dict(units=time_units, long_name=time_units_name.title(), sampling=sampling, start_date=start_date, end_date=end_date)
            add_coord(ncds, name='time', data=time_coord, atts=tatts, zlib=lzlib)
            # add attributes
            if ds_atts is not None:
                for key,value in coerceAtts(ds_atts).items(): 
                    ncds.setncattr(key,value)
        
    elif isinstance(filepath,nc.Dataset) or ( isinstance(filepath,str) and os.path.exists(filepath) ):
        
        assert not loverwrite, filepath
        
        if isinstance(filepath,str):
            # open exising dataset
            ncargs['clobber'] = False
            ncargs['mode'] = 'a' # append 
            ncds = nc.Dataset(filepath, **ncargs)                
        else:
            ncds = filepath 
        
        # check mapping stuff
        if griddef is None:
            raise NotImplementedError("Inferring GridDef from file is not implemented yet.")
        assert griddef.isProjected == ncds.getncattr('is_projected'), griddef.projection.ExportToProj4()
        xlon, ylat = (griddef.xlon, griddef.ylat)
        for ax in (xlon, ylat):
            assert ax.name in ncds.dimensions, ax.name
            assert len(ax) == len(ncds.dimensions[ax.name]), ax.name
            assert ax.name in ncds.variables, ax.name
        
        # set time interval    
        if start_date is not None:
            print("Overwriting start date with NetCDF start date.")
        start_date = ncds.getncattr('start_date')
        if lfeedback: print("Setting start date to:",start_date)
        start_date_dt = pd.to_datetime(start_date)
        # check existing time axis
        assert 'time' in ncds.dimensions, ncds.dimensions
        assert 'time' in ncds.variables, ncds.variables
        nc_end_date_dt = pd.to_datetime(ncds.getncattr('end_date'))
        # check consistency of dates
        days = (nc_end_date_dt - start_date_dt).days # timedelta
        if sampling.upper() == 'D': time_len = days
        else: raise NotImplementedError(sampling)
        if time_len != len(ncds.dimensions['time']):
            raise NetCDFError("Registered start and end dates are inconsistent with array length: {} != {}".format(time_len,len(ncds.dimensions['time'])))
        
        # generate list of datetimes from end dates and frequency
        datetime64_array = np.arange(start_date,end_date, dtype='datetime64[{}]'.format(sampling))
        # verify time coordinate
        start_datetime = datetime64_array[0]
        time_coord = ( ( datetime64_array - start_datetime ) / np.timedelta64(1,sampling) ).astype('int64')
        time_units = samplingUnits(sampling) + ' since ' + str(start_datetime)
        tvar = ncds.variables['time']
        assert tvar.units == time_units, tvar
        assert np.all(tvar[:time_len] == time_coord[:time_len]), time_coord
        # update time coordinate
        tvar[time_len:] = time_coord[time_len:]
        # update time stamps
        if 'time_stamp' in ncds.variables:
            ncds.variables['time_stamp'][time_len:] = np.datetime_as_string(datetime64_array[time_len:], unit=sampling)
        
    else:
        raise TypeError(filepath)
                
    # add variables
    if var_start_idx is None: var_start_idx = dict()
    dimlist = ('time', ncds.ylat, ncds.xlon)
    shape = tuple(len(ncds.dimensions[dim]) for dim in dimlist) # xlon is usually the innermost dimension
    var_shp = (None,)+shape[1:] # time dimension remains extendable
    for varname,varatts in vardefs.items():
        nc_name = varatts.get('name',varname)
        if nc_name in ncds.variables:
            ncvar = ncds.variables[nc_name]
            assert ncvar.shape == shape, shape # time dim is already updated/extended due to time coordinate
            assert ncvar.dimensions == ('time',ylat.name,xlon.name), ('time',ylat.name,xlon.name)
            if varname not in var_start_idx:
                var_start_idx[varname] = time_len # where to start writing new data (can vary by variable)
        else:
            atts = dict(original_name=varname, units=varatts['units'])
            add_var(ncds, name=nc_name, dims=dimlist, data=None, shape=var_shp, atts=atts, 
                    dtype=varatts['dtype'], zlib=lzlib, lusestr=True)
            var_start_idx[varname] = 0
    ncds.sync()
    #print(filepath)
    #print(ncds)
    
    if not lmask: 
        raise NotImplementedError("Need to handle missing values without mask - use lna=True?")

    # get fillValues
    fillValues = {varname:varatts.get('fillValue',None) for varname,varatts in vardefs.items()}
                
    ## loop over datetimes
    for i,dt64 in enumerate(datetime64_array):
        
        datetime = pd.to_datetime(dt64)
        
        ## loop over variables
        for varname,varatts in vardefs.items():
          
            i0 = var_start_idx[varname]
            # skip existing data
            if i >= i0:
                # actually add data now
                if i0 == i and i > 0 and lfeedback:
                    print("{}: appending after {} timesteps.".format(varname,i))
                nc_name = varatts.get('name',varname)
                fillValue = fillValues[varname]
                # construct file names
                raster_path = raster_path_func(datetime, varname, **varatts)
                raster_path = raster_folder + raster_path
                if lfeedback: print(raster_path)
                # load raster data and save to NetCDF
                if lgzip is None: # will only trigger once
                    lgzip = raster_path.endswith('.gz')
                if os.path.exists(raster_path):
                    raster_data, geotrans, nodata = readASCIIraster(raster_path, lgzip=lgzip, lgdal=lgdal, dtype=varatts.get('dtype',np.float32), 
                                                                    lmask=lmask, fillValue=varatts.get('fillValue',None), lgeotransform=True, 
                                                                    lna=True)
                    assert all(np.isclose(geotrans,griddef.geotransform)), geotrans           
                    if fillValue is None: 
                        fillValues[varname] = nodata # remember for next field
                    elif fillValue != nodata:
                        raise NotImplementedError('No data/fill values need to be consistent: {} != {}'.format(fillValue,nodata)) 
                    # scale, if appropriate
                    if 'scalefactor' in varatts: raster_data *= varatts['scalefactor']
                    if 'offset' in varatts: raster_data += varatts['offset']
                elif lskipMissing:
                    print("Skipping missing raster: '{}'".format(raster_path))  
                    # create an array of missing data
                    if lmask:
                        raster_data = np.ma.masked_all(shape=var_shp[1:], dtype=varatts['dtype']) # no time dim; all masked
                    elif fillValue is not None:
                        raster_data = np.full(shape=var_shp[1:], fill_value=fillValue, dtype=varatts['dtype']) # no time dim; all filled
                    else:
                        NotImplementedError("Need to be able to generate missing data in order to skip missing raster.")
                # save data to NetCDF
                ncds.variables[nc_name][i,:,:] = raster_data
        ## maybe compute some derived variables?

    # close file
    ncds.sync(); ncds.close()
    return filepath

## functions to construct Variables and Datasets from ASCII raster data

def rasterDataset(name=None, title=None, vardefs=None, axdefs=None, atts=None, projection=None, griddef=None,
                  lgzip=None, lgdal=True, lmask=True, fillValue=None, lskipMissing=True, lgeolocator=True,
                  file_pattern=None, lfeedback=True, **kwargs):
    ''' function to load a set of variables that are stored in raster format in a systematic directory tree into a Dataset
        Variables and Axis are defined as follows:
          vardefs[varname] = dict(name=string, units=string, axes=tuple of strings, atts=dict, plot=dict, dtype=np.dtype, fillValue=value)
          axdefs[axname]   = dict(name=string, units=string, atts=dict, coord=array or list) or None
        The path to raster files is constructed as variable_pattern+axes_pattern, where axes_pattern is defined through the axes, 
        (as in rasterVarialbe) and variable_pattern takes the special keywords VAR, which is the variable key in vardefs.
    '''
  
    ## prepare input data and axes
    if griddef: 
        xlon,ylat = griddef.xlon,griddef.ylat
        if projection is None: 
            projection = griddef.projection
        elif projection != griddef.projection:
            raise ArgumentError("Conflicting projection and GridDef!")
        geotransform = griddef.geotransform
        isProjected = griddef.isProjected
    else: 
        xlon = ylat = geotransform = None
        isProjected = False if projection is None else True
    # construct axes dict
    axes = dict()
    for axname,axdef in list(axdefs.items()):
        assert 'coord' in axdef, axdef
        assert ( 'name' in axdef and 'units' in axdef ) or 'atts' in axdef, axdef
        if axdef is None:
            axes[axname] = None
        else:
            ax = Axis(**axdef)
            axes[ax.name] = ax
    # check for map Axis
    if isProjected:
        if 'x' not in axes: axes['x'] = xlon
        if 'y' not in axes: axes['y'] = ylat
    else:
        if 'lon' not in axes: axes['lon'] = xlon
        if 'lat' not in axes: axes['lat'] = ylat
      
    ## load raster data into Variable objects
    varlist = []
    for varname,vardef in list(vardefs.items()):
        # check definitions
        assert 'axes' in vardef and 'dtype' in vardef, vardef
        assert ( 'name' in vardef and 'units' in vardef ) or 'atts' in vardef, vardef 
        # determine relevant axes
        vardef = vardef.copy()
        axes_list = [None if ax is None else axes[ax] for ax in vardef.pop('axes')]
        # define path parameters (with varname)
        path_params = vardef.pop('path_params',None)
        path_params = dict() if path_params is None else path_params.copy()
        if 'VAR' not in path_params: path_params['VAR'] = varname # a special key
        # add kwargs and relevant axis indices
        relaxes = [ax.name for ax in axes_list if ax is not None] # relevant axes
        for key,value in list(kwargs.items()):
          if key not in axes or key in relaxes:
              vardef[key] = value
        # create Variable object
        var = rasterVariable(projection=projection, griddef=griddef, file_pattern=file_pattern, lgzip=lgzip, lgdal=lgdal, 
                             lmask=lmask, lskipMissing=lskipMissing, axes=axes_list, path_params=path_params, 
                             lfeedback=lfeedback, **vardef)
        # vardef components: name, units, atts, plot, dtype, fillValue
        varlist.append(var)
        # check that map axes are correct
        for ax in var.xlon,var.ylat:
            if axes[ax.name] is None: axes[ax.name] = ax
            elif axes[ax.name] != ax: raise AxisError("{} axes are incompatible.".format(ax.name))
        if griddef is None: griddef = var.griddef
        elif griddef != var.griddef: raise AxisError("GridDefs are inconsistent.")
        if geotransform is None: geotransform = var.geotransform
        elif geotransform != var.geotransform: 
            raise AxisError("Conflicting geotransform (from Variable) and GridDef!\n {} != {}".format(var.geotransform,geotransform))
        
    ## create Dataset
    # create dataset
    dataset = Dataset(name=name, title=title, varlist=varlist, axes=axes, atts=atts)
    # add GDAL functionality
    dataset = addGDALtoDataset(dataset, griddef=griddef, projection=projection, geotransform=geotransform, gridfolder=None, 
                               lwrap360=None, geolocator=lgeolocator, lforce=False)
    # N.B.: for some reason we also need to pass the geotransform, otherwise it is recomputed internally and some consistency
    #       checks fail due to machine-precision differences
    
    # return GDAL-enabled Dataset
    return dataset
    

def rasterVariable(name=None, units=None, axes=None, atts=None, plot=None, dtype=None, projection=None, griddef=None,
                   file_pattern=None, lgzip=None, lgdal=True, lmask=True, fillValue=None, lskipMissing=True, 
                   path_params=None, offset=0, scalefactor=1, transform=None, time_axis=None, lfeedback=False, **kwargs):
    ''' function to read multi-dimensional raster data and construct a GDAL-enabled Variable object '''

    # print status
    if lfeedback: print("Loading variable '{}': ".format(name), end=' ')  # no newline

    ## figure out axes arguments and load data
    # figure out axes (list/tuple of axes has to be ordered correctly!)
    axes_list = [ax.name for ax in axes[:-2]]
    # N.B.: the last two axes are the two horizontal map axes (x&y); they can be None and will be inferred from raster
    # N.B.: coordinate values can be overridden with keyword arguments, but length must be consistent
    # figure out coordinates for axes
    for ax in axes[:-2]:
        if ax.name in kwargs:
            # just make sure the dimensions match, but use keyword argument
            if not len(kwargs[ax.name]) == len(ax):
                raise AxisError("Length of Variable axis and raster file dimension have to be equal.")
        else:
            # use Axis coordinates and add to kwargs for readRasterArray call
            kwargs[ax.name] = tuple(ax.coord)
    # load raster data
    if lfeedback: print(("'{}'".format(file_pattern)))
    data, geotransform = readRasterArray(file_pattern, lgzip=lgzip, lgdal=lgdal, dtype=dtype, lmask=lmask, 
                                         fillValue=fillValue, lgeotransform=True, axes=axes_list, lna=False, 
                                         lskipMissing=lskipMissing, path_params=path_params, lfeedback=lfeedback, **kwargs)
    # shift and rescale
    if offset != 0: data += offset
    if scalefactor != 1: data *= scalefactor
    ## create Variable object and add GDAL
    # check map axes and generate if necessary
    xlon, ylat = getAxes(geotransform, xlen=data.shape[-1], ylen=data.shape[-2], 
                         projected=griddef.isProjected if griddef else bool(projection))
    axes = list(axes)
    if axes[-1] is None: axes[-1] = xlon
    elif len(axes[-1]) != len(xlon): raise AxisError(axes[-1])
    if axes[-2] is None: axes[-2] = ylat
    elif len(axes[-2]) != len(ylat): raise AxisError(axes[-2])
    # create regular Variable with data in memory
    var = Variable(name=name, units=units, axes=axes, data=data, dtype=dtype, mask=None, fillValue=fillValue, 
                   atts=atts, plot=plot)
    # apply transform (if any), now that we have axes etc.
    if transform is not None: var = transform(var=var, time_axis=time_axis)
    # add GDAL functionality
    if griddef is not None:
        # perform some consistency checks ...
        if projection is None: 
            projection = griddef.projection
        elif projection != griddef.projection:
            raise ArgumentError("Conflicting projection and GridDef!\n {} != {}".format(projection,griddef.projection))
        if not np.isclose(geotransform, griddef.geotransform).all():
            raise ArgumentError("Conflicting geotransform (from raster) and GridDef!\n {} != {}".format(geotransform,griddef.geotransform))
        # ... and use provided geotransform (due to issues with numerical precision, this is usually better)
        geotransform = griddef.geotransform # if we don't pass the geotransform explicitly, it will be recomputed from the axes
    # add GDAL functionality
    var = addGDALtoVar(var, griddef=griddef, projection=projection, geotransform=geotransform, gridfolder=None)
    
    # return final, GDAL-enabled variable
    return var


## functions to load ASCII raster data

def readRasterArray(file_pattern, lgzip=None, lgdal=True, dtype=np.float32, lmask=True, fillValue=None, lfeedback=False,
                    lgeotransform=True, axes=None, lna=False, lskipMissing=False, path_params=None, **kwargs):
    ''' function to load a multi-dimensional numpy array from several structured ASCII raster files '''
    
    if axes is None: raise NotImplementedError
    #TODO: implement automatic detection of axes arguments and axes order
    
    ## expand path argument and figure out dimensions
    
    # collect axes arguments
    shape = []; axes_kwargs = dict()
    for ax in axes:
        if ax not in kwargs: raise AxisError(ax)
        coord = kwargs.pop(ax)
        shape.append(len(coord))
        axes_kwargs[ax] = coord
    assert len(axes) == len(shape) == len(axes_kwargs)
    shape = tuple(shape)
    #TODO: add handling of embedded inner product expansion
    
    # argument expansion using outer product
    file_kwargs_list = expandArgumentList(outer_list=axes, **axes_kwargs)
    assert np.prod(shape) == len(file_kwargs_list)
    
    ## load data from raster files and assemble array
    path_params = dict() if path_params is None else path_params.copy() # will be modified
    
    # find first valid 2D raster to determine shape
    i0 = 0 
    path_params.update(file_kwargs_list[i0]) # update axes parameters
    filepath = file_pattern.format(**path_params) # construct file name
    if not os.path.exists(filepath): 
        if lskipMissing: # find first valid
            while not os.path.exists(filepath):
                i0 += 1 # go to next raster file
                if i0 >= len(file_kwargs_list): 
                  raise IOError("No valid input raster files found!\n'{}'".format(filepath))
                if lfeedback: print(' ', end=' ')
                path_params.update(file_kwargs_list[i0]) # update axes parameters
                filepath = file_pattern.format(**path_params) # nest in line
        else: # or raise error
            raise IOError(filepath)
      
    # read first 2D raster file
    data2D = readASCIIraster(filepath, lgzip=lgzip, lgdal=lgdal, dtype=dtype, lna=True,
                             lmask=lmask, fillValue=fillValue, lgeotransform=lgeotransform, **kwargs)
    if lgeotransform: data2D, geotransform0, na = data2D
    else: data2D, na = data2D # we might still need na, but no need to check if it is the same
    shape2D = data2D.shape # get 2D raster shape for later use
    
    # allocate data array
    list_shape = (np.prod(shape),)+shape2D # assume 3D shape to concatenate 2D rasters
    if lmask:
        data = ma.empty(list_shape, dtype=dtype)
        if fillValue is None: data._fill_value = data2D._fill_value 
        else: data._fill_value = fillValue
        data.mask = True # initialize everything as masked 
    else: data = np.empty(list_shape, dtype=dtype) # allocate the array
    assert data.shape[0] == len(file_kwargs_list), (data.shape, len(file_kwargs_list))
    # insert (up to) first raster before continuing
    if lskipMissing and i0 > 0:
      data[:i0,:,:] = ma.masked if lmask else fillValue # mask all invalid rasters up to first valid raster
    data[i0,:,:] = data2D # add first (valid) raster
    
    # loop over remaining 2D raster files
    for i,file_kwargs in enumerate(file_kwargs_list[i0:]):
        
        path_params.update(file_kwargs) # update axes parameters
        filepath = file_pattern.format(**path_params) # construct file name
        if os.path.exists(filepath):
            if lfeedback: print('.', end=' ') # indicate data with bar/pipe
            # read 2D raster file
            data2D = readASCIIraster(filepath, lgzip=lgzip, lgdal=lgdal, dtype=dtype, lna=False,
                                     lmask=lmask, fillValue=fillValue, lgeotransform=lgeotransform, **kwargs)
            # check geotransform
            if lgeotransform: 
                data2D, geotransform = data2D
                if not geotransform == geotransform0:
                    raise AxisError(geotransform) # to make sure all geotransforms are identical!
            else: geotransform = None
            # size information
            if not shape2D == data2D.shape:
                raise AxisError(data2D.shape) # to make sure all geotransforms are identical!            
            # insert 2D raster into 3D array
            data[i+i0,:,:] = data2D # raster shape has to match
        elif lskipMissing:
            # fill with masked values
            data[i+i0,:,:] = ma.masked # mask missing raster
            if lfeedback: print(' ', end=' ') # indicate missing with dot
        else:
          raise IOError(filepath)

    # complete feedback with linebreak
    if lfeedback: print('')
    
    # reshape and check dimensions
    assert i+i0 == data.shape[0]-1, (i,i0)
    data = data.reshape(shape+shape2D) # now we have the full shape
    gc.collect() # remove duplicate data
    
    # return data and optional meta data
    if lgeotransform or lna:
        return_data = (data,)
        if lgeotransform: return_data += (geotransform,)
        if lna: return_data += (na,)
    else: 
        return_data = data
    return return_data


def readASCIIraster(filepath, lgzip=None, lgdal=True, dtype=np.float32, lmask=True, fillValue=None, 
                    lgeotransform=True, lna=False, **kwargs):
    ''' load a 2D field from an ASCII raster file (can be compressed); return (masked) numpy array and geotransform '''
    
    # handle compression (currently only gzip)
    if lgzip is None: lgzip = filepath[-3:] == '.gz' # try to auto-detect
      
    if lgdal:
  
        # gdal imports (allow to skip if GDAL is not installed)
        from osgeo import gdal        
        os.environ.setdefault('GDAL_DATA','/usr/local/share/gdal') # set default environment variable to prevent problems in IPython Notebooks
        gdal.UseExceptions() # use exceptions (off by default)
          
        if lgzip and not ( ramdisk and os.path.exists(ramdisk) ):
            raise IOError("RAM disk '{}' not found; RAM disk is required to unzip raster files for GDAL.".format(ramdisk) + 
                          "\nSet the RAM disk location using the RAMDISK environment variable.")
              
        ## use GDAL to read raster and parse meta data
        ds = tmp = None # for graceful exit
        try: 
          
            # if file is compressed, create temporary decompresse file
            if lgzip:
              with gzip.open(filepath, mode='rb') as gz, tempfile.NamedTemporaryFile(mode='wb', dir=ramdisk, delete=False) as tmp:
                shutil.copyfileobj(gz, tmp)
              filepath = tmp.name # full path of the temporary file (must not be deleted upon close!)
              
            # open file as GDAL dataset and read raster band into Numpy array
            ds = gdal.Open(filepath)
              
            assert ds.RasterCount == 1, ds.RasterCount
            band = ds.GetRasterBand(1)
            
            # get some meta data
            ie, je = band.XSize, band.YSize
            na = band.GetNoDataValue()
            if lgeotransform: 
                geotransform = ds.GetGeoTransform()
                lflip = geotransform[5] < 0 
            else: lflip = True
            
            # get data array and transform into a masked array
            data = band.ReadAsArray(0, 0, ie, je).astype(dtype)
            if lflip:
                data = flip(data, axis=-2) # flip y-axis
                if lgeotransform:
                    assert geotransform[4] == 0, geotransform
                    geotransform = geotransform[:3]+(geotransform[3]+je*geotransform[5],0,-1*geotransform[5])
            if lmask: 
              data = ma.masked_equal(data, value=na, copy=False)
              if fillValue is not None: data._fill_value = fillValue
            elif fillValue is not None: 
              data[data == na] = fillValue # replace original fill value 
          
        except Exception as e:
          
            raise e
          
        finally:
          
            # clean-up 
            del ds # neds to be deleted, before tmp-file can be deleted - Windows is very pedantic about this...
            if lgzip and tmp is not None: 
                os.remove(filepath) # remove temporary file
            del tmp # close GDAL dataset and temporary file
  
    else:
        
        ## parse header manually and use Numpy's genfromtxt to read array
        
        # handle compression on the fly (no temporary file)
        if lgzip: Raster = gzip.open(filepath, mode='rb')
        else: Raster = open(filepath, mode='rb')
    
        # open file
        with Raster:
        
            # read header information
            headers = ('NCOLS','NROWS','XLLCORNER','YLLCORNER','CELLSIZE','NODATA_VALUE')
            hdtypes = (int,int,float,float,float,dtype)
            hvalues = []
            # loop over items
            for header,hdtype in zip(headers,hdtypes):
                name, val = Raster.readline().split()
                if name.upper() != header: 
                    raise IOError("Unknown header info: '{:s}' != '{:s}'".format(name,header))
                hvalues.append(hdtype(val))
            ie, je, xll, yll, d, na = hvalues
            # derive geotransform
            if lgeotransform: geotransform = (xll, d, 0., yll, 0., d)
            
            # read data
            #print ie, je, xll, yll, d, na
            # N.B.: the file cursor is already moved to the end of the header, hence skip_header=0
            data = np.genfromtxt(Raster, skip_header=0, dtype=dtype, usemask=lmask, 
                                 missing_values=na, filling_values=fillValue, **kwargs)
          
        if not data.shape == (je,ie):
            raise IOError(data.shape, ie, je, xll, yll, d, na,)
      
    # return data and optional meta data
    if lgeotransform or lna:
        return_data = (data,)
        if lgeotransform: return_data += (geotransform,)
        if lna: return_data += (na,)
    else: 
        return_data = data
    return return_data
