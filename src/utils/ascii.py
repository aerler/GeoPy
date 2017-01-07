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
from geodata.misc import AxisError, ArgumentError
from utils.misc import flip, expandArgumentList

# the environment variable RAMDISK contains the path to the RAM disk
ramdisk = os.getenv('RAMDISK', None)
if ramdisk and not os.path.exists(ramdisk): 
  raise IOError(ramdisk)


## functions to construct Variables and Datasets from ASCII raster data


def rasterDataset(name=None, title=None, vardefs=None, axdefs=None, atts=None, projection=None, griddef=None,
                  lgzip=None, lgdal=True, lmask=True, fillValue=None, lskipMissing=True, lgeolocator=True,
                  file_pattern=None, **kwargs):
    ''' function to load a set of variables that are stored in raster format in a systematic directory tree into a Dataset
        Variables and Axis are defined as follows:
          vardefs[varname] = dict(name=string, units=string, axes=tuple of strings, atts=dict, plot=dict, dtype=np.dtype, fillValue=value)
          axdefs[axname]   = dict(name=string, units=string, atts=dict, coords=array or list) or None
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
    for axname,axdef in axdefs.items():
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
    for varname,vardef in vardefs.items():
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
        for key,value in kwargs.items():
          if key not in axes or key in relaxes:
              vardef[key] = value
        # create Variable object
        var = rasterVariable(projection=projection, griddef=griddef, file_pattern=file_pattern, lgzip=lgzip, lgdal=lgdal, 
                             lmask=lmask, lskipMissing=lskipMissing, axes=axes_list, path_params=path_params, **vardef) 
        # vardef components: name, units, atts, plot, dtype, fillValue
        varlist.append(var)
        # check that map axes are correct
        for ax in var.xlon,var.ylat:
            if axes[ax.name] is None: axes[ax.name] = ax
            elif axes[ax.name] != ax: raise AxisError("{} axes are incompatible.".format(ax.name))
        if griddef is None: griddef = var.griddef
        elif griddef != var.griddef: raise AxisError("GridDefs are inconsistent.")
        if geotransform is None: geotransform = var.geotransform
        elif geotransform != var.geotransform: raise AxisError("Geotransforms are inconsistent.")
        
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
                   path_params=None, **kwargs):
    ''' function to read multi-dimensional raster data and construct a GDAL-enabled Variable object '''
    
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
    data, geotransform = readRasterArray(file_pattern, lgzip=lgzip, lgdal=lgdal, dtype=dtype, lmask=lmask, 
                                         fillValue=fillValue, lgeotransform=True, axes=axes_list, lna=False, 
                                         lskipMissing=lskipMissing, path_params=path_params, **kwargs)
    
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
    # add GDAL functionality
    if griddef is not None:
        # perform some consistency checks
        if projection is None: 
            projection = griddef.projection
        elif projection != griddef.projection:
            raise ArgumentError("Conflicting projection and GridDef!")
        if geotransform != griddef.geotransform:
            print geotransform,griddef.geotransform
            raise ArgumentError("Conflicting geotransform (from raster) and GridDef!")
    # add GDAL functionality
    var = addGDALtoVar(var, griddef=griddef, projection=projection, geotransform=geotransform, gridfolder=None)
    
    # return final, GDAL-enabled variable
    return var


## functions to load ASCII raster data

def readRasterArray(file_pattern, lgzip=None, lgdal=True, dtype=np.float32, lmask=True, fillValue=None, 
                    lgeotransform=True, axes=None, lna=False, lskipMissing=False, path_params=None, **kwargs):
    ''' function to load a multi-dimensional numpy array from several structured ASCII raster files '''
    
    if lskipMissing and not lmask: raise ArgumentError
    
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
                  raise IOError("No valid input raster files found!")
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
      data[:i0,:,:] = ma.masked # mask all invalid rasters up to first valid raster
    data[i0,:,:] = data2D # add first (valid) raster
    
    # loop over remaining 2D raster files
    for i,file_kwargs in enumerate(file_kwargs_list[i0:]):
        
        path_params.update(file_kwargs) # update axes parameters
        filepath = file_pattern.format(**path_params) # construct file name
        if os.path.exists(filepath):
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
        else:
          raise IOError(filepath)
    
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
          
        ## use GDAL to read raster and parse meta data
        try: 
          
            # if file is compressed, create temporary decompresse file
            if lgzip:
              with gzip.open(filepath, mode='rb') as gz, tempfile.NamedTemporaryFile(mode='wb', dir=ramdisk, delete=False) as tmp:
                shutil.copyfileobj(gz, tmp)
              filepath = tmp.name # full path of the temporary file (must not be deleted upon close!)
            else: tmp = None
              
            # open file as GDAL dataset and read raster band into Numpy array
            ds = gdal.Open(filepath)
              
            assert ds.RasterCount == 1, ds.RasterCount
            band = ds.GetRasterBand(1)
            
            # get some meta data
            ie, je = band.XSize, band.YSize
            na = band.GetNoDataValue()
            if lgeotransform: geotransform = ds.GetGeoTransform()
            
            # get data array and transform into a masked array
            data = band.ReadAsArray(0, 0, ie, je).astype(dtype)
            data = flip(data, axis=-2) # flip y-axis
            data = ma.masked_equal(data, value=na, copy=False)
            if fillValue is not None: data._fill_value = fillValue
          
        finally:
          
            # clean-up
            del ds, tmp # close GDAL dataset and temporary file
            #print tmp.name # print full path of temporary file
            if lgzip: os.remove(filepath) # remove temporary file
  
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
