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
from geodata.gdal import addGDALtoDataset, addGDALtoVar
from geodata.misc import AxisError, ArgumentError
from utils.misc import flip, expandArgumentList

# the environment variable RAMDISK contains the path to the RAM disk
ramdisk = os.getenv('RAMDISK', None)
if ramdisk and not os.path.exists(ramdisk): 
  raise IOError(ramdisk)


## functions to construct Variables and Datasets from ASCII raster data

def rasterVariable(name=None, units=None, axes=None, atts=None, plot=None, dtype=None, projection=None, 
                   file_pattern=None, lgzip=None, lgdal=True, lmask=True, fillValue=None,):
    ''' function to read multi-dimensional raster data and construct a GDAL-enabled Variable object '''
    pass
    


## functions to load ASCII raster data

def readRasterArray(file_pattern, lgzip=None, lgdal=True, dtype=np.float32, lmask=True, fillValue=None, 
                    lgeotransform=True, axes=None, lna=False, lskipMissing=False, **kwargs):
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
    
    # find first valid 2D raster to determine shape
    i0 = 0; filepath = file_pattern.format(**file_kwargs_list[i0]) # construct file name
    if not os.path.exists(filepath): 
        if lskipMissing: # find first valid
            while not os.path.exists(filepath):
                i0 += 1; filepath = file_pattern.format(**file_kwargs_list[i0]) # try next in line
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
        
        filepath = file_pattern.format(**file_kwargs) # construct file name
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
