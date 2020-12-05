'''
Created on Nov. 26, 2020

A helper module for newer dataset module (e.g. using xarray and rasterio)

@author: Andre R. Erler, GPL v3
'''

# external imports
import os
import dask
import numpy as np
import xarray as xr
# internal imports
from geodata.misc import DatasetError, ArgumentError
from datasets.common import getRootFolder
from geospatial.xarray_tools import loadXArray, default_lat_coords, default_lon_coords

def getFolderFileName(varname=None, dataset=None, filetype=None, resolution=None, bias_correction=None, grid=None, resampling=None, 
                      mode=None, period=None, shape=None, station=None, lcreateFolder=True, dataset_index=None, **kwargs):
    ''' function to provide the folder and filename for the requested dataset parameters '''
    if mode is None:
        mode = 'clim' if period else 'daily'
    else:
        if mode.lower() not in ('clim','monthly','daily'):
            raise ArgumentError(mode)
    # some default settings
    if dataset is None and varname is not None: 
        if dataset_index is None:
            raise ArgumentError("A 'dataset_index' dict is required to infer a preferred dataset for a variable.")
        dataset = dataset_index.get(varname,'MergedForcing')
    # dataset-specific settings
    if dataset.lower() == 'mergedforcing' or dataset.lower() == 'merged': 
        filetype = 'merged'; dataset = 'MergedForcing'
    elif dataset.lower() == 'nrcan':
        if not resolution: resolution = 'CA12' # default
        filetype = dataset.lower() # no filetypes
    elif dataset[:4].lower() == 'era5' and len(dataset) > 4:
        if filetype is None: filetype = dataset.lower() 
        dataset = 'era5'
    else:
        if filetype is None: filetype = dataset.lower()
    # folder and filename strings
    ds_str_folder = ds_str_file = filetype.lower()
    # resolution
    if resolution: ds_str_file += '_' + resolution.lower()
    # construct filename
    gridstr = '_' + grid.lower() if grid else ''
    # add shape or station identifier in front of grid
    if shape and station: 
        raise DatasetError((shape,station))
    elif shape: gridstr = '_' + shape.lower() + gridstr
    elif station: gridstr = '_' + station.lower() + gridstr
    bcstr = '_' + bias_correction.lower() if bias_correction else ''
    if mode.lower() == 'daily': name_str = bcstr + '_' + varname.lower() + gridstr
    else: name_str = bcstr + gridstr
    mode_str = mode.lower()
    if period is None: pass
    elif isinstance(period,str): mode_str += '_'+period
    elif isinstance(period,(tuple,list)): mode_str += '_{}-{}'.format(*period)
    else: raise NotImplementedError(period)
    filename = '{}{}_{}.nc'.format(ds_str_file, name_str, mode_str)
    # construct folder
    folder = getRootFolder(dataset_name=dataset, fallback_name='MergedForcing')
    if mode.lower() == 'daily':
        folder += ds_str_folder+'_daily'
        if grid: 
            folder = '{}/{}'.format(folder,grid) # non-native grids are stored in sub-folders
            if resampling is None:
                # complicated auto-detection of resampling folders ... 
                old_folder = os.getcwd()
                os.chdir(folder)
                # inspect folder
                nc_file = False; default_folder = False; folder_list = []
                for item in os.listdir():
                    if os.path.isfile(item):
                        if item.endswith('.nc'): nc_file = True
                    elif os.path.isdir(item):
                        if item.lower() == 'default': default_folder = item
                        folder_list.append(item)
                    else:
                        raise IOError(item)
                os.chdir(old_folder) # return
                # evaluate findings
                if nc_file: resampling = None
                elif default_folder: resampling = default_folder
                elif len(folder_list) == 1: resampling = folder_list[0]
            if resampling: 
                folder = '{}/{}'.format(folder,resampling) # different resampling options are stored in subfolders                
    else: folder += ds_str_folder+'avg'
    if folder[-1] != '/': folder += '/'
    if lcreateFolder: os.makedirs(folder, exist_ok=True)
    # return folder and filename
    return folder,filename


def addConstantFields(xds, const_list=None, grid=None):
    ''' add constant auxiliary fields like topographic elevation and geographic coordinates to dataset '''
    if const_list is None:
        const_list = ['lat2D', 'lon2D']
    # find horizontal coordinates
    dims = (xds.ylat,xds.xlon)
    for rv in xds.data_vars.values():
        if xds.ylat in rv.dims and xds.xlon in rv.dims: break
    if dask.is_dask_collection(rv):
        chunks = {dim:max(chk) for dim,chk in zip(rv.dims, rv.chunks) if dim in dims}
    else:
        chunks = None # don't chunk if nothing else is chunked...
    # add constant variables
    llat2D = 'lat2D' in const_list; llon2D = 'lon2D' in const_list  
    if llat2D or llon2D:
        # add geographic coordinate fields 
        if grid is None: 
            xlon = xlat = None
            # infer from lat/lon coordinates
            for dim in xds.dims:
                if dim in default_lon_coords: xlon = xds[dim].values
                if dim in default_lat_coords: xlat = xds[dim].values
            if xlon is None or xlat is None:
                raise DatasetError("Need latitude and longitude coordinates or GridDef object to infer lat2D or lon2D.")
            lon2D,lat2D = np.meshgrid(xlon,xlat)
            assert lon2D.shape == lat2D.shape
            assert lon2D.shape == (len(xlat),len(xlon)), lon2D.shape
        else:
            # get fields from griddef
            from geodata.gdal import loadPickledGridDef
            griddef = loadPickledGridDef(grid=grid)
            lat2D = griddef.lat2D; lon2D = griddef.lon2D
        # add local latitudes
        if llat2D:
            atts = dict(name='lat2d', long_name='Latitude', units='deg N')
            xvar = xr.DataArray(data=lat2D, attrs=atts, dims=dims)
            if chunks: xvar = xvar.chunk(chunks=chunks)
            xds['lat2D'] = xvar
        # add local longitudes
        if llon2D:
            atts = dict(name='lon2d', long_name='Longitude', units='deg E')
            xvar = xr.DataArray(data=lon2D, attrs=atts, dims=dims)
            if chunks: xvar = xvar.chunk(chunks=chunks)
            xds['lon2D'] = xvar 
    if 'zs' in const_list:
        print("Loading of surface/topographic elevation is not yet implemented")
    return xds        


def loadXRDataset(varname=None, varlist=None, dataset=None, grid=None, bias_correction=None, resolution=None, 
                  period=None, shape=None, station=None, mode='daily', filetype=None, resampling=None, 
                  varmap=None, varatts=None, default_varlist=None, mask_and_scale=True,  
                  lgeoref=True, geoargs=None, chunks=True, lautoChunk=False, lskip=False, **kwargs):
    ''' load data from standardized NetCDF files into an xarray Dataset '''
    # first, get folder and filename pattern
    folder,filename = getFolderFileName(varname='{var:s}', dataset=dataset, filetype=filetype, resolution=resolution, grid=grid,
                                        bias_correction=bias_correction, resampling=resampling, period=period, mode=mode,  
                                        shape=shape, station=station, lcreateFolder=False, dataset_index=None)
    # load XR dataset
    xds = loadXArray(varname=varname, varlist=varlist, folder=folder, varatts=varatts, filename_pattern=filename,  
                     default_varlist=default_varlist, varmap=varmap, mask_and_scale=mask_and_scale, grid=grid,  
                     lgeoref=lgeoref, geoargs=geoargs, chunks=chunks, lautoChunk=lautoChunk, lskip=lskip, **kwargs)
    # supplement annotation/attributes
    if bias_correction is not None and 'bias_correction' not in xds.attrs: xds.attrs['bias_correction'] = bias_correction
    if resolution is not None and 'resolution' not in xds.attrs: xds.attrs['resolution'] = resolution
    if resampling is not None and 'resampling' not in xds.attrs: xds.attrs['resampling'] = resampling 
    if 'name' not in xds.attrs: xds.attrs['name'] = dataset
    if 'title' not in xds.attrs:
        title_str = ' Climatology' if mode.lower() == 'clim' else ' {:s} Timeseries'.format(mode.title())
        xds.attrs['title'] = xds.attrs['name'] + title_str
    # return dataset
    return xds
    

## abuse for testing
if __name__ == '__main__':
    
    # select mode
#     mode = 'filefolder'
    mode = 'loadXR'
    
    # run tests
    
    if mode == 'loadXR':
        
        xds = loadXRDataset(varname=None, varlist=['precip','T2'], dataset='NRCan', grid='snw2', bias_correction=None, resolution='SON60', 
                            period=None, shape=None, station=None, mode='daily', filetype=None, resampling=None, 
                            varmap=None, varatts=None, default_varlist=None, mask_and_scale=True,  
                            lgeoref=True, geoargs=None, chunks=None, lautoChunk=False, lskip=False,)
        print(xds)
    
    elif mode == 'filefolder':
        
        # test with generic variable place holder
        folder,filename = getFolderFileName(varname='{VAR:s}', dataset='NRCan', filetype=None, resolution=None, bias_correction=None, 
                                            grid='snw2', resampling=None, mode='daily', period=None, shape=None, station=None, 
                                            lcreateFolder=False, dataset_index=None)
        print(folder)
        print(filename)
    