'''
Created on 2013-08-13, adapted on 2013-09-13

This module provides a class that contains definitions of source and target datasets and methods to process 
variables in these datasets. 
The class is designed to be imported and extended by modules that perform more specific tasks.
Simple methods for copying and averaging variables will already be provided in this class.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import functools
from osgeo import gdal, osr
# internal imports
from geodata.misc import VariableError, AxisError, PermissionError, DatasetError, GDALError, ArgumentError #, DateError
from geodata.base import Axis, Dataset, Variable
from geodata.netcdf import DatasetNetCDF, asDatasetNC
from geodata.nctools import writeNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition, gdalInterp,\
  NamedShape
from collections import OrderedDict
# default data types
dtype_int = np.dtype('int16')
dtype_float = np.dtype('float32')

class ProcessError(Exception):
  ''' Error class for exceptions occurring in methods of the CPU (CentralProcessingUnit). '''
  pass

class CentralProcessingUnit(object):
  
  def __init__(self, source, target=None, varlist=None, ignorelist=None, tmp=True, feedback=True):
    ''' Initialize processor and pass input and output datasets. '''
    # check varlist
    if varlist is None: varlist = source.variables.keys() # all source variables
    elif not isinstance(varlist,(list,tuple)): raise TypeError
    self.varlist = varlist # list of variable to be processed
    # ignore list (e.g. variables that will cause errors)
    if ignorelist is None: ignorelist = [] # an empty list
    elif not isinstance(ignorelist,(list,tuple)): raise TypeError
    self.ignorelist = ignorelist # list of variable *not* to be processed    
    # check input
    if not isinstance(source,Dataset): raise TypeError
    if isinstance(source,DatasetNetCDF) and not 'r' in source.mode: raise PermissionError
    self.input = source
    self.source = source
    # check output
    if target is not None:       
      if not isinstance(target,Dataset): raise TypeError
      if isinstance(target,DatasetNetCDF) and not 'w' in target.mode: raise PermissionError
    else:
      if not tmp: raise DatasetError, "Need target location, if temporary storage is disables (tmp=False)." 
    self.output = target
    # temporary dataset
    self.tmp = tmp
    if tmp: self.tmpput = Dataset(name='tmp', title='Temporary Dataset', varlist=[], atts={})
    else: self.tmpput = None
    # determine if temporary storage is used and assign target dataset
    if self.tmp: self.target = self.tmpput
    else: self.target = self.output 
    # whether or not to print status output
    self.feedback = feedback
        
  def getTmp(self, asNC=False, filename=None, deepcopy=False, **kwargs):
    ''' Get a copy of the temporary data in dataset format. '''
    if not self.tmp: raise DatasetError
    # make new dataset (name and title should transfer in atts dict)
    if asNC:
      if not isinstance(filename,basestring): raise TypeError
      writeData = kwargs.pop('writeData',False)
      ncformat = kwargs.pop('ncformat','NETCDF4')
      zlib = kwargs.pop('zlib',True)
      dataset = asDatasetNC(self.tmpput, ncfile=filename, mode='wr', deepcopy=deepcopy, 
                            writeData=writeData, ncformat=ncformat, zlib=zlib, **kwargs)
    else:
      dataset = self.tmpput.copy(varsdeep=deepcopy, atts=self.input.atts.copy(), **kwargs)
    # return dataset
    return dataset
  
  def sync(self, varlist=None, flush=False, gdal=True, copydata=True):
    ''' Transfer contents of temporary storage to output/target dataset. '''
    if not isinstance(self.output,Dataset): raise DatasetError, "Cannot sync without target Dataset!"
    if self.tmp:
      if varlist is None: varlist = self.tmpput.variables.keys()  
      for varname in varlist:
        if varname in self.tmpput.variables:
          var = self.tmpput.variables[varname]
          self.output.addVariable(var, loverwrite=True, deepcopy=copydata)
          # N.B.: without copydata/deepcopy, only the variable header is created but no data is written
          if flush: var.unload() # remove unnecessary references (unlink data)
      if gdal and 'gdal' in self.tmpput.__dict__: 
        if self.tmpput.gdal: 
          projection = self.tmpput.projection; geotransform = self.tmpput.geotransform
          #xlon = self.tmpput.xlon; ylat = self.tmpput.ylat 
        else: 
          projection=None; geotransform=None; #xlon = None; ylat = None 
        self.output = addGDALtoDataset(self.output, projection=projection, geotransform=geotransform)
#           self.source = self.output # future operations will write to the output dataset directly
#           self.target = self.output # future operations will write to the output dataset directly                     
        
  def writeNetCDF(self, filename=None, folder=None, ncformat='NETCDF4', zlib=True, writeData=True, close=False, flush=False):
    ''' Write current temporary storage to a NetCDF file. '''
    if self.tmp:
      if not isinstance(filename,basestring): raise TypeError
      if folder is not None: filename = folder + filename       
      output = writeNetCDF(self.tmpput, filename, ncformat=ncformat, zlib=zlib, writeData=writeData, close=False)
      if flush: self.tmpput.unload()
      if self.feedback: print('\nOutput written to {0:s}\n'.format(filename))
    else: 
      self.output.sync()
      output = self.output.dataset # get (primary) NetCDF file
      if self.feedback: print('\nSynchronized dataset {0:s} with temporary storage.\n'.format(output.name))
    # flush?
    if flush: self.output.unload()      
    # close file or return file handle
    if close: output.close()
    else: return output

  def process(self, function, flush=False):
    ''' This method applies the desired operation/function to each variable in varlist. '''
    if flush: # this function is to save RAM by flushing results to disk immediately
      if not isinstance(self.output,DatasetNetCDF):
        raise ProcessError, "Flush can only be used with NetCDF Datasets (and not with temporary storage)."
      if self.tmp: # flush requires output to be target
        if self.source.gdal and not self.tmpput.gdal:
          self.tmpput = addGDALtoDataset(self.tmpput, projection=self.source.projection, geotransform=self.source.geotransform)
        self.source = self.tmpput
        self.target = self.output
        self.tmp = False # not using temporary storage anymore
    # loop over input variables
    for varname in self.varlist:
      # check agaisnt ignore list
      if varname not in self.ignorelist: 
        # check if variable already exists
        if self.target.hasVariable(varname):
          # "in-place" operations
          var = self.target.variables[varname]         
          newvar = function(var) # perform actual processing
          if newvar.ndim != var.ndim or newvar.shape != var.shape: raise VariableError
          if newvar is not var: self.target.replaceVariable(var,newvar)
        elif self.source.hasVariable(varname):        
          var = self.source.variables[varname]
          ldata = var.data # whether data was pre-loaded 
          # perform operation from source and copy results to target
          newvar = function(var) # perform actual processing
          if not ldata: var.unload() # if it was already loaded, don't unload        
          self.target.addVariable(newvar, copy=True) # copy=True allows recasting as, e.g., a NC variable
        else:
          raise DatasetError, "Variable '%s' not found in input dataset."%varname
        assert varname == newvar.name
        # flush data to disk immediately      
        if flush: 
          self.output.variables[varname].unload() # again, free memory
        newvar.unload(); del var, newvar # free space; already added to new dataset
    # after everything is said and done:
    self.source = self.target # set target to source for next time
    
    
  ## functions (or function pairs, rather) that perform operations on the data
  # every function pair needs to have a setup function and a processing function
  # the former sets up the target dataset and the latter operates on the variables
  
  # function pair to average data over a given collection of shapes      
  def ShapeAverage(self, shape_dict=None, shape_name=None, shpax=None, xlon=None, ylat=None, **kwargs):
    ''' Average over a limited area of a gridded datasets; calls processAverageShape. 
        A dictionary of NamedShape objects is expected to define the averaging areas. '''
    if not self.source.gdal: raise DatasetError, "Source dataset must be GDAL enabled! {:s} is not.".format(self.source.name)
    if not isinstance(shape_dict,OrderedDict): raise TypeError
    if not all(isinstance(shape,NamedShape) for shape in shape_dict.itervalues()): raise TypeError
    # make temporary dataset
    if self.source is self.target:
      if self.tmp: assert self.source == self.tmpput and self.target == self.tmpput
      # the operation can not be performed "in-place"!
      self.target = Dataset(name='tmptoo', title='Temporary target dataset for non-in-place operations', varlist=[], atts={})
      ltmptoo = True
    else: ltmptoo = False
    src = self.source; tgt = self.target # short-cuts 
    # determine source dataset grid definition
    if src.griddef is None:  
      srcgrd = GridDefinition(projection=self.source.projection, geotransform=self.source.geotransform, 
                              size=self.source.mapSize, xlon=self.source.xlon, ylat=self.source.ylat)
    else: srcgrd = src.griddef
    # figure out horizontal axes (will be replaced with station axis)
    if isinstance(xlon,Axis): 
      if not src.hasAxis(xlon, check=True): raise DatasetError
    elif isinstance(xlon,basestring): xlon = src.getAxis(xlon)
    else: xlon = src.x if srcgrd.isProjected else src.lon
    if isinstance(ylat,Axis):
      if not src.hasAxis(ylat, check=True): raise DatasetError
    elif isinstance(ylat,basestring): ylat = src.getAxis(ylat)
    else: ylat = src.y if srcgrd.isProjected else src.lat
    # check/create shapes axis
    if shpax: # not in source dataset!
      # if shape axis supplied
      if src.hasAxis(shpax, check=True): raise DatasetError, "Source dataset must not have a 'shape' axis!"
      if len(shpax) != len(shape_dict): raise AxisError
    else:
      # creat shape axis, if not supplied
      shpatts = dict(name='shape', long_name='Ordinal Number of Shape', units='#')
      shpax = Axis(coord=np.arange(1,len(shape_dict)+1), atts=shpatts) # starting at 1
    assert isinstance(xlon,Axis) and isinstance(ylat,Axis) and isinstance(shpax,Axis)
    # prepare target dataset
    # N.B.: attributes should already be set in target dataset (by caller module)
    #       we are also assuming the new dataset has no axes yet
    assert len(tgt.axes) == 0
    # add station axis (trim to valid coordinates)
    tgt.addAxis(shpax, asNC=True, copy=True) # already new copy
    # add axes from source data
    for axname,ax in src.axes.iteritems():
      if axname not in (xlon.name,ylat.name):
        tgt.addAxis(ax, asNC=True, copy=True)
    # add shape names
    shape_names = [shape.name for shape in shape_dict.itervalues()] # can construct Variable from list!
    atts = dict(name='shape_name', long_name='Name of Shape', units='')
    tgt.addVariable(Variable(data=shape_names, axes=(shpax,), atts=atts), asNC=True, copy=True)
    # add proper names
    shape_long_names = [shape.long_name for shape in shape_dict.itervalues()] # can construct Variable from list!
    atts = dict(name='shp_long_name', long_name='Proper Name of Shape', units='')
    tgt.addVariable(Variable(data=shape_long_names, axes=(shpax,), atts=atts), asNC=True, copy=True)    
    # add shape category
    shape_type = [shape.shapetype for shape in shape_dict.itervalues()] # can construct Variable from list!
    atts = dict(name='shp_type', long_name='Type of Shape', units='')
    tgt.addVariable(Variable(data=shape_type, axes=(shpax,), atts=atts), asNC=True, copy=True)    
    # collect rasterized masks from shape files 
    mask_array = np.zeros((len(shpax),)+srcgrd.size[::-1], dtype=np.bool) 
    # N.B.: rasterize() returns mask in (y,x) shape, size is ordered as (x,y)
    shape_masks = []; shp_full = []; shp_empty = []; shp_encl = []
    for i,shape in enumerate(shape_dict.itervalues()):
      mask = shape.rasterize(griddef=srcgrd, asVar=False)
      mask_array[i,:] = mask
      masksum = mask.sum() 
      lfull = masksum == 0; shp_full.append( lfull )
      lempty = masksum == mask.size; shp_empty.append( lempty )
      shape_masks.append( mask if not lempty else None )
      if lempty: shp_encl.append( False )
      else:
        shp_encl.append( np.all( mask[[0,-1],:] == True ) and np.all( mask[:,[0,-1]] == True ) )
        # i.e. if boundaries are masked
    # N.B.: shapes that have no overlap with grid will be skipped and filled with NaN
    # add rasterized masks to new dataset
    atts = dict(name='shp_mask', long_name='Rasterized Shape Mask', units='')
    tgt.addVariable(Variable(data=mask_array, atts=atts, axes=(shpax,srcgrd.ylat.copy(),srcgrd.xlon.copy())), 
                    asNC=True, copy=True)
    # add area enclosed by shape
    da = srcgrd.geotransform[1]*srcgrd.geotransform[5]
    mask_area = (1-mask_array).mean(axis=2).mean(axis=1)*da
    atts = dict(name='shp_area', long_name='Area Contained in the Shape', 
                units= 'm^2' if srcgrd.isProjected else 'deg^2' )
    tgt.addVariable(Variable(data=mask_area, axes=(shpax,), atts=atts), asNC=True, copy=True)
    # add flag to indicate if shape is fully enclosed by domain
    atts = dict(name='shp_encl', long_name='If Shape is fully included in Domain', units= '')
    tgt.addVariable(Variable(data=shp_encl, axes=(shpax,), atts=atts), asNC=True, copy=True)
    # add flag to indicate if shape fully covers domain
    atts = dict(name='shp_full', long_name='If Shape fully covers Domain', units= '')
    tgt.addVariable(Variable(data=shp_full, axes=(shpax,), atts=atts), asNC=True, copy=True)
    # add flag to indicate if shape and domain have no overlap
    atts = dict(name='shp_empty', long_name='If Shape and Domain have no Overlap', units= '')
    tgt.addVariable(Variable(data=shp_empty, axes=(shpax,), atts=atts), asNC=True, copy=True)
    # save all the meta data
    tgt.sync()
    # prepare function call    
    function = functools.partial(self.processShapeAverage, masks=shape_masks, ylat=ylat, xlon=xlon, shpax=shpax) # already set parameters
    # start process
    if self.feedback: print('\n   +++   processing shape/area averaging   +++   ') 
    self.process(function, **kwargs) # currently 'flush' is the only kwarg
    if self.feedback: print('\n')
    if self.tmp: self.tmpput = self.target
    if ltmptoo: assert self.tmpput.name == 'tmptoo' # set above, when temp. dataset is created    
  # the previous method sets up the process, the next method performs the computation
  def processShapeAverage(self, var, masks=None, ylat=None, xlon=None, shpax=None):
    ''' Compute masked area averages from variable data. '''
    # process gdal variables (if a variable has a horiontal grid, it should be GDAL enabled)
    if var.gdal and ( np.issubdtype(var.dtype,np.integer) or np.issubdtype(var.dtype,np.inexact) ):
      if self.feedback: print('\n'+var.name),
      assert var.hasAxis(xlon) and var.hasAxis(ylat)
      assert len(masks) == len(shpax)
      tgt = self.target
      assert tgt.hasAxis(shpax, strict=False) and shpax not in var.axes 
      # assemble new axes
      axes = [tgt.getAxis(shpax.name)]      
      for ax in var.axes:
        if ax not in (xlon,ylat) and ax.name != shpax.name: # these axes are just transferred 
          axes.append(tgt.getAxis(ax.name))
      # N.B.: shape axis well be outer axis
      axes = tuple(axes)
      # pre-allocate
      shape = tuple(len(ax) for ax in axes)
      tgtdata = np.zeros(shape, dtype=np.float32) 
      # now we loop over all shapes/masks
      var.load()
      if var.ndim == 2:
        for i,mask in enumerate(masks): 
          if mask is None: tgtdata[i] = np.NaN # NaN for missing values (i.e. no overlap)
          else: tgtdata[i] = var.mapMean(mask=mask, asVar=False, squeeze=True) # compute the averages
          #print i,tgtdata[i]
      elif var.ndim > 2:
        for i,mask in enumerate(masks):
          if mask is None: tgtdata[i,:] = np.NaN # NaN for missing values (i.e. no overlap) 
          else: tgtdata[i,:] = var.mapMean(mask=mask, asVar=False, squeeze=True) # compute the averages
          #print i,tgtdata[i]
      else: raise AxisError 
      # create new Variable
      assert shape == tgtdata.shape
      newvar = var.copy(axes=axes, data=tgtdata) # new axes and data
      del tgtdata # clean up (just to make sure)      
    else:
      var.load() # need to load variables into memory to copy it (and we are not doing anything else...)
      newvar = var # just pass over the variable to the new dataset
    # return variable
    return newvar
  
  # function pair to extract station data from a time-series (or climatology)      
  def Extract(self, template=None, stnax=None, xlon=None, ylat=None, laltcorr=True, **kwargs):
    ''' Extract station data points from gridded datasets; calls processExtract. 
        A station dataset can be passed as template (must have station coordinates. '''
    if not self.source.gdal: raise DatasetError, "Source dataset must be GDAL enabled! {:s} is not.".format(self.source.name)
    if template is None: raise NotImplementedError
    elif isinstance(template, Dataset):
      if not template.hasAxis('station'): raise DatasetError, "Template station dataset needs to have a station axis."
      if not ( (template.hasVariable('lat') or template.hasVariable('stn_lat')) and 
               (template.hasVariable('lon') or template.hasVariable('stn_lon')) ): 
        raise DatasetError, "Template station dataset needs to have lat/lon arrays for the stations."      
    else: raise TypeError
    # make temporary dataset
    if self.source is self.target:
      if self.tmp: assert self.source == self.tmpput and self.target == self.tmpput
      # the operation can not be performed "in-place"!
      self.target = Dataset(name='tmptoo', title='Temporary target dataset for non-in-place operations', varlist=[], atts={})
      ltmptoo = True
    else: ltmptoo = False
    src = self.source; tgt = self.target # short-cuts 
    # determine source dataset grid definition
    if src.griddef is None:  
      srcgrd = GridDefinition(projection=self.source.projection, geotransform=self.source.geotransform, 
                              size=self.source.mapSize, xlon=self.source.xlon, ylat=self.source.ylat)
    else: srcgrd = src.griddef
    # figure out horizontal axes (will be replaced with station axis)
    if isinstance(xlon,Axis): 
      if not src.hasAxis(xlon, check=True): raise DatasetError
    elif isinstance(xlon,basestring): xlon = src.getAxis(xlon)
    else: xlon = src.x if srcgrd.isProjected else src.lon
    if isinstance(ylat,Axis):
      if not src.hasAxis(ylat, check=True): raise DatasetError
    elif isinstance(ylat,basestring): ylat = src.getAxis(ylat)
    else: ylat = src.y if srcgrd.isProjected else src.lat
    if stnax: # not in source dataset!
      if src.hasAxis(stnax, check=True): raise DatasetError, "Source dataset must not have a 'station' axis!"
    elif template: stnax = template.station # station axis
    else: raise ArgumentError, "A station axis needs to be supplied." 
    assert isinstance(xlon,Axis) and isinstance(ylat,Axis) and isinstance(stnax,Axis)
    # transform to dataset-native coordinate system
    if template: 
      if template.hasVariable('lat'): lats = template.lat.getArray()
      else: lats = template.stn_lat.getArray()
      if template.hasVariable('lon'): lons = template.lon.getArray()
      else: lons = template.stn_lon.getArray()
    else: raise NotImplementedError, "Cannot extract station data without a station template Dataset"
    # adjust longitudes
    if srcgrd.isProjected:
      if lons.max() > 180.: lons = np.where(lons > 180., 360.-lons, lons)
      # reproject coordinate
      latlon = osr.SpatialReference() 
      latlon.SetWellKnownGeogCS('WGS84') # a normal lat/lon coordinate system
      tx = osr.CoordinateTransformation(latlon,srcgrd.projection)
      xs = []; ys = [] 
      for i in xrange(len(lons)):
        x,y,z = tx.TransformPoint(lons[i].astype(np.float64),lats[i].astype(np.float64))
        xs.append(x); ys.append(y); del z
      lons = np.array(xs); lats = np.array(ys)
      #lons,lats = tx.TransformPoints(lons,lats) # doesn't seem to work...
    else:
      if lons.min() < 0. and xlon.coord.max() > 180.: lons = np.where(lons < 0., lons + 360., lons)
      elif lons.max() > 180. and xlon.coord.min() < 0.: lons = np.where(lons > 180., 360.-lons, lons)
      else: pass # source and template do not conflict
    # generate index list
    ixlon = []; iylat = []; istn = []; zs_err = [] # also record elevation error
    lzs = src.hasVariable('zs')
    lstnzs = template.hasVariable('zs') or  template.hasVariable('stn_zs')
    if laltcorr and lzs and lstnzs:
      if src.zs.ndim != 2 or not src.zs.gdal or src.zs.units != 'm': raise VariableError
      # consider altidue of surrounding points as well      
      zs = src.zs.getArray(unmask=True,fillValue=-300)
      if template.hasVariable('zs'): stn_zs = template.zs.getArray(unmask=True,fillValue=-300)
      else: stn_zs = template.stn_zs.getArray(unmask=True,fillValue=-300)
      if src.zs.axisIndex(xlon.name) == 0: zs.transpose() # assuming lat,lon or y,x order is more common
      ye,xe = zs.shape # assuming order lat,lon or y,x
      xe -= 1; ye -= 1 # last valid index, not length
      for n,lon,lat in zip(xrange(len(stnax)),lons,lats):
        ip = xlon.getIndex(lon, mode='left', outOfBounds=True)
        jp = ylat.getIndex(lat, mode='left', outOfBounds=True)
        if ip is not None and jp is not None:
          # find neighboring point with smallest altitude error 
#           ip = im+1 if im < xe else im  
#           jp = jm+1 if jm < ye else jm
          im = ip-1 if ip > 0 else ip  
          jm = jp-1 if jp > 0 else jp
          zdiff = np.Infinity # initialize, so that it triggers at least once
          # check four closest grid points
          for i in im,ip:
            for j in jm,jp:
              ze = zs[j,i]-stn_zs[n]
              zd = np.abs(ze) # compute elevation error
              if zd < zdiff: ii,jj,zdiff,zerr = i,j,zd,ze # preliminary selection, triggers at least once               
          ixlon.append(ii); iylat.append(jj); istn.append(n); zs_err.append(zerr) # final selection          
    else: 
      # just choose horizontally closest point 
      for n,lon,lat in zip(xrange(len(stnax)),lons,lats):
        i = xlon.getIndex(lon, mode='closest', outOfBounds=True)
        j = ylat.getIndex(lat, mode='closest', outOfBounds=True)
        if i is not None and j is not None: 
          if lzs: # compute elevation error
            zs_err.append(zs[j,i]-stn_zs[n])          
          ixlon.append(i); iylat.append(j); istn.append(n)
    # N.B.: it is necessary to append, because we don't know the number of valid points
    ixlon = np.array(ixlon); iylat = np.array(iylat); istn = np.array(istn); zs_err = np.array(zs_err)
    # prepare target dataset
    # N.B.: attributes should already be set in target dataset (by caller module)
    #       we are also assuming the new dataset has no axes yet
    assert len(tgt.axes) == 0
    # add axes from source data
    for axname,ax in src.axes.iteritems():
      if axname not in (xlon.name,ylat.name):
        tgt.addAxis(ax, asNC=True, copy=True)
    # add station axis (trim to valid coordinates)
    newstnax = stnax.copy(coord=stnax.coord[istn]) # same but with trimmed coordinate array
    tgt.addAxis(newstnax, asNC=True, copy=True) # already new copy
    # create variable for elevation error
    if lzs:
      assert len(zs_err) > 0
      zs_err = Variable(name='zs_err', units='m', data=zs_err, axes=(newstnax,),
                        atts=dict(long_name='Station Elevation Error'))
      tgt.addVariable(zs_err, asNC=True, copy=True); del zs_err # need to copy to make NC var
    # add a bunch of other variables with station meta data
    for var in template.variables.itervalues():
      if var.ndim == 1 and var.hasAxis(stnax): # station attributes
        if var.name[-4:] != '_len' or var.name == 'stn_rec_len': # exclude certain attributes
          newvar = var.copy(data=var.getArray()[istn], axes=(newstnax,))
          if newvar.name[:4] != 'stn_' and newvar.name[:8] != 'station_': 
            newvar.name = 'stn_'+newvar.name
          # N.B.: we need to rename, or name collisions will happen! 
          tgt.addVariable(newvar, asNC=True, copy=True); del newvar # need to copy to make NC var
    # save all the meta data
    tgt.sync()
    # prepare function call    
    function = functools.partial(self.processExtract, ixlon=ixlon, iylat=iylat, ylat=ylat, xlon=xlon, stnax=stnax) # already set parameters
    # start process
    if self.feedback: print('\n   +++   processing point-data extraction   +++   ') 
    self.process(function, **kwargs) # currently 'flush' is the only kwarg
    if self.feedback: print('\n')
    if self.tmp: self.tmpput = self.target
    if ltmptoo: assert self.tmpput.name == 'tmptoo' # set above, when temp. dataset is created    
  # the previous method sets up the process, the next method performs the computation
  def processExtract(self, var, ixlon=None, iylat=None, ylat=None, xlon=None, stnax=None):
    ''' Extract grid poitns corresponding to stations. '''
    # process gdal variables (if a variable has a horiontal grid, it should be GDAL enabled)
    if var.gdal:
      if self.feedback: print('\n'+var.name),
      tgt = self.target
      assert xlon in var.axes and ylat in var.axes
      assert tgt.hasAxis(stnax, strict=False) and stnax not in var.axes 
      # assemble new axes
      axes = [tgt.getAxis(stnax.name)]      
      for ax in var.axes:
        if ax.name not in (xlon.name,ylat.name) and ax.name != stnax.name: # these axes are just transferred 
          axes.append(tgt.getAxis(ax.name))
      axes = tuple(axes)
      shape = tuple(len(ax) for ax in axes)
      srcdata = var.getArray(copy=False) # don't make extra copy
      # roll x & y axes to the front (xlon first, then ylat, then the rest)
      srcdata = np.rollaxis(srcdata, axis=var.axisIndex(ylat.name), start=0)
      srcdata = np.rollaxis(srcdata, axis=var.axisIndex(xlon.name), start=0)
      assert srcdata.shape == (len(xlon),len(ylat))+shape[1:]
      # here we extract the data points
      if srcdata.ndim == 2:
        tgtdata = srcdata[ixlon,iylat] # constructed above
      elif srcdata.ndim > 2:
        tgtdata = srcdata[ixlon,iylat,:] # constructed above
      else: raise AxisError
      #try: except: print srcdata.shape, [slc.max() for slc in slices] 
      # create new Variable
      assert shape == tgtdata.shape
      newvar = var.copy(axes=axes, data=tgtdata) # new axes and data
      del srcdata, tgtdata # clean up (just to make sure)      
    else:
      var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var # just pass over the variable to the new dataset
    # return variable
    return newvar
    
  # function pair to compute a climatology from a time-series      
  def Regrid(self, griddef=None, projection=None, geotransform=None, size=None, xlon=None, ylat=None, 
             lmask=True, int_interp=None, float_interp=None, **kwargs):
    ''' Setup climatology and start computation; calls processClimatology. '''
    # make temporary gdal dataset
    if self.source is self.target:
      if self.tmp: assert self.source == self.tmpput and self.target == self.tmpput
      # the operation can not be performed "in-place"!
      self.target = Dataset(name='tmptoo', title='Temporary target dataset for non-in-place operations', varlist=[], atts={})
      ltmptoo = True
    else: ltmptoo = False 
    # make sure the target dataset is a GDAL-enabled dataset
    if 'gdal' in self.target.__dict__: 
      # gdal info alread present      
      if griddef is not None or projection is not None or geotransform is not None: 
        raise AttributeError, "Target Dataset '%s' is already GDAL enabled - cannot overwrite settings!"%self.target.name
      if self.target.xlon is None: raise GDALError, "Map axis 'xlon' not found!"
      if self.target.ylat is None: raise GDALError, "Map axis 'ylat' not found!"
      xlon = self.target.xlon; ylat = self.target.ylat
    else:
      # need to set GDAL parameters
      if self.tmp and 'gdal' in self.output.__dict__:
        # transfer gdal settings from output to temporary dataset 
        assert self.target is not self.output 
        projection = self.output.projection; geotransform = self.output.geotransform
        xlon = self.output.xlon; ylat = self.output.ylat
      else:
        # figure out grid definition from input 
        if griddef is None: 
          griddef = GridDefinition(projection=projection, geotransform=geotransform, size=size, xlon=xlon, ylat=ylat)
        # pass arguments through GridDefinition, if not provided
        projection=griddef.projection; geotransform=griddef.geotransform
        xlon=griddef.xlon; ylat=griddef.ylat                     
      # apply GDAL settings target dataset 
      for ax in (xlon,ylat): self.target.addAxis(ax, loverwrite=True) # i.e. replace if already present
      self.target = addGDALtoDataset(self.target, projection=projection, geotransform=geotransform)
    # use these map axes
    xlon = self.target.xlon; ylat = self.target.ylat
    assert isinstance(xlon,Axis) and isinstance(ylat,Axis)
    # determine source dataset grid definition
    if self.source.griddef is None:  
      srcgrd = GridDefinition(projection=self.source.projection, geotransform=self.source.geotransform, 
                              size=self.source.mapSize, xlon=self.source.xlon, ylat=self.source.ylat)
    else: srcgrd = self.source.griddef
    srcres = srcgrd.scale; tgtres = griddef.scale
    # determine if shift is necessary to insure correct wrapping
    if not srcgrd.isProjected and not griddef.isProjected:
      lwrapSrc = srcgrd.wrap360
      lwrapTgt = griddef.wrap360
      # check grids
      for grd in (srcgrd,griddef):
        if grd.wrap360:            
          assert grd.geotransform[0] + grd.geotransform[1]*(len(grd.xlon)-1) > 180        
          assert np.round(grd.geotransform[1]*len(grd.xlon), decimals=2) == 360 # require 360 deg. to some accuracy... 
          assert any( grd.xlon.getArray() > 180 ) # need to wrap around
          assert all( grd.xlon.getArray() >= 0 )
          assert all( grd.xlon.getArray() <= 360 )
        else:
          assert grd.geotransform[0] + grd.geotransform[1]*(len(grd.xlon)-1) < 180
          assert all( grd.xlon.getArray() >= -180 )
          assert all( grd.xlon.getArray() <= 180 )  
    else: 
      lwrapSrc = False # no need to shift, if a projected grid is involved!
      lwrapTgt = False # no need to shift, if a projected grid is involved!
    # determine GDAL interpolation
    if int_interp is None: int_interp = gdalInterp('nearest')
    else: int_interp = gdalInterp(int_interp)
    if float_interp is None:
      if srcres < tgtres: float_interp = gdalInterp('convolution') # down-sampling: 'convolution'
      else: float_interp = gdalInterp('cubicspline') # up-sampling
    else: float_interp = gdalInterp(float_interp)      
    # prepare function call    
    function = functools.partial(self.processRegrid, ylat=ylat, xlon=xlon, lwrapSrc=lwrapSrc, lwrapTgt=lwrapTgt, # already set parameters
                                 lmask=lmask, int_interp=int_interp, float_interp=float_interp)
    # start process
    if self.feedback: print('\n   +++   processing regridding   +++   ') 
    self.process(function, **kwargs) # currently 'flush' is the only kwarg
    # now make sure we have a GDAL dataset!
    self.target = addGDALtoDataset(self.target, griddef=griddef)
    if self.feedback: print('\n')
    if self.tmp: self.tmpput = self.target
    if ltmptoo: assert self.tmpput.name == 'tmptoo' # set above, when temp. dataset is created    
  # the previous method sets up the process, the next method performs the computation
  def processRegrid(self, var, ylat=None, xlon=None, lwrapSrc=False, lwrapTgt=False, lmask=True, int_interp=None, float_interp=None):
    ''' Compute a climatology from a variable time-series. '''
    # process gdal variables
    if var.gdal:
      if self.feedback: print('\n'+var.name),
      # replace axes
      axes = list(var.axes)
      axes[var.axisIndex(var.ylat)] = ylat
      axes[var.axisIndex(var.xlon)] = xlon
      # create new Variable
      var.load() # most rebust way to determine the dtype! and we need it later anyway
      newvar = var.copy(axes=axes, data=None, projection=self.target.projection) # and, of course, load new data
      # if necessary, shift array back, to ensure proper wrapping of coordinates
      # prepare regridding
      # get GDAL dataset instances
      srcdata = var.getGDAL(load=True, wrap360=lwrapSrc)
      tgtdata = newvar.getGDAL(load=False, wrap360=lwrapTgt, allocate=True, fillValue=var.fillValue)
      # determine GDAL interpolation
      if 'gdal_interp' in var.__dict__: gdal_interp = var.gdal_interp
      elif 'gdal_interp' in var.atts: gdal_interp = var.atts['gdal_interp'] 
      else: # use default based on variable type
        if np.issubdtype(var.dtype, np.integer): gdal_interp = int_interp # can't process logicals anyway...
        else: gdal_interp = float_interp                          
      # perform regridding
      err = gdal.ReprojectImage(srcdata, tgtdata, var.projection.ExportToWkt(), newvar.projection.ExportToWkt(), gdal_interp)
      #print srcdata.ReadAsArray().std(), tgtdata.ReadAsArray().std()
      #print var.projection.ExportToWkt()
      #print newvar.projection.ExportToWkt()
      del srcdata # clean up (just to make sure)
      # N.B.: the target array should be allocated and prefilled with missing values, otherwise ReprojectImage
      #       will just fill missing values with zeros!  
      if err != 0: raise GDALError, 'ERROR CODE %i'%err
      #tgtdata.FlushCash()  
      # load data into new variable
      newvar.loadGDAL(tgtdata, mask=lmask, wrap360=lwrapTgt, fillValue=var.fillValue)      
      del tgtdata # clean up (just to make sure)
    else:
      var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var # just pass over the variable to the new dataset
    # return variable
    return newvar
  
  # function pair to compute a climatology from a time-series      
  def Climatology(self, timeAxis='time', climAxis=None, period=None, offset=0, shift=0, timeSlice=None, **kwargs):
    ''' Setup climatology and start computation; calls processClimatology. '''
    if period is not None and not isinstance(period,(np.integer,int)): raise TypeError # period in years
    if not isinstance(offset,(np.integer,int)): raise TypeError # offset in years (from start of record)
    if not isinstance(shift,(np.integer,int)): raise TypeError # shift in month (if first month is not January)
    # construct new time axis for climatology
    if climAxis is None:        
      climAxis = Axis(name=timeAxis, units='month', length=12, coord=np.arange(1,13,1), dtype=dtype_int) # monthly climatology
    else: 
      if not isinstance(climAxis,Axis): raise TypeError
    # add axis to output dataset    
    if self.target.hasAxis(climAxis.name): 
      self.target.repalceAxis(climAxis, check=False) # will have different shape
    else: 
      self.target.addAxis(climAxis, copy=True) # copy=True allows recasting as, e.g., a NC variable
    climAxis = self.target.axes[timeAxis] # make sure we have exactly that instance
    # figure out time slice
    if period is not None:
      start = offset * len(climAxis); end = start + period * len(climAxis)
      timeSlice = slice(start,end,None)
    else: 
      if not isinstance(timeSlice,slice): raise TypeError, timeSlice
    # add variables that will cause errors to ignorelist (e.g. strings)
    for varname,var in self.source.variables.iteritems():
      if var.hasAxis(timeAxis) and var.dtype.kind == 'S': self.ignorelist.append(varname)
    # prepare function call
    function = functools.partial(self.processClimatology, # already set parameters
                                 timeAxis=timeAxis, climAxis=climAxis, timeSlice=timeSlice, shift=shift)
    # start process
    if self.feedback: print('\n   +++   processing climatology   +++   ')     
    if self.source.gdal: griddef = self.source.griddef
    else: griddef = None 
    self.process(function, **kwargs) # currently 'flush' is the only kwarg    
    # add GDAL to target
    if griddef is not None:
      self.target = addGDALtoDataset(self.target, griddef=griddef)
    # N.B.: if the dataset is empty, it wont do anything, hence we do it now    
    if self.feedback: print('\n')    
  # the previous method sets up the process, the next method performs the computation
  def processClimatology(self, var, timeAxis='time', climAxis=None, timeSlice=None, shift=0):
    ''' Compute a climatology from a variable time-series. '''
    # process variable that have a time axis
    if var.hasAxis(timeAxis):
      if self.feedback: print('\n'+var.name),
      # prepare averaging
      tidx = var.axisIndex(timeAxis)
      interval = len(climAxis)
      newshape = list(var.shape)
      newshape[tidx] = interval # shape of the climatology field  
      if not (interval == 12): raise NotImplementedError
      # load data
      if timeSlice is not None:
        idx = tuple([timeSlice if ax.name == timeAxis else slice(None) for ax in var.axes])
      else: idx = None
      dataarray = var.getArray(idx=idx, unmask=False, copy=False)    
      if var.masked: avgdata = ma.zeros(newshape, dtype=var.dtype) # allocate array
      else: avgdata = np.zeros(newshape, dtype=var.dtype) # allocate array    
      # average data
      timelength = dataarray.shape[tidx]
      if timelength % interval == 0:
        # use array indexing
        climelts = np.arange(interval, dtype=dtype_int)
        for t in xrange(0,timelength,interval):
          if self.feedback: print('.'), # t/interval+1
          avgdata += dataarray.take(t+climelts, axis=tidx)
        del dataarray # clean up
        # normalize
        avgdata /= (timelength/interval) 
      else: 
        # simple indexing
        climcnt = np.zeros(interval, dtype=dtype_int)
        for t in xrange(timelength):
          if self.feedback and t%interval == 0: print('.'), # t/interval+1
          idx = int(t%interval)
          climcnt[idx] += 1
          if dataarray.ndim == 1:
            avgdata[idx] = avgdata[idx] + dataarray[t]
          else: 
            avgdata[idx,:] = avgdata[idx,:] + dataarray[t,:]
        del dataarray # clean up
        # normalize
        for i in xrange(interval):
          if avgdata.ndim == 1:
            if climcnt[i] > 0: avgdata[i] /= climcnt[i]
            else: avgdata[i] = 0 if np.issubdtype(var.dtype, np.integer) else np.NaN
          else:
            if climcnt[i] > 0: avgdata[i,:] /= climcnt[i]
            else: avgdata[i,:] = 0 if np.issubdtype(var.dtype, np.integer) else np.NaN
      # shift data (if first month was not January)
      if shift != 0: avgdata = np.roll(avgdata, shift, axis=tidx)
      # create new Variable
      axes = tuple([climAxis if ax.name == timeAxis else ax for ax in var.axes]) # exchange time axis
      newvar = var.copy(axes=axes, data=avgdata, dtype=var.dtype) # and, of course, load new data
      del avgdata # clean up - just to make sure
      #     print newvar.name, newvar.masked
      #     print newvar.fillValue
      #     print newvar.data_array.__class__
    else:
      var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var.copy()
    # return variable
    return newvar
  
  def Shift(self, shift=0, axis=None, byteShift=False, **kwargs):
    ''' Method to initialize shift along a coordinate axis. '''
    # kwarg input
    if shift == 0 and axis == None:
      for key,value in kwargs.iteritems():
        if self.target.hasAxis(key) or self.input.hasAxis(key):
          if axis is None: axis = key; shift = value
          else: raise ProcessError, "Can only process one coordinate shift at a time."
      del kwargs[axis] # remove entry 
    # check input
    if isinstance(axis,basestring):
      if self.target.hasAxis(axis): axis = self.target.axes[axis]
      elif self.input.hasAxis(axis): axis = self.input.axes[axis].copy()
      else: raise AxisError, "Axis '%s' not found in Dataset."%axis
    else: 
      if not isinstance(axis,Axis): raise TypeError
    # apply shift to new axis
    if byteShift:
      # shift coordinate vector like data
      coord = np.roll(axis.getArray(unmask=False), shift) # 1-D      
    else:              
      coord = axis.getArray(unmask=False) + shift # shift coordinates
      # transform coordinate shifts into index shifts (linear scaling)
      shift = int( shift / (axis[1] - axis[0]) )    
    axis.coord = coord
    # add axis to output dataset      
    if self.target.hasAxis(axis, strict=True): pass
    elif self.target.hasAxis(axis.name): self.target.repalceAxis(axis)
    else: self.target.addAxis(axis, copy=True) # copy=True allows recasting as, e.g., a NC variable
    axis = self.target.axes[axis.name] # make sure we have the right version!
    # prepare function call
    function = functools.partial(self.processShift, # already set parameters
                                 shift=shift, axis=axis)
    # start process
    if self.feedback: print('\n   +++   processing shift/roll   +++   ')     
    self.process(function, **kwargs) # currently 'flush' is the only kwarg    
    if self.feedback: print('\n')
  # the previous method sets up the process, the next method performs the computation
  def processShift(self, var, shift=None, axis=None):
    ''' Method that shifts a data array along a given axis. '''
    # only process variables that have the specified axis
    if var.hasAxis(axis.name):
      if self.feedback: print('\n'+var.name), # put line break before test, instead of after      
      # shift data array
      newdata = np.roll(var.getArray(unmask=False), shift, axis=var.axisIndex(axis))
      # create new Variable
      axes = tuple([axis if ax.name == axis.name else ax for ax in var.axes]) # replace axis with shifted version
      newvar = var.copy(axes=axes, data=newdata) # and, of course, load new data
      var.unload(); del var, newdata
    else:
      var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var  
      var.unload(); del var
    # return variable
    return newvar

