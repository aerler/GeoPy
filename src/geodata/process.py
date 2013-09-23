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
from osgeo import gdal
# internal imports
from geodata.misc import VariableError, AxisError, PermissionError, DatasetError, GDALError
from geodata.base import Axis, Dataset
from geodata.netcdf import DatasetNetCDF, asDatasetNC
from geodata.nctools import writeNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition, gdalInterp

class ProcessError(Exception):
  ''' Error class for exceptions occurring in methods of the CPU (CentralProcessingUnit). '''
  pass


class CentralProcessingUnit(object):
  
  def __init__(self, source, target=None, varlist=None, tmp=False):
    ''' Initialize processor and pass input and output datasets. '''
    # check varlist
    if varlist is None: varlist = source.variables.keys() # all source variables
    if not isinstance(varlist,(list,tuple)): raise TypeError
    self.__dict__['varlist'] = varlist # list of variable to be processed
    # check input
    if not isinstance(source,Dataset): raise TypeError
    if isinstance(source,DatasetNetCDF) and not 'r' in source.mode: raise PermissionError
    self.__dict__['input'] = source
    self.__dict__['source'] = source
    # check output
    if target is not None:       
      if not isinstance(target,Dataset): raise TypeError
      if isinstance(target,DatasetNetCDF) and not 'w' in target.mode: raise PermissionError
    else:
      if not tmp: raise DatasetError, "Need target location, if temporary storage is disables (tmp=False)." 
    self.__dict__['output'] = target
    # temporary dataset
    self.__dict__['tmp'] = tmp
    if tmp: self.__dict__['tmpput'] = Dataset(name='tmp', title='Temporary Dataset', varlist=[], atts={})
    else: self.__dict__['tmpput'] = None
    # determine if temporary storage is used and assign target dataset
    if self.tmp: self.__dict__['target'] = self.tmpput
    else: self.__dict__['target'] = self.output 
        
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
  
  def sync(self, varlist=None, flush=False, gdal=True, deepcopy=False):
    ''' Transfer contents of temporary storage to output/target dataset. '''
    if not isinstance(self.output,Dataset): raise DatasetError, "Cannot sync without target Dataset!"
    if self.tmp:
      if varlist is None: varlist = self.tmpput.variables.keys()  
      for varname in varlist:
        if varname in self.tmpput.variables:
          var = self.tmpput.variables[varname]
          self.output.addVariable(var, overwrite=True, deepcopy=deepcopy)
          if flush: var.unload() # remove unnecessary references (unlink data)
      print self.output
      if gdal and 'gdal' in self.tmpput.__dict__: 
        if self.tmpput.gdal: 
          projection = self.tmpput.projection; geotransform = self.tmpput.geotransform
          #xlon = self.tmpput.xlon; ylat = self.tmpput.ylat 
        else: 
          projection=None; geotransform=None; #xlon = None; ylat = None 
        self.output = addGDALtoDataset(self.output, projection=projection, geotransform=geotransform)
#           self.source = self.output # future operations will write to the output dataset directly
#           self.target = self.output # future operations will write to the output dataset directly                     
        
  def writeNetCDF(self, filename=None, ncformat='NETCDF4', zlib=True, writeData=True, close=False, flush=False):
    ''' Write current temporary storage to a NetCDF file. '''
    if self.tmp:
      if not isinstance(filename,basestring): raise TypeError
      if flush: self.tmpput.unload()
      output = writeNetCDF(self, filename, ncformat=ncformat, zlib=zlib, writeData=writeData, close=False)
    else: 
      self.output.sync()
      output = self.output.dataset # get (primary) NetCDF file
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
        self.source = self.tmpput
        self.target = self.output
        self.tmp = False # not using temporary storage anymore
    # loop over input variables
    for varname in self.varlist:
      # check if variable already exists
      if self.target.hasVariable(varname):
        # "in-place" operations
        var = self.target.variables[varname] 
        newvar = function(var)
        if newvar.ndim != var.ndim or newvar.shape != var.shape: raise VariableError
        if newvar is not var: self.target.replaceVariable(var,newvar)
      elif self.source.hasVariable(varname):        
        var = self.source.variables[varname] 
        # perform operation from source and copy results to target
        newvar = function(var)
        self.target.addVariable(newvar, copy=True) # copy=True allows recasting as, e.g., a NC variable
      else:
        raise DatasetError, "Variable '%s' not found in input dataset."%varname
      # free space (in case garbage collection fails...) 
      var.unload() # not needed anymore
      newvar.unload() # already added to new dataset
      # flush data to disk immediately
      if flush:
        outvar = self.output.variables[newvar.name]
        outvar.sync() # sync this variable
        self.output.dataset.sync() # sync NetCDF dataset, but don't call sync on all the other variables...
        outvar.unload() # again, free memory
    # after everything is said and done:
    self.source = self.target # set target to source for next time
    
    
  ## functions (or function pairs) that perform operations on the data
  
  # function pair to compute a climatology from a time-series      
  def Regrid(self, griddef=None, projection=None, geotransform=None, size=None, xlon=None, ylat=None, 
             mask=True, int_interp='nearest', float_interp='bilinear', **kwargs):
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
        if griddef is None: griddef = GridDefinition(projection=projection, geotransform=geotransform, 
                                                     size=size, xlon=xlon, ylat=ylat)
        # pass arguments through GridDefinition, if not provided
        projection=griddef.projection; geotransform=griddef.geotransform
        xlon=griddef.xlon; ylat=griddef.ylat                     
      # apply GDAL settings target dataset 
      for ax in (xlon,ylat):
        #if self.target.hasVariable(ax.name): self.target.removeVariable(ax)
        self.target.addAxis(ax, overwrite=True) # i.e. replace is already present
      self.target = addGDALtoDataset(self.target, projection=projection, geotransform=geotransform)
    # use these map axes
    xlon = self.target.xlon; ylat = self.target.ylat
    assert isinstance(xlon,Axis) and isinstance(ylat,Axis)
    # determine GDAL interpolation
    int_interp = gdalInterp(int_interp)
    float_interp = gdalInterp(float_interp)      
    # prepare function call
    function = functools.partial(self.processRegrid, ylat=ylat, xlon=xlon, # already set parameters
                                 mask=mask, int_interp=int_interp, float_interp=float_interp)
    # start process
    self.process(function, **kwargs) # currently 'flush' is the only kwarg
    if self.tmp: self.tmpput = self.target
    if ltmptoo: assert self.tmpput.name == 'tmptoo' # set above, when temp. dataset is created    
  # the previous method sets up the process, the next method performs the computation
  def processRegrid(self, var, ylat=None, xlon=None, mask=True, int_interp=None, float_interp=None):
    ''' Compute a climatology from a variable time-series. '''
    # process gdal variables
    if var.gdal:
      print('\n'+var.name),
      # replace axes
      axes = list(var.axes)
      axes[var.axisIndex(var.ylat)] = ylat
      axes[var.axisIndex(var.xlon)] = xlon
      # create new Variable
      newvar = var.copy(axes=axes, data=None, projection=self.target.projection) # and, of course, load new data
      # prepare regridding
      # get GDAL dataset instances
      srcdata = var.getGDAL(load=True)
      tgtdata = newvar.getGDAL(load=False, allocate=True, fillValue=var.fillValue)
#       print tgtdata.ReadAsArray().mean()
#       print newvar.projection
      # determine GDAL interpolation
      if 'gdal_interp' in var.__dict__: gdal_interp = var.gdal_interp
      elif 'gdal_interp' in var.atts: gdal_interp = var.atts['gdal_interp'] 
      else: # use default based on variable type
        if np.issubdtype(var.dtype, np.integer): gdal_interp = int_interp # can't process logicals anyway...
        else: gdal_interp = float_interp                          
      # perform regridding
      err = gdal.ReprojectImage(srcdata, tgtdata, var.projection.ExportToWkt(), newvar.projection.ExportToWkt(), gdal_interp)
      # N.B.: the target array should be allocated and prefilled with missing values, otherwise ReprojectImage
      #       will just fill missing values with zeros!  
      if err != 0: raise GDALError, 'ERROR CODE %i'%err  
      # load data into new variable
      newvar.loadGDAL(tgtdata, mask=mask, fillValue=var.fillValue)
    else:
      if not var.data: var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var  
    # return variable
    return newvar
  
  # function pair to compute a climatology from a time-series      
  def Climatology(self, timeAxis='time', climAxis=None, period=None, offset=0, **kwargs):
    ''' Setup climatology and start computation; calls processClimatology. '''
    # construct new time axis for climatology
    if climAxis is None:        
      climAxis = Axis(name=timeAxis, units='month', length=12, data=np.arange(1,13,1)) # monthly climatology
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
      if not isinstance(timeSlice,slice): raise TypeError
    # prepare function call
    function = functools.partial(self.processClimatology, # already set parameters
                                 timeAxis=timeAxis, climAxis=climAxis, timeSlice=timeSlice)
    # start process
    self.process(function, **kwargs) # currently 'flush' is the only kwarg    
  # the previous method sets up the process, the next method performs the computation
  def processClimatology(self, var, timeAxis='time', climAxis=None, timeSlice=None):
    ''' Compute a climatology from a variable time-series. '''
    # process variable that have a time axis
    if var.hasAxis(timeAxis):
      print('\n'+var.name),
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
      if var.masked: avgdata = ma.zeros(newshape) # allocate array
      else: avgdata = np.zeros(newshape) # allocate array    
      # average data
      timelength = dataarray.shape[tidx]
      if timelength % interval == 0:
        # use array indexing
        climelts = np.arange(interval)
        for t in xrange(0,timelength,interval):
          print('.'), # t/interval+1
          avgdata += dataarray.take(t+climelts, axis=tidx)
        # normalize
        avgdata /= (timelength/interval) 
      else: raise NotImplementedError
      # create new Variable
      axes = tuple([climAxis if ax.name == timeAxis else ax for ax in var.axes]) # exchange time axis
      newvar = var.copy(axes=axes, data=avgdata) # and, of course, load new data
      #     print newvar.name, newvar.masked
      #     print newvar.fillValue
      #     print newvar.data_array.__class__
    else:
      if not var.data: var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var  
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
      coord = np.roll(axis.getArray(unmask=False), shift=shift)      
    else:              
      coord = axis.getArray(unmask=False) + shift # shift coordinates
      # transform coordinate shifts into index shifts (linear scaling)
      shift = int( shift / (axis[1] - axis[0]) )    
    axis.updateCoord(coord=coord)
    # add axis to output dataset      
    if self.target.hasAxis(axis, strict=True): pass
    elif self.target.hasAxis(axis.name): self.target.repalceAxis(axis)
    else: self.target.addAxis(axis, copy=True) # copy=True allows recasting as, e.g., a NC variable
    axis = self.target.axes[axis.name] # make sure we have the right version!
    # prepare function call
    function = functools.partial(self.processShift, # already set parameters
                                 shift=shift, axis=axis)
    # start process
    self.process(function, **kwargs) # currently 'flush' is the only kwarg
  # the previous method sets up the process, the next method performs the computation
  def processShift(self, var, shift=None, axis=None):
    ''' Method that shifts a data array along a given axis. '''
    # only process variables that have the specified axis
    if var.hasAxis(axis.name):
      print('\n'+var.name), # put line break before test, instead of after      
      # shift data array
      newdata = np.roll(var.getArray(unmask=False), shift, axis=var.axisIndex(axis))
      # create new Variable
      axes = tuple([axis if ax.name == axis.name else ax for ax in var.axes]) # replace axis with shifted version
      newvar = var.copy(axes=axes, data=newdata) # and, of course, load new data
    else:
      var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var  
    # return variable
    return newvar


"""

## Class for regridding  datasets
class NetcdfRegrid(NetcdfProcessor):

  ## member variables
  inCoords = None # names and values of input coordinate vectors (dict)
  outCoords = None # names and values of output coordinate vectors (dict)
  mapCoords = None # mapping of map coordinates, i.e. lon -> x / lat -> y (dict)

  ## member methods
  # constructor
  def __init__(self, **kwargs):
    ''' Define names of input and output datasets and set general processing parameters. '''
    super(NetcdfRegrid,self).__init__(**kwargs)
    self.inCoords = dict()
    self.outCoords = dict()
    self.mapCoords = dict()      
  
  # define parameters of input dataset
  def initInput(self, epsg=4326, **kwargs):
    ''' This method defines parameters of the input dataset. '''
    # open input datasets
    super(NetcdfRegrid,self).initInput(**kwargs)
    # add regridding functionality
    if epsg == 4326:
      # spherical coordinates
      from regrid import LatLonProj
      lon = self.indata.variables['lon'][:]; self.inCoords['lon'] = lon
      lat = self.indata.variables['lat'][:]; self.inCoords['lat'] = lat
      self.inProj = LatLonProj(lon, lat)
    elif epsg is None:
      # euclidian coordinates 
      x = self.indata.variables['x'][:]; self.inCoords['x'] = x
      y = self.indata.variables['y'][:]; self.inCoords['y'] = y      
    
  # define parameters of output dataset
  def initOutput(self, template=None, lon=None, lat=None, x=None, y=None, epsg=4326, **kwargs):
    ''' This method defines output parameters and initializes the output dataset. '''
    assert ( isinstance(lon,np.ndarray) and isinstance(lat,np.ndarray) ) or \
           ( isinstance(x,np.ndarray) and isinstance(y,np.ndarray) ) or \
           ( isinstance(template,Dataset) or isinstance(template,str) ), \
           'Either input arguments \'lon\'/\'lat\' or \'x\'/\'y\' need to be defined (as numpy arrays)!' 
    # create output dataset
    super(NetcdfRegrid,self).initOutput(**kwargs)
    # check template
    if template: 
      if isinstance(template,str): template = Dataset(template)
      if template.variables.has_key('lon') and template.variables.has_key('lat') and \
         (len(template.variables['lon'].dimensions) == 1) and \
         (len(template.variables['lat'].dimensions) == 1): epsg = 4326 
    # add regridding functionality
    if epsg == 4326:
      # spherical coordinates
      from regrid import LatLonProj
      if template: # get coordinate arrays from template 
        lon = template.variables['lon'][:]; lat = template.variables['lat'][:]
      self.outCoords['lon'] = lon; self.outCoords['lat'] = lat
      self.outProj = LatLonProj(lon, lat)
    elif epsg is None: 
      # euclidian coordinates
      if template: # get coordinate arrays from template 
        x = template.variables['x'][:]; y = template.variables['y'][:]
      self.outCoords['x'] = x; self.outCoords['y'] = x
  
  # set operation parameters
  def defineOperation(self, interpolation='', **kwargs):
    ''' This method defines the operation and the parameters for the operation performed on the dataset. '''
    super(NetcdfRegrid,self).defineOperation(**kwargs)
    self.interpolation = interpolation
    self.mapCoords = dict(zip(self.inCoords.keys(), self.outCoords.keys()))
  
  # perform operation (dummy method)
  def performOperation(self, **kwargs):
    ''' This method performs the actual operation on the variables; it is defined in specialized child classes. '''
    # regridding is performed by regridArray function
    from regrid import regridArray
    # get variable
    varname = kwargs['name']
    ncvar = self.indata.variables[varname]
    # copy meta data 
    newname = varname
    newdims = [self.mapCoords.get(dim,dim) for dim in ncvar.dimensions] # map horizontal coordinate dimensions
    newdtype = ncvar.dtype
    newatts = dict(zip(ncvar.ncattrs(),[ncvar.getncattr(att) for att in ncvar.ncattrs()]))
    # decide what to do
#     print ncvar
#     print self.inCoords.keys()
#     print self.outCoords.keys()
#     print '\n\n'
    if self.inCoords.viewkeys() <= set(ncvar.dimensions): # 2D or more will be regridded
      data = ncvar[:] # the netcdf module returns masked arrays! 
      if '_FillValue' in ncvar.ncattrs(): 
        fillValue = ncvar.getncattr('_FillValue')
        if isinstance(data,np.ma.masked_array): data = data.filled(fillValue)
      else: fillValue = None
      newvals = regridArray(data, self.inProj, self.outProj, interpolation=self.interpolation, missing=fillValue)
#       if fillValue: 
#         newvals = np.ma.masked_where(newvals == 0, newvals)
#       import pylab as pyl
#       pyl.imshow(np.flipud(newvals[0,:,:])); pyl.colorbar(); pyl.show(block=True)
    elif varname in self.inCoords: 
      newname = self.mapCoords[varname] # new name for map coordinate
      newvals = self.outCoords[newname] # assign new coordinate values
    else: # other coordinate variables are left alone
      newvals = ncvar[:]
    # this method needs to return all the information needed to create a new netcdf variable    
    return newname, newvals, newdims, newatts, newdtype

## some code for testing 
if __name__ == '__main__':

  # input dataset
  infolder = '/media/tmp/' # RAM disk
  prismfile = infolder + 'prismavg/prism_clim.nc'
  gpccfile = infolder + 'gpccavg/gpcc_05_clim_1979-1981.nc' 
  # output dataset
  outfolder = '/media/tmp/test/' # RAM disk
#   outfile = outfolder + 'prism_test.nc'
  outfile = outfolder + 'prism_10km.nc'

  ## launch test
#   ncpu = NetcdfProcessor(infile=infile, outfile=outfile, prefix='test_')
#   ncpu.initInput(); ncpu.initOutput(); ncpu.defineOperation()
  ncpu = NetcdfRegrid(infile=prismfile, outfile=outfile, prefix='test_')
  ncpu.initInput()
  dlon = dlat = 1./6.
  lon = -142. + np.arange(int(29./dlon))*dlon # PRISM longitude: -142 to -113
  lat = 47. + np.arange(int(25./dlat))*dlat # PRISM latitude: 47 to 72
  ncpu.initOutput(lon=lon,lat=lat)
  ncpu.defineOperation(interpolation='cubicspline')
  outdata = ncpu.processDataset()
  
  ## show output
  outdata = Dataset(outfile, 'r')
  print
  print outdata
  
    # display
  import pylab as pyl
  vardata = outdata.variables['rain']
  for i in xrange(1):
#     pyl.imshow(outdata[i,:,:]); pyl.colorbar(); pyl.show(block=True)
#     pyl.imshow(np.flipud(likeData.variables['rain'][i,:,:])); pyl.colorbar(); pyl.show(block=True)
    pyl.imshow(np.flipud(vardata[i,:,:])); pyl.colorbar(); pyl.show(block=True)
#     pyl.imshow(np.flipud(outdata[i,:,:]-likeData.variables['rain'][i,:,:])); pyl.colorbar(); pyl.show(block=True)  
    
"""
  