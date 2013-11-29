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
from geodata.misc import VariableError, AxisError, PermissionError, DatasetError, GDALError #, DateError
from geodata.base import Axis, Dataset
from geodata.netcdf import DatasetNetCDF, asDatasetNC
from geodata.nctools import writeNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition, gdalInterp

class ProcessError(Exception):
  ''' Error class for exceptions occurring in methods of the CPU (CentralProcessingUnit). '''
  pass

class CentralProcessingUnit(object):
  
  def __init__(self, source, target=None, varlist=None, tmp=True, feedback=True):
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
    # whether or not to print status output
    self.__dict__['feedback'] = feedback
        
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
          self.output.addVariable(var, overwrite=True, deepcopy=copydata)
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
        newvar.unload() # free space; already added to new dataset
      else:
        raise DatasetError, "Variable '%s' not found in input dataset."%varname
      # free space (in case garbage collection fails...) 
      var.unload() # not needed anymore
      #print self.target.pmsl.data_array.mean()
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
             mask=True, int_interp=None, float_interp=None, **kwargs):
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
    # determine source dataset grid definition 
    srcgrd = GridDefinition(projection=self.source.projection, geotransform=self.source.geotransform, 
                            size=self.source.mapSize, xlon=self.source.xlon, ylat=self.source.ylat)
    srcres = srcgrd.scale; tgtres = griddef.scale
    # determine GDAL interpolation
    if int_interp is None: int_interp = gdalInterp('nearest')
    else: int_interp = gdalInterp(int_interp)
    if float_interp is None:
      if srcres < tgtres: float_interp = gdalInterp('convolution') # down-sampling: 'convolution'
      else: float_interp = gdalInterp('cubicspline') # up-sampling
    else: float_interp = gdalInterp(float_interp)      
    # prepare function call    
    function = functools.partial(self.processRegrid, ylat=ylat, xlon=xlon, # already set parameters
                                 mask=mask, int_interp=int_interp, float_interp=float_interp)
    # start process
    if self.feedback: print('\n   +++   processing regridding   +++   ') 
    self.process(function, **kwargs) # currently 'flush' is the only kwarg
    if self.feedback: print('\n')
    if self.tmp: self.tmpput = self.target
    if ltmptoo: assert self.tmpput.name == 'tmptoo' # set above, when temp. dataset is created    
  # the previous method sets up the process, the next method performs the computation
  def processRegrid(self, var, ylat=None, xlon=None, mask=True, int_interp=None, float_interp=None):
    ''' Compute a climatology from a variable time-series. '''
    # process gdal variables
    if var.gdal:
      if self.feedback: print('\n'+var.name),
      # replace axes
      axes = list(var.axes)
      axes[var.axisIndex(var.ylat)] = ylat
      axes[var.axisIndex(var.xlon)] = xlon
      # create new Variable
      newvar = var.copy(axes=axes, data=None, dtype=var.dtype, projection=self.target.projection) # and, of course, load new data
      # prepare regridding
      # get GDAL dataset instances
      srcdata = var.getGDAL(load=True)
      tgtdata = newvar.getGDAL(load=False, allocate=True, fillValue=var.fillValue)
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
      # N.B.: the target array should be allocated and prefilled with missing values, otherwise ReprojectImage
      #       will just fill missing values with zeros!  
      if err != 0: raise GDALError, 'ERROR CODE %i'%err
      #tgtdata.FlushCash()  
      # load data into new variable
      newvar.loadGDAL(tgtdata, mask=mask, fillValue=var.fillValue)
    else:
      if not var.data: var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var  
    # return variable
    return newvar
  
  # function pair to compute a climatology from a time-series      
  def Climatology(self, timeAxis='time', climAxis=None, period=None, offset=0, shift=0, **kwargs):
    ''' Setup climatology and start computation; calls processClimatology. '''
    if period is not None and not isinstance(period,(np.integer,int)): raise TypeError # period in years
    if not isinstance(offset,(np.integer,int)): raise TypeError # offset in years (from start of record)
    if not isinstance(shift,(np.integer,int)): raise TypeError # shift in month (if first month is not January)
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
    # add GDAL to target
    if self.source.gdal: 
      self.target = addGDALtoDataset(self.target, projection=self.source.projection, geotransform=self.source.geotransform)
    # prepare function call
    function = functools.partial(self.processClimatology, # already set parameters
                                 timeAxis=timeAxis, climAxis=climAxis, timeSlice=timeSlice, shift=shift)
    # start process
    if self.feedback: print('\n   +++   processing climatology   +++   ')     
    self.process(function, **kwargs) # currently 'flush' is the only kwarg    
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
      if var.masked: avgdata = ma.zeros(newshape) # allocate array
      else: avgdata = np.zeros(newshape) # allocate array    
      # average data
      timelength = dataarray.shape[tidx]
      if timelength % interval == 0:
        # use array indexing
        climelts = np.arange(interval)
        for t in xrange(0,timelength,interval):
          if self.feedback: print('.'), # t/interval+1
          avgdata += dataarray.take(t+climelts, axis=tidx)
        # normalize
        avgdata /= (timelength/interval) 
      else: 
        # simple indexing
        climcnt = np.zeros(interval)
        for t in xrange(timelength):
          if self.feedback and t%interval == 0: print('.'), # t/interval+1
          idx = int(t%interval)
          climcnt[idx] += 1
          if dataarray.ndim == 1:
            avgdata[idx] = avgdata[idx] + dataarray[t]
          else: 
            avgdata[idx,:] = avgdata[idx,:] + dataarray[t,:]
        # normalize
        for i in xrange(interval):
          if avgdata.ndim == 1:
            if climcnt[i] > 0: avgdata[i] /= climcnt[i]
            else: avgdata[i] = np.NaN
          else:
            if climcnt[i] > 0: avgdata[i,:] /= climcnt[i]
            else: avgdata[i,:] = np.NaN
        #raise NotImplementedError, "The length of the time series has to be divisible by {0:d}.".format(interval)
      # shift data (if first month was not January)
      if shift != 0: avgdata = np.roll(avgdata, shift, axis=tidx)
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
    else:
      var.load() # need to load variables into memory, because we are not doing anything else...
      newvar = var  
    # return variable
    return newvar

