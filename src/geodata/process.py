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
# internal imports
from geodata.misc import VariableError
from geodata.base import Variable, Axis


class CentralProcessingUnit(object):
   
  def __init__(self, source, target, function, **kwargs):
    ''' Pass input and output datasets and define the processing function (kwargs are passed to function). 
        The pattern for 'function' is: outvar = function(invar)
    '''
    self.__dict__['input'] = source
    self.__dict__['output'] = target
    self.__dict__['function'] = functools.partial(function, **kwargs) # already set kw-parameters
    
  def process(self, flush=True):
    ''' This method applies the desired operation to each variable in the input dataset. '''    
    # loop over input variables
    for var in self.input:
      # check if variable already exists
      if self.output.hasVariable(var.name):
        newvar = self.function(var)
        oldvar = self.output.variable[var.name]
        if newvar.ndim != oldvar.ndim or newvar.shape != oldvar.shape: raise VariableError
        self.output.variable[var.name].load(newvar.getArray(unmask=False,copy=False))
      else:        
        newvar = self.function(var)
        self.output.addVariable(newvar, copy=True)
        var.unload() # not needed anymore
      # sync data and free space
      if flush:
        newvar.sync()
        newvar.unload()
    
#         print
#         newvar = self.output.precip     
#         print newvar.name, newvar.masked
#         print newvar.fillValue
#         print newvar.data_array.__class__


class ClimatologyProcessingUnit(CentralProcessingUnit):
  
  def __init__(self, source, target, timeAxis='time', climAxis=None, period=None, offset=0):
    ''' Pass input and output datasets and define the processing function (kwargs are passed to function). 
        The pattern for 'function' is: outvar = function(invar)
    '''
    if climAxis is None: # construct new time axis for climatology       
      climAxis = Axis(name=timeAxis, units='month', length=12, data=np.arange(1,13,1)) # monthly climatology
    target.addAxis(climAxis, copy=True)
    climAxis = target.axes[timeAxis] 
    if period is not None:
      start = offset * len(climAxis); end = start + period * len(climAxis)
      timeSlice = slice(start,end,None)      
    # call superior constructor with climatology function and parameters
    super(ClimatologyProcessingUnit,self).__init__(source, target, Climatology, timeAxis=timeAxis, 
                                                   climAxis=climAxis, timeSlice=timeSlice)  

# this is not actually a class method...
def Climatology(var, timeAxis='time', climAxis=None, timeSlice=None):
  ''' Compute a climatology from a time-series. '''
  # process variable that have a time axis
  if var.hasAxis(timeAxis):
    print('\n'+var.name),
    # prepare averaging
    tidx = var.axisIndex(timeAxis)
    interval = len(climAxis)
    newshape = list(var.shape)
    newshape[tidx] = interval # shape of the climatology field  
    if not (interval == 12): raise NotImplemented
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
    newvar = Variable(name=var.name, units=var.units, axes=axes, data=avgdata, dtype=var.dtype, 
                      mask=None, fillValue=var.fillValue, atts=var.atts, plot=var.plot)
    #     print newvar.name, newvar.masked
    #     print newvar.fillValue
    #     print newvar.data_array.__class__
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
  