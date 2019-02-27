'''
Created on 2012-11-10

Some simple functions built on top of the netCDF4-python module. 

@author: Andre R. Erler, GPL v3
'''
# external imports
import netCDF4 as nc # netCDF4-python module: Dataset is probably all we need
import numpy as np
import numpy.ma as ma
import collections as col
from warnings import warn
import os
# internal imports
# N.B.: there should be no dependencies on this package, so that it can be imported independently

## definitions

# NC4 compression options
zlib_default = dict(zlib=True, complevel=1, shuffle=True) # my own default compression settings

# data error class
class NCDataError(Exception):
  ''' Exceptions related to data passed to NetCDF datasets. '''
  pass

# axis error class
class NCAxisError(Exception):
  ''' Exceptions related to axis/dimensions in NetCDF datasets. '''
  pass


## helper functions

def checkFillValue(fillValue, dtype):
  ''' check a fill value and return either return a valid value or raise an exception '''
  lstrvar = dtype.kind == 'S'
  if not lstrvar and isinstance(fillValue, basestring): 
    fillValue = None # invalid value
    warn("Removed malformed fill_value '{:s}'.")
  if fillValue is not None:
    if isinstance(fillValue, np.ndarray): fillValue = np.asscalar(fillValue)
    fillValue = dtype.type(fillValue) # transform into appropriate numpy scalar
    if not np.issubdtype(fillValue,dtype): 
        raise TypeError(fillValue) # convert to Numpy type      
  return fillValue

def coerceAtts(atts):
  ''' Convert an attribute dictionary to a NetCDF compatible format. '''
  if not isinstance(atts,dict): raise TypeError
  ncatts = dict()
  # loop over items
  for key,value in atts.iteritems():
    if key in ('missing_value','fillValue','_FillValue'): pass
    # N.B.: these are special attributes that the NetCDF module will try to read
    elif isinstance(key,basestring) and key[0] == '_' : pass # skip (internal attributes)
    elif value is None: pass # skip (invalid value / not assigned)
    elif not isinstance(value,(basestring,np.ndarray,np.inexact,float,np.integer,int)):
      if 'name' in dir(value):
        ncatts[key] = value.name # mostly for datasets and variables
      elif isinstance(value,col.Iterable):
        if len(value) == 0: ncatts[key] = '' # empty attribute
        elif all([isinstance(val,(int,np.integer,float,np.inexact)) for val in value]):
          # N.B.: int & float are not part of their numpy equivalents
          ncatts[key] = np.array(value)         
        else:
          l = '(' # fake list representation
          for elt in value[0:-1]: l += '{0:s}, '.format(str(elt))
          l += '{0:s})'.format(str(value[-1]))
          ncatts[key] = l          
      else: ncatts[key] = str(value) 
    else: ncatts[key] = value
  return ncatts


## generic netcdf functions

def add_strvar(dst, name, strlist, dim, atts=None):
  ''' Function that adds a list of string variables as a variable along a specified dimension. '''
  warn("WARNING: Use of 'add_strvar' is now deprecated - 'add_var' can also handle string variables. ")
  # determine max length of string
  strlen = 0 
  for string in strlist: strlen = max(strlen,len(string))
  # figure out dimension
  dimlen = len(strlist) # length of dimension
  if dim in dst.dimensions:
    if dimlen != len(dst.dimensions[dim]): raise NCAxisError
  else:
    dst.createDimension(dim, size=dimlen) # create dimension on the fly
  # allocate array
  chararray = np.ndarray((dimlen,strlen), dtype='S1')
  for i in xrange(dimlen):
    jlen = len(strlist[i])
    for j in xrange(jlen):
      chararray[i,j] = strlist[i][j] # unfortunately, direct assignment of sequences does not work
    # fill remaining with spaces 
    if jlen < strlen: chararray[i,jlen:] = ' '
  # create netcdf dimension and variable
  dst.createDimension(name, strlen) # name of month string
  strvar = dst.createVariable(name,'S1',(dim,name))
  strvar[:] = chararray
  # add attributes
  if atts: strvar.setncatts(coerceAtts(atts))
  # return string variable
  return strvar

def add_coord(dst, name, data=None, length=None, atts=None, dtype=None, zlib=True, fillValue=None, **kwargs):
  ''' Function to add a Coordinate Variable to a NetCDF Dataset; returns the Variable reference. '''
  # check input
  if length is None:
    pass # that means unlimited record dimensions (can only have one of these)
  elif length is not None:
    if isinstance(length,(int,np.integer)): length=(length,)
  elif data is not None and data.ndim == 1:
    length=data.shape
  else: 
    raise NCDataError
#   # create dimension
#   if dtype is None and data is None:
#     dst.createDimension(name, size=length[0])
#     return None
#   else:
  # basically a simplified interface for add_var
  coord = add_var(dst, name, (name,), data=data, shape=length, atts=atts, dtype=dtype, 
                  zlib=zlib, fillValue=fillValue, **kwargs)  
  return coord

def add_var(dst, name, dims, data=None, shape=None, atts=None, dtype=None, zlib=True, fillValue=None, 
            lusestr=True, **kwargs):
  ''' Function to add a Variable to a NetCDF Dataset; returns the Variable reference. '''
  # all remaining kwargs are passed on to dst.createVariable()
  # use data array to infer dimensions and data type
  if data is not None:
    if not isinstance(data,np.ndarray): raise TypeError     
    if len(dims) != data.ndim: raise NCDataError("Number of dimensions in '%s' does not match data array."%(name,))    
    if shape: 
      if shape != data.shape: raise NCDataError("Shape of '%s' does not match data array."%(name,))
    else: shape = data.shape
    # get dtype 
    if dtype: 
      if dtype != data.dtype: data = data.astype(dtype)
        # raise NCDataError, "Data type in '%s' does not match data array."%(name,) 
    else: dtype = data.dtype
  if dtype is None: raise NCDataError("Cannot construct a NetCDF Variable without a data array or an abstract data type.")
  dtype = np.dtype(dtype) # use numpy types
  if dtype is np.dtype('bool_'): dtype = np.dtype('i1') # cast numpy bools as 8-bit integers
  lstrvar = ( dtype.kind == 'S' and not lusestr )
  # check/create dimensions
  if shape is None: shape = [None,]*len(dims)
  else: shape = list(shape)
  if len(shape) != len(dims): raise NCAxisError 
  for i,dim in zip(xrange(len(dims)),dims):
    if dim in dst.dimensions:
      if shape[i] is None: 
        shape[i] = len(dst.dimensions[dim])
      else: 
        if shape[i] != len(dst.dimensions[dim]): 
          raise NCAxisError('Size of dimension %s does not match records! %i != %i'%(dim,shape[i],len(dst.dimensions[dim])))
    else: 
      dst.createDimension(dim, size=shape[i])
  dims = tuple(dims); shape = tuple(shape)
  # figure out parameters for variable
  varargs = dict() # arguments to be passed to createVariable
  if isinstance(zlib,dict): varargs.update(zlib)
  elif zlib: varargs.update(zlib_default)
  varargs.update(kwargs)
  if fillValue is None:
    if atts and '_FillValue' in atts: fillValue = atts['_FillValue'] # will be removed later
    elif atts and 'missing_value' in atts: fillValue = atts['missing_value']
    elif data is not None and isinstance(data,ma.MaskedArray): # defaults values for numpy masked arrays
      fillValue = ma.default_fill_value(dtype)
      # if isinstance(dtype,np.bool_): fillValue = True
      # elif isinstance(dtype,np.integer): fillValue = 999999
      # elif isinstance(dtype,np.floating): fillValue = 1.e20
      # elif isinstance(dtype,np.complexfloating): fillValue = 1.e20+0j
      # elif isinstance(dtype,np.flexible): fillValue = 'N/A'
      # else: fillValue = None # for 'object'
    else: pass # if it is not a masked array and no missing value information was passed, don't assign fillValue 
  else:  
    if data is not None and isinstance(data,ma.MaskedArray): data._fill_value = fillValue 
  # make sure fillValue is OK (there have been problems...)    
  fillValue = checkFillValue(fillValue, dtype)
  if fillValue is not None:
    atts['missing_value'] = fillValue # I use fillValue and missing_value the same way
  # add extra dimension for strings
  if lstrvar and dtype.itemsize > 1:
    # add extra dimension
    shape = shape + (dtype.itemsize,)
    dims = dims + ('str_dim_'+name,) # naming pattern for string dimensions
    dst.createDimension(dims[-1], size=shape[-1])
    # change dtype to single char string  
    dtype = np.dtype('|S1')
    # convert string arrays to char arrays
    if data is not None: 
      data = nc.stringtochar(data)
      assert data.dtype == dtype, str(data.dtype)+', '+str(dtype)    
  # create netcdf variable  
  var = dst.createVariable(name, dtype, dims, fill_value=fillValue, **varargs)
  # add attributes
  if atts: var.setncatts(coerceAtts(atts))
  # assign coordinate data if given
  if data is not None: var[:] = data   
  # return var reference
  return var


## copy functions

# copy attributes from a variable or dataset to another
def copy_ncatts(dst, src, prefix = '', incl_=True):
  for att in src.ncattrs(): 
    if att in ('missing_value','fillValue','_FillValue'): pass
    elif att[0] != '_' or incl_: # these seem to cause problems
      dst.setncattr(prefix+att,src.getncattr(att))
      
# copy variables from one dataset to another
def copy_vars(dst, src, varlist=None, namemap=None, dimmap=None, remove_dims=None, copy_data=True, copy_atts=True, \
              zlib=True, prefix='', incl_=True, fillValue=None, **kwargs):
  # prefix is passed to copy_ncatts, the remaining kwargs are passed to dst.createVariable()
  if varlist is None: varlist = src.variables.keys() # just copy all
  if dimmap: midmap = dict(zip(dimmap.values(),dimmap.keys())) # reverse mapping
  varargs = dict() # arguments to be passed to createVariable
  if zlib: varargs.update(zlib_default)
  varargs.update(kwargs)
  dtype = varargs.pop('dtype', None) 
  # loop over variable list
  for name in varlist:
    if namemap and (name in namemap.keys()): rav = src.variables[namemap[name]] # apply name mapping 
    else: rav = src.variables[name]
    dims = [] # figure out dimension list
    for dim in rav.dimensions:
      if dimmap and midmap.has_key(dim): dim = midmap[dim] # apply name mapping (in reverse)
      if not (remove_dims and dim in remove_dims): dims.append(dim)
    # create new variable
    dtype = dtype or rav.dtype
    if '_FillValue' in rav.ncattrs(): fillValue = rav.getncattr('_FillValue')
    var = dst.createVariable(name, dtype, dims, fill_value=fillValue, **varargs)
    if copy_atts: copy_ncatts(var, rav, prefix=prefix, incl_=incl_) # copy attributes, if desired (default) 
    if copy_data: var[:] = rav[:] # copy actual data, if desired (default)

# copy dimensions and coordinate variables from one dataset to another
def copy_dims(dst, src, dimlist=None, namemap=None, copy_coords=True, **kwargs):
  # all remaining kwargs are passed on to dst.createVariable()
  if dimlist is None: dimlist = src.dimensions.keys() # just copy all
  if namemap is None: namemap = dict() # a dummy - assigning pointers in argument list is dangerous! 
  # loop over dimensions
  for name in dimlist:
    mid = src.dimensions[namemap.get(name,name)]
    # create actual dimensions
    dst.createDimension(name, size=len(mid))
  # create coordinate variable
  if copy_coords:
#    if kwargs.has_key('dtype'): kwargs['datatype'] = kwargs.pop('dtype') # different convention... 
    remove_dims = [dim for dim in src.dimensions.keys() if dim not in dimlist] # remove_dims=remove_dims
    dimlist = [dim for dim in dimlist if dim in src.variables] # only the ones that have coordinates
    copy_vars(dst, src, varlist=dimlist, namemap=namemap, dimmap=namemap, remove_dims=remove_dims, **kwargs)
    

## Dataset functions

def writeNetCDF(dataset, ncfile, ncformat='NETCDF4', zlib=True, writeData=True, overwrite=True, skipUnloaded=False, 
                feedback=False, close=True):
  ''' A function to write the data in a generic Dataset to a NetCDF file. '''
  if feedback: print("Writing to file: '{:s}'".format(ncfile)) # print feedback
  # open file
  if isinstance(ncfile,basestring): 
    if not overwrite and os.path.exists(ncfile): raise IOError("File '{:s}' already exists and 'overwrite' set to False.".format(ncfile))
    ncfile = nc.Dataset(ncfile, mode='w', format=ncformat, clobber=overwrite)
  elif not isinstance(ncfile,nc.Dataset): raise TypeError
  #if ncfile.mode == 'r': raise NCDataError, "Need write permission on NetCDF dataset."
  ncfile.setncatts(coerceAtts(dataset.atts))
  # add coordinate variables first
  for name,ax in dataset.axes.items():
    # only need to add real coordinate axes; simple dimensions are added on-the-fly by ariables
    data = ax.getArray(unmask=True) if writeData and ( ax.data or not skipUnloaded ) else None
    add_coord(ncfile, name, length=len(ax), data=data, atts=coerceAtts(ax.atts), dtype=ax.dtype, zlib=zlib, fillValue=ax.fillValue)
  # now add variables
  for name,var in dataset.variables.items():
    dims = tuple([ax.name for ax in var.axes])
    #data = var.getArray(unmask=True) if writeData and ( var.data or not skipUnloaded ) else None  
    add_var(ncfile, name, dims=dims, data=var.data_array, atts=coerceAtts(var.atts), 
            dtype=var.dtype, zlib=zlib, fillValue=var.fillValue)
  # close file or return file handle
  ncfile.sync()
  if close: ncfile.close()
  else: return ncfile
  