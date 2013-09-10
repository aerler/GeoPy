'''
Created on 2012-11-10

Some simple functions built on top of the netCDF4-python module. 

@author: Andre R. Erler, GPL v3
'''
# external imports
import netCDF4 as nc # netCDF4-python module: Dataset is probably all we need
import numpy as np
import collections as col
# internal imports
from geodata.misc import isNumber

## definitions
# NC4 compression options
zlib_default = dict(zlib=True, complevel=1, shuffle=True) # my own default compression settings

## generic functions

# add a new dimension with coordinate variable
def add_coord(dst, name, values=None, length=None, atts=None, dtype=None, zlib=True, fillValue=None, **kwargs):
  # all remaining kwargs are passed on to dst.createVariable()
  # create dimension
  if dst.dimensions.has_key(name):
    assert len(values) == len(dst.dimensions[name]), '\nWARNING: Dimensions %s already present and size does not match!\n'%(name,) 
  else:
    if values is not None:
      if not dtype: dtype = values.dtype # should be standard... 
      dst.createDimension(name, size=len(values))
    elif length is not None:
      dst.createDimension(name, size=length)
    else:
      dst.createDimension(name, size=None) # unlimited dimension
  # create coordinate variable
  varargs = dict() # arguments to be passed to createVariable
  if zlib: varargs.update(zlib_default)
  varargs.update(kwargs)
  if fillValue is not None: atts['_FillValue'] = fillValue
  elif atts and '_FillValue' in atts: fillValue = atts['_FillValue']
  else: fillValue = None # masked array handling could go here 
  coord = dst.createVariable(name, dtype, (name,), fill_value=fillValue,  **varargs)
  if values is not None: coord[:] = values # assign coordinate values if given  
  if atts: # add attributes
    for key,value in atts.iteritems():
      coord.setncattr(key,value) 
      
def add_var(dst, name, dims, values=None, atts=None, dtype=None, zlib=True, fillValue=None, **kwargs):
  # all remaining kwargs are passed on to dst.createVariable()
  # use values array to infer dimensions and data type
  if values is not None: 
    # check/create dimension
    assert len(dims) == values.ndim, '\nWARNING: Number of dimensions does not match (%s)!\n'%(name,)    
    for i in xrange(len(dims)):
      if dst.dimensions.has_key(dims[i]):
        assert values.shape[i] == len(dst.dimensions[dims[i]]), \
              '\nWARNING: Size of dimension %s does not match!\n'%(dims[i],)
      else: dst.createDimension(dims[i], size=values.shape[i])
    if not dtype: dtype = values.dtype # infer data type, if not specified 
  # create coordinate variable
  varargs = dict() # arguments to be passed to createVariable
  if zlib: varargs.update(zlib_default)
  varargs.update(kwargs)
  if fillValue is not None: atts['_FillValue'] = fillValue
  elif atts and '_FillValue' in atts: fillValue = atts['_FillValue']
  else: fillValue = None # masked array handling could go here
  if fillValue is not None: atts['missing_value'] = fillValue # I use fillValue and missing_value the same way
  var = dst.createVariable(name, dtype, dims, fill_value=fillValue, **varargs)
  if atts: # add attributes
    for key,value in atts.iteritems():
#       print key, value
      if key[0] != '_': var.setncattr(key,value)  
  if values is not None: var[:] = values # assign coordinate values if given  


## copy functions

# copy attributes from a variable or dataset to another
def copy_ncatts(dst, src, prefix = '', incl_=True):
  for att in src.ncattrs(): 
    if att[0] != '_' or incl_: # these seem to cause problems
      dst.setncattr(prefix+att,src.getncattr(att))
      
# copy variables from one dataset to another
def copy_vars(dst, src, varlist=None, namemap=None, dimmap=None, remove_dims=None, copy_data=True, copy_atts=True, \
              zlib=True, prefix='', incl_=True, fillValue=None, **kwargs):
  # prefix is passed to copy_ncatts, the remaining kwargs are passed to dst.createVariable()
  if not varlist: varlist = src.variables.keys() # just copy all
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
    if copy_data: var[:] = rav[:] # copy actual data, if desired (default)
    if copy_atts: copy_ncatts(var, rav, prefix=prefix, incl_=incl_) # copy attributes, if desired (default) 

# copy dimensions and coordinate variables from one dataset to another
def copy_dims(dst, src, dimlist=None, namemap=None, copy_coords=True, **kwargs):
  # all remaining kwargs are passed on to dst.createVariable()
  if not dimlist: dimlist = src.dimensions.keys() # just copy all
  if not namemap: namemap = dict() # a dummy - assigning pointers in argument list is dangerous! 
  # loop over dimensions
  for name in dimlist:
    mid = src.dimensions[namemap.get(name,name)]
    # create actual dimensions
    dst.createDimension(name, size=len(mid))
  # create coordinate variable
  if copy_coords:
#    if kwargs.has_key('dtype'): kwargs['datatype'] = kwargs.pop('dtype') # different convention... 
    remove_dims = [dim for dim in src.dimensions.keys() if dim not in dimlist] # remove_dims=remove_dims
    copy_vars(dst, src, varlist=dimlist, namemap=namemap, dimmap=namemap, remove_dims=remove_dims, **kwargs)
    

## Dataset functions

def coerceAtts(atts):
  ''' Convert an attribute dictionary to a NetCDF compatible format. '''
  if not isinstance(atts,dict): raise TypeError
  ncatts = atts.copy()
  # loop over items
  for key,value in ncatts.iteritems():
    if not isinstance(value,(basestring,np.ndarray,float,int)):
      if isinstance(value,col.Iterable):
        if len(value) == 0: ncatts[key] = '' # empty attribute
        elif all(isNumber(value)): ncatts[key] = np.array(value)
        else:
          l = '(' # fake list representation
          for elt in value[0:-1]: l += '{0:s}, '.format(str(elt))
          l += '{0:s})'.format(str(value[-1]))
          ncatts[key] = l          
      else: ncatts[key] = str(value) 
  return ncatts

def writeNetCDF(dataset, filename, ncformat='NETCDF4', zlib=True, writeData=True, close=True):
  ''' A function to write the data in a generic Dataset to a NetCDF file. '''
  # open file
  print filename,ncformat
  ncfile = nc.Dataset(filename, mode='w', format=ncformat)
  ncfile.setncatts(coerceAtts(dataset.atts))
  # add coordinate variables first
  for name,ax in dataset.axes.iteritems():
    if ax.data: # only need to add real coordinate axes; simple dimensions are added on-the-fly below
      if writeData:
        add_coord(ncfile, name, values=ax.getArray(unmask=True), atts=coerceAtts(ax.atts), dtype=ax.dtype, zlib=zlib, fillValue=ax.fillValue)
      else:
        add_coord(ncfile, name, length=len(ax), atts=coerceAtts(ax.atts), dtype=ax.dtype, zlib=zlib, fillValue=ax.fillValue)
  # now add variables
  for name,var in dataset.variables.iteritems():
    dims = tuple([ax.name for ax in var.axes])
    if writeData: 
      add_var(ncfile, name, dims=dims, values=var.getArray(unmask=True), atts=coerceAtts(var.atts), dtype=var.dtype, zlib=zlib, fillValue=var.fillValue)
    else: 
      add_var(ncfile, name, dims=dims, values=None, atts=coerceAtts(var.atts), dtype=var.dtype, zlib=zlib, fillValue=var.fillValue)
  # close file or return file handle
  if close: ncfile.close()
  else: return ncfile
  