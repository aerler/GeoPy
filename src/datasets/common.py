'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

# external imports
from importlib import import_module
from warnings import warn
import numpy as np
import pickle
import os
# internal imports
from geodata.misc import AxisError, DatasetError, DateError
from geodata.base import Dataset, Variable, Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, GridDefinition, loadPickledGridDef, griddef_pickle


# days per month
days_per_month = np.array([31,28.2425,31,30,31,30,31,31,30,31,30,31], dtype='float32') # 97 leap days every 400 years
# N.B.: the Gregorian calendar repeats every 400 years
days_per_month_365 = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype='int16') # no leap day
# human-readable names
name_of_month = ['January  ', 'February ', 'March    ', 'April    ', 'May      ', 'June     ', #
                 'July     ', 'August   ', 'September', 'October  ', 'November ', 'December ']


# attributes for variables in standardized climatologies 
# variable attributes and name
default_varatts = dict(pmsl     = dict(name='pmsl', units='Pa'), # sea-level pressure
                       ps       = dict(name='ps', units='Pa'), # surface pressure
                       Ts       = dict(name='Ts', units='K'), # average skin temperature
                       T2       = dict(name='T2', units='K'), # 2m average temperature
                       T        = dict(name='T', units='K'), # average temperature
                       Tmin     = dict(name='Tmin', units='K'), # 2m minimum temperature
                       Tmax     = dict(name='Tmax', units='K'), # 2m maximum temperature
                       Q2       = dict(name='Q2', units='Pa'), # 2m water vapor pressure
                       pet      = dict(name='pet', units='kg/m^2/s'), # potential evapo-transpiration
                       evap     = dict(name='evap', units='kg/m^2/s'), # actual evapo-transpiration
                       precip   = dict(name='precip', units='kg/m^2/s'), # total precipitation                       
                       solprec  = dict(name='solprec', units='kg/m^2/s'), # solid precipitation
                       liqprec  = dict(name='liqprec', units='kg/m^2/s'), # liquid precipitation
                       pwtr     = dict(name='pwtr', units='kg/m^2'), # total precipitable water (kg/m^2)
                       snow     = dict(name='snow', units='kg/m^2'), # snow water equivalent
                       snowh    = dict(name='snowh', units='m'), # snow depth
                       sfroff   = dict(name='sfroff', units='kg/m^2/s'), # surface run-off                      
                       ugroff   = dict(name='ugroff', units='kg/m^2/s'), # sub-surface/underground run-off      
                       runoff   = dict(name='runoff', units='kg/m^2/s'), # total surface and sub-surface run-off
                       stations = dict(name='stations', units='#'), # number of gauges for observation
                       zs       = dict(name='zs', units='m'), # surface elevation
                       landmask = dict(name='landmask', units=''), # land mask
                       lon2D    = dict(name='lon2D', units='deg E'), # geographic longitude field
                       lat2D    = dict(name='lat2D', units='deg N'), # geographic latitude field
                       # axes (don't have their own file; listed in axes)
                       time     = dict(name='time', units='month'), # time coordinate for climatology
                       lon      = dict(name='lon', units='deg E'), # geographic longitude field
                       lat      = dict(name='lat', units='deg N'), # geographic latitude field
                       x        = dict(name='x', units='m'), # projected west-east coordinate
                       y        = dict(name='y', units='m')) # projected south-north coordinate
default_varatts['p-et'] = dict(name='p-et', units='kg/m^2/s') # net precipitation; only legal as a string                                


# data root folder
import socket
hostname = socket.gethostname()
if hostname=='komputer':
  data_root = '/data/'  
#  root = '/media/tmp/' # RAM disk for development
elif hostname=='cryo':
  data_root = '/scratch/marcdo/Data/'
elif hostname=='erlkoenig':
  data_root = '/media/PortableHD/DATA/'
else:
  raise NotImplementedError, "No 'data_root' folder set!"
# standard folder for grids and shapefiles  
grid_folder = data_root + '/grids/' # folder for pickled grids
shape_folder = data_root + '/shapes/' # folder for pickled grids
 

# convenience function to extract landmask variable from another masked variable
def addLandMask(dataset, varname='precip', maskname='landmask', atts=None):
  ''' Add a landmask variable with meta data from a masked variable to a dataset. '''
  # check
  if not isinstance(dataset,Dataset): raise TypeError
  if dataset.hasVariable(maskname): 
    raise DatasetError, "The Dataset '%s' already has a field called '%s'."%(dataset.name,maskname)
  # attributes and meta data
  if atts is None:
    atts = default_varatts[maskname].copy()
    atts['long_name'] = 'Geographic Mask for Climatology Fields' 
    atts['description'] = 'data are valid where this mask is zero'  
  # axes and data
  var = dataset.variables[varname]
  axes = var.axes[-2:] # last two axes (i.e. map axes)
  data = var.getMask().__getitem__((0,)*(var.ndim-2)+(slice(None),)*2)
  if 'gdal' in dataset.__dict__ and dataset.gdal:
    if dataset.xlon not in axes or dataset.ylat not in axes: raise AxisError
  if not all([ax.name in ('x','y','lon','lat') for ax in axes]): raise AxisError
  # create variable and add to dataset
  if isinstance(dataset, DatasetNetCDF) and 'w' in dataset.mode: 
    dataset.addVariable(Variable(axes=axes, name=maskname, data=data, atts=atts), asNC=True)
  else: dataset.addVariable(Variable(axes=axes, name=maskname, data=data, atts=atts))
  # return mask variable
  return dataset.variables[maskname]


# annotate dataset with names and length of months (for climatology mostly)
def addLengthAndNamesOfMonth(dataset, noleap=False, length=None, names=None):
  ''' Function to add the names and length of month to a NetCDF dataset. '''
  if not isinstance(dataset,Dataset): raise TypeError
  # attributes
  lenatts = dict(name='length_of_month', units='days',long_name='Length of Month')
  stratts = dict(name='name_of_month', units='', long_name='Name of the Month')
  # data
  if length is None: # leap year or no leap year
    if noleap: length = days_per_month_365
    else: length = days_per_month
  if names is None: names = name_of_month
  # create variables
  if isinstance(dataset, DatasetNetCDF) and 'w' in dataset.mode: 
    #dataset += VarNC(dataset.dataset, axes=(dataset.time,), data=length, atts=lenatts)
    dataset.addVariable(Variable(axes=(dataset.time,), data=length, atts=lenatts), asNC=True)
    dataset.axisAnnotation(stratts['name'], names, 'time', atts=stratts)
  else:
    # N.B.: char/string arrays are currently not supported as Variables
    dataset.addVariable(Variable(axes=(dataset.time,), data=length, atts=lenatts))
  # return length variable
  return dataset.variables[lenatts['name']]


# helper function to convert monthly precip amount into precip rate
def convertPrecip(precip):
  ''' convert monthly precip amount to SI units (mm/s) '''
  warn("Use of method 'convertPrecip' is depricated; use the on-the-fly transformPrecip function instead")
  if precip.units == 'kg/m^2/month' or precip.units == 'mm/month':
    precip /= (days_per_month.reshape((12,1,1)) * 86400.) # convert in-place
    precip.units = 'kg/m^2/s'
  return precip

# transform function to convert monthly precip amount into precip rate on-the-fly
def transformPrecip(data, var=None, slc=None):
  ''' convert monthly precip amount to SI units (mm/s) '''
  if not isinstance(var,VarNC): raise TypeError
  if var.units == 'kg/m^2/month' or var.units == 'mm/month':
    if not ( data.ndim == 3 and data.shape[0]%12 == 0 ): raise NotImplementedError
    tax = var.axisIndex('time'); te = len(var.time)
    if not ( tax == 0 and te%12 == 0 ): raise NotImplementedError  
    data /= np.repeat(days_per_month.reshape((12,1,1)) * 86400., te/12, axis=tax) # convert in-place
    var.units = 'kg/m^2/s'
  return data      
      

# convenience function to invert variable name mappings
def translateVarNames(varlist, varatts):
  ''' Simple function to replace names in a variable list with their original names as inferred from the 
      attributes dictionary. Note that this requires the dictionary to have the field 'name'. '''
  if not isinstance(varlist,(list,tuple,set)) or not isinstance(varatts,dict): raise TypeError 
  varlist = list(varlist) # make copy, since operation is in-place 
  # cycle over names in variable attributes (i.e. final names, not original names)  
  for key,atts in varatts.iteritems():
    if 'name' in atts and atts['name'] in varlist: 
      varlist[varlist.index(atts['name'])] = key # original name is used as key in the attributes dict
  # return varlist with final names replaced by original names
  return varlist


# universal function to generate file names for climatologies
def getFileName(name=None, resolution=None, period=None, filetype='climatology', grid=None, filepattern=None):
  ''' A function to generate a standardized filename for climatology and time-series files, based on grid type and period.  '''
  if name is None: name = ''
  # grid (this is a *non-native grid*)
  if grid is None or grid == name: gridstr = ''
  else: gridstr = '_{0:s}'.format(grid.lower()) # only use lower case for filenames
  # resolution is the native resolution (behind dataset name, prepended to the grid 
  if resolution: gridstr = '_{0:s}{1:s}'.format(resolution,gridstr)
  # period
  if filetype == 'time-series':
    # assemble filename
    if filepattern is None: filepattern = name.lower() + '{0:s}_monthly.nc' 
    filename = filepattern.format(gridstr)
  elif filetype == 'climatology':
    if isinstance(period,(tuple,list)): pass
    elif isinstance(period,basestring): pass
    elif period is None: pass
    elif isinstance(period,(int,np.integer)):
      period = (1979, 1979+period)
    else: raise DateError   
    if period is None or period == '': periodstr = ''
    elif isinstance(period,basestring): periodstr = '_{0:s}'.format(period)
    else: periodstr = '_{0:4d}-{1:4d}'.format(*period)  
    # assemble filename
    if filepattern is None: filepattern = name.lower() + '{0:s}_clim{1:s}.nc' 
    filename = filepattern.format(gridstr,periodstr)
  else: raise NotImplementedError, "Unrecognized filetype: '{:s}'".format(filetype)
  # return final name
  assert filename == filename.lower(), "By convention, climatology files only have lower-case names!"
  return filename
  
  
# universal load function that will be imported by datasets
def loadClim(name, folder, resolution=None, period=None, grid=None, varlist=None, varatts=None, filepattern=None, 
             filelist=None, projection=None, geotransform=None, axes=None, lautoregrid=False):
  ''' A function to load standardized climatology datasets. '''
  # prepare input
  # varlist (varlist = None means all variables)
  if varatts is None: varatts = default_varatts.copy()
  if varlist is not None: varlist = translateVarNames(varlist, varatts)
  # filelist
  if filelist is None: 
    filename = getFileName(name=name, resolution=resolution, period=period, grid=grid, filepattern=filepattern)
    # check existance
    filepath = '{:s}/{:s}'.format(folder,filename)
    if not os.path.exists(filepath):
      nativename = getFileName(name=name, resolution=resolution, period=period, grid=None, filepattern=filepattern)
      nativepath = '{:s}/{:s}'.format(folder,nativename)
      if os.path.exists(nativepath):
        if lautoregrid: 
          from processing.regrid import performRegridding # causes circular reference if imported earlier
          griddef = loadPickledGridDef(grid=grid, res=resolution, folder=grid_folder)
          dataargs = dict(period=period, resolution=resolution)
          performRegridding(name, griddef, dataargs) # default kwargs
        else: raise IOError, "The dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.".format(filename,grid) 
      else: raise IOError, "The dataset file '{:s}' does not exits!".format(filename)
  # load dataset
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=[filename], varlist=varlist, varatts=varatts, 
                          axes=axes, multifile=False, ncformat='NETCDF4')
  # figure out grid
  if grid is None or grid == name:
    dataset = addGDALtoDataset(dataset, projection=projection, geotransform=geotransform, gridfolder=grid_folder)
  elif isinstance(grid,basestring): # load from pickle file
#     griddef = loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder)
    # add GDAL functionality to dataset 
    dataset = addGDALtoDataset(dataset, griddef=grid, gridfolder=grid_folder)
  else: raise TypeError
  # N.B.: projection should be auto-detected, if geographic (lat/lon)
  return dataset

def checkItemList(itemlist, length, dtype, default=NotImplemented, iterable=False, trim=True):
  ''' return a list based on item and check type '''
  # N.B.: default=None is not possible, because None may be a valid default...
  if itemlist is None: itemlist = []
  if iterable:
    # if elements are lists or tuples etc.
    if not isinstance(itemlist,(list,tuple,set)): raise TypeError, str(itemlist)
    if len(itemlist) == 0: 
      if isinstance(default,(list,tuple,set)): itemlist = [default]*length
      else: raise TypeError, str(itemlist) # here the default has tp be a list of items
    else:
      if isinstance(itemlist[0],(list,tuple,set)):     
        if trim:
          if len(itemlist) > length: del itemlist[length:]
          elif len(itemlist) < length: itemlist += [default]*(length-len(itemlist))
        else:
          if len(itemlist) != length: raise TypeError, str(itemlist)    
        if dtype is not None:
          if not all([isinstance(item,dtype) for item in itemlist if item != default]):
            raise TypeError, str(itemlist) # only checks the non-default values
      else:
        if not all([isinstance(item,dtype) for item in itemlist if item is not None]): 
          raise TypeError, str(itemlist)
        itemlist = [itemlist]*length
  else:
    if isinstance(itemlist,(list,tuple,set)):     
      itemlist = list(itemlist)
      if default is NotImplemented: 
        if len(itemlist)>0: default = itemlist[0] # use first item
        else: default = None   
      if trim:
        if len(itemlist) > length: del itemlist[length:]
        elif len(itemlist) < length:
          itemlist += [default]*(length-len(itemlist))
      else:
        if len(itemlist) != length: raise TypeError, str(itemlist)    
      if dtype is not None:
        if not all([isinstance(item,dtype) for item in itemlist if item != default]):
          raise TypeError, str(item) # only checks the non-default values
    else:
      if not isinstance(itemlist,dtype): raise TypeError, str(itemlist)
      itemlist = [itemlist]*length
  return itemlist

# helper function for loadDatasets (see below)
def loadDataset(exp, prd, dom, grd, res, filetypes=None, varlist=None, lbackground=True, 
                lWRFnative=True, lautoregrid=False):
  ''' A function that loads a dataset, based on specified parameters '''
  from datasets.WRF import loadWRF
  from projects.WRF_experiments import WRF_exps, Exp
  from datasets.CESM import CESM_exps, loadCESM 
  from datasets.GPCC import loadGPCC
  from datasets.CRU import loadCRU
  from datasets.PCIC import loadPCIC
  from datasets.PRISM import loadPRISM
  from datasets.CFSR import loadCFSR
  from datasets.NARR import loadNARR
  from datasets.Unity import loadUnity
  if not isinstance(exp,(basestring,Exp)): raise TypeError
  if exp[0].isupper():
    if exp == 'Unity': 
      ext = loadUnity(resolution=res, period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'Merged Observations'        
    elif exp == 'GPCC': 
      ext = loadGPCC(resolution=res, period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'GPCC Observations'
    elif exp == 'CRU': 
      ext = loadCRU(period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'CRU Observations' 
    elif exp == 'PCIC': # PCIC with some background field
      if lbackground:
        if (len(varlist) == 1 and 'precip' in varlist) or (len(varlist) == 3 and 
            'precip' in varlist and 'lon2D' in varlist and 'lat2D' in varlist): 
          ext = (loadGPCC(grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPCIC(grid=grd, varlist=varlist, lautoregrid=lautoregrid),)
          axt = 'PCIC PRISM (and GPCC)'
        else: 
          ext = (loadCRU(period='1971-2001', grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPCIC(grid=grd, varlist=varlist, lautoregrid=lautoregrid)) 
          axt = 'PCIC PRISM (and CRU)'
      else:
        ext = loadPCIC(grid=grd, varlist=varlist, lautoregrid=lautoregrid); axt = 'PCIC PRISM'
    elif exp == 'PRISM': # PRISM with some background field
      if lbackground:
        if (len(varlist) == 1 and 'precip' in varlist) or (len(varlist) == 3 and 
            'precip' in varlist and 'lon2D' in varlist and 'lat2D' in varlist): 
          ext = (loadGPCC(grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid),)
          axt = 'PRISM (and GPCC)'
        else: 
          ext = (loadCRU(period='1979-2009', grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid)) 
          axt = 'PRISM (and CRU)'
      else:
        ext = loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid); axt = 'PRISM'
    elif exp == 'CFSR': 
      ext = loadCFSR(period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'CFSR Reanalysis' 
    elif exp == 'NARR': 
      ext = loadNARR(period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'NARR Reanalysis'
    else: # all other uppercase names are CESM runs
      exp = CESM_exps[exp]
      #print exp.name, exp.title
      ext = loadCESM(experiment=exp, period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = exp.title
  else: 
    # WRF runs are all in lower case
    exp = WRF_exps[exp]      
    parent = None
    if isinstance(dom,(list,tuple)):
      #if not lbackground: raise ValueError, 'Can only plot one domain, if lbackground=False'
      if 0 == dom[0]:
        dom = dom[1:]
        parent, tmp = loadDataset(exp.parent, prd, dom, grd, res, varlist=varlist, lbackground=False, lautoregrid=lautoregrid); del tmp    
    #if 'xtrm' in WRFfiletypes: 
    varatts = None #dict(T2=dict(name='Ts')) 
    if lWRFnative: grd = None
    ext = loadWRF(experiment=exp, period=prd, grid=grd, domains=dom, filetypes=filetypes, 
                  varlist=varlist, varatts=varatts, lautoregrid=lautoregrid)
    if parent is not None: ext = (parent,) + tuple(ext)
    axt = exp.title # defaults to name...
  # return values
  return ext, axt    
    
# function to load a list of datasets/experiments based on names and other common parameters
def loadDatasets(explist, n=None, varlist=None, titles=None, periods=None, domains=None, grids=None,
                 resolutions='025', filetypes=None, lbackground=True, lWRFnative=True, ltuple=True, 
                 lautoregrid=False):
  ''' function to load a list of datasets/experiments based on names and other common parameters '''
  # for load function (below)
  from projects.WRF_experiments import Exp
  if lbackground and not ltuple: raise ValueError
  # check and expand lists
  if n is None: n = len(explist)
  elif not isinstance(n, (int,np.integer)): raise TypeError
  explist = checkItemList(explist, n, (basestring,Exp,tuple))
  titles = checkItemList(titles, n, basestring, default=None)
  periods  = checkItemList(periods, n, (basestring,int,np.integer), default=None, iterable=False)
  if isinstance(domains,tuple): ltpl = ltuple
  else: ltpl = False # otherwise this causes problems with expanding this  
  domains  = checkItemList(domains, n, (int,np.integer,tuple), default=None, iterable=ltpl) # to return a tuple, give a tuple of domains
  grids  = checkItemList(grids, n, basestring, default=None)
  resolutions  = checkItemList(resolutions, n, basestring, default=None)  
  # resolve experiment list
  dslist = []; axtitles = []
  for exp,tit,prd,dom,grd,res in zip(explist,titles,periods,domains,grids,resolutions): 
    if isinstance(exp,tuple):
      if lbackground: raise ValueError, 'Adding Background is not supported in combination with experiment tuples!'
      if not isinstance(dom,(list,tuple)): dom =(dom,)*len(exp)
      if len(dom) != len(exp): raise ValueError, 'Only one domain is is not supported for each experiment!'          
      ext = []; axt = []        
      for ex,dm in zip(exp,dom):
        et, at = loadDataset(ex, prd, dm, grd, res, filetypes=filetypes, varlist=varlist, 
                           lbackground=False, lWRFnative=lWRFnative, lautoregrid=lautoregrid)
        #if isinstance(et,(list,tuple)): ext += list(et); else: 
        ext.append(et)
        #if isinstance(at,(list,tuple)): axt += list(at); else: 
        axt.append(at)
      ext = tuple(ext); axt = tuple(axt)
    else:
      ext, axt = loadDataset(exp, prd, dom, grd, res, filetypes=filetypes, varlist=varlist, 
                           lbackground=lbackground, lWRFnative=lWRFnative, lautoregrid=lautoregrid)
    dslist.append(ext) 
    if tit is not None: axtitles.append(tit)
    else: axtitles.append(axt)  
  # count experiment tuples (layers per panel)
  if ltuple:
    nlist = [] # list of length for each element (tuple)
    for n in xrange(len(dslist)):
      if not isinstance(dslist[n],(tuple,list)): # should not be necessary
        dslist[n] = (dslist[n],)
      elif isinstance(dslist[n],list): # should not be necessary
        dslist[n] = tuple(dslist[n])
      nlist.append(len(dslist[n])) # layer counter for each panel  
  # return list with datasets and plot titles
  if ltuple:
    return dslist, axtitles, nlist
  else:
    return dslist, axtitles
  

# function to return grid definitions for some common grids
def getCommonGrid(grid, res=None):
  ''' return grid definitions of some commonly used grids '''
  # look in known datasets first
  try :
    dataset = import_module(grid)
    if res is None:
      griddef = dataset.default_grid
    else:
      griddef = dataset.grid_def[res]
  except ImportError:
    lgrid = True
    # select grid
    if grid == 'ARB_small':   slon, slat, elon, elat = -160.25, 32.75, -90.25, 72.75
    elif grid == 'ARB_large': slon, slat, elon, elat = -179.75, 3.75, -69.75, 83.75
    else: lgrid = False
    # select resolution:
    lres = True
    if res is None: res = '025' # default    
    if res == '025':   dlon = dlat = 0.25 # resolution
    elif res == '05':  dlon = dlat = 0.5
    elif res == '10':  dlon = dlat = 1.0
    elif res == '25':  dlon = dlat = 2.5
    else: lres = False
    if lgrid and lres:    
      assert (elon-slon) % dlon == 0 
      lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
      assert (elat-slat) % dlat == 0
      lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
      # add new geographic coordinate axes for projected map
      xlon = Axis(coord=lon, atts=dict(name='lon', long_name='longitude', units='deg E'))
      ylat = Axis(coord=lat, atts=dict(name='lat', long_name='latitude', units='deg N'))
      gridstr = '{0:s}_{1:s}'.format(grid,res) if res is not None else grid
      griddef = GridDefinition(name=gridstr, projection=None, xlon=xlon, ylat=ylat) # projection=None >> lat/lon
    else: 
      griddef = None
  # return grid definition object
  return griddef

## (ab)use main execution for quick test
if __name__ == '__main__':
    
  
#   mode = 'test_climatology'
#   mode = 'test_timeseries'
  mode = 'pickle_grid'
  grids = dict( ARB_small=['025','05','10','25'],
                ARB_large=['025','05','10','25'],
                CFSR=['031','05'],
                GPCC=['025','05','10','25'],
                CRU=[None],NARR=[None],PRISM=[None],PCIC=[None])
  #grids = dict( CFSR=['031','05'],)
    
  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,reses in grids.items():
      
      if reses is None: reses = [None] # default grid
      
      for res in reses: 
      
        print('')        
        if res is None:
          gridstr = grid
          print('   ***   Pickling Grid Definition for {0:s}   ***   '.format(grid))
        else:
          gridstr = '{0:s}_{1:s}'.format(grid,res)  
          print('   ***   Pickling Grid Definition for {0:s} Resolution {1:s}   ***   '.format(grid,res))
        print('')
        
        # load GridDefinition      
        griddef = getCommonGrid(grid,res)         
        
        if griddef is None:
          print('GridDefinition object for {0:s} not found!'.format(gridstr))         
        else:
          # save pickle
          filename = '{0:s}/{1:s}'.format(grid_folder,griddef_pickle.format(gridstr))
          filehandle = open(filename, 'w')
          pickle.dump(griddef, filehandle)
          filehandle.close()
          
          print('   Saving Pickle to \'{0:s}\''.format(filename))
          print('')
          
          # load pickle to make sure it is right
          del griddef
          griddef = loadPickledGridDef(grid, res=res, folder=grid_folder)
          print(griddef)
        print('')
