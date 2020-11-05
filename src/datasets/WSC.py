'''
Created on 2014-02-12

A module to load station data from the Water Survey of Canada and associate the data with river basins;
the data is stored in human-readable text files and tables. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os, functools
from copy import deepcopy
from collections import OrderedDict
# internal imports
from datasets.common import selectElements, data_root
from geodata.netcdf import DatasetNetCDF
from geodata.misc import FileError, isNumber, ArgumentError
from utils import nanfunctions as nf
from geodata.gdal import Shape, ShapeSet
from geodata.base import Dataset, Variable, Axis
# from geodata.utils import DatasetError

## WSC (Water Survey Canada) Meta-data

dataset_name = 'WSC'
shape_root = '{:s}/shapes/'.format(data_root) # the shapefile root folder
if not os.path.exists(shape_root):
    raise IOError("Folder for shape files not found!\n('{:s}')".format(shape_root))
root_folder = shape_root

# variable attributes and name
variable_attributes_kgs = dict(runoff = dict(name='runoff', units='kg/m^2/s', atts=dict(long_name='Average Runoff Rate')), # average flow rate
                               roff_std = dict(name='roff_std', units='kg/m^2/s', atts=dict(long_name='Runoff Rate Variability')), # flow rate variability
                               roff_sem = dict(name='roff_sem', units='kg/m^2/s', atts=dict(long_name='Runoff Rate Error')), # flow rate error
                               roff_max = dict(name='roff_max', units='kg/m^2/s', atts=dict(long_name='Maximum Runoff Rate')), # maximum flow rate
                               roff_min = dict(name='roff_min', units='kg/m^2/s', atts=dict(long_name='Minimum Runoff Rate')), # minimum flow rate
                               discharge = dict(name='discharge', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Average Flow Rate')), # average flow rate
                               discstd = dict(name='StdDisc', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Flow Rate Variability')), # flow rate variability
                               discsem = dict(name='SEMDisc', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Flow Rate Error')), # flow rate error
                               discmax = dict(name='MaxDisc', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Maximum Flow Rate')), # maximum flow rate
                               discmin = dict(name='MinDisc', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Minimum Flow Rate')), # minimum flow rate
                               level = dict(name='level', units='m', atts=dict(long_name='Water Level'))) # water level
kgs_to_mms = {'kg/m^2/s':'mm/s', 'kg/s':'m^3/s'}
variable_attributes_mms = deepcopy(variable_attributes_kgs)
for varatts in list(variable_attributes_mms.values()):
    varatts['units'] = kgs_to_mms.get(varatts['units'],varatts['units'])
# list of variables to load
variable_list = list(variable_attributes_kgs.keys()) # also includes coordinate fields    
# alternate WSC runoff variable names (for use with aggregation mode)
agg_varatts_kgs = dict()
for varname,varatts in list(variable_attributes_kgs.items()):
  tmpatts = varatts.copy() 
  if tmpatts['units'] == 'kg/m^2/s': tmpatts['name'] = 'runoff'  # always name 'runoff'
  elif tmpatts['units'] == 'kg/s': tmpatts['name'] = 'discharge'  # always name 'discharge'
  agg_varatts_kgs[varname] = tmpatts    
agg_varatts_mms = deepcopy(agg_varatts_kgs)
for varatts in list(agg_varatts_mms.values()):
    varatts['units'] = kgs_to_mms.get(varatts['units'],varatts['units'])


# change scalefactor and update PlotAtts of Variables
def updateScalefactor(dataset, varlist, scalefactor=None):
    ''' a function to update the scalefactor for plotting in a set of variables, usually used for river discharge '''
    # determine variable list
    if isinstance(varlist,dict): vardict = varlist
    elif isinstance(varlist,(list,tuple)): vardict = {varname:scalefactor for varname in varlist}
    else: raise TypeError(varlist)
    # scalefactor string
    scalestr = lambda s: None if s == 1 else '10^{:d}'.format(-1*int(np.log10(s)))
    # loop over variables
    for varname,scalefactor in list(vardict.items()):
        if varname in dataset and scalefactor is not None:
            var = dataset[varname]
            oldstr = scalestr(var.plot.scalefactor)
            newstr = scalestr(scalefactor)
            # replace old PlotAtts with new ones
            if oldstr and newstr: 
              if oldstr == newstr: units = None
              else: units = var.plot.units.replace(oldstr,newstr)
            elif oldstr: units = var.plot.units.replace(oldstr,'').strip()
            elif newstr is not None: units = r'${}$ {}'.format(newstr,var.plot.units)
            else: units = None
            if units is not None:
                var.plot = var.plot.copy(units=units, scalefactor=scalefactor)
    # return dataset, although not really necessary
    return dataset


# custom exception for missing gage station data
class GageStationError(FileError):
  ''' Exception indicating that gage station data is missing '''
  pass


# shape class for lakes
class Province(Shape):
  ''' a Shape class for provinces '''
  def __init__(self, name=None, long_name=None, shapefile=None, folder=None, load=False, ldebug=False,
               data_source=None, shapetype=None):
    ''' some additional information '''
    if shapetype is None: shapetype = 'PRV'
    if folder is None: folder = '{:s}/Provinces/{:s}/'.format(shape_root,long_name)
    super(Province,self).__init__(name=name, long_name=long_name, shapefile=shapefile, folder=folder, 
                               load=load, ldebug=ldebug, data_source=data_source, shapetype=shapetype)    
    
# a container class for lake meta data
class Nation(ShapeSet,Province): 
  ''' a container class for sets of provinces '''
  _ShapeClass = Province # the class that is used to initialize the shape collection
  
  def __init__(self, name=None, long_name=None, provinces=None, data_source=None, 
               folder=None, shapetype=None):
    ''' some common operations and inferences '''
    # call parent constructor 
    if shapetype is None: shapetype = 'NAT'
    if folder is None: folder = '{:s}/Provinces/{:s}/'.format(shape_root,long_name)
    super(Nation,self).__init__(name=name, long_name=long_name, shapefiles=provinces, 
                                 data_source=data_source, folder=folder, shapetype=shapetype) # ShapeSet arguments
                                  # N.B.: addition arguments for Basin constructor
    # add lake specific stuff
    self.provinces = self.shapes # alias


# shape class for lakes
class Lake(Shape):
  ''' a Shape class for lakes with associated gage station information '''
  def __init__(self, name=None, long_name=None, shapefile=None, folder=None, load=False, ldebug=False,
               data_source=None, shapetype=None):
    ''' save meta information; should be initialized from a BasinInfo instance '''
    if shapetype is None: shapetype = 'LKE'
    if folder is None: folder = '{:s}/Lakes/{:s}/'.format(shape_root,long_name)
    super(Lake,self).__init__(name=name, long_name=long_name, shapefile=shapefile, folder=folder, 
                              load=load, ldebug=ldebug, data_source=data_source, shapetype=shapetype)    
    
# a container class for lake meta data
class LakeSet(ShapeSet,Lake): 
  ''' a container class for sets of lakes with associated lakes '''
  _ShapeClass = Lake # the class that is used to initialize the shape collection
  
  def __init__(self, name=None, long_name=None, lakes=None, data_source=None, folder=None, shapetype=None):
    ''' some common operations and inferences '''
    # call parent constructor 
    if shapetype is None: shapetype = 'LKE'
    if folder is None: folder = '{:s}/Lakes/{:s}/'.format(shape_root,long_name)
    #shapefiles = lakes.keys() if isinstance(lakes, dict) else lakes
    super(LakeSet,self).__init__(name=name, long_name=long_name, shapefiles=lakes, 
                                 data_source=data_source, folder=folder, shapetype=shapetype,) # ShapeSet arguments
                                  # N.B.: addition arguments for Basin constructor
    # add lake specific stuff
    self.lakes = self.shapes # alias

 
# shape class for basins
class Basin(Shape):
  ''' a Shape class for river basins with associated gauge station information '''
  def __init__(self, name=None, long_name=None, shapefile=None, folder=None, load=False, ldebug=False,
               subbasins=None, rivers=None, stations=None, data_source=None, shapetype=None):
    ''' save meta information; should be initialized from a BasinInfo instance '''
    if shapetype is None: shapetype = 'BSN'
    if folder is None: folder = '{:s}/Basins/{:s}/'.format(shape_root,long_name)
    super(Basin,self).__init__(name=name, long_name=long_name, shapefile=shapefile, folder=folder, 
                               load=load, ldebug=ldebug, data_source=data_source, shapetype=shapetype)
    # figure out if we are the outline/main basin
    name = self.name
    if subbasins:
        if 'Whole{:s}'.format(self.name) in subbasins: name = 'Whole{:s}'.format(self.name)
        assert name in subbasins, (name,subbasins)
    # add gauge station from dict (based on name)
    if isinstance(subbasins,dict) and name in subbasins and subbasins[name]:
      maingage = subbasins[name]
      if isinstance(station, (list,tuple)): maingage = '{}_{}'.format(*maingage)        
    elif rivers and stations and name[:5] == 'Whole':
      maingage = '{}_{}'.format(rivers[0],stations[rivers[0]][0]) # first station of first river (from outflow)
    else: maingage = None # no gage station defined
    # initialize gage station
    self.maingage = None if maingage is None else GageStation(basin=self.name, name=maingage, folder=folder) # just name, for now
    
  def getMainGage(self, varlist=None, varatts=None, aggregation=None, mode='timeseries', 
                  filetype='monthly', lkgs=True):
    ''' return a dataset with data from the main gaging station (default: timeseries) '''
    if self.maingage is not None:
      station = loadGageStation(basin=self, station=self.maingage, varlist=varlist, varatts=varatts, 
                                aggregation=aggregation, mode=mode, filetype=filetype, lkgs=lkgs)
    else: station = None 
    return station

# a container class for basin meta data
class BasinSet(ShapeSet,Basin): 
  ''' a container class for basins with associated subbasins '''
  _ShapeClass = Basin # the class that is used to initialize the shape collection
  
  def __init__(self, name=None, long_name=None, rivers=None, stations=None, subbasins=None, 
               data_source=None, folder=None, shapetype=None):
    ''' some common operations and inferences '''
    # call parent constructor 
    if shapetype is None: shapetype = 'BSN'
    if folder is None: folder = '{:s}/Basins/{:s}/'.format(shape_root,long_name)
    #shapefiles = subbasins.keys() if isinstance(subbasins, dict) else subbasins
    super(BasinSet,self).__init__(name=name, long_name=long_name, shapefiles=subbasins, 
                                  data_source=data_source, folder=folder, shapetype=shapetype, # ShapeSet arguments
                                  rivers=rivers, stations=stations, subbasins=subbasins) 
                                  # N.B.: addition arguments for Basin constructor
    # add basin specific stuff
    self.subbasins = self.shapes # alias
    # self.maingage = stations[rivers[0]][0] if stations else None # should already be set by Basin init
    assert self.maingage is not None or not stations, stations # internal check
    self.rivers = rivers
    # add list of gage stations
    self.river_stations = OrderedDict(); self.stations = OrderedDict()
    for river in rivers:
      if  river in stations:
          station_list = stations[river]
          self.river_stations = [GageStation(name=station, river=river, folder=folder) 
                                 for station in station_list]
          for station in self.river_stations: self.stations[station.name] = station     
        

# shape class for catchments associated with a gauge
class Catchment(Shape):
  ''' a Shape class for the drainage area (catchment) associated with a gauge station '''
  gauge = None
  metadata = None
  
  def __init__(self, name=None, long_name=None, shapefile=None, folder=None, load=False, ldebug=False,
               station=None, data_source='WSC', shapetype='CAT'):
    ''' save meta information; should be initialized from a BasinInfo instance '''
    if folder is None: 
        folder = '{:s}/WSC_Gauge_Catchments/{:s}/{:s}/'.format(shape_root,name[:2],name)
    if shapefile is None:
        shapefile = name+'_1' if len(name) < 9 else name
    super(Catchment,self).__init__(name=name, long_name=long_name, shapefile=shapefile, folder=folder, 
                                   load=load, ldebug=ldebug, data_source=data_source, shapetype=shapetype)
    # most attributes are set in Shape class
    self.station = station or self.name # name of actual gauge station
    self.gauge = None
    # add gage station (based on name)
    
  def getGaugeSation(self, lcheck=True):
    ''' load and return the GageStation object for this catchment '''
    # load gauge object
    gauge = GageStation(name=self.station, folder=self.folder, meta_file='Metadata.csv', 
                        monthly_file='Monthly_flow.csv', lcheck=lcheck)
    # assign variables
    self.gauge = gauge
    self.metadata = gauge.getMetaData(lcheck=lcheck)
    return gauge
    
  def getGaugeDataset(self, varlist=None, varatts=None, aggregation=None, mode='timeseries', 
                      filetype='monthly', lkgs=True):
    ''' return a dataset with data from the main gaging station (default: timeseries) '''
    if self.gauge is None:
        self.getGaugeSation(lcheck=True)
    dataset = loadGageStation(station=self.gauge, varlist=varlist, varatts=varatts, 
                              aggregation=aggregation, mode=mode, filetype=filetype, lkgs=lkgs)
    return dataset
      
      
# a class to hold meta data of gaging stations
class GageStation(object):
  ''' a class that provides access to station meta data and gage data '''
  meta_ext = '_Metadata.csv'
  meta_file = None
  monthly_ext = '_Monthly.csv'
  monthly_file = None
  
  def __init__(self, name=None, basin=None, river=None, folder=None, meta_file=None, monthly_file=None, 
               lcheck=False):
    ''' initialize gage station based on various input data '''
    if name is None: raise ArgumentError()
    if folder is None: folder = '{:s}/Basins/{:s}/'.format(root_folder,basin)
    if not os.path.isdir(folder): IOError(folder)
    if river is not None and river+'_' not in name: name = '{:s}_{:s}'.format(river,name)
    self.folder = folder # usually basin folder
    self.name = name # or prefix...
    self.basin_name = basin # has to be a long_name in order to construct the folder
    if meta_file is None: meta_file = name + self.meta_ext
    self.meta_file = '{:s}/{:s}'.format(folder,meta_file)  
    if not os.path.isfile(self.meta_file): 
      if lcheck: raise IOError(self.meta_file)
      else: self.meta_file = None # clear if not available
    if monthly_file is None: monthly_file = name + self.monthly_ext
    self.monthly_file = '{:s}/{:s}'.format(folder,monthly_file)
    if not os.path.isfile(self.monthly_file): 
      if lcheck: raise IOError(self.monthly_file)
      else: self.monthly_file = None # clear if not available
    
  def getMetaData(self, lcheck=False):
    ''' parse meta data file and save and return as dictionary '''
    if self.meta_file:
      # parse file and load data into a dictionary
      with open(self.meta_file, mode='r') as filehandle:
        lines = [line.encode('ascii',errors='ignore').decode() for line in filehandle.readlines()]
        # N.B.: .encode('ascii',errors='ignore').decode() is to remove all non-ascii characters
        assert len(lines) == 2, lines
        keys = lines[0].split(',')
        values = lines[1].split(',')
      assert len(keys) == len(values)
      # add some additional attributes
      metadata = {key:value for key,value in zip(keys,values)}
      metadata['long_name'] = metadata['Station Name'].replace('"', '').title() # clean up name
      metadata['ID'] = metadata['Station Number']
      metadata['shape_name'] = self.basin_name
      metadata['shp_area'] = float(metadata['Drainage Area']) * 1e6 # km^2 == 1e6 m^2
    elif lcheck:
      raise GageStationError("No metadata available for gage station '{}'.".format(self.name))
    else:
      metadata = None
    # add to station
    self.atts = metadata
    return metadata
  
  def getTimeseriesData(self, units='kg/s', lcheck=True, lexpand=True, lfill=True, period=None, lflatten=True):
    ''' extract time series data and time coordinates from a WSC monthly CSV file '''
    if self.monthly_file:
      # use numpy's CSV functionality
      # get timeseries data
      with open(self.monthly_file, 'r') as f:
          for i,line in enumerate(f.readlines()):
              if 'ID,PARAM,TYPE,Year'.lower() in line.lower(): break
      header = i+1
      if header > 2: raise ValueError('Header not detected correctly - file format may have changed.')
      data = np.genfromtxt(self.monthly_file, dtype=np.float32, delimiter=',', skip_header=header, filling_values=np.nan,  
                           usecols=np.arange(4,28,2), usemask=True, loose=True, invalid_raise=True)
      assert data.shape[1] == 12, data.shape
      # for some reason every value is followed by an extra comma...
      #data = np.ma.masked_less(data, 10) # remove some invalid values
      # N.B.: some values appear unrealistically small, however, these are removed in the check-
      #       section below (it appears they consistently fail the ckeck test)
      if units.lower() == 'kg/s': data *= 1000. # m^3 == 1000 kg (water)
      elif units.lower() == 'm^3/s': pass # original units
      else: raise ArgumentError("Unknown units: {}".format(units))
      # get time coordinates and verification flag
      check = np.genfromtxt(self.monthly_file, dtype=np.int, delimiter=',', skip_header=header, filling_values=-9999,  
                           usecols=np.arange(1,4,1), usemask=True, loose=True, invalid_raise=True)
      assert check.shape[0] == data.shape[0], check.shape
      assert np.all(check >= 0), np.sum(check < 0)
      time = check[:,2].astype(np.int) # this is the year (time coordinate)
      # determine valid entries
      if lcheck:
        check = np.all(check[:,:2]==1, axis=1) # require all entries to be one
        # N.B.: I'm not sure what it means if values are not equal to one, but the flow values look 
        #       unrealistically small (see above); probably different units...
        data = data[check,:]; time = time[check]
        assert time.shape[0] == data.shape[0], check.shape
      # slice off values outside the period of interest
      if period:
        valid = np.logical_and(time >= period[0],time < period[1])
        time = time[valid]; data = data[valid]
      # fill in missing time periods/years
      if lfill:
        if period: time0 = period[0]; time1 = period[1]
        else: time0 = time[0]; time1 = time[-1]+1
        idx = np.asarray(time - time0, dtype=np.int32); tlen = time1 - time0 # start at 0; length is last value (+1)
        pad_time = np.arange(time0,time1) # form continuous sequence
        #assert np.all( pad_time[idx] == time ), idx # potentially expensive
        time = pad_time # new continuous time coordinate
        pad_data = np.ma.zeros((tlen,12), dtype=np.float32)*np.NaN # pre-allocate with NaN
        pad_data.mask = True # mask everywhere for now
        pad_data[idx,:] = data; #pad_data.mask[idx,:] = data.mask
        #assert np.all( pad_data.mask[idx,:] == data.mask ) # potentially expensive
        data = pad_data
      # now, expand time coordinate by adding month
      if lexpand:
        time = time.reshape((time.size,1))
        coord = np.repeat((time-1979)*12, 12, axis=1) + np.arange(0,12).reshape((1,12))
        assert coord.shape == data.shape, coord.shape
        #assert np.all( np.diff(coord.flatten()) == 1 ), coord  # potentially expensive
        time = coord
      if lflatten:
        time = time.flatten(); data = data.flatten()
      # return data array and coordinate vector
      return data, time
    else:
      raise IOError("No timeseries file defined or file not found for gage station '{}'.\n(folder: '{}')".format(self.name,self.folder))
        
# function to get a GageStation instance
def getGageStation(basin=None, station=None, name=None, folder=None, river=None, basin_list=None, **kwargs):
  ''' return an initialized GageStation instance, but infer parameters from input '''
  if isinstance(station, GageStation):
    return station # very simple shortcut!
  # resolve basin
  if basin_list is not None:
      if isinstance(basin,str): 
          # select basin from basin list
          if basin not in basin_list: raise GageStationError(basin)
          else: basin = basin_list[basin]
      elif isinstance(station,str) and basin is None:
          # search for station in basin list to determine basin
          for basin_set in list(basin_list.values()):
              for station_list in list(basin_set.stations.values()):
                  if station in station_list:
                      basin = basin_set # station found!
                      break # break inner loop
              if basin is not None: 
                  break # break outer loop
      else:
        raise GageStationError(basin)
  else: basin_name = basin
  name = name or station # name actually has precedence
  # determine basin meta data
  if isinstance(basin,Basin):
    folder = basin.folder if folder is None else folder
    basin_name = basin.name
    # figure out station
    if name is None and basin.maingage is not None: 
      return basin.maingage # simple!
    if isinstance(basin,BasinSet):
      if name in basin.stations: 
        return basin.stations[name] # also straight forward
      if not basin.rivers: pass
      elif river is None: river = basin.rivers[0]
      elif river not in basin.rivers: raise GageStationError(river,basin.rivers)
      name = '{}_{}'.format(river,name) # try with adding river name
      if name in basin.stations: 
        return basin.stations[name] # also straight forward      
  elif river is not None and river+'_' not in name:
      name = '{}_{}'.format(river,name)
  # if we are not done yet, make sure we have valid folder and file names now!
  if not (isinstance(folder,str) and isinstance(name,str)):
    raise GageStationError('Specify either basin (and station) or folder & station prefix/name.')
    # N.B.: this Error also indicates that no gage station is available
  if not os.path.exists(folder): raise IOError(folder)
  # instantiate and return GageStation
  return GageStation(basin=basin_name, river=None, name=name, folder=folder, **kwargs)
      
    
## Functions that handle access to ASCII files
def loadGageStation(basin=None, station=None, varlist=None, varatts=None, mode='climatology', 
                    aggregation=None, filetype='monthly', folder=None, name=None, period=None,
                    basin_list=None, lcheck=True, lexpand=True, lfill=True, lflatten=True,
                    lkgs=True, scalefactors=None, title=None):
  ''' function to load hydrograph climatologies and timeseries for a given basin '''
  ## resolve input
  if mode == 'timeseries' and aggregation: 
    raise ArgumentError('Timeseries does not support aggregation.')
  # get GageStation instance (can be passes directly)
  if isinstance(station,str) or station is None:
      station = getGageStation(basin=basin, station=station, name=name, folder=folder, 
                               river=None, basin_list=basin_list, lcheck=True)
  # variable attributes
  if varlist is None: varlist = variable_list
  elif not isinstance(varlist,(list,tuple)): raise TypeError  
  varlist = list(varlist) # make copy of varlist to avoid interference
  if varatts is None: 
    if aggregation is None: varatts = variable_attributes_kgs if lkgs else variable_attributes_mms
    else: varatts = agg_varatts_kgs if lkgs else agg_varatts_mms
  elif not isinstance(varatts,dict): raise TypeError
  
  ## read csv data
  # time series data and time coordinates
  lexpand = True; lfill = True
  if mode == 'climatology': lexpand = False; lfill = False; lflatten = False
  data, time = station.getTimeseriesData(units='kg/s' if lkgs else 'm^3/s', lcheck=True, lexpand=lexpand, 
                                         lfill=lfill, period=period, lflatten=lflatten)
  # station meta data
  metadata = station.getMetaData(lcheck=True)
  den = metadata['shp_area'] if lkgs else ( metadata['shp_area'] / 1000. )
  ## create dataset for station
  dataset = Dataset(name='WSC', title=title or metadata['Station Name'], varlist=[], atts=metadata,) 
  if mode.lower() in ('timeseries','time-series'): 
    time = time.flatten(); data = data.flatten() # just to make sure...
    # make time axis based on time coordinate from csv file
    timeAxis = Axis(name='time', units='month', coord=time, # time series centered at 1979-01
                    atts=dict(long_name='Month since 1979-01'))
    dataset += timeAxis
    # load mean discharge
    dataset += Variable(axes=[timeAxis], data=data, atts=varatts['discharge'])
    # load mean runoff
    doa = data / den 
    dataset += Variable(axes=[timeAxis], data=doa, atts=varatts['runoff'])
  elif mode == 'climatology': 
    # N.B.: this is primarily for backwards compatibility; it should not be used anymore...
    # make common time axis for climatology
    te = 12 # length of time axis: 12 month
    climAxis = Axis(name='time', units='month', length=12, coord=np.arange(1,te+1,1)) # monthly climatology
    dataset.addAxis(climAxis, copy=False)
    # extract variables (min/max/mean are separate variables)
    # N.B.: this is mainly for backwards compatibility
    doa = data / den
    if aggregation is None or aggregation.lower() == 'mean':
      # load mean discharge
      tmpdata = nf.nanmean(data, axis=0)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['discharge'])
      dataset.addVariable(tmpvar, copy=False)
      # load mean runoff
      tmpdata = nf.nanmean(doa, axis=0)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['runoff'])
      dataset.addVariable(tmpvar, copy=False)
    if aggregation is None or aggregation.lower() == 'std':
      # load  discharge standard deviation
      tmpdata = nf.nanstd(data, axis=0, ddof=1) # very few values means large uncertainty!
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['discstd'])
      dataset.addVariable(tmpvar, copy=False)
      # load  runoff standard deviation
      tmpdata = nf.nanstd(doa, axis=0, ddof=1)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['roff_std'])
      dataset.addVariable(tmpvar, copy=False)
    if aggregation is None or aggregation.lower() == 'sem':
      # load  discharge standard deviation
      tmpdata = nf.nansem(data, axis=0, ddof=1) # very few values means large uncertainty!
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['discsem'])
      dataset.addVariable(tmpvar, copy=False)
      # load  runoff standard deviation
      tmpdata = nf.nansem(doa, axis=0, ddof=1)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['roff_sem'])
      dataset.addVariable(tmpvar, copy=False)
    if aggregation is None or aggregation.lower() == 'max':
      # load maximum discharge
      tmpdata = nf.nanmax(data, axis=0)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['discmax'])
      dataset.addVariable(tmpvar, copy=False)
      # load maximum runoff
      tmpdata = nf.nanmax(doa, axis=0)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['roff_max'])
      dataset.addVariable(tmpvar, copy=False)
    if aggregation is None or aggregation.lower() == 'min':
      # load minimum discharge
      tmpdata = nf.nanmin(data, axis=0)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['discmin'])
      dataset.addVariable(tmpvar, copy=False)
      # load minimum runoff
      tmpdata = nf.nanmin(doa, axis=0)
      tmpvar = Variable(axes=[climAxis], data=tmpdata, atts=varatts['roff_min'])
      dataset.addVariable(tmpvar, copy=False)
  else: 
    raise NotImplementedError("Time axis mode '{}' is not supported.".format(mode))
  # adjust scalefactors, if necessary
  if scalefactors:
      if isinstance(scalefactors,dict):
          dataset = updateScalefactor(dataset, varlist=scalefactors, scalefactor=None)
      elif isNumber(scalefactors):
          scalelist = ('discharge','StdDisc','SEMDisc','MaxDisc','MinDisc',)
          dataset = updateScalefactor(dataset, varlist=scalelist, scalefactor=scalefactors)
      else: 
          raise TypeError(scalefactors) 
  # return station dataset
  return dataset   

# helper function to convert arguments
def _sliceArgs(slices=None, basin=None, station=None, period=None, years=None):
    ''' a helper function to translate arguments from standard datasets to WSC specifics; the main difference is
        that basins and stations can be loaded directly (without slicing) '''
    if slices is None: slices = dict()
    period = period or slices.get('period',None) or years or slices.get('years',None)
    basin = basin or slices.get('basin',None) or slices.get('shape_name',None)
    station = station or slices.get('station',None) or slices.get('station_name',None)
    return dict(period=period, basin=basin, station=station) 

# function to load gage station climatologies
def loadWSC_Stn(name=dataset_name, title=None, station=None, basin=None, varlist=None, varatts=None, folder=None, 
                period=None, years=None, filetype='monthly', basin_list=None, slices=None, lkgs=True, 
                scalefactors=None):
    ''' Get monthly WSC gage station climatology by station name. '''
    dataset = loadGageStation(varlist=varlist, varatts=varatts, mode='climatology', aggregation='mean',  title=title,
                              filetype=filetype, folder=folder, name=None, basin_list=basin_list, 
                              lkgs=lkgs, scalefactors=scalefactors,
                              **_sliceArgs(slices=slices, basin=basin, station=station, period=period, years=years))
    if name: dataset.name = name
    # return formatted dataset
    return dataset

# function to load station time-series
def loadWSC_StnTS(name=dataset_name, title=None, station=None, basin=None, varlist=None, varatts=None, folder=None, 
                  period=None, years=None, filetype='monthly', basin_list=None, slices=None, lkgs=True, 
                  scalefactors=None):
    ''' Get monthly WSC gage station time-series by station name. '''
    dataset = loadGageStation(varlist=varlist, varatts=varatts, mode='time-series', aggregation=None,  title=title,
                              filetype=filetype, folder=folder, name=None, basin_list=basin_list, 
                              lkgs=lkgs, scalefactors=scalefactors,
                              **_sliceArgs(slices=slices, basin=basin, station=station, period=period, years=years))
    if name: dataset.name = name
    # return formatted dataset
    return dataset

# function to load regionally averaged climatologies
def loadWSC_Shp(name=dataset_name, title=None, shape=None, basin=None, station=None, varlist=None, varatts=None,
                folder=None, period=None, years=None, filetype='monthly', basin_list=None, slices=None, lkgs=True, 
                scalefactors=None):
    ''' Get monthly WSC gage station climatology by river basin or region. '''
    dataset = loadGageStation(varlist=varlist, varatts=varatts, mode='climatology', aggregation='mean',  title=title,
                              filetype=filetype, folder=folder, name=None, basin_list=basin_list, 
                              lkgs=lkgs, scalefactors=scalefactors, 
                              **_sliceArgs(slices=slices, basin=basin, station=station, period=period, years=years))
    if name: dataset.name = name
    # return formatted dataset
    return dataset

# function to load regional/shape time-series
def loadWSC_ShpTS(name=dataset_name, title=None, shape=None, basin=None, station=None, varlist=None, varatts=None, 
                  folder=None, period=None, years=None, filetype='monthly', basin_list=None, slices=None, lkgs=True,
                  scalefactors=None):
    ''' Get monthly WSC gage station time-series by river basin or region. '''
    dataset = loadGageStation(varlist=varlist, varatts=varatts, mode='time-series', aggregation=None, title=title, 
                              filetype=filetype, folder=folder, name=None, basin_list=basin_list, 
                              lkgs=lkgs, scalefactors=scalefactors,
                              **_sliceArgs(slices=slices, basin=basin, station=station, period=period, years=years))
    if name: dataset.name = name
    # return formatted dataset
    return dataset

## some helper functions to test conditions
# defined in module main to facilitate pickling
def test_encl(val, index,dataset,axis):
  ''' check if shape is fully enclosed by grid ''' 
  return dataset.shp_encl[index] == val  
def test_full(val, index,dataset,axis):
  ''' check if shape fully covers the grid ''' 
  return dataset.shp_full[index] == val
def test_empty(val, index,dataset,axis):
  ''' check if shape is outside of grid ''' 
  return dataset.shp_empty[index] == val 
def test_mina(val,index,dataset,axis):
  ''' check minimum area ''' 
  return dataset.shp_area[index] >= val
def test_maxa(val,index,dataset,axis):
  ''' check maximum area ''' 
  return dataset.shp_area[index] <= val
 
# apply tests to list
def apply_test_suite(tests, index, dataset, axis):
  ''' apply an entire test suite to '''
  # just call all individual tests for given index 
  return all(test(index,dataset,axis) for test in tests)

## select a set of common stations for an ensemble, based on certain conditions
def selectStations(datasets, shpaxis='shape', imaster=None, linplace=True, lall=False, **kwcond):
  ''' A wrapper for selectCoords that selects stations based on common criteria '''
  # pre-load NetCDF datasets
  for dataset in datasets: 
    if isinstance(dataset,DatasetNetCDF): dataset.load() 
  # list of possible constraints
  tests = [] # a list of tests to run on each station
  #loadlist =  (datasets[imaster],) if not lall and imaster is not None else datasets 
  # test definition
  for key,val in kwcond.items():
    key = key.lower()
    if key[:4] == 'encl' or key[:4] == 'cont':
      val = bool(val)
      tests.append(functools.partial(test_encl, val))
    elif key == 'full':
      val = bool(val)
      tests.append(functools.partial(test_full, val))
    elif key[:4] == 'empt':
      val = bool(val)
      tests.append(functools.partial(test_empty, val))
    elif key == 'min_area':
      if not isNumber(val): raise TypeError
      val = val*1e6 # units in km^2  
      tests.append(functools.partial(test_mina, val))    
    elif key == 'max_area':
      if not isNumber(val): raise TypeError
      val = val*1e6 # units in km^2  
      tests.append(functools.partial(test_maxa, val))
    else:
      raise NotImplementedError("Unknown condition/test: '{:s}'".format(key))
  # define test function (all tests must pass)
  if len(tests) > 0:
    testFct = functools.partial(apply_test_suite, tests)
  else: testFct = None
  # pass on call to generic function selectCoords
  datasets = selectElements(datasets=datasets, axis=shpaxis, testFct=testFct, imaster=imaster, linplace=linplace, lall=lall)
  # return sliced datasets
  return datasets


## abuse main block for testing
if __name__ == '__main__':
  
#   from projects.WSC_basins import basin_list, basins, BasinSet
  # N.B.: importing BasinInfo through WSC_basins is necessary, otherwise some isinstance() calls fail
  GRW_stations = {'Grand River':['Brantford','Marsville'], 'Conestogo River':['Glen Allan'],
                  'Fairchild Creek':['Brantford'],'Speed River':['Guelph'], 'Whitemans Creek':['Mount Vernon']}
  basin_list = dict(GRW=BasinSet(name='GRW', long_name='Grand River Watershed', data_source='Aquanty', stations=GRW_stations, 
                                 rivers=['Grand River', 'Conestogo River', 'Fairchild Creek', 'Speed River', 'Whitemans Creek'], 
                                 subbasins=['WholeGRW','UpperGRW','LowerGRW','NorthernGRW','SouthernGRW','WesternGRW']),
                    SSR=BasinSet(name='SSR', long_name='South Saskatchewan River', rivers=['South Saskatchewan'], 
                                 data_source='Aquanty', stations={'South Saskatchewan':['St Louis','Saskatoon']}, 
                                 subbasins=['WholeSSR']),
                    PRW=BasinSet(name='PRW', long_name='Payne River Watershed', data_source='Aquanty',
                                 rivers=['Payne River', ], stations={'Payne River':['Berwick',],}, subbasins=['WholePRW',]),
                    )


#   mode = 'catchment'
  mode = 'gauge'
#   mode = 'basin'

#   basin_name = 'PRW'; station = 'Payne River_Berwick'
#   basin_name = 'SSR'; station = 'St Louis'
  basin_name = 'GRW'; station = 'Grand River_Brantford'
  
  if mode == 'catchment':
    
      # test new Catchment class
      name = '02GB001'
      catch = Catchment(name=name, ldebug=True)
      print(catch.station)
      assert catch.gauge == None
      
      gauge = catch.getGaugeSation(lcheck=True)
      assert isinstance(gauge,GageStation)
      print(catch.metadata)
      assert catch.metadata['ID'] == name
      
      gauge_ds = catch.getGaugeDataset(lkgs=False)
      print(gauge_ds)
      print(gauge_ds.discharge.mean())
    
      name = '02GB006'
      catch = Catchment(name=name, ldebug=True)
      print(catch.station)
      assert catch.gauge == None
      
      gauge = catch.getGaugeSation(lcheck=True)
      print(gauge)
      assert gauge is None
      
  elif mode == 'gauge':
    
      # load a random station
#       stnds = loadGageStation(basin=basin_name, basin_list=basin_list, station=station,
#                               scalefactors=1e-4, lkgs=True)
      stnds = loadWSC_StnTS(name=dataset_name, title=None, station=station, basin=basin_name, 
                            varlist=None, varatts=None, folder=None, 
                            period=None, years=None, filetype='monthly', 
                            basin_list=basin_list, slices=None, lkgs=True, scalefactors=1e-4)
      print((stnds))
      
      print()
      print((stnds.discharge.plot))
      
  elif mode == 'basin':
    
      # verify basin info
      basin_set = basin_list[basin_name]
      print(basin_set.long_name)
      print(basin_set.stations)
      
      # load basins
      basin = basin_list[basin_name]
      print(basin.long_name)
      print(basin)
      assert basin.name == basin_name, basin.name
      
      # load station data
      station = basin.getMainGage(aggregation=None, lkgs=False)
      print()
      print(station)
      print()
      assert station.atts.ID == loadGageStation(basin=basin_name, basin_list=basin_list).atts.ID
      print(station.discharge.climMean()[:])
      
      # print basins
      print()
      for bsn in basin_list.keys():
        print(bsn)