'''
Created on 2014-08-20

A module to load station daily data from Environment Canada from ASCII files and convert them to monthly 
NetCDF datasets (with extremes); the module also provides a wrapper to load the NetCDF datasets. 

@author: Andre R. Erler, GPL v3
'''

# external imports
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
from copy import deepcopy
import fileinput
import inspect
# internal imports
from datasets.common import days_per_month, name_of_month, data_root
from geodata.misc import ParseError, ArgumentError, DateError
from geodata.gdal import Shape
from geodata.station import StationDataset, Variable, Axis
# from geodata.misc import DatasetError
from warnings import warn

## EC (Environment Canada) Meta-data

dataset_name = 'EC'
root_folder = data_root + dataset_name + '/'
orig_ts_file = '{0:s}{1:s}.txt' # filename pattern: variable name and station ID
tsfile = 'ec{0:s}_monthly.nc' # filename pattern: station type 
avgfile = 'ec{0:s}_clim{1:s}.nc' # filename pattern: station type and ('_'+period)
avgfolder = root_folder + 'ecavg/'  # folder for user data

# variable attributes and name
varatts = dict(T2       = dict(name='T2', units='K', atts=dict(long_name='Average 2m Temperature')), # 2m average temperature
               Tmin     = dict(name='Tmin', units='K', atts=dict(long_name='Minimum 2m Temperature')), # 2m minimum temperature
               Tmax     = dict(name='Tmax', units='K', atts=dict(long_name='Maximum 2m Temperature')), # 2m maximum temperature
               precip   = dict(name='precip', units='kg/m^2/s', atts=dict(long_name='Total Precipitation')), # total precipitation
               solprec  = dict(name='solprec', units='kg/m^2/s', atts=dict(long_name='Solid Precipitation')), # solid precipitation
               liqprec  = dict(name='liqprec', units='kg/m^2/s', atts=dict(long_name='Liquid Precipitation')), # liquid precipitation
               # axes (don't have their own file; listed in axes)
               time     = dict(name='time', units='month', atts=dict(long_name='Month of the Year')), # time coordinate
               station  = dict(name='station', units='#', atts=dict(long_name='Station Number'))) # ordinal number of station
# list of variables to load
variable_list = varatts.keys() # also includes coordinate fields    


## a class that handles access to station records in ASCII files
class DailyStationRecord(object):
  '''
    A class that is used by StationRecords to facilitate access to daily station records from ASCII files.  
  '''
  # list of station parameters that need to be supplied
  id         = '' # station ID
  name       = '' # station name
  variable   = '' # variable name (full name used in header)
  units      = '' # data units used in record 
  filename   = '' # absolute path of ASCII file containing data record
  prov       = '' # province (in Canada)  
  joined     = False # whether station record was merged with another nearby  
  begin_year = 0 # year of first record
  begin_mon  = 0 # month of first record
  end_year   = 0 # year of last record 
  end_mon    = 0 # month of last record
  lat        = 0. # latitude of station location
  lon        = 0. # longitude of station location
  alt        = 0. # station elevation (altitude)
  
  # id='', name='', datatype='', filename='', prov='', begin_year=0, begin_mon=0, end_year=0, end_mon=0, lat=0, lon=0
  def __init__(self, **kwargs):
    ''' initialize station parameters '''
    # generate attribute list
    parameters = inspect.getmembers(self, lambda att: not(inspect.isroutine(att)))
    parameters = [key for key,val in parameters if key[:2] != '__' and key[-2:] != '__']
    cls = self.__class__
    # parse input    
    for key,value in kwargs.iteritems():
      if key in parameters:
        # simple type checking
        if isinstance(cls.__dict__[key], basestring) and not isinstance(value, basestring):
          raise TypeError, "Parameter '{:s}' has to be of type 'basestring'.".format(key)  
        elif ( isinstance(cls.__dict__[key], (float,np.inexact)) and 
               not isinstance(value, (int,np.integer,float,np.inexact)) ):
          raise TypeError, "Parameter '{:s}' has to be of a numeric type.".format(key)
        elif ( isinstance(cls.__dict__[key], (int,np.integer)) and 
               not isinstance(value, (int,np.integer)) ):
          raise TypeError, "Parameter '{:s}' has to be of an integer type.".format(key)
        elif isinstance(cls.__dict__[key], (bool,np.bool)) and not isinstance(value, (bool,np.bool)):
          raise TypeError, "Parameter '{:s}' has to be of boolean type.".format(key)  
        # unfortunately this automated approach makes type checking a bit clumsy
        self.__dict__[key] = value
      else: raise ArgumentError, "Invalid parameter: '{:s}'".format(key)
    # check that all parameters are set
    for param in parameters:
      if param not in self.__dict__:
        raise ArgumentError, "Parameter '{:s}' was not set (missing argument).".format(param)
      
  def validateHeader(self, headerline):
    ''' validate header information against stored meta data '''
    # parse header line (print header if an error occurs)
    header = [elt.strip().lower() for elt in headerline.split(',')]
    if self.id.lower() != header[0]: raise ParseError, headerline # station ID
    if self.name.lower() != header[1]: raise ParseError, headerline # station name
    if self.prov.lower() != header[2]: raise ParseError, headerline # province
    if 'joined' not in header[3]: raise ParseError, headerline # station joined or not
    else:
      if self.joined and 'not' in header[3]: raise ParseError, headerline # station joined or not
      if not self.joined and 'not' not in header[3]: raise ParseError, headerline # station joined or not
    if 'daily' not in header[4]: raise ParseError, headerline # this class only deals with daily values
    if self.variable.lower() not in header[4]: raise ParseError, headerline # variable name
    if self.units.lower() not in header[5]: raise ParseError, headerline # variable units
    # if no error was raised, we are good
    
  def parseRecord(self):
    ''' open the station file and parse records; return a daiy time-series '''
    # open file
    f = open(self.filename)
    self.validateHeader(f.readline()) # read first line as header
    # allocate daily data array (31 days per month, filled with NaN for missing values)
    tlen = ( (self.end_year - self.begin_year) * 12 + (self.end_mon - self.begin_mon +1) ) * 31
    data = np.zeros((tlen,), dtype=np.float16) # only three significant digits...
    data[:] = np.NaN # use NaN as missing values
    # iterate over line
    oldyear = self.begin_year; oldmon = self.begin_mon -1; z = 0
    for line in f:      
      ll = line.replace('-9999.9', ' -9999.9').split() # without the replace, the split doesn't work
      if ll[0].isdigit() and ll[1].isdigit():
        year = int(ll[0]); mon = int(ll[1])
        # check date bounds
        if year == self.begin_year and mon < self.begin_mon: raise DateError, line
        elif year < self.begin_year: raise DateError, line
        if year == self.end_year and mon > self.end_mon: raise DateError, line
        elif year > self.end_year: raise DateError, line
        # check continuity
        if year == oldyear and mon == oldmon+1: pass
        elif year == oldyear+1 and oldmon == 12 and mon ==1: pass 
        else: raise DateError, line
        oldyear = year; oldmon = mon
        # parse values
        if len(ll[2:]) > 5: # need more than 5 valid values
          zz = z 
          for num in ll[2:]:
            if num[:7] == '-9999.9' or num[-1] == 'M': pass # missing value; already pre-filled NaN
            elif 3 < len(num): # at least 3 digits plus decimal, i.e. ignore the flag
              if num.isdigit(): n = float(num)
              elif num[:-1].isdigit: n = float(num[:-1])
              else: raise ParseError, "Unable to process value '{:s}' in line:\n {:s}".format(num,line)
              if n < 0: raise ParseError, "Encountered negative value '{:s}' in line:\n {:s}".format(num,line)
              data[zz] = n
            else: raise ParseError, "Unable to process value '{:s}' in line:\n {:s}".format(num,line)
            zz += 1
          if zz != z+31: raise ParseError, 'Line has {:d} values instead of 31:\n {:s}'.format(zz-z,line)  
        # increment counter
        z += 31
      elif ll[0] != 'Year' or ll[1] != 'Mo':
        raise ParseError, "No valid title or data found at begining of file:\n {:s}".format(self.filename)
    if z != tlen: raise ParseError, 'Number of lines in file is inconsistent with begin and end date: {:s}'.format(self.filename)
    # close again
    f.close()
    # return array
    return data
  
    
## class to read station records and return a dataset
class StationRecords(object):
  '''
    A class that provides methods to load station data and associated meta data from files of a given format;
    The format itself will be defines in child classes.
    The data will be converted to monthly statistics and accessible as a PyGeoData dataset or can be written 
    to a NetCDF file.
  '''
  # list of format parameters
  metafile = 'stations.txt' # file that contains station meta data (to load station records)
  folder   = '{0:s}_{1:s}' # root folder for station data: interval and datatype
  interval = '' # source data interval (currently only daily)
  datatype = '' # variable class, e.g. temperature or precipitation tyes
  vardefs  = None # parameters and definitions associated with variables
  
  def __init__(self):
    ''' Parse station file and initialize station records. '''
    # open and parse station file
    
    # initialize station objects and add to list
    
    # initialize station dataset
    pass

## class that implements particularities of EC temperature station records
class DailyTemp(StationRecords):
  '''
    A class to load daily temperature records from EC stations. 
  '''
  interval = 'daily' # source data interval (currently only daily)
  datatype = 'temp' # variable class, e.g. temperature or precipitation tyes
  vardefs  = None # parameters and definitions associated with variables
    

## load pre-processed EC station time-series
def loadEC_TS(): 
  ''' Load a monthly time-series of pre-processed EC station data. '''
  return NotImplementedError

## load pre-processed EC station climatology
def loadEC(): 
  ''' Load a pre-processed EC station climatology. '''
  return NotImplementedError
  
## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = orig_ts_file # filename pattern: variable name and resolution
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: variable name and resolution
data_folder = avgfolder # folder for user data
grid_def = None # no grid here...
LTM_grids = None 
TS_grids = None
grid_res = None
default_grid = None
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = None # time-series data
loadClimatology = None # pre-processed, standardized climatology
loadStationTimeSeries = loadEC_TS # time-series data
loadStationClimatology = loadEC # pre-processed, standardized climatology

if __name__ == '__main__':

  mode = 'test_ASCII_station'
#   mode = 'convert_ASCII'
  
  # do some tests
  if mode == 'test_ASCII_station':  
    
    # initialize station
    test = DailyStationRecord(id='250M001', name='MOULD BAY', variable='precipitation', units='mm', prov='NT',  
                              begin_year=1948, begin_mon=1, end_year=2012, end_mon=12, lat=76.2, lon=-119.3,
                              alt=2, joined=True, filename='/data/EC/daily_precip/dt/dt250M001.txt')
    data = test.parseRecord()
    print data.shape, data.dtype
    print np.nanmin(data), np.nanmean(data), np.nanmax(data)