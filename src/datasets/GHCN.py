# coding: utf-8
'''
Created on 2016-07-2

A module to load daily station data from the Global Historical Climatology Network in ASCII format and convert them to monthly 
NetCDF datasets (with extremes); the module also provides a wrapper to load the NetCDF datasets. 

@author: Yiling Huo, Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
from copy import deepcopy
import codecs, calendar, functools
from warnings import warn
# internal imports
from datasets.CRU import loadCRU_StnTS
from datasets.common import days_per_month, getRootFolder, selectElements
from datasets.common import CRU_vars, stn_params, nullNaN
from geodata.misc import ParseError, ArgumentError, DatasetError, AxisError
from geodata.misc import RecordClass, StrictRecordClass, isNumber, isInt 
from geodata.base import Axis, Variable, Dataset
from utils.nctools import writeNetCDF
from geodata.netcdf import DatasetNetCDF
# import derived variables from the WRF Tools package wrfavg
import imp, os
# read code root folder from environment variable
code_root = os.getenv('CODE_ROOT')
if not code_root : raise ArgumentError('No CODE_ROOT environment variable set!')
if not os.path.exists(code_root): raise ImportError("The code root '{:s}' directory set in the CODE_ROOT environment variable does not exist!".format(code_root))
# import module from WRF Tools explicitly to avoid name collision
if os.path.exists(code_root+'/WRF-Tools/Python/wrfavg/derived_variables.py'):
  dv = imp.load_source('derived_variables', code_root+'/WRF-Tools/Python/wrfavg/derived_variables.py') # need explicit absolute import due to name collision
elif os.path.exists(code_root+'/WRF Tools/Python/wrfavg/derived_variables.py'):
  dv = imp.load_source('derived_variables', code_root+'/WRF Tools/Python/wrfavg/derived_variables.py') # need explicit absolute import due to name collision
#dv = importlib.import_module('wrfavg.derived_variables') # need explicit absolute import due to name collision
#import wrfavg.derived_variables as dv
from utils.constants import precip_thresholds
#from wrfavg.derived_variables import precip_thresholds

## GHCN (Environment Canada) Meta-data

dataset_name = 'GHCN'
root_folder = getRootFolder(dataset_name=dataset_name) # get dataset root folder based on environment variables
orig_ts_file = '{1:s}.dly' # filename pattern: variable name and station ID
tsfile = 'ghcn{0:s}_monthly.nc' # filename pattern: station type
#tsfile_prov = 'ghcn{0:s}_{1:s}_monthly.nc' # filename pattern with province: station type, province  
avgfile = 'ghcn{0:s}_clim{1:s}.nc' # filename pattern: station type and ('_'+period)
avgfolder = root_folder + 'ghcnavg/'  # folder for user data


# variable attributes and name
varatts = dict(T2         = dict(name='T2', units='C', atts=dict(long_name='Average Temperature')), # average temperature
               Tmin       = dict(name='Tmin', units='C', atts=dict(long_name='Minimum Temperature')), # minimum temperature
               Tmax       = dict(name='Tmax', units='C', atts=dict(long_name='Maximum Temperature')), # maximum temperature
               precip     = dict(name='precip', units='mm', atts=dict(long_name='Precipitation')), # total precipitation
               solprec    = dict(name='solprec', units='kg/m^2/s', atts=dict(long_name='Solid Precipitation')), # solid precipitation
               #liqprec    = dict(name='liqprec', units='kg/m^2/s', atts=dict(long_name='Liquid Precipitation')), # liquid precipitation
               # N.B.: note that some variables are defined after the PrecipDef and TempDef classes below
               # secondary variables for consistent loading (currently only precip)
               MaxPrecip     = dict(name='MaxPrecip_1d', units='kg/m^2/s'), # maximum daily precip                    
               MaxPrecip_5d  = dict(name='MaxPrecip_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
               MaxSolprec    = dict(name='MaxSolprec_1d', units='kg/m^2/s'), # maximum daily precip
               MaxSolprec_5d = dict(name='MaxSolprec_5d', units='kg/m^2/s'), # maximum pendat (5 day) precip
               # meta/constant data variables
               # N.B.: 'stn'/'station' prefix is to allow consistent naming and avoid name collisions with variables in other datasets
               name        = dict(name='station_name', units='', atts=dict(long_name='Station Name')), # the proper name of the station
               lat         = dict(name='stn_lat', units='deg N', atts=dict(long_name='Latitude')), # geographic latitude field
               lon         = dict(name='stn_lon', units='deg E', atts=dict(long_name='Longitude')), # geographic longitude field
               alt         = dict(name='stn_zs', units='m', atts=dict(long_name='Station Elevation')), # station elevation
               begin_date  = dict(name='stn_begin_date', units='month', atts=dict(long_name='Month since 1980-01', # begin of station record
                                                                              description='Begin of Station Record (relative to 1980-01)')), 
               end_date    = dict(name='stn_end_date', units='month', atts=dict(long_name='Month since 1980-01', # begin of station record
                                                                          description='End of Station Record (relative to 1980-01)')),
               stn_rec_len = dict(name='stn_rec_len', units='month', atts=dict(long_name='Length of Record', # actual length of station record
                                                                      description='Number of Month with valid Data')),
               # axes (also sort of meta data)
               time     = dict(name='time', units='month', atts=dict(long_name='Month since 1980-01')), # time coordinate
               station  = dict(name='station', units='#', atts=dict(long_name='Station Number'))) # ordinal number of statio
varatts['SummerDays_+25'] = dict(name='sumfrq', units='', atts=dict(long_name='Fraction of Summer Days (>25C)')), # N.B.: rename on load,
varatts['FrostDays_+0']   = dict(name='frzfrq', units='', atts=dict(long_name='Fraction of Frost Days (< 0C)')),  #       same as WRF
# add variables with different wet-day thresholds
for threshold in precip_thresholds:
    suffix = '_{:03d}'.format(int(10*threshold))
    varatts['WetDays'+suffix]      = dict(name='wetfrq'+suffix, units='') # fraction of wet/rainy days                    
    varatts['WetDayRain'+suffix]   = dict(name='dryprec'+suffix, units='kg/m^2/s') # precipitation rate above dry-day thre
    varatts['WetDayPrecip'+suffix] = dict(name='wetprec'+suffix, units='kg/m^2/s', 
                                          atts=dict(fillValue=0), transform=nullNaN) # wet-day precipitation rate (kg/m^2/s)

# list of variables to load
variable_list = list(varatts.keys()) # also includes coordinate fields    

class DailyStationRecord(StrictRecordClass):
  '''
    A class that is used by StationRecords to facilitate access to daily station records from ASCII files.  
  '''
  # list of station parameters that need to be supplied
  id         = '' # station ID
  name       = '' # station name
  variable   = '' # variable name (full name used in header)
  units      = '' # data units used in record
  dtype      = '' # data type (default: float32)
  missing    = '' # string indicating missing value
  mflag       = ''
  qflag       = ''
  sflag       = ''
  varmin     = 0. # smallest allowed value in data
  varmax     = 0. # largest allowed value in data
  filename    = '' # absolute path of ASCII file containing data record
  encoding   = '' # text file encoding
  begin_year = 0 # year of first record
  begin_mon  = 0 # month of first record
  end_year   = 0 # year of last record 
  end_mon    = 0 # month of last record
  lat        = 0. # latitude of station location
  lon        = 0. # longitude of station location
  alt        = 0. # station elevation (altitude)
  # id='', name='', datatype='', filename='', prov='', begin_year=0, begin_mon=0, end_year=0, end_mon=0, lat=0, lon=0
      
  def validateHeader(self, headerline):
    ''' validate header information against stored meta data '''
    # parse header line (print header if an error occurs)
    header = headerline
    if self.id.lower() != header[:11].lower(): raise ParseError(headerline) # station ID
    
  def checkHeader(self):
    ''' open the station file and validate the header information; then close '''
    # open file
    f = codecs.open(self.filename, 'r', encoding=self.encoding)
    self.validateHeader(f.readline()) # read first line as header
    f.close()
  
  def parseRecord(self):
    ''' open the station file and parse records; return a daily time-series '''
    # open file
    f = codecs.open(self.filename, 'r', encoding=self.encoding)
    infoline=f.readlines()
    ll=infoline[0]
    self.begin_year = int(ll[11:15])
    self.begin_mon = int(ll[15:17])
    #existflag=False
    off=-270 #number of characters in a line
    f.seek(off, 2)#-270 from the end of file
    infoline = f.readlines() 
    if len(infoline)>=2: 
      raise ParseError('last line incomplete')
    else:
      ll=infoline[-1]
      self.end_year = int(ll[11:15])
      self.end_mon = int(ll[15:17])
    oldyear = self.begin_year; oldmon = self.begin_mon -1;
    # allocate daily data array (31 days per month, filled with NaN for missing values)
    tlen = ( (self.end_year - self.begin_year) * 12 + (self.end_mon - self.begin_mon +1) ) * 31
    data = np.empty((tlen,), dtype=self.dtype); data.fill(np.NaN) # use NaN as missing values
    # some stuff to remember
    lm = len(self.missing)
    # iterate over line
    z = 0
    f.seek(0,0)
    for line in f:
      ll=line
      if self.variable==ll[17:21]:#check if it's the variable we're looking for
        year = int(ll[11:15]); mon = int(ll[15:17])  
        # check continuity
        discontinuity_flag = False
        if year == oldyear and mon == oldmon+1: pass
        elif year == oldyear+1 and oldmon == 12 and mon ==1: pass 
        else:
          discontinuity_flag = True
          print('discontinuity', oldyear, oldmon)
          while (discontinuity_flag):
            z+=31
            if oldmon == 12:
              oldmon=1;oldyear+=1
            else:
              oldmon+=1
            if year == oldyear and mon == oldmon+1: discontinuity_flag=False
            elif year == oldyear+1 and oldmon == 12 and mon ==1: discontinuity_flag=False 
        oldyear=year;oldmon=mon
          #assert len(ll) ==269, line
        zz = z 
          # loop over daily values
        count=21
        while(count < 266):
            num=ll[count:count+5]
            # evaluate daily value
            assert zz < data.size, line              
            if num[:lm] == self.missing: 
                pass # missing value; already pre-filled NaN;  or num[-1] == 'M'
            else:
              n = int(num)
              if n < self.varmin: warn("Encountered value '{:s}' below minimum in line (ignored):\n {:s}".format(num,line))
              elif n > self.varmax: warn("Encountered value '{:s}' above maximum in line (ignored):\n {:s}".format(num,line))
              else: data[zz] = n # only now, we can accept the value
            # increment count in line
            zz += 1 # here each month has 31 days (padded with missing values)
            count += 8 # here each month has 31 days (padded with missing values)
        if zz != z+31: raise ParseError('Line has {:d} values instead of 31:\n {:s}'.format(zz-z,line))  
        # increment counter
        z += 31
    #if existflag==False:tlen=0
    #if z < tlen: raise ParseError, 'Reached end of file before specified end date: {:s}'.format(self.filename)    
    f.close() # close again
    # return array
    return data
  

## class that defines variable properties (specifics are implemented in children)
class VarDef(RecordClass):
  # variable specific
  name        = '' # full variable name (used in source files)
  atts        = None # dictionary with GeoPy variable attributes
  prefix      = '' # file prefix for source file name (used with station ID)
  fileext     = '.dly' # file name extension (used for source files)
  # type specific
  datatype    = '' # defined in child class; type of source file
  recordunits = '' # units used for data  (in source files)
  units       = '' # units after conversion
  scalefactor = 1 # constant scaling factor for conversion 
  offset      = 0 # constant offset for conversion
  dtype       = 'float32' # data type used for data in source files
  encoding    = 'UTF-8' # file encoding (used in source files)
  missing     = '-9999' # string indicating missing value
  mflag       = ' BDHKLOPTW' # legal data flags (case sensitive; 'M' for missing should be screened earlier)
  qflag       = ' DGIKLMNORSTWXZ'
  sflag       = ' 067AaBbCEFGHIKMNQRrSsTUuWXZz'
  varmin      = 0. # smallest allowed value in data
  varmax      = 0. # largest allowed value in data
  # inferred variables
  variable    = '' # alias for name
  filepath    = '' # inferred from prefix
  
  def __init__(self, **kwargs):
    super(VarDef,self).__init__(**kwargs)
    self.variable = self.name
    self.filepath = '{0:s}{1:s}'.format('{:s}',self.fileext)
    
  def convert(self, data): 
    if self.scalefactor != 1: data *= self.scalefactor;
    if self.offset != 0: data += self.offset
    return data
  
  def getKWargs(self, *args):
    ''' Return a dictionary with the specified arguments and their values '''
    if len(args) == 0: 
      args = ['variable', 'units', 'varmin', 'varmax', 'missing', 'mflag', 'qflag', 'sflag', 'dtype', 'encoding']
    kwargs = dict()
    for arg in args:
      kwargs[arg] = getattr(self,arg)
    return kwargs
  
## variable definitions for GHCN datasets
  
# definition for precipitation files
class PrecipDef(VarDef):
  scalefactor = .1/(24.*60.*60.) # convert from mm/day to mm/s 
  offset      = 0 
  datatype    = 'precip'
  title       = 'GHCN Precipitation Records'
  varmin      = 0. # smallest allowed value in data
  varmax      = 1.e4 # largest allowed value in data

# daily precipitation variables in data
precip_vars = dict(precip=PrecipDef(name='PRCP', atts=varatts['precip']),
                   solprec=PrecipDef(name='SNOW', atts=varatts['solprec']))
# precipitation extremes (and other derived variables)
precip_xtrm = []
for threshold in precip_thresholds:
  precip_xtrm.append(dv.WetDays(threshold=threshold, ignoreNaN=True))
  precip_xtrm.append(dv.WetDayRain(threshold=threshold, ignoreNaN=True))
  precip_xtrm.append(dv.WetDayPrecip(threshold=threshold, ignoreNaN=True))
for var in precip_vars:
  for mode in ('min','max'):
    # ordinary & interval extrema: var, mode, [interval=7,] name=None, dimmap=None
    precip_xtrm.append(dict(var=var, mode=mode, klass=dv.Extrema))      
    precip_xtrm.append(dict(var=var, mode=mode, interval=5, klass=dv.MeanExtrema))      
# consecutive events: var, mode, threshold=0, name=None, long_name=None, dimmap=None
for threshold in precip_thresholds:
  suffix = '_{:03d}'.format(int(10*threshold)); name_suffix = '{:3.1f} mm/day)'.format(threshold)
  tmpatts = dict(var='precip', threshold=threshold/86400., klass=dv.ConsecutiveExtrema)
  precip_xtrm.append(dict(name='CWD'+suffix, mode='above', long_name='Consecutive Wet Days (>'+name_suffix, **tmpatts))
  precip_xtrm.append(dict(name='CDD'+suffix, mode='below', long_name='Consecutive Dry Days (<'+name_suffix, **tmpatts))

# definition for temperature files
class TempDef(VarDef):
  scalefactor = 0.1 
  offset      = 273.15 # convert to Kelvin 
  datatype    = 'temp'
  title       = 'GHCN Temperature Records'
  varmin      = -1000. # smallest allowed value in data
  varmax      = 1000. # largest allowed value in data

# daily temperature variables in data
temp_vars   = dict(T2=TempDef(name='TAVG', atts=varatts['T2']),
                   Tmax=TempDef(name='TMAX', atts=varatts['Tmax']),  
                   Tmin=TempDef(name='TMIN', atts=varatts['Tmin']),)
# temperature extremes (and other derived variables)
temp_xtrm   = [dv.FrostDays(threshold=0., temp='TAMIN', ignoreNaN=True), dv.SummerDays(threshold=25., temp='TAMAX'),]
for var in temp_vars:
  for mode in ('min','max'):
    # ordinary & interval extrema: var, mode, [interval=7,] name=None, dimmap=None
    temp_xtrm.append(dict(var=var, mode=mode, klass=dv.Extrema))      
    temp_xtrm.append(dict(var=var, mode=mode, interval=5, klass=dv.MeanExtrema))      
# consecutive events: var, mode, threshold=0, name=None, long_name=None, dimmap=None
# tmpatts = dict(mode='below', threshold=273.15, klass=dv.ConsecutiveExtrema) # threshold after conversion
# temp_xtrm.append(dict(name='CFD', var='T2', long_name='Consecutive Frost Days', **tmpatts))
temp_xtrm.append(dict(name='CSD', var='Tmax', mode='above', threshold=273.15+25., 
                      long_name='Consecutive Summer Days (>25C)', klass=dv.ConsecutiveExtrema))
temp_xtrm.append(dict(name='CFD', var='Tmin', mode='below', threshold=273.15, 
                      long_name='Consecutive Frost Days (< 0C)', klass=dv.ConsecutiveExtrema))
# map from common variable names to WRF names (which are used in the derived_variables module)
ghcn_varmap = dict(RAIN='precip', south_north='time', time='station', west_east=None, # swap order of axes                 
                 TAMIN='Tmin', TAMAX='Tmax', FrostDays='frzfrq', SummerDays='sumfrq') 
for threshold in precip_thresholds:
  suffix = '_{:03d}'.format(int(10*threshold))
  ghcn_varmap['WetDays'+suffix] = 'wetfrq'+suffix
  ghcn_varmap['WetDayRain'+suffix] ='dryprec'+suffix
  ghcn_varmap['WetDayPrecip'+suffix] ='wetprec'+suffix

# combine temperature and precip variables
all_vars = precip_vars.copy(); all_vars.update(temp_vars)
all_xtrm = precip_xtrm + temp_xtrm
# definition of station meta data format 
ghcn_station_format = tuple([('id', str),                           
                            ('lat', float),
                            ('lon', float),
                            ('alt', float), 
                            ('name', str), # the name variable can have multiple words
                            ('begin_year', int), # these are processed later    
                            ('begin_mon', int),
                            ('end_year', int),   
                            ('end_mon', int),])                               
#TODO: We should probably add the columns/field numbers corresponding to each variable and parse based 
#      on column/field number, rather than separating by white space; the begin- and end-years can be 
#      added to the station definition independently, since here they are not read from the file.
#      In its current form the station name is cut off after the first word (e.g. 'ST' for 'ST JOHNS').
    
## class to read station records and return a dataset
class StationRecords(object):
  '''
    A class that provides methods to load station data and associated meta data from files of a given format;
    The format itself will be defines in child classes.
    The data will be converted to monthly statistics and accessible as a GeoPy dataset or can be written to a NetCDF file.
  '''
  # arguments
  folder      = root_folder # root folder for station data: interval and datatype (source folder)
  stationfile = 'ghcnd-stations.txt' # file that contains station meta data (to load station records)
  encoding    = '' # encoding of station file
  interval    = '' # source data interval (currently only daily)
  datatype    = 'all' # GHCN does not seperate different data types
  title       = 'GHCN Station Data' # dataset title
  variables   = None # parameters and definitions associated with variables
  extremes    = None # list of derived/extreme variables to be computed as well
  atts        = None # attributes of resulting dataset (including name and title)
  header_format  = '' # station format definition (for validation)
  station_format = '' # station format definition (for reading)
  constraints    = None # constraints to limit the number of stations that are loaded
  # internal variables
  stationlists   = None # list of station objects
  dataset        = None # GeoPy Dataset (will hold results) 
  
  def __init__(self, folder=root_folder, stationfile='ghcnd-stations.txt', variables=None, extremes=None, interval='daily', 
               encoding='', header_format=None, station_format=None, constraints=None, atts=None, varmap=None):
    ''' Parse station file and initialize station records. '''
    # some input checks
    if not isinstance(stationfile,str): raise TypeError
    if interval != 'daily': raise NotImplementedError
    if station_format is None: station_format = ghcn_station_format # default    
    elif not isinstance(station_format,(tuple,list)): raise TypeError
    if not isinstance(constraints,dict) and constraints is not None: raise TypeError
    # variables et al.
    if not isinstance(variables,dict): raise TypeError
    extremes = deepcopy(all_xtrm) # no datatype distinction
    # N.B.: need to use deepcopy, because we are modifying the objects      
    if not isinstance(extremes,(list,tuple)): raise TypeError
    if varmap is None: varmap = ghcn_varmap
    elif not isinstance(varmap, dict): raise TypeError
    # utils
    encoding = encoding or list(variables.values())[0].encoding 
    if atts is None: atts = dict(name=self.datatype, title=self.title) # default name
    elif not isinstance(atts,dict): raise TypeError # resulting dataset attributes
    if not isinstance(encoding,str): raise TypeError
    folder = folder #or '{:s}/{:s}_{:s}/'.format(root_folder,interval,datatype) # default folder scheme 
    if not isinstance(folder,str): raise TypeError
    # save arguments
    self.folder = folder
    self.stationfile = stationfile
    self.encoding = encoding
    self.interval = interval
    self.variables = variables
    self.extremes = extremes
    self.varmap = varmap
    self.ravmap = dict((value,key) for key,value in varmap.items() if value is not None) # reverse var map
    self.atts = atts
    self.header_format = header_format
    self.station_format = station_format
    self.constraints = constraints
    ## initialize station objects from file
    # open and parse station file
    stationfile = '{:s}/{:s}'.format(folder,stationfile)
    f = codecs.open(stationfile, 'r', encoding=encoding)
    # initialize station list
    self.stationlists = {varname:[] for varname in variables.keys()} # a separate list for each variable
    z = 0 # row counter 
    ns = 0 # station counter
    # loop over lines (each defines a station)
    for line in f:
      z += 1 # increment counter
      collist=line[0:38].split()
      collist.append(line[41:71])
      lat=float(collist[1])
      lon=float(collist[2])
      if lat>5 and lat<49 and lon>55 and lon<135:#Tibet lat and lon range
#XXX: this needs to be revised based on column/field number, not white space delimiters
        stdef = dict() # station specific arguments to instantiate station object
        # loop over column titles
        zz = 0 # column counter
        for key,fct in station_format[0:5]: # loop over columns
          stdef[key] = fct(collist[zz]) # convert value and assign to argument
          zz += 1 # increment column
        assert zz <= len(collist) # not done yet
        for key,fct in station_format[-4:]:
          stdef[key]=fct('0')
        if constraints is None: ladd = True
        else:
          ladd = True
          for key,val in constraints.items():
            if stdef[key] not in val: ladd = False
        # instantiate station objects for each variable and append to lists
        if ladd:
          ns += 1
          # loop over variable definitions
          for varname,vardef in variables.items():
            filename = '{0:s}/ghcnd_all/{1:s}.dly'.format(folder, format(stdef['id']))
            kwargs = dict() # combine station and variable attributes
            kwargs.update(stdef); kwargs.update(vardef.getKWargs())
            station = DailyStationRecord(filename=filename, **kwargs)
            station.checkHeader() 
            self.stationlists[varname].append(station)
            begin_end_dates=[station.begin_year,station.begin_mon,station.end_year,station.end_mon]
            count=0
            for key,fct in station_format[-4:]:
              stdef[key]=fct(str(begin_end_dates[count]))
              count+=1
#XXX: we don't really need to use the station format to add these...

    assert len(self.stationlists[varname]) == ns # make sure we got all (lists should have the same length)
    
  def prepareDataset(self, filename=None, folder=None, station_folder=None):
    ''' prepare a GeoPy dataset for the station data (with all the meta data); 
        create a NetCDF file for monthly data; also add derived variables          '''
    if folder is None: folder = avgfolder # default folder scheme 
    elif not isinstance(folder,str): raise TypeError(folder)
    if station_folder is None: station_folder = '{:s}/ghcnd_all/'.format(root_folder) # default folder scheme 
    elif not isinstance(station_folder,str): raise TypeError(station_folder)
    if filename is None: filename = 'ghcn{:s}_monthly.nc'.format(self.datatype)# default folder scheme 
    elif not isinstance(filename,str): raise TypeError(filename)
    # meta data arrays
    dataset = Dataset(atts=self.atts)
    # station axis (by ordinal number)
    stationlist = list(self.stationlists.values())[0] # just use first list, since meta data is the same
    assert all([len(stationlist) == len(stnlst) for stnlst in list(self.stationlists.values())]) # make sure none is missing
    station = Axis(coord=np.arange(1,len(stationlist)+1, dtype='int'), **varatts['station']) # start at 1
    # station name
    namelen = max([len(stn.name) for stn in stationlist])
    strarray = np.array([stn.name.ljust(namelen) for stn in stationlist], dtype='|S{:d}'.format(namelen))
    dataset += Variable(axes=(station,), data=strarray, **varatts['name'])
    # geo locators (lat/lon/alt)
    for coord in ('lat','lon','alt'):
      coordarray = np.array([getattr(stn,coord) for stn in stationlist], dtype='float32') # single precision float
      dataset += Variable(axes=(station,), data=coordarray, **varatts[coord])
    # start/end dates (month relative to 1980-01)
    begin_year = np.zeros(len(stationlist), dtype='int16');
    begin_mon  = np.zeros(len(stationlist), dtype='int16');
    end_year   = np.zeros(len(stationlist), dtype='int16');
    end_mon    = np.zeros(len(stationlist), dtype='int16');
    datearray  = np.empty([2,len(stationlist)], dtype='int16');
    pntcounter = 0
    print(self.variables)
    for pnt in ('begin','end'):
      s = 0 #station counter
      for stations in stationlist:
        filenamedly = '{0:s}/{1:s}.dly'.format(station_folder,stations.id)
        # open file
        f = codecs.open(filenamedly, 'r','UTF-8')
        infoline=f.readlines()
        ll=infoline[0]
        begin_year[s] = int(ll[11:15])
        begin_mon[s] = int(ll[15:17])
        off=-270
        f.seek(off, 2)
        infoline = f.readlines()
        if len(infoline)>=2:
          raise ParseError('last line incomplete')
        else:
          ll=infoline[-1]
          end_year[s] = int(ll[11:15])
          end_mon[s] = int(ll[15:17])
        f.close()     
        datearray[0,s] = ( begin_year[s] - 1980 )*12 + begin_mon[s] - 1  # compute month relative to 1980-01
        datearray[1,s] = ( end_year[s] - 1980 )*12 + end_mon[s] - 1  # compute month relative to 1980-01
        s+=1
      dataset += Variable(axes=(station,), data=datearray[pntcounter], **varatts[pnt+'_date'])
      pntcounter+=1   
      # save bounds to determine size of time dimension
      if pnt == 'begin':
        begin_date = np.min(datearray)
        if begin_date%12: begin_date = (begin_date//12)*12 # always rounds down, no correction necessary 
      elif pnt == 'end':
        end_date = np.max(datearray)
        if end_date%12: end_date = (end_date//12+1)*12 # exclusive, i.e. this value is not included in range
    assert begin_date%12 == 0 and end_date%12 == 0
    # actual length of record (number of valid data points per station; filled in later)
    dataset += Variable(axes=(station,), data=np.zeros(len(station), dtype='int16'),  **varatts['stn_rec_len'])
    # add variables for monthly values
    time = Axis(coord=np.arange(begin_date, end_date, dtype='int16'), **varatts['time'])
    # loop over variables
    for varname,vardef in self.variables.items():
      # add actual variables
      dataset += Variable(axes=(station,time), dtype=vardef.dtype, **vardef.atts)
      # add length of record variable
      tmpatts = varatts['stn_rec_len'].copy(); recatts = tmpatts['atts'].copy()
      recatts['long_name'] = recatts['long_name']+' for {:s}'.format(varname.title())
      tmpatts['name'] = 'stn_'+varname+'_len'; tmpatts['atts'] = recatts
      dataset += Variable(axes=(station,), data=np.zeros(len(station), dtype='int16'),  **tmpatts)
    # write dataset to file
    ncfile = '{:s}/{:s}'.format(folder,filename)      
    #zlib = dict(chunksizes=dict(station=len(station))) # compression settings; probably OK as is 
    ncset = writeNetCDF(dataset, ncfile, feedback=False, overwrite=True, writeData=True, 
                        skipUnloaded=True, close=False, zlib=True)
    # add derived variables
    extremes = []
    for xvar in self.extremes:
      if isinstance(xvar,dict):
        var = xvar.pop('var'); mode = xvar.pop('mode'); Klass = xvar.pop('klass')
        xvar = Klass(ncset.variables[var], mode, ignoreNaN=True, **xvar)
        xvar.prerequisites = [self.ravmap.get(varname,varname) for varname in xvar.prerequisites]
      elif isinstance(xvar,dv.DerivedVariable):
        xvar.axes = tuple([self.varmap.get(varname,varname) for varname in xvar.axes if self.varmap.get(varname,varname)])
      else: raise TypeError
      # adapt variable instances for this dataset (i.e. axes and dependencies)
      xvar.normalize = False # different aggregation
      # check axes
      if len(xvar.axes) != 2 or xvar.axes != (varatts['station']['name'], varatts['time']['name']):
        raise dv.DerivedVariableError("Axes ('station', 'time') are required; adjust varmap as needed.")
      # finalize
      xvar.checkPrerequisites(ncset, const=None, varmap=self.varmap)
      if xvar.name in self.varmap: xvar.name = self.varmap[xvar.name] # rename
      xvar.createVariable(ncset)
      extremes.append(xvar)
    self.extremes = extremes
    # reopen netcdf file with netcdf dataset
    self.dataset = DatasetNetCDF(dataset=ncset, mode='rw', load=True) # always need to specify mode manually
    
  def readStationData(self):
    ''' read station data from source files and store in dataset '''
    assert self.dataset
    # determine record begin and end indices
    all_begin = self.dataset.time.coord[0] # coordinate value of first time step
    begin_idx = ( self.dataset.stn_begin_date.getArray() - all_begin ) * 31.
    end_idx = ( self.dataset.stn_end_date.getArray() - all_begin + 1 ) * 31.
    # loop over variables
    dailydata = dict() # to store daily data for derived variables
    monlydata = dict() # monthly data, but transposed
    ravmap = self.ravmap # shortcut for convenience  
    print(("\n   ***   Preparing {:s}   ***\n   Constraints: {:s}\n".format(self.title,str(self.constraints))))
    for var,vardef in self.variables.items():
      print(("\n {:s} ('{:s}'):\n".format(vardef.name.title(),var)))
      varobj = self.dataset[var] # get variable object
      wrfvar = ravmap.get(varobj.name,varobj.name)
      # allocate array
      shape = (varobj.shape[0], varobj.shape[1]*31) # daily data!
      dailytmp = np.empty(shape, dtype=varobj.dtype); dailytmp.fill(np.NaN) # initialize all with NaN
      # loop over stations
      s = 0 # station counter
      for station in self.stationlists[var]:
        print(("   {:<15s} {:s}".format(station.name,station.filename)))
        # read station file
        dailytmp[s,begin_idx[s]:end_idx[s]] = station.parseRecord()  
        s += 1 # next station
      assert s == varobj.shape[0]
      dailytmp = vardef.convert(dailytmp) # apply conversion function
      # compute monthly average
      dailytmp = dailytmp.reshape(varobj.shape+(31,))
      monlytmp = np.nanmean(dailytmp,axis=-1) # squeezes automatically
      # store daily and monthly data for computation of derived variables
      dailydata[wrfvar] = dailytmp
      monlydata[wrfvar] = monlytmp
      # load data
      varobj.load(monlytmp) # varobj.sync()
      del dailytmp, monlytmp
    # loop over derived nonlinear variables/extremes
    if any(not var.linear for var in self.extremes): print('\n computing (nonlinear) daily variables:')
    for var in self.extremes:      
      if not var.linear:
        print(("   {:<15s} {:s}".format(var.name,str(tuple(self.varmap.get(varname,varname) for varname in var.prerequisites)))))
        varobj = self.dataset[var.name] # get variable object
        if var.name not in ravmap: ravmap[var.name] = var.name # naming convention for tmp storage 
        wrfvar = ravmap[var.name] 
        # allocate memory for monthly values
        tmp = np.ma.empty(varobj.shape, dtype=varobj.dtype); tmp.fill(np.NaN) 
        # N.B.: some derived variable types may return masked arrays
        monlydata[wrfvar] = tmp
    # loop over time steps      
    tmpvars = dict()
    for m,mon in enumerate(varobj.axes[1].coord):
      # figure out length of month
      if mon%12 == 1: # February
        if calendar.isleap(1980 + mon/12): lmon = 29
        else: lmon = 28
      else: lmon = days_per_month[mon%12]
      # construct arrays for this month
      tmpdata = {varname:data[:,m,0:lmon] for varname,data in dailydata.items()}      
      for var in self.extremes:      
        if not var.linear:
          varobj = self.dataset[var.name] # get variable object
          wrfvar = ravmap[var.name]
          dailytmp = var.computeValues(tmpdata, aggax=1, delta=86400., tmp=tmpvars)        
          tmpdata[wrfvar] = dailytmp
          monlytmp = var.aggregateValues(dailytmp, aggdata=None, aggax=1) # last axis
          assert monlytmp.shape == (s,)
          monlydata[wrfvar][:,m] = monlytmp
    # loop over linear derived variables/extremes
    if any(var.linear for var in self.extremes): print('\n computing (linear) monthly variables:')
    for var in self.extremes:      
      varobj = self.dataset[var.name] # get variable object
      wrfvar = ravmap[var.name]
      if var.linear:
        # compute from available monthly data
        print(("   {:<15s} {:s}".format(var.name,str(tuple(self.varmap.get(varname,varname) for varname in var.prerequisites)))))
        monlytmp = var.computeValues(monlydata, aggax=1, delta=86400.)
        monlydata[wrfvar] = monlytmp
      tmpload = monlydata[wrfvar]
      assert varobj.shape == tmpload.shape
      varobj.load(tmpload)
    # determine actual length of records (valid data points)
    minlen = None 
    for varname in self.variables.keys():
      rec_len = self.dataset['stn_'+varname+'_len'] # get variable object
      varobj = self.dataset[varname]
      stlen,tlen = varobj.shape
      assert stlen == rec_len.shape[0]
      tmp = tlen - np.isnan(varobj.getArray(unmask=True, fillValue=np.NaN)).sum(axis=1)
      tmp = np.asanyarray(tmp, dtype=rec_len.dtype)
      assert tmp.shape == rec_len.shape
      rec_len.load(tmp) 
      if minlen is None: minlen = tmp
      else: minlen = np.minimum(minlen,tmp)
    # save minimum as overall record length
    self.dataset['stn_rec_len'].load(minlen)
    # synchronize data, i.e. write to disk
    self.dataset.sync()
    
    
## load pre-processed GHCN station time-series
def loadGHCN_TS(name=None, filetype='all', varlist=None, varatts=None, 
              filelist=None, folder=None, **kwargs): 
  ''' Load a monthly time-series of pre-processed GHCN station data. '''
  if filetype != 'all': raise NotImplementedError(filetype)
  if name is None: name = 'GHCN'
  if folder is None: folder = avgfolder
  if filelist is None: filelist = [tsfile.format(filetype)]
  # open NetCDF file (name, varlist, and varatts are passed on directly)
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, 
                            multifile=False, ncformat='NETCDF4', **kwargs)
  return dataset

## wrapper for loadGHCN_TS with additional functionality
def loadGHCN_StnTS(name=None, varlist=None, varatts=varatts, lloadCRU=False, filetype='all', **kwargs):
  ''' Load a monthly time-series of pre-processed GHCN station data. '''
  if filetype != 'all': raise NotImplementedError(filetype)
  
  if varlist is not None: 
    if isinstance(varlist,str): varlist = [varlist]
    varlist = list(set(varlist).union(stn_params)) 
  # load station data
  #print varlist  
  dataset = loadGHCN_TS(name=name, varlist=varlist, varatts=varatts, **kwargs) # just an alias
  # make sure we have a time-dependent variable
  if not dataset.hasAxis('time'):  
    #raise DatasetError, "No time-dependent variables in Dataset:\n{:s}".format(str(dataset)) # error, if no time-dependent variable
    dataset = loadGHCN_TS(name=name, varlist=varlist+['precip','T'], 
                        varatts=varatts, **kwargs) # just an alias
    # N.B.: for some operations we need a time axis...
    
  # supplement with CRU gridded data, if necessary
  if lloadCRU and varlist and any(var not in dataset for var in varlist):
    dataset.load() # not much data anyway..
    crulist = [var for var in varlist if ( var not in dataset and var in CRU_vars )]
    #print crulist
    if len(crulist) > 0:
      cru = loadCRU_StnTS(station='ghcn{}'.format(filetype), varlist=crulist).load() # need to load for slicing
      if cru.hasAxis('time'): # skip, if no time-dependent variable
        #print dataset
        cru = cru(time=dataset.time.limits()) # slice to same length
        dataset = dataset(time=cru.time.limits()) # slice to same length
        for varname in crulist: 
          dataset += cru[varname] # add auxiliary variables
      else: raise AxisError("Time-dependent variables not found in fall-bak dataset (CRU).")
  return dataset

## load pre-processed EC station climatology
def loadGHCN(): 
  ''' Load a pre-processed GHCN station climatology. '''
  return NotImplementedError
loadGHCN_Stn = loadGHCN


## some helper functions to test conditions
# defined in module main to facilitate pickling
#def test_prov(val,index,dataset,axis):
#  ''' check if station province is in provided list ''' 
#  return dataset.stn_prov[index] in val
def test_begin(val,index,dataset,axis):
  ''' check if station record begins before given year ''' 
  return dataset.stn_begin_date[index] <= val # converted to month beforehand 
def test_end(val,index,dataset,axis):
  ''' check if station record ends after given year ''' 
  return dataset.stn_end_date[index] >= val # converted to month beforehand 
def test_minlen(val,index,dataset,axis):
  ''' check if station record is longer than a minimum period ''' 
  return dataset.stn_rec_len[index] >= val 
def test_maxzse(val,index,dataset,axis, lcheckVar=True):
  ''' check that station elevation error does not exceed a threshold ''' 
  if not dataset.hasVariable('zs_err'):
    if lcheckVar: raise DatasetError
    else: return True # EC datasets don't have this field...
  else: return np.abs(dataset.zs_err[index]) <= val
def test_maxz(val,index,dataset,axis, lcheckVar=True):
  ''' check that station elevation does not exceed a threshold ''' 
  if not dataset.hasVariable('stn_zs'):
    if lcheckVar: raise DatasetError
    else: return True # EC datasets don't have this field...
  else: return np.abs(dataset.stn_zs[index]) <= val
def test_lat(val,index,dataset,axis):
  ''' check if station is located within selected latitude band '''
  return val[0] <= dataset.stn_lat[index] <= val[1] 
def test_lon(val,index,dataset,axis):
  ''' check if station is located within selected longitude band ''' 
  return val[0] <= dataset.stn_lon[index] <= val[1] 
def test_cluster(val,index,dataset,axis, cluster_name='cluster_id', lcheckVar=True):
  ''' check if station is located within selected longitude band '''
  if not dataset.hasVariable(cluster_name):
    if lcheckVar: raise DatasetError
    else: return True # most datasets don't have this field...
  elif isinstance(val, (int,np.integer)): 
    return dataset[cluster_name][index] == val
  elif isinstance(val, (tuple,list,np.ndarray)):
    return dataset[cluster_name][index] in val
  else: ValueError, val
# apply tests to list
def apply_test_suite(tests, index, dataset, axis):
  ''' apply an entire test suite to '''
  # just call all individual tests for given index
  results = [test(index,dataset,axis) for test in tests]
  results = [np.all(res) if isinstance(res,np.ndarray) else res for res in results]  
  return all(results)

## select a set of common stations for an ensemble, based on certain conditions
def selectStations(datasets, stnaxis='station', master=None, linplace=False, lall=False, 
                  lcheckVar=False, cluster_name='cluster_id', **kwcond):
  ''' A wrapper for selectCoords that selects stations based on common criteria '''
  if linplace: raise NotImplementedError("Option 'linplace' does not work currently.")
  # pre-load NetCDF datasets
  for dataset in datasets: 
    if isinstance(dataset,DatasetNetCDF): dataset.load()
    if dataset.station_name.ndim > 1 and not dataset.station_name.hasAxis(stnaxis):
      raise DatasetError("Meta-data fields must only have a 'station' axis and no other!") 
  # list of possible constraints
  tests = [] # a list of tests to run on each station
  #loadlist =  (datasets[imaster],) if not lall and imaster is not None else datasets 
  # test definition
  varcheck = [True]*len(datasets)
  for key,val in kwcond.items():
    key = key.lower()
    if key == 'min_len':
      varname = 'stn_rec_len'
      if not isNumber(val): raise TypeError
      val = val*12 # units in dataset are month  
      tests.append(functools.partial(test_minlen, val))    
    elif key == 'begin_before':
      varname = 'stn_begin_date'
      if not isNumber(val): raise TypeError
      val = (val-1980.)*12. # units in dataset are month since Jan 1980  
      tests.append(functools.partial(test_begin, val))    
    elif key == 'end_after':
      varname = 'stn_end_date'
      if not isNumber(val): raise TypeError
      val = (val-1980.)*12. # units in dataset are month since Jan 1980  
      tests.append(functools.partial(test_end, val))    
    elif key == 'max_zerr':
      varname = 'zs_err'
      if not isNumber(val): raise TypeError  
      tests.append(functools.partial(test_maxzse, val, lcheckVar=lcheckVar))
    elif key == 'max_z':
      varname = 'stn_zs'
      if not isNumber(val): raise TypeError  
      tests.append(functools.partial(test_maxz, val, lcheckVar=lcheckVar))
    elif key == 'lat':
      varname = 'stn_lat'
      if not isinstance(val,(list,tuple)) or len(val) != 2 or not all(isNumber(l) for l in val): raise TypeError  
      tests.append(functools.partial(test_lat, val))
    elif key == 'lon':
      varname = 'stn_lon'
      if not isinstance(val,(list,tuple)) or len(val) != 2 or not all(isNumber(l) for l in val): raise TypeError  
      tests.append(functools.partial(test_lon, val))
    elif key == 'cluster':
      varname = cluster_name
      if ( not isinstance(val,(list,tuple,np.ndarray)) or not all(isInt(l) for l in val)) and not isInt(val): raise TypeError  
      tests.append(functools.partial(test_cluster, val, cluster_name=cluster_name, lcheckVar=lcheckVar))
    else:
      raise NotImplementedError("Unknown condition/test: '{:s}'".format(key))
    # record, which datasets have all variables 
    varcheck = [dataset.hasVariable(varname) and vchk for dataset,vchk in zip(datasets,varcheck)]
  if not all(varcheck): 
    if lall and lcheckVar: raise DatasetError(varcheck)
    else: warn("Some Datasets do not have all variables: {:s}".format(varcheck))
  # define test function (all tests must pass)
  if len(tests) > 0:
    testFct = functools.partial(apply_test_suite, tests)
  else: testFct = None
  # pass on call to generic function selectCoords
  datasets = selectElements(datasets=datasets, axis=stnaxis, testFct=testFct, master=master, linplace=linplace, lall=lall)
  # return sliced datasets
  return datasets
  
## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = orig_ts_file # filename pattern: variable name and resolution
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: variable name and resolution
data_folder = avgfolder # folder for user data
grid_def = None # no grid here...
LTM_grids = [] 
TS_grids = ['']
grid_res = {}
default_grid = None
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = None # time-series data
loadClimatology = None # pre-processed, standardized climatology
loadStationTimeSeries = loadGHCN_StnTS # time-series data
loadStationClimatology = None # pre-processed, standardized climatology

if __name__ == '__main__':

#   mode = 'test_selection'
#   mode = 'test_timeseries'
#   mode = 'test_station_object'
#   mode = 'test_station_reader'
#   mode = 'test_conversion'
  mode = 'convert_all_stations'
  
  if mode == 'test_selection':
    
    # some foreign imports
    from geodata.base import Ensemble
    from datasets.WRF import loadWRF_StnEns
    # load pre-processed time-series file
    stn='ghcnprecip'
    print('')
    stnens = Ensemble(loadGHCN_StnTS(station=stn), loadWRF_StnEns(ensemble='max-ens-2100', station=stn, 
                      filetypes='hydro', domains=2)) # including WRF data for test
    print((stnens[0]))    
    print('')
    var = stnens[-1].axes['station']; print(''); print(var)
    for var in stnens.station: print((var.min(),var.mean(),var.max()))
    # test station selector
    cluster = (4,7,8); cluster_name = 'cluster_projection';  max_zserr = 300; lat = (40,50)
    min_len = 50; begin_before = 1985; end_after = 2000
#     stnens = selectStations(stnens, prov=prov, min_len=min_len, lat=lat, max_zerr=max_zserr, lcheckVar=True,
#                             stnaxis='station', imaster=1, linplace=False, lall=False); cluster = None
    stnens = selectStations(stnens, min_len=min_len, cluster=cluster, lat=lat, max_zerr=max_zserr,
                            begin_before=begin_before, end_after=end_after, cluster_name=cluster_name,
                            lcheckVar=False, stnaxis='station', master=None, linplace=False, lall=True)
    # N.B.: clusters effectively replace provinces, but currently only the EC datasets have that 
    #       information, hence they have to be master-set (imaster=0)
    print(stnens)    
    print('')
    var = stnens[-1].axes['station']; print(''); print(var)
    for var in stnens.station: print((var.min(),var.mean(),var.max()))
    
    print('')
    #print(stnens[0].stn_prov.data_array)
    print((stnens[0][cluster_name].data_array))
    for stn in stnens:
    # assert all(elt in prov for elt in stn.stn_prov.data_array)
      assert all(lat[0]<=elt<=lat[1] for elt in stn.stn_lat.data_array)
      assert all(min_len<=elt for elt in stn.stn_rec_len.data_array)
      if cluster is not None and cluster_name in stn: # only GHCN stations
        assert all(elt in cluster for elt in stn[cluster_name].data_array)
      if 'zs_err' in stn: # only WRF datasets
        assert all(elt < max_zserr for elt in stn.zs_err.data_array)
    
        
  # test wrapper function to load time series data from GHCN stations
  elif mode == 'test_timeseries':
    
    # load pre-processed time-series file
    print('')
    dataset = loadGHCN_TS(filetype='temp').load()
    #dataset = loadGHCN_StnTS(station='ghcnprecip', varlist=['solprec','T']).load()
    print(dataset)
    print('')
    print(('DHABI', dataset.station_name.findValues('DHABI')))
    print('')
    print((dataset.time))
    print((dataset.time.coord))
    print((dataset.stn_begin_date.min()))
    assert dataset.time.coord[0]%12. == 0
    assert (dataset.time.coord[-1]+1)%12. == 0
        
  # test station object initialization
  elif mode == 'test_station_object':  
    
    # initialize station (new way with VarDef)
    var = PrecipDef(name='PRCP', prefix='dt', atts=varatts['precip'])
    test = DailyStationRecord(id='BR002955013', name='ALEGRETE-ELETROSUL', filename=root_folder+'/ghcnd_all/BR002955013.dly',
                              begin_year=1986, begin_mon=2, end_year=1998, end_mon=4, 
                              lat=-29.78, lon=-59.77, alt=80, **var.getKWargs())
    test.checkHeader() # fail early...
    data = var.convert(test.parseRecord())    
    print(data.shape, data.dtype)
    print(np.nanmin(data), np.nanmean(data), np.nanmax(data))  
  
  # tests station reader initialization
  elif mode == 'test_station_reader':
    
    # prepare input
    test = StationRecords(folder=root_folder, variables=all_vars, stationfile='ghcnd-stations-debug.txt')
    # show dataset
    test.prepareDataset()
    print(test.dataset)
    print(test.dataset.filelist)
    print('')
    # test netcdf file
    dataset = DatasetNetCDF(filelist=[avgfolder+'ghcnall_monthly.nc'])
    print(dataset)
    print('')
    print(dataset.station_name[:]) # test string variable recall
    
  
  # tests entire conversion process
  elif mode == 'test_conversion':
    
    # initialize station record container (PE only has 3 stations - ideal for testing!)
    test = StationRecords(folder=root_folder, variables=all_vars, stationfile='ghcnd-stations-debug.txt')
    # create netcdf file
    print('')
    filename = tsfile.format(list(all_vars.values())[0].datatype)        
    test.prepareDataset(filename=filename, folder=None)
    # read actual station data
    test.readStationData()
    dataset = test.dataset
    print('')
    print(dataset)
    print('\n')
    precip_list = list(precip_vars.keys()) + precip_xtrm
    for varname,var in dataset.variables.items():
      if var.hasAxis('time') and var.hasAxis('station'):
        data = var.getArray()
        if var.units == 'kg/m^2/s': 
          data  *= 86400. 
          if np.all(data.mask):
            # some stations do not have all variables
            print(('{:>14s}:    NaN  |    NaN  |    NaN  [{:s}]'.format(var.name, var.units))) 
          else:
            print(('{:>14s}: {:5.2f} | {:5.2f} | {:5.2f} [{:s}]'.format(
                  var.name, np.nanmin(data), np.nanmean(data), np.nanmax(data), var.units))) 
        else: 
          print(('{:>14s}: {:5.1f} | {:5.1f} | {:5.1f} [{:s}]'.format(
                var.name, np.nanmin(data), np.nanmean(data), np.nanmax(data), var.units)))
    # record length
    for pfx in ['stn_rec',]: # +variables.keys() # currently the only one (all others are the same!)
      var = dataset[pfx+'_len']
      data = var.getArray()
      print(('{:>14s}: {:5d} | {:5.1f} | {:5d} [{:s}]'.format(var.name,np.min(data), np.mean(data), 
                                                      np.max(data), var.units)))
    
  
  # convert all station date to NetCDF
  elif mode == 'convert_all_stations':
    
    # loop over variable types
    #for variables in ( temp_vars,): # precip_vars, temp_vars, all_vars
    for variables in (all_vars,):
      
      # initialize station record container
      stations = StationRecords(variables=variables, constraints=None)
      # create netcdf file
      stations.prepareDataset(filename=None, folder=None) # default settings
      # read actual station data
      stations.readStationData()
