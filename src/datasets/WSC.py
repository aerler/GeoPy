'''
Created on 2014-02-12

A module to load station data from the Water Survey of Canada and associate the data with river basins;
the data is stored in human-readable text files and tables. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import functools
import fileinput
from collections import OrderedDict
from copy import deepcopy
# internal imports
from datasets.common import selectElements, data_root
from geodata.netcdf import DatasetNetCDF
from geodata.misc import FileError, isNumber
from geodata.gdal import NamedShape, ShapeInfo
from geodata.station import StationDataset, Variable, Axis
# from geodata.utils import DatasetError
from warnings import warn

## WSC (Water Survey Canada) Meta-data

dataset_name = 'WSC'
root_folder = data_root + dataset_name + '/'

# variable attributes and name
variable_attributes = dict(runoff = dict(name='runoff', units='kg/m^2/s', atts=dict(long_name='Average Runoff Rate')), # average flow rate
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
# list of variables to load
variable_list = variable_attributes.keys() # also includes coordinate fields    
# alternate WSC runoff variable names (for use with aggregation mode)
agg_varatts = dict()
for varname,varatts in variable_attributes.iteritems():
  tmpatts = varatts.copy() 
  if tmpatts['units'] == 'kg/m^2/s': tmpatts['name'] = 'runoff'  # always name 'runoff'
  elif tmpatts['units'] == 'kg/s': tmpatts['name'] = 'discharge'  # always name 'discharge'
  agg_varatts[varname] = tmpatts    

# custom exception for missing gage station data
class GageStationError(FileError):
  ''' Exception indicating that gage station data is missing '''
  pass

  
# container class for stations and area files
class Basin(NamedShape):
  ''' Just a container for basin information and associated station data '''
  def __init__(self, basin=None, subbasin=None, folder=None, shapefile=None, basins_dict=None, load=False, ldebug=False):
    ''' save meta information; should be initialized from a BasinInfo instance '''
    super(Basin,self).__init__(area=basin,  subarea=subbasin, folder=folder, shapefile=shapefile, shapes_dict=basins_dict, load=load, ldebug=ldebug)
    self.maingage = basin.maingage if basin is not None else None 
    
  def getMainGage(self, varlist=None, varatts=None, aggregation=None, mode='climatology', filetype='monthly'):
    ''' return a dataset with data from the main gaging station '''
    if self.maingage is not None:
      station = loadGageStation(basin=self.info, varlist=varlist, varatts=varatts, aggregation=aggregation, mode=mode, filetype=filetype)
    else: station = None 
    return station

# a container class for basin meta data
class BasinInfo(ShapeInfo): 
  ''' basin meta data '''
  def __init__(self, name=None, long_name=None, rivers=None, stations=None, subbasins=None, data_source=None, folder=None):
    ''' some common operations and inferences '''
    # call parent constructor 
    if folder is None: folder = root_folder + '/Basins/'
    super(BasinInfo,self).__init__(name=name, long_name=long_name, shapefiles=subbasins, shapetype='BSN', 
                                   data_source=data_source, folder=folder)
    # add basin specific stuff
    self.subbasins = subbasins
    self.maingage = stations[rivers[0]][0] if stations else None 
    self.stationfiles = dict()
    for river,station_list in stations.items():
      for station in station_list: 
        #filename = '{0:s}_{1:}.dat'.format(river,station)
        filename = '{0:s}_{1:}_Monthly.csv'.format(river,station)
        if station in self.stationfiles: 
          warn('Duplicate station name: {}\n  {}\n  {}'.format(station,self.stationfiles[station],filename))
        else: self.stationfiles[station] = filename
      
# dictionary with basin meta data
basins_info = OrderedDict() # maintain order
# meta data for specific basins

basins_info['AY'] = BasinInfo(name='AY', long_name='Alaska and Yukon', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholeAY'])
basins_info['AO'] = BasinInfo(name='AO', long_name='Arctic Ocean', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholeAO'])
basins_info['ARB'] = BasinInfo(name='ARB', long_name='Athabasca River Basin', rivers=['Athabasca'], data_source='WSC',
                               stations=dict(Athabasca=['Embarras','McMurray']),
                               subbasins=['WholeARB','UpperARB','LowerARB'])
basins_info['CRB'] = BasinInfo(name='CRB', long_name='Columbia River Basin', rivers=['Columbia'], data_source='WSC',
                               stations=dict(), subbasins=['WholeCRB'])
basins_info['FRB'] = BasinInfo(name='FRB', long_name='Fraser River Basin', rivers=['Fraser'], data_source='WSC',
                               stations=dict(Fraser=['PortMann','Mission']),
                               subbasins=['WholeFRB','UpperFRB','LowerFRB'])
basins_info['GLB'] = BasinInfo(name='GLB', long_name='Great Lakes Basin', rivers=['Upper Saint Lawrence'], data_source='WSC',
                               stations=dict(), subbasins=['WholeGLB'])
basins_info['GSL'] = BasinInfo(name='GSL', long_name='Great Slave Lake', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholeGSL'])
basins_info['LS'] = BasinInfo(name='LS', long_name='Labrador Sea', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholeLS'])
basins_info['MKB'] = BasinInfo(name='MKB', long_name='MacKenzie Basin', rivers=['MacKenzie'], data_source='',
                               stations=dict(), subbasins=['WholeMKB'])
basins_info['MRB'] = BasinInfo(name='MRB', long_name='Missouri River Basin', rivers=['Missouri'], data_source='WSC',
                               stations=dict(), subbasins=['WholeMRB'])
basins_info['NRB'] = BasinInfo(name='NRB', long_name='Nelson River Basin', rivers=['Nelson'], data_source='WSC',
                               stations=dict(), subbasins=['WholeNRB'])
basins_info['NHB'] = BasinInfo(name='NHB', long_name='Northern Hudson Bay', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholeNHB'])
basins_info['NO'] = BasinInfo(name='NO', long_name='Northern Ontario', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholeNO'])
basins_info['PO'] = BasinInfo(name='PO', long_name='Pacific Ocean', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholePO'])
basins_info['PSB'] = BasinInfo(name='PSB', long_name='Pacific Seaboard', rivers=[], data_source='WSC',
                               stations=dict(), subbasins=['WholePSB','NorthernPSB','SouthernPSB'])
basins_info['SLR'] = BasinInfo(name='SLR', long_name='Saint Lawrence River', rivers=['Saint Lawrence'], data_source='WSC',
                               stations=dict(), subbasins=['WholeSLR'])

# N.B.: all shapefiles from Water Survey of Canada

# dictionary of basins
basins = OrderedDict() # maintain order
for name,basin in basins_info.iteritems():
  # add main basin
  basins[basin.name] = Basin(basin=basin, subbasin=None)
  if len(basin.subbasins) > 1 :
    # preserve grouping
    for subbasin in basin.subbasins[1:]: # skip first
      basins[subbasin] = Basin(basin=basin, subbasin=subbasin)
    
# get hydrographs from WSC here: https://wateroffice.ec.gc.ca/search/search_e.html?sType=h2oArc

## Functions that handle access to ASCII files
def loadGageStation(basin=None, station=None, varlist=None, varatts=None, mode='climatology', 
                    aggregation=None, filetype='monthly', folder=None, filename=None):
  ''' Function to load hydrograph climatologies for a given basin '''
  # resolve input
  if isinstance(basin,(basestring,BasinInfo)):
    if isinstance(basin,basestring):
      if basin in basins: basin = basins_info[basin]
      else: raise ValueError, 'Unknown basin: {}'.format(basin)
    folder = basin.folder
    if station is None: station = basin.maingage      
    elif not isinstance(station,basestring): raise TypeError
    if station in basin.stationfiles: filename = basin.stationfiles[station]
    else: raise GageStationError, 'Unknown station: {}'.format(station)
    river = filename.split('_')[0].lower()
    atts = dict(basin=basin.name, river=river) # first component of file name       
  elif isinstance(folder,basestring) and isinstance(filename,basestring):
    atts = None; river = None
  else: raise TypeError, 'Specify either basin & station or folder & filename.'
  # variable attributes
  if varlist is None: varlist = variable_list
  elif not isinstance(varlist,(list,tuple)): raise TypeError  
  varlist = list(varlist) # make copy of varlist to avoid interference
  if varatts is None: 
    if aggregation is None: varatts = deepcopy(variable_attributes) # because of nested dicts
    else: varatts = deepcopy(agg_varatts) # because of nested dicts
  elif not isinstance(varatts,dict): raise TypeError
  ## read csv data
  filepath = '{}/{}'.format(folder,filename)
  data = np.genfromtxt(filepath, dtype=np.float32, delimiter=',', skip_header=1, filling_values=np.nan,  
                       usecols=np.arange(4,28,2), usemask=True, loose=True, invalid_raise=True)
  # for some reason every value if followed by an extra comma...
  data = np.ma.masked_less(data, 10) # remove some invalid values
  data *= 1000. # m^3 == 1000 kg (water)
  ## load meta data
  # open namelist file for reading   
  filehandle = fileinput.FileInput(['{}/{}'.format(folder,filename.replace('Monthly', 'Metadata'))], mode='r')  
  # parse file and load data into a dictionary
  l = 0
  for line in filehandle:
    linesplit = line.split(',')
    if l == 0: keys = linesplit
    elif l == 1: values = linesplit
    else: raise IOError, line
    l += 1 # count lines...
  assert len(keys) == len(values)
  metadata = {key:value for key,value in zip(keys,values)}
  # create dataset for station
  atts['long_name'] = metadata['Station Name']
  atts['ID'] = metadata['Station Number']
  atts['shape_name'] = basin.name if basin is not None else None
  atts['shp_area'] = float(metadata['Drainage Area']) * 1e6 # km^2 == 1e6 m^2
  metadata.update(atts)
  dataset = StationDataset(name='WSC', title=metadata['Station Name'], varlist=[], atts=metadata,) 
  if mode == 'climatology': 
    # make common time axis for climatology
    te = 12 # length of time axis: 12 month
    climAxis = Axis(name='time', units='month', length=12, coord=np.arange(1,te+1,1)) # monthly climatology
  else: raise NotImplementedError, 'Currently only climatologies are supported.'
  dataset.addAxis(climAxis, copy=False)
  # extract variables (min/max/mean are separate variables)
  doa = data / atts['shp_area']
  from utils import nanfunctions as nf
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
  # return station dataset
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
  for key,val in kwcond.iteritems():
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
      raise NotImplementedError, "Unknown condition/test: '{:s}'".format(key)
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
  
  basin_name = 'ARB'
    
  # verify basin info
  basin_info = basins_info[basin_name]
  print basin_info.long_name
  print basin_info.stationfiles
  
  # load basins
  basin = basins[basin_name]
  print basin.long_name
  print basin
  assert basin.info == basin_info
  assert basin.shapetype == 'BSN'
  
  # load station data
  station = basin.getMainGage(aggregation='std')
  print
  print station
  print
  assert station.ID == loadGageStation(basin=basin_name).ID
  print station.discharge.getArray()
  
  # print basins
  print
  for bsn in basins.iterkeys():
    print bsn