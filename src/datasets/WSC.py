'''
Created on 2014-02-12

A module to load station data from the Water Survey if Canada and associate the data with river basins;
the data is stored in human-readable text files and tables. 

@author: Andre R. Erler, GPL v3
'''

# external imports
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
from copy import deepcopy
import fileinput
# internal imports
from datasets.common import days_per_month, name_of_month, data_root
from geodata.misc import ParseError
from geodata.gdal import Shape
from geodata.station import StationDataset, Variable, Axis
# from geodata.misc import DatasetError
from warnings import warn

## WSC Meta-data

dataset_name = 'WSC'
root_folder = data_root + dataset_name + '/'

# variable attributes and name
variable_attributes = dict(discharge = dict(name='discharge', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Average Flow Rate')), # average flow rate
                           discmax = dict(name='discmax', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Maximum Flow Rate')), # maximum flow rate
                           discmin = dict(name='discmin', units='kg/s', fileunits='m^3/s', scalefactor=1000., atts=dict(long_name='Minimum Flow Rate')), # minimum flow rate
                           level = dict(name='level', units='m', atts=dict(long_name='Water Level'))) # water level
# list of variables to load
variable_list = variable_attributes.keys() # also includes coordinate fields    

# container class for stations and shape files
class Basin(Shape):
  ''' Just a container for basin information and associated station data '''
  def __init__(self, basin=None, subbasin=None, folder=None, shapefile=None, rivers=None, stations=None, load=False, ldebug=False):
    ''' save meta information; should be initialized from a BasinInfo instance '''
    # resolve input
    if isinstance(basin,(basestring,BasinInfo)):
      if isinstance(basin,basestring):
        if basin in basins: basin = basins[basin]
        else: raise ValueError, 'Unknown basin: {}'.format(basin)
      folder = basin.folder
      if subbasin is None: subbasin = basin.outline      
      elif not isinstance(subbasin,basestring): raise TypeError
      #shapefile = subbasin if subbasin[-4:] == '.shp' else subbasin + '.shp'
      if subbasin not in basin.shapefiles: raise ValueError, 'Unknown subbasin: {}'.format(subbasin)
    elif isinstance(folder,basestring) and isinstance(shapefile,basestring): pass
    else: raise TypeError, 'Specify either basin & station or folder & shapefile.'
    # call Shape constructor
    super(Basin,self).__init__(name=basin.name, long_name=basin.long_name, shapefile=basin.shapefiles[subbasin], load=load, ldebug=ldebug)
    # add info
    self.info = basin
    self.maingage = basin.maingage if basin is not None else None 
    
  def getMainGage(self, varlist=None, varatts=None, mode='climatology', filetype='monthly'):
    ''' return a dataset with data from the main gaging station '''
    if self.maingage is not None:
      station = loadGageStation(basin=self.info, varlist=varlist, varatts=varatts, mode=mode, filetype=filetype)
    else: station = None 
    return station

# a container class for basin meta data
class BasinInfo(object): 
  ''' basin meta data '''
  def __init__(self):
    ''' some common operations and inferences '''
    self.folder = root_folder+self.long_name+'/'; self.__doc__ = self.long_name
    shapefiles = OrderedDict()
    for shp in self.shapefiles:
      if shp[-4:] == '.shp':
	shapefiles[shp[:-4]] = self.folder + shp
      else: 
	shapefiles[shp] = self.folder + shp + '.shp'
    self.shapefiles = shapefiles 
    #self.shapefiles = [shp if shp[-4:] == '.shp' else shp+'.shp' for shp in self.shapefiles]
    self.outline = self.shapefiles.keys()[0]; self.maingage = self.stations[self.rivers[0]][0] if self.stations else None 
    self.stationfiles = dict()
    for river,stations in self.stations.items():
      for station in stations: 
        filename = '{0:s}_{1:}.dat'.format(river,station)
        if station in self.stationfiles: 
          warn('Duplicate station name: {}\n  {}\n  {}'.format(station,self.stationfiles[station],filename))
        else: self.stationfiles[station] = filename 
      
# meta data for specific basins
class ARB(BasinInfo):
  def __init__(self):
    self.name = 'ARB'; self.long_name = 'Athabasca River Basin'; self.rivers = ['Athabasca']
    self.stations = dict(Athabasca=['Embarras','McMurray'])
    self.shapefiles = ['WholeARB','LowerARB','LowerCentralARB','UpperCentralARB','UpperARB']        
    BasinInfo.__init__(self)
class FRB(BasinInfo):
  def __init__(self):
    self.name = 'FRB'; self.long_name = 'Fraser River Basin'; self.rivers = ['Fraser']
    self.stations = dict(Fraser=['PortMann','Mission'])
    self.shapefiles = ['WholeFRB','LowerFRB','UpperFRB']
    BasinInfo.__init__(self)
class PSB(BasinInfo):
  def __init__(self):
    self.name = 'PSB'; self.long_name = 'Pacific Seaboard'; self.rivers = []
    self.stations = dict()
    self.shapefiles = ['WholePSB','NorthernPSB','SouthernPSB']
    BasinInfo.__init__(self)
# dictionary with basin meta data
basins = dict(ARB=ARB(), FRB=FRB(), PSB=PSB())

## Functions that handle access to ASCII files

def loadGageStation(basin=None, station=None, varlist=None, varatts=None, mode='climatology', 
                    filetype='monthly', folder=None, filename=None):
  ''' Function to load hydrograph climatologies for a given basin '''
  # resolve input
  if isinstance(basin,(basestring,BasinInfo)):
    if isinstance(basin,basestring):
      if basin in basins: basin = basins[basin]
      else: raise ValueError, 'Unknown basin: {}'.format(basin)
    folder = basin.folder
    if station is None: station = basin.maingage      
    elif not isinstance(station,basestring): raise TypeError
    if station in basin.stationfiles: filename = basin.stationfiles[station]
    else: raise ValueError, 'Unknown station: {}'.format(station)
    river = filename.split('_')[0].lower()
    atts = dict(basin=basin.name, river=river) # first component of file name       
  elif isinstance(folder,basestring) and isinstance(filename,basestring):
    atts = None; river = None
  else: raise TypeError, 'Specify either basin & station or folder & filename.'
  # variable attributes
  if varlist is None: varlist = variable_list
  elif not isinstance(varlist,(list,tuple)): raise TypeError  
  if varatts is None: varatts = deepcopy(variable_attributes.copy()) # because of nested dicts
  elif not isinstance(varatts,dict): raise TypeError
  # create dataset for station
  dataset = StationDataset(name=station, title=filename.split('.')[0], ID=None, varlist=[], atts=atts) 
  if mode == 'climatology': 
    # make common time axis for climatology
    te = 12 # length of time axis: 12 month
    climAxis = Axis(name='time', units='month', length=12, data=np.arange(1,te+1,1)) # monthly climatology
  else: raise NotImplementedError, 'Currently only climatologies are supported.'
  dataset.addAxis(climAxis, copy=False)
  # a little helper function
  def makeVariable(varname):  
    atts = varatts[varname] # these are more specific than below (mean/min/max)
    data = np.asarray(linesplit[1:te+1], dtype='float')
    if 'scalefactor' in atts: data = data * atts.pop('scalefactor')
    if 'fileunits' in atts: atts.pop('fileunits') 
    #print varname, data.shape, len(climAxis)
    return Variable(axes=[climAxis], data=data, **atts)      
  # open namelist file for reading   
  filehandle = fileinput.FileInput(['{}/{}'.format(folder,filename)], mode='r')  
  # parse file and load variables
  l = 0; filevar = None; lmean = False; lreadVar = False; offset = 0
  for line in filehandle:
    l += 1 # count lines...
    linesplit = line.split()
    # blank line
    if len(linesplit) == 0:
      if filevar is None: 
        if offset == 0: l -= 1 # empty header lines
        else: offset += 1 # empty lines after header
      else:
        if not lreadVar or lmean: offset = l # go to next record
        else: raise ParseError, 'Data not found at line {}.'.format(l)    
    # first line of file
    elif l == 1:
      if basin is not None: # check
        fileriver = linesplit[0].lower()
        if fileriver != river: raise ParseError, 'Inconsistent river names: {} != {}'.format(fileriver,river)
      tmp = linesplit[-1]
      if tmp[0] == '(' and tmp[-1] == ')': 
        ID = tmp[1:-1]; print('Station ID: {}'.format(ID))
      else: 
        ID = None; warn('No station ID available.')
      dataset.ID = ID # set station ID
      offset = 1
    # first line of record
    elif l == 1 + offset:
      if filetype == 'monthly':
        if not linesplit[0].lower() == 'monthly': raise ParseError, 'Unknown filetype; not monthly.'
      filevar = linesplit[2].lower() # third field       
      if filevar in varatts and filevar in varlist: 
        lreadVar = True # read this variable
        tmpatts = varatts[filevar]
        units = tmpatts.pop('fileunits',tmpatts['units'])        
        fileunits = linesplit[-1].lower()
        if units == fileunits[1:-1]: pass
        elif fileunits[0] == '(' and fileunits[-1] == ')' and units == fileunits[1:-1]: pass         
        else: raise ParseError,'No compatible units found; expected {}, found {}.'.format(units,fileunits)
      else:
        lreadVar = False # skip this variables
        if filevar in varatts: print('Skipping variable: {}.'.format(filevar))  
        else: print('Unknown variable: {}.'.format(filevar))
    # third line
    elif l == 2 + offset:      
      if filetype == 'monthly':
        if not ( linesplit[0].lower() == 'year' and len(linesplit) == 14 ): 
          raise ParseError, 'Unknown file format; expected yearly rows.'
    # skip the entire time-series section
    # extract variables (min/max/mean are separate variables)
    # load mean
    elif l > 2 + offset and linesplit[0].lower() == 'mean' and lreadVar:
      if filevar == 'discharge': varname = filevar
      dataset.addVariable(makeVariable(varname), copy=False)
      lmean = True 
    # load maximum
    elif l > 2 + offset and linesplit[0].lower() == 'max' and lreadVar:
      if filevar == 'discharge': varname = 'discmax'
      dataset.addVariable(makeVariable(varname), copy=False) 
    # load minimum
    elif l > 2 + offset and linesplit[0].lower() == 'min' and lreadVar:
      if filevar == 'discharge': varname = 'discmin'
      dataset.addVariable(makeVariable(varname), copy=False)
  # return station dataset
  return dataset   


## abuse main block for testing
if __name__ == '__main__':
    
  # verify basin info
  ARB = basins['ARB']
  print ARB.long_name
  print ARB.stationfiles
  FRB = basins['FRB']
  print FRB.long_name
  print FRB.stationfiles
  
  # load station data
  station = loadGageStation(basin='ARB') # , station='Mission'
  station = loadGageStation(basin='FRB') # , station='Mission'
  print
  print station
  print
  print station.discharge.getArray()
    
