'''
Created on 2014-02-12

A module to load station data from the Water Survey if Canada and associate the data with river basins;
the data is stored in human-readable text files and tables. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
# internal imports
from datasets.common import days_per_month, name_of_month, data_root
from geodata.gdal import Shape
# from geodata.misc import DatasetError
from warnings import warn

## WSC Meta-data

dataset_name = 'WSC'
root_folder = data_root + 'shapes' + '/'

# variable attributes and name
varatts = dict(discharge = dict(name='discharge', long_name='Stream Flow Rate', units='kg/s', scalefactor=1./1000.), # total flow rate
               level = dict(name='level', long_name='Water Level', units='m')) # water level
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# container class for stations and shape files
class Basin(Shape):
  ''' Just a container for basin information and associated station data '''
  def __init__(self, name=None, folder=None, shapefile=None, rivers=None, stations=None):
    ''' save meta information '''

# basin abbreviations and full names
basins = dict()
basins['ARB'] = 'Athabasca River Basin'
basins['FRB'] = 'Fraser River Basin'
# inverse dictionary, i.e. sorted by full name
fullbasin = {value:key for key,value in basins.iteritems()}
# river basin look-up
rivers = dict() # the basin a river belongs to
rivers['Athabasca'] = 'ARB'
rivers['Fraser'] = 'FRB'
 
# variable and file lists settings
foldername = root_folder + '{0:s}' + '/'
filename = '{0:s}_{1:s}{2:s}.dat' # basin abbreviation, variable name (full), tag (optional, prepend '_')


## Functions that handle access to ASCII files

def loadWSCstation(basin=None, river=None, stations=None, varlist=None, varatts=None, 
                   filetype='monthly_station', folder=foldername, filename=filename):
  ''' Function to load hydrograph climatologies for a given basin '''
  # resolve input
  
  # create dataset for all stations
  # make common time axis for climatology
  
  # loop over stations (one file per station, containing all variables)  
  # parse files and load variables 
  # extract variables (min/max/mean are separate variables)
  # this will work with case distinctions for variables (elif)
  
  ## example
  # open namelist file for reading 
  file = fileinput.FileInput([nmlstwps], mode='r')
  # loop over entries/lines
  for line in file: 
    # search for relevant entries
    if imd==0 and 'max_dom' in line:
      imd = file.filelineno()
      maxdom = int(line.split()[2].strip(','))
    elif isd==0 and 'start_date' in line:
      isd = file.filelineno()
      # extract start time of main and sub-domains
      dates = extractValueList(line)
      startdates = [date[1:14] for date in dates] # strip quotes and cut off after hours 
    elif ied==0 and 'end_date' in line:
      ied = file.filelineno()
      # extract end time of main domain (sub-domains irrelevant)
      dates = extractValueList(line)
      enddates = [date[1:14] for date in dates] # strip quotes and cut off after hours
    if imd>0 and isd>0 and ied>0:
      break # exit as soon as all are found
  
  # cast data into Geodat variables and add to dataset
  # return dataset
  return   

# load data from ASCII files and return numpy arrays
# data is assumed to be stored in monthly intervals
def loadWSC_hydrographs(var, fileformat=filename, arrayshape=(601,697)):
  # local imports
  from numpy.ma import zeros
  from numpy import genfromtxt, flipud
  # definitions
  datadir = root_folder + 'Climatology/ASCII/' # data folder   
  ntime = len(days_per_month) # number of month
  # allocate space
  data = zeros((ntime,)+arrayshape) # time = ntime, (x, y) = arrayshape  
  # loop over month
  print('  Loading variable %s from file.'%(var))
  for m in xrange(ntime):
    # read data into array
    filename = fileformat%(var,m+1)
    tmp = genfromtxt(datadir+filename, dtype=float, skip_header=5, 
                     missing_values=-9999, filling_values=-9999, usemask=True)
    data[m,:] = flipud(tmp)  
    # N.B.: the data is loaded in a masked array (where missing values are omitted)   
  # return array
  return data

if __name__ == '__main__':
    pass