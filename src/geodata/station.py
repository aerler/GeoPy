'''
Created on 2014-02-13

A module that introduces a special class intended for station datasets (i.e. time-series only).

@author: Andre R. Erler, GPL v3
'''

# internal imports
from geodata.base import Dataset, Variable, Axis
from geodata.netcdf import DatasetNetCDF

## the basic station class, without any geographic information  
class StationDataset(Dataset):
  '''
    A Dataset class that is intended for station time-series data (usually one-dimensional), at one 
    particular location; this class also holds additional meta data.
  '''

  def __init__(self,  name=None, title=None, ID=None, varlist=None, atts=None, **kwargs):
    ''' 
      This class can be initialized simply with a name and a (optionally) a set of variables. 
      
      Station Attributes:
        ID = @property # station ID      
      Basic Attributes:
        name = @property # short name (links to name in atts)
        title = @property # descriptive name (links to name in atts)
        variables = dict() # dictionary holding Variable instances
        axes = dict() # dictionary holding Axis instances (inferred from Variables)
        atts = AttrDict() # dictionary containing global attributes / meta data   
    '''
    # initialize Dataset using parent constructor (kwargs are necessary, in order to support multiple inheritance)
    super(StationDataset,self).__init__(name=name, title=title, varlist=varlist, atts=atts, **kwargs)
    # set remaining attibutes
    self.atts['ID'] = ID
    
    @property
    def ID(self):
      ''' The station ID, usually an alphanumerical code. '''
      return self.atts['ID']
    @ID.setter
    def ID(self, ID):
      self.atts['ID'] = ID

      
## the NetCDF version of the station dataset
class StationNetCDF(StationDataset,DatasetNetCDF):
  '''
    A StationDataset, associated with a NetCDF file, inheriting the properties of DatasetNetCDF.
    WARNING: this class has not been tested!  
  '''
  
  def __init__(self, name=None, title=None, ID=None, dataset=None, filelist=None, varlist=None, varatts=None, 
               atts=None, axes=None, multifile=False, check_override=None, folder='', mode='r', ncformat='NETCDF4', 
               squeeze=True):
    ''' 
      Create a Dataset from one or more NetCDF files; Variables are created from NetCDF variables. 
      
      Station Attributes:
        ID = @property # station ID      
      NetCDF Attributes:
        mode = 'r' # a string indicating whether read ('r') or write ('w') actions are intended/permitted
        datasets = [] # list of NetCDF datasets
        dataset = @property # shortcut to first element of self.datasets
        filelist = [] # files used to create datasets 
      Basic Attributes:        
        variables = dict() # dictionary holding Variable instances
        axes = dict() # dictionary holding Axis instances (inferred from Variables)
        atts = AttrDict() # dictionary containing global attributes / meta data
    '''
    # call parent constructor
    super(StationNetCDF,self).__init__(self,self, name=None, title=None, ID=None, dataset=None, filelist=None, 
                                       varlist=None, varatts=None, atts=None, axes=None, multifile=False, 
                                       check_override=None, folder='', mode='r', ncformat='NETCDF4', squeeze=True)
    
