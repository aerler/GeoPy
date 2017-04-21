'''
Created on 2013-08-23

Miscellaneous decorators, methods. and classes, as well as exception classes. 

@author: Andre R. Erler, GPL v3
'''

# numpy imports
import numpy as np
import numpy.ma as ma
import collections as col
import inspect

# days per month
days_per_month = np.array([31,28.2425,31,30,31,30,31,31,30,31,30,31], dtype='float32') # 97 leap days every 400 years
seconds_per_month = days_per_month * 86400.
# N.B.: the Gregorian calendar repeats every 400 years
days_per_month_365 = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype='float32') # no leap day
seconds_per_month_365 = days_per_month_365 * 86400.
# human-readable names
name_of_month = ['January  ', 'February ', 'March    ', 'April    ', 'May      ', 'June     ', #
                 'July     ', 'August   ', 'September', 'October  ', 'November ', 'December ']
stripped_month = [mon.strip().lower() for mon in name_of_month] # better case-isensitive
abbr_of_month = [mon[:3].lower() for mon in name_of_month] # better case-isensitive

## useful decorators

# decorator function that applies a function to a list 
class ElementWise():
  ''' 
    A decorator class that applies a function to a list (element-wise), and returns the results in a tuple. 
  '''
  
  def __init__(self, f):
    ''' Save original (scalar) function in decorator. '''
    self.f = f
    
  def __call__(self, *args, **kwargs):
    ''' Call function element-wise, by iterating over the argument list. Multiple arguments are supported using
        multiple list arguments (of the same length); key-word arguments are passed directly to the function. '''
    if isinstance(args[0], col.Iterable) and not isinstance(args[0], basestring):
      l = len(args[0])
      for arg in args: # check consistency
        if not isinstance(arg, col.Iterable) and len(arg)==l: 
          raise TypeError('All list arguments have to be of the same length!')
      results = [] # output list
      for i in xrange(l):
        eltargs = [arg[i] for arg in args] # construct argument list for this element
        results.append(self.f(*eltargs, **kwargs)) # results are automatically wrapped into tuples, if necessary
    else:
      results = (self.f( *args, **kwargs),) # just call function normally and wrap results in a tuple
    # make sure result is a tuple
    return tuple(results)
      

## useful functions

# check if input is a valid index or slice
@ElementWise
def checkIndex(idx, floatOK=False):
  ''' Check if idx is a valid index or slice; if floatOK=True, floats can also be indices. '''
  # check if integer or slice object
  if isinstance(idx,(list,tuple,np.ndarray)): 
    isIdx = isNumber(idx) if floatOK else isInt(idx)
  elif isinstance(idx,(int,np.integer)) or isinstance(idx,slice): isIdx = True
  elif floatOK and isinstance(idx,(float,np.inexact)): isIdx = True 
  else: isIdx = False         
  # return logical 
  return isIdx

# check if input is an integer
@ElementWise
def isInt(arg): return isinstance(arg,(int,np.integer))

# check if input is a float
@ElementWise
def isFloat(arg): return isinstance(arg,(float,np.inexact))

# check if input is a number
@ElementWise
def isNumber(arg): return isinstance(arg,(int,np.integer,float,np.inexact))

# define machine precision
floateps = np.finfo(np.float).eps
# check if an array is zero within machine precision
def isZero(data, eps=None, masked_equal=True):
  ''' This function checks if a numpy array or scalar is zero within machine precision, and returns a scalar logical. '''
  if isinstance(data,np.ndarray):
    if np.issubdtype(data.dtype, np.inexact): # also catch float32 etc
      return ma.allclose(np.zeros_like(data), data, masked_equal=True)
#     if eps is None: eps = 100.*floateps # default
#       return np.all( np.absolute(array) <= eps )
    elif np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool):
      return np.all( data == 0 )
  elif isinstance(data,float) or isinstance(data, np.inexact):
      if eps is None: eps = 100.*floateps # default
      return np.absolute(data) <= eps
  elif isinstance(data,(int,bool)) or isinstance(data, (np.integer,np.bool)):
      return data == 0
  else: raise TypeError(data)
# check if an array is one within machine precision
def isOne(data, eps=None, masked_equal=True):
  ''' This function checks if a numpy array or scalar is one within machine precision, and returns a scalar logical. '''
  if isinstance(data,np.ndarray):
    if np.issubdtype(data.dtype, np.inexact): # also catch float32 etc
      return ma.allclose(np.ones_like(data), data, masked_equal=True)
    elif np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool):
      return np.all( data == 1 )
  elif isinstance(data,float) or isinstance(data, np.inexact):
      if eps is None: eps = 100.*floateps # default
      return np.absolute(data-1) <= eps
  elif isinstance(data,(int,bool)) or isinstance(data, (np.integer,np.bool)):
      return data == 1
  else: raise TypeError(data)

# check if two arrays are equal within machine precision
def isEqual(left, right, eps=None, masked_equal=True):
  ''' This function checks if two numpy arrays or scalars are equal within machine precision, and returns a scalar logical. '''
  diff_type = "Both arguments to function 'isEqual' must be of the same class!"
  if isinstance(left,np.ndarray):
    # ndarray
    if not isinstance(right,np.ndarray): raise TypeError(diff_type)
    if not left.dtype==right.dtype:
      right = right.astype(left.dtype) # casting='same_kind' doesn't work...
    if np.issubdtype(left.dtype, np.inexact): # also catch float32 etc
      if eps is None: return ma.allclose(left, right, masked_equal=masked_equal)
      else: return ma.allclose(left, right, masked_equal=masked_equal, atol=eps)
    elif np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool):
      return np.all( left == right ) # need to use numpy's all()
  elif isinstance(left,(float,np.inexact)):
    # numbers
    if not isinstance(right,(float,np.inexact)): raise TypeError(diff_type)
    if eps is None: eps = 100.*floateps # default
    if ( isinstance(right,float) or isinstance(right,float) ) or left.dtype.itemsize == right.dtype.itemsize: 
      return np.absolute(left-right) <= eps
    else:
      if left.dtype.itemsize < right.dtype.itemsize: right = left.dtype.type(right)
      else: left = right.dtype.type(left)
      return np.absolute(left-right) <= eps  
  elif isinstance(left,(int,bool,np.integer,np.bool)):
    # logicals
    if not isinstance(right,(int,bool,np.integer,np.bool)): raise TypeError(diff_type)
    return left == right
  else: raise TypeError(left)
 

def translateSeasons(months):
  ''' determine indices for months from a string or integer identifying a season or month '''
  # determine season
  if isinstance(months,(int,np.integer)): 
    idx = [months-1] # indexing starts at 0, calendar month at 1
  elif isinstance(months,(list,tuple)):
    if all([isinstance(s,(int,np.integer)) for s in months]): 
      idx = [mon-1 for mon in months] # list of integers refering to calendar month
    elif all([isinstance(s,basestring) for s in months]):
      # list of names of month, optionally abbreviated, case insensitive
      idx = [abbr_of_month.index(mon[:3].lower()) for mon in months] 
    else: raise TypeError(months)
  elif isinstance(months,basestring):
    ssn = months.lower() # ignore case
    if ssn in abbr_of_month: # abbreviated name of a month, case insensitive 
      idx = np.asarray([abbr_of_month.index(ssn[:3])])
    elif ssn in stripped_month: # name of a month, case insensitive 
      idx = np.asarray([stripped_month.index(ssn)])
    else: # some definition of a season
      year = 'jfmamjjasondjfmamjjasond' # all month, twice
      # N.B.: regular Python indexing, starting at 0 for Jan and going to 11 for Dec
      if ssn == 'jfmamjjasond' or ssn == 'annual': idx = range(12)
      elif ssn == 'jja' or ssn == 'summer': idx = [5,6,7]
      elif ssn == 'djf' or ssn == 'winter': idx = [0,1,11]
      elif ssn == 'mam' or ssn == 'spring': idx = [2,3,4] # need to sort properly
      elif ssn == 'son' or ssn == 'fall'  or ssn == 'autumn': idx = [8,9,10]
      elif ssn == 'mamjja' or ssn == 'warm': idx = [2,3,4,5,6,7]
      elif ssn == 'sondjf' or ssn == 'cold': idx = [0,1,8,9,10,11] # need to sort properly
      elif ssn == 'amj' or ssn == 'melt': idx = [3,4,5,]
      elif ssn in year: 
        s = year.find(ssn) # find first occurrence of sequence
        idx = np.arange(s,s+len(ssn))%12 # and use range of months
      else: raise ValueError("Unknown key word/months/season: '{:s}'".format(str(months)))
  else: raise TypeError("Unknown identifier for months/season: '{:s}'".format(str(months)))
  # return indices for selected month
  idx = np.asarray(idx, dtype=np.int32) # return integers
  return idx

# utility function
def genStrArray(string_list):
  ''' utility function to generate a string array from a list of strings '''
  if not isinstance(string_list,(list,tuple)): raise TypeError(string_list)
  strlen = 0; new_list = []
  for string in string_list:
    if not isinstance(string,basestring): raise TypeError(string)
    strlen = max(len(string),strlen)
    new_list.append(string.ljust(strlen))
  strarray = np.array(new_list, dtype='|S{:d}'.format(strlen))
  assert strarray.shape == (len(string_list),)
  return strarray
    
# utility function to separate a run-together camel-casestring
def separateCamelCase(string, **kwargs):
  ''' Utility function to separate a run-together camel-casestring and replace string sequences. '''
  # insert white spaces
  charlist = [string[0]]
  for i in xrange(1,len(string)):
    if string[i-1].islower() and string[i].isupper(): charlist.append(' ')
    charlist.append(string[i])
  string = str().join(charlist)
  # replace names
  for old,new in kwargs.iteritems(): string = string.replace(old,new)
  # return new string
  return string 

# function to generate a sting representation of a list of numbers
def printList(iterable):
  ''' Small function to generate a sting representation of a list of numbers. '''
  string = '('
  for item in iterable: string += '{0:s},'.format(str(item))
  string += ')'
  return string


# import definitions from a script into global namespace
def loadGlobals(filename, warning=True):
  '''This method imports variables from a file into the global namespace; a common application is, to load
     physical constants. If warning is True, name collisions are reported. 
     (This method was adapted from the PyGeode module constants.py; original version by Peter Hitchcock.)
     Use like this:

        import os
        path, fname = os.path.split(__file__)
        load_constants(path + '/atm_const.py')
        del path, fname, os '''
  new_symb = {} # store source namespace in this dictionary
  execfile(filename, new_symb, None) # generate source namespace
  new_symb.pop('__builtins__')
  if warning: # detect any name collisions with existing global namespace
    coll = [k for k in new_symb.keys() if k in globals().keys()]
    if len(coll) > 0: # report name collisions
      from warnings import warn
      warn ("%d constants have been redefined by %s.\n%s" % (len(coll), filename, coll.__repr__()), stacklevel=2)
  # update global namespace
  globals().update(new_symb)


## a useful class to use dictionaries like "structs"

class AttrDict(dict):
  '''
    This class provides a dictionary where the items can be accessed like instance attributes.
    Use like this: 
    
        a = attrdict(x=1, y=2)
        print a.x, a.y
        
        b = attrdict()
        b.x, b.y  = 1, 2
        print b.x, b.y
  '''
  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs) # initialize as dictionary
    self.__dict__ = self # use itself (i.e. its own entries) as instance attributes

def joinDicts(*dicts):
      ''' Join dictionaries, but remove all entries that are conflicting. '''      
      joined = dict(); conflicting = set()
      for d in dicts:
        for key,value in d.iteritems():
          if key in conflicting: pass # conflicting entry
          elif key in joined: # either conflicting or same
            # check equality
            try: 
                equal = isEqual(value,joined[key]) # try numerical equality first
            except TypeError: 
                equal = (value == joined[key]) # fallback, if not an array or numerical
            if not equal:
              del joined[key] # remove conflicting
              conflicting.add(key)
          else: joined[key] = value # new entry
      # return joined dictionary
      return joined    
    
## another useful container class

class RecordClass(object):
  '''
    A class that takes keyword arguments and assigns their value to class attributes; limited type checking
    is performed. Defaults can be set in the child class definition
  '''
  
  def __init__(self, **kwargs):
    ''' initialize station parameters '''
    # generate attribute list
    def tmp_fct(att): return not(inspect.isroutine(att)) # lambda's cause problems when pickling
    parameters = inspect.getmembers(self, tmp_fct)
    parameters = [key for key,val in parameters if key[:2] != '__' and key[-2:] != '__']; del val
    self.__params__ = parameters # save parameter list for references
    cls = self.__class__
    # parse input    
    for key,value in kwargs.iteritems():
      if key in parameters:
        # simple type checking
        if getattr(self,key) is None: pass # no type checking...
        elif isinstance(getattr(self,key), basestring) and not isinstance(value, basestring):
          raise TypeError("Parameter '{:s}' has to be of type 'basestring'.".format(key)  )
        elif ( isinstance(getattr(self,key), (float,np.inexact)) and 
               not isinstance(value, (int,np.integer,float,np.inexact)) ):
          raise TypeError("Parameter '{:s}' has to be of a numeric type.".format(key))
        elif ( isinstance(getattr(self,key), (int,np.integer)) and 
               not isinstance(value, (int,np.integer)) ):
          raise TypeError("Parameter '{:s}' has to be of an integer type.".format(key))
        elif isinstance(getattr(self,key), (bool,np.bool)) and not isinstance(value, (bool,np.bool)):
          raise TypeError("Parameter '{:s}' has to be of boolean type.".format(key))
        # unfortunately this automated approach makes type checking a bit clumsy
        self.__dict__[key] = value
      else: raise ArgumentError("Invalid parameter: '{:s}'".format(key))
      
class StrictRecordClass(RecordClass):
  '''
    A version of RecordClass that enforces that all attributes are set explicitly; defaults are only used
    for type checking.
  '''      
  def __init__(self, **kwargs):
    # call original constructor
    super(StrictRecordClass,self).__init__(**kwargs)
    # check that all parameters are set
    for param in self.__params__:
      if param not in self.__dict__:
        raise ArgumentError("Parameter '{:s}' was not set (missing argument).".format(param))


## Error handling classes
# from base import Variable

class VariableError(Exception):
  ''' Base class for exceptions occurring in Variable methods. '''
  
  lvar = False # indicate that variable information is available
  var = None # variable instance
  lmsg = False # indicate that a print message is avaialbe
  msg = '' # Error message
  
  def __init__(self, *args):
    ''' Initialize with parent constructor and add message or variable instance. '''
    # parse special input
    from base import Variable
    lmsg = False; msg = ''; lvar = False; var = None; arglist = []
    for arg in args:
      if isinstance(arg,str): # string input
        if not lmsg: 
          lmsg = True; msg = arg 
      elif isinstance(arg,Variable): # variable instance
        if not lvar:
          lvar = True; var = arg
      else: arglist.append(arg) # assemble remaining arguments        
    # call parent constructor
    super(VariableError,self).__init__(*arglist)
    self.lmsg = lmsg; self.msg = msg; self.lvar = lvar; self.var = var
    # assign special attributes 
    if self.lvar:
      self.type = self.var.__class__.__name__
      self.name = self.var.name
    
  def __str__(self):
    ''' Print informative message based on available information. '''
    if self.lvar: # print message based on variable instance
      return 'An Error occurred in the %s instance \'%s\'.'%(self.type,self.name)
    if self.lmsg: # return standard message 
      return self.msg 
    else: # fallback
      return super(VariableError,self).__str__()

class DataError(VariableError):
  '''Exception raised when data is requested that is not loaded. '''

  nodata = False # if data was loaded; need to check variable...
  
  def __init__(self, *args):
    ''' Initialize with parent constructor and add message or variable instance. '''
    super(DataError,self).__init__(*args)
    # add special parameters
    if self.lvar:
      self.nodata = not self.var.data # inverse
    
  def __str__(self):
    ''' Print error message for data request. '''
    if self.lmsg: # custom message has priority
      return self.msg
    elif self.nodata: # check variable state
      return '%s instance \'%s\' has no data loaded!'%(self.type,self.name)
    else: # fallback
      return super(VariableError,self).__str__()
     
class AxisError(VariableError):
  ''' Exceptions related to Axes. '''
  pass

class TimeAxisError(VariableError):
  ''' Errors specifically related to the time Axes. '''
  pass

class PermissionError(VariableError):
  ''' Exceptions raised when permissions are missing (such as defined by 'mode'). '''
  pass

class FileError(IOError):
  ''' Exceptions indicating a file access problem (usually a missing file). '''
  pass

class ParseError(FileError):
  ''' Errors that occur while parsing a file. '''
  pass

class NetCDFError(FileError):
  ''' Exceptions related to NetCDF file access. '''
  pass

class ArgumentError(Exception):
  ''' Exceptions related to passed arguments. '''
  pass

class GDALError(VariableError):
  ''' Base class for exceptions related to GDAL. '''
  pass

class DatasetError(VariableError):
  ''' Base class for exceptions occurring in Dataset methods. '''
  pass

class EmptyDatasetError(Exception):
  ''' Error to indicate that a loaded Dataset is empty. '''
  pass

class DateError(VariableError):
  ''' Exception indicating invalid dates passed to processor. '''
  pass

class ListError(TypeError):
  ''' Error class for failed list expansion. '''
  pass

class DistVarError(VariableError):
  ''' Exception indicating invalid use of overloaded method. '''
  pass

## simple application code
if __name__ == '__main__':

  print('Floating-point precision on this machine:')
  print(' '+str(floateps))