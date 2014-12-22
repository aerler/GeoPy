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
# import numbers as num

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
    if isinstance(args[0], col.Iterable):
      l = len(args[0])
      for arg in args: # check consistency
        if not isinstance(arg, col.Iterable) and len(arg)==l: 
          raise TypeError, 'All list arguments have to be of the same length!'
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
  else: raise TypeError
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
  else: raise TypeError

# check if two arrays are equal within machine precision
def isEqual(left, right, eps=None, masked_equal=True):
  ''' This function checks if two numpy arrays or scalars are equal within machine precision, and returns a scalar logical. '''
  diff_type = "Both arguments to function 'isEqual' must be of the same class!"
  if isinstance(left,np.ndarray):
    # ndarray
    if not isinstance(right,np.ndarray): raise TypeError, diff_type 
    if not left.dtype==right.dtype:
      right = right.astype(left.dtype) # casting='same_kind' doesn't work...
    if np.issubdtype(left.dtype, np.inexact): # also catch float32 etc
      if eps is None: return ma.allclose(left, right, masked_equal=masked_equal)
      else: return ma.allclose(left, right, masked_equal=masked_equal, atol=eps)
    elif np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool):
      return np.all( left == right ) # need to use numpy's all()
  elif isinstance(left,(float,np.inexact)):
    # numbers
    if not isinstance(right,(float,np.inexact)): raise TypeError, diff_type
    if eps is None: eps = 100.*floateps # default
    return np.absolute(left-right) <= eps
  elif isinstance(left,(int,bool,np.integer,np.bool)):
    # logicals
    if not isinstance(right,(int,bool,np.integer,np.bool)): raise TypeError, diff_type
    return left == right
  else: raise TypeError
 
 
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
            if value.__class__ != joined[key].__class__: equal = False  
            elif isinstance(value,basestring): equal = (value == joined[key])
            else: equal = isEqual(value,joined[key])              
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
          raise TypeError, "Parameter '{:s}' has to be of type 'basestring'.".format(key)  
        elif ( isinstance(getattr(self,key), (float,np.inexact)) and 
               not isinstance(value, (int,np.integer,float,np.inexact)) ):
          raise TypeError, "Parameter '{:s}' has to be of a numeric type.".format(key)
        elif ( isinstance(getattr(self,key), (int,np.integer)) and 
               not isinstance(value, (int,np.integer)) ):
          raise TypeError, "Parameter '{:s}' has to be of an integer type.".format(key)
        elif isinstance(getattr(self,key), (bool,np.bool)) and not isinstance(value, (bool,np.bool)):
          raise TypeError, "Parameter '{:s}' has to be of boolean type.".format(key)  
        # unfortunately this automated approach makes type checking a bit clumsy
        self.__dict__[key] = value
      else: raise ArgumentError, "Invalid parameter: '{:s}'".format(key)
      
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
        raise ArgumentError, "Parameter '{:s}' was not set (missing argument).".format(param)    


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