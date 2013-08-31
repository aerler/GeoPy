'''
Created on 2013-08-23

Miscellaneous decorators, methods. and classes, as well as exception classes. 

@author: Andre R. Erler, GPL v3
'''

# numpy imports
import numpy as np
import collections as col

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
        assert isinstance(arg, col.Iterable) and len(arg)==l, 'All list arguments have to be of the same length!'
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
  if isinstance(idx,int) or isinstance(idx,slice): isIdx = True
  else: isIdx = False       
  # if a float is also allowed, check that
  if floatOK and isinstance(idx,float): isIdx = True 
  # return logical 
  return isIdx

# check if input is a float
@ElementWise
def isFloat(arg): return isinstance(arg,float)

# define machine precision
floateps = np.finfo(np.float).eps
# check if an array is zero within machine precision
def isZero(array, eps=None):
  ''' This function checks if a numpy array (or scalar) is zero within machine precision, and returns a scalar logical. '''
  if isinstance(array,np.ndarray):
    if array.dtype == 'float':
      if eps is None: eps = 100.*floateps # default
      return ( np.absolute(array) <= eps ).all()
    elif array.dtype == 'int' or array.dtype == 'bool':
      return all( array == 0 )
  elif isinstance(array,float):
      if eps is None: eps = 100.*floateps # default
      return np.absolute(array) <= eps
  elif isinstance(array,(int,bool)):
      return array == 0
# check if two arrays are equal within machine precision
def isEqual(left, right, eps=None):
  ''' This function checks if two numpy arrays are equal within machine precision, and returns a scalar logical. '''
  assert left.dtype==right.dtype, 'Both arguments to function \'isEqual\' must be of the same type!'
  assert left.__class__==right.__class__, 'Both arguments to function \'isEqual\' must be of the same class!'
  if (left.dtype == 'bool') or (left.dtype == 'int'):
    return ( left == right ).all()
  else:
    return isZero(left - right, eps=eps)


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


## simple application code
if __name__ == '__main__':

  print('Floating-point precision on this machine:')
  print(' '+str(floateps))