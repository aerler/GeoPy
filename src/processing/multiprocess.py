'''
Created on 2013-10-29


A simple script to test the usage of pool and how to return values...

@author: Andre R. Erler
'''

import multiprocessing
import logging
import sys
import gc # garbage collection
import types
import os
import numpy as np
from datetime import datetime
from time import sleep


## test functions

global_var = 0

# test apply_along_axis
def test_aax_helper(arr, kw=0, axis=0): # lambda fct can't be pickled
  return ( arr - np.mean(arr) ) / arr.std() - kw, kw
 
def test_aax(arr, kw=0, axis=0): return test_aax_helper(arr, kw=kw)[0]
 
def test_noaax(arr, axis=0, kw=0):
  shape = arr.shape[:-1]+(1,)
  mean = np.mean(arr,axis=axis).reshape(shape)
  std = np.std(arr,axis=axis).reshape(shape)
  return (arr - mean) / std -kw


def test_func(n, wait=None, queue=None):
  global global_var
#   print(n,global_var)
  global_var = n
#   print(n,global_var)
  sleep(wait or n)  
  print((n,global_var))
  result = [n, 'hello', None]
  if queue is not None:
    queue.put(result)
  else:
    return result

def test_mq(func, NP):
  ''' using a managed queue '''
  print('\n   ***   using managed queue   ***\n')
  m = multiprocessing.Manager()
  q =m.Queue()
  pool = multiprocessing.Pool(processes=NP)
  for n in range(NP):
      pool.apply_async(func,(n,),dict(queue=q))
  pool.close()
  pool.join()
  results = []
  for n in range(NP): 
    results.append(q.get())
  for result in results:
    print(result)
  
def test_async(func, NP):
  ''' using pool async results '''
  print('\n   ***   using pool async results   ***\n')    
  pool = multiprocessing.Pool(processes=NP)
  results = []
  for n in range(NP):
      result = pool.apply_async(func,(n,))
      results.append(result)
  pool.close()
  pool.join()
  #print('\n   ***   joined   ***\n')
  results = [result.get() for result in results]
  for result in results:
    print(result)

def test_func_ec(n, wait=1, lparallel=True, pidstr='', logger=None, ldebug=False):
  ''' test function that conforms to asyncPool requirements '''
  sleep(wait)  
  pid = int(multiprocessing.current_process().name.split('-')[-1]) # start at 1
  logger.info('{:s} Current Process ID: {:d}'.format(pidstr,pid))
  assert int(pidstr[-3:-1]) == pid
  # return n as exit code
  return n 

def test_func_dec(n, wait=1, lparallel=False, pidstr='', logger=None, ldebug=False):
  ''' test function for decorator '''
  sleep(wait)
  pid = int(multiprocessing.current_process().name.split('-')[-1]) # start at 1
  logger.info('{:s} Current Process ID: {:d}'.format(pidstr,pid))
  assert int(pidstr[-3:-1]) == pid
  

## production functions

# a decorator class that handles loggers and exit codes for functions inside asyncPool_EC  
class TrialNError():
  ''' 
    A decorator class that handles errors and returns an exit code for a pool worker function;
    also handles loggers and some multiprocessing stuff. 
  '''
  
  def __init__(self, func):
    ''' Save original function in decorator class. '''
    self.func = func
    
  def __call__(self, *args, **kwargs):
    ''' connect to logger, figure out process ID, execute decorated function in try-block,
        report errors, and return exit code '''
    # input arguments
    lparallel = kwargs['lparallel']
    logger = kwargs['logger']
    # type checks
    if not isinstance(lparallel,bool): raise TypeError
    if logger is not None and not isinstance(logger,str): raise TypeError

    # logging
    if logger is None: 
      logger = logging.getLogger() # new logger
      logger.addHandler(logging.StreamHandler())
    else: logger = logging.getLogger(name=logger) # connect to existing one    
    logger.propagate = False # suppress duplicate output
    # parallelism
    if lparallel:
      pid = int(multiprocessing.current_process().name.split('-')[-1]) # start at 1
      pidstr = '[proc{0:02d}]'.format(pid) # pid for parallel mode output
    else:
      pidstr = '' # don't print process ID, sicne there is only one

    # execute decorated function in try-block
    kwargs['logger'] = logger
    try:
      # decorated function
      ec = self.func(*args, pidstr=pidstr, **kwargs)
      gc.collect() # enforce garbage collection
      # return exit code
      return ec or 0 # everything OK (and ec = None is OK, too)    
    except Exception: # , err
      # an error occurred
      logging.exception(pidstr) # print stack trace of last exception and current process ID 
      return 1 # indicate failure


def asyncPoolEC(func, args, kwargs, NP=1, ldebug=False, ltrialnerror=True):
  ''' 
    A function that executes func with arguments args (len(args) times) on NP number of processors;
    args must be a list of argument tuples; kwargs are keyword arguments to func, which do not change
    between calls.
    Func is assumed to take a keyword argument lparallel to indicate parallel execution, and return 
    a common exit status (0 = no error, > 0 for an error code).
    This function returns the number of failures as the exit code. 
  '''
  # input checking
  if not isinstance(func,types.FunctionType): raise TypeError
  if not isinstance(args,list): raise TypeError
  if not isinstance(kwargs,dict): raise TypeError
  if NP is not None and not isinstance(NP,int): raise TypeError
  if not isinstance(ldebug,(bool,np.bool)): raise TypeError
  if not isinstance(ltrialnerror,(bool,np.bool)): raise TypeError
  
  # figure out if running parallel
  if NP is not None and NP == 1: lparallel = False
  else: lparallel = True
  kwargs['ldebug'] = ldebug
  kwargs['lparallel'] = lparallel  

  # logging level
  if ldebug: loglevel = logging.DEBUG
  else: loglevel = logging.INFO
  # set up parallel logging (multiprocessing)
  if lparallel:
    multiprocessing.log_to_stderr()
    mplogger = multiprocessing.get_logger()
    #if ldebug: mplogger.setLevel(logging.DEBUG)
    if ldebug: mplogger.setLevel(logging.INFO)
    else: mplogger.setLevel(logging.ERROR)
  # set up general logging
  logger = logging.getLogger('multiprocess.asyncPoolEC') # standard logger
  logger.setLevel(loglevel)
  ch = logging.StreamHandler(sys.stdout) # stdout, not stderr
  ch.setLevel(loglevel)
  ch.setFormatter(logging.Formatter('%(message)s'))
  logger.addHandler(ch)
  kwargs['logger'] = logger.name
#   # process sub logger
#   sublogger = logging.getLogger('multiprocess.asyncPoolEC.func') # standard logger
#   sublogger.propagate = False # only print message in this logger
#   sch = logging.StreamHandler(sys.stdout) # stdout, not stderr
#   sch.setLevel(loglevel)
#   if lparallel: fmt = logging.Formatter('[%(processName)s] %(message)s')
#   else: fmt = logging.Formatter('%(message)s')
#   sch.setFormatter(fmt)
#   sublogger.addHandler(sch)
#   kwargs['logger'] = sublogger.name
  
  # apply decorator
  if ltrialnerror: func = TrialNError(func)
  
  # print first logging message
  logger.info(datetime.today())
  logger.info('\nTHREADS: {0:s}, DEBUG: {1:s}\n'.format(str(NP),str(ldebug)))
  exitcodes = [] # list of results  
  def callbackEC(result):
    # custom callback function that appends the results to the list
    exitcodes.append(result)
  ## loop over and process all job sets
  if lparallel:
    # create pool of workers   
    pool = multiprocessing.Pool(processes=NP) # NP=None uses all available CPUs
    # distribute tasks to workers
    for arguments in args:
      #exitcodes.append(pool.apply_async(func, arguments, kwargs))
      #print arguments      
      pool.apply_async(func, arguments, kwargs, callback=callbackEC) 
      # N.B.: we do not record result objects, since we have callback, which just extracts the exitcodes
    # wait until pool and queue finish
    pool.close()
    pool.join() 
    logger.debug('\n   ***   all processes joined   ***   \n')
  else:
    # don't parallelize, if there is only one process: just loop over files    
    for arguments in args:       
      exitcodes.append(func(*arguments, **kwargs))
    
  # evaluate exit codes    
  exitcode = 0
  for ec in exitcodes:
    #if lparallel: ec = ec.get() # not necessary, if callback is used
    if ec < 0: raise ValueError('Exit codes have to be zero or positive!') 
    elif ec > 0: ec = 1
    # else ec = 0, i.e. no errors
    exitcode += ec
  # N.B.: returnign None is interpreted as  
  nop = len(args) - exitcode
  
  # print summary (to log)
  if exitcode == 0:
    logger.info('\n   >>>   All {:d} operations completed successfully!!!   <<<   \n'.format(nop))
  else:
    logger.info('\n   ===   {:2d} operations completed successfully!    ===   \n'.format(nop) +
          '\n   ###   {:2d} operations did not complete/failed!   ###   \n'.format(exitcode))
  logger.info(datetime.today())
  # return with exit code
  return exitcode

def apply_along_axis(fct, axis, data, NP=0, chunksize=200, ldebug=False, laax=True, *args, **kwargs):
  ''' a parallelized version of numpy's apply_along_axis; the preferred way of passing arguments is,
      by using functools.partial, but arguments can also be passed to this function; the call-signature
      is the same as for np.apply_along_axis, except for NP=OMP_NUM_THREADS, chunksize=200, 
      ldebug=False, and laax=True; the latter can be set to False, if fct is fully vectorized and only
      the parallelization feature is required, otherwise Numpy's apply_along_axis will be called within
      child processes. '''  
  if NP == 0: NP = int(os.environ['OMP_NUM_THREADS'])
  # pre-processing: move sampel axis to the back
  if not axis == data.ndim-1:
    data = np.rollaxis(data, axis=axis, start=data.ndim) # roll sample axis to last (innermost) position
  arrayshape,samplesize = data.shape[:-1],data.shape[-1]
#   arraysize = np.prod(arrayshape) if arrayshape else 1
  arraysize = int(np.prod(arrayshape))
  # flatten array for redistribution
  data = np.reshape(data,(arraysize,samplesize))
  # compute
  if chunksize == 0: chunksize = 1 
  if not laax: kwargs['axis'] = 1 # for ufunc-like functions
  elif len(kwargs) > 0: raise NotImplementedError("np.apply_along_axis doesn't take kwargs")
  if ldebug: print(("Arraysize: {}, Chunksize: {}".format(arraysize,chunksize)))
  if (NP == 1 or arraysize < 1.1*chunksize):
    # just use regular Numpy version... but always apply over last dimension
    if ldebug: print('\n   ***   Running in Serial Mode   ***')
    if laax: results = np.apply_along_axis(fct, 1, data, *args, **kwargs)
    else: results = fct(data, *args, **kwargs)
  else:
    # adjust number of processors
    NP = int(min(NP,np.around(arraysize/chunksize)))
    if ldebug: print(('NP: {}'.format(NP)))
    # split up data
    if arraysize < (NP+1)*chunksize:
      cs = int(arraysize//NP) # chunksize; use integer division
      if arraysize%NP != 0: cs += 1
      nc = NP
    else:
      nc = int(arraysize//chunksize) # number of chunks; use integer division
      if arraysize%chunksize != 0: nc += 1
      cs = chunksize
    chunks = [data[i*cs:(i+1)*cs,:] for i in range(nc)] # views on subsets of the data
    # initialize worker pool
    if ldebug: print('\n   ***   firing up pool (using async results)   ***')
    if ldebug: print(('         OMP_NUM_THREADS = {:d}\n'.format(NP)))
    pool = multiprocessing.Pool(processes=NP)
    results = [] # list of resulting chunks (concatenated later    
    for n in range(nc):
      # run computation on individual subsets/chunks
      if ldebug: print(('   Starting Chunk #{:d}'.format(n+1)))
      if laax: # use Numpy's apply_along_axis
        result = pool.apply_async(np.apply_along_axis, (fct,1,chunks[n],)+args, kwargs)
      else: # for ufunc-like functions that can operate on multi-dimensional arrays
        result = pool.apply_async(fct, (chunks[n],)+args, kwargs)
      results.append(result)
    pool.close()
    pool.join()
    if ldebug: print('\n   ***   joined worker pool (getting results)   ***\n')
    # retrieve and assemble results 
    results = tuple(result.get() for result in results)
    results = np.concatenate(results, axis=0) 
  # check and reshape
  assert results.shape[0] == arraysize
  if results.ndim == 1: # if the second dimension was reduced to a scalar
    results = np.reshape(results,arrayshape)
    assert results.ndim == len(arrayshape) and results.shape == arrayshape
  elif results.ndim == 2: # if the second dimansion was replaced
    results = np.reshape(results,arrayshape+(results.shape[1],))
    assert results.ndim == len(arrayshape)+1 and results.shape[:-1] == arrayshape
    if not axis == results.ndim-1:
      results = np.rollaxis(results, axis=results.ndim-1, start=axis) # roll sample axis back to original position
  # return results
  return results

if __name__ == '__main__':

  NP = 4
  from geodata.misc import isEqual, isZero
  from functools import partial
  # test apply_along_axis
  axis = 1
  def func(arr, kw=0, axis=0): # lambda fct can't be pickled
    return ( arr - np.mean(arr) ) / arr.std() - kw, kw
   
  def func1(arr, kw=0, axis=0): return func(arr, kw=kw)[0]
   
  def nolaax(arr, axis=0, kw=0):
    shape = arr.shape[:-1]+(1,)
    mean = np.mean(arr,axis=axis).reshape(shape)
    std = np.std(arr,axis=axis).reshape(shape)
    return (arr - mean) / std -kw
  
  def run_test(fct, kw=0, laax=True):
    ff = partial(fct, kw=kw)
    shape = (500,100)
    data = np.arange(np.prod(shape), dtype='float').reshape(shape)
    assert data.shape == shape
    # parallel implementation using my wrapper
    pres = apply_along_axis(ff, axis, data, NP=2, ldebug=True, laax=laax)
    print(pres.shape)
    assert pres.shape == data.shape
    assert isZero(pres.mean(axis=axis)+kw) and isZero(pres.std(axis=axis)-1.)
    # straight-forward numpy version
    res = np.apply_along_axis(ff, axis, data)
    assert res.shape == data.shape
    assert isZero(res.mean(axis=axis)+kw) and isZero(res.std(axis=axis)-1.)
    # final test
    assert isEqual(pres, res) 
    
  # run tests 
  run_test(nolaax, kw=1, laax=False) # without Numpy's apply_along_axis
  run_test(func1, kw=1, laax=True) # Numpy's apply_along_axis
  
  #print logging.DEBUG,logging.INFO,logging.WARNING,logging.ERROR,logging.CRITICAL
    
  #test_mq(test_func, NP)
#   print(global_var)
#   test_async(test_func, NP)
#   print(global_var)
  
  # test asyncPool
#   args = [(n,) for n in xrange(5)]
#   kwargs = dict(wait=1)
#   asyncPoolEC(test_func_dec, args, kwargs, NP=NP, ldebug=True, ltrialnerror=True)
  #asyncPoolEC(test_func_ec, args, kwargs, NP=NP, ldebug=True)
  
