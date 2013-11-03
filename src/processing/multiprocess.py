'''
Created on 2013-10-29


A simple script to test the usage of pool and how to return values...

@author: Andre R. Erler
'''

import multiprocessing
import logging
import sys
import types
import numpy as np
from datetime import datetime
from time import sleep


## test functions

def test_func(n, wait=3, queue=None):
  sleep(wait)
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
  for n in xrange(NP):
      pool.apply_async(func,(n,),dict(queue=q))
  pool.close()
  pool.join()
  results = []
  for n in xrange(NP): 
    results.append(q.get())
  for result in results:
    print result
  
def test_async(func, NP):
  ''' using pool async results '''
  print('\n   ***   using pool async results   ***\n')    
  pool = multiprocessing.Pool(processes=NP)
  results = []
  for n in xrange(NP):
      result = pool.apply_async(func,(n,))
      results.append(result)
  pool.close()
  pool.join()
  #print('\n   ***   joined   ***\n')
  results = [result.get() for result in results]
  for result in results:
    print result

def test_func_ec(n, wait=1, lparallel=True, logger=None):
  ''' test function that conforms to asyncPool requirements '''
  if logger is None: 
    logger = logging.getLogger() # new logger
    logger.addHandler(logging.StreamHandler())
  else: logger = logging.getLogger(name=logger) # connect to existing one
  sleep(wait)  
  pid = int(multiprocessing.current_process().name.split('-')[-1]) # start at 1
  logger.info('Current Process ID: %i\n'%pid)
  # return n as exit code
  return n 

def test_func_dec(n, wait=1, lparallel=False, pidstr='', logger=None):
  ''' test function for decorator '''
  sleep(wait)
  logger.info('\n{0:s} Current Process ID: {0:s}\n'.format(pidstr))  


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
    if logger is not None and not isinstance(logger,basestring): raise TypeError

    # logging
    if logger is None: 
      logger = logging.getLogger() # new logger
      logger.addHandler(logging.StreamHandler())
    else: logger = logging.getLogger(name=logger) # connect to existing one      
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
      self.func(*args, pidstr=pidstr, **kwargs)
      # return exit code
      return 0 # everything OK    
    except Exception: # , err
      # an error occurred
      logging.exception(pidstr) # print stack trace of last exception and current process ID 
      return 1 # indicate failure


def asyncPoolEC(func, args, kwargs, NP=1, ldebug=True, ltrialnerror=True):
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
  if not isinstance(NP,int): raise TypeError
  if not isinstance(ldebug,(bool,np.bool)): raise TypeError
  if not isinstance(ltrialnerror,(bool,np.bool)): raise TypeError
  
  # figure out if running parallel
  if NP is not None and NP == 1: lparallel = False
  else: lparallel = True
  #lparallel = True
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
  logger.info('\nTHREADS: {0:d}, DEBUG: {1:s}\n'.format(NP,str(ldebug)))
  exitcodes = [] # list of results  
  ## loop over and process all job sets
  if lparallel:
    # create pool of workers   
    if NP is None: pool = multiprocessing.Pool() 
    else: pool = multiprocessing.Pool(processes=NP)
    # distribute tasks to workers
    #print kwargs
    for arguments in args:
      #print arguments 
      exitcodes.append(pool.apply_async(func, arguments, kwargs))
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
    if lparallel: ec = ec.get()
    if ec < 0: raise ValueError, 'Exit codes have to be zero or positive!' 
    elif ec > 0: ec = 1
    # else ec = 0, i.e. no errors
    exitcode += ec
  nop = len(args) - exitcode
  
  # print summary (to log)
  if exitcode == 0:
    logger.info('\n   >>>   All {:d} operations completed successfully!!!   <<<   \n'.format(nop))
  else:
    logger.info('\n   ===   {:2d} operations completed successfully!    ===   \n'.format(nop) +
          '\n   ###   {:2d} operations did not complete/failed!   ###   \n'.format(exitcode))
  logger.info(datetime.today())
  # return with exit code
  exit(exitcode)


if __name__ == '__main__':

  NP = 2
  #print logging.DEBUG,logging.INFO,logging.WARNING,logging.ERROR,logging.CRITICAL
    
  #test_mq(test_func, NP)
  #test_async(test_func, NP)
  
  # test asyncPool
  args = [(n,) for n in xrange(5)]
  kwargs = dict(wait=1)
  asyncPoolEC(test_func_dec, args, kwargs, NP=NP, ldebug=True, ltrialnerror=True)
  #asyncPoolEC(test_func_ec, args, kwargs, NP=NP, ldebug=True)
  