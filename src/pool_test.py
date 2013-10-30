'''
Created on 2013-10-29

A simple script to test the usage of pool and how to return values...

@author: Andre R. Erler
'''

from multiprocessing import Pool
from multiprocessing import Manager
from time import sleep

def f(n, queue=None):
  sleep(3)
  result = [n, None, 'hello']
  if queue is not None:
    q.put(result)
  else:
    return result
  

if __name__ == '__main__':

    NP = 2
    
    ## using a managed queue
    print('\n   ***   using managed queue   ***\n')
    m = Manager()
    q =m.Queue()
    pool = Pool(processes=NP)
    results = []
    for n in xrange(NP):
        pool.apply_async(f,(n,),dict(queue=q))
    pool.close()
    pool.join()
    results = []
    for n in xrange(NP): 
      results.append(q.get())
    for result in results:
      print result

    ## using pool async results
    print('\n   ***   using pool async results   ***\n')    
    pool = Pool(processes=NP)
    results = []
    for n in xrange(NP):
        result = pool.apply_async(f,(n,))
        results.append(result)
    pool.close()
    pool.join()
    #print('\n   ***   joined   ***\n')
    results = [result.get() for result in results]
    for result in results:
      print result
    