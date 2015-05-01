'''
Created on 2015-01-19 

Unittest for assorted PyGeoDat components not included elsewhere.

@author: Andre R. Erler, GPL v3
'''

import unittest
import numpy as np
import os, sys, gc
import multiprocessing
import logging
from time import sleep


# import geodata modules
from utils.nctools import writeNetCDF
from geodata.misc import isZero, isOne, isEqual
from geodata.base import Variable, Axis, Dataset, Ensemble
from datasets.common import data_root
# import modules to be tested


# RAM disk settings ("global" variable)
RAM = True # whether or not to use a RAM disk
ramdisk = '/media/tmp/' # folder where RAM disk is mounted
NP = 2
ldebug = False


## tests for multiprocess module
class MultiProcessTest(unittest.TestCase):  
   
  def setUp(self):
    ''' create two test variables '''
    pass
      
  def tearDown(self):
    ''' clean up '''
    gc.collect()

  def testApplyAlongAxis(self):
    ''' test parallelized version of Numpy's apply_along_axis '''    
    from processing.multiprocess import apply_along_axis, test_aax, test_noaax
    import functools
    
    def run_test(fct, kw=0, axis=1, laax=True):
      ff = functools.partial(fct, kw=kw)
      shape = (500,100)
      data = np.arange(np.prod(shape), dtype='float').reshape(shape)
      assert data.shape == shape
      # parallel implementation using my wrapper
      pres = apply_along_axis(ff, axis, data, NP=2, ldebug=True, laax=laax)
      print pres.shape
      assert pres.shape == data.shape
      assert isZero(pres.mean(axis=axis)+kw) and isZero(pres.std(axis=axis)-1.)
      # straight-forward numpy version
      res = np.apply_along_axis(ff, axis, data)
      assert res.shape == data.shape
      assert isZero(res.mean(axis=axis)+kw) and isZero(res.std(axis=axis)-1.)
      # final test
      assert isEqual(pres, res) 
      
    # run tests 
    run_test(test_noaax, kw=1, laax=False) # without Numpy's apply_along_axis
    run_test(test_aax, kw=1, laax=True) # Numpy's apply_along_axis

  
  def testAsyncPool(self):
    ''' test asyncPool wrapper '''    
    from processing.multiprocess import asyncPoolEC, test_func_dec, test_func_ec
    args = [(n,) for n in xrange(5)]
    kwargs = dict(wait=1)
    ec = asyncPoolEC(test_func_dec, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
    assert ec == 0 
    ec = asyncPoolEC(test_func_ec, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=True)
    assert ec == 4
    ec = asyncPoolEC(test_func_ec, args, kwargs, NP=NP, ldebug=ldebug, ltrialnerror=False)
    assert ec == 0
    

  
## tests related to loading datasets
class DatasetsTest(unittest.TestCase):  
   
  def setUp(self):
    ''' create two test variables '''
    pass
      
  def tearDown(self):
    ''' clean up '''
    gc.collect()

  def testExpArgList(self):
    ''' test function to expand argument lists '''    
    from datasets.common import expandArgumentList
    # test arguments
    args1 = [0,1,2]; args2 = ['0','1','2']; args3 = ['test']*3; arg4 = 'static1'; arg5 = 'static2' 
    explist = ['arg1','arg2','arg3']
    # test inner product expansion
    arg_list = expandArgumentList(arg1=args1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  expand_list=explist, lproduct='inner')
    assert len(arg_list) == len(args1) and len(arg_list) == len(args2)
    for args,arg1,arg2,arg3 in zip(arg_list,args1,args2,args3):
      assert args['arg1'] == arg1
      assert args['arg2'] == arg2
      assert args['arg3'] == arg3
      assert args['arg4'] == arg4
      assert args['arg5'] == arg5
    # test outer product expansion
    arg_list = expandArgumentList(arg1=args1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  expand_list=explist, lproduct='outer')
    assert len(arg_list) == len(args1) * len(args2) * len(args3)
    n = 0
    for arg1 in args1:
      for arg2 in args2:
        for arg3 in args3:
          args = arg_list[n]
          assert args['arg1'] == arg1
          assert args['arg2'] == arg2
          assert args['arg3'] == arg3
          assert args['arg4'] == arg4
          assert args['arg5'] == arg5
          n += 1
    assert n == len(arg_list)
    # test simultaneous inner and outer product expansion
    n1 = len(args2) * len(args3) / len(args1)
    tmp1 = args1*n1
    arg_list = expandArgumentList(arg1=tmp1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  outer_list=['arg2','arg3'], inner_list=['arg1'])
    assert len(arg_list) == len(args2) * len(args3) == len(tmp1)
    n = 0
    for arg2 in args2:
      for arg3 in args3:
          args = arg_list[n]
          assert args['arg1'] == tmp1[n]
          assert args['arg2'] == arg2
          assert args['arg3'] == arg3
          assert args['arg4'] == arg4
          assert args['arg5'] == arg5
          n += 1
    assert n == len(arg_list)
    # test parallel outer product expansion
    assert len(args1) == len(args2) # necessary for test
    arg_list = expandArgumentList(arg1=args1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  expand_list=[('arg1','arg2'),'arg3'], lproduct='outer')
    assert len(arg_list) == len(args1) * len(args3)
    n = 0
    for arg1,arg2 in zip(args1,args2):
      for arg3 in args3:
        args = arg_list[n]
        assert args['arg1'] == arg1
        assert args['arg2'] == arg2
        assert args['arg3'] == arg3
        assert args['arg4'] == arg4
        assert args['arg5'] == arg5
        n += 1
    assert n == len(arg_list)
    
  def testLoadDataset(self):
    ''' test universal dataset loading function '''
    from datasets.common import loadDataset, loadClim, loadStnTS 
    # test climtology
    ds = loadClim(name='PCIC', grid='arb2_d02', varlist=['precip'])
    assert isinstance(ds, Dataset)
    assert ds.name == 'PCIC'
    assert 'precip' in ds
    assert ds.gdal and ds.isProjected
    # test CVDP
    ds = loadDataset(name='HadISST', period=None, varlist=None, mode='CVDP')
    assert isinstance(ds, Dataset)
    assert ds.name == 'HadISST'
    assert 'PDO' in ds
    assert ds.gdal and not ds.isProjected
    # test CVDP with WRF
    ds = loadDataset(name='phys-ens-2100', period=None, varlist=None, mode='CVDP')
    assert isinstance(ds, Dataset)
    assert ds.name == 'phys-ens-2100'
    assert 'PDO' in ds
    assert ds.gdal and not ds.isProjected    
    # test WRF station time-series
    ds = loadStnTS(name='ctrl-1_d02', varlist=['MaxPrecip_1d'], station='ecprecip', filetypes='hydro')
    assert isinstance(ds, Dataset)
    assert ds.name == 'ctrl-1_d02'
    assert 'MaxPrecip_1d' in ds
    # test example with list expansion
    # test EC station time-series
    dss = loadStnTS(name=['EC','ctrl-1'], varlist=['MaxPrecip_1d','precip'],
                    station='ecprecip', filetypes='hydro',
                    load_list=['name','varlist'], lproduct='outer')
    assert len(dss) == 4
    assert isinstance(ds, Dataset)
    assert dss[1].name == 'EC' and dss[2].name == 'ctrl-1'
    assert 'MaxPrecip_1d' in dss[0] and 'precip' in dss[1]
    assert 'MaxPrecip_1d' not in dss[3] and 'precip' not in dss[2]
    
  def testBasicLoadEnsembleTS(self):
    ''' test station data load functions (ensemble and list) '''
    from datasets.common import loadEnsembleTS    
    # test simple ensemble with basins
    names = ['GPCC', 'phys-ens_d01','max-ens-2100']; varlist = ['precip'] 
    aggregation = None; slices = dict(shape_name='ARB'); obsslices = dict(years=(1939,1945)) 
    shpens = loadEnsembleTS(names=names, season=None, shape='shpavg', aggregation=aggregation, 
                            slices=slices, varlist=varlist, filetypes=['hydro'], obsslices=obsslices)
    assert isinstance(shpens, Ensemble)
    assert shpens.basetype.__name__ == 'Dataset'
    assert all(shpens.hasVariable(varlist[0]))
    assert names[0] in shpens
    assert len(shpens[names[0]].time) == 72 # time-series
    assert len(shpens[names[-1]].time) == 720 # ensemble
    assert all('ARB' == ds.atts.shape_name for ds in shpens)
    # test list expansion of ensembles loading
    names = ['EC', 'phys-ens']; varlist = ['MaxPrecip_1d'] 
    prov = ['BC','AB']; season = ['summer','winter']; mode = ['max','min']
    constraints = dict(min_len=50, lat=(50,55), max_zerr=300,)
    enslst = loadEnsembleTS(names=names, prov=prov, season=season, mode=mode, station='ecprecip', 
                            constraints=constraints, varlist=varlist, filetypes=['hydro'], domain=2,
                            load_list=[('mode','season'),'prov',], lproduct='outer', lwrite=False)
    assert len(enslst) == 4
    assert all(isinstance(ens, Ensemble) for ens in enslst)
    assert all(ens.basetype.__name__ == 'Dataset' for ens in enslst)
    assert all(ens.hasVariable(varlist[0]) for ens in enslst)
    assert all('EC' in ens for ens in enslst)

  def testAdvancedLoadEnsembleTS(self):
    ''' test station data load functions (ensemble and list) '''
    from datasets.common import loadEnsembleTS 
    lwrite = False   
    # test ensemble (inner) list expansion
    names = 'CRU'; varlist = ['precip']; slices = dict(shape_name='FRB'); 
    obsslices = [dict(years=(1914,1918)), dict(years=(1939,1945))]
    name_tags = ['_1914','_1939']
    shpens = loadEnsembleTS(names=names, shape='shpavg', name_tags=name_tags, obsslices=obsslices,
                            slices=slices, varlist=varlist, filetypes=['hydro'],
                            aggregation=None, season=None, 
                            ensemble_list=['obsslices', 'name_tags'])
    assert isinstance(shpens, Ensemble)
    assert shpens.basetype.__name__ == 'Dataset'
    assert all(shpens.hasVariable(varlist[0]))
    assert all('CRU' == ds.name[:3] for ds in shpens)
    assert len(shpens['CRU_1914'].time) == 48 # time-series
    assert len(shpens['CRU_1939'].time) == 72 # time-series
    assert all('FRB' == ds.atts.shape_name for ds in shpens)
    # test ensemble (inner) list expansion with outer list expansion    
    varlist = ['MaxPrecip_1d']; constraints = dict(min_len=50, lat=(50,55), max_zerr=300,)
    # inner expansion
    names = ['EC', 'EC', 'cfsr-max']; name_tags = ['_1990','_1940','WRF_1990']
    obsslices = [dict(years=(1929,1945)), dict(years=(1979,1995)), dict()]
    # outer expansion
    prov = ['BC','AB']; season = ['summer','winter']; mode = ['max']
    # load data
    enslst = loadEnsembleTS(names=names, prov=prov, season=season, mode=mode, station='ecprecip', 
                            constraints=constraints, name_tags=name_tags, obsslices=obsslices,  
                            domain=2, filetypes=['hydro'], varlist=varlist, ensemble_product='inner',  
                            ensemble_list=['names','name_tags','obsslices',], lwrite=lwrite,
                            load_list=['mode','season','prov',], lproduct='outer',)
    assert len(enslst) == 4
    assert all(isinstance(ens, Ensemble) for ens in enslst)
    assert all(ens.basetype.__name__ == 'Dataset' for ens in enslst)
    assert all(ens.hasVariable(varlist[0]) for ens in enslst)
    assert all('EC_1990' in ens for ens in enslst)
    assert all('EC_1940' in ens for ens in enslst)
    assert all('WRF_1990' in ens for ens in enslst)
    # add CVDP data
    cvdp = loadEnsembleTS(names=names, prov=prov, season=season, mode=mode, 
                          name_tags=name_tags, obsslices=obsslices,  
                          varlist=['PDO'], ensemble_product='inner',  
                          ensemble_list=['names','name_tags','obsslices',], lwrite=lwrite,
                          load_list=['mode','season','prov',], lproduct='outer',
                          dataset_mode='CVDP')
    assert all(ens.hasVariable('PDO') for ens in enslst)
    # add PDO time-series to datasets
    for ts,cv in zip(enslst,cvdp):
      ts.addVariable(cv.PDO, lautoTrim=True)  
    all(ens.hasVariable('PDO') for ens in enslst)  
    # test slicing by PDO
    ds = enslst[0]['WRF_1990']
    slcds = ds(PDO=(-1,0.), lminmax=True)
    ## some debugging test
    # NetCDF datasets to add cluster_id to
    wrfensnc = ['max-ctrl','max-ens-A','max-ens-B','max-ens-C', # Ensembles don't have unique NetCDF files
                'max-ctrl-2050','max-ens-A-2050','max-ens-B-2050','max-ens-C-2050',
                'max-ctrl-2100','max-ens-A-2100','max-ens-B-2100','max-ens-C-2100',]
    wrfensnc = loadEnsembleTS(names=wrfensnc, name='WRF_NC', title=None, varlist=None, 
                              station='ecprecip', filetypes=['hydro'], domain=2, lwrite=lwrite)
    # climatology
    constraints = dict()
    constraints['min_len'] = 10 # for valid climatology
    constraints['lat'] = (45,60) 
    #constraints['max_zerr'] = 100 # can't use this, because we are loading EC data separately from WRF
    constraints['prov'] = ('BC','AB')
    wrfens = loadEnsembleTS(names=['max-ens','max-ens-2050','max-ens-2100'], name='WRF', title=None, 
                            varlist=None, 
                            aggregation='mean', station='ecprecip', constraints=constraints, filetypes=['hydro'], 
                            domain=2, lwrite=False)
    wrfens = wrfens.copy(asNC=False) # read-only DatasetNetCDF can't add new variables (not as VarNC, anyway...)    
#     gevens = [ens.fitDist(lflatten=True, axis=None) for ens in enslst]
#     print(''); print(gevens[0][0])

  def testLoadStandardDeviation(self):
    ''' test station data load functions (ensemble and list) '''
    from datasets.common import loadEnsembleTS
    # just a random function call that exposes a bug in Numpy's nanfunctions.py    
    slices = {'shape_name': 'FRB', 'years': (1979, 1994)}
    loadEnsembleTS(names='CRU', season=None, aggregation='SEM', slices=slices, 
                   varlist=['precip'], shape='shpavg', ldataset=True)
    # N.B.: the following link to a patched file should fix the problem:
    #  /home/data/Enthought/EPD/lib/python2.7/site-packages/numpy/lib/nanfunctions.py 
    #  -> /home/data/Code/PyGeoData/src/utils/nanfunctions.py
    # But diff first, to check for actual updates!
    # P/S at the moment I'm importing the custom nanfunctions directly
    
    
if __name__ == "__main__":

    
    specific_tests = []
#     specific_tests += ['ApplyAlongAxis']
#     specific_tests += ['AsyncPool']    
#     specific_tests += ['ExpArgList']
#     specific_tests += ['LoadDataset']
#     specific_tests += ['BasicLoadEnsembleTS']
#     specific_tests += ['AdvancedLoadEnsembleTS']
    specific_tests += ['LoadStandardDeviation']


    # list of tests to be performed
    tests = [] 
    # list of variable tests
#     tests += ['MultiProcess']
    tests += ['Datasets'] 
    

    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.iteritems():
      if key[-4:] == 'Test':
        test_classes[key[:-4]] = val


    # run tests
    report = []
    for test in tests: # test+'.test'+specific_test
      if len(specific_tests) > 0: 
        test_names = ['misc_test.'+test+'Test.test'+s_t for s_t in specific_tests]
        s = unittest.TestLoader().loadTestsFromNames(test_names)
      else: s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      report.append(unittest.TextTestRunner(verbosity=2).run(s))
      
    # print summary
    runs = 0; errs = 0; fails = 0
    for name,test in zip(tests,report):
      #print test, dir(test)
      runs += test.testsRun
      e = len(test.errors)
      errs += e
      f = len(test.failures)
      fails += f
      if e+ f != 0: print("\nErrors in '{:s}' Tests: {:s}".format(name,str(test)))
    if errs + fails == 0:
      print("\n   ***   All {:d} Test(s) successfull!!!   ***   \n".format(runs))
    else:
      print("\n   ###     Test Summary:      ###   \n" + 
            "   ###     Ran {:2d} Test(s)     ###   \n".format(runs) + 
            "   ###      {:2d} Failure(s)     ###   \n".format(fails)+ 
            "   ###      {:2d} Error(s)       ###   \n".format(errs))
    