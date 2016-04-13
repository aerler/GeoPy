'''
Created on 2010-11-12

Unittest for TPdef package (tropopause analysis tools and plot functions)

@author: Andre R. Erler
'''

import unittest
from matplotlib.pylab import show  


def modelOutput(var, all):
  # function to generate diagnostic output for a new variable  
  print
  print(' * '+var.atts['long_name']+' ['+var.atts['units']+']')
  print(var.axes)  
  if all:
    # compute slice only once and load in memory
    slicedVar = var(lon=(20,25),lat=(40,45)).load()
    # compute some stats
    min = slicedVar.min()
    mean = slicedVar.mean()
    max = slicedVar.max()
    # print output
    print
    print('    min: %.4f'%(min))    
    print('   mean: %.4f'%(mean))
    print('    max: %.4f'%(max))
    
class TPanalysis(unittest.TestCase):
  # whether or not to test lengthy computations
  all = True
  show = True
  dataset = 'ifs'  
  
  def setUp(self):
    from pygeode.atmdyn import meteoVar 
    # load main dataset      
    if self.dataset == 'lm':      
      from myDatasets.loadLM import openLM, allLM, LMroot, LCs, hostname
      # time-step/file definition for COSMO data
      LC = 'LC1' # smallest data set     
      self.rootFolder = LMroot + LCs[LC] + '/' # path to data set        
      # load COSMO data set
      if hostname=='komputer': 
        data = openLM(LC)
        data += meteoVar.Entropy(data.T, data.p) # add entropy   
      elif hostname=='erlkoenig': 
        data = allLM(LC=LC,tag='test',specialVars=['PV','PVs','s'],LMdata=False)
      # add more variables to dataset
      data += meteoVar.Theta(data.T, data.p)
      data += meteoVar.N2(data.th)
    elif self.dataset == 'ifs':
      from myDatasets.loadIFS import allIFS, IFSroot
#      # some pre-computed data
#      file = 'ecmwfTestData183JanNH.nc'  
      # load ECMWF-IPY data set      
      data = allIFS('JanNH', serverData=False)
      # add N2
      data += meteoVar.N2(data.s, z=data.z, entropy=True)
      self.rootFolder = IFSroot               
    # save dataset
    self.data = data
        
  def tearDown(self):
    pass

  def testOutput(self):
    # select a variable
    var = self.data.T # Entropy
    # print some diagnostics
    print; print
    print(' --- '+var.name+' --- ')      
    modelOutput(var, self.all)
          
  def testProfile(self):      
    # plot a profile
    from myPlots.plots import linePlot
    # select slice        
    if self.dataset == 'lm':
      slice1 = {'time':8*86400, 'lon':0,'lat':55,'z':(0,20e3)}
      slice2 = {'time':9*86400, 'lon':0,'lat':55,'z':(0,20e3)}      
      zAxis = self.data.z
    elif self.dataset == 'ifs':
      slice1 = {'time':3.5, 'lon':0,'lat':45,'eta':(30,91)}
      slice2 = {'time':3.5, 'lon':5,'lat':55,'eta':(30,91)}
      zAxis = self.data.phi
    ## first example
    # select some options
    kwargs = {}
    kwargs['clevs'] = [None, None, (150,450,10), None] # data plot limits / (0,7,7) 
    kwargs['labels'] = [None, '', ['Temperature', 'Pot. Temp.'], 'Pressure Profile'] # for legend / [ , ]
    kwargs['legends'] = [4, 3, 4, 1] # location; can also be kwargs / 
    # plot
#    f1 = linePlot([[self.data.T, self.data.th], [self.data.p]], [slice1, slice2], axis=zAxis, **kwargs)
    ## second example
    # select some options
    kwargs = {}
    kwargs['clevs'] = [None, None] # data plot limits / (0,7,7) 
    kwargs['labels'] = [None, ['Day 8', 'Day 9']] # for legend / [ , ]
    kwargs['legends'] = [3, 1] # location; can also be kwargs /
    # make new axis
    pAxis = self.data.p; pAxis.plotatts['plotorder'] = -1; pAxis.plotatts['plotscale'] = 'log' 
    # plot
    f2 = linePlot([[self.data.T], [self.data.p]], [[slice1, slice2]], axis=[[pAxis],[pAxis]], **kwargs)
    show()
    
  def testLine(self):      
    # plot a a horizontal line
    from myPlots.plots import linePlot
    # select slice        
    if self.dataset == 'lm':
      tSlice = {'time':(0,15*86400),'lon':0,'lat':55,'z':0}
      ySlice = {'time':9*86400, 'lon':0,'lat':(25,75),'z':10e3}      
    elif self.dataset == 'ifs':
      tSlice = {'time':(3,14), 'lon':0,'lat':45,'eta':91}
      ySlice = {'time':3.5, 'lon':5,'lat':(25,75),'eta':60}
    ## line plot example
    # select some options
    kwargs = {}
    kwargs['clevs'] = [None, None, None, None] # data plot limits / (0,7,7) 
    kwargs['labels'] = ['T, meridional', 'T, time-series', 'u, meridional', 'u, time-series'] # for legend / [ , ]
    kwargs['legends'] = [1, 3, 2, 4] # location; can also be kwargs / 
    # plot
    f = linePlot([[self.data.T], [self.data.u]], [ySlice, tSlice], axis=None, transpose=True, **kwargs)
    show()
      
  def testSynopCombi(self):
    # slice and projection setup
    days = 3
    lambConv = {'projection':'gall', 'llcrnrlon':-15, 'llcrnrlat':30, 'urcrnrlon':45, 'urcrnrlat':70,
                'parallels':[30,40,50,60,70], 'meridians':[-10,0,10,20,30,40]}
    # plot setup
    cbar = {'manual':False,'orientation':'horizontal'}
    cbls = [(4,16,7),(0,5,6)]
    clevs = [(4,16,10),(10,50,10),(00,300,10),(0,20,10)]
    sbplt = (2,2)
    # plot a map of TP height and sharpness 
    f = self.data.synop(['TPhgt','TPsharp','TPval','RMSE'], days ,clevs=clevs, subplot=sbplt, geos=lambConv) 
#    folder='/home/me/Research/Tropopause Definition/Figures/'
#    sf = dict(dpi=600,transparent=True)
#    f.savefig(folder+'HTPdN2day0304.pdf',**sf)
    show()

  def testSynopSingle(self):
    # slice and projection setup
    days=[3.0,3.25,3.5,3.75]
    lambConv = {'projection':'lcc', 'lat_0':50, 'lat_2':50, 'lon_0':15, 'width':4e7/6, 'height':4e7/8,
                'parallels':[20,40,60,80], 'meridians':[-30,0,30,60], 'labels':[1,0,0,1]}
    # plot setup
    colorbar = {'manual':True,'location':'right'}
    margins = {}#{'left':0.025,'wspace':0.025,'right': 0.95}
    cbls = 7
    clevs = (4,16,50)
    # plot a map of TP height and sharpness 
    f = self.data.synop('TPhgt', days, clevs=clevs,cbls=cbls,colorbar=colorbar,margins=margins,geos=lambConv) 
#    folder='/home/me/Research/Tropopause Definition/Figures/'
#    sf = dict(dpi=600,transparent=True)
#    f.savefig(folder+'HTPdN2day0304.pdf',**sf)
    show()
      
  def testHovmoeller(self):
    # make hovmoeller plot of TP height and TP sharpness
    # plot setup
    cbar = {'manual':True, 'location':'right', 'orientation':'vertical'}
    cbls = [(4,16,7),(0,50,6)]
    clevs = [(4,16,50),(0,50,50)]
    sbplt = (1,2)
    # plot a map of TP height and sharpness 
    f = self.data.synop(['TPhgt','TPsharp'], 0 ,clevs=clevs, cbls=cbls, subplot=sbplt, colorbar=cbar)
#    f = self.data.hovmoeller(['TPhgt','TPsharp'], slice={'lat':(30,70)},transpose=True,clevs=clevs,cbls=cbls,colorbar=colorbar) 
#    folder='/home/me/Research/Tropopause Definition/Figures/'
#    sf = dict(dpi=600,transparent=True)
#    f.savefig(folder+'HTPdN2hovmoeller.pdf',**sf)
    show()
    
def suite(tests=[]):    
  # creates a testsuite  
  if tests != []: # only include sets given in list 
    s = unittest.TestSuite(map(TPanalysis,tests))
  else: # test everything
    s = unittest.TestLoader().loadTestsFromTestCase(TPanalysis)
  return s

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLoad']
    #unittest.main(verbosity=1) 
    # list of test cases:
    # 'testProfile', 'testTPana', 'testSynop', 'testHovmoeller'
    tests = ['testSynopSingle']   
    # run tests
    unittest.TextTestRunner(verbosity=2).run(suite(tests))
    # show output
    show()