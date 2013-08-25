'''
Created on 2010-11-12, adapted on 2013-08-24 

Unittest for the atmdyn package; mainly for meteoVar, etaVar, f2pyVar, and srfcVar 
(the code was adapted from the PyGeode plugin AtmDyn).

@author: Andre R. Erler, GPL v3
'''

import unittest
import matplotlib as mpl
#from matplotlib.pyplot import rcParams
#mpl.rc('lines', linewidth=1.5)
#mpl.rc('font', size=22)
axlbl = dict(labelsize='large')
mpl.rc('axes', **axlbl)
mpl.rc('xtick', **axlbl)
mpl.rc('ytick', **axlbl) 
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)
import matplotlib.pylab as pyl
# my own imports

from atmdyn import meteoVar, etaVar, f2pyVar, srfcVar


class Variables(unittest.TestCase):  
  all = False # whether or not to test lengthy computations
  plot = False # whether or not to display plots
  dataset = 'CRU' # dataset to use (some tests require a specific dataset)
  slice = {'time':1, 'lon':(20,30),'lat':(30,40)}
  
  def setUp(self):
    if self.dataset == 'CRU':
      from myDatasets.loadLM import openLM, allLM, LMroot, LCs, hostname
      # time-step/file definition for COSMO data
      LC = 'LC1' # smallest data set      
      self.slice = self.sliceLM
      # load COSMO data set
      if hostname=='komputer': self.data = openLM(LC)
      elif hostname=='erlkoenig': 
        self.data = allLM(LC=LC,tag='test',specialVars=[],LMdata=True) # 'PV','PVs','s'
      # save root folder for later 
      self.rootFolder = LMroot + LCs[LC] + '/'
#     # add potential temperature (because it's so ubiquitous) 
#     if ('T' in self.data.vardict) and ('p' in self.data.vardict):   
#       self.data += f2pyVar.Theta(self.data.T, self.data.p) # use Fortran implementation  
    
  def tearDown(self):     
    pass
  
  def output(self, var, all, slice=None):
    # function to generate diagnostic output for a new variable  
    print
    print
    print(' * '+var.atts['long_name']+' ['+var.atts['units']+']')
    print(var.axes)  
    if all:
      # compute slice only once and load in memory
      if slice is None: slice = self.slice
      slicedVar = var(**slice).load()
      # compute some stats
      min = slicedVar.min()
      mean = slicedVar.mean()
      max = slicedVar.max()
      # print output
      print
      print('    min: %g'%(min))    
      print('   mean: %g'%(mean))
      print('    max: %g'%(max))
      
  def makePlot(self, var, z=None, plotAxis='', slice=None, xlim=None, ylim=None):
    from numpy import mean
    # figure out slice
    if not slice: slice = self.slice
    # determine plot dimension
    if not plotAxis:
      vertical = True
      if self.dataset=='lm': slice['z'] = (0,20e3); plotAxis = 'z' 
      elif self.dataset=='ifs': slice['eta'] = (30,91); plotAxis = 'eta'
    else:
      if plotAxis=='z' or plotAxis=='eta': vertical = True
      else: vertical = False   
    # collapse all dimensions except plotAxis 
    for dim,lim in slice.iteritems():
      if not dim==plotAxis: slice[dim] = mean(lim)  
    # construct plot
    ax = pyl.axes()
#    if not vertical:
#      if self.dataset=='lm': 
#        plotvar(var(**self.slice), ax=ax, lblx=True)
#        if self.dataset=='lm': ax.set_ylim((0,20))       
#      elif self.dataset=='ifs':
#        if not z: z  = self.data.z
#        vv = var(**self.slice).squeeze().get()*var.plotatts['scalefactor']+var.plotatts['offset']
#        if var.hasaxis(z.name): 
#          z(**{z.name:self.slice[z.name]}).values*z.plotatts['scalefactor']+z.plotatts['offset']
#        else: zz = z(**self.slice).squeeze().get()*z.plotatts['scalefactor']+z.plotatts['offset'] 
#        pyl.plot(vv,zz)
#        if z.name=='z': ax.set_ylim((0,20)) 
#    else: 
    var.plotatts['plotname'] = r'$\theta$'
    plotvar(var(**self.slice), ax=ax, lblx=True, lbly=True)
    # axes limits
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    pyl.show()
    # return axes for modification...
    return ax

  ## test cases

  def testLoad(self):
    # print final dataset
    print(self.data)
    
  def testIntersect(self):
    from pygeode.formats.netcdf import open     
    from myDatasets.utils import intersectDatasets
    # load new test data
    file = "TPfittest.nc"
    newData = open(self.rootFolder+file)
    # merge datasets with different slicing
    mergedData = intersectDatasets(newData,self.data)
    print(mergedData)
    
  def testTemperature(self):
    # print diagnostics
    self.output(self.data.T,self.all)
    # test plot
    if self.plot: self.makePlot(self.data.T, xlim=(150,350))
    
  def testPressure(self):
    # this method is specific to hybrid coordinates
    if self.dataset == 'ifs':
      # print pressure on eta levels diagnostics
      self.output(self.data.p,self.all)
      # test plot: either as function of eta or z
#      if self.plot: self.makePlot(self.data.p,z=self.data.eta,plotAxis='eta')
      if self.plot: self.makePlot(self.data.p)
    
  def testGeopotHeight(self):
    # this method is specific to hybrid coordinates
    if self.dataset == 'ifs':
      # compute geopotential on eta levels
      z = etaVar.GeopotHeight(self.data.T, self.data.ps, self.data.zs)
      # print diagnostics
      self.output(self.data.z,self.all)
      # test plot: this should be a strait 45 degree line!
      if self.plot: self.makePlot(self.data.z, xlim=(0,20))
    
  def testVerticalVelocity(self):  
    # print vertical velocity diagnostics
    if self.dataset == 'lm':
#      # interpolate vertical velocity to full levels  
#      vv = meteoVar.verticalVelocity(self.data.w, self.data.z)
      assert self.data.w.hasaxis('z')
      assert self.data.w.axes == self.data.u.axes
      # w is added to dataset upon load
      self.output(self.data.w,self.all)
    
  def testRho(self):  
    # print density diagnostics
    rho = meteoVar.Rho(self.data.T, self.data.p)
    self.output(rho,self.all)
    
  def testF2pyTheta(self):
    # Fortran implementation of theta (mainly for test purpose)
    theta = f2pyVar.Theta(self.data.T, self.data.p)
    # print potential temperature diagnostics
    self.output(theta,self.all)
    # test plot
    if self.plot: self.makePlot(theta, xlim=(250,550))
      
  def testIsentropicSrfc(self):  
    # trim
    z = self.data.p(z=(4e3,16e3))
    th = self.data.th(z=(4e3,16e3))  
    # interpolate Z to isentrope
    Z320K = srfcVar.interp2theta(z, th, [320])
    # print potential temperature diagnostics
    self.output(Z320K,self.all)
    # test plot
    if self.plot: self.makePlot(Z320K, plotAxis='lat', ylim=(0,20))
    
  def testTheta(self):  
    # print potential temperature diagnostics
    th = meteoVar.Theta(self.data.T, self.data.p)
    self.output(th,self.all)
    
  def testEntropy(self):  
    # compute entropy
    s = meteoVar.Entropy(self.data.T, self.data.p)
    # print entropy diagnostics    
    self.output(s,self.all)
    
  def testLR(self):  
    # compute temperature lapse-rate
    lr = meteoVar.LR(self.data.T,z=self.data.z)  
    self.output(lr,self.all)
    
  def testThetaLR(self):      
    # compute potential temperature lapse-rate
    thlr = meteoVar.ThetaLR(self.data.th,z=self.data.z)  
    self.output(thlr,self.all)
    
  def testN2(self):  
    # compute Brunt-Vaisaila Frequency Squared
    nn = meteoVar.N2(self.data.th,z=self.data.z)  
    self.output(nn,self.all)
    # test plot
    if self.plot: self.makePlot(nn, xlim=(0,7))
      
  def testF2pyZeta(self):
    # compute relative vorticity 
    zeta = f2pyVar.RelativeVorticity(self.data.u, self.data.v)
    # print diagnostics
    self.output(zeta,self.all, self.slice)
    # test plot
    if self.plot: self.makePlot(zeta, xlim=(-1,1))
    
  def testF2pyPV(self):
    # some required fields
    rho = meteoVar.Rho(self.data.T, self.data.p)
    # Fortran implementation of PV
    if self.dataset=='lm':
      PV = f2pyVar.PotentialVorticity(self.data.u, self.data.v, self.data.th, rho, w=self.data.w)
    elif self.dataset=='ifs':
      PV = f2pyVar.PotentialVorticity(self.data.u, self.data.v, self.data.th, rho, z=self.data.z)
    # print potential temperature diagnostics
    self.output(PV,self.all, self.slice)
    # test plot
    if self.plot: self.makePlot(PV, xlim=(0,10))
      
  def testF2pyWMOTP(self):
    # compute TP height
    slice = {'lat':(30,70),'lon':(0,1)}
    if self.dataset=='lm':
      bnd=(4e3,18e3)
      zTP = srfcVar.WMOTP(self.data.T(z=bnd,**slice), axis=self.data.z(z=bnd), bnd=bnd)
    elif self.dataset=='ifs':
      bnd = (35,75)
      zTP = srfcVar.WMOTP(self.data.T(eta=bnd,**slice), z=self.data.z(eta=bnd,**slice), bnd=bnd)
    # print potential temperature diagnostics
    self.output(zTP,self.all,slice)
    # test plot
    if self.plot: self.makePlot(zTP, plotAxis='lat', ylim=(0,20))
    
def suite(tests=[]):    
  # creates a testsuite  
  if tests != []: # only include sets given in list 
    s = unittest.TestSuite(map(Variables,tests))
  else: # test everything
    s = unittest.TestLoader().loadTestsFromTestCase(Variables)
  return s

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLoad']
    #unittest.main(verbosity=1) 
    # list of test cases:
    # 'testLoad', 'testPressure', 'testGeopotHeight', 'testLR', 'testTheta', 'testThetaLR', 'testEntropy', 'testN2'
    tests = ['testF2pyPV']   
    # run tests
    unittest.TextTestRunner(verbosity=2).run(suite(tests))