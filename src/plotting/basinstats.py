'''
Created on 2013-11-14

A simple script to plot basin-averaged monthly climatologies. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import matplotlib.pylab as pyl
import matplotlib as mpl
mpl.rc('lines', linewidth=1.)
mpl.rc('font', size=10)
# internal imports
# PyGeoDat stuff
from datasets.WRF import loadWRF
from datasets.WRF_experiments import exps as WRF_exps
# from datasets.CESM import loadCESM, CESMtitle
from datasets.CFSR import loadCFSR
from datasets.NARR import loadNARR
from datasets.GPCC import loadGPCC
from datasets.CRU import loadCRU
from datasets.PRISM import loadPRISM
from datasets.common import days_per_month, days_per_month_365 # for annotation
from plotting.settings import getFigureSettings, getVariableSettings
# ARB project related stuff
from plotting.ARB_settings import getARBsetup, arb_figure_folder, arb_map_folder

## helper functions

## start computation
if __name__ == '__main__':
  
  ## settings
  exp = 'noah'
  varlist = ['T2','Tmin','Tmax']; lsum = False; leg = 2
#   varlist = ['p-et','precip','liqprec']; lsum = True; leg = 1
#   varlist = ['waterflx','runoff','snwmlt','p-et','precip']; lsum = True; leg = 3
#   varlist = ['ugroff','runoff','sfroff']
  period = 10
  domain = 2
  filetypes = ['srfc','xtrm','hydro']
  varatts = None # dict(Runoff=dict(name='runoff'))
  
  ## load data
  exp = WRF_exps[exp] # resolve short form
  # load WRF dataset
  dataset = loadWRF(experiment=exp, domains=domain, period=period, filetypes=filetypes, varlist=varlist, varatts=varatts)
  # observations
  cru = loadCRU(period=period, grid='arb2_d02', varlist=varlist, varatts=varatts)
  gpcc = loadGPCC(period=None, grid='arb2_d02', varlist=varlist, varatts=varatts)
  print dataset
  
  ## apply basin mask
  dataset.load(); cru.load(); gpcc.load()
  print 
  dataset.maskShape(name='Athabasca_River_Basin')
  if len(cru.variables) > 0: cru.maskShape(name='Athabasca_River_Basin')
  if len(gpcc.variables) > 0: gpcc.maskShape(name='Athabasca_River_Basin')
  print 
  
  # display
#   pyl.imshow(np.flipud(dataset.Athabasca_River_Basin.getArray()))
#   pyl.imshow(np.flipud(dataset.precip.getMapMask()))
#   pyl.colorbar(); 
  
  # scale factor
  if lsum: S = ( 1 - dataset.Athabasca_River_Basin.getArray() ).sum() * (1e4)**2 / 1e3
  else: S = 1.
  time = dataset.time.coord # time axis
  plotdata = []; plotlegend = []
  for var in varlist:
    # compute spatial average
    vardata = dataset.variables[var].mean(x=None,y=None)
    plotdata.append(time)
    plotdata.append(S*vardata.getArray())    
    plotlegend.append('%s (%s)'%(var,exp.name))
    print
    print exp.name, vardata.name, S*vardata.getArray().mean()
    if cru.hasVariable(var, strict=False):
      # compute spatial average for CRU
      vardata = cru.variables[var].mean(x=None,y=None)
      plotdata.append(time)
      plotdata.append(S*vardata.getArray())    
      plotdata.append('-.')
      plotlegend.append('%s (%s)'%(var,cru.name))
      print cru.name, vardata.name, S*vardata.getArray().mean()
    if gpcc.hasVariable(var, strict=False):
      # compute spatial average for CRU
      vardata = gpcc.variables[var].mean(x=None,y=None)
      plotdata.append(time)
      plotdata.append(S*vardata.getArray())
      plotdata.append('--')    
      plotlegend.append('%s (%s)'%(var,gpcc.name))
      print gpcc.name, vardata.name, S*vardata.getArray().mean()
      
    
    
    # average discharge below Fort McMurray: 620 m^3/s
    
  # plot
  import pylab as pyl
  pyl.plot(*plotdata)
  pyl.legend(plotlegend, loc=leg)

  pyl.show(block=True)
  