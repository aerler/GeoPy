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
  exp = 'max'
  varlist = ['waterflx','runoff','snwmlt']
#   varlist = ['ugroff','runoff','sfroff']
  period = 10
  domain = 2
  filetypes = ['lsm','hydro']
  varatts = dict(Runoff=dict(name='runoff'))
  
  ## load data
  exp = WRF_exps[exp] # resolve short form
  # load WRF dataset
  dataset = loadWRF(experiment=exp, domains=domain, period=period, filetypes=filetypes, varlist=varlist, varatts=varatts)
  print dataset
  
  ## apply basin mask
  dataset.load()
  dataset.maskShape(name='Athabasca_River_Basin')
  print 
  
  # display
#   pyl.imshow(np.flipud(dataset.waterflx.getMapMask()))
#   pyl.colorbar(); 
  
  # scale factor
  S = ( 1 - dataset.Athabasca_River_Basin.getArray() ).sum() * (1e4)**2 / 1e3
  time = dataset.time.coord # time axis
  plotdata = []
  for var in varlist:
    # compute spatial average
    var = dataset.variables[var].mean(x=None,y=None)
    plotdata.append(time)
    plotdata.append(S*var.getArray())
    
    print
    print var.name, S*var.getArray().mean()
    
  # plot
  import pylab as pyl
  pyl.plot(*plotdata)
  pyl.legend(varlist)

  pyl.show(block=True)
  