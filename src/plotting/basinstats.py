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
mpl.rc('lines', linewidth=1.5)
mpl.rc('font', size=12)
# internal imports
# PyGeoDat stuff
from datasets.GPCC import loadGPCC
from datasets.CRU import loadCRU
from datasets.common import loadDatasets # for annotation
from plotting.settings import getFigureSettings, getVariableSettings
# ARB project related stuff
from plotting.ARB_settings import getARBsetup, arb_figure_folder, arb_map_folder

# settings
folder = arb_figure_folder
lprint = False 

## start computation
if __name__ == '__main__':
  
  ## settings
  explist = ['max']
  varlist = ['T2','Tmin','Tmax']; filetypes = ['srfc','xtrm']; lsum = False; leg = 2
#   varlist = ['p-et','precip','liqprec']; filetypes = ['hydro']; lsum = True; leg = 1
  varlist = ['waterflx','runoff','snwmlt','p-et','precip']; filetypes = ['srfc','hydro','lsm']; lsum = True; leg = 3
  varlist = ['waterflx','snwmlt','p-et','precip']; filetypes = ['srfc','hydro']; lsum = True; leg = 3
#   varlist = ['ugroff','runoff','sfroff']; filetypes = ['lsm']; lsum = True; leg = 2
  period = 10
  domain = 2
  grid='arb2_d02'
  varatts = None # dict(Runoff=dict(name='runoff'))
  
  ## load data  
  exps, titles = loadDatasets(explist, n=None, varlist=varlist, titles=None, periods=period, domains=domain, 
               grids='arb2_d02', resolutions=None, filetypes=filetypes, lWRFnative=True, ltuple=False)
  ref = exps[0]; nlen = len(exps)
  # observations
  cru = loadCRU(period=period, grid='arb2_d02', varlist=varlist, varatts=varatts)
  gpcc = loadGPCC(period=None, grid='arb2_d02', varlist=varlist, varatts=varatts)
  print ref
  
  ## apply basin mask
  for exp in exps:
    exp.load(); exp.maskShape(name='Athabasca_River_Basin')
  print 
  
  if len(cru.variables) > 0: 
    cru.load(); cru.maskShape(name='Athabasca_River_Basin')
  if len(gpcc.variables) > 0: 
    gpcc.load(); gpcc.maskShape(name='Athabasca_River_Basin')
  print 
  
  
  # display
#   pyl.imshow(np.flipud(dataset.Athabasca_River_Basin.getArray()))
#   pyl.imshow(np.flipud(dataset.precip.getMapMask()))
#   pyl.colorbar(); 
  # scale factor
  if lsum: S = ( 1 - ref.Athabasca_River_Basin.getArray() ).sum() * (1e4)**2 / 1e3
  else: S = 1.

  ## setting up figure
  # figure parameters for saving
  sf, figformat, margins, subplot, figsize = getFigureSettings(nlen, cbar=False)
  # make figure and axes
  f = pyl.figure(facecolor='white', figsize=figsize)
  axes = []
  for n in xrange(nlen):
    axes.append(f.add_subplot(subplot[0],subplot[1],n+1))
  f.subplots_adjust(**margins) # hspace, wspace
  
  # loop over axes
  for ax,exp in zip(axes,exps): 
    
    # make plots
    time = exp.time.coord # time axis
    plotdata = []; plotlegend = []
    # loop over vars    
    for var in varlist:
      # compute spatial average
      vardata = exp.variables[var].mean(x=None,y=None)
      plotdata.append(time)
      plotdata.append(S*vardata.getArray())    
      #plotlegend.append('%s (%s)'%(var,exp.name))
      plotlegend.append(var)
      print
      print exp.name, vardata.name, S*vardata.getArray().mean()
      if cru.hasVariable(var, strict=False):
        # compute spatial average for CRU
        vardata = cru.variables[var].mean(x=None,y=None)
        plotdata.append(time)
        plotdata.append(S*vardata.getArray())    
        plotdata.append('o')
        plotlegend.append('%s (%s)'%(var,cru.name))
        print cru.name, vardata.name, S*vardata.getArray().mean()
      if gpcc.hasVariable(var, strict=False):
        # compute spatial average for CRU
        vardata = gpcc.variables[var].mean(x=None,y=None)
        plotdata.append(time)
        plotdata.append(S*vardata.getArray())
        plotdata.append('x')    
        plotlegend.append('%s (%s)'%(var,gpcc.name))
        print gpcc.name, vardata.name, S*vardata.getArray().mean()
      # make plot
      ax.plot(*plotdata)
      pyl.legend(ax, plotlegend, loc=leg)
      pyl.title(ax, exp.name)

    
    
    # average discharge below Fort McMurray: 620 m^3/s
    
  # save figure to disk
  if lprint:
    print('\nSaving figure in '+filename)
    f.savefig(folder+filename, **sf) # save figure to pdf
    print(folder)
  
  ## show plots after all iterations
  pyl.show()
