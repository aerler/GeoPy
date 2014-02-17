'''
Created on 2014-02-14

Script to generate plots for my first downscaling paper!

@author: Andre R. Erler, GPL v3
'''


# external imports
import numpy as np
import matplotlib.pylab as pyl
import matplotlib as mpl
linewidth = 1.
mpl.rc('lines', linewidth=linewidth)
if linewidth == 1.5: mpl.rc('font', size=12)
else: mpl.rc('font', size=10)
# internal imports
# PyGeoDat stuff
from datasets.WRF import loadWRF
from datasets.Unity import loadUnity
from datasets.WSC import Basin
from plotting.settings import getFigureSettings
# ARB project related stuff
from projects.ARB_settings import figure_folder

if __name__ == '__main__':
    
  ## runoff differences plot
  
  #settings
  basins = ['FRB','ARB']
  basins.reverse()
  exp = 'max-ens'
  period = 15
  grid = 'arb2_d02'
  # figure
  # figure parameters for saving
  sf, figformat, margins, subplot, figsize = getFigureSettings(2, cbar=False, sameSize=False)
  # make figure and axes
  fig, axes = pyl.subplots(*subplot, sharex=True, sharey=False, facecolor='white', figsize=figsize)
  fig.subplots_adjust(**margins) # hspace, wspace
              
  # loop over panels/basins
  for ax,basin in zip(axes,basins):
#   for basin in basins:
    
    # load meteo data
    fullwrf = loadWRF(experiment=exp, domains=2, period=period, grid=grid, 
                  varlist=['precip','runoff','sfroff'], filetypes=['srfc','lsm']) # WRF
    fullunity = loadUnity(period=period, grid=grid, varlist=['precip'])
    fullwrf.load(); fullunity.load()
    
    # load basin data
    basin = Basin(basin=basin)
    gage = basin.getMainGage()
    # mask fields and get time series
    mask = basin.rasterize(griddef=fullwrf.griddef)
    #basinarea = (1-mask).sum()*100. # 100 km^2 per grid point    
    # average over basins
    wrf = fullwrf.mapMean(basin, integral=True)
    unity = fullunity.mapMean(basin, integral=True)
    # compute misfit
    runoff = wrf.runoff; runoff.units = 'kg/s'
    sfroff = wrf.sfroff; sfroff.units = 'kg/s'
    gagero = gage.discharge
#     print sfroff.name, sfroff.units
#     print gagero.name, gagero.units
    rodiff = sfroff - gagero    
    
    def linePlot(ax, var, scalefactor=1, linestyle=None, **kwargs):
      ''' simple function to draw a line plot based on a variable '''
      print var.name, var.units
      xx = var.axes[0].getArray(unmask=True) # should only have one axis by now
      yy = var.getArray(unmask=True)*scalefactor # the data to plot
      print xx.mean(), yy.mean()
      if linestyle is None: plt = ax.plot(xx, yy, **kwargs)[0]
      else: plt = ax.plot(xx, yy, linestyle, **kwargs)[0]
      # return handle
      return plt      
    
    # plot runoff
    plt = []; leg = []
    # WRF runoff    
    plt.append(linePlot(ax, runoff, scalefactor=1e-6, color='purple'))
    plt.append(linePlot(ax, sfroff, scalefactor=1e-6, color='green'))
    plt.append(linePlot(ax, gagero, scalefactor=1e-6, linestyle='o', color='green'))
    plt.append(linePlot(ax, rodiff, scalefactor=1e-6, color='red'))
#     leg.append(label)
                
    
  filename = 'runoff_test.png'
  print('\nSaving figure in '+filename)
  fig.savefig(figure_folder+filename, **sf) # save figure to pdf
  print(figure_folder)
      
  ## show plots after all iterations
  pyl.show()

    