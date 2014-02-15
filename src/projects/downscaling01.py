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
  exp = 'max-ens'
  period = 15
  grid = 'arb2_d02'
  # figure
  # figure parameters for saving
  sf, figformat, margins, subplot, figsize = getFigureSettings(2, cbar=False)
  # make figure and axes
  fig, axes = pyl.subplots(*subplot, sharex=True, sharey=True, facecolor='white', figsize=figsize)
  fig.subplots_adjust(**margins) # hspace, wspace
              
  # loop over panels/basins
  for basin in basins:
    
    # load basin data
    basin = Basin(basin=basin)
    gage = basin.getMainGage()
    # load meteo data
    wrf = loadWRF(experiment=exp, domains=2, period=period, grid=grid, 
                  varlist=['precip','runoff','sfroff'], filetypes=['srfc','lsm']) # WRF
    unity = loadUnity(period=period, grid=grid, varlist=['precip'])
    # mask fields and get time series
    mask = basin.rasterize(griddef=wrf.griddef)
    basinarea = (1-mask).sum()*100. # 100 km^2 per grid point
    wrf.load(); wrf.mask(mask)
    unity.load(); unity.mask(mask)
    
    
    