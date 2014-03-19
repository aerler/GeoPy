'''
Created on 2014-03-19

some useful functions to make map and surface plots that take advantage of variable meta data

@author: Andre R. Erler, GPL v3
'''

# external imports
from types import NoneType
import numpy as np
import matplotlib.pylab as pyl
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import ImageGrid
linewidth = .75
mpl.rc('lines', linewidth=linewidth)
if linewidth == 1.5: mpl.rc('font', size=12)
elif linewidth == .75: mpl.rc('font', size=8)
else: mpl.rc('font', size=10)
# prevent figures from closing: don't run in interactive mode, or plt.show() will not block
pyl.ioff()
# internal imports
from utils import getPlotValues, expandLevelList
from geodata.base import Variable
from geodata.misc import AxisError, ListError, VariableError



# function to plot 


# function to place (shared) colorbars at a specified figure margins
def sharedColorbar(fig, cf, clevs, colorbar, cbls, subplot, margins):
  loc = colorbar.pop('location','bottom')      
  # determine size and spacing
  if loc=='top' or loc=='bottom':
    orient = colorbar.pop('orientation','horizontal') # colorbar orientation
    je = subplot[1] # number of colorbars: number of rows
    ie = subplot[0] # number of plots per colorbar: number of columns
    cbwd = colorbar.pop('cbwd',0.025) # colorbar height
    sp = margins['wspace']
    wd = (margins['right']-margins['left'] - sp*(je-1))/je # width of each colorbar axis 
  else:
    orient = colorbar.pop('orientation','vertical') # colorbar orientation
    je = subplot[0] # number of colorbars: number of columns
    ie = subplot[1] # number of plots per colorbar: number of rows
    cbwd = colorbar.pop('cbwd',0.025) # colorbar width
    sp = margins['hspace']
    wd = (margins['top']-margins['bottom'] - sp*(je-1))/je # width of each colorbar axis
  shrink = colorbar.pop('shrinkFactor',1)
  # shift existing subplots
  if loc=='top': newMargin = margins['top']-margins['hspace'] -cbwd
  elif loc=='right': newMargin = margins['right']-margins['left']/2 -cbwd
  else: newMargin = 2*margins[loc] + cbwd    
  fig.subplots_adjust(**{loc:newMargin})
  # loop over variables (one colorbar for each)
  for i in range(je):
    if dir=='vertical': ii = je-i-1
    else: ii = i
    offset = (wd+sp)*float(ii) + wd*(1-shrink)/2 # offset due to previous colorbars
    # horizontal colorbar(s) at the top
    if loc == 'top': ci = i; cax = [margins['left']+offset, newMargin+margins['hspace'], shrink*wd, cbwd]             
    # horizontal colorbar(s) at the bottom
    elif loc == 'bottom': ci = i; cax = [margins['left']+offset, margins[loc], shrink*wd, cbwd]        
    # vertical colorbar(s) to the left (get axes reference right!)
    elif loc == 'left': ci = i*ie; cax = [margins[loc], margins['bottom']+offset, cbwd, shrink*wd]        
    # vertical colorbar(s) to the right (get axes reference right!)
    elif loc == 'right': ci = i*ie; cax = [newMargin+margins['wspace'], margins['bottom']+offset, cbwd, shrink*wd]
    # make colorbar 
    fig.colorbar(mappable=cf[ci],cax=fig.add_axes(cax),ticks=expandLevelList(cbls[i],clevs[i]),
                 orientation=orient,**colorbar)
  # return figure with colorbar (just for the sake of returning something) 
  return fig

if __name__ == '__main__':
    pass