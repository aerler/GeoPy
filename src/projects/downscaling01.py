'''
Created on 2014-02-14

Script to generate plots for my first downscaling paper!

@author: Andre R. Erler, GPL v3
'''


# external imports
from types import NoneType
import numpy as np
import matplotlib.pylab as pyl
import matplotlib as mpl
linewidth = 1.
mpl.rc('lines', linewidth=linewidth)
if linewidth == 1.5: mpl.rc('font', size=12)
else: mpl.rc('font', size=10)
# prevent figures from closing: don't run in interactive mode, or plt.show() will not block
pyl.ioff()
# internal imports
# PyGeoDat stuff
from geodata.base import Variable
from geodata.misc import AxisError, ListError
from datasets.WRF import loadWRF
from datasets.Unity import loadUnity
from datasets.WSC import Basin
from plotting.settings import getFigureSettings
# ARB project related stuff
from projects.ARB_settings import figure_folder


def linePlot(ax, varlist, linestyles=None, varatts=None, legend=None, xline=None, yline=None, 
             xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):
  ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on variable properties. '''
  # varlist is the list of variable objects that are to be plotted
  if isinstance(varlist,Variable): varlist = [varlist]
  elif not isinstance(varlist,(tuple,list)) or not all([isinstance(var,Variable) for var in varlist]): raise TypeError
  for var in varlist: var.squeeze() # remove singleton dimensions
  # linestyles is just a list of line styles for each plot
  if isinstance(linestyles,(basestring,NoneType)): linestyles = [linestyles]*len(varlist)
  elif not isinstance(linestyles,(tuple,list)): 
    if not all([isinstance(linestyles,basestring) for var in varlist]): raise TypeError
    if len(varlist) != len(linestyles): raise ListError, "Failed to match linestyles to varlist!"
  # varatts are variable-specific attributes that are parsed for special keywords and then passed on to the  
  if isinstance(varatts,dict):
    tmp = [varatts[var.name] if var.name in varatts else None for var in varlist]
    if any(tmp): varatts = tmp # if any variable names were found
    else: varatts = [varatts]*len(varlist) # assume it is one varatts dict, which will be used for all variables
  elif not isinstance(varatts,(tuple,list)): raise TypeError
  if not all([isinstance(atts,(dict,NoneType)) for atts in varatts]): raise TypeError
  # check axis: they need to have only one axes, which has to be the same for all!
  if len(varatts) != len(varlist): raise ListError, "Failed to match varatts to varlist!"  
  varname, varunits = var.name,var.units # determine variable properties from first variable
  axname, axunits = var.axes[0].name,var.axes[0].units # determine axes from first variable
  for var in varlist:
    if not var.ndim: raise AxisError, "Variable '{}' has more than one dimension.".format(var.name)
    if not var.hasAxis(axname): raise AxisError, "Variable {} does not have a '{}' axis.".format(var.name,axname)
    if not axunits == var.axes[0].units: raise AxisError, "Axis '{}' in Variable {} does not have a '{}' units.".format(axname,var.name,axunits)    
  # loop over variables
  flipxy = kwargs.pop('flipxy',False)
  plts = [] # list of plot handles
  for var,linestyle,varatt in zip(varlist,linestyles,varatts):
    axe = var.axes[0].getArray(unmask=True) # should only have one axis by now
    val = var.getArray(unmask=True) # the data to plot
    # figure out keyword options
    kwatts = kwargs.copy(); kwatts.update(varatt) # join individual and common attributes
    if 'label' not in kwatts: kwatts['label'] = var.name # default label: variable name
    if 'scalefactor' in kwatts: val *= kwatts.pop('scalefactor')
    # N.B.: other scaling behavior could be added here
    print var.name, var.units, axe.mean(), val.mean()
    # figure out orientation
    if flipxy: xx,yy = val, axe 
    else: xx,yy = axe, val
    # call plot function
    if linestyle is None: plts.append(ax.plot(xx, yy, **kwatts)[0])
    else: plts.append(ax.plot(xx, yy, linestyle, **kwatts)[0])
  # set axes labels
  labelpad = 3
  if flipxy:
    ax.set_xlabel('{} [{}]'.format(varname,varunits), labelpad=labelpad)
    ax.set_ylabel('{} [{}]'.format(axname,axunits), labelpad=labelpad)
  else: 
    ax.set_xlabel('{} [{}]'.format(axname,axunits), labelpad=labelpad)
    ax.set_ylabel('{} [{}]'.format(varname,varunits), labelpad=labelpad)
  # else: ax.set_xticklabels([])
  #ax.minorticks_on()
  ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    
  # add orientation lines
  if isinstance(xline,(int,np.integer,float,np.inexact)): ax.axhline(y=xline, color='black')
  elif isinstance(xline,dict): ax.axhline(**xline)
  if isinstance(yline,(int,np.integer,float,np.inexact)): ax.axvline(x=yline, color='black')
  elif isinstance(xline,dict): ax.axvline(**yline)
  # return handle
  return plts      


if __name__ == '__main__':
    
  ## runoff differences plot
  
  #settings
  basins = ['FRB','ARB']
  basins.reverse()
  exp = 'max-ens'
  period = 15
  grid = 'arb2_d02'
  # figure
  lprint = False
  # figure parameters for saving
  sf, figformat, margins, subplot, figsize = getFigureSettings(2, cbar=False, sameSize=False)
  # make figure and axes
  fig, axes = pyl.subplots(*subplot, sharex=True, sharey=False, facecolor='white', figsize=figsize)
  margins = dict(bottom=0.11, left=0.11, right=.975, top=.95, hspace=0.05, wspace=0.05)
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
    discharge = gage.discharge
    difference = sfroff - discharge
    difference.name = 'difference'    
#     print sfroff.name, sfroff.units
    
    # plot properties    
    varatts = dict()
    varatts['runoff'] = dict(color='purple')
    varatts['sfroff'] = dict(color='green')
    varatts['discharge'] = dict(color='green', linestyle='', marker='o')
    varatts['difference'] = dict(color='red')    
    # plot runoff
    varlist = [discharge]
    varlist += [runoff, sfroff, difference]
    plts = linePlot(ax, varlist, varatts=varatts, xline=0, scalefactor=1e-6)
                
  if lprint:
    filename = 'runoff_test.png'
    print('\nSaving figure in '+filename)
    fig.savefig(figure_folder+filename, **sf) # save figure to pdf
    print(figure_folder)
      
  ## show plots after all iterations  
  pyl.show()
    