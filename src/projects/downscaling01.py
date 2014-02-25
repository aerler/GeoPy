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
from mpl_toolkits.axes_grid1 import ImageGrid
linewidth = 1.
mpl.rc('lines', linewidth=linewidth)
if linewidth == 1.5: mpl.rc('font', size=12)
else: mpl.rc('font', size=10)
# prevent figures from closing: don't run in interactive mode, or plt.show() will not block
pyl.ioff()
# internal imports
# PyGeoDat stuff
from geodata.base import Variable
from geodata.misc import AxisError, ListError, VariableError
from datasets.WRF import loadWRF
from datasets.Unity import loadUnity
from datasets.WSC import Basin
# from plotting.settings import getFigureSettings
# ARB project related stuff
from projects.ARB_settings import figure_folder

# method to check units and name, and return scaled plot value
def getPlotValues(var, checkunits=None, checkname=None):
  ''' Helper function to check variable/axis, get (scaled) values for plot, and return appropriate units. '''
  if var.plot is not None and 'plotname' in var.plot: 
    varname = var.plot['plotname'] 
    if checkname is not None and varname != checkname: # only check plotname! 
      raise VariableError, "Expected variable name '{}', found '{}'.".format(checkname,varname)
  else: varname = var.atts['name']
  val = var.getArray(unmask=True) # the data to plot
  if var.plot is not None and 'scalefactor' in var.plot: 
    val *= var.plot['scalefactor']
    varunits = var.plot.plotunits
  else: varunits = var.atts.units
  if checkunits is not None and  varunits != checkunits: 
    raise VariableError, "Units for variable '{}': expected {}, found {}.".format(var.name,checkunits,varunits) 
  if var.plot is not None and 'offset' in var.plot: val += var.plot['offset']    
  # return values, units, name
  return val, varunits, varname     

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
  axname = varlist[0].axes[0].name
  for var in varlist:
    if not var.ndim: raise AxisError, "Variable '{}' has more than one dimension.".format(var.name)
    if not var.hasAxis(axname): raise AxisError, "Variable {} does not have a '{}' axis.".format(var.name,axname)
  # loop over variables
  flipxy = kwargs.pop('flipxy',False)
  plts = []; varname = None; varunits = None; axname = None; axunits = None # list of plot handles
  for var,linestyle,varatt in zip(varlist,linestyles,varatts):
    axe, axunits, axname = getPlotValues(var.axes[0], checkunits=axunits, checkname=axname)
    val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=varname)
    # figure out keyword options
    kwatts = kwargs.copy(); kwatts.update(varatt) # join individual and common attributes     
    if 'label' not in kwatts: kwatts['label'] = var.name # default label: variable name
    # N.B.: other scaling behavior could be added here
    print varname, varunits, val.mean()
    # figure out orientation
    if flipxy: xx,yy = val, axe 
    else: xx,yy = axe, val
    # call plot function
    if linestyle is None: plts.append(ax.plot(xx, yy, **kwatts)[0])
    else: plts.append(ax.plot(xx, yy, linestyle, **kwatts)[0])
  # set axes limits
  if isinstance(xlim,(list,tuple)) and len(xlim)==2: ax.set_xlim(*xlim)
  elif xlim is not None: raise TypeError
  if isinstance(ylim,(list,tuple)) and len(ylim)==2: ax.set_ylim(*ylim)
  elif ylim is not None: raise TypeError  
  # set axes labels
  xpad = 2; ypad = -2
  if flipxy:
    ax.set_xlabel('{} [{}]'.format(varname,varunits), labelpad=xpad)
    ax.set_ylabel('{} [{}]'.format(axname,axunits), labelpad=ypad)
  else: 
    ax.set_xlabel('{} [{}]'.format(axname,axunits), labelpad=xpad)
    ax.set_ylabel('{} [{}]'.format(varname,varunits), labelpad=ypad)
  # make monthly ticks
  if axname == 'time' and axunits == 'month':
    #ax.minorticks_on()
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
  # add legend
  if isinstance(legend,dict): ax.legend(**legend)
  elif isinstance(legend,(int,np.integer,float,np.inexact)): ax.legend(loc=legend)
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
#   basins.reverse()
  exp = 'max-ens'
  period = 15
  grid = 'arb2_d02'
  # figure
  lprint = True
  lfield = True
  lgage = True 
  ldisc = True # scale sfroff to discharge
  lprecip = True # scale sfroff by precip bias
  # figure parameters for saving
#   sf, figformat, margins, subplot, figsize = getFigureSettings(2, cbar=False, sameSize=False)
  # make figure and axes
#   fig, axes = pyl.subplots(*subplot, sharex=True, sharey=False, facecolor='white', figsize=figsize)
#   margins = dict(bottom=0.11, left=0.11, right=.975, top=.95, hspace=0.05, wspace=0.05)
#   fig.subplots_adjust(**margins) # hspace, wspace
  nax = len(basins)
  fig = pyl.figure(1, (5.5, 3.5))
  axes = ImageGrid(fig, (0.09,0.11,0.88,0.85), nrows_ncols = (1, nax), axes_pad = 0.2, aspect=False, label_mode = "L")
              
  # loop over panels/basins
  for n,ax,basin in zip(xrange(nax),axes,basins):
#   for basin in basins:
    
    # load meteo data
    if lfield:
      fullwrf = loadWRF(experiment=exp, domains=2, period=period, grid=grid, 
                    varlist=['precip','runoff','sfroff'], filetypes=['srfc','lsm']) # WRF
      fullunity = loadUnity(period=period, grid=grid, varlist=['precip'])
      fullwrf.load(); fullunity.load()
    
    # load basin data
    basin = Basin(basin=basin)
    if lgage: gage = basin.getMainGage()
    # mask fields and get time series
    if lfield: 
      mask = basin.rasterize(griddef=fullwrf.griddef)
      # average over basins
      wrf = fullwrf.mapMean(basin, integral=True)
      unity = fullunity.mapMean(basin, integral=True)
    # load/compute variables
    varlist = []
    if lgage: 
      discharge = gage.discharge      
      varlist += [discharge]
    if lfield:
      runoff = wrf.runoff; runoff.units = discharge.units
      sfroff = wrf.sfroff; sfroff.units = discharge.units
      varlist += [runoff, sfroff]
      if ldisc:
        s_sfroff = sfroff.copy(deepcopy=True)
        s_sfroff.name = 'scaled sfroff'
        s_sfroff *= discharge.getArray().mean()/sfroff.getArray().mean()
        print s_sfroff 
        varlist += [s_sfroff]
      elif lprecip:
        assert unity.precip.units == wrf.precip.units
        scale = unity.precip.getArray().mean()/wrf.precip.getArray().mean()
        s_sfroff = sfroff.copy(deepcopy=True); s_sfroff *= scale; s_sfroff.name = 'scaled sfroff'
        s_runoff = runoff.copy(deepcopy=True); s_runoff *= scale; s_runoff.name = 'scaled runoff'
        varlist += [s_sfroff, s_runoff]
    if lfield and lgage:
      difference = sfroff - discharge
      difference.name = 'difference'
      varlist += [difference]
      if ldisc or lprecip:
        s_difference = s_sfroff - discharge
        s_difference.name = 'scaled difference'      
        varlist += [s_difference]    
    for var in varlist: var.plot = discharge.plot # harmonize plotting
    #print sfroff.plot.name, sfroff.plot.units
    
    # plot properties    
    varatts = dict()
    varatts['runoff'] = dict(color='purple', linestyle='--')
    varatts['sfroff'] = dict(color='green', linestyle='--')
    varatts['discharge'] = dict(color='green', linestyle='', marker='o')
    varatts['difference'] = dict(color='red', linestyle='--')
    # add scaled variables
    satts = {}
    for key,val in varatts.iteritems():
      val = val.copy(); val['linestyle'] = '-'
      satts['scaled '+key] = val
    varatts.update(satts)
    # determine legend
    if n == 0: legend = None
    else: legend = dict(loc=1, labelspacing=0.125, handlelength=2, handletextpad=0.5, fancybox=True)
    # plot runoff
    plts = linePlot(ax, varlist, varatts=varatts, xline=0, ylim=(-6,16), legend=legend) # , scalefactor=1e-6
                
  if lprint:
    if ldisc: filename = 'runoff_discharge.png'
    elif lprecip: filename = 'runoff_precip.png'
    else: filename = 'runoff_test.png'
    print('\nSaving figure in '+filename)
    fig.savefig(figure_folder+filename, dpi=150) # save figure to pdf
    print(figure_folder)
      
  ## show plots after all iterations  
  pyl.show()
    