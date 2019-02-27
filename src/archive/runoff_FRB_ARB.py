'''
Created on 2014-02-14

Script to generate plots for my first downscaling paper!

@author: Andre R. Erler, GPL v3
'''
# external imports
# from types import NoneType
# from warnings import warn
# import numpy as np
# use common MPL instance
# import matplotlib as mpl
import matplotlib.pylab as pyl
# from plotting.legacy import loadMPL
# mpl,pyl = loadMPL(linewidth=.75)
# from plotting.misc import loadStyleSheet
# loadStyleSheet(stylesheet='myggplot', lpresentation=False, lpublication=True)
# from mpl_toolkits.axes_grid1 import ImageGrid
# internal imports
# PyGeoDat stuff
# from utils.signalsmooth import smooth
from plotting.figure import getFigAx
# from plotting.misc import getPlotValues
# from geodata.base import Variable
# from geodata.misc import AxisError, ListError
from datasets.WRF import loadWRF_ShpEns
from datasets.Unity import loadUnity_ShpTS
from datasets.WSC import basins as basin_dict
# ARB project related stuff
from projects.ARB_settings import figure_folder

# def linePlot(varlist, ax=None, fig=None, linestyles=None, varatts=None, legend=None,
#               xline=None, yline=None, title=None, flipxy=None, xlabel=None, ylabel=None, xlim=None,
#               ylim=None, lsmooth=False, lprint=False, **kwargs):
#   ''' A function to draw a list of 1D variables into an axes, and annotate the plot based on variable properties. '''
#   warn('Deprecated function: use Figure or Axes class methods.')
#   # create axes, if necessary
#   if ax is None: 
#     if fig is None: fig,ax = getFigAx(1) # single panel
#     else: ax = fig.axes[0]
#   # varlist is the list of variable objects that are to be plotted
#   #print varlist
#   if isinstance(varlist,Variable): varlist = [varlist]
#   elif not isinstance(varlist,(tuple,list)) or not all([isinstance(var,Variable) for var in varlist]): raise TypeError
#   for var in varlist: var.squeeze() # remove singleton dimensions
#   # linestyles is just a list of line styles for each plot
#   if isinstance(linestyles,(basestring,NoneType)): linestyles = [linestyles]*len(varlist)
#   elif not isinstance(linestyles,(tuple,list)): 
#     if not all([isinstance(linestyles,basestring) for var in varlist]): raise TypeError
#     if len(varlist) != len(linestyles): raise ListError, "Failed to match linestyles to varlist!"
#   # varatts are variable-specific attributes that are parsed for special keywords and then passed on to the
#   if varatts is None: varatts = [dict()]*len(varlist)  
#   elif isinstance(varatts,dict):
#     tmp = [varatts[var.name] if var.name in varatts else dict() for var in varlist]
#     if any(tmp): varatts = tmp # if any variable names were found
#     else: varatts = [varatts]*len(varlist) # assume it is one varatts dict, which will be used for all variables
#   elif not isinstance(varatts,(tuple,list)): raise TypeError
#   if not all([isinstance(atts,dict) for atts in varatts]): raise TypeError
#   # check axis: they need to have only one axes, which has to be the same for all!
#   if len(varatts) != len(varlist): raise ListError, "Failed to match varatts to varlist!"  
#   for var in varlist: 
#     if var.ndim > 1: raise AxisError, "Variable '{}' has more than one dimension; consider squeezing.".format(var.name)
#     elif var.ndim == 0: raise AxisError, "Variable '{}' is a scalar; consider display as a line.".format(var.name)
#   # loop over variables
#   plts = []; varname = None; varunits = None; axname = None; axunits = None # list of plot handles
#   for var,linestyle,varatt in zip(varlist,linestyles,varatts):
#     varax = var.axes[0]
#     # scale axis and variable values 
#     axe, axunits, axname = getPlotValues(varax, checkunits=axunits, checkname=None)
#     val, varunits, varname = getPlotValues(var, checkunits=varunits, checkname=None)
#     # variable and axis scaling is not always independent...
#     if var.plot is not None and varax.plot is not None: 
#       if 'preserve' in var.plot and 'scalefactor' in varax.plot:
#         if varax.units != axunits and var.plot.preserve == 'area':
#           val /= varax.plot.scalefactor  
#     # figure out keyword options
#     kwatts = kwargs.copy(); kwatts.update(varatt) # join individual and common attributes     
#     if 'label' not in kwatts: kwatts['label'] = var.name # default label: variable name
#     # N.B.: other scaling behavior could be added here
#     if lprint: print varname, varunits, val.mean()    
#     if lsmooth: val = smooth(val)
#     # figure out orientation
#     if flipxy: xx,yy = val, axe 
#     else: xx,yy = axe, val
#     # call plot function
#     if linestyle is None: plts.append(ax.plot(xx, yy, **kwatts)[0])
#     else: plts.append(ax.plot(xx, yy, linestyle, **kwatts)[0])
#   # set axes limits
#   if isinstance(xlim,(list,tuple)) and len(xlim)==2: ax.set_xlim(*xlim)
#   elif xlim is not None: raise TypeError
#   if isinstance(ylim,(list,tuple)) and len(ylim)==2: ax.set_ylim(*ylim)
#   elif ylim is not None: raise TypeError 
#   # set title
#   if title is not None:
#     ax.set_title(title, dict(fontsize='x-large'))
#     pos = ax.get_position()
#     pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-0.03)    
#     ax.set_position(pos)
#   # set axes labels  
#   if flipxy: xname,xunits,yname,yunits = varname,varunits,axname,axunits
#   else: xname,xunits,yname,yunits = axname,axunits,varname,varunits
#   if not xlabel: xlabel = '{0:s} [{1:s}]'.format('Seasonal Cycle',xunits) if xunits else '{0:s}'.format(xname)
#   else: xlabel = xlabel.format(xname,xunits)
#   if not ylabel: ylabel = '{0:s} [{1:s}]'.format(yname,yunits) if yunits else '{0:s}'.format(yname)
#   else: ylabel = ylabel.format(yname,yunits)
#   # a typical custom label that makes use of the units would look like this: 'custom label [{1:s}]', 
#   # where {} will be replaced by the appropriate default units (which have to be the same anyway)
#   xpad =  2; xticks = ax.get_xaxis().get_ticklabels()
#   ypad = -2; yticks = ax.get_yaxis().get_ticklabels()
#   # len(xticks) > 0 is necessary to avoid errors with AxesGrid, which removes invisible tick labels 
#   if len(xticks) > 0 and xticks[-1].get_visible(): ax.set_xlabel(xlabel, labelpad=xpad)
#   elif len(yticks) > 0 and not title: yticks[0].set_visible(False) # avoid overlap
#   if len(yticks) > 0 and yticks[-1].get_visible(): ax.set_ylabel(ylabel, labelpad=ypad)
#   elif len(xticks) > 0: xticks[0].set_visible(False) # avoid overlap
#   # make monthly ticks
#   if axname == 'time' and axunits == 'month':
#     ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2)) # ax.minorticks_on()
#   # add legend
#   if legend:
#     legatts = dict()
#     if ax.get_yaxis().get_label():
#       legatts['fontsize'] = ax.get_yaxis().get_label().get_fontsize()
#     if isinstance(legend,dict): legatts.update(legend) 
#     elif isinstance(legend,(int,np.integer,float,np.inexact)): legatts['loc'] = legend
#     ax.legend(**legatts)
#   # add orientation lines
#   if isinstance(xline,(int,np.integer,float,np.inexact)): ax.axhline(y=xline, color='black')
#   elif isinstance(xline,dict): ax.axhline(**xline)
#   if isinstance(yline,(int,np.integer,float,np.inexact)): ax.axvline(x=yline, color='black')
#   elif isinstance(xline,dict): ax.axvline(**yline)
#   # return handle
#   return plts      


if __name__ == '__main__':
    
  ## runoff differences plot
  
  #settings
  basins = ['FRB','ARB']
#   basins.reverse()
  exp = 'max-ens'
  period = 15
  grid = 'arb2_d02'
  variables = ['precip','runoff','sfroff']
  # figure
  lprint = True
  lfield = True
  lgage = True 
  ldisc = False # scale sfroff to discharge
  lprecip = True # scale sfroff by precip bias
  # figure parameters for saving
#   sf, figformat, margins, subplot, figsize = getFigureSettings(2, cbar=False, sameSize=False)
  # make figure and axes
#   fig, axes = pyl.subplots(*subplot, sharex=True, sharey=False, facecolor='white', figsize=figsize)
#   margins = dict(bottom=0.11, left=0.11, right=.975, top=.95, hspace=0.05, wspace=0.05)
#   fig.subplots_adjust(**margins) # hspace, wspace
  nax = len(basins)
  paper_folder = '/home/data/Figures/Basins/'
#   fig = pyl.figure(1, figsize=(6.25,3.75))
#   axes = ImageGrid(fig, (0.07,0.11,0.91,0.82), nrows_ncols = (1, nax), axes_pad = 0.2, aspect=False, label_mode = "L")
  fig, axes = getFigAx((1,2), name=None, title='IC Ensemble Average (Hist., Mid-, End-Century)', 
                       title_font='x-large', figsize=(6.25,3.75),  
                       stylesheet='myggplot', lpublication=True, yright=False, xtop=False,
                       variable_plotargs=None, dataset_plotargs=None, plot_labels=None, 
                       sharex=True, AxesGrid=False, direction='row',
                       axes_pad = 0., aspect=False, margins=(0.075,0.11,0.95,0.81),)
  # loop over panels/basins
  for n,ax,basin in zip(range(nax),axes,basins):
#   for basin in basins:
        
    # load meteo data
    if lfield: 
      print(' - loading Data')
      unity = loadUnity_ShpTS(varlist=['precip'], shape='shpavg')
      unity = unity(shape_name=basin).load().climMean()
#       unity['precip'][:] *= 86400. # scale with basin area
      wrf = loadWRF_ShpEns(name=exp, domains=2, shape='shpavg', filetypes=['srfc','lsm'],
                           varlist=variables[:]) # WRF
      wrf = wrf(shape_name=basin).load().climMean()
#       for varname in variables: wrf[varname][:] *= 86400. # scale with basin area
    # load basin data
    basin = basin_dict[basin] # Basin(basin=basin, basins_dict=)
    if lgage: gage = basin.getMainGage()
    # load/compute variables
    varlist = []
    if lgage: 
#       discharge = gage.discharge
#       print discharge.mean()      
#       discharge[:] /= ( unity.atts.shp_area * 86400)
#       discharge.plot = wrf.runoff.plot
#       discharge.units = wrf.runoff.units
#       print discharge.mean()
      discharge = gage.runoff
      discharge.name = 'Observed River Runoff'
      varlist += [discharge]
    if lfield:
      runoff = wrf.runoff; runoff.name = 'Total Runoff'
      sfroff = wrf.sfroff; sfroff.name = 'Surface Runoff'
      varlist += [runoff, sfroff]
      if ldisc:
        s_sfroff = sfroff.copy(deepcopy=True)
        s_sfroff.name = 'Scaled Sfroff'
        s_sfroff *= discharge.getArray().mean()/sfroff.getArray().mean()
        print(s_sfroff) 
        varlist += [s_sfroff]
      elif lprecip:
        assert unity.precip.units == wrf.precip.units
        scale = unity.precip.getArray().mean()/wrf.precip.getArray().mean()
        s_sfroff = sfroff.copy(deepcopy=True); s_sfroff *= scale; s_sfroff.name = 'Scaled Surface Runoff'
        s_runoff = runoff.copy(deepcopy=True); s_runoff *= scale; s_runoff.name = 'Scaled Total Runoff'
        varlist += [s_sfroff, s_runoff]
    if lfield and lgage:
      difference = sfroff - discharge
      difference.name = 'Difference'
      varlist += [difference]
      if ldisc or lprecip:
        s_difference = s_sfroff - discharge
        s_difference.name = 'Scaled Difference'      
        varlist += [s_difference]    
    for var in varlist: var.plot = discharge.plot # harmonize plotting
    #print sfroff.plot.name, sfroff.plot.units
    
    # plot properties    
    varatts = dict()
    varatts['Total Runoff'] = dict(color='purple', linestyle='--')
    varatts['Surface Runoff'] = dict(color='green', linestyle='--')
    varatts['Observed River Runoff'] = dict(color='green', linestyle='', marker='o', markersize=5)
    varatts['Difference'] = dict(color='red', linestyle='--')
    # add scaled variables
    satts = {}
    for key,val in varatts.items():
      val = val.copy(); val['linestyle'] = '-'
      satts['Scaled '+key] = val
    varatts.update(satts)
    # determine legend
    if n == 0: legend = None
    else: legend = dict(loc=1, labelspacing=0.125, handlelength=2.5, handletextpad=0.5, fancybox=True)
    # plot runoff
    print(' - creating plot')
    ax.title_size = 'large'
    ax.title_height = 0.04
    plts = ax.linePlot(varlist, plotatts=varatts, title=basin.long_name, hline=0, 
                       xlim=(0.5,12.5), lperi=True, lparasiteMeans=True,
                       ylim=(-2.5,5), xlabel='Seasonal Cycle [{1:s}]', legend=legend, lprint=True)
                
  if lprint:
    print(' - writing file')
    if ldisc: filename = 'runoff_discharge.pdf'
    elif lprecip: filename = 'runoff_precip.pdf'
    else: filename = 'runoff_test.pdf'
    print(('\nSaving figure in '+filename))
    fig.savefig(paper_folder+filename, dpi=300) # save figure to pdf
    print(figure_folder)
      
  ## show plots after all iterations  
  pyl.show()
    