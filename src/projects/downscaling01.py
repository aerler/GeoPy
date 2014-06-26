'''
Created on 2014-02-14

Script to generate plots for my first downscaling paper!

@author: Andre R. Erler, GPL v3
'''

# use common MPL instance
from plotting.utils import loadMPL
mpl,pyl = loadMPL(linewidth=.75)
from mpl_toolkits.axes_grid1 import ImageGrid
# PyGeoDat stuff
from plotting.lineplots import linePlot
from datasets.WRF import loadWRF
from datasets.Unity import loadUnity
from datasets.WSC import Basin
# ARB project related stuff
from projects.ARB_settings import figure_folder



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
  ldisc = False # scale sfroff to discharge
  lprecip = True # scale sfroff by precip bias
  # figure parameters for saving
#   sf, figformat, margins, subplot, figsize = getFigureSettings(2, cbar=False, sameSize=False)
  # make figure and axes
#   fig, axes = pyl.subplots(*subplot, sharex=True, sharey=False, facecolor='white', figsize=figsize)
#   margins = dict(bottom=0.11, left=0.11, right=.975, top=.95, hspace=0.05, wspace=0.05)
#   fig.subplots_adjust(**margins) # hspace, wspace
  nax = len(basins)
  paper_folder = '/home/me/Research/Dynamical Downscaling/Report/JClim Paper 2014/figures/'
  fig = pyl.figure(1, facecolor='white', figsize=(6.25,3.75))
  axes = ImageGrid(fig, (0.09,0.11,0.88,0.82), nrows_ncols = (1, nax), axes_pad = 0.2, aspect=False, label_mode = "L")
              
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
    else: legend = dict(loc=1, labelspacing=0.125, handlelength=2.5, handletextpad=0.5, fancybox=True)
    # plot runoff
    plts = linePlot(ax, varlist, varatts=varatts, title=basin.long_name, xline=0, xlim=(1,12), ylim=(-6,16), legend=legend) # , scalefactor=1e-6
                
  if lprint:
    if ldisc: filename = 'runoff_discharge.png'
    elif lprecip: filename = 'runoff_precip.png'
    else: filename = 'runoff_test.png'
    print('\nSaving figure in '+filename)
    fig.savefig(paper_folder+filename, dpi=150) # save figure to pdf
    print(figure_folder)
      
  ## show plots after all iterations  
  pyl.show()
    