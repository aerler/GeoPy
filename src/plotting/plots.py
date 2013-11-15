'''
Created on 2011-03-15

some useful plotting functions

@author: Andre R. Erler
'''

## add subplot/axes label
def addLabel(ax, label=None, loc=1, stroke=False, size=None, prop=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText 
    from matplotlib.patheffects import withStroke
    from string import lowercase    
    # expand list
    if not isinstance(ax,(list,tuple)): ax = [ax] 
    l = len(ax)
    if not isinstance(label,(list,tuple)): label = [label]*l
    if not isinstance(loc,(list,tuple)): loc = [loc]*l
    if not isinstance(stroke,(list,tuple)): stroke = [stroke]*l
    # settings
    if prop is None:
      prop = dict()
    if not size: prop['size'] = 18
    args = dict(pad=0., borderpad=1.5, frameon=False)
    args.update(kwargs)
    # cycle over axes
    at = [] # list of texts
    for i in xrange(l):
      if label[i] is None:
        label[i] = '('+lowercase[i]+')'
      elif isinstance(label[i],int):
        label[i] = '('+lowercase[label[i]]+')'
      # create label    
      at.append(AnchoredText(label[i], loc=loc[i], prop=prop, **args))
      ax[i].add_artist(at[i]) # add to axes
      if stroke[i]: 
        at[i].txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at
  
# plots with error shading 
def addErrorPatch(ax, var, err, color, axis='p', lpygeode=True, xerr=True, alpha=0.25, check=False, cap=-1):
    from numpy import append, where, isnan
    from matplotlib.patches import Polygon 
    if lpygeode:
      y = var.axes[var.whichaxis(axis)].values
      x = var.get(); e = err.get()
    else:
      y = axis; x = var; e = err
    if check:
      e = where(isnan(e),0,e)
      if cap > 0: e = where(e>cap,0,e)
    if xerr: 
      ix = append(x-e,(x+e)[::-1])
      iy = append(y,y[::-1])
    else:
      ix = append(y,y[::-1])
      iy = append(x-e,(x+e)[::-1])
    patch = Polygon(zip(ix,iy), alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_patch(patch)
    return patch 

## plot profile (line or scatter plot) of variable with multiplot support
def linePlot(varlist, coord, axis=None, clevs=None,title='',subplot=(),expand=True,figargs={'figsize':(10,8)},transpose=False, mplrc=None, **plotargs):
  from pygeode.axis import Axis, ZAxis
  from utils import multiPlot, loadMPL
  # load matplotlib and apply config dictionary
  mpl = loadMPL(mplrc)
  import matplotlib.pyplot as pyl
    
  # precondition input
  if isinstance(expand,bool): expand = (expand,)*2 # handle both levels equally
  if not isinstance(varlist,list): varlist = [[varlist]] # list of variables to be plotted
  else: varlist = [var if isinstance(var,list) else [var] for var in varlist]
  if not isinstance(coord,list): coord = [[coord]] # slices that will be applied to the variables
  else: coord = [co if isinstance(co,list) else [co] for co in coord]
  lv = len(varlist); lc = len(coord)  
  # expand auxiliary arguments to variable length
  if not isinstance(clevs,list): clevs = [clevs]*lv
  assert len(clevs)==lv or len(clevs)==lv*lc, 'length of clevs and varlist/coord are incompatible'
  # model axis list after varlist
  if not isinstance(axis,list): axis = [axis]*lv # alternate axes (variables with the same axes as the variables)
  assert len(axis)==lv, 'length of varlist and axis lists do not match'
  for i in xrange(lv):
    if not isinstance(axis[i],list): axis[i] = [axis[i]]*len(varlist[i])
    assert len(axis[i])==len(varlist[i]), 'length of varlist and axis lists in axis #%g do not match'%i
  
  ## handle multiplots (in the end varlist and coord should have the same shape)
  # variables arranged horizontally, coordinates/slices arranged vertically
  # expand outer list (multiplot layout)
  if expand[0]:
    if not subplot: subplot = (lc,lv) # use this as multiplot layout
    # expand lists
    if transpose:
      coord = [co for v in xrange(lv) for co in coord] # varying slow (outer loop; horizontal)
      varlist = [var for var in varlist for c in xrange(lc)] # varying fast (inner loop; vertical)
      axis = [ax for ax in axis for c in xrange(lc)] # analogous to varlist
      if len(clevs) == lv: # expand clevs like varlist if length matches
        clevs = [clv for clv in clevs for c in xrange(lc)]
    else:
      coord = [co for co in coord for v in xrange(lv)] # varying fast (inner loop; vertical)
      varlist = [var for c in xrange(lc) for var in varlist] # varying slow (outer loop; horizontal)
      axis = [ax for c in xrange(lc) for ax in axis] # analogous to varlist
      if len(clevs) == lv: # expand clevs like varlist if length matches
        clevs = [clv for c in xrange(lc) for clv in clevs]
  else:
    # do some simple adjustments and checks and use linear layout
    if lv==1: varlist = varlist*lc; axis = axis*lc
    elif lc==1: coord = coord*lv
    else: assert len(coord)==len(varlist), 'length of variable and coordinate list do not match; consider using the expand option'    
  le = len(varlist)
  assert le==len(coord), 'length mismatch between variable and coordinate list'
  if subplot: assert le==subplot[0]*subplot[1], 'subplot layout incompatible with number of variables and slices'
  # expand inner list (plots within one axis)  
  if expand[1]:
    for n in xrange(le):
      # expand lists
      lc = len(coord[n]); lv = len(varlist[n])
      coord[n] = [co for v in xrange(lv) for co in coord[n]] # varying fast
      varlist[n] = [var for var in varlist[n] for c in xrange(lc)] # varying slow      
      axis[n] = [ax for ax in axis[n] for c in xrange(lc)] # analogous to varlist 
  else:
    # do some simple adjustments and checks and use linear layout
    for n in xrange(le):
      if len(varlist[n])==1: varlist[n] = varlist[n]*len(coord[n]); axis[n] = axis[n]*len(coord[n])
      elif len(coord[n])==1: coord[n] = coord[n]*len(varlist[n])
      else: assert len(coord[n])==len(varlist[n]), 'length of variable and coordinate list do not match; consider using the expand option'
      
  ## make (sub-)plots  
  # prepare list of variables for plotting
  plotlist = []
  # loop over axes
  for n in xrange(le):
    lm = len(varlist[n])
    assert len(coord[n])==lm, 'length mismatch between variable and coordinate list in axis number %.0f'%n
    linelist = [] # list of variables in one axis
    # loop over line plots
    for m in xrange(lm):
      if varlist[n][m] is not None:
        # select slice
        slvar = varlist[n][m](**coord[n][m])
        # copy plot properties of replaced axes
        for i in xrange(slvar.naxes):
          slvar.axes[i].atts.update(varlist[n][m].axes[i].atts)
          slvar.axes[i].plotatts.update(varlist[n][m].axes[i].plotatts)
        # replace remaining axis with variable in axis-list, if given, 
        #  and provided it does not already have this axis
        if axis[n][m] and not slvar.hasaxis(axis[n][m].name):
          slax = axis[n][m](**coord[n][m])
          # find relevant axis
          for ax in slvar.axes: 
            if ax.values.size>1: 
              replaceAxis = ax.name; break
          # set attributes while preserving orientation w.r.t. horizontal or vertical
          axargs = {'name':slax.name, 'atts':slax.atts, 'plotatts':slax.plotatts}
          if isinstance(ax,ZAxis): 
            axisdict = {replaceAxis:ZAxis(values=slax.squeeze().get(), **axargs)}
          else:
            axisdict = {replaceAxis:Axis(values=slax.squeeze().get(), **axargs)}
          # replace axis
          slvar = slvar.replace_axes(axisdict=axisdict)        
        linelist.append(slvar)
      else:
        linelist = None
    # store variable list in plotlist
    plotlist.append(linelist)               
  # construct figure    
  pyl.ioff() # non-interactive mode
  f = pyl.figure(**figargs)        
  f.clf()
  # set default margins  
  defaultMargins = {'left':0.075,'right':0.965,'bottom':0.075,'top':0.975,'wspace':0.025,'hspace':0.025}
  defaultMargins.update(plotargs.pop('margins',{}))
  # call multiplot
  (f,cf,subplot) = multiPlot(f=f,varlist=plotlist,clevs=clevs,subplot=subplot,margins=defaultMargins, transpose=transpose,**plotargs)  
  if title: f.suptitle(title,fontsize=14) 
  
  # finalize
  pyl.draw(); # pyl.ion(); 
  return f  

## plot 2D map of variable (with multiplot support)
def surfacePlot(varlist, times=None, clevs=None,cbls=None,title='',axTitles=True,subplot=(),slices={},figargs={'figsize':(10,8)},**plotargs):
  import matplotlib.pylab as pyl
  from utils import multiPlot, sharedColorbar
  if not isinstance(varlist,list): varlist = [varlist]
  if not isinstance(times,list): times = [times]
  if not isinstance(clevs,list): clevs = [clevs]
  if not isinstance(cbls,list): cbls = [cbls]  
  pyl.ioff() # non-interactive mode
  # construct figure and axes    
  f = pyl.figure(**figargs)    
  f.clf()
  ## handle multiplot meta data
  if len(varlist)==1: 
    # plot single variable, possibly multiple times
    titles = []; plotlist = [];
    for time in times: 
      if time and axTitles: titles.append('time '+str(time))
      else: titles.append('')
      if varlist[0].hasaxis('time'): slices['time'] = time
      plotlist.append(varlist[0](**slices))
  else: 
    # plot multiple variables side-by-side, each time-step 
    if not subplot: subplot = (len(times),len(varlist))
    titles = []; plotlist = []; 
    for time in times:
      for var in varlist:
        if axTitles: titles.append(var.name+', time '+str(time))
        else: titles.append('')
        if var.hasaxis('time'): slices['time'] = time
        plotlist.append(var(**slices)) # sliced for time and latitude
  # expansion of other lists
  clevlist = clevs*len(times) # create contour level list for each plot
  cbllist = cbls*len(times) # create contour level list for each plot
  # organize colorbar (cleanup arguments) 
  colorbar = plotargs.pop('colorbar',{})
  manualCbar = colorbar.pop('manual',False)
  if manualCbar: cbar = False
  else: cbar = colorbar 
  # set default margins  
  defaultMargins = {'left':0.065,'right':0.975,'bottom':0.05,'top':0.95,'wspace':0.025,'hspace':0.1}
  defaultMargins.update(plotargs.pop('margins',{}))
  ## draw (sub-)plots  
  (f,cf,subplot) = multiPlot(f=f,varlist=plotlist,titles=titles,clevs=clevlist,cbls=cbllist, #
                     subplot=subplot,colorbar=cbar,margins=defaultMargins,**plotargs)  
  if title: f.suptitle(title,y=defaultMargins['top'],fontsize=plotargs.get('fontsize',12)+2) 
  ## add common colorbar
  if manualCbar:
    if len(varlist)==1: subplot = (1,1) # need only one colorbar, reference to first axes
    sharedColorbar(f, cf, clevs, colorbar, cbls, subplot, defaultMargins)   
  # finalize
  pyl.draw(); # pyl.ion(); 
  return f

## generates a hovmoeller plot of the given variables (with multiplot support)
def hovmoellerPlot(varlist, clevs=None,cbls=None,title='',subplot=(),slices={},figargs={'figsize':(8,8)},**plotargs):
  import matplotlib.pylab as pyl    
  from pygeode.axis import XAxis, YAxis, TAxis
  from utils import multiPlot, sharedColorbar
  if not isinstance(varlist,list): varlist = [varlist]
  if not isinstance(clevs,list): clevs = [clevs]
  if not isinstance(cbls,list): cbls = [cbls]
  pyl.ioff() # non-interactive mode
  # construct figure and axes    
  f = pyl.figure(**figargs)    
  f.clf()
  # create zonal-mean variables
  titles = [var.name for var in varlist]
  plotlist = [var(**slices).mean(XAxis).transpose(YAxis,TAxis) for var in varlist] # latitude sliced
  # organize colorbar (cleanup arguments) 
  colorbar = plotargs.pop('colorbar',{})
  manualCbar = colorbar.pop('manual',False)
  if manualCbar: cbar = False
  else: cbar = colorbar 
  # set default margins  
  defaultMargins = {'left':0.065,'right':0.975,'bottom':0.05,'top':0.95,'wspace':0.05,'hspace':0.1}
  defaultMargins.update(plotargs.pop('margins',{}))
  ## make subplots
  (f,cf,subplot) = multiPlot(f=f,varlist=plotlist,titles=titles,clevs=clevs,cbls=cbls,subplot=subplot, #
                     colorbar=cbar,margins=defaultMargins,**plotargs)
  if title: f.suptitle(title,fontsize=14) 
  ## add common colorbar
  if manualCbar:
    f = sharedColorbar(f, cf, clevs, colorbar, cbls, subplot, defaultMargins)        
  # finalize
  pyl.draw(); # pyl.ion();
  return f