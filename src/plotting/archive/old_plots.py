'''
Created on 2011-03-15

some useful plotting functions

@author: Andre R. Erler
'''

# function that creates the most "square" arrangement of subplots for a given number of plots
def multiPlot(f,varlist,titles='',clevs=None,labels=None,legends=None,cbls=None,geos={},sharex=True,sharey=True,margins={},subplot=(),transpose=False,axargs=None,**kwargs):
  from matplotlib.pylab import setp
  
  le = len(varlist)      
  # expand some list, if necessary (one element per plot) 
  if not isinstance(titles,list): titles = [titles]*le
  else: assert len(titles)==le, 'number of titles not the same as number of plots'
  if not isinstance(labels,list): labels = [labels]*le
  else: assert len(labels)==le, 'number of labels not the same as number of plots'
  if not isinstance(legends,list): legends = [legends]*le
  else: assert len(legends)==le, 'number of legend dictionaries not the same as number of plots'
  if not isinstance(clevs,list): clevs = [clevs]*le # contour list; numpy array or tuple (3 elements)
  else: assert len(clevs)==le, 'number of contour lists not the same as number of plots'
  if not isinstance(cbls,list): cbls = [cbls]*le # tick list; numpy array, number, or tuple (3 elements)
  else: assert len(cbls)==le, 'number of colorbar tick lists not the same as number of plots'
  if not isinstance(geos,list): geos = [geos]*le # geographic projection parameters (False=no projection)
  else: assert len(geos)==le, 'number of geographic projections not the same as number of plots'
  ## determine subplot division
  if subplot: 
    if transpose:
      (je,ie) = subplot
      subplot = (ie,je) # save (return to caller)
    else: 
      (ie,je) = subplot 
  else: # assume scalar    
    # all possible subdivisions 
    s = [(i,le/i) for i in range(1,le+1) if le%i==0]
    # select "most square" 
    ss = [t[0]+t[1] for t in s]
    (ie,je) = s[ss.index(min(ss))]
    if transpose: (je,ie) = (ie,je) # swap vertical and horizontal dimension length 
    subplot = (ie,je) # save (return to caller)  
    
  # colorbar default as in plotvar
  colorbar = kwargs.pop('colorbar',{'orientation':'vertical'})      
  ## create axes and draw plots
  axs = []; cf = []; n=0 # reset counter
  for i in range(ie):
    for j in range(je):          
      
      var = varlist[n]
      # if var is None, leave and empty space
      if var is not None: 
        # plot shorthands
        geo = geos[n].copy() 
        # expand clevs tuple (or whatever) into nparray
        clev = expandLevelList(clevs[n])       
        ## handle annotation and shared axes pre-processing
        plotargs = {'lblx':kwargs.get('lblx',True), 'lbly':kwargs.get('lbly',True)} 
        if titles: plotargs['title'] = titles[n] # plot titles
        share={'sharex':None, 'sharey':None}
        if geo: labels=geo.get('labels',[1,0,0,1]) # labels at left and bottom side
        # x-axis annotation
        if i<(ie-1): # no bottom labels
          plotargs['lblx'] = False # ... and bottom border
          if geo and sharex: labels[3] = 0 
        if sharex and not geo and i>0: share['sharex'] = axs[j] # same column, first row
        # y-axis annotation     
        if j>0: 
          plotargs['lbly'] = False # only annotation on left border...
          if sharey: # no left labels
            if geo: labels[0] = 0  
            else: share['sharey'] = axs[i*je] # same row, first column  
#        else:
#          if ie==1: # use title instead of name if space is available
#            if isinstance(var,list):
#              for m in xrange(len(var)): 
#                var[m].plotatts['plotname'] = var[m].plotatts['plottitle']
#            else:
#              var.plotatts['plotname'] = var.plotatts['plottitle']      
        if geo:
          geo.update({'labels':labels}) 
          plotargs['projection'] = geo # save projection parameters
        # create axis with appropriate axes share references and label visibility
        axs.append(f.add_subplot(ie,je,n+1,**share)); ax = axs[n]       
        plotargs.update(kwargs) # update plot args and overwrite defaults      
                
        ## either line plot (line plots have only one dimensions)
        if isinstance(var,list) or len(var.squeeze().axes)==1:
          label = labels[n]; legend = legends[n]         
          # either multiple line plots in one axis              
          if isinstance(var,list):
            ll = len(var)          
            if isinstance(label,list): 
              assert len(label)==ll, 'number of labels in axis %g not the same as number of plots'%n
              for lbl in label: assert isinstance(lbl,str), 'labels have to be strings' 
            elif label is None: label = [v.name for v in var] # default labels: name of variable
            else:
              assert isinstance(label,str), 'labels have to be strings'
              label = [label]*ll
            # Note: here var is a *list* of all line plots that go into this axis!
            cf.append([plotvar(var[m],ax=ax,label=label[m],**plotargs) for m in range(ll)])
            zz = isinstance(var[m].squeeze().axes[0],ZAxis)
          # or a single line plot per axis
          else: # len(var.squeeze().axes)==1:  
            cf.append(plotvar(var,ax=ax,label=label,**plotargs))
            zz = isinstance(var.squeeze().axes[0],ZAxis)
          # use clevs to adjust value axis scaling 
          if (clev is not None) and len(clev)>1:         
            if zz: ax.set_xlim([clev[0],clev[-1]])
            else: ax.set_ylim([clev[0],clev[-1]])
          # add legend
          if legend:    
            if isinstance(legend,dict): lp = legend
            elif isinstance(legend,int): lp = {'loc':legend}
            else: lp
            ax.legend(**lp)
        
        ## or surface/contour plot
        else: # surface plots have two dimensions
          assert len(var.squeeze().axes)==2, 'surface plots can only have two dimensions'
          cbl = cbls[n];
          # figure out colorbar ticks/levels
          if colorbar: colorbar['ticks'] = expandLevelList(cbl,clev)
          # draw surface or contour plot with colorbar
          cf.append(plotvar(var,ax=ax,clevs=clev,colorbar=colorbar,**plotargs)) 
        
        ## apply axes properties and shared axes post-processing
#        if axArgs: pset  
        if sharex and plotargs['lblx']==False: 
          setp(ax.get_xticklabels(minor=False),visible=False)
          setp(ax.get_xticklabels(minor=True),visible=False) # also apply to minor ticks
        if sharey and plotargs['lbly']==False:
          setp(ax.get_yticklabels(minor=False),visible=False) 
          setp(ax.get_yticklabels(minor=True),visible=False)
        n+=1 # counter
          
  # set margins
  if margins: f.subplots_adjust(**margins)
  # return axes
  return (f,cf,subplot) 

## plot profile (line or scatter plot) of variable with multiplot support
def linePlot(varlist, coord, axis=None, clevs=None,title='',subplot=(),expand=True,figargs={'figsize':(10,8)},transpose=False, mplrc=None, **plotargs):
  from plotting.misc import multiPlot, loadMPL
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
  for i in range(lv):
    if not isinstance(axis[i],list): axis[i] = [axis[i]]*len(varlist[i])
    assert len(axis[i])==len(varlist[i]), 'length of varlist and axis lists in axis #%g do not match'%i
  
  ## handle multiplots (in the end varlist and coord should have the same shape)
  # variables arranged horizontally, coordinates/slices arranged vertically
  # expand outer list (multiplot layout)
  if expand[0]:
    if not subplot: subplot = (lc,lv) # use this as multiplot layout
    # expand lists
    if transpose:
      coord = [co for v in range(lv) for co in coord] # varying slow (outer loop; horizontal)
      varlist = [var for var in varlist for c in range(lc)] # varying fast (inner loop; vertical)
      axis = [ax for ax in axis for c in range(lc)] # analogous to varlist
      if len(clevs) == lv: # expand clevs like varlist if length matches
        clevs = [clv for clv in clevs for c in range(lc)]
    else:
      coord = [co for co in coord for v in range(lv)] # varying fast (inner loop; vertical)
      varlist = [var for c in range(lc) for var in varlist] # varying slow (outer loop; horizontal)
      axis = [ax for c in range(lc) for ax in axis] # analogous to varlist
      if len(clevs) == lv: # expand clevs like varlist if length matches
        clevs = [clv for c in range(lc) for clv in clevs]
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
    for n in range(le):
      # expand lists
      lc = len(coord[n]); lv = len(varlist[n])
      coord[n] = [co for v in range(lv) for co in coord[n]] # varying fast
      varlist[n] = [var for var in varlist[n] for c in range(lc)] # varying slow      
      axis[n] = [ax for ax in axis[n] for c in range(lc)] # analogous to varlist 
  else:
    # do some simple adjustments and checks and use linear layout
    for n in range(le):
      if len(varlist[n])==1: varlist[n] = varlist[n]*len(coord[n]); axis[n] = axis[n]*len(coord[n])
      elif len(coord[n])==1: coord[n] = coord[n]*len(varlist[n])
      else: assert len(coord[n])==len(varlist[n]), 'length of variable and coordinate list do not match; consider using the expand option'
      
  ## make (sub-)plots  
  # prepare list of variables for plotting
  plotlist = []
  # loop over axes
  for n in range(le):
    lm = len(varlist[n])
    assert len(coord[n])==lm, 'length mismatch between variable and coordinate list in axis number %.0f'%n
    linelist = [] # list of variables in one axis
    # loop over line plots
    for m in range(lm):
      if varlist[n][m] is not None:
        # select slice
        slvar = varlist[n][m](**coord[n][m])
        # copy plot properties of replaced axes
        for i in range(slvar.naxes):
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
  from plotting.misc import multiPlot, sharedColorbar
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
  from plotting.misc import multiPlot, sharedColorbar
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