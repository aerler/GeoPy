'''
Created on 2011-02-28

utility functions, mostly for plotting, that are not called directly

@author: Andre R. Erler
'''

## Log-axis ticks
# N
nTicks1st = ['2','','4','','6','','','']
nTicks2nd = ['20','','40','','60','','','']
nTicks3rd = ['200','','400','','600','','','']
# p
pTicks1st = ['2','3','','5','','7','','']
pTicks2nd = ['20','30','','50','','70','','']
pTicks3rd = ['200','300','','500','','700','','']


def loadMPL(mplrc=None):
  # load matplotlib (default)
  import matplotlib as mpl
  # apply rc-parameters from dictionary
  if (mplrc is not None) and isinstance(mplrc,dict):
    # loop over parameter groups
    for (key,value) in mplrc.iteritems():
      mpl.rc(key,**value)  # apply parameters
  # return matplotlib instance with new parameters
  return mpl
  
# function to smooth a vector (numpy array): moving mean, nothing fancy
def smoothVector(x,i):
  xs = x.copy() # smoothed output vector
  i = 2*i
  d = i+1 # denominator  for later
  while i>0:    
    t = x.copy(); t[i:] = t[:-i];  xs += t
    t = x.copy(); t[:-i] = t[i:];  xs += t
    i-=2
  return xs/d

# function to traverse nested lists recursively and perform the operation fct on the end members
def traverseList(lsl, fct):
  # traverse nested lists recursively
  if isinstance(lsl, list):
    return [traverseList(lsl[i], fct) for i in range(len(lsl))]
  # break recursion and apply rescaling when dictionary is reached 
  else: return fct(lsl)
  
# function to expand level lists and colorbar ticks
def expandLevelList(arg, vec=None):
  from numpy import asarray, ndarray, linspace, min, max
  ## figure out level list and return numpy array of levels
  # trivial case: already numpy array
  if isinstance(arg,ndarray):
    return arg 
  # list: recast as array
  elif isinstance(arg,list):
    return asarray(arg)
  # tuple with three or two elements: use as argument to linspace 
  elif isinstance(arg,tuple) and (len(arg)==3 or len(arg)==2):
    return linspace(*arg)
  # use additional info in vec to determine limits
  else:
    # figure out vec limits
    # use first two elements, third is number of levels
    if isinstance(vec,(tuple,list)) and len(vec)==3:  
      minVec = min(vec[:2]); maxVec = max(vec[:2])
    # just treat as level list
    else: 
      minVec = min(vec); maxVec = max(vec)
    # interpret arg as number of levels in given interval
    # only one element: just number of levels
    if isinstance(arg,(tuple,list,ndarray)) and len(arg)==1: 
      return linspace(minVec,maxVec,arg[0])
    # numerical value: use as number of levels
    elif isinstance(arg,(int,float)):
      return linspace(minVec,maxVec,arg)        
  
# function to place (shared) colorbars at a specified figure margins
def sharedColorbar(f, cf, clevs, colorbar, cbls, subplot, margins):
  loc = colorbar.pop('location','bottom')      
  # determine size and spacing
  if loc=='top' or loc=='bottom':
    dir = colorbar.pop('orientation','horizontal') # colorbar orientation
    je = subplot[1] # number of colorbars: number of rows
    ie = subplot[0] # number of plots per colorbar: number of columns
    cbwd = colorbar.pop('cbwd',0.025) # colorbar height
    sp = margins['wspace']
    wd = (margins['right']-margins['left'] - sp*(je-1))/je # width of each colorbar axis 
  else:
    dir = colorbar.pop('orientation','vertical') # colorbar orientation
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
  f.subplots_adjust(**{loc:newMargin})
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
    f.colorbar(mappable=cf[ci],cax=f.add_axes(cax),ticks=expandLevelList(cbls[i],clevs[i]),orientation=dir,**colorbar)
  # return figure with colorbar (just for the sake of returning something) 
  return f

# function that creates the most "square" arrangement of subplots for a given number of plots
def multiPlot(f,varlist,titles='',clevs=None,labels=None,legends=None,cbls=None,geos={},sharex=True,sharey=True,margins={},subplot=(),transpose=False,axargs=None,**kwargs):
  from pygeode.plot import plotvar
  from pygeode.axis import ZAxis
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
    s = [(i,le/i) for i in xrange(1,le+1) if le%i==0]
    # select "most square" 
    ss = [t[0]+t[1] for t in s]
    (ie,je) = s[ss.index(min(ss))]
    if transpose: (je,ie) = (ie,je) # swap vertical and horizontal dimension length 
    subplot = (ie,je) # save (return to caller)  
    
  # colorbar default as in plotvar
  colorbar = kwargs.pop('colorbar',{'orientation':'vertical'})      
  ## create axes and draw plots
  axs = []; cf = []; n=0 # reset counter
  for i in xrange(ie):
    for j in xrange(je):          
      
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
            cf.append([plotvar(var[m],ax=ax,label=label[m],**plotargs) for m in xrange(ll)])
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
