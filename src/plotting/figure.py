'''
Created on Dec 11, 2014

A custom Figure class that provides some specialized functions and uses a custom Axes class.

@author: Andre R. Erler, GPL v3
'''

# external imports
from warnings import warn
from types import NoneType
from matplotlib.figure import Figure, SubplotBase, subplot_class_factory
from mpl_toolkits.axes_grid.axes_divider import LocatableAxes
import numpy as np
# internal imports
from geodata.misc import isInt , ArgumentError
from plotting.axes import MyAxes, MyLocatableAxes, Axes, MyPolarAxes, TaylorAxes
from plotting.misc import loadStyleSheet, toGGcolors
import matplotlib as mpl
# just for convenience
from matplotlib.pyplot import show, figure

## my new figure class
class MyFigure(Figure):
  ''' 
    A custom Figure class that provides some specialized functions and uses a custom Axes class.
    This is achieved by overloading add_axes and add_subplot.
    (This class does not support built-in projections; use the Basemap functionality instead.)  
  '''
  # some default parameters
  title_height    = 0.05
  title_size      = 'x-large'
  print_settings  = None
  shared_legend   = None
  legend_axes     = None
  shared_colorbar = None
  colorbar_axes   = None
  
  def __init__(self, *args, **kwargs):
    ''' constructor that accepts custom axes_class as keyword argument '''
    # parse arguments
    if 'axes_class' in kwargs:
      axes_class = kwargs.pop('axes_class')
      if not issubclass(axes_class, Axes): raise TypeError(axes_class)
    else: axes_class = MyAxes # default
    if 'axes_args' in kwargs:
      axes_args = kwargs.pop('axes_args')
      if axes_args is not None and not isinstance(axes_args, dict): raise TypeError
    else: axes_args = None # default
    if 'print_settings' in kwargs:
      print_settings = kwargs.pop('print_settings')
    else: print_settings = None
    # call parent constructor
    super(MyFigure,self).__init__(*args, **kwargs)
    # save axes class for later
    self.axes_class = axes_class   
    self.axes_args = axes_args 
    # print options
    self.print_settings = dict(dpi=300, transparent=False) # defaults
    if print_settings: self.print_settings.update(print_settings)
    
# N.B.: using the built-in mechanism to choose Axes seems to cause more problems
#     from matplotlib.projections import register_projection
#     # register custom class with mpl
#     register_projection(axes_class)
#   def add_axes(self, *args, **kwargs):
#     ''' overloading original add_subplot in order to use custom Axes (adapted from parent) '''
#     if 'projection' not in kwargs:
#       kwargs['projection'] = 'my'
#     super(MyFigure,self).__init__(*args, **kwargs)
  
  def add_axes(self, *args, **kwargs):
    ''' overloading original add_subplot in order to use custom Axes (adapted from parent) '''
    if not len(args):
        return
    # shortcut the projection "key" modifications later on, if an axes
    # with the exact args/kwargs exists, return it immediately.
    key = self._make_key(*args, **kwargs)
    ax = self._axstack.get(key)
    if ax is not None:
        self.sca(ax)
        return ax
    if isinstance(args[0], Axes): # allow all Axes, if passed explicitly
      a = args[0]
      assert(a.get_figure() is self)
    else:
      rect = args[0]
      # by registering the new Axes class as a projection, it may be possible 
      # to use the old axes creation mechanism, but it doesn't work this way...
      #       from matplotlib.figure import process_projection_requirements 
      #       if 'projection' not in kwargs: kwargs['projection'] = 'my'
      #       axes_class, kwargs, key = process_projection_requirements(
      #           self, *args, **kwargs)
      axes_class = kwargs.pop('axes_class',None)  
      if axes_class is None: axes_class = self.axes_class # defaults to my new custom axes (MyAxes)
      key = self._make_key(*args, **kwargs)
      # check that an axes of this type doesn't already exist, if it
      # does, set it as active and return it
      ax = self._axstack.get(key)
      if ax is not None and isinstance(ax, axes_class):
          self.sca(ax)
          return ax
      # create the new axes using the axes class given
      # add default axes arguments
      if self.axes_args is not None:
        axes_args = self.axes_args.copy()
        axes_args.update(kwargs)
      else: axes_args = kwargs
      a = axes_class(self, rect, **axes_args)
    self._axstack.add(key, a)
    self.sca(a)
    # attach link to figure (self)
    a.figure = self
    return a
  def add_subplot(self, *args, **kwargs):
    ''' overloading original add_subplot in order to use custom Axes (adapted from parent) '''
    if not len(args):
        return
    if len(args) == 1 and isinstance(args[0], int):
        args = tuple([int(c) for c in str(args[0])])
    if isinstance(args[0], SubplotBase):
      # I'm not sure what this does...
      a = args[0]
      assert(a.get_figure() is self)
      # make a key for the subplot (which includes the axes object id
      # in the hash)
      key = self._make_key(*args, **kwargs)
    else:
      #       if 'projection' not in kwargs: kwargs['projection'] = 'my'
      #       axes_class, kwargs, key = process_projection_requirements(
      #           self, *args, **kwargs)    
      axes_class = kwargs.pop('axes_class',None)  
      if axes_class is None: axes_class = self.axes_class # defaults to my new custom axes (MyAxes)
      key = self._make_key(*args, **kwargs)
      # try to find the axes with this key in the stack
      ax = self._axstack.get(key)
      if ax is not None:
        if isinstance(ax, axes_class):
          # the axes already existed, so set it as active & return
          self.sca(ax)
          return ax
        else:
          # Undocumented convenience behavior:
          # subplot(111); subplot(111, projection='polar')
          # will replace the first with the second.
          # Without this, add_subplot would be simpler and
          # more similar to add_axes.
          self._axstack.remove(ax)
      # add default axes arguments
      if self.axes_args is not None:
        axes_args = self.axes_args.copy()
        axes_args.update(kwargs)      
      else: axes_args = kwargs
      # generate subplot class and create axes instance
      a = subplot_class_factory(axes_class)(self, *args, **axes_args)
    self._axstack.add(key, a)
    self.sca(a)
    return a
  
  # function to adjust subplot parameters
  def updateSubplots(self, mode='shift', **kwargs):
    ''' simple helper function to move (relocate), shift, or scale subplot margins '''
    pos = self.subplotpars
    margins = dict() # original plot margins
    margins['left'] = pos.left; margins['right'] = pos.right 
    margins['top'] = pos.top; margins['bottom'] = pos.bottom
    margins['wspace'] = pos.wspace; margins['hspace'] = pos.hspace
    # update subplot margins
    if mode == 'move': margins.update(kwargs)
    else: 
      for key,val in kwargs.iteritems():
        if key in margins:
          if mode == 'shift': margins[key] += val
          elif mode == 'scale': margins[key] *= val
    # finally, actually update figure
    self.subplots_adjust(**margins)
    # and now repair damage: restore axes
    for ax in self.axes:
      if isinstance(ax,LocatableAxes):
        warn('Adjusting subplots does not work with LocatableAxes')
      ax.updateAxes(mode='adjust')
        
  # add common/shared legend to a multi-panel plot
  def addSharedLegend(self, plots=None, labels=None, fontsize=None, hscl=1., hpad=0.005, location='bottom', loc=None, ncols=None, **kwargs):
    ''' add a common/shared legend to a multi-panel plot '''
    # complete input
    if labels is None: labels = [plt.get_label() for plt in plots]
    elif not isinstance(labels, (list,tuple)): raise TypeError
    if plots is None: plots = self.axes[0].plots
    elif not isinstance(plots, (list,tuple,NoneType)): raise TypeError
    # figure out fontsize and row numbers  
    fontsize = fontsize or self.axes[0].get_yaxis().get_label().get_fontsize() # or fig._suptitle.get_fontsize()
    nlen = len(plots) if plots else len(labels)
    if ncols is None:
        if fontsize > 11: ncols = 2 if nlen == 4 else 3
        else: ncols = 3 if nlen == 6 else 4              
    # make room for legend
    if location.lower() == 'bottom':
        leghgt = ( np.ceil(float(nlen)/float(ncols)) * fontsize/300.) * hscl
        self.updateSubplots(mode='shift', bottom=leghgt+hpad) # shift bottom upwards (add height pad)
        ax = self.add_axes([0, hpad-0.005, 1,leghgt-hpad], axes_class=MyAxes) # new axes to hold legend, with some attributes
        if loc is None: loc = 9 
    elif location.lower() == 'right':
        leghgt = ( ncols * fontsize/40.) * hscl
        self.updateSubplots(mode='shift', right=-leghgt-hpad) # shift bottom upwards (add height pad)
        ax = self.add_axes([0.99-leghgt-hpad, 0, leghgt+hpad,1-self.title_height-hpad], axes_class=MyAxes) # new axes to hold legend, with some attributes
        if loc is None: loc = 2 # upper left
    ax.set_frame_on(False); ax.axes.get_yaxis().set_visible(False); ax.axes.get_xaxis().set_visible(False)
    # define legend parameters
    legargs = dict(loc=loc, ncol=ncols, borderaxespad=0., fontsize=fontsize, frameon=True,
                   labelspacing=0.1, handlelength=1.3, handletextpad=0.3, fancybox=True)
    legargs.update(kwargs)
    # create legend and return handle
    if plots: legend = ax.legend(plots, labels, **legargs)
    else: legend = ax.legend(labels, **legargs)
    # store axes handle and legend
    self.legend_axes = ax
    self.shared_legend = legend
    return legend
    
  # add subplot/axes labels
  def addLabels(self, labels=None, loc=1, lstroke=False, lalphabet=True, size=None, prop=None, **kwargs):
    # expand list
    axes = self.axes
    n = len(axes)
    if not isinstance(labels,(list,tuple)): labels = [labels]*n
    if not isinstance(loc,(list,tuple)): loc = [loc]*n
    if not isinstance(lstroke,(list,tuple)): lstroke = [lstroke]*n
    # settings
    if prop is None: prop = dict()
    if not size: prop['size'] = 'large'
    args = dict(pad=0., borderpad=1.5, frameon=False)
    args.update(kwargs)
    # cycle over axes
    ats = [] # list of texts
    for i,ax in enumerate(axes):
      # skip shared legend or colorbar
      if ax is not self.legend_axes and ax is not self.colorbar_axes:
        # default label
        label = labels[i]
        if label is None: 
          label = i
          if not lalphabet: label += 1
        # create label artist
        ats.append(ax.addLabel(label, loc=loc[i], lstroke=lstroke[i], lalphabet=lalphabet, 
                               prop=prop, **args))      
    return ats
  
  # save figure
  def save(self, *args, **kwargs):
    ''' save figure with some sensible default settings '''
    if len(args) == 0: raise ArgumentError
    # get option
    folder = kwargs.pop('folder', None)
    lfeedback = kwargs.pop('lfeedback', None) or kwargs.pop('feedback', None)
    lreplaceSpace = kwargs.pop('lreplaceSpace', True) and kwargs.pop('lreplaceSpace', True)
    filetype = kwargs.pop('filetype', 'pdf')
    if not filetype.startswith('.'): filetype = '.'+filetype
    # construct filename
    filename = ''
    for arg in args: 
      if arg is not None:
        if isinstance(arg, (list,tuple)):
          for a in arg: filename += str(a)
        else: filename += str(arg)
        filename += '_'
    filename = filename[:-1] # remove last underscore
    if not filename.endswith(filetype): filename += filetype
    # replace spaces, if desired
    if lreplaceSpace:
        filename = filename.replace(' ', '_')
    # update print settings
    sf = self.print_settings.copy() # print properties
    sf.update(kwargs) # update with kwargs
    # save file
    if lfeedback: print('Saving figure in '+filename)
    if folder is not None:
      filename = '{:s}/{:s}'.format(folder,filename)
      if lfeedback: print("('{:s}')".format(folder))
    self.savefig(filename, **sf) # save figure to pdf


## convenience function to return a figure and an array of ImageGrid axes
def getFigAx(subplot, name=None, title=None, title_font='x-large', title_height=None, figsize=None,
             variable_plotargs=None, dataset_plotargs=None, plot_labels=None, yright=False, xtop=False,
             sharex=None, sharey=None, lAxesGrid=False, ngrids=None, direction='row',
             lPolarAxes=False, lTaylor = False, 
             axes_pad = None, add_all=True, share_all=None, aspect=False, margins=None,
             label_mode='L', cbar_mode=None, cbar_location='right', lreduce=True,
             cbar_pad=None, cbar_size='5%', axes_class=None, axes_args=None,  stylesheet=None,
             lpresentation=False, lpublication=False, figure_class=None, **figure_args):
  # load stylesheet
  if stylesheet is not None:
    loadStyleSheet(stylesheet, lpresentation=lpresentation, lpublication=lpublication) 
    if stylesheet in ('myggplot','ggplot'):
      warn("Rewriting built-in color definitions to GG-plot defaults.")
      if dataset_plotargs is not None: dataset_plotargs = toGGcolors(dataset_plotargs) # modifies in-place!      
  # default figure class
  if figure_class is None: figure_class = MyFigure
  elif not issubclass(figure_class, Figure): raise TypeError 
  # figure out subplots
  if isinstance(subplot,(np.integer,int)):
    if subplot == 1:   subplot = (1,1)
    elif subplot == 2: subplot = (1,2)
    elif subplot == 3: subplot = (1,3)
    elif subplot == 4: subplot = (2,2)
    elif subplot == 6: subplot = (2,3)
    elif subplot == 9: subplot = (3,3)
    else: raise NotImplementedError
  elif not (isinstance(subplot,(tuple,list)) and len(subplot) == 2) and all(isInt(subplot)): raise TypeError    
  # create figure
  if figsize is None:
    if lpublication: 
      if subplot == (1,1): figsize = (3.75,3.75)
      elif subplot == (1,2) or subplot == (1,3): figsize = (6.25,3.75)
      elif subplot == (2,1) or subplot == (3,1): figsize = (3.75,7)
      else: figsize = (6.25,6.25)
    elif lpresentation: 
      if subplot == (1,2) or subplot == (1,3): figsize = (5,3)
      elif subplot == (2,1) or subplot == (3,1): figsize = (3,5)
      else: figsize = (5,5)
    else:
      if subplot == (1,1): figsize = (5,5)
      elif subplot == (1,2) or subplot == (1,3): figsize = (9,5)
      elif subplot == (2,1) or subplot == (3,1): figsize = (5,9)
      else: figsize = (9,9)
  # figure out margins
  if margins is None:
    # N.B.: the rectangle definition is presumably left, bottom, width, height
    if subplot == (1,1): margins = (0.1,0.1,0.85,0.85)
    elif subplot == (1,2) or subplot == (1,3): margins = (0.06,0.1,0.92,0.87)
    elif subplot == (2,1) or subplot == (3,1): margins = (0.09,0.11,0.88,0.82)
    elif subplot == (2,2) or subplot == (3,3): margins = (0.06,0.08,0.92,0.92)
    else: margins = (0.09,0.11,0.88,0.82)
    if title_height is None: title_height = getattr(figure_class, 'title_height', 0.05) # use default from figure
    if title is not None: margins = margins[:3]+(margins[3]-title_height,) # make room for title
#   # some style sheets have different label sizes
#   if stylesheet.lower() in ('myggplot','ggplot'):
#     margins = list(margins)
#     margins[0] += 0.015; margins[1] -= 0.01 # left, bottom
#     margins[2] += 0.02; margins[3] += 0.02 # width, height
  # handle special TaylorPlot axes
  if lTaylor:
      if not lPolarAxes: lPolarAxes = True
      if not axes_class: axes_class = TaylorAxes
  # handle mixed Polar/Axes
  if isinstance(axes_class, (list,tuple,np.ndarray)):
      for i,axcls in enumerate(axes_class):
          if axcls is None:
              if lTaylor: axes_class[i] = TaylorAxes
              elif lPolarAxes: axes_class[i] = MyPolarAxes
              else: axes_class[i] = MyAxes
          elif axcls.lower() == 'taylor': axes_class[i] = TaylorAxes
          elif axcls.lower() == 'polar': axes_class[i] = MyPolarAxes
          elif axcls.lower() in ('regular','default'): axes_class[i] = MyAxes
          if not issubclass(axcls, Axes): raise TypeError(axcls) 
  # create axes
  if lAxesGrid:
    if share_all is None: share_all = True
    if axes_pad is None: axes_pad = 0.05
    # adjust margins for ignored label pads
    margins = list(margins)
    margins[0] += 0.005; margins[1] -= 0.02 # left, bottom
    margins[2] -= 0.005; margins[3] -= 0.00 # width, height
    # create axes using the Axes Grid package
    if axes_class is None: axes_class=MyLocatableAxes
    fig = mpl.pylab.figure(facecolor='white', figsize=figsize, axes_class=axes_class, 
                           FigureClass=figure_class, **figure_args)
    if axes_args is None: axes_class = (axes_class,{})
    elif isinstance(axes_args,dict): axes_class = (axes_class,axes_args)
    else: raise TypeError
    from mpl_toolkits.axes_grid1 import ImageGrid
    # AxesGrid: http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
    grid = ImageGrid(fig, margins, nrows_ncols = subplot, ngrids=ngrids, direction=direction, 
                     axes_pad=axes_pad, add_all=add_all, share_all=share_all, aspect=aspect, 
                     label_mode=label_mode, cbar_mode=cbar_mode, cbar_location=cbar_location, 
                     cbar_pad=cbar_pad, cbar_size=cbar_size, axes_class=axes_class)
    # return figure and axes
    axes = np.asarray(grid).reshape(subplot) # don't want flattened array
    #axes = tuple([ax for ax in grid]) # this is already flattened  
  elif isinstance(axes_class, (list,tuple,np.ndarray)):
    # PolarAxes can't share axes and by default don't have labels
    if figure_args is None: figure_args = dict()
    fig = figure(facecolor='white', figsize=figsize, FigureClass=figure_class, **figure_args)
    # now create list of axes
    if axes_args is None: axes_args = dict()
    axes = np.empty(subplot, dtype=object); n = 0
    for i in range(subplot[0]):
        for j in range(subplot[1]):
            n += 1
            axes[i,j] = fig.add_subplot(subplot[0], subplot[1], n, axes_class=axes_class[n-1], **axes_args)      
    # just adjust margins
    if axes_pad is None: axes_pad = 0.03
    wspace = hspace = 0.1
    margin_dict = dict(left=margins[0], bottom=margins[1], right=margins[0]+margins[2], 
                       top=margins[1]+margins[3], wspace=wspace, hspace=hspace)
    fig.subplots_adjust(**margin_dict)      
  else:
    # select default axes based on other arguments 
    if axes_class is None:
        if lPolarAxes:
            axes_class = MyPolarAxes
            share_all = sharex = sharey = False
            # N.B.: PolarAxes does not support sharing of axes, and
            #       default behavior is to hide labels
        else:
            axes_class = MyAxes 
    # create axes using normal subplot routine
    if axes_pad is None: axes_pad = 0.03
    wspace = hspace = axes_pad
    if share_all: 
      sharex='all'; sharey='all'
    if sharex is True or sharex is None: sharex = 'col' # default
    if sharey is True or sharey is None: sharey = 'row'
    if sharex: hspace -= 0.015
    if sharey: wspace -= 0.015
    # other axes arguments
    if axes_args is not None and not isinstance(axes_args,dict): raise TypeError
    # create figure
    from matplotlib.pyplot import subplots    
    # GridSpec: http://matplotlib.org/users/gridspec.html 
    fig, axes = subplots(subplot[0], subplot[1], sharex=sharex, sharey=sharey,squeeze=lreduce, 
                         facecolor='white', figsize=figsize, FigureClass=figure_class, 
                         subplot_kw=axes_args, axes_class=axes_class, **figure_args)    
    # there is also a subplot_kw=dict() and fig_kw=dict()
    # just adjust margins
    margin_dict = dict(left=margins[0], bottom=margins[1], right=margins[0]+margins[2], 
                       top=margins[1]+margins[3], wspace=wspace, hspace=hspace)
    fig.subplots_adjust(**margin_dict)
  # apply reduction
  if lreduce:
      if isinstance(axes,np.ndarray): axes = axes.squeeze() # remove singleton dimensions
      if isinstance(axes,(list,tuple,np.ndarray)) and len(axes) == 1: axes = axes[0] # return a bare axes instance, if there is only one axes
  ## set label positions
  if not lPolarAxes:
      # X-/Y-labels and -ticks
      yright = not sharey and subplot[0]==2 if yright is None else yright
      xtop = not sharex and subplot[1]==2 if xtop is None else xtop
      if isinstance(axes, Axes): 
        axes.yright = yright
        axes.xtop = xtop
      else:
        if axes.ndim == 1:
          if subplot[0] == 2: axes[-1].yright = yright # right panel
          if subplot[1] == 2: axes[0].xtop = xtop # top panel
        elif axes.ndim == 2:
          for ax in axes[:,-1]: ax.yright = yright # right column
          for ax in axes[0,:]: ax.xtop = xtop # top row
        else: raise ValueError
  # add figure title
  if name is None: name = title
  if name is not None: fig.canvas.set_window_title(name) # window title
  if title is not None:
      y = 1. - ( title_height / ( 5. if 'x' in title_font else 8. ) ) # smaller title closer to the top
      if isinstance(title_font,basestring): title_font = dict(fontsize=title_font, y=y)
      fig.suptitle(title, **title_font) # title on figure (printable)
  fig.title_height = title_height # save value
  # add default line styles for variables and datasets to axes (figure doesn't need to know)
  if isinstance(axes, np.ndarray):
    for ax in axes.ravel(): 
      ax.variable_plotargs = variable_plotargs
      ax.dataset_plotargs = dataset_plotargs
      ax.plot_labels = plot_labels
  else:
    axes.variable_plotargs = variable_plotargs
    axes.dataset_plotargs = dataset_plotargs
    axes.plot_labels = plot_labels
  # return Figure/ImageGrid and tuple of axes
  #if AxesGrid: fig = grid # return ImageGrid instead of figure
  return fig, axes

if __name__ == '__main__':
    pass