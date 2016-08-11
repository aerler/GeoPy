'''
Created on 2011-05-30

a module containing functions to compute common atmospheric 2D/surface variables

@author: Andre R. Erler 
'''

from pygeode.atmdyn.properties import variablePlotatts # import plot properties from different file

# new axes for multiple surface values 
from pygeode.axis import ZAxis

## WMO-Tropopause

# axis for TP height 
class TPdef(ZAxis):
  name = 'TPdef' # default name
  units = ''
  plotatts = ZAxis.plotatts.copy()
  plotatts['formatstr'] = '%d' # just print integers
  # Formatting attributes for axis labels and ticks (see formatter method for application)
  plotatts['plottitle'] = 'TP Def. #' # name displayed in plots (axis label)
  plotatts['plotunits'] = '' # displayed units (after offset and scalefactor have been applied) 
  
# a variable class that computes parameters computed from a single column
from pygeode.var import Var
class WMOTP(Var):      
  # initialization
  def __init__(self, T, z=None,axis=ZAxis,bnd=None,threshold=2e-3,deltaZ=2e3,revIdx=False,index=1,name='WMOTP',atts=None,plotatts=None,**kwargs):
    from numpy import asarray        
    # some checks and defaults
    assert T.hasaxis(axis), 'target variable is not defined along fit axis'    
    if z: assert z.hasaxis(axis), 'source variable is not defined along fit axis'              
    ## precondition target variable         
    iaxis = T.whichaxis(axis)
    #TODO: defaults for bnd, depending on type of coordinate
    # if vertical boundaries are given, apply first
    if bnd: T = T(**{T.axes[iaxis].name: bnd})            
    # make vertical axis varying the fastest    
    order = range(0,iaxis) + range(iaxis+1,T.naxes) + [iaxis]
    T = T.transpose (*order)        
    ## precondition height data
    # if a source field is given
    if z:
      iaxis = z.whichaxis(axis) 
      # if vertical boundaries are given, apply first
      if bnd: z = z(**{z.axes[iaxis].name: bnd})
      # make vertical axis varying the fastest    
      order = range(0,iaxis) + range(iaxis+1,z.naxes) + [iaxis]
      z = z.transpose (*order)
    # shorten axis, check again
    axis = T.getaxis(axis.name) # get new (shortened) axis    
    assert T.axes[-1]==axis, 'there is a problem with the vertical axes in T'
    if z: assert z.axes[-1]==axis, 'there is a problem with the vertical axes in z'
    ## create axis for TP height (currently degenerate)
    paxis = TPdef(index) # variable index serves as enumerator for TP definitions
    axes = T.axes[:-1] + (paxis,)
    ## set attributes
    # fit attributes
    defatts = dict()
    defatts['T'] = T.name
    defatts['tgtUnits'] = T.atts['units']
    defatts['threshold'] = threshold
    defatts['deltaZ'] = deltaZ
    if z: 
      defatts['z'] = z.name
      defatts['srcUnits'] = z.atts['units']
    else: 
      defatts['z'] = axis.name
      defatts['srcUnits'] = axis.atts['units']
    defatts['axis'] = axis.name
    defatts['bnd'] = bnd or ''
    defatts['revIdx'] = revIdx
    # variable attributes
    defatts['name'] = name
    defatts['units'] = defatts['srcUnits']
    defatts['long_name'] = 'WMO Tropopause Height' # change name 
    defatts['standard_name'] = 'TP Height'
    if atts: defatts.update(atts)
    # plotatts
    defplotatts = variablePlotatts['z']
    if plotatts: defplotatts.update(plotatts)
    Var.__init__(self, name=name, axes=axes, dtype=T.dtype, atts=defatts, plotatts=defplotatts)
    # save references
    self.T = T # the fit variable
    self.threshold = threshold # lapse-rate threshold that defines the tropopause
    self.deltaZ = deltaZ # vertical depth for which the threshold criterion has to hold 
    self.z = z # the domain on which the model fct. is defined
    self.axis = axis # the axis along which the model operates
    self.bnd = bnd # upper and lower profile boundaries (axis indices or coordinate values)
    self.revIdx = revIdx # reverse index order of target and source field 
    self.args = kwargs # keyword arguments that can be passed to fct (static)   
  # method to compute parameters
  def getview(self, view, pbar):
    from numpy import empty, prod, flipud, fliplr
    from f2py import meteo
#    from multiprocessing import Pool
    ## construct new view
    inView = view.replace_axis(self.naxes-1, self.axis)
    ## get data
    T = inView.map_to(self.T).get(self.T)
    te = prod(T.shape[:-1]) # number of profiles
    ke = T.shape[-1] # vertical levels
    if self.z:
      z = inView.map_to(self.z).get(self.z)
      assert z.shape==T.shape
      ze = te
    else:
      z = self.axis.values
      ze = 1        
    # cast arrays into two-dimensional array to loop over profiles
    T = T.reshape((te,ke))
    z = z.reshape((ze,ke))
    # allocate output data
    zTP = empty(view.shape, dtype=self.dtype)
    zTP = zTP.reshape((te,zTP.shape[-1])) # TP-axis is degenerate anyway, but well...
    # if array starts with highest level, flip array
    if self.revIdx:
      T = fliplr(T)
      if self.z: z = fliplr(z)
      else: z = flipud(z)
    # determine tropopause height according to WMO definition        
    zTP = meteo.tropopausewmo(T, z, self.threshold, self.deltaZ, te, ze, ke)
    return zTP.reshape(view.shape)  


## Isentropic Coordinates

# axis for isentropic coordinates 
class Isentrope(ZAxis):
  name = 'theta' # default name
  units = 'K'
  plotatts = ZAxis.plotatts.copy()
  plotatts['formatstr'] = '%d' # just print integers
  # Formatting attributes for axis labels and ticks (see formatter method for application)
  plotatts['plottitle'] = 'Potentential Temperature' # name displayed in plots (axis label)
  plotatts['plotunits'] = 'K' # displayed units (after offset and scalefactor have been applied) 
   
# interpolate to isentropic surface
def interp2theta(var, theta, values, interp='linear', **kwargs):
  from pygeode.interp import interpolate, sorted
  # inaxis axis
  iaxis = var.whichaxis(ZAxis)
#  # sort theta (must be monotonically increasing)
#  var = sorted(var, iaxis, reverse=False)
#  theta = sorted(theta, iaxis, reverse=False)
  # prepare input
  inaxis = var.axes[iaxis]
  assert theta.hasaxis(inaxis), 'vertical axis of var and theta are incompatible' 
  # create new axis
  outaxis = Isentrope(values=values, **kwargs)  
  # interpolate to isentropic levels
  ivar = interpolate(var, inaxis, outaxis, inx=theta, interp_type=interp)
  # return variable interpolated to isentropic level(s)
  return ivar    

## Dynamical Tropopause
  
# axis for PV iso-surfaces (dynamical tropopause)
  class DynamicalTP(ZAxis):  
    name = 'PViso' # default name
    units = '(K m^2)/(s kg)'
    plotatts = ZAxis.plotatts.copy()
    plotatts['formatstr'] = '%3.1f' # one digit behind decimal 
    # Formatting attributes for axis labels and ticks (see formatter method for application)
    plotatts['plottitle'] = 'Dynamical TP' # name displayed in plots (axis label)
    plotatts['plotunits'] = 'PVU' # displayed units (after offset and scalefactor have been applied)
    plotatts['scalefactor'] = 1e6 # conversion factor; assumed units are meters