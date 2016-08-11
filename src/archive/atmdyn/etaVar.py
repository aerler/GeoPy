'''
Created on 2011-02-24

compute some quantities from hybrid vertical coordinates
everything here is specific to hybrid coordinates 

@author: Andre R. Erler
'''

from pygeode.atmdyn.constants import Rd, g0
from pygeode.atmdyn.properties import variablePlotatts # import plot properties from different file
from pygeode.var import Var

# surface pressure from log-surface-pressure
def SurfacePressure(lnsp):
  # exponentiate
  ps = lnsp.exp()
  # and add meta data
  ps.name = 'ps'
  # attributes
  ps.atts = {}
  ps.atts['name'] = ps.name
  ps.atts['long_name'] = 'surface pressure' 
  ps.atts['standard_name'] = 'surface pressure'
  ps.atts['units'] = 'Pa' # currently units are Pa (as in surface pressure)
  # plot attributes (defaults)
  ps.plotatts = variablePlotatts['ps']
  return ps
  
# pressure from eta coordinates (on half levels)
def Pressure(ps, eta):
  from numpy import empty
  from pygeode.axis import TAxis
  from pygeode.varoperations import transpose 
  # Note: the 'A' and 'B' coefficients are assumed to be stored as 
  # auxiliary arrays with the eta-axis 
  # for now just extend A and B
  A = empty(len(eta)+1); B = empty(len(eta)+1);
  A[1:] = eta.auxarrays['A']; B[1:] = eta.auxarrays['B']
  A[0] = 0; B[0] = 0; # this is correct: highest half-level has zero pressure  
  # compute A and B on full levels and substitute back into eta axis
  eta.auxarrays['A'] = (A[1:] + A[0:-1]) /2
  eta.auxarrays['B'] = (B[1:] + B[0:-1]) /2
  # compute pressure values from parameters stored with axis eta
  p = eta.auxasvar('A') + eta.auxasvar('B')*ps
  # transpose into common order: time first, then eta
  if p.hasaxis(TAxis): 
    p = transpose(p,p.axes[p.whichaxis(TAxis)].name,eta.name)  
  # short name
  p.name = 'p'
  # attributes
  p.atts['name'] = p.name
  p.atts['long_name'] = 'pressure' 
  p.atts['standard_name'] = 'pressure'
  p.atts['units'] = 'Pa' # currently units are Pa (as in surface pressure)
  # plot attributes (defaults)
  p.plotatts = variablePlotatts['p']
  return p

# return a geopotential variable 
def GeopotHeight(T, ps, phis):
  Z = GeopotHeightVar(T, ps, phis)
  # enforce same axis order as T
  order = []
  for ax in T.axes:
    order.append(Z.whichaxis(ax))    
  return Z.transpose(*order) 

# geopotential height in eta coordinates (on full model levels)
class GeopotHeightVar(Var):
  # geopotential height is geopotential divided by standard gravity g0
  # geopotential can be computed from temperature and pressure
  # initialization   
  # Note: currently the pressure field is computed on the fly from eta-coefficients 
  # and surface pressure; this is faster than using a pre-computed 3D pressure field. 
  def __init__(self, T, ps, phis):
    from numpy import diff
    from pygeode.axis import Hybrid
    from pygeode.varoperations import sorted    
    # precondition input and store internally
    assert T.hasaxis(Hybrid), 'this function only computes geopotential from hybrid coordinates'      
    # T: make vertical axis varying the slowest (default is 2nd slowest, after time)   
    ietaT = T.whichaxis(Hybrid)
    inOrder = [ietaT] + range(0,ietaT) + range(ietaT+1,T.naxes)
    self.T = T.transpose(*inOrder)
    # surface fields
    self.ps = ps # surface pressure
    self.phis = phis
    # get vertical coefficients for hybrid coordinate
    self.A = T.axes[ietaT].auxarrays['A']
    self.B = T.axes[ietaT].auxarrays['B']
    # construct output axes: make eta varying the fastest, to prevent break-up in loop-over routine    
    outAxes = T.axes[0:ietaT] + T.axes[ietaT+1:] + (T.axes[ietaT],)    
    # ensure eta axis is ordered properly
    if not all(diff(self.T.eta.values)>0):
      self.T = sorted(self.T, eta=1)
      from warnings import warn
      warn('The vertical axis (eta) was not in the expected order - the sorted-fct. has been applied.')            
    # attributes
    atts = {}
    atts['name'] = 'z'
    atts['long_name'] = 'geopotential height' 
    atts['standard_name'] = 'geopotential'
    atts['units'] = 'm' 
    atts['g0'] = g0
    atts['Rd'] = Rd
    # plot attributes (defaults)
    plotatts = variablePlotatts['z']
    # make proper Var-instance
    Var.__init__(self, axes=outAxes, dtype=self.T.dtype, name='z', values=None, atts=atts, plotatts=plotatts)
    self.ieta = self.whichaxis(Hybrid)
    # make sure axes are assigned properly and eta is the innermost axis
    assert self.naxes == T.naxes
    assert self.ieta == T.naxes-1    
  # actual computation
  def getview(self, view, pbar):
    from numpy import empty, prod, log #, arange, min
    # Geopotential requires the integration of pressure differences and temperature;
    # technically T has to be virtual temperature, but I'm ignoring that for now.
    # The computation is performed explicitly, level by level.
    # I believe the temperature data is actually on full levels. 
    # Geopotential is computed on full levels as well.
    # source: IFS (31r1) Documentation, Part 3, pp. 6-8 
    # url: http://www.ecmwf.int/research/ifsdocs/CY31r1/index.html
    # detect size and shape
    lev = view.integer_indices[self.ieta] # actually requested levels
    ie = prod(view.shape[0:self.ieta]+view.shape[self.ieta+1:]) # all non-vertical coordinates      
    # construct new view for input variables 
    #TODO: implement slicing more efficiently: only extend axis towards bottom (top is not necessary)
#    minLev = min(lev); lev = lev - minLev # adjust lev to use for indexing of extended field
#    newLev = arange(minLev,self.shape[self.ieta])
#    inView = view.modify_slice(self.ieta, newLev) 
    # NOTE: it entirely escapes my comprehension, why the above does not work, but it gives ridiculous results
    inView = view.unslice(self.ieta) # just request entire axis... actually not necessary but simpler
    ke = inView.shape[self.ieta] # length of vertical coordinate
    # get data and cast into 2D array
    T = inView.get(self.T).reshape(ke,ie) # map_to(self.T.axes).    
    T = (Rd/g0) * T # scale T (avoids some unnecessary operations)    
    # allocate output data
    phi = empty((ke,ie), dtype=self.dtype)
    # ps & phi0 have different axes (2D)
    ps = view.get(self.ps).reshape(1,ie)
    phis = view.get(self.phis).reshape((1,ie))    
    # initial conditions on half-levels
    hlPhi = phis.copy()/g0 # convert to height (divide by g0)
    # compute half-level pressures adjacent to first model level               
    pp = self.A[ke-1] + self.B[ke-1]*ps
    pm = self.A[ke-2] + self.B[ke-2]*ps
    # special treatment of first model level (full level) 
    tmp = log(pp/pm) # used later
    phi[ke-1,:] = hlPhi + T[ke-1,:]*(1 - tmp*pm/(pp-pm));    
    # loop over levels in reverse order    
    for k in range(ke-2,0,-1):      
      # compute half-level geopotential
      hlPhi += T[k,:] * tmp 
      # advance pressure calculation 
      pp = pm.copy(); pm = self.A[k-1] + self.B[k-1]*ps
      tmp = log(pp/pm)      
      # correction has to be applied to get full levels      
      phi[k,:] = hlPhi + T[k,:]*(1 - tmp*pm/(pp-pm)) # apply correction, store value
    # last step requires special treatment (to avoid index out of bounds in pm      
    hlPhi += T[k,:] * log(pp/pm)
    phi[0,:] = hlPhi + T[0,:]*log(2) # apply different correction
    # extract the requested slice
    phi = phi[lev,:]
    # return value in correct shape: transpose to make eta innermost axis, apply desired shape    
    return phi.transpose().reshape(view.shape)

# compute geopotential height using the integrate function 
#def GeopotHeight(T, p, phi0):
#  from pygeode.intgr import integrate
#  from pygeode.deriv import deriv
#  eta = T.getaxis(Hybrid)
#  dphi = Rd/g0 * T * deriv(p.log(), eta)
#  phi = integrate(dphi, eta, v0=phi0, order=-1)
#  phi.name = 'phi'
#  # attributes
#  phi.atts = {}
#  phi.atts['name'] = 'phi'
#  phi.atts['long_name'] = 'geopotential height' 
#  phi.atts['standard_name'] = 'geopotential'
#  phi.atts['units'] = 'm' 
#  return phi 