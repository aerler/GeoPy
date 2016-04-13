'''
Created on 2010-11-17

Module that contains functions to compute common meteorological quantities.

@author: Andre R. Erler
'''

from pygeode.atmdyn.constants import Rd, kappa, g0, cp, p0, T0
from pygeode.atmdyn.properties import variablePlotatts # import plot properties from different file

# function to look for available vertical axis
def findZAxis(var):
  from pygeode.axis import Height, ZAxis
  # use geometric height axis if available
  if var.hasaxis(Height): z = var.axes[var.whichaxis(Height)]
  # otherwise use any vertical axis available, and go ahead anyway
  else: z = var.axes[var.whichaxis(ZAxis)]
  return z

# figure out proper vertical axis and perform some checks
def findAxis(var, z, ax):
  from pygeode.axis import Axis, ZAxis, Height
  ## sort out axis ax and field z
  # only var given: use vertical axis of var
  if not z and not ax: z = ax = findZAxis(var)
  # no height field/vector, but axis given: use axis as height vector
  elif not z: z = ax
  # no axis given, but a height field/vector
  elif not ax:  
    # if z is an axis, use it
    if isinstance(z,Axis): ax = z
    # otherwise use vertical axis from var
    else: ax = findZAxis(var)
  # else: everything given
  ## perform error checking
  assert var.hasaxis(ax), 'variable %s does not have axis %s'%(var.name,ax.name)
  # if we only have an axis (no field/vector)
  if isinstance(z,Axis):
    assert ax==z, 'inconsistent input axes'
    assert isinstance(ax,Height), 'axis %s is not an Height axis'%ax.name
  # if z is a field/vector
  else:
    assert isinstance(ax,ZAxis), 'no vertical (ZAxis) axis found'
    assert z.hasaxis(ax), 'field/vector %s does not have axis %s'%(z.name,ax.name) 
  # return properly formated axis
  return z,ax

# density of dry air
def verticalVelocity(w, z, interpType='linear', **kwargs):
  from pygeode.axis import Height, ZAxis
  from pygeode.interp import interpolate
  from warnings import warn
  # figure out axes
  oldZ = findZAxis(w)
  assert isinstance(z,ZAxis), 'interpolation only works along vertical axis'
  if (not isinstance(z,Height)) or (not isinstance(oldZ,Height)):
    warn('The current implementation of verticalVelocity is designed for interpolation between Height coordinates only.')
  assert oldZ.__class__ == z.__class__, 'old and new axes are not of the same type (class)' 
  # interpolate values
  vv = interpolate(w, oldZ, z, interp_type=interpType)
  # attributes (defaults)
  vv.atts['name'] = 'w'
  vv.atts['units'] = 'm/s'
  vv.atts['long_name'] = 'vertical velocity on full model levels' 
  vv.atts['standard_name'] = 'vertical velocity'
  vv.atts['interpolation'] = interpType
  # plot attributes (defaults)
  vv.plotatts = variablePlotatts['w'] 
  # apply user-defined attributes (from kwargs; override defaults)
  vv.atts.update(kwargs)
  # assign short name
  vv.name = vv.atts['name']
  vv.units = vv.atts['units']
  return vv 

# density of dry air
def Rho(T, p, **kwargs): 
  # compute values
  rho = p/(Rd*T)
  # attributes (defaults)
  rho.atts['name'] = 'rho'
  rho.atts['units'] = r'$kg/m^3$'
  rho.atts['long_name'] = 'density of dry air' 
  rho.atts['standard_name'] = r'$\rho$'
  rho.atts['Rd'] = Rd # constants  
  # plot attributes (defaults)
  rho.plotatts = variablePlotatts['rho']
  # apply user-defined attributes (from kwargs; override defaults)
  rho.atts.update(kwargs)
  # assign short name
  rho.name = rho.atts['name']
  rho.units = rho.atts['units']
  return rho 

# potential temperature
def Theta(T, p, **kwargs): 
  # compute values
  th = T*(p0/p)**kappa
  # attributes (defaults)
  th.atts['name'] = 'th'
  th.atts['units'] = 'K'
  th.atts['long_name'] = 'Potential Temperature' # change name 
  th.atts['standard_name'] = r'$\theta$'
  th.atts['p0'] = p0 # constants
  th.atts['kappa'] = kappa
  # plot attributes (defaults)
  th.plotatts = variablePlotatts['th']
  # apply user-defined attributes (from kwargs; override defaults)
  th.atts.update(kwargs)
  # assign short name
  th.name = th.atts['name']
  th.units = th.atts['units']
  return th 

# entropy (of an ideal gas)
def Entropy(t, p=None, **kwargs):
  from pygeode.ufunc import log
  # compute values
  if p: s = cp*log(t/T0) - Rd*log(p/p0) # if pressure is given, assume t is normal temperature
  else: s = cp*log(t/T0) # is no p is given, assume t is potential temperature
  # attributes (defaults)
  s.atts['name'] = 's'
  s.atts['units'] = 'J/(kg K)'
  s.atts['long_name'] = 'Entropy' # change name 
  s.atts['standard_name'] = 's'
  s.atts['cp'] = cp # constants
  s.atts['T0'] = T0
  s.atts['Rd'] = Rd
  s.atts['p0'] = p0
  # plot attributes (defaults)
  s.plotatts = variablePlotatts['s']
  # apply user-defiend attributes (from kwargs; override defaults)
  s.atts.update(kwargs)
  # assign short name
  s.name = s.atts['name']
  s.units = s.atts['units']
  return s

# lapse-rate
def LR(T, z=None, ax=None, **kwargs):
  # figure out axis
  (z, ax) = findAxis(T,z,ax)            
  # Note: if height is available as a variable, pass it explicitly as z-argument
  # compute values
  lr = -1*T.deriv(ax, dx=z)
  # attributes (defaults)
  lr.atts['name'] = 'lr'
  lr.atts['units'] = T.atts['units']+'/'+z.atts['units'] # change units
  lr.atts['long_name'] = 'temperature lapse-rate' # change name 
  lr.atts['standard_name'] = r'$\gamma$'
  # plot attributes (defaults)
  lr.plotatts = variablePlotatts['lr']
  # apply user-defined attributes (from kwargs; override defaults)
  lr.atts.update(kwargs)
  # assign short name
  lr.name = lr.atts['name']
  lr.units = lr.atts['units']
  return lr 

# potential temperature lapse-rate
def ThetaLR(th, z=None, ax=None, **kwargs):
  # figure out axis
  (z, ax) = findAxis(th,z,ax)       
  # Note: if height is available as a variable, pass it explicitly as z-argument
  # compute values
  thlr = th.deriv(ax,dx=z)  
  # attributes (defaults) 
  thlr.atts['name'] = 'thlr'
  thlr.atts['units'] = th.atts['units']+'/'+z.atts['units'] # change units
  thlr.atts['long_name'] = 'potential temperature lapse-rate' # change name 
  thlr.atts['standard_name'] = r'$\partial_z\theta$'
  thlr.atts['p0'] = p0 # theta constants
  thlr.atts['kappa'] = kappa
  # plot attributes (defaults)
  thlr.plotatts = variablePlotatts['thle']
  # apply user-defined attributes (from kwargs; override defaults)
  thlr.atts.update(kwargs)
  # assign short name
  thlr.name = thlr.atts['name']
  thlr.units = thlr.atts['units']
  return thlr 

# Brunt-Vaeisaelae Frequency Squared N2
def N2(th, z=None, ax=None, entropy=False, **kwargs):
  # figure out axis
  (z, ax) = findAxis(th,z,ax)         
  # Note: if height is available as a variable, pass it explicitly as z-argument
  # compute values
  if entropy: # assume w is in fact entropy, and not theta
    nn = (g0/cp)*th.deriv(ax,dx=z) 
  else:  # standard
    nn = g0*th.deriv(ax,dx=z)/th   
  # attributes (defaults)
  nn.atts['name'] = 'N2'
  nn.atts['units'] = r'$1/s^2$' # assign units
  nn.atts['long_name'] = 'Brunt-V\"ais\"al\"a Frequency Squared' # change name 
  nn.atts['standard_name'] = r'$N^2$'
  nn.atts['g0'] = g0 # constants
  nn.atts['cp'] = cp # entropy constants
  nn.atts['T0'] = T0
  nn.atts['Rd'] = Rd
  nn.atts['p0'] = p0
  # plot attributes (defaults)
  nn.plotatts = variablePlotatts['N2']
  # apply user-defined attributes (from kwargs; override defaults)
  nn.atts.update(kwargs)
  # assign short name
  nn.name = nn.atts['name']
  nn.units = nn.atts['units']
  return nn 
  