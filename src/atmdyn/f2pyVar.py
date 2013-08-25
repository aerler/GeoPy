'''
Created on 2011-05-17

f2py wrappers for Fortran routines

@author: Andre R. Erler
'''

from pygeode.atmdyn.properties import variablePlotatts # import plot properties from different file
from pygeode.var import Var

class RelativeVorticity(Var):
  '''
  Reltive Vorticity (vertical component).
  This is a wrapper class that overloads getview and calls a Fortran library via f2py. 
  '''
  def __init__(self, u, v, perix=True, name=None, atts=None, plotatts=None):
    '''
      relativeVorticity(u, v, perix=True, name=None, atts=None, plotatts=None)
    '''
    from pygeode.axis import Lat, Lon
    from pygeode.atmdyn.constants import Re       
    ## check axes
    # order: latitude and longitude have to be the last two
    assert u.whichaxis(Lat)==u.naxes-2 and u.whichaxis(Lon)==u.naxes-1, 'Unsupported axes order.'
    # homogeneity
    assert u.axes==v.axes, 'Axes are incompatible!'            
    # handle meta data
    if name is None: name = 'ze'
    # parameter (set defaults, if not present)
    defatts = {'name': 'zeta', 'units': r'$s^{-1}$', 'long_name': 'Relative Vorticity (vertical)', 
               'standard_name': 'Relative Vorticity', 'Re': Re, 'perix':perix}      
    if atts: defatts.update(atts)
    # plot parameter
    defplot = variablePlotatts['ze'] 
    if plotatts: defplot.update(plotatts)
    # make proper Var-instance
    Var.__init__(self, axes=u.axes, dtype=u.dtype, name=name, values=None, atts=defatts, plotatts=defplot)
    # save variables and parameters
    self.perix = perix
    self.u = u; self.v = v
    self.lon = u.getaxis(Lon); self.lat = u.getaxis(Lat)
    self.Re = self.atts['Re']
  
  def getview(self, view, pbar):
    # Here the call to the Fortran library happens in all its glory. 
    from numpy import empty, append, product
    from warnings import warn
    from f2py import meteo  
    from pygeode.axis import XAxis
#    print(" >>>>> Hi, I'm a zeta instance! <<<<<") # to see how often this is called
    # extend view
    extView = view
    lperix = self.perix # switch for x-boundary handling for fortran code
    # extend view in lat and lon direction
    for i in [self.naxes-2, self.naxes-1]: 
      # check if extension is possible
      if not view.shape[i]==self.shape[i]:
        intidx = extView.integer_indices[i] # get all indices
        if min(intidx)>0: intidx = append((min(intidx)-1),intidx) # extend below
        if max(intidx)<self.shape[i]-1: intidx = append(intidx,(max(intidx)+1)) # extend above
        # substitute new slice  
        extView = extView.modify_slice(i, intidx)   
        # don't assume periodicity when x-axis is split
        if isinstance(self.axes[i],XAxis): self.perix = False
        #TODO: implement smart handling (periodic extension) of boundary intervalls    
    # get input data    
    u = extView.get(self.u)
    v = extView.get(self.v)
    extShape = u.shape # old shape (used to reshape output)
    # compress external axes
    zipShape = (product(extShape[0:-2]),extShape[-2],extShape[-1])
    u = u.reshape(zipShape)
    v = v.reshape(zipShape)
    # handle internal axes (lat&lon)      
    lat = extView.get(self.lat).squeeze() # vector required for Coriolis force
    dlon = self.lon.values[1]-self.lon.values[0] # only spacing required (assumed regular)
    warn ("assuming regular spacing of '%s' axis"%self.lon.name)
    # allocate output array (this helps!)
    zeta = empty(zipShape)
    # do number crunching (in Fortran)    
    zeta = meteo.relativevorticity(u,v,lat,dlon,self.Re,lperix,*zipShape)
    # inflate external axes
    zeta = zeta.reshape(extShape)
    # discard boundary points (trim to original view)
    for i in [self.naxes-2, self.naxes-1]:
      # check if reduction is necessary
      if not view.shape[i]==extView.shape[i]:
        extidx = extView.integer_indices[i] # get current indices
        idx = view.integer_indices[i] # get requested indices
        lower = 0; upper = len(extidx) 
        if min(extidx)<min(idx): lower = lower+1 # cut off first
        if max(extidx)>max(idx): upper = upper-1 # cut off last
        # trim axis
        zeta = zeta.take(range(lower,upper),axis=i)                        
    # return output
    return zeta


class PotentialVorticity(Var):
  '''
  Potential Vorticity.
  This is a wrapper class that overloads getview and calls a Fortran library via f2py. 
  '''
  def __init__(self, u, v, th, rho, w=None, z=None, perix=True, name=None, atts=None, plotatts=None):
    '''
      PotentialVorticity(u, v, th, rho, w=None, z=None, name=None, atts=None, plotatts=None)
    '''
    from pygeode.axis import Lat, Lon, Height, ZAxis, TAxis
    from pygeode.atmdyn.constants import Re, Omega       
    ## check axes
    # order
    assert u.whichaxis(TAxis)==0 and u.whichaxis(Lat)==2 and u.whichaxis(Lon)==3, 'Unsupported axes order.'
    # homogeneity
    assert u.axes==v.axes, 'Axes are incompatible!'
    assert th.axes==rho.axes, 'Axes are incompatible!'    
    assert u.axes==th.axes, 'Axes are incompatible!'        
    if w: assert u.axes==w.axes, 'Axes are incompatible!' # should have same axes as u & v
    if z: # z is a field, e.g. geopotential on hybrid axis
      zField = True
      assert u.whichaxis(ZAxis)==1, 'Position of vertical axis is not supported.'
      if not z.axes==th.axes: # sort z's axes if necessary
        order = []
        for ax in th.axes:
          order.append(ax.name)
          z = z.transpose(*order)
#      assert z.axes==th.axes, 'Axes are incompatible!' # should have same axes as th & rho 
    else: # just use height axis as z-field 
      zField = False
      u.whichaxis(Height)==1, 'Position of vertical axis is not supported.'
      z = u.getaxis(Height) # expand to field later
    # handle meta data
    if th.name=='th':
      if name is None: name = 'PV'
      # parameter (set defaults, if not present
      defatts = {'name': 'PV', 'units': r'$K m^2 (s kg)^{-1}$', 'long_name': 'Ertel Potential Vorticity', 
                 'standard_name': 'isentropic PV', 'Re': Re, 'Omega': Omega, 'perix':perix}      
    elif th.name=='s':
      if name is None: name = 'PVs'
      # parameter (set defaults, if not present
      defatts = {'name': 'PVs', 'units': r'$J m^2 (K s)^{-1} kg^{-2}$', 'long_name': 'Entropy Potential Vorticity', 
                 'standard_name': 'Entropy PV', 'Re': Re, 'Omega': Omega, 'perix':perix}      
    if atts: defatts.update(atts)
    # plot parameter
    defplot = variablePlotatts[name] 
    if plotatts: defplot.update(plotatts)
    # make proper Var-instance
    Var.__init__(self, axes=th.axes, dtype=th.dtype, name=name, values=None, atts=defatts, plotatts=defplot)
    # save variables and parameters
    self.perix = perix
    self.zField = zField
    self.u = u; self.v = v; self.w = w
    self.th = th; self.rho = rho; self.z = z
    self.lon = u.getaxis(Lon); self.lat = u.getaxis(Lat)
    self.Re = self.atts['Re']; self.Omega = self.atts['Omega']
  
  def getview(self, view, pbar):
    '''
      Here the call to the Fortran library happens in all its glory.
      Note: My own Fortran/f2py implementation is more than twice as fast as the NumPy version! 
    '''
    from numpy import empty, zeros, append
    from warnings import warn
    from f2py import meteo  
    from pygeode.axis import XAxis
#    print(" >>>>> Hi, I'm a PV instance! <<<<<") # to see how often this is called
    # extend view
    extView = view
    lperix = self.perix # switch for x-boundary handling for fortran code
    # extend view in vertical, lat, and lon direction (but not in time!)
    for i in range(1,self.naxes): 
      # check if extension is possible
      if not view.shape[i]==self.shape[i]:
        intidx = extView.integer_indices[i] # get all indices
        if min(intidx)>0: intidx = append((min(intidx)-1),intidx) # extend below
        if max(intidx)<self.shape[i]-1: intidx = append(intidx,(max(intidx)+1)) # extend above
        # substitute new slice  
        extView = extView.modify_slice(i, intidx)   
        # don't assume periodicity when x-axis is split
        if isinstance(self.axes[i],XAxis): self.perix = False
        #TODO: implement smart handling (periodic extension) of boundary intervalls    
    # get input data    
    u = extView.get(self.u)
    v = extView.get(self.v)
    th = extView.get(self.th)
    rho = extView.get(self.rho)
    extShape = u.shape # old shape (used to reshape output)
    # handle optional input fields
    if self.zField: 
      z = extView.get(self.z) 
    else: 
      z = extView.get(self.z).squeeze()
      # w is only needed if z is a vector
      if self.w: w = extView.get(self.w)
      else: w = zeros(extShape) # just neglect w in computation
    # remaining axes      
    lat = extView.get(self.lat).squeeze() # vector required for Coriolis force
    dlon = self.lon.values[1]-self.lon.values[0] # only spacing required (assumed regular)
    warn ("assuming regular spacing of '%s' axis"%self.lon.name)
    # allocate output array (this helps!)
    PV = empty(extShape)
    # do number crunching (in Fortran)    
    if self.zField:
      PV = meteo.potentialvorticityifs(u,v,th,rho,z,lat,dlon,self.Re,self.Omega,lperix,*extShape)
    else:
      PV = meteo.potentialvorticitylm(u,v,w,th,rho,lat,dlon,z,self.Re,self.Omega,lperix,*extShape)      
    # discard boundary points (trim to original view)
    for i in range(self.naxes):
      # check if reduction is necessary
      if not view.shape[i]==extView.shape[i]:
        extidx = extView.integer_indices[i] # get current indices
        idx = view.integer_indices[i] # get requested indices
        lower = 0; upper = len(extidx) 
        if min(extidx)<min(idx): lower = lower+1 # cut off first
        if max(extidx)>max(idx): upper = upper-1 # cut off last
        # trim axis
        PV = PV.take(range(lower,upper),axis=i)                        
    # return output
    return PV

class Theta(Var):
  '''
  Potential Temperature.
  This is a wrapper class that overloads getview and calls a Fortran library via f2py. 
  '''
  def __init__(self, T, p, name=None, atts=None, plotatts=None):
    '''
      Theta(T, p, name=None, atts=None, plotatts=None)
    '''
    from pygeode.atmdyn.constants import p0, kappa
    # input checks    
    if not T.axes==p.axes:
      # need to transpose p to fix axes order between T and p      
      iaxes = range(len(T.axes))
      for i in range(len(T.axes)):
        
        assert p.hasaxis(T.axes[i]), 'Axes of T and p are incompatible!'
        iaxes[p.whichaxis(T.axes[i])] = i # order of axes in p
      p = p.transpose(*iaxes)
    # handle meta data
    if name is None: name = 'th'
    # parameter (set defaults, if not present
    defatts = {'name': 'th', 'units': 'K', 'long_name': 'potential temperature', 
               'standard_name': r'$\theta$', 'p0': p0, 'kappa': kappa}
    if atts: defatts.update(atts)
    # plot parameter
    defplot = {'plottitle': 'theta', 'plotunits': 'K'}
    if plotatts: defplot.update(plotatts)
    # make proper Var-instance
    Var.__init__(self, axes=T.axes, dtype=T.dtype, name=name, values=None, atts=defatts, plotatts=plotatts)
    # save T & p and parameters
    self.T = T; self.p = p
    self.p0 = self.atts['p0']; self.kappa = self.atts['kappa']
  
  def getview(self, view, pbar):
    '''
      Here the call to the Fortran library happens in all its glory.
      Note: My own Fortran/f2py implementation is more than twice as fast as the NumPy version! 
    '''
    from f2py import meteo    
    # get input data
    inview = view # trivial here
    T = inview.get(self.T)
    p = inview.get(self.p)
    # reshape to 1D
    viewShape = T.shape # old shape (used to reshape output)
    T = T.reshape(T.size)
    p = p.reshape(p.size)
    # do number crunching (in Fortran)    
    th = meteo.potentialtemperature(T,p,self.p0,self.kappa)
#    th = T*(self.p0/p)**self.kappa # NumPy
    # reshape output
    th = th.reshape(viewShape)
    # return output
    return th