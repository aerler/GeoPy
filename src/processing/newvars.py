'''
Created on May 5, 2016

A collection of functions to compute derived variables, primarily for WRF but also in general.

@author: Andre R. Erler, GPL v3
'''

# external imports
from warnings import warn
import numpy as np
from numexpr import evaluate, set_num_threads, set_vml_num_threads
# numexpr parallelisation: don't parallelize at this point!
set_num_threads(1); set_vml_num_threads(1)
# internal imports
from geodata.base import Dataset, Variable, VariableError, AxisError
from geodata.misc import days_per_month, days_per_month_365, DatasetError
from utils.constants import sig, lw, pi # used in calculation for clack body radiation

## helper functions

# net radiation balance using black-body radiation from skin temperature
def radiation_black(A, SW, LW, e, Ts, TSmax=None):
  ''' net radiation  [W/m^2] at the surface: downwelling long and short wave minus upwelling terrestrial radiation '''
  if TSmax is None:
    # using average skin temperature for terrestrial long wave emission
    warn('Using average skin temperature; diurnal min/max skin temperature is preferable due to strong nonlinearity.')
    return evaluate('( ( 1 - A ) * SW ) + ( LW * e ) - ( e * sig * Ts**4 )')
  else:
    # using min/max skin temperature to account for nonlinearity
    return evaluate('( ( 1 - A ) * SW ) + ( LW * e ) - ( e * sig * ( Ts**4 + TSmax**4 ) / 2 )')

# net radiation balance using accumulated quantities
def radiation(SWDN, LWDN, SWUP, LWUP, ):
  ''' net radiation  [W/m^2] at the surface: downwelling long and short wave minus upwelling '''
  # using min/max skin temperature to account for nonlinearity
  return evaluate('SWDN + LWDN - SWUP - LWUP')

# 2m wind speed [m/s]
def wind(u, v=None, z=10):
  ''' approximate 2m wind speed [m/s] from wind speed at different height (z [m]; default 10m)
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#wind%20profile%20relationship)
  '''
  if v is not None: u = evaluate('sqrt( 5*u**2 + 10*v**2 )') # estimate wind speed from u and v components
  # N.B.: the scale factors (5&10) are necessary, because absolute wind speeds are about 2.5 times higher than
  #       the mean of the compnents. This is because opposing directions average to zero.
  return evaluate('( u * 4.87 ) / log( 67.8 * z - 5.42 )')

def computeSrfcPressure(dataset, zs=None, lpmsl=True):
  ''' compute surface pressure [Pa] from elevation using Eq. 7 from FAO:
      http://www.fao.org/3/x0490e/x0490e07.htm#atmospheric%20pressure%20(p) 
  '''
  zs, zs_units = _inferElev(dataset, zs=zs, latts=True, lunits=True) 
  assert zs_units is None or zs_units == 'm', zs
  pmsl = dataset['pmsl'][:] if lpmsl and 'pmsl' in dataset else 1013.
  p = evaluate('pmsl * ( 1 - ( 0.0065*zs/293 ) )**5.26 ')
  # return surface pressure estimate in Pa
  return p      

# psychrometric constant [Pa/K]
def gamma(p):
  ''' psychrometric constant [Pa/K] for a given air pressure [Pa]
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#psychrometric%20constant%20%28g%29) 
  '''
  return 665.e-6 * p

# slope of saturation vapor pressure [Pa/K]
def Delta(T):
  ''' compute the slope of saturation vapor pressure [Pa/K] relative to temperature T [K]
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#calculation%20procedures)
  '''
  return evaluate('4098 * ( 610.8 * exp( 17.27 * (T - 273.15) / (T - 35.85) ) ) / (T - 35.85)**2')

# saturation vapor pressure [Pa]
def e_sat(T, Tmax=None):
  ''' compute saturation vapor pressure [Pa] for given temperature [K]; average from Tmin & Tmax
      is also supported
      (Magnus Formula: http://www.fao.org/docrep/x0490e/x0490e07.htm#calculation%20procedures) 
  '''
  if Tmax is None: 
    # Magnus formula
    #warn('Using average 2m temperature; diurnal min/max 2m temperature is preferable due to strong nonlinearity.')
    return evaluate('610.8 * exp( 17.27 * (T - 273.15) / (T - 35.85) )')
  else:
    # use average of saturation pressure from Tmin and Tmax (because of nonlinearity)
    return evaluate('305.4 * ( exp( 17.27 * (T - 273.15) / (T - 35.85) ) + exp( 17.625 * (Tmax - 273.15) / (Tmax - 35.85) ) )')

## functions to compute relevant variables (from a dataset)

# compute net radiation (for PET)
def computeNetRadiation(dataset, asVar=True, lA=True, lem=True, lrad=True, name='netrad'):
  ''' function to compute net radiation at surface for Penman-Monteith equation
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#radiation)
  '''
  if lrad and 'SWDNB' in dataset and 'LWDNB' in dataset and 'SWUPB' in dataset and 'LWUPB' in dataset:
    data = radiation(dataset['SWDNB'][:],dataset['LWDNB'][:],dataset['SWUPB'][:],dataset['LWUPB'][:]) # downward total net radiation
  else:
    if not lem: em = 0.93 # average value for soil
    elif lem and 'e' in dataset: em = dataset['e'][:]
    else:
        raise VariableError("Emissivity is not available for radiation calculation.")
    if not lA: A = 0.23 # reference Albedo for grass
    elif lA and 'A' in dataset: A = dataset['A'][:]
    else: 
        raise VariableError("Actual Albedo is not available for radiation calculation.")
    if 'TSmin' in dataset and 'TSmax' in dataset: Ts = dataset['TSmin'][:]; TSmax = dataset['TSmax'][:]
    elif 'TSmean' in dataset: Ts = dataset['TSmean'][:]; TSmax = None
    elif 'Ts' in dataset: Ts = dataset['Ts'][:]; TSmax = None
    elif 'Tmin' in dataset and 'Tmax' in dataset: Ts = dataset['Tmin'][:]; TSmax = dataset['Tmax'][:]
    elif 'T2' in dataset: Ts = dataset['T2'][:]; TSmax = None
    else: 
        raise VariableError("Either 'Ts' or 'TSmean' are required to compute net radiation for PET calculation.")
    if 'LWDNB' in dataset: GLW = dataset['LWDNB'][:]
    elif 'GLW' in dataset: GLW = dataset['GLW'][:]
    else: 
        raise VariableError("Downwelling LW radiation is not available for radiation calculation.")
    if 'SWDNB' in dataset: SWD = dataset['SWDNB'][:]
    elif 'SWD' in dataset: SWD = dataset['SWD'][:]
    else: 
        raise VariableError("Downwelling SW radiation is not available for radiation calculation.")
    data = radiation_black(A,SWD,GLW,em,Ts,TSmax) # downward total net radiation
  # cast as Variable
  if asVar:
    var = Variable(data=data, name=name, units='W/m^2', axes=dataset['SWD'].axes)
  else: var = data
  # return new variable
  return var

# compute potential evapo-transpiration
def computeVaporDeficit(dataset):
  ''' function to compute water vapor deficit for Penman-Monteith PET
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#air%20humidity)
  '''
  if 'Q2' in dataset: ea = dataset['Q2'][:] # actual vapor pressure
  elif 'q2' in dataset and 'ps' in dataset: # water vapor mixing ratio
    ea = dataset['q2'][:] * dataset['ps'][:] * 28.96 / 18.02
  else: raise VariableError("Cannot determine 2m water vapor pressure for PET calculation.")
  # get saturation water vapor
  if 'Tmin' in dataset and 'Tmax' in dataset: es = e_sat(dataset['Tmin'][:],dataset['Tmax'][:])
  # else: Es = e_sat(T) # backup, but not very accurate
  else: raise VariableError("'Tmin' and 'Tmax' are required to compute saturation water vapor pressure for PET calculation.")
  var = Variable(data=es-ea, name='vapdef', units='Pa', axes=dataset['Tmin'].axes)
  # return new variable
  return var

def _getTemperatures(dataset, lmeans=False, lxarray=False):
    ''' helper functions to extract temperature variables '''
    # get diurnal min/max 2m temperatures
    if 'Tmin' in dataset: 
        Tmin = dataset['Tmin']; min_units = dataset['Tmin'].units
        Tmin = Tmin.data if lxarray else Tmin[:]
    else: raise VariableError("Cannot determine diurnal minimum 2m temperature for PET calculation.")
    if 'Tmax' in dataset: 
        Tmax = dataset['Tmax']; max_units = dataset['Tmax'].units
        Tmax = Tmax.data if lxarray else Tmax[:]
    else: raise VariableError("Cannot determine diurnal maximum 2m temperature for PET calculation.")
    assert max_units == min_units, (max_units,min_units)
    # get 2m mean temperature
    if lmeans and 'Tmean' in dataset: 
        T = dataset['Tmean'][:]; t_units = dataset['Tmean'].units
    elif 'T2' in dataset: 
        T = dataset['T2'][:]; t_units = dataset['T2'].units
    else:
        # estimate mean temperature as average of diurnal minimum and maximum
        T = (Tmax+Tmin)/2.; t_units = min_units
    assert t_units == min_units, (t_units,min_units)
    # return 
    return T, Tmin, Tmax, t_units
  

# compute potential evapo-transpiration with all bells and whistles
def computePotEvapPM(dataset, lterms=True, lmeans=False, lrad=True, lgrdflx=True, lpmsl=True, lxarray=False, **kwargs):
  ''' function to compute potential evapotranspiration (according to Penman-Monteith method:
      https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation,
      http://www.fao.org/docrep/x0490e/x0490e06.htm#formulation%20of%20the%20penman%20monteith%20equation)
  '''
  if lxarray: raise NotImplementedError()
  # get net radiation at surface
  if 'netrad' in dataset: Rn = dataset['netrad'][:] # net radiation
  if 'Rn' in dataset: Rn = dataset['Rn'][:] # alias
  else: Rn = computeNetRadiation(dataset, lrad=lrad, asVar=False) # try to compute
  # heat flux in and out of the ground
  if lgrdflx:
      if 'grdflx' in dataset: G = dataset['grdflx'][:] # heat release by the soil
      else: raise VariableError("Cannot determine soil heat flux for PET calculation.")
  else: G = 0
  # get wind speed
  if 'U2' in dataset: u2 = dataset['U2'][:]
  elif lmeans and 'U10' in dataset: u2 = wind(dataset['U10'][:], z=10)
  elif 'u10' in dataset and 'v10' in dataset: u2 = wind(u=dataset['u10'][:],v=dataset['v10'][:], z=10)
  else: raise VariableError("Cannot determine 2m wind speed for PET calculation.")
  # get psychrometric variables
  if 'ps' in dataset: p = dataset['ps'][:]
  elif 'zs' in dataset: p = computeSrfcPressure(dataset, lpmsl=lpmsl)
  else: raise VariableError("Cannot determine surface air pressure for PET calculation.")
  g = gamma(p) # psychrometric constant (pressure-dependent)
  if 'Q2' in dataset: ea = dataset['Q2'][:]
  elif 'q2' in dataset:
      q2 = dataset['q2'][:] 
      ea =  evaluate('q2 * p * 28.96 / 18.02')
  else: raise VariableError("Cannot determine 2m water vapor pressure for PET calculation.")
  # get temperature
  T, Tmin, Tmax, t_units = _getTemperatures(dataset, lmeans=lmeans)
  assert t_units == 'K', t_units
  # get saturation water vapor
  if 'Tmin' in dataset and 'Tmax' in dataset: es = e_sat(Tmin,Tmax)
  # else: Es = e_sat(T) # backup, but not very accurate
  else: raise VariableError("'Tmin' and 'Tmax' are required to compute saturation water vapor pressure for PET calculation.")
  D = Delta(T) # slope of saturation vapor pressure w.r.t. temperature
  # determine reference variable (see what's available)
  for refvar in ('ps','T2','Tmax','Tmin','Q2','q2','U10','u10',):
      if refvar in dataset: break
  # compute potential evapotranspiration according to Penman-Monteith method 
  # (http://www.fao.org/docrep/x0490e/x0490e06.htm#fao%20penman%20monteith%20equation)
  if lterms:
    Dgu = evaluate('( D + g * (1 + 0.34 * u2) ) * 86400') # common denominator
    rad = evaluate('0.0352512 * D * (Rn + G) / Dgu') # radiation term
    wnd = evaluate('g * u2 * (es - ea) * 0.9 / T / Dgu') # wind term (vapor deficit)
    pet = evaluate('( 0.0352512 * D * (Rn + G) + ( g * u2 * (es - ea) * 0.9 / T ) ) / ( D + g * (1 + 0.34 * u2) ) / 86400')
    assert np.allclose(pet, rad+wnd, equal_nan=True)
    rad = Variable(data=rad, name='petrad', units='kg/m^2/s', axes=dataset[refvar].axes)
    wnd = Variable(data=wnd, name='petwnd', units='kg/m^2/s', axes=dataset[refvar].axes)
  else:
    pet = evaluate('( 0.0352512 * D * (Rn + G) + ( g * u2 * (es - ea) * 0.9 / T ) ) / ( D + g * (1 + 0.34 * u2) ) / 86400')
  # N.B.: units have been converted to SI (mm/day -> 1/86400 kg/m^2/s, kPa -> 1000 Pa, and Celsius to K)
  atts = dict(name='pet', units='kg/m^2/s', long_name='PET (Penman-Monteith)')
  pet = Variable(data=pet, axes=dataset[refvar].axes, atts=atts)
  assert 'liqwatflx' not in dataset or pet.units == dataset['liqwatflx'].units, pet
  assert 'precip' not in dataset or pet.units == dataset['precip'].units, pet
  # return new variable(s)
  return (pet,rad,wnd) if lterms else pet


# compute potential evapo-transpiration; this method requires radiative flux data and mean temperature
def computePotEvapPT(dataset, alpha=1.26, lmeans=False, lrad=True, lgrdflx=True, lpmsl=True, lxarray=False, **kwargs):
  ''' function to compute potential evapotranspiration based on the Priestley-Taylor method (1972):
      Priestley & Taylor (1972, MWR): On the Assessment of Surface Heat Flux and Evaporation Using Large-Scale Parameters
      Note that different values for 'alpha' may be appropriate for different climates.
  '''
  if lxarray: raise NotImplementedError()
  # get net radiation at surface
  if 'netrad' in dataset: Rn = dataset['netrad'][:] # net radiation
  elif 'Rn' in dataset: Rn = dataset['Rn'][:] # alias
  else: Rn = computeNetRadiation(dataset, lrad=lrad, asVar=False) # try to compute
  # heat flux in and out of the ground
  if lgrdflx:
      if 'grdflx' in dataset: G = dataset['grdflx'][:] # heat release by the soil
      else: raise VariableError("Cannot determine soil heat flux for PET calculation.")
  else: G = 0
  # get psychrometric variables
  if 'ps' in dataset: p = dataset['ps'][:]
  else: p = computeSrfcPressure(dataset, lpmsl=lpmsl)
#   elif 'zs' in dataset: p = computeSrfcPressure(dataset, lpmsl=lpmsl)
#   else: 
#       raise VariableError("Cannot determine surface air pressure for PET calculation.")
  g = gamma(p) # psychrometric constant (pressure-dependent)
  # get temperature
  if lmeans and 'Tmean' in dataset: T = dataset['Tmean'][:]
  elif 'T2' in dataset: T = dataset['T2'][:]
  else: raise VariableError("Cannot determine 2m mean temperature for PET calculation.")
  # get slope of saturation vapor pressure w.r.t. temperature
  D = Delta(T)
  # determine reference variable (see what's available)
  for refvar in ('ps','T2','Tmean',):
      if refvar in dataset: break
  # compute potential evapotranspiration according to Priestley-Taylor method (1972)
  pet = evaluate('alpha * D * (Rn + G) / ( D + g ) / lw') # Eq. 12, Stannard, 1993, WRR
  # N.B.: units have been converted to SI (mm/day -> 1/86400 kg/m^2/s, kPa -> 1000 Pa, and Celsius to K)
  atts = dict(name='pet_pt', units='kg/m^2/s', long_name='PET (Priestley-Taylor)')
  pet = Variable(data=pet, axes=dataset[refvar].axes, atts=atts)
  assert 'liqwatflx' not in dataset or pet.units == dataset['liqwatflx'].units, pet
  assert 'precip' not in dataset or pet.units == dataset['precip'].units, pet
  # return new variable
  return pet


# function to extract a variable from suitable options
def _extractVariable(dataset, var=None, name_list=None, latts=True, lunits=False, lxarray=False):
  ''' extract a variable (value or array) from dataset based on a list of options '''
  units = None
  if var is None:
      variable = None
      # search for lat coordinate or field
      for varname in name_list:
          if varname in dataset:
              variable = dataset[varname]
              break
      if variable is None:
          var = None
          # search for attribute (constant value)
          if latts:
              for attname in name_list:
                  if lxarray:
                      if attname in dataset.attrs:
                          var = dataset.attrs[attname]
                          break
                  else: 
                      if attname in dataset.atts:
                          var = dataset.atts[attname]
                          break
          if var is None:
              raise DatasetError("No suitable variable {} or attribute found in dataset '{}'.".format(str(name_list),dataset.name))
      else:
          units = variable.units
          var = variable.data if lxarray else variable[:]
  elif isinstance(var,str):
      if var in dataset:
          variable = dataset[var] # select this field
      else:
          raise DatasetError("Variable '{}' not found in Dataset '{}'".format(var,dataset.name))
      units = variable.units
      var = variable.data if lxarray else variable[:]
  else:
      if isinstance(var,Variable):
          variable = var
          units = variable.units
          var = variable.data if lxarray else variable[:]
      else:
          pass # just use this value
  # return data array
  return (var, units) if lunits else var

# wrapper to extract elevation
def _inferElev(dataset, zs=None, name_list=None, latts=True, lunits=False):
    ''' wrapper to extract elevation based on naem_list'''
    if name_list is None: name_list = ('zs', 'elev', 'stn_zs')
    return _extractVariable(dataset, var=zs, name_list=name_list, latts=latts, lunits=lunits)   


# compute potential evapo-transpiration based on Tmin/Tmax only (and elevation)
def computePotEvapHog(dataset, lmeans=False, lq2=False, zs=None, lxarray=False, **kwargs):
  ''' function to compute potential evapotranspiration based on Hogg's simplified formula (1997):
      Hogg (1997, AgForMet): Temporal scaling of moisture and the forest-grassland boundary in western Canada
  '''
  if lxarray: import xarray as xr
  # get surface elevation variables
  zs, zs_units = _inferElev(dataset, zs=zs, latts=True, lunits=True) 
  assert zs_units is None or zs_units == 'm', zs
  T, Tmin, Tmax, t_units = _getTemperatures(dataset, lmeans=lmeans, lxarray=lxarray)
  # make sure T is in Celsius
  if t_units == 'K':
      assert lxarray or T.min() > 150, T
      TC = T - 273.15 # no need to convert min/max since only difference is used
  elif 'C' in t_units:
      assert lxarray or T.max() < 70, T
      TC = T
  else:
      raise VariableError("Cannot infer temperature units from unit string",t_units)
  # get saturation water vapor
  es = e_sat(Tmin,Tmax)
  # derive actual vapor pressure
  if lq2 and 'Q2' in dataset: ea = dataset['Q2'][:]
  elif lq2 and 'q2' in dataset and 'ps' in dataset:
      q2 = dataset['q2'][:];  p = dataset['ps'][:]
      ea =  evaluate('q2 * p * 28.96 / 18.02')
  else: 
      # estimate actual vapor pressure from Tmin
      ea = e_sat(Tmin - 2.5) # 2.5K below Tmin - about 85% at night
  # compute potential evapotranspiration based on temperature (Hogg 1997, Eq. 4)
  D = evaluate('(es - ea) * exp( zs/9300 ) / 1000.')
  pet = np.where(np.isfinite(TC),0,np.NaN)
  pet = np.where(TC>-5,evaluate('(6.2*TC + 31) * D'),pet)
  pet = np.where(TC>10,evaluate('93 * D'),pet)
  # convert from mm/month to kg/m^2/s
  pet /= days_per_month.mean()*86400
  # N.B.: units have been converted to SI (mm/day -> 1/86400 kg/m^2/s, kPa -> 1000 Pa, and Celsius to K)
  # create DataArray/Variable
  refvar = dataset['Tmax']
  atts = dict(name='pet_hog', units='kg/m^2/s', long_name='PET (Hogg 1997)')
  if lxarray:
      var = xr.DataArray(coords=refvar.coords, data=pet, name=atts['name'], attrs=atts)
  else: 
      if refvar.masked:
          pet = np.ma.masked_array(pet, mask=refvar.data_array.mask)
      var = Variable(data=pet, axes=refvar.axes, atts=atts)
  assert 'liqwatflx' not in dataset or pet.units == dataset['liqwatflx'].units, pet
  assert 'precip' not in dataset or pet.units == dataset['precip'].units, pet
  # return new variable
  return var


## some helper functions to compute solar irradiance (ToA) and daylight hours

# function to extract a suitable latitude value/array from a dataset
def _inferLat(dataset, lat=None, name_list=None, latts=True, ldeg=True, lunits=True, lxarray=False):
  ''' extract a latitude value or array from dataset and expand to reference dimensions '''
  if name_list is None: name_list = ('lat2D', 'xlat', 'lat', 'latitude', 'stn_lat')
  lat, units = _extractVariable(dataset, var=lat, name_list=name_list, latts=latts, lunits=True, lxarray=lxarray)
  if units:
      if ldeg and 'deg' not in units and 'rad' in units: 
          lat = np.rad2deg(lat); units = 'deg'
      elif not ldeg and 'deg' in units and 'rad' not in units: 
          lat = np.deg2rad(lat); units = 'rad' 
  # return data array
  return (lat, units) if lunits else lat

# calendar day of the middle of the month
def day_of_year(month, l365=False, time_offset=0):
    ''' calculate the calendar day (day of year) of the middle of the month; values in time series will 
        be used as zero- or one-based indices '''
    if np.isscalar(month): month = np.asarray([month])
    elif not isinstance(month,np.ndarray): month = np.asarray(month)
    # compute day of the middle of the month
    dpm = days_per_month_365 if l365 else days_per_month
    mid_day_month = np.cumsum(np.concatenate(([0],dpm[:-1]))) + dpm/2.
    # create index array
    month_idx = ( month if time_offset == 0 else month+time_offset ) % 12
    month_idx = month_idx.astype(np.int)
    # select the mid-day of the year for each month
    J = mid_day_month[month_idx]
    # return day of the year of the middle of the month
    return J

# calculate solar declination angle from Julian day
def solar_declination(J, ldeg=True):
    ''' calculate the solar declination based on the day of the year; this is the CBM model from 
        Forsythe et al. 1995 (Ecological Modelling), which is supposed to be more accurate and 
        accounts for ellipticity of the orbit '''
    th = 0.2163108 + 2*np.arctan(0.9671396 * np.tan(0.00860 * (J - 186) ) ) # solar orbital angle
    dec = np.arcsin( 0.39795 * np.cos(th) ) # declination
    if ldeg: dec = np.rad2deg(dec)
    return dec

# calculate length of day from declination and latitude
def fraction_of_daylight(dec, lat, p='center', ldeg=True):
    ''' calculate length of day from declination and latitude, with a daylight definition parameter; formulas have been adapted from 
        Forsythe et al, (1995, Ecological Modeling 80): A model comparison for daylength as a function of latitude and day of year'''
    if isinstance(p,str):
        if p == 'center': p = 0
        elif p == 'top': p = 0.26667
        elif p == 'refrac': p = 0.8333
        elif p == 'nautical': p = 12
        elif p == 'civil': p = 18
        elif p == 'astro': p = 6
        else:
            raise ValueError(p)
        # values in degree from Table 1 in Forsythe et al. 1995
        if not ldeg: p = np.deg2rad(p)
    if ldeg:
        dec = np.deg2rad(dec); lat = np.deg2rad(lat); p = np.deg2rad(p)
    tmp = ( np.sin(p) + np.sin(lat)*np.sin(dec) ) / ( np.cos(lat)*np.cos(dec) )
    np.clip(tmp, a_min=-1, a_max=1, out=tmp) # in place
    frac = 1 - np.arccos(tmp)/pi
    # return fraction of daylight ( 1 == 24 hours )
    return frac

def _prepCoords(time, lat, lmonth=True, l365=False, time_offset=0, ldeg=True):
    ''' compute calendar day and solar declination from timeseries, expand and convert latitude; 
        declination and latitude are converted to radians '''
    if np.issubdtype(time.dtype,np.datetime64):
        # get day of year from datetime64 array
        J = 1 + ( ( time.astype('datetime64[D]') - time.astype('datetime64[Y]') ) / np.timedelta64(1,'D') )
        if lmonth: J += 15 # move from beginning to middle of the month
    else:
        if lmonth:
            # compute day of year
            J = day_of_year(time, l365=l365, time_offset=time_offset)        
        else: J = time # assume time is calendar day
    assert np.all(J < 367), J.max()
    # compute solar declination
    D = solar_declination(J, ldeg=False)
    if not np.isscalar(lat) and not np.isscalar(D):
        lat = np.asarray(lat); D = np.asarray(D)
        assert D.ndim == 1, D.shape
        D = D.reshape(D.shape+(1,)*lat.ndim) # add singleton spatial dimensions
        lat = lat.reshape((1,)+lat.shape) # add singleton time dimension        
    if ldeg: lat = np.deg2rad(lat) # ldeg refers to input, not output!
    # return calendar day, solar declination and latitude (declination and latitude in radians)
    return J, D, lat # D & lat always in radians !!!


# compute daylight hours from month and latitude
def monthlyDaylight(time, lat, lmonth=True, l365=False, time_offset=0, ldeg=True, p='center'):
    ''' compute fraction of daylight based on month (middle of the month), latitude and a 'horizon definition' p '''
    J, D, lat = _prepCoords(time, lat, lmonth=lmonth, l365=l365, time_offset=time_offset, ldeg=ldeg); del J
    # compute hours of daylight
    frac = fraction_of_daylight(D, lat, p=p, ldeg=False)
    # return fraction of daylight ( 1 == 24 hours )
    return frac


# compute top-of-atmosphere (extra-terrestrial) radiation
def toa_rad(time, lat, lmonth=True, l365=False, time_offset=0, ldeg=True):
    ''' solar radiation at the top of the atmosphere; from Allen et al. (1998), FAO, Eq. 21 '''
    # N.B.: ldeg refers to lat input
    J, D, lat = _prepCoords(time, lat, lmonth=lmonth, l365=l365, time_offset=time_offset, ldeg=ldeg) 
    # compute inverse relative distance to sun (Eq. 23, Allen et al.)
    rr = evaluate('1 + 0.033*cos( pi*2*J/365.2425 )').reshape((len(J),)+(1,)*(lat.ndim-1))
    # compute sunset hour angle (Eq. 26, Allen et al.)
    ws = evaluate('arccos( -1*tan(lat)*tan(D) )')
    # compute top-of-atmosphere solar radiation (extra-terrestrial)
    Ra = evaluate('1366.67 * rr * ( ws*sin(lat)*sin(D) + cos(lat)*cos(D)*sin(ws) ) / pi')
    # return solar radiation at ToA
    return Ra


# compute potential evapotranspiration based on Hargreaves method; requires only Tmin/Tmax and ToA radiation (i.e. latitude and date)
def computePotEvapHar(dataset, lat=None, lmeans=False, l365=None, time_offset=0, lAllen=False, lxarray=False, **kwargs):
    ''' function to compute potetnial evapotranspiration following the Hargreaves method;
        (Hargreaves & Allen, 2003, Eq. 8) '''
    if lxarray: import xarray as xr
    T, Tmin, Tmax, t_units = _getTemperatures(dataset, lmeans=lmeans, lxarray=lxarray)
    # make sure T is in Celsius
    if t_units == 'K':
        assert lxarray or T.min() > 150, T
        TC = T - 273.15 # no need to convert min/max since only difference is used
    elif 'C' in t_units:
        assert lxarray or TC.max() < 70, T
        TC = T
    else:
        raise VariableError("Cannot infer temperature units from unit string",t_units)
    # infer latitude
    lat = _inferLat(dataset, lat=lat, ldeg=True, lunits=False, lxarray=lxarray)
    # compute top-of-atmosphere solar radiation
    time = dataset['time'].data if lxarray else dataset.time[:]
    lmonth = not lxarray and ('month' in time.units.lower()) # whether time axis is datetime or month index
    Ra = toa_rad(time, lat=lat, lmonth=lmonth, ldeg=True, l365=l365, time_offset=time_offset)    
    # compute PET (need to convert Ra from J/m^/s to kg/m^2/s = mm/s and Kelvin to Celsius)    
    if lAllen:
        pet = evaluate('(0.0029/lw) * Ra * (TC + 20.) * (Tmax - Tmin)**0.4') # seems high-biased
    else:
        pet = evaluate('(0.0023/lw) * Ra * (TC + 17.8) * (Tmax - Tmin)**0.5')
    # create a DataArray/Variable instance
    refvar = dataset['Tmax']
    atts = dict(name='pet_har', units='kg/m^2/s', long_name='PET (Hargreaves)')
    if lxarray:
        if pet.size == 0: pet = None
        var = xr.DataArray(coords=refvar.coords, data=pet, name=atts['name'], attrs=atts)
    else: 
        if refvar.masked:
            pet = np.ma.masked_array(pet, mask=refvar.data_array.mask)
        var = Variable(data=pet, axes=refvar.axes, atts=atts)
    assert 'liqwatflx' not in dataset or pet.units == dataset['liqwatflx'].units, pet
    assert 'precip' not in dataset or pet.units == dataset['precip'].units, pet
    # return new variable
    return var


# compute heat index for Thornthwaite PET formulation
def heatIndex(T2, lKelvin=True, lkeepDims=True):
    ''' formula to compute heat index (Thornthwaite 1948) from monthly normal temperatures '''
    if T2.size > 0 and T2.shape[0] != 12: 
        raise ValueError(T2.shape)    
    t = T2[:].copy()
    if lKelvin: t -= 273.15 # convert temperatur
    np.clip(t, a_min=0, a_max=None, out=t) # in-place
    t /= 5. # operate in-place - if the arrays are small, this is trivial anyway
    t **= 1.514
    # compute sum along time axis
    I = np.sum(t, axis=0, keepdims=lkeepDims)
    # return cumulative heat index
    return I

# compute potential evapo-transpiration following Thornthwaite method; only requires T2 (monthly/daily and climatological)
def computePotEvapTh(dataset, climT2=None, lat=None, l365=None, time_offset=0, p='center', lxarray=False, **kwargs):
    ''' function to compute potential evapotranspiration according to Thornthwaite method
        (Thornthwaite, 1948, Appendix 1 or https://en.wikipedia.org/wiki/Potential_evaporation) '''
    if lxarray:
        import xarray as xr
        if isinstance(dataset, xr.Dataset): T2 = dataset['T2']
        else: raise TypeError(dataset) # has to be a Dataset, so we can also pass climT2
        time = T2.coords['time']
        t_units = time.long_name
        time = time.data
        assert T2.dims[0] == 'time', T2
        t = T2.data
    else:
        # check prerequisites
        if isinstance(dataset, Dataset): T2 = dataset['T2'].load()
        elif isinstance(dataset, Variable): T2 = dataset.load()
        else: raise TypeError(dataset)
        time = T2.getAxis('time')
        t_units = time.units
        time = time[:]
        assert T2.axisIndex('time') == 0, T2
        t = T2[:]
    # convert units
    T2units = T2.units # same for xarray and geopy
    if T2units.upper().startswith('K'):
        t = t - 273.15
    elif T2units.upper().startswith('C'):
        t = t.copy() # will clip in place later
    else:
        raise VariableError("Cannot infer temperature units from unit string",T2units)
    # check climatology
    if lxarray:
        if not isinstance(climT2,str):
            raise TypeError("If xarray is used, monthly normals have to be provided as a variable in the main dataet.")
        climT2 = dataset[climT2]
        lKelvin = climT2.units.upper().startswith('K')
        climt = climT2.data
    else:
        if climT2 is None:
            climT2 = T2.climMean()
        elif isinstance(climT2,str):
            if isinstance(dataset, Dataset):
                climT2 = dataset[climT2]
            else:
                raise TypeError("Providing climT2 as a string argument (Variable name) requires dataset to be an actual Dataset.")
        if isinstance(climT2,Variable):
            lKelvin = ( climT2.units == 'K' )
            climt = climT2.load()[:]
            if lKelvin: assert climt.min() > 150, climt
            else: assert climt.max() < 70, climt
        else:
            climt = climT2
            lKelvin = ( climt.min() > 150 )
    # compute heat index
    I = heatIndex(climt, lKelvin=lKelvin, lkeepDims=True)
    a = evaluate('6.75e-7*I**3 - 7.71e-5*I**2 + 1.792e-2*I + 0.49239')
    # infer latitude
    lat = _inferLat(dataset, lat=lat, ldeg=True, latts=True, lunits=False, lxarray=lxarray)
    # compute PET 
    np.clip(t, a_min=0, a_max=None, out=t)
    pet = evaluate('(16./30.) * ( 10. * t/I )**a') # in mm/day for 12 hours of daylight
    # compute daylight hours
    lmonth = not lxarray and 'month' in t_units.lower()
    dlf = monthlyDaylight(time, lat, p=p, l365=l365, time_offset=time_offset, ldeg=True, lmonth=lmonth)
    if dlf.ndim < pet.ndim:
        nd = dlf.ndim
        if dlf.shape == pet.shape[:nd]:
            dlf = dlf.reshape(dlf.shape+(1,)*(pet.ndim-nd)) # usually the longitude axis will be expanded
        else:
            raise NotImplementedError((dlf.shape,pet.shape))
    elif pet.size == 0:
        dlf = dlf.reshape(pet.shape) # to facilitate xarray output probing
    else:
        assert dlf.shape == pet.shape, (dlf.shape,pet.shape)
    pet *= 2*dlf/86400 # here 'unity' should corresponds to 12h, not 24h
    # N.B.: we also need to convert from per day to per second!
    # create a DataArray/Variable instance
    atts = dict(name='pet_th', units='kg/m^2/s', long_name='PET (Thornthwaite)')
    if lxarray:
        if pet.size == 0: pet = None
        var = xr.DataArray(coords=T2.coords, data=pet, name=atts['name'], attrs=atts)
    else: 
        if T2.masked:
            pet = np.ma.masked_array(pet, mask=T2.data_array.mask)
        var = Variable(data=pet, axes=T2.axes, atts=atts)
    assert 'liqwatflx' not in dataset or pet.units == dataset['liqwatflx'].units, pet
    assert 'precip' not in dataset or pet.units == dataset['precip'].units, pet
    # return new variable
    return var

# recompute total precip from solid and liquid precip
def computeTotalPrecip(dataset):
  ''' function to recompute total precip from solid and liquid precip '''
  # check prerequisites
  for pv in ('liqprec','solprec'): 
    if pv not in dataset: raise VariableError("Prerequisite '{:s}' for net water flux not found.".format(pv))
    else: dataset[pv].load() # load data for computation
  # recompute total precip (returns a Variable instance)
  var = dataset['liqprec'] + dataset['solprec']
  var.name = 'precip' # give correct name (units should be correct)
  assert var.units == dataset['liqprec'].units, var
  # return new variable
  return var

# compute surface water flux
def computeWaterFlux(dataset):
  ''' function to compute the net water flux at the surface '''
  # check prerequisites
  if 'liqprec' in dataset: # this is the preferred computation
    for pv in ('evap','snwmlt'): 
      if pv not in dataset: raise VariableError("Prerequisite '{:s}' for net water flux not found.".format(pv))
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['liqprec'] + dataset['snwmlt'] - dataset['evap']
  elif 'solprec' in dataset: # alternative computation, mainly for CESM
    for pv in ('precip','evap','snwmlt'): 
      if pv not in dataset: raise VariableError("Prerequisite '{:s}' for net water flux not found.".format(pv))
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['precip'] - dataset['solprec'] + dataset['snwmlt'] - dataset['evap']
  else: 
    raise VariableError("No liquid or solid precip found to compute net water flux.")
  var.name = 'waterflx' # give correct name (units should be correct)
  assert var.units == dataset['evap'].units, var
  # return new variable
  return var

# compute downward/liquid component of surface water flux
def computeLiquidWaterFlux(dataset, shift=0):
  ''' function to compute the downward/liquid component of water flux at the surface '''
  if 'snwmlt' not in dataset: raise VariableError("Prerequisite 'snwmlt' for liquid water flux not found.")  
  snwmlt = dataset['snwmlt'].load() # load data for computation
  if shift: snwmlt = shiftVariable(snwmlt, shift=shift)
  # check prerequisites
  if 'liqprec' in dataset: # this is the preferred computation
    if 'liqprec' not in dataset: raise VariableError("Prerequisite 'liqprec' for liquid water flux not found.")
    else: dataset['liqprec'].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['liqprec'] + snwmlt
  elif 'solprec' in dataset: # alternative computation, mainly for CESM
    for pv in ('precip','solprec'): 
      if pv not in dataset: raise VariableError("Prerequisite '{:s}' for net water flux not found.".format(pv))
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['precip'] - dataset['solprec'] + snwmlt
  else: 
    raise VariableError("No liquid or solid precip found to compute net water flux.")
  var.name = 'liqwatflx' # give correct name (units should be correct)
  var.atts.long_name = 'Liquid Water Flux'
  if shift: 
      var.name += snwmlt.atts.bc_shift_str
      var.atts.long_name += ' (Snowmelt shifted {})'
      bc_note = snwmlt.atts.bc_note
      bc_note = bc_note[bc_note.find('shifted by')]
      if bc_note in var.atts and len(var.atts.bc_note): var.atts.bc_note += '; ' + bc_note
      else: var.atts.bc_note = bc_note
      assert var.name[-2] in ('-','+') or var.name[-3] in ('-','+'), var
  assert var.units == snwmlt.units, var  
  # return new variable
  return var


# generic function to shift a variable foreward or backward in time
def shiftVariable(variable, shift=0, time_axis='time'):
  ''' function to shift a single variable foreward or backward in time (roll over) '''
  assert -10 < shift < 10, shift
  variable.load()
  # process shift
  tax = variable.axisIndex(time_axis)
  assert variable.shape[tax]%12 == 0, var # this is really only applicable to periodic monthly data...
  assert 'month' in variable.axes[tax].units.lower(), variable.axes[tax]
  if shift == int(shift):
      data = np.roll(variable.data_array, shift=shift, axis=tax) 
      shift_str = '{:+2d}'.format(int(shift))
  else:
      # use weighted average of upper and lower integer shift
      data_floor = np.roll(variable.data_array, shift=int(np.floor(shift)), axis=tax) 
      data_ceil  = np.roll(variable.data_array, shift=int(np.ceil(shift)), axis=tax)
      data = (np.ceil(shift)-shift)*data_floor + (shift-np.floor(shift))*data_ceil
      shift_str = '{:+03d}'.format(int(np.round(shift, decimals=1)*10))
  # create new variable object
  newvar = variable.copy(deepcopy=False,)
  newvar.load(data)
  newvar.name += shift_str
  newvar.atts.bc_shift = shift
  newvar.atts.bc_shift_str = shift_str
  taxis = variable.axes[tax]
  bc_note = "shifted by {} {} along axis '{}'".format(shift,taxis.units,taxis.name)
  if bc_note in newvar.atts and len(newvar.atts.bc_note): newvar.atts.bc_note += '; ' + bc_note
  else: newvar.atts.bc_note = bc_note
  # return shifted variable
  return newvar


if __name__ == '__main__':
    

#   from projects.WesternCanada.WRF_experiments import Exp, WRF_exps, ensembles
  from projects.GreatLakes import loadWRF
  # N.B.: importing Exp through WRF_experiments is necessary, otherwise some isinstance() calls fail
    
  mode = 'test_PenmanMonteith'
    
  # load averaged climatology file
  if mode == 'test_PenmanMonteith':
    
    print('')
    load_list = ['ps','u10','v10','Q2','Tmin','Tmax','T2','TSmin','TSmax','Tmean','U10', # wind
                 'grdflx','A','SWD','e','GLW','SWDNB','SWUPB','LWDNB','LWUPB'] # radiation
        
    dataset = loadWRF(experiment='g-ensemble-2100', domains=1, grid='grw2', filetypes=['hydro','srfc','xtrm','lsm','rad'], 
                      period=15, varlist=load_list)
    print(dataset)
    print('')
    var = computePotEvapPM(dataset, lterms=False, lmeans=True)
    print(var)
    print('')
    print('PET using MEAN variables from WRF xtrm files:')
    print((var.min(),var.mean(),var.std(),var.max()))
    print('PET using averages from WRF srfc files:')
    var = computePotEvapPM(dataset, lterms=False, lmeans=False)
    print((var.min(),var.mean(),var.std(),var.max()))
    import numpy as np
    print(('Ratio of Wind Terms:', np.mean( dataset['U10'][:] / np.sqrt(5*dataset['u10'][:]**2 + 10*dataset['v10'][:]**2) ) ))
    print(('Difference of Temperature Terms:', np.mean( dataset['T2'][:] - dataset['Tmean'][:]) ))
