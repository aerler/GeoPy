'''
Created on May 5, 2016

A collection of functions to compute derived variables, primarily for WRF but also in general.

@author: Andre R. Erler, GPL v3
'''

# external imports
from warnings import warn
from numexpr import evaluate, set_num_threads, set_vml_num_threads
# numexpr parallelisation: don't parallelize at this point!
set_num_threads(1); set_vml_num_threads(1)
# internal imports
from geodata.base import Variable, VariableError
from utils.constants import sig

## helper functions

# net radiation balance 
def radiation(A, SW, LW, e, Ts, TSmax=None):
  ''' net radiation  [W/m^2] at the surface: downwelling long and short wave minus upwelling terrestrial radiation '''
  if TSmax is None:
    # using average skin temperature for terrestrial long wave emission
    warn('Using average skin temperature; diurnal min/max skin temperature is preferable due to strong nonlinearity.')
    return evaluate('( ( 1 - A ) * SW ) + ( LW * e ) - ( e * sig * Ts**4 )')
  else:
    # using min/max skin temperature to account for nonlinearity
    return evaluate('( ( 1 - A ) * SW ) + ( LW * e ) - ( e * sig * ( Ts**4 + TSmax**4 ) / 2 )')

# 2m wind speed [m/s]
def wind(u, z=10):
  ''' approximate 2m wind speed from wind speed at different height (z [m]; default 10m)
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#wind%20profile%20relationship)
  '''
  return evaluate('( u * 4.87 ) / log( 67.8 * z - 5.42 )')

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
    warn('Using average 2m temperature; diurnal min/max 2m temperature is preferable due to strong nonlinearity.')
    return evaluate('610.8 * exp( 17.27 * (T - 273.15) / (T - 35.85) )')
  else:
    # use average of saturation pressure from Tmin and Tmax (because of nonlinearity)
    return evaluate('305.4 * ( exp( 17.27 * (T - 273.15) / (T - 35.85) ) + exp( 17.625 * (Tmax - 273.15) / (Tmax - 35.85) ) )')

## functions to compute relevant variables (from a dataset)

# compute net radiation (for PET)
def computeNetRadiation(dataset, asVar=True):
  ''' function to compute net radiation at surface for Penman-Monteith equation
      (http://www.fao.org/docrep/x0490e/x0490e06.htm#formulation%20of%20the%20penman%20monteith%20equation)
  '''
  if 'A' in dataset and 'SWD' in dataset and 'GLW' in dataset and 'e' in dataset:
    if 'TSmin' in dataset and 'TSmax' in dataset: Ts = dataset['TSmin'][:]; TSmax = dataset['TSmax'][:]
    elif 'TSmean' in dataset: Ts = dataset['TSmean'][:]; TSmax = None
    elif 'Ts' in dataset: Ts = dataset['Ts'][:]; TSmax = None
    else: raise VariableError, "Either 'Ts' or 'TSmean' are required to compute net radiation for PET calculation."
    data = radiation(dataset['A'][:],dataset['SWD'][:],dataset['GLW'][:],dataset['e'][:],Ts,TSmax) # downward total net radiation
  else: raise VariableError, "Cannot determine net radiation for PET calculation."
  # cast as Variable
  if asVar:
    var = Variable(data=data, name='netrad', units='kg/m^2/s', axes=dataset['SWD'].axes)
    assert var.units == dataset['waterflx'].units, var
  else: var = data
  # return new variable
  return var

# compute potential evapo-transpiration
def computePotEvapPM(dataset):
  ''' function to compute potential evapotranspiration (according to Penman-Monteith method:
      https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation,
      http://www.fao.org/docrep/x0490e/x0490e06.htm#formulation%20of%20the%20penman%20monteith%20equation)
  '''
  # get net radiation at surface
  if 'netrad' in dataset: Rn = dataset['netrad'][:] # net radiation
  if 'Rn' in dataset: Rn = dataset['Rn'][:] # alias
  else: Rn = computeNetRadiation(dataset, asVar=False) # try to compute
  # heat flux in and out of the ground
  if 'grdflx' in dataset: G = dataset['grdflx'][:] # heat release by the soil
  else: raise VariableError, "Cannot determine soil heat flux for PET calculation."
  # get wind speed
  if 'U2' in dataset: u2 = dataset['U2'][:]
  elif 'U10' in dataset: u2 = wind(dataset['U10'][:], z=10)
  else: raise VariableError, "Cannot determine 2m wind speed for PET calculation."
  # get psychrometric variables
  if 'ps' in dataset: p = dataset['ps'][:]
  else: raise VariableError, "Cannot determine surface air pressure for PET calculation."
  g = gamma(p) # psychrometric constant (pressure-dependent)
  if 'Q2' in dataset: ea = dataset['Q2'][:]
  elif 'q2' in dataset: ea = dataset['q2'][:] * dataset['ps'][:] * 28.96 / 18.02
  else: raise VariableError, "Cannot determine 2m water vapor pressure for PET calculation."
  # get temperature
  if 'Tmean' in dataset: T = dataset['Tmean'][:]
  elif 'T2' in dataset: T = dataset['T2'][:]
  else: raise VariableError, "Cannot determine 2m mean temperature for PET calculation."
  # get saturation water vapor
  if 'Tmin' in dataset and 'Tmax' in dataset: es = e_sat(dataset['Tmin'][:],dataset['Tmax'][:])
  # else: Es = e_sat(T) # backup, but not very accurate
  else: raise VariableError, "'Tmin' and 'Tmax' are required to compute saturation water vapor pressure for PET calculation."
  D = Delta(T) # slope of saturation vapor pressure w.r.t. temperature
  # compute potential evapotranspiration according to Penman-Monteith method 
  # (http://www.fao.org/docrep/x0490e/x0490e06.htm#fao%20penman%20monteith%20equation)
  data = evaluate('( 0.0352512 * D * (Rn + G) + ( g * u2 * (es - ea) * 0.9 / T ) ) / ( D + g * (1 + 0.34 * u2) ) / 86400')
  # N.B.: units have been converted to SI (mm/day -> 1/86400 kg/m^2/s, kPa -> 1000 Pa, and Celsius to K)
  var = Variable(data=data, name='pet', units='kg/m^2/s', axes=dataset['ps'].axes)
  assert var.units == dataset['waterflx'].units, var
  # return new variable
  return var

# compute potential evapo-transpiration
def computePotEvapTh(dataset):
  ''' function to compute potential evapotranspiration (according to Thornthwaite method:
      https://en.wikipedia.org/wiki/Potential_evaporation) '''
  raise NotImplementedError
  # check prerequisites
  for pv in (): 
    if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for potential evapo-transpiration not found.".format(pv)
    else: dataset[pv].load() # load data for computation
  # compute waterflux (returns a Variable instance)
  var = dataset['']
  var.name = 'pet' # give correct name (units should be correct)
  assert var.units == dataset[''].units, var
  # return new variable
  return var

# compute surface water flux
def computeWaterFlux(dataset):
  ''' function to compute the net water flux at the surface '''
  # check prerequisites
  if 'liqprec' in dataset: # this is the preferred computation
    for pv in ('evap','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for net water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['liqprec'] + dataset['snwmlt'] - dataset['evap']
  elif 'solprec' in dataset: # alternative computation, mainly for CESM
    for pv in ('precip','evap','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for net water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['precip'] - dataset['solprec'] + dataset['snwmlt'] - dataset['evap']
  else: 
    raise VariableError, "No liquid or solid precip found to compute net water flux."
  var.name = 'waterflx' # give correct name (units should be correct)
  assert var.units == dataset['evap'].units, var
  # return new variable
  return var

# compute downward/liquid component of surface water flux
def computeLiquidWaterFlux(dataset):
  ''' function to compute the downward/liquid component of water flux at the surface '''
  # check prerequisites
  if 'liqprec' in dataset: # this is the preferred computation
    for pv in ('liqprec','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for liquid water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['liqprec'] + dataset['snwmlt']
  elif 'solprec' in dataset: # alternative computation, mainly for CESM
    for pv in ('precip','snwmlt'): 
      if pv not in dataset: raise VariableError, "Prerequisite '{:s}' for net water flux not found.".format(pv)
      else: dataset[pv].load() # load data for computation
    # compute waterflux (returns a Variable instance)
    var = dataset['precip'] - dataset['solprec'] + dataset['snwmlt']
  else: 
    raise VariableError, "No liquid or solid precip found to compute net water flux.".format(pv)
  var.name = 'liqwatflx' # give correct name (units should be correct)
  assert var.units == dataset['snwmlt'].units, var
  # return new variable
  return var


if __name__ == '__main__':
    

#   from projects.WesternCanada.WRF_experiments import Exp, WRF_exps, ensembles
  from projects.GreatLakes import loadWRF
  # N.B.: importing Exp through WRF_experiments is necessary, otherwise some isinstance() calls fail
    
  mode = 'test_PenmanMonteith'
    
  # load averaged climatology file
  if mode == 'test_PenmanMonteith':
    
    print('')
    varlist = ['hfx','A','SWD','e','GLW','ps','U10','Q2','Tmin','Tmax','Tmean','TSmin','TSmax','waterflx','liqwatflx']
    dataset = loadWRF(experiment='g-ctrl', domains=2, grid='grw2', filetypes=['hydro','srfc','xtrm','lsm'], 
                      period=15, varlist=varlist)
    print(dataset)
    print('')
    var = computePotEvapPM(dataset)
    print(var)
    print(var.min(),var.mean(),var.std(),var.max())
