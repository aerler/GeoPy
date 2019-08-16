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
from geodata.base import Variable, VariableError
from utils.constants import sig # used in calculation for clack body radiation

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
  ''' approximate 2m wind speed from wind speed at different height (z [m]; default 10m)
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#wind%20profile%20relationship)
  '''
  if v is not None: u = evaluate('sqrt( 5*u**2 + 10*v**2 )') # estimate wind speed from u and v components
  # N.B.: the scale factors (5&10) are necessary, because absolute wind speeds are about 2.5 times higher than
  #       the mean of the compnents. This is because opposing directions average to zero.
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
    #warn('Using average 2m temperature; diurnal min/max 2m temperature is preferable due to strong nonlinearity.')
    return evaluate('610.8 * exp( 17.27 * (T - 273.15) / (T - 35.85) )')
  else:
    # use average of saturation pressure from Tmin and Tmax (because of nonlinearity)
    return evaluate('305.4 * ( exp( 17.27 * (T - 273.15) / (T - 35.85) ) + exp( 17.625 * (Tmax - 273.15) / (Tmax - 35.85) ) )')

## functions to compute relevant variables (from a dataset)

# compute net radiation (for PET)
def computeNetRadiation(dataset, asVar=True, lA=True, lrad=True, name='netrad'):
  ''' function to compute net radiation at surface for Penman-Monteith equation
      (http://www.fao.org/docrep/x0490e/x0490e07.htm#radiation)
  '''
  if lrad and 'SWDNB' in dataset and 'LWDNB' in dataset and 'SWUPB' in dataset and 'LWUPB' in dataset:
    data = radiation(dataset['SWDNB'][:],dataset['LWDNB'][:],dataset['SWUPB'][:],dataset['LWUPB'][:]) # downward total net radiation
  else:
    if 'e' not in dataset: raise VariableError("Emissivity is not available for radiation calculation.")
    if not lA: A = 0.23 # reference Albedo for grass
    elif lA and 'A' in dataset: A = dataset['A'][:]
    else: raise VariableError("Actual Albedo is not available for radiation calculation.")
    if 'TSmin' in dataset and 'TSmax' in dataset: Ts = dataset['TSmin'][:]; TSmax = dataset['TSmax'][:]
    elif 'TSmean' in dataset: Ts = dataset['TSmean'][:]; TSmax = None
    elif 'Ts' in dataset: Ts = dataset['Ts'][:]; TSmax = None
    else: raise VariableError("Either 'Ts' or 'TSmean' are required to compute net radiation for PET calculation.")
    if 'LWDNB' in dataset: GLW = dataset['LWDNB'][:]
    elif 'GLW' in dataset: GLW = dataset['GLW'][:]
    else: raise VariableError("Downwelling LW radiation is not available for radiation calculation.")
    if 'SWDNB' in dataset: SWD = dataset['SWDNB'][:]
    elif 'SWD' in dataset: SWD = dataset['SWD'][:]
    else: raise VariableError("Downwelling LW radiation is not available for radiation calculation.")
    data = radiation_black(A,SWD,GLW,dataset['e'][:],Ts,TSmax) # downward total net radiation
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

# compute potential evapo-transpiration
def computePotEvapPM(dataset, lterms=True, lmeans=False, lrad=True):
  ''' function to compute potential evapotranspiration (according to Penman-Monteith method:
      https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation,
      http://www.fao.org/docrep/x0490e/x0490e06.htm#formulation%20of%20the%20penman%20monteith%20equation)
  '''
  # get net radiation at surface
  if 'netrad' in dataset: Rn = dataset['netrad'][:] # net radiation
  if 'Rn' in dataset: Rn = dataset['Rn'][:] # alias
  else: Rn = computeNetRadiation(dataset, lrad=lrad, asVar=False) # try to compute
  # heat flux in and out of the ground
  if 'grdflx' in dataset: G = dataset['grdflx'][:] # heat release by the soil
  else: raise VariableError("Cannot determine soil heat flux for PET calculation.")
  # get wind speed
  if 'U2' in dataset: u2 = dataset['U2'][:]
  elif lmeans and 'U10' in dataset: u2 = wind(dataset['U10'][:], z=10)
  elif 'u10' in dataset and 'v10' in dataset: u2 = wind(u=dataset['u10'][:],v=dataset['v10'][:], z=10)
  else: raise VariableError("Cannot determine 2m wind speed for PET calculation.")
  # get psychrometric variables
  if 'ps' in dataset: p = dataset['ps'][:]
  else: raise VariableError("Cannot determine surface air pressure for PET calculation.")
  g = gamma(p) # psychrometric constant (pressure-dependent)
  if 'Q2' in dataset: ea = dataset['Q2'][:]
  elif 'q2' in dataset: ea = dataset['q2'][:] * dataset['ps'][:] * 28.96 / 18.02
  else: raise VariableError("Cannot determine 2m water vapor pressure for PET calculation.")
  # get temperature
  if lmeans and 'Tmean' in dataset: T = dataset['Tmean'][:]
  elif 'T2' in dataset: T = dataset['T2'][:]
  else: raise VariableError("Cannot determine 2m mean temperature for PET calculation.")
  # get saturation water vapor
  if 'Tmin' in dataset and 'Tmax' in dataset: es = e_sat(dataset['Tmin'][:],dataset['Tmax'][:])
  # else: Es = e_sat(T) # backup, but not very accurate
  else: raise VariableError("'Tmin' and 'Tmax' are required to compute saturation water vapor pressure for PET calculation.")
  D = Delta(T) # slope of saturation vapor pressure w.r.t. temperature
  # compute potential evapotranspiration according to Penman-Monteith method 
  # (http://www.fao.org/docrep/x0490e/x0490e06.htm#fao%20penman%20monteith%20equation)
  if lterms:
    Dgu = evaluate('( D + g * (1 + 0.34 * u2) ) * 86400') # common denominator
    rad = evaluate('0.0352512 * D * (Rn + G) / Dgu') # radiation term
    wnd = evaluate('g * u2 * (es - ea) * 0.9 / T / Dgu') # wind term (vapor deficit)
    pet = evaluate('( 0.0352512 * D * (Rn + G) + ( g * u2 * (es - ea) * 0.9 / T ) ) / ( D + g * (1 + 0.34 * u2) ) / 86400')
    import numpy as np
    assert np.allclose(pet, rad+wnd, equal_nan=True)
    rad = Variable(data=rad, name='petrad', units='kg/m^2/s', axes=dataset['ps'].axes)
    wnd = Variable(data=wnd, name='petwnd', units='kg/m^2/s', axes=dataset['ps'].axes)
  else:
    pet = evaluate('( 0.0352512 * D * (Rn + G) + ( g * u2 * (es - ea) * 0.9 / T ) ) / ( D + g * (1 + 0.34 * u2) ) / 86400')
  # N.B.: units have been converted to SI (mm/day -> 1/86400 kg/m^2/s, kPa -> 1000 Pa, and Celsius to K)
  pet = Variable(data=pet, name='pet', units='kg/m^2/s', axes=dataset['ps'].axes)
  assert 'waterflx' not in dataset or pet.units == dataset['waterflx'].units, pet
  # return new variable(s)
  return (pet,rad,wnd) if lterms else pet

# compute potential evapo-transpiration
def computePotEvapTh(dataset):
  ''' function to compute potential evapotranspiration (according to Thornthwaite method:
      https://en.wikipedia.org/wiki/Potential_evaporation) '''
  raise NotImplementedError
  # check prerequisites
  for pv in (): 
    if pv not in dataset: raise VariableError("Prerequisite '{:s}' for potential evapo-transpiration not found.".format(pv))
    else: dataset[pv].load() # load data for computation
  # compute waterflux (returns a Variable instance)
  var = dataset['']
  var.name = 'pet' # give correct name (units should be correct)
  assert var.units == dataset[''].units, var
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
