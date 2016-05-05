'''
Created on May 5, 2016

A collection of functions to compute derived variables, primarily for WRF but also in general.

@author: Andre R. Erler, GPL v3
'''

# internal imports
from geodata.base import VariableError


## functions to compute relevant variables (from a dataset)

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

# compute potential evapo-transpiration
def computePotEvapPM(dataset):
  ''' function to compute potential evapotranspiration (according to Penman-Monteith method:
      https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation,
      http://www.fao.org/docrep/x0490e/x0490e06.htm#formulation%20of%20the%20penman%20monteith%20equation)
  '''
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


if __name__ == '__main__':
    pass