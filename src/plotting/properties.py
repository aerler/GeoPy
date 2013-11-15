'''
Created on 2011-05-29, adapted on 2013-08-24

Properties of commonly used meteorological and climate variables.
Note that default scale-factors assume that the data is in SI units.

@author: Andre R. Erler, GPL v3
'''

from atmdyn.constants import cp, g0

## dictionary of plot attributes for some common variables
defaultPlotatts = dict()
# general default values
defaultPlotatts['plotname'] = 'unknown'
defaultPlotatts['plottitle'] = 'unknown variable'
defaultPlotatts['plotunits'] = 'n/a'
defaultPlotatts['potscale'] = 'linear'
defaultPlotatts['preserve'] = 'value'
defaultPlotatts['scalefactor'] = 1
defaultPlotatts['offset'] = 0  

# dictionary of variable specific plotatts
variablePlotatts = dict()

## Generic Category
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Cat.'
tmp['plottitle'] = 'Category'
tmp['plotunits'] = ''
# add to collection
variablePlotatts['cat'] = tmp

## *** Dimensions/Axes (1D) ***

## Time in Minutes (WRF)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Time'
tmp['plottitle'] = 'Elapsed Time'
tmp['plotunits'] = r'$min$'
# add to collection
variablePlotatts['xtime'] = tmp

## Longitude
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Lon'
tmp['plottitle'] = 'Longitude'
tmp['plotunits'] = r'$^\circ$E'
# add to collection
variablePlotatts['lon'] = tmp

## Latitude
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Lat'
tmp['plottitle'] = 'Latitude'
tmp['plotunits'] = r'$^\circ$N'
# add to collection
variablePlotatts['lat'] = tmp

## ** Surface Vars (2D) ***

## Land Mask
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Land'
tmp['plottitle'] = 'Land Mask'
tmp['plotunits'] = ''
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['lnd'] = tmp

## Land Classes
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Land Cat.'
tmp['plottitle'] = 'Land Classes'
tmp['plotunits'] = ''
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['lndcls'] = tmp

## Station Distribution
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Stations'
tmp['plottitle'] = 'Station Distribution'
tmp['plotunits'] = '#'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['stns'] = tmp

## Surface Geopotential Height
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$Z_s$'
tmp['plottitle'] = 'Terrain Height'
tmp['plotunits'] = 'km'
tmp['scalefactor'] = 1e-3
# add to collection
variablePlotatts['zs'] = tmp

## Surface Pressure
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$p_s$'
tmp['plottitle'] = 'Surface Pressure'
tmp['plotunits'] = 'hPa'
tmp['scalefactor'] = 1e-2
# add to collection
variablePlotatts['ps'] = tmp
## Logarithm of Surface Pressure
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'ln $p_s$'
tmp['plottitle'] = 'Logarithmic Surface Pressure'
tmp['plotunits'] = ''
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['lnps'] = tmp

## Mean Sea-level Pressure
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$p_{msl}$'
tmp['plottitle'] = 'Surface Pressure (MSL)'
tmp['plotunits'] = 'hPa'
tmp['scalefactor'] = 1e-2
# add to collection
variablePlotatts['pmsl'] = tmp

## 10m Zonal Wind
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$u_{10m}$'
tmp['plottitle'] = 'Zonal Wind at 10m'
tmp['plotunits'] = 'm/s'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['u10'] = tmp

## 10m Meridional Wind
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$v_{10m}$'
tmp['plottitle'] = 'Meridional Wind at 10m'
tmp['plotunits'] = 'm/s'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['v10'] = tmp

## 2m Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$T_{2m}$'
tmp['plottitle'] = 'Temperature at 2m'
tmp['plotunits'] = 'K'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['T2'] = tmp

## 2m Maximum Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$T_{max}$'
tmp['plottitle'] = 'Maximum 2m Temperature'
tmp['plotunits'] = 'K'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['Tmax'] = tmp

## 2m Minimum Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$T_{min}$'
tmp['plottitle'] = 'Minimum 2m Temperature'
tmp['plotunits'] = 'K'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['Tmin'] = tmp

## Skin Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$T_s$'
tmp['plottitle'] = 'Skin Temperature'
tmp['plotunits'] = 'K'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['Ts'] = tmp

## Skin Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'sea-ice'
tmp['plottitle'] = 'Sea Ice Fraction'
tmp['plotunits'] = ''
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['seaice'] = tmp

## Snow (water-equivalent)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'snow'
tmp['plottitle'] = 'Snow (water-equivalent)'
tmp['plotunits'] = r'$kg/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['snow'] = tmp

## Snow (depth/height)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'snowh'
tmp['plottitle'] = 'Snow Depth'
tmp['plotunits'] = 'm'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['snowh'] = tmp

## 2m Water Vapor Mixing Ratio
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Q_2m'
tmp['plottitle'] = 'Water Vapor at 2m'
tmp['plotunits'] = 'hPa'
tmp['scalefactor'] = 1.e-2
# add to collection
variablePlotatts['Q2'] = tmp

## Accumulated Cumulus Precipitation
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'acc. precip (cu)'
tmp['plottitle'] = 'Accumulated Cumulus Precipitation'
tmp['plotunits'] = 'mm' # = kg/m^2
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['acpreccu'] = tmp

## Accumulated Grid-scale Precipitation
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'acc. precip (grid)'
tmp['plottitle'] = 'Accumulated Grid-scale Precipitation'
tmp['plotunits'] = 'mm'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['acprecnc'] = tmp

## Accumulated Total Precipitation
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'acc. precip'
tmp['plottitle'] = 'Accumulated Total Precipitation'
tmp['plotunits'] = 'mm'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['acprec'] = tmp

## Cumulus Precipitation Rate
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'precip (cu)'
tmp['plottitle'] = 'Cumulus Precipitation Rate'
tmp['plotunits'] = 'mm/day' # = kg/m^2/(86400s)
tmp['scalefactor'] = 86400
# add to collection
variablePlotatts['preccu'] = tmp

## Accumulated Grid-scale Precipitation
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'precip (grid)'
tmp['plottitle'] = 'Grid-scale Precipitation Rate'
tmp['plotunits'] = 'mm/day'
tmp['scalefactor'] = 86400
# add to collection
variablePlotatts['precnc'] = tmp

## Total Precipitation Rate
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'precip'
tmp['plottitle'] = 'Total Precipitation Rate'
tmp['plotunits'] = 'mm/day'
tmp['scalefactor'] = 86400
# add to collection
variablePlotatts['precip'] = tmp

## Surface Sensible Heat Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$Q_s$'
tmp['plottitle'] = 'Surface Heat Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['hfx'] = tmp

## Surface Latent Heat Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$Q_{LH}$'
tmp['plottitle'] = 'Surface Latent Heat Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['lhfx'] = tmp

## Total Surface Heat Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$Q_{tot}$'
tmp['plottitle'] = 'Total Surface Heat Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['qtfx'] = tmp

## Relative Latent Heat Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$Q_{LH} / Q_{tot}$'
tmp['plottitle'] = 'Relative Latent Heat Flux'
tmp['plotunits'] = ''
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['lhfr'] = tmp

## Surface Downward LW Radiative Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$LW_{down}$'
tmp['plottitle'] = 'Downward LW Radiative Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['GLW'] = tmp

## Surface Downward SW Radiative Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$SW_{down}$'
tmp['plottitle'] = 'Downward SW Radiative Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['SWDOWN'] = tmp

## Surface Normal SW Radiative Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$SW_{norm}$'
tmp['plottitle'] = 'Surface Normal (SW) Radiative Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['SWNORM'] = tmp

## Residual Upward Energy Flux (basically upward LW)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$LW_{up}$'
tmp['plottitle'] = '(Residual) Upward LW Radiative Flux'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['rfx'] = tmp

## Outgoing Longwave Radiation
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'OLR'
tmp['plottitle'] = 'Outgoing Longwave Radiation'
tmp['plotunits'] = r'$W/m^2$'
tmp['scalefactor'] = 1
# add to collection
variablePlotatts['OLR'] = tmp

## Potential Evapo-Transpiration
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'PET'
tmp['plottitle'] = 'Potential Evapo-Transpiration'
tmp['plotunits'] = r'$kg m^{-2} day^{-1}$'
tmp['scalefactor'] = 86400
# add to collection
variablePlotatts['pet'] = tmp

## Evapo-Transpiration
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'ET'
tmp['plottitle'] = 'Evapo-Transpiration'
tmp['plotunits'] = r'$kg m^{-2} day^{-1}$'
tmp['scalefactor'] = 86400
# add to collection
variablePlotatts['evap'] = tmp

## Net Surface Moisture Flux
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'P - ET'
tmp['plottitle'] = 'Precipitation - Evaporation'
tmp['plotunits'] = r'$kg m^{-2} day^{-1}$'
tmp['scalefactor'] = 86400
# add to collection
variablePlotatts['p-et'] = tmp


## *** Standard Vars (3D) ***

## Geopotential Height
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'Z'
tmp['plottitle'] = 'Geopotential Height'
tmp['plotunits'] = 'km'
tmp['scalefactor'] = 1e-3
# add to collection
variablePlotatts['z'] = tmp

## Zonal Wind
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'u'
tmp['plottitle'] = 'Zonal Wind'
tmp['plotunits'] = 'm/s'
# add to collection
variablePlotatts['u'] = tmp

## Meridional Wind
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'v'
tmp['plottitle'] = 'Meridional Wind'
tmp['plotunits'] = 'm/s'
# add to collection
variablePlotatts['v'] = tmp

## Vertical Velocity
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'w'
tmp['plottitle'] = 'Vertical Velocity'
tmp['plotunits'] = 'm/s'
# add to collection
variablePlotatts['w'] = tmp

## Pressure
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'p'
tmp['plottitle'] = 'Pressure'
tmp['plotunits'] = 'hPa'
tmp['scalefactor'] = 1e-2
# add to collection
variablePlotatts['p'] = tmp

## Density (of dry air)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'rho'
tmp['plottitle'] = 'Density'
tmp['plotunits'] = r'$kg/m^3$'
# add to collection
variablePlotatts['rho'] = tmp

## Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'T'
tmp['plottitle'] = 'Temperature'
tmp['plotunits'] = 'K'
# add to collection
variablePlotatts['T'] = tmp

## Potential Temperature
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'theta'
tmp['plottitle'] = 'Potential Temperature'
tmp['plotunits'] = 'K'
# add to collection
variablePlotatts['th'] = tmp 

## Entropy
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 's'
tmp['plottitle'] = 'Entropy'
tmp['plotunits'] = r'$J kg^{-1} K^{-1}$'
# add to collection
variablePlotatts['s'] = tmp

## Lapse-rate (Temperature)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'lr'
tmp['plottitle'] = 'Lapse-rate'
tmp['plotunits'] = 'K/km'
tmp['scalefactor'] = 1e3
# add to collection
variablePlotatts['lr'] = tmp

## Potential Temperature Lapse-rate
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'theta_z'
tmp['plottitle'] = 'Theta Lapse-rate'
tmp['plotunits'] = 'K/km'
tmp['scalefactor'] = 1e3
# add to collection
variablePlotatts['thlr'] = tmp

## Brunt-Vaeisaelae Frequency Squared
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'N^2'
tmp['plottitle'] = r'$N^2$'
tmp['plotunits'] = r'$10^{-4}s^{-2}$'
tmp['scalefactor'] = 1e4
# add to collection
variablePlotatts['N2'] = tmp

## Relative Vorticity (vertical component)
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'zeta'
tmp['plottitle'] = 'Relative Vorticity'
tmp['plotunits'] = '10^-4 s^-1'
tmp['scalefactor'] = 1e4
# add to collection
variablePlotatts['ze'] = tmp

## Potential Vorticity
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'PV'
tmp['plottitle'] = 'Potential Vorticity'
tmp['plotunits'] = 'PVU'
tmp['scalefactor'] = 1e6
# add to collection
variablePlotatts['PV'] = tmp

## Potential Vorticity
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'PV_s'
tmp['plottitle'] = 'Entropy Potential Vorticity'
tmp['plotunits'] = 'PVU_s'
tmp['scalefactor'] = 1e6
# add to collection
variablePlotatts['PVs'] = tmp

## Vertical Potential Vorticity Gradient
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'dPV/dz '
tmp['plottitle'] = 'Vertical Gradient of Potential Vorticity'
tmp['plotunits'] = 'PVU/km'
tmp['scalefactor'] = 1e9
# add to collection
variablePlotatts['dPV'] = tmp

## Brunt-Vaeisaelae Frequency Squared
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'N^2'
tmp['plottitle'] = 'Entropy Gradient N^2'
tmp['plotunits'] = '10^-4 s^-2'
tmp['scalefactor'] = g0/cp*1e4
# add to collection
variablePlotatts['N2s'] = tmp

## *** Spectral Vars ***

## Wavenumber
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'n'
tmp['plottitle'] = 'Wavenumber'
tmp['plotunits'] = ''
tmp['plotscale'] = 'log'
# add to collection
variablePlotatts['n'] = tmp

## Kinetic Energy Spectral Density per Unit Mass
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$E_n$'
tmp['plottitle'] = 'Kinetic Energy' #  per Unit Mass
tmp['plotunits'] = r'$m^2 s^{-2}$'
tmp['plotscale'] = 'log'
# add to collection
variablePlotatts['En'] = tmp

## Spectral Energy Flux per Unit Mass
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$F_n$'
tmp['plottitle'] = 'Energy Flux' # per Unit Mass
tmp['scalefactor'] = 1.e4
tmp['plotunits'] = r'$10^{-4} m^2 s^{-3}$'
tmp['plotscale'] = 'linear' 
# add to collection
variablePlotatts['Fn'] = tmp

## Spectral Energy Tendencies / Interaction Terms per Unit Mass
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$I_n$'
tmp['plottitle'] = 'Energy Tendencies' # per Unit Mass
tmp['scalefactor'] = 1.e4
tmp['plotunits'] = r'$10^{-4} m^2 s^{-3}$'
tmp['plotscale'] = 'linear'
tmp['preserve'] = 'area'  
# add to collection
variablePlotatts['In'] = tmp

## Enstrophy Spectral Density per Unit Mass
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$Z_n$'
tmp['plottitle'] = 'Enstrophy' # per Unit Mass
tmp['plotunits'] = r'$s^{-2}$'
tmp['plotscale'] = 'log' 
# add to collection
variablePlotatts['Zn'] = tmp

## Spectral Enstrophy Flux per Unit Mass
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$H_n$'
tmp['plottitle'] = 'Enstrophy Flux' # per Unit Mass
tmp['scalefactor'] = 1.e15
tmp['plotunits'] = r'$10^{-15} s^{-3}$'
tmp['plotscale'] = 'linear' 
# add to collection
variablePlotatts['Hn'] = tmp

## Spectral Enstrophy Tendencies / Interaction Terms per Unit Mass
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = r'$J_n$'
tmp['plottitle'] = 'Enstrophy Tendencies' # per Unit Mass
tmp['scalefactor'] = 1.e15
tmp['plotunits'] = r'$10^{-15} s^{-3}$'
tmp['plotscale'] = 'linear'
tmp['preserve'] = 'area' 
# add to collection
variablePlotatts['Jn'] = tmp

## *** Misc Vars ***

## WMO Tropopause Height
tmp = defaultPlotatts.copy() 
# specific properties
tmp['plotname'] = 'WMO-TP'
tmp['plottitle'] = 'WMO Tropopause Height'
tmp['plotunits'] = 'km'
tmp['scalefactor'] = 1e-3
# add to collection
variablePlotatts['WMOTP'] = tmp

## function to update all variable properties in a dataset
def updatePlotatts(dataset, mode='update', plotatts=None, default=True):
  # defaults (see above)
  if not plotatts: plotatts = variablePlotatts
  # loop over variables
  for var in dataset.vardict.iterkeys():
    if plotatts.has_key(var):
      if not hasattr(dataset.vardict[var],'plotatts'):
        # add plotatts if not present
        dataset.vardict[var] = plotatts[var]
      else:
        # update if present
        if mode=='update': 
          # overwrite changes with defaults (default behaviour)
          dataset.vardict[var].plotatts.update(plotatts[var])
        elif mode=='replace':
          # replace old dictionary with new one (new keys are lost)
          dataset.vardict[var].plotatts=plotatts[var]
        elif mode=='add':
          # respect changes, only add keys not previously present
          # Note: this option will not change already present default values!
          dataset.vardict[var].plotatts = plotatts[var].update(dataset.vardict[var].plotatts)
        else:
          pass # do nothing
    # if the variable/name is not in plotatts...
    else: 
      print('Variable \'%s\' not found in plotatts dictionary - no update performed.'%var)
      if default:
        # generate default dictionary
        defpa = defaultPlotatts
        defpa['plotname'] = dataset.vardict[var].atts.get('name','unknown')
        defpa['plottitle'] = dataset.vardict[var].atts.get('standard_name',defpa['plotname'])
        defpa['plotunits'] = dataset.vardict[var].atts.get('units','n/a')
        # add default plotatts
        if hasattr(dataset.vardict[var],'plotatts'):
          dataset.vardict[var].plotatts.update(defpa)
        else:
          dataset.vardict[var].plotatts = defpa
  # return same old dataset with updated plotatts
  return dataset
  