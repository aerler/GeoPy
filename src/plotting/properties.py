'''
Created on 2011-05-29, adapted on 2013-08-24

Properties of commonly used meteorological and climate variables.
Note that default scale-factors assume that the data is in SI units (not plot units).

@author: Andre R. Erler, GPL v3
'''

from collections import namedtuple
from types import NoneType
from atmdyn.constants import cp, g0

## dictionary of plot attributes for some common variables

# a class of plot attributes based on named tuples
class PlotAtts(namedtuple('PlotAtts', ['name','title','units','scale', 'preserve','scalefactor','offset'], verbose=False, rename=False)):
  # define some sensible default values
  def __new__(cls, name        = 'unknown', 
                   title       = 'unknown variable', 
                   units       = 'n/a', 
                   scale       = 'linear', 
                   preserve    = 'value', 
                   scalefactor = 1, 
                   offset      = 0):
    # create new instance with default values
    return super(PlotAtts,cls).__new__(cls,name=name,title=title,units=units,scale=scale,
                                       preserve=preserve,scalefactor=scalefactor,offset=offset)
  # also provide wrapper to _replace to present a similar interface as dict and AttrDict
  def copy(self, **kwargs):
    ''' create a copy of the namedtuple (new instance); fields specified as kwargs will be replaced '''
    return self._replace(**kwargs)
      

# dictionary of variable specific plot
variablePlotatts = dict()

## Generic Category
tmp = PlotAtts(name = 'Cat.', title = 'Category', units = '')
# add to collection
variablePlotatts['cat'] = tmp

##Histogram
tmp = PlotAtts(name = 'Histogram', title = 'Histogram', 
               units = '#', preserve = 'area')
# add to collection
variablePlotatts['hist'] = tmp
## Density Distribution
tmp = PlotAtts(name = 'PDF', title = 'Density Distribution', 
               units = '', preserve = 'area')
# add to collection
variablePlotatts['pdf'] = tmp
## Cumulative Distribution
tmp = PlotAtts(name = 'CDF', title = 'Cumulative Distribution', 
               units = '', preserve = 'area')
# add to collection
variablePlotatts['cdf'] = tmp

## *** Dimensions/Axes (1D) ***

## Time in Minutes (WRF)
tmp = PlotAtts(name = 'Time', title = 'Elapsed Time', 
               units = r'$min$')
# add to collection
variablePlotatts['xtime'] = tmp

## Longitude
tmp = PlotAtts(name = 'Lon', title = 'Longitude', 
               units = r'$^\circ$E')
# add to collection
variablePlotatts['lon'] = tmp

## Latitude
tmp = PlotAtts(name = 'Lat', title = 'Latitude', 
               units = r'$^\circ$N')
# add to collection
variablePlotatts['lat'] = tmp

## ** Surface Vars (2D) ***

## Land Mask
tmp = PlotAtts(name = 'Land', title = 'Land Mask', 
               units = '', scalefactor = 1)
# add to collection
variablePlotatts['lnd'] = tmp

## Land Classes
tmp = PlotAtts(name = 'Land Cat.', title = 'Land Classes', 
               units = '', scalefactor = 1)
# add to collection
variablePlotatts['lndcls'] = tmp

## Station Distribution
tmp = PlotAtts(name = 'Stations', title = 'Station Distribution', 
               units = '#', scalefactor = 1)
# add to collection
variablePlotatts['stns'] = tmp

## Surface Geopotential Height
tmp = PlotAtts(name = r'$Z_s$', title = 'Terrain Height', 
               units = 'km', scalefactor = 1e-3)
# add to collection
variablePlotatts['zs'] = tmp

## Surface Pressure
tmp = PlotAtts(name = r'$p_s$', title = 'Surface Pressure', 
               units = 'hPa', scalefactor = 1e-2)
# add to collection
variablePlotatts['ps'] = tmp
## Logarithm of Surface Pressure
tmp = PlotAtts(name = r'ln $p_s$', title = 'Logarithmic Surface Pressure', 
               units = '', scalefactor = 1)
# add to collection
variablePlotatts['lnps'] = tmp

## Mean Sea-level Pressure
tmp = PlotAtts(name = r'$p_{msl}$', title = 'Surface Pressure (MSL)', 
               units = 'hPa', scalefactor = 1e-2)
# add to collection
variablePlotatts['pmsl'] = tmp

## 10 m Zonal Wind
tmp = PlotAtts(name = r'$u_{10m}$', title = '10 m Zonal Wind', 
               units = 'm/s', scalefactor = 1)
# add to collection
variablePlotatts['u10'] = tmp

## 10 m Meridional Wind
tmp = PlotAtts(name = r'$v_{10m}$', title = '10 m Meridional Wind', 
               units = 'm/s', scalefactor = 1)
# add to collection
variablePlotatts['v10'] = tmp

## 2 m Temperature
tmp = PlotAtts(name = r'$T_{2m}$', title = '2 m Temperature', 
               units = 'K', scalefactor = 1)
# add to collection
variablePlotatts['T2'] = tmp

## 2 m Maximum Temperature
tmp = PlotAtts(name = r'$T_{max}$', title = 'Maximum 2 m Temperature', 
               units = 'K', scalefactor = 1)
# add to collection
variablePlotatts['Tmax'] = tmp

## 2 m Minimum Temperature
tmp = PlotAtts(name = r'$T_{min}$', title = 'Minimum 2 m Temperature', 
               units = 'K', scalefactor = 1)
# add to collection
variablePlotatts['Tmin'] = tmp

## Skin Temperature
tmp = PlotAtts(name = r'$T_s$', title = 'Skin Temperature', 
               units = 'K', scalefactor = 1)
# add to collection
variablePlotatts['Ts'] = tmp

## Skin Temperature
tmp = PlotAtts(name = 'sea-ice', title = 'Sea Ice Fraction', 
               units = '', scalefactor = 1)
# add to collection
variablePlotatts['seaice'] = tmp

## Snow (water-equivalent)
tmp = PlotAtts(name = 'snow', title = 'Snow (water-equivalent)', 
               units = r'$kg/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['snow'] = tmp

## Snow (depth/height)
tmp = PlotAtts(name = 'snowh', title = 'Snow Depth', 
               units = 'm', scalefactor = 1)
# add to collection
variablePlotatts['snowh'] = tmp

## 2 m Water Vapor Mixing Ratio
tmp = PlotAtts(name = '$Q_{2m}$', title = '2 m Water Vapor', 
               units = 'hPa', scalefactor = 1.e-2)
# add to collection
variablePlotatts['Q2'] = tmp

## Accumulated Cumulus Precipitation
tmp = PlotAtts(name = 'acc. precip (cu)', title = 'Accumulated Cumulus Precipitation', 
               units = 'mm', scalefactor = 1)
# add to collection
variablePlotatts['acpreccu'] = tmp

## Accumulated Grid-scale Precipitation
tmp = PlotAtts(name = 'acc. precip (grid)', title = 'Accumulated Grid-scale Precipitation', 
               units = 'mm', scalefactor = 1)
# add to collection
variablePlotatts['acprecnc'] = tmp

## Accumulated Total Precipitation
tmp = PlotAtts(name = 'acc. precip', title = 'Accumulated Total Precipitation', 
               units = 'mm', scalefactor = 1)
# add to collection
variablePlotatts['acprec'] = tmp

## Cumulus Precipitation Rate
tmp = PlotAtts(name = 'precip (cu)', title = 'Cumulus Precipitation Rate', 
               units = 'mm/day', scalefactor = 86400)
# add to collection
variablePlotatts['preccu'] = tmp

## Accumulated Grid-scale Precipitation
tmp = PlotAtts(name = 'precip (grid)', title = 'Grid-scale Precipitation Rate', 
               units = 'mm/day', scalefactor = 86400)
# add to collection
variablePlotatts['precnc'] = tmp

## Total Precipitation Rate
tmp = PlotAtts(name = 'precip', title = 'Total Precipitation Rate', 
               units = 'mm/day', scalefactor = 86400)
# add to collection
variablePlotatts['precip'] = tmp

## Surface Sensible Heat Flux
tmp = PlotAtts(name = r'$Q_s$', title = 'Surface Heat Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['hfx'] = tmp

## Surface Latent Heat Flux
tmp = PlotAtts(name = r'$Q_{LH}$', title = 'Surface Latent Heat Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['lhfx'] = tmp

## Total Surface Heat Flux
tmp = PlotAtts(name = r'$Q_{tot}$', title = 'Total Surface Heat Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['qtfx'] = tmp

## Relative Latent Heat Flux
tmp = PlotAtts(name = r'$Q_{LH} / Q_{tot}$', title = 'Relative Latent Heat Flux', 
               units = '', scalefactor = 1)
# add to collection
variablePlotatts['lhfr'] = tmp

## Surface Downward LW Radiative Flux
tmp = PlotAtts(name = r'$LW_{down}$', title = 'Downward LW Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['GLW'] = tmp

## Surface Downward SW Radiative Flux
tmp = PlotAtts(name = r'$SW_{down}$', title = 'Downward SW Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['SWDOWN'] = tmp

## Surface Normal SW Radiative Flux
tmp = PlotAtts(name = r'$SW_{norm}$', title = 'Surface Normal (SW) Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['SWNORM'] = tmp

## Residual Upward Energy Flux (basically upward LW)
tmp = PlotAtts(name = r'$LW_{up}$', title = '(Residual) Upward LW Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['rfx'] = tmp

## Outgoing Longwave Radiation
tmp = PlotAtts(name = r'OLR', title = 'Outgoing Longwave Radiation', 
               units = r'$W/m^2$', scalefactor = 1)
# add to collection
variablePlotatts['OLR'] = tmp

## Potential Evapo-Transpiration
tmp = PlotAtts(name = 'PET', title = 'Potential Evapo-Transpiration', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['pet'] = tmp

## Evapo-Transpiration
tmp = PlotAtts(name = 'ET', title = 'Evapo-Transpiration', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['evap'] = tmp

## Net Precipitation
tmp = PlotAtts(name = 'P - ET', title = 'Precipitation - Evaporation', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['p-et'] = tmp

## Net Surface Moisture Flux
tmp = PlotAtts(name = 'WaterFlx', title = 'Net Water Flux', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['waterflx'] = tmp

## Snowmelt (water equivalent)
tmp = PlotAtts(name = 'Snowmelt', title = 'Snowmelt', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['snwmlt'] = tmp

## Total Runoff
tmp = PlotAtts(name = 'Runoff', title = 'Total Runoff', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['runoff'] = tmp

## Surface Runoff
tmp = PlotAtts(name = 'Srfc. RO', title = 'Surface Runoff', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['sfroff'] = tmp

## Underground Runoff
tmp = PlotAtts(name = 'Sub-srfc. RO', title = 'Underground Runoff', 
               units = r'$kg m^{-2} day^{-1}$', scalefactor = 86400)
# add to collection
variablePlotatts['ugroff'] = tmp

## Discharge (river flow)
tmp = PlotAtts(name = 'Discharge', title = 'Discharge', 
               units = r'$10^6 kg s^{-1}$', scalefactor = 1e-6)
# add to collection
variablePlotatts['discharge'] = tmp

## *** Standard Vars (3D) ***

## Geopotential Height
tmp = PlotAtts(name = 'Z', title = 'Geopotential Height', units = 'km', scalefactor = 1e-3)
# add to collection
variablePlotatts['z'] = tmp

## Zonal Wind
tmp = PlotAtts(name = 'u', title = 'Zonal Wind', units = 'm/s')
# add to collection
variablePlotatts['u'] = tmp

## Meridional Wind
tmp = PlotAtts(name = 'v', title = 'Meridional Wind',units = 'm/s')
# add to collection
variablePlotatts['v'] = tmp

## Vertical Velocity
tmp = PlotAtts(name = 'w', title = 'Vertical Velocity',units = 'm/s')
# add to collection
variablePlotatts['w'] = tmp

## Pressure
tmp = PlotAtts(name = 'p', title = 'Pressure', units = 'hPa', scalefactor = 1e-2)
# add to collection
variablePlotatts['p'] = tmp

## Density (of dry air)
tmp = PlotAtts(name = 'rho', title = 'Density', units = r'$kg/m^3$')
# add to collection
variablePlotatts['rho'] = tmp

## Temperature
tmp = PlotAtts(name = 'T', title = 'Temperature', units = 'K')
# add to collection
variablePlotatts['T'] = tmp

## Potential Temperature
tmp = PlotAtts(name = 'theta', title = 'Potential Temperature', units = 'K')
# add to collection
variablePlotatts['th'] = tmp 

## Entropy
tmp = PlotAtts(name = 's', title = 'Entropy', units = r'$J kg^{-1} K^{-1}$')
# add to collection
variablePlotatts['s'] = tmp

## Lapse-rate (Temperature)
tmp = PlotAtts(name = 'lr', title = 'Lapse-rate', units = 'K/km', scalefactor = 1e3)
# add to collection
variablePlotatts['lr'] = tmp

## Potential Temperature Lapse-rate
tmp = PlotAtts(name = 'theta_z', title = 'Theta Lapse-rate', 
               units = 'K/km', scalefactor = 1e3)
# add to collection
variablePlotatts['thlr'] = tmp

## Brunt-Vaeisaelae Frequency Squared
tmp = PlotAtts(name = 'N^2', title = r'$N^2$', 
               units = r'$10^{-4}s^{-2}$', scalefactor = 1e4)
# add to collection
variablePlotatts['N2'] = tmp

## Relative Vorticity (vertical component)
tmp = PlotAtts(name = 'zeta', title = 'Relative Vorticity', 
               units = '10^-4 s^-1', scalefactor = 1e4)
# add to collection
variablePlotatts['ze'] = tmp

## Potential Vorticity
tmp = PlotAtts(name = 'PV', title = 'Potential Vorticity', 
               units = 'PVU', scalefactor = 1e6)
# add to collection
variablePlotatts['PV'] = tmp

## Potential Vorticity
tmp = PlotAtts(name = 'PV_s', title = 'Entropy Potential Vorticity', 
               units = 'PVU_s', scalefactor = 1e6)
# add to collection
variablePlotatts['PVs'] = tmp

## Vertical Potential Vorticity Gradient
tmp = PlotAtts(name = 'dPV/dz ', title = 'Vertical Gradient of Potential Vorticity', 
               units = 'PVU/km', scalefactor = 1e9)
# add to collection
variablePlotatts['dPV'] = tmp

## Brunt-Vaeisaelae Frequency Squared
tmp = PlotAtts(name = 'N^2', title = 'Entropy Gradient N^2', 
               units = '10^-4 s^-2', scalefactor = g0/cp*1e4)
# add to collection
variablePlotatts['N2s'] = tmp

## *** Spectral Vars ***

## Wavenumber
tmp = PlotAtts(name = 'n', title = 'Wavenumber', 
               units = '', scale = 'log')
# add to collection
variablePlotatts['n'] = tmp

## Kinetic Energy Spectral Density per Unit Mass
tmp = PlotAtts(name = r'$E_n$', title = 'Kinetic Energy', 
               units = r'$m^2 s^{-2}$', scale = 'log')
# add to collection
variablePlotatts['En'] = tmp

## Spectral Energy Flux per Unit Mass
tmp = PlotAtts(name = r'$F_n$', title = 'Energy Flux', scalefactor = 1.e4, 
               units = r'$10^{-4} m^2 s^{-3}$', scale = 'linear' )
# add to collection
variablePlotatts['Fn'] = tmp

## Spectral Energy Tendencies / Interaction Terms per Unit Mass
tmp = PlotAtts(name = r'$I_n$', title = 'Energy Tendencies', scalefactor = 1.e4, 
               units = r'$10^{-4} m^2 s^{-3}$', scale = 'linear', preserve = 'area'  )
# add to collection
variablePlotatts['In'] = tmp

## Enstrophy Spectral Density per Unit Mass
tmp = PlotAtts(name = r'$Z_n$', title = 'Enstrophy', 
               units = r'$s^{-2}$', scale = 'log' )
# add to collection
variablePlotatts['Zn'] = tmp

## Spectral Enstrophy Flux per Unit Mass
tmp = PlotAtts(name = r'$H_n$', title = 'Enstrophy Flux', scalefactor = 1.e15, 
               units = r'$10^{-15} s^{-3}$', scale = 'linear')
# add to collection
variablePlotatts['Hn'] = tmp

## Spectral Enstrophy Tendencies / Interaction Terms per Unit Mass
tmp = PlotAtts(name = r'$J_n$', title = 'Enstrophy Tendencies', scalefactor = 1.e15, 
               units = r'$10^{-15} s^{-3}$', scale = 'linear', preserve = 'area') 
# add to collection
variablePlotatts['Jn'] = tmp

## *** Misc Vars ***

## WMO Tropopause Height
tmp = PlotAtts(name = 'WMO-TP', title = 'WMO Tropopause Height', 
               units = 'km', scalefactor = 1e-3)
# add to collection
variablePlotatts['WMOTP'] = tmp

from geodata.misc import ArgumentError

## function to retrieve plot atts based on a variable name, units, and atts
def getPlotAtts(name=None, units=None, atts=None, plot=None, plotatts_dict=None):
  ''' figure out sensible plotatts atts based on name, units, and atts '''
  # check input
  if name is not None: pass 
  elif atts is not None and 'name' in atts: name = atts['name']  
  else: raise ArgumentError
  if units is not None: pass 
  elif atts is not None and 'units' in atts: units = atts['units']  
  else: raise ArgumentError
  if not isinstance(atts,(dict,NoneType)): raise TypeError
  if not isinstance(plot,(dict,PlotAtts,NoneType)): raise TypeError
  plotatts_dict = plotatts_dict or variablePlotatts
  if not isinstance(plotatts_dict,dict): raise TypeError
  # find variable in plotatts_dict (based on name)
  prefix = postfix = ''  
  basename = name
  # get base plotatts atts
  if isinstance(plot,PlotAtts):
    plotatts = plot.copy() 
  elif name in variablePlotatts: 
    plotatts = variablePlotatts[name].copy()
  else:
    if name[:3].lower() in ('min','max'):
      prefix = name[:3].lower()
      basename = name[3:]
    if basename in variablePlotatts:
      plotatts = variablePlotatts[basename].copy()
    elif basename.lower() in variablePlotatts:
      plotatts = variablePlotatts[basename.lower()].copy()
    else:
      namelist = basename.split('_')
      basename = namelist[0]
      postfix = namelist[1] if len(namelist)>1 else ''
      if basename in variablePlotatts: 
        plotatts = variablePlotatts[basename].copy()
      else:
        # last resort...
        plotatts = PlotAtts(name=name, units=units, title=name)  
  # modify according to variable specifics
  if prefix == 'max': 
    plotatts = plotatts.copy(name = 'Max. '+plotatts.name, title = 'Maximum '+plotatts.title) 
  elif prefix == 'min': 
    plotatts = plotatts.copy(name = 'Min. '+plotatts.name, title = 'Minimum '+plotatts.title)
  if len(postfix)>0: # these are mostly things like, e.g., '7d' for 7d means
    plotatts = plotatts.copy(name = plotatts.name+' ({:s})'.format(postfix),
                     title = plotatts.title+' ({:s})'.format(postfix))
  # adjust units
  if units == plotatts.units: 
    plotatts = plotatts.copy(scalefactor = 1, offset = 0) # no conversion necessary
  # plotatts dictionary override
  if isinstance(plot,dict): plotatts = plotatts.copy(**plot)
  # return variable with new plotatts  
  return plotatts

## function to update all variable properties in a dataset
def updateAllPlotAtts(dataset, mode='update', plot=None, default=True):
  # defaults (see above)
  if not plot: plot = variablePlotatts
  # loop over variables
  for var in dataset.vardict.iterkeys():
    if plot.has_key(var):
      if not hasattr(dataset.vardict[var],'plot'):
        # add plot if not present
        dataset.vardict[var] = plot[var]
      else:
        # update if present
        if mode=='update': 
          # overwrite changes with defaults (default behaviour)
          dataset.vardict[var].plot.update(plot[var])
        elif mode=='replace':
          # replace old dictionary with new one (new keys are lost)
          dataset.vardict[var].plot=plot[var]
        elif mode=='add':
          # respect changes, only add keys not previously present
          # Note: this option will not change already present default values!
          dataset.vardict[var].plot = plot[var].update(dataset.vardict[var].plot)
        else:
          pass # do nothing
    # if the variable/name is not in plot atts...
    else: 
      print('Variable \'%s\' not found in plot attributes - no update performed.'%var)
      if default:
        # generate default dictionary
        defpa = PlotAtts()
#         defpa.name = dataset.vardict[var].atts.get('name','unknown')
#         defpa.title = dataset.vardict[var].atts.get('standard_name',defpa['plotname'])
#         defpa.units = dataset.vardict[var].atts.get('units','n/a')
        # add default plot atts
        if hasattr(dataset.vardict[var],'plot'):
          dataset.vardict[var].plot.update(defpa)
        else:
          dataset.vardict[var].plot = defpa
  # return same old dataset with updated plot atts
  return dataset
  