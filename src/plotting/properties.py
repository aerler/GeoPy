'''
Created on 2011-05-29, adapted on 2013-08-24

Properties of commonly used meteorological and climate variables.
Note that default scale-factors assume that the data is in SI units (not plot units).

@author: Andre R. Erler, GPL v3
'''

from collections import namedtuple
from types import NoneType
from utils.constants import cp, g0

## dictionary of plot attributes for some common variables

# a class of plot attributes based on named tuples
class PlotAtts(namedtuple('PlotAtts', ['name','title','units','scale', 'preserve','scalefactor','offset'], verbose=False, rename=False)):
  # define some sensible default values
  def __new__(cls, name        = 'unknown', 
                   title       = 'unknown variable', 
                   units       = 'n/a', 
                   scale       = 'linear', 
                   preserve    = 'value', 
                   scalefactor = 1., 
                   offset      = 0.):
    # create new instance with default values
    return super(PlotAtts,cls).__new__(cls,name=name,title=title,units=units,scale=scale,
                                       scalefactor=float(scalefactor),offset=float(offset),
                                       preserve=preserve)
  # also provide wrapper to _replace to present a similar interface as dict and AttrDict
  def copy(self, **kwargs):
    ''' create a copy of the namedtuple (new instance); fields specified as kwargs will be replaced '''
    return self._replace(**kwargs)
      
precip_units = r'$mm/day$' # equivalent to '$kg m^{-2} day^{-1}$'

# dictionary of variable specific plot
variablePlotatts = dict()

## Generic Category
tmp = PlotAtts(name = 'Cat.', title = 'Category', units = '')
# add to collection
variablePlotatts['cat'] = tmp

## Histogram
tmp = PlotAtts(name = 'Histogram', title = 'Histogram', 
               units = '#', preserve = 'area')
# add to collection
variablePlotatts['hist'] = tmp
## Density Distribution
tmp = PlotAtts(name = 'PDF', title = 'Probability Density', 
               units = '', preserve = 'area')
# add to collection
variablePlotatts['pdf'] = tmp
## Cumulative Distribution
tmp = PlotAtts(name = 'CDF', title = 'Cumulative Probability', 
               units = '', preserve = 'value')
# add to collection
variablePlotatts['cdf'] = tmp
## p-value of statistic
tmp = PlotAtts(name = 'p-value', title = 'p-Value', 
               units = '', preserve = 'value')
# add to collection
variablePlotatts['pval'] = tmp
## quantile of statistic
tmp = PlotAtts(name = 'Quantile', title = 'Quantile', 
               units = '', preserve = 'value')
# add to collection
variablePlotatts['quant'] = tmp

## *** Dimensions/Axes (1D) ***

## Time in Minutes (WRF)
tmp = PlotAtts(name = 'Time', title = 'Elapsed Time', 
               units = r'$min$')
# add to collection
variablePlotatts['xtime'] = tmp

## Longitude
tmp = PlotAtts(name = 'Longitude', title = 'Longitude', 
               units = r'$^\circ$E')
# add to collection
variablePlotatts['lon'] = tmp

## Latitude
tmp = PlotAtts(name = 'Latitude', title = 'Latitude', 
               units = r'$^\circ$N')
# add to collection
variablePlotatts['lat'] = tmp

## West-East Coordinate
tmp = PlotAtts(name = 'Easting', title = 'UTM Easting', 
               units = 'km', scalefactor = 1e-3)
# add to collection
variablePlotatts['x'] = tmp

## South-North Coordinate
tmp = PlotAtts(name = 'Northing', title = 'UTM Northing', 
               units = 'km', scalefactor = 1e-3)
# add to collection
variablePlotatts['y'] = tmp


## ** Surface Vars (2D) ***

## Land Mask
tmp = PlotAtts(name = 'Land', title = 'Land Mask', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['lnd'] = tmp

## Land Use Classes
tmp = PlotAtts(name = 'LU Cat.', title = 'Land Use Classes', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['lu'] = tmp

## Station Distribution
tmp = PlotAtts(name = 'Stations', title = 'Station Distribution', 
               units = '#', scalefactor = 1.)
# add to collection
variablePlotatts['stns'] = tmp

## Wet-day Frequency
tmp = PlotAtts(name = 'Wet-days', title = 'Wet-day Frequency', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['wetfrq'] = tmp

## Frost-day Frequency
tmp = PlotAtts(name = 'Frost-days', title = 'Frost-day Frequency', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['frzfrq'] = tmp

## Surface Geopotential Height
tmp = PlotAtts(name = r'$Z_s$', title = 'Terrain Height', 
               units = 'km', scalefactor = 1e-3)
# tmp = PlotAtts(name = r'$Z_s$', title = 'Terrain Height', 
#                units = 'm', scalefactor = 1.)
# add to collection
variablePlotatts['zs'] = tmp

## Saturation
tmp = PlotAtts(name = r'Sat.', title = 'Relative Soil Saturation', 
               units = '%', scalefactor = 1e2)
# add to collection
variablePlotatts['sat'] = tmp

## Surface Pressure
tmp = PlotAtts(name = r'$p_s$', title = 'Surface Pressure', 
               units = 'hPa', scalefactor = 1e-2)
# add to collection
variablePlotatts['ps'] = tmp
## Logarithm of Surface Pressure
tmp = PlotAtts(name = r'ln $p_s$', title = 'Logarithmic Surface Pressure', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['lnps'] = tmp

## Mean Sea-level Pressure
tmp = PlotAtts(name = r'$p_{msl}$', title = 'Surface Pressure (MSL)', 
               units = 'hPa', scalefactor = 1e-2)
# add to collection
variablePlotatts['pmsl'] = tmp

## 10 m Zonal Wind
tmp = PlotAtts(name = r'$u_{10m}$', title = '10 m Zonal Wind', 
               units = r'$m s^{-1}$', scalefactor = 1.)
# add to collection
variablePlotatts['u10'] = tmp

waterflux_scale = 100.

## Column-integrated Zonal Water (Vapor) Transport
tmp = PlotAtts(name = r'$\overline{q^w_u}$', title = 'Zonal Water Transport', 
               units = r'$kg m^{-1} s^{-1}$', scalefactor = waterflux_scale)
# add to collection
variablePlotatts['cqwu'] = tmp

## Column-integrated Zonal Heat Flux
tmp = PlotAtts(name = r'$\overline{q^h_u}$', title = 'Zonal Heat Transport', units = 'J/m/s')
# add to collection
variablePlotatts['cqhu'] = tmp

## 10 m Meridional Wind
tmp = PlotAtts(name = r'$v_{10m}$', title = '10 m Meridional Wind', 
               units = r'$m s^{-1}$', scalefactor = 1.)
# add to collection
variablePlotatts['v10'] = tmp

## Column-integrated Meridional Water (Vapor) Transport
tmp = PlotAtts(name = r'$\overline{q^w_v}$', title = 'Meridional Water Transport',
               units = r'$kg m^{-1} s^{-1}$', scalefactor = waterflux_scale)
# add to collection
variablePlotatts['cqwv'] = tmp

## Column-integrated Meridional Heat Wind
tmp = PlotAtts(name = r'$\overline{q^h_v}$', title = 'Meridional Heat Transport',units = 'J/m/s')
# add to collection
variablePlotatts['cqhv'] = tmp

## Column-integrated Water (Vapor) Content
tmp = PlotAtts(name = r'$\overline{q^w}$', title = 'Column-integrated Water',units = 'kg/m^2')
# add to collection
variablePlotatts['cqw'] = tmp

## Column-integrated Heat Content
tmp = PlotAtts(name = r'$\overline{q^h}$', title = 'Column-integrated Heat',units = 'J/m^2')
# add to collection
variablePlotatts['cqh'] = tmp

## 2 m Temperature
tmp = PlotAtts(name = r'$T_{2m}$', title = '2 m Temperature', 
               units = 'K', scalefactor = 1.)
# add to collection
variablePlotatts['T2'] = tmp

## 2 m Maximum Temperature
tmp = PlotAtts(name = r'$T_{max}$', title = 'Maximum 2 m Temperature', 
               units = 'K', scalefactor = 1.)
# add to collection
variablePlotatts['Tmax'] = tmp

## 2 m Minimum Temperature
tmp = PlotAtts(name = r'$T_{min}$', title = 'Minimum 2 m Temperature', 
               units = 'K', scalefactor = 1.)
# add to collection
variablePlotatts['Tmin'] = tmp

## Skin Temperature
tmp = PlotAtts(name = r'$T_s$', title = 'Skin Temperature', 
               units = 'K', scalefactor = 1.)
# add to collection
variablePlotatts['Ts'] = tmp

## Skin Temperature
tmp = PlotAtts(name = r'$T_soil$', title = 'Soil Temperature', 
               units = 'K', scalefactor = 1.)
# add to collection
variablePlotatts['Tslb'] = tmp

## Sea Ice Fraction
tmp = PlotAtts(name = 'sea-ice', title = 'Sea Ice Fraction', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['seaice'] = tmp

## Snow (water-equivalent)
tmp = PlotAtts(name = 'snow', title = 'Snow (water-equivalent)', 
               units = r'$kg/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['snow'] = tmp

## Snow (depth/height)
tmp = PlotAtts(name = 'snowh', title = 'Snow Depth', 
               units = 'm', scalefactor = 1.)
# add to collection
variablePlotatts['snowh'] = tmp

## (absolute) Soil Moisture
tmp = PlotAtts(name = 'Soil Moisture', title = '(absolute) Soil Moisture', 
               units = '$m^3/m^3$', scalefactor = 1.)
# add to collection
variablePlotatts['aSM'] = tmp

## 2 m Water Vapor Pressure
tmp = PlotAtts(name = '$Q_{2m}$', title = '2 m Water Vapor', 
               units = 'hPa', scalefactor = 1.e-2)
# add to collection
variablePlotatts['Q2'] = tmp

## Accumulated Cumulus Precipitation
tmp = PlotAtts(name = 'acc. precip (cu)', title = 'Accumulated Cumulus Precipitation', 
               units = 'mm', scalefactor = 1.)
# add to collection
variablePlotatts['acpreccu'] = tmp

## Accumulated Grid-scale Precipitation
tmp = PlotAtts(name = 'acc. precip (grid)', title = 'Accumulated Grid-scale Precipitation', 
               units = 'mm', scalefactor = 1.)
# add to collection
variablePlotatts['acprecnc'] = tmp

## Accumulated Total Precipitation
tmp = PlotAtts(name = 'acc. precip', title = 'Accumulated Total Precipitation', 
               units = 'mm', scalefactor = 1.)
# add to collection
variablePlotatts['acprec'] = tmp

## Cumulus Precipitation Rate
tmp = PlotAtts(name = 'precip (cu)', title = 'Cumulus Precipitation Rate', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['preccu'] = tmp

## Accumulated Grid-scale Precipitation
tmp = PlotAtts(name = 'precip (grid)', title = 'Grid-scale Precipitation Rate', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['precnc'] = tmp

## Solid Precipitation Rate
tmp = PlotAtts(name = 'solprec', title = 'Solid Precipitation Rate', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['solprec'] = tmp

## Liquid Precipitation Rate
tmp = PlotAtts(name = 'liqprec', title = 'Liquid Precipitation Rate', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['liqprec'] = tmp

## Total Precipitation Rate
tmp = PlotAtts(name = 'precip', title = 'Total Precipitation Rate', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['precip'] = tmp

## Wet-day Precipitation Rate
tmp = PlotAtts(name = 'Wet-day Precip', title = 'Wet-day Precipitation Rate', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['wetprec'] = tmp

## Wet-day Precipitation Rate
tmp = PlotAtts(name = 'Corrected Precip', title = 'Precipitation Exceeding Dry-day Threshold', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['dryprec'] = tmp

## Consecutive Wet Days
tmp = PlotAtts(name = 'CWD', title = 'Consecutive Wet Days', 
               units = 'days', scalefactor = 1.)
# add to collection
variablePlotatts['CWD'] = tmp

## Consecutive Dry Days
tmp = PlotAtts(name = 'CDD', title = 'Consecutive Dry Days', 
               units = 'days', scalefactor = 1.)
# add to collection
variablePlotatts['CDD'] = tmp

## Consecutive Net Wet Days
tmp = PlotAtts(name = 'CNWD', title = 'Consecutive Net Wet Days', 
               units = 'days', scalefactor = 1.)
# add to collection
variablePlotatts['CNWD'] = tmp

## Consecutive Dry Days
tmp = PlotAtts(name = 'CNDD', title = 'Consecutive Net Dry Days', 
               units = 'days', scalefactor = 1.)
# add to collection
variablePlotatts['CNDD'] = tmp

## Surface Sensible Heat Flux
tmp = PlotAtts(name = r'$Q_s$', title = 'Surface Heat Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['hfx'] = tmp

## Surface Latent Heat Flux
tmp = PlotAtts(name = r'$Q_{LH}$', title = 'Surface Latent Heat Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['lhfx'] = tmp

## Total Surface Heat Flux
tmp = PlotAtts(name = r'$Q_{tot}$', title = 'Total Surface Heat Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['qtfx'] = tmp

## Relative Latent Heat Flux
tmp = PlotAtts(name = r'$Q_{LH} / Q_{tot}$', title = 'Relative Latent Heat Flux', 
               units = '', scalefactor = 1.)
# add to collection
variablePlotatts['lhfr'] = tmp

## Surface Downward LW Radiative Flux
tmp = PlotAtts(name = r'$LW_{down}$', title = 'Downward LW Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['GLW'] = tmp
variablePlotatts['LWDNB'] = tmp

## Surface Downward SW Radiative Flux
tmp = PlotAtts(name = r'$SW_{down}$', title = 'Downward SW Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['SWDOWN'] = tmp
variablePlotatts['SWDNB'] = tmp
variablePlotatts['SWD'] = tmp

## Surface Normal SW Radiative Flux
tmp = PlotAtts(name = r'$SW_{norm}$', title = 'Surface Normal (SW) Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['SWNORM'] = tmp

## Residual Upward Energy Flux (basically upward LW)
tmp = PlotAtts(name = r'$LW_{up}$', title = '(Residual) Upward LW Radiative Flux', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['rfx'] = tmp

## Outgoing Longwave Radiation
tmp = PlotAtts(name = 'OLR', title = 'Outgoing Longwave Radiation', 
               units = r'$W/m^2$', scalefactor = 1.)
# add to collection
variablePlotatts['OLR'] = tmp
variablePlotatts['LWUPT'] = tmp

## Potential Evapo-Transpiration (WRF)
tmp = PlotAtts(name = 'WRF PET', title = 'Potential Evapo-Transpiration (WRF)', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['pet_wrf'] = tmp
## Potential Evapo-Transpiration
tmp = PlotAtts(name = 'PET', title = 'Potential Evapo-Transpiration', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['pet'] = tmp
## Radiation Term (PET)
tmp = PlotAtts(name = 'Rad. Term', title = 'Radiation Term of PET', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['petrad'] = tmp
## Wind Term (PET)
tmp = PlotAtts(name = 'Wind Term', title = 'Wind Term of PET', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['petwnd'] = tmp

## Vapor Deficit (PET)
tmp = PlotAtts(name = 'Vapor Deficit', title = 'Water Vapor Deficit', 
               units = 'hPa', scalefactor = 1.e-2)
# add to collection
variablePlotatts['vapdef'] = tmp

## Evapo-Transpiration
tmp = PlotAtts(name = 'ET', title = 'Evapo-Transpiration', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['evap'] = tmp

## Net Precipitation
tmp = PlotAtts(name = 'P - ET', title = 'Precipitation - Evaporation', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['p-et'] = tmp

## Net Surface Moisture Flux
tmp = PlotAtts(name = 'Water Flux', title = 'Net Water Flux', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['waterflx'] = tmp

## Water Flux into the Land Surface
tmp = PlotAtts(name = 'Water Flux', title = 'Liquid Water Flux', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['liqwatflx'] = tmp

## Snowmelt (water equivalent)
tmp = PlotAtts(name = 'Snowmelt', title = 'Snowmelt', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['snwmlt'] = tmp

## Total Runoff
tmp = PlotAtts(name = 'Runoff', title = 'Total Runoff', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['runoff'] = tmp

## Surface Runoff
tmp = PlotAtts(name = 'Srfc. RO', title = 'Surface Runoff', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['sfroff'] = tmp

## Underground Runoff
tmp = PlotAtts(name = 'Sub-srfc. RO', title = 'Underground Runoff', 
               units = precip_units, scalefactor = 86400.)
# add to collection
variablePlotatts['ugroff'] = tmp


## *** HGS Variables ***

## Vertical Darcy flux
tmp = PlotAtts(name = 'Vertical Flux', title = 'Vertical Darcy Flux', 
               units = precip_units, scalefactor = -86400000.)
# add to collection
variablePlotatts['dflx'] = tmp

## Surface-subsurface Exchange Flux
tmp = PlotAtts(name = 'Exchange Flux', title = 'Surface-subsurface Exchange Flux', 
               units = precip_units, scalefactor = -86400000.)
# add to collection
variablePlotatts['exflx'] = tmp

## Porous Media Evaporation
tmp = PlotAtts(name = 'PM Evaporation', title = 'Porous Media Evaporation', 
               units = precip_units, scalefactor = -86400000.)
# add to collection
variablePlotatts['evap_pm'] = tmp

## Surface Evaporation
tmp = PlotAtts(name = 'Evaporation', title = 'Surface Evaporation', 
               units = precip_units, scalefactor = -86400000.)
# add to collection
variablePlotatts['evap'] = tmp

## Porous Media Transpiration
tmp = PlotAtts(name = 'Transpiration', title = 'Porous Media Transpiration', 
               units = precip_units, scalefactor = -86400000.)
# add to collection
variablePlotatts['trans'] = tmp

## Total Evapo-Transpiration
tmp = PlotAtts(name = 'Total ET', title = 'Total Evapo-Transpiration', 
               units = precip_units, scalefactor = -86400000.)
# add to collection
variablePlotatts['ET'] = tmp

## Infiltration into the Subsurface
tmp = PlotAtts(name = 'Infiltration', title = 'Infiltration into the Subsurface', 
               units = precip_units, scalefactor = 86400000.)
# add to collection
variablePlotatts['infil'] = tmp

## Exfiltration from the Subsurface
tmp = PlotAtts(name = 'Exfiltration', title = 'Exfiltration from the Subsurface', 
               units = precip_units, scalefactor = 86400000.)
# add to collection
variablePlotatts['exfil'] = tmp

## Groundwater Recharge
tmp = PlotAtts(name = 'Recharge', title = 'Groundwater Recharge', 
               units = precip_units, scalefactor = 86400000.)
# add to collection
variablePlotatts['recharge'] = tmp

## Depth to Groundwater Table
tmp = PlotAtts(name = 'GW Depth', title = 'Depth to Groundwater Table', 
               units = 'm', scalefactor = 1.)
# add to collection
variablePlotatts['d_gw'] = tmp

## Soil Saturation
tmp = PlotAtts(name = 'Saturation', title = 'Soil Saturation', 
               units = '$\%$', scalefactor = 100.)
# add to collection
variablePlotatts['sat'] = tmp


# ## Discharge (river flow)
# tmp = PlotAtts(name = 'Discharge', title = 'Discharge', 
#                units = r'$m^3 s^{-1}$', scalefactor = 1e-3)
# # add to collection
# variablePlotatts['discharge'] = tmp

## *** Standard Vars (3D) ***

## Geopotential Height
tmp = PlotAtts(name = 'Z', title = 'Geopotential Height', units = 'km', scalefactor = 1e-3)
# add to collection
variablePlotatts['z'] = tmp

## Zonal Wind
tmp = PlotAtts(name = 'u', title = 'Zonal Wind', units = 'm/s')
# add to collection
variablePlotatts['u'] = tmp

## Zonal Water (Vapor) Flux
tmp = PlotAtts(name = r'$q^w_u$', title = 'Zonal Water Flux', units = r'$kg m^{-2} s^{-1}$',
               scalefactor = 1e2)
# add to collection
variablePlotatts['qwu'] = tmp

## Zonal Heat Flux
tmp = PlotAtts(name = r'$q^h_u$', title = 'Zonal Heat Flux', units = r'$J m^{-2} s^{-1}$')
# add to collection
variablePlotatts['qhu'] = tmp

## Meridional Wind
tmp = PlotAtts(name = 'v', title = 'Meridional Wind',units = 'm/s')
# add to collection
variablePlotatts['v'] = tmp

## Meridional Water (Vapor) Wind
tmp = PlotAtts(name = r'$q^w_v$', title = 'Meridional Water Flux', units = r'$kg m^{-2} s^{-1}$',
               scalefactor = 1e2)
# add to collection
variablePlotatts['qwv'] = tmp

## Meridional Heat Wind
tmp = PlotAtts(name = r'$q^h_v$', title = 'Meridional Heat Flux',units = r'$J m^{-2} s^{-1}$')
# add to collection
variablePlotatts['qhv'] = tmp

## Vertical Velocity
tmp = PlotAtts(name = 'w', title = 'Vertical Velocity',units = r'$m s^{-1}$')
# add to collection
variablePlotatts['w'] = tmp

## Pressure
tmp = PlotAtts(name = 'p', title = 'Pressure', units = 'hPa', scalefactor = 1e-2)
# add to collection
variablePlotatts['p'] = tmp

## Density (of dry air)
tmp = PlotAtts(name = 'rho', title = 'Density', units = r'$kg m^{-3}$')
# add to collection
variablePlotatts['rho'] = tmp

## Temperature
tmp = PlotAtts(name = 'T', title = 'Temperature', units = 'K')
# add to collection
variablePlotatts['T'] = tmp

## Dew-point Temperature
tmp = PlotAtts(name = r'T_d', title = 'Dew-point', units = 'K')
# add to collection
variablePlotatts['Td'] = tmp

## Temperature
tmp = PlotAtts(name = 'RH', title = 'Relative Humidity', units = r'%')
# add to collection
variablePlotatts['RH'] = tmp

## Potential Temperature
tmp = PlotAtts(name = 'theta', title = 'Potential Temperature', units = 'K')
# add to collection
variablePlotatts['th'] = tmp 

## Entropy
tmp = PlotAtts(name = 's', title = 'Entropy', units = r'$J kg^{-1} K^{-1}$')
# add to collection
variablePlotatts['s'] = tmp

## Lapse-rate (Temperature)
tmp = PlotAtts(name = 'lr', title = 'Lapse-rate', units = r'$K km^{-1}$', scalefactor = 1e3)
# add to collection
variablePlotatts['lr'] = tmp

## Potential Temperature Lapse-rate
tmp = PlotAtts(name = r'$\theta_z$', title = 'Theta Lapse-rate', 
               units = r'$K km^{-1}$', scalefactor = 1e3)
# add to collection
variablePlotatts['thlr'] = tmp

## Brunt-Vaeisaelae Frequency Squared
tmp = PlotAtts(name = r'$N^2$', title = r'$N^2$', 
               units = r'$10^{-4}s^{-2}$', scalefactor = 1e4)
# add to collection
variablePlotatts['N2'] = tmp

## Relative Vorticity (vertical component)
tmp = PlotAtts(name = r'$\zeta$', title = 'Relative Vorticity', 
               units = r'$10^{-4} s^{-1}$', scalefactor = 1e4)
# add to collection
variablePlotatts['ze'] = tmp

## Potential Vorticity
tmp = PlotAtts(name = 'PV', title = 'Potential Vorticity', 
               units = 'PVU', scalefactor = 1e6)
# add to collection
variablePlotatts['PV'] = tmp

## Potential Vorticity
tmp = PlotAtts(name = r'$PV_s$', title = 'Entropy Potential Vorticity', 
               units = r'$PVU_s$', scalefactor = 1e6)
# add to collection
variablePlotatts['PVs'] = tmp

## Vertical Potential Vorticity Gradient
tmp = PlotAtts(name = r'$dPV/dz$', title = 'Vertical Gradient of Potential Vorticity', 
               units = r'$PVU km^{-1}$', scalefactor = 1e9)
# add to collection
variablePlotatts['dPV'] = tmp

## Brunt-Vaeisaelae Frequency Squared
tmp = PlotAtts(name = r'$N^2$', title = r'Entropy Gradient $N^2$', 
               units = r'$10^{-4} s^{-2}$', scalefactor = g0/cp*1e4)
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
  if not isinstance(atts,(dict,NoneType)): raise TypeError(atts)
  if not isinstance(plot,(dict,PlotAtts,NoneType)): raise TypeError(plot)
  if plotatts_dict is None: plotatts_dict = variablePlotatts
  if not isinstance(plotatts_dict,dict): raise TypeError(plotatts_dict)
  # find variable in plotatts_dict (based on name)
  prefix = postfix = ''
  basename = name
  ldistvar = False # distribution variables are identified by postfix
  # get base plotatts atts
  if isinstance(plot,PlotAtts):
    plotatts = plot.copy() 
  elif name in plotatts_dict: 
    plotatts = plotatts_dict[name].copy()
  else:
    if name[:3].lower() in ('min','max'):
      prefix = name[:3].lower()
      basename = name[3:]
    if basename in plotatts_dict:
      plotatts = plotatts_dict[basename].copy()
    elif basename.lower() in plotatts_dict:
      plotatts = plotatts_dict[basename.lower()].copy()
    else:
      namelist = basename.split('_')
      basename = namelist[0]
      postfix = namelist[1] if len(namelist)>1 else ''
      if postfix in plotatts_dict:
        ldistvar = True
        plotatts = plotatts_dict[postfix].copy()
      elif postfix.lower() in plotatts_dict:
        ldistvar = True
        plotatts = plotatts_dict[postfix.lower()].copy()
      elif basename in plotatts_dict: 
        plotatts = plotatts_dict[basename].copy()
      elif basename.lower() in plotatts_dict: 
        plotatts = plotatts_dict[basename.lower()].copy()
      else:
        # last resort...
        name = name.title() if name.islower() else name
        if '^' in units or '{' in units and '$' not in units: units = '$'+units+'$'
        plotatts = PlotAtts(name=name, units=units, title=name)  
  # intercept variables that express relative change 
  # i.e. display var1 / var0 - 1 as +/--percentage
  if units == '':
      if isinstance(atts,dict): 
          binop_name = atts.get('binop_name',name)
      else: binop_name = name
      binop_name = binop_name.split()
      if len(binop_name) >= 4: # needs at least 4 components: var1 / var2 -1
        if (binop_name[-1] == '-1' or ( binop_name[-2] == '-' and binop_name[-1] == '1' ) ):
          if '/' in binop_name:
            name = plotatts.name; title = plot.title
            name = name.title() if name.islower() else name
            title = title.title() if title.islower() else title
            plotatts = PlotAtts(name = '{:s} Ratio'.format(name), 
                                title = 'Relative {:s} Differences'.format(title), 
                                units = '$\%$', scalefactor = 1e2)
  # modify according to variable specifics
  if len(postfix) > 0: # these are mostly things like, e.g., '7d' for 7d means
    if ldistvar: 
      # N.B.: this is for distribution objects - we usually cast these into a proper variable before plotting 
      tmpname = basename if basename == basename.upper() else basename.title() 
      plotatts = plotatts.copy(name = '{:s} '.format(tmpname)+plotatts.name,
                               title = '{:s} '.format(tmpname)+plotatts.title)
    else:
      # N.B.: this is the usualy path for variables we are actually plotting
      tmpname = plotatts.name if plotatts.name == plotatts.name.upper() else plotatts.name.title()
      if tmpname in ('CWD','CDD'):
        postfix = '{:2.0f}'.format(int(postfix)/10.) if int(postfix) >= 100 else '{:2.1f}'.format(int(postfix)/10.) 
        if not atts is None and 'long_name' in atts: title = atts['long_name'] 
        else: title = plotatts.title+' ({:s})'.format(postfix)
      else: title = plotatts.title+' ({:s})'.format(postfix)
      plotatts = plotatts.copy(name = tmpname+' ({:s})'.format(postfix), title = title)
  if prefix == 'max': 
    plotatts = plotatts.copy(name = 'Max. '+plotatts.name, title = 'Maximum '+plotatts.title) 
  elif prefix == 'min': 
    plotatts = plotatts.copy(name = 'Min. '+plotatts.name, title = 'Minimum '+plotatts.title)
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
  if plot is None: plot = variablePlotatts.copy()
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
  