'''
Created on 2010-11-26, adapted on 2013-08-24

Some default physical constants (mostly used in meteoVar).

@author: Andre R. Erler, GPL v3
'''

from numpy import pi, sin

# actual constants
R = 8.31447215 # J/(mol K), universal gas constant (Wikipedia)
cp = 1005.7 # J/(kg K), specific heat of dry air per mass (AMS Glossary)
g0 = 9.80665 # m/s**2, for geopotential altitude (else actually y-dependent g(y))
Mair = 0.0289644 # kg/mol, Molecular mass of dry air
Re = 6371229 # m, Radius of planet earth
T0 = 273.15 # K, Temperature at 0 deg C, i.e. negative absolute zero in Celsius
Omega = 2*pi/((23*60+56)*60+4.1) # 1/s, Earth's rotation rate (using siderial day)
# some derived constants, for convenience
Cp = cp*Mair # J/(mol K), specific heat of dry air per mole
Rd = R/Mair # gas constant for dry air
kappa = R/Cp # ~7/2, adiabatic exponent for dry air
# not exactly physical constants
fc = 2*Omega*sin(pi/4) # Coriolis parameter at 45 deg N
p0 = 1e5 # reference pressure (e.g. for potential temperature)
