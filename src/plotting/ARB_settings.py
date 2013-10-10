'''
Created on 2013-10-10

Meta data related to the Athabasca River Basin downscaling project; primarily WRF settings. 

@author: Andre R. Erler, GPL v3
'''


# data root folder
from socket import gethostname
hostname = gethostname()
if hostname=='erlkoenig':
  WRFroot = '/home/me/DATA/WRF/'
elif hostname=='komputer':
  WRFroot = '/home/DATA/DATA/WRF/' + 'Downscaling/'

## Project Parameters
projRoot = WRFroot # + 'Downscaling/' # the project root folder
# Experiments: define non-standard folders 
experiment = dict() # dictionary of experiments
# hitop
experiment['hitop-ctrl'] =  projRoot + 'hitop-ctrl'
experiment['nofdda-ctrl'] =  projRoot + 'hitop-test/nofdda-ctrl'
experiment['nofdda-hitop'] =  projRoot + 'hitop-test/nofdda-hitop'
experiment['hitop-old'] =  projRoot + 'hitop-test/hitop-ctrl'
## shorthands for common experiments
# proper names
WRFname = dict()
# these are all based on the "new" configuration (ARB3 domain)
WRFname['new'] = 'new-ctrl'
WRFname['nogulf'] = 'new-nogulf' # ARB2 domain
WRFname['noah'] = 'new-noah' # ARB2 domain
WRFname['noah35'] = 'v35-noah' # ARB2 domain
# these are all based on the "max" configuration
WRFname['max'] = 'max-ctrl'
WRFname['max-2050'] = 'max-ctrl-2050'
# these are all based on the old configuration (original + RRTMG, ARB2)
WRFname['ctrl-1'] = 'ctrl-1'
WRFname['tiedt'] = 'tiedtke-ctrl'
WRFname['tom'] = 'tom-ctrl'
WRFname['wdm6'] = 'wdm6-ctrl'
WRFname['milb'] = 'milbrandt-ctrl'
WRFname['nmpsnw'] = 'nmpsnw-ctrl'
WRFname['nmpbar'] = 'nmpbar-ctrl'
# these are all based on the original configuration (mostly ARB1 domain)
WRFname['cam3'] = 'cam-ctrl'
WRFname['pbl4'] = 'pbl1-arb1'
WRFname['grell'] = 'grell3-arb1'
WRFname['moris'] = 'moris-ctrl'
WRFname['nmpdef'] = 'noahmp-arb1'
WRFname['clm4'] = 'v35-clm'
WRFname['pwrf'] = 'polar-arb1'
WRFname['modis'] = 'modis-arb1'
WRFname['cfsr'] = 'cfsr-cam'
WRFname['ens-Z'] = 'cam-ctrl'
WRFname['ens-A'] = 'cam-ens-A'
WRFname['ens-B'] = 'cam-ens-B'
WRFname['ens-C'] = 'cam-ens-C'
WRFname['ctrl-1-2000'] = 'cam-ctrl'
WRFname['ctrl-1-2050'] = 'cam-ctrl-1-2050' #'ctrl-2-2050'
WRFname['ctrl-1-2100'] = 'cam-ctrl-2-2100'
WRFname['ctrl-2-2050'] = 'cam-ctrl-2-2050' #'ctrl-arb1-2050'
# titles (alphabetical order)
WRFtitle = dict()
WRFtitle['cam3'] = 'CAM3 Radiation'
WRFtitle['cfsr'] = 'CFSR Forcing (CAM3)'
WRFtitle['clm4'] = 'CLM-4'
WRFtitle['ctrl-1'] = 'WRF Control'
WRFtitle['ctrl-1-2000'] = 'Ctrl-1 Historical'
WRFtitle['ctrl-1-2050'] = 'Ctrl-1 2050'
WRFtitle['ctrl-1-2100'] = 'Ctrl-1 2100'
WRFtitle['ctrl-2-2050'] = 'Ctrl-2 2050'
WRFtitle['ens-Z'] = 'WRF Control (CAM3)'
WRFtitle['ens-A'] = 'Ensemble A'
WRFtitle['ens-B'] = 'Ensemble B'
WRFtitle['ens-C'] = 'Ensemble C'
WRFtitle['grell'] = 'Grell-3 Cumulus (CAM3)'
WRFtitle['max'] = 'Morrison & Grell-3'
WRFtitle['milb'] = 'Milbrandt-Yau MP'
WRFtitle['modis'] = 'MODIS Land Classes (CAM3)'
WRFtitle['moris'] = 'Morrison MP (CAM3)'
WRFtitle['new'] = 'New Configuration'
WRFtitle['nmpbar'] = 'Noah-MP V3'
WRFtitle['nmpdef'] = 'Noah-MP V1 (CAM3)'
WRFtitle['nmpsnw'] = 'Noah-MP V2 (CAM3)'
WRFtitle['noah'] = 'New Config (with Noah LSM)'
WRFtitle['noah35'] = 'New Config (V3.5, Noah LSM)'
WRFtitle['nogulf'] = 'New Config (old domain)'
WRFtitle['pbl4'] = 'QNSE PBL (CAM3)'
WRFtitle['pwrf'] = 'Polar WRF (CAM3)'
WRFtitle['tiedt'] = 'Tiedtke Cumulus'
WRFtitle['tom'] = 'Thompson MP'
WRFtitle['wdm6'] = 'WDM-6 MP'

## setup projection: lambert conformal
def getProjectionSettings(projtype):
  # lon_0,lat_0 is central point. lat_ts is latitude of true scale.
  if projtype == 'lcc-new':
    ## Lambert Conic Conformal - New Fine Domain
    projection = dict(projection='lcc', lat_0=55, lon_0=-120, lat_1=52, rsphere=(6378137.00,6356752.3142),#
                width=180*10e3, height=180*10e3, area_thresh = 1000., resolution='l')
  elif projtype == 'lcc-fine':
    ## Lambert Conic Conformal - Fine Domain
    projection = dict(projection='lcc', lat_0=58, lon_0=-132, lat_1=53, rsphere=(6378137.00,6356752.3142),#
                width=200*10e3, height=300*10e3, area_thresh = 1000., resolution='l')
  elif projtype == 'lcc-small':
    ## Lambert Conic Conformal - Small Domain
    projection = dict(projection='lcc', lat_0=56, lon_0=-130, lat_1=53, rsphere=(6378137.00,6356752.3142),#
                width=2500e3, height=2650e3, area_thresh = 1000., resolution='l')
  elif projtype == 'lcc-intermed':
    ## Lambert Conic Conformal - Intermed Domain
    projection = dict(projection='lcc', lat_0=57, lon_0=-140, lat_1=53, rsphere=(6378137.00,6356752.3142),#
                width=4000e3, height=3400e3, area_thresh = 1000., resolution='l')
  elif projtype == 'lcc-large':
    ## Lambert Conic Conformal - Large Domain
    projection = dict(projection='lcc', lat_0=54.5, lon_0=-140, lat_1=53, #rsphere=(6378137.00,6356752.3142),#
                width=11000e3, height=7500e3, area_thresh = 10e3, resolution='l')
  elif projtype == 'laea': 
    ## Lambert Azimuthal Equal Area
    projection = dict(projection='laea', lat_0=57, lon_0=-137, lat_ts=53, resolution='l', #
                width=259*30e3, height=179*30e3, rsphere=(6378137.00,6356752.3142), area_thresh = 1000.)  
  elif projtype == 'ortho-NA':
    ## Orthographic Projection
    projection = dict(projection='ortho', lat_0 = 75, lon_0 = -137, resolution = 'l', area_thresh = 1000.)
  # resolution of coast lines
  grid = 10; res = projection['resolution']
  # return values
  return projection, grid, res


if __name__ == '__main__':
    pass