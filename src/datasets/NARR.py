'''
Created on 2013-09-08

This module contains meta data and access functions for NARR datasets. 

@author: Andre R. Erler, GPL v3
'''

# from atmdyn.properties import variablePlotatts
from geodata.netcdf import NetCDFDataset
from geodata.gdal import GDALDataset, getProjFromDict 

## NARR Climatology (LTM - Long Term Mean)

# projection
projdict = dict(proj = 'lcc', # Lambert Conformal Conic  
                lat1 =   50., # Latitude of first standard parallel
                lat2 =   50., # Latitude of second standard parallel
                lat0 =   50., # Latitude of natural origin
                lon0 = -107., # Longitude of natural origin
                x0   = 5632642.22547, # False Origin Easting
                y0   = 4612545.65137) # False Origin Northing

# variable attributes and name
varatts = dict(air   = dict(name='T2', units='K'), # 2m Temperature
                   prate = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)
                   prmsl = dict(name='pmsl', units='Pa'), # sea-level pressure
                   lon   = dict(name='lon', units='deg E'), # geographic longitude field
                   lat   = dict(name='lat', units='deg N'), # geographic latitude field
                   x     = dict(name='x', units='m', offset=-1*projdict['x0']), # projected west-east coordinate
                   y     = dict(name='y', units='m', offset=-1*projdict['y0'])) # projected south-north coordinate

# variable and file lists
folder = '/home/DATA/DATA/NARR/LTM/'
pfx = '.mon.ltm.nc'
filelist = dict(air='air.2m',prate='prate',prmsl='prmsl')
filelist = {key:value+pfx for key,value in filelist.iteritems()} # add postfix
varlist = filelist.keys() + ['lon','lat']

def loadNARRclim(varlist=varlist, varatts=varatts, projection=projdict, filelist=filelist, folder=folder):
  ''' Function to return a properly formatted NARR data set. '''
  # prepare input
  if varlist is None: 
    varlist = filelist.keys()
    filelist=filelist.values()
  else: 
    filelist = [filelist[var] for var in varlist if var in filelist] 
  # load dataset
  dataset = NetCDFDataset(folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, atts=projdict, multifile=False, ncformat='NETCDF4_CLASSIC')
  # add projection
  projection = getProjFromDict(projdict, name='NARR Coordinate System')
  dataset = GDALDataset(dataset, projection=projection, geotransform=None)
  # return formatted dataset
  return dataset


## (ab)use main execution for quick test
if __name__ == '__main__':
    
    # load dataset
    dataset = loadNARRclim()
    
    # print dataset
    print(dataset)