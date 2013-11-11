'''
Created on 2013-06-06, adapted on 2013-09-09

This module contains meta data and access functions for the PRISM climatology, as well as
functionality to read PRISM data from ASCII files and write to NetCDF format.

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import numpy.ma as ma
import netCDF4 as nc # netcdf python module
# internal imports
from datasets.common import days_per_month, name_of_month, data_root, loadClim
from geodata.misc import DatasetError
from warnings import warn
from geodata.gdal import GridDefinition

## PRISM Meta-data

dataset_name = 'PRISM'

# PRISM grid definition
dlat = dlon = 1./24. #  0.041666666667
dlat2 = dlon2 = 1./48. #  0.02083333333331
nlat = 601 # slat = 25 deg
nlon = 697 # slon = 29 deg
# N.B.: coordinates refer to grid points (CF convention), commented values refer to box edges (GDAL convention) 
llclat = 47. # 46.979166666667
# urclat = 72.; urclon = -113.
llclon = -142. # -142.020833333333
           
geotransform = (llclon-dlon2, dlon, 0.0, llclat-dlat2, 0.0, dlat)
size = (nlon,nlat) # (x,y) map size of PRISM grid
# make GridDefinition instance
PRISM_grid = GridDefinition(name=dataset_name, projection=None, geotransform=geotransform, size=size)

# variable attributes and name (basically no alterations necessary...)
varatts = dict(T2 = dict(name='T2', units='K', atts=dict(long_name='Average 2m Temperature')), # 2m average temperature
               Tmin = dict(name='Tmin', units='K', atts=dict(long_name='Minimum 2m Temperature')), # 2m minimum temperature
               Tmax = dict(name='Tmax', units='K', atts=dict(long_name='Maximum 2m Temperature')), # 2m maximum temperature
               precip = dict(name='precip', units='kg/m^2/s', atts=dict(long_name='Total Precipitation')), # total precipitation
               # axes (don't have their own file; listed in axes)
               time=dict(name='time', units='month', atts=dict(long_name='Month of the Year')), # time coordinate
               lon  = dict(name='lon', units='deg E', atts=dict(long_name='Longitude')), # geographic longitude field
               lat  = dict(name='lat', units='deg N', atts=dict(long_name='Latitude'))) # geographic latitude field
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    

# variable and file lists settings
root_folder = data_root + dataset_name + '/' # long-term mean folder


## Functions that provide access to well-formatted PRISM NetCDF files

# pre-processed climatology files (varatts etc. should not be necessary)
avgfile = 'prism{0:s}_clim{1:s}.nc' # formatted NetCDF file
avgfolder = root_folder + 'prismavg/' # prefix
# function to load these files...
def loadPRISM(name=dataset_name, period=None, grid=None, varlist=None, varatts=None, folder=avgfolder, filelist=None):
  ''' Get the pre-processed monthly PRISM climatology as a DatasetNetCDF. '''
  # only the climatology is available
  if period is not None: 
    warn('Only the full climatology is currently available: setting \'period\' to None.')
    period = None
  # load standardized climatology dataset with PRISM-specific parameters  
  dataset = loadClim(name=name, folder=folder, projection=None, period=period, grid=grid, varlist=varlist, 
                     varatts=varatts, filepattern=avgfile, filelist=filelist)
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
file_pattern = avgfile # filename pattern
data_folder = avgfolder # folder for user data
grid_def = {'':PRISM_grid} # no special name, since there is only one...
grid_res = {'':0.04} # approximate resolution in degrees at 45 degrees latitude
default_grid = PRISM_grid
# grid_def = {0.04:PRISM_grid} # standardized grid dictionary, addressed by grid resolution
# grid_tag = {0.04:''} # no special name, since there is only one...
# functions to access specific datasets
loadLongTermMean = None # climatology provided by publisher
loadTimeSeries = None # time-series data
loadClimatology = loadPRISM # pre-processed, standardized climatology

## Functions that handle access to PRISM ASCII files

# loads data from original ASCII files and returns numpy arrays
# data is assumed to be stored in monthly intervals
def loadASCII(var, fileformat='BCY_%s.%02ia', arrayshape=(601,697)):
  # local imports
  from numpy.ma import zeros
  from numpy import genfromtxt, flipud
  # definitions
  datadir = root_folder + 'Climatology/ASCII/' # data folder   
  ntime = len(days_per_month) # number of month
  # allocate space
  data = zeros((ntime,)+arrayshape) # time = ntime, (x, y) = arrayshape  
  # loop over month
  print('  Loading variable %s from file.'%(var))
  for m in xrange(ntime):
    # read data into array
    filename = fileformat%(var,m+1)
    tmp = genfromtxt(datadir+filename, dtype=float, skip_header=5, 
                     missing_values=-9999, filling_values=-9999, usemask=True)
    data[m,:] = flipud(tmp)  
    # N.B.: the data is loaded in a masked array (where missing values are omitted)   
  # return array
  return data

# function to generate lat/lon coordinate axes for the data set
# the data is assumed to be in a simple lat/lon projection
def genCoord():
  # imports
  from numpy import diff, finfo, arange # linspace 
  eps = finfo(float).eps
  # N.B.: PRISM meta data is defined in the header of this file  
  # generate coordinate arrays
  lat = llclat + arange(nlat)*dlat # + dlat/2.
  assert (diff(lat).mean() - dlat) < eps # sanity check
  lon = llclon + arange(nlon) *dlon #  + dlon/2.
  assert (diff(lon).mean() - dlon) < eps # sanity check  
  # return coordinate arrays (in degree)
  return lat, lon

if __name__ == '__main__':
    
  mode = 'test_climatology'
#   mode = 'convert_ASCII'
  
  # do some tests
  if mode == 'test_climatology':  
    
    # load NetCDF dataset
    dataset = loadPRISM()
    print(dataset)
    print('')
    print(dataset.geotransform)
    print(dataset.precip.masked)
    print(dataset.precip.getArray().mean())
    
  ## convert ASCII files to NetCDF
  elif mode == 'convert_ASCII': 
    
    # import netcdf tools
    from geodata.nctools import add_coord, add_var, add_strvar
    
    ## load data
    data = dict()
    
    # read precip data        
    ppt = loadASCII('Ppt')
    # rescale data (divide by 100 * days per month * seconds per day)
    ppt /= days_per_month.reshape((len(days_per_month),1,1)) * 8640000. # convert to kg/m^2/s
    # print diagnostic
    print('Mean Precipitation: %f kg/m^2/s'%ppt.mean())
    data['precip'] = ppt
    
    # read temperature data
    # rescale (multiply by 100) and convert to Kelvin
    Tmin = loadASCII('Tmin'); Tmin /= 100.; Tmin += 273.15
    Tmax = loadASCII('Tmax'); Tmax /= 100.; Tmax += 273.15    
#     Tavg = loadASCII('Tavg'); Tavg /= 100.; Tavg += 273.15
    T2 = ( Tmin + Tmax ) / 2. # temporary solution for Tavg, because the data seems to be corrupted
    # print diagnostic
    print('Min/Mean/Max Temperature: %3.1f / %3.1f / %3.1f K'%(Tmin.mean(),T2.mean(),Tmax.mean()))
    data['T2'] = T2; data['Tmin'] = Tmin; data['Tmax'] = Tmax
    
    # get coordinate axes
    lat, lon = genCoord()
    
#     # display
#     import pylab as pyl
#     pyl.pcolormesh(lon, lat, ppt.mean(axis=0))
#     pyl.colorbar()
#     pyl.show(block=True)

    ## create NetCDF file
    
    # initialize netcdf dataset structure
    outfile = avgfolder+avgfile
    print('\nWriting data to disk: %s'%outfile)
    # create groups for different resolution
    outdata = nc.Dataset(outfile, 'w', format='NETCDF4') # outgrp.createGroup('fineres')
    # global attributes
    outdata.title = 'Climatology of Monthly PRISM Data'
    outdata.creator = 'Andre R. Erler' 
    
    # create ncatts dictionary from varatts
    ncatts = dict()
    for var,atts in varatts.iteritems():
      newatts = dict() # new atts dictionary
      # and iterate over atts dict contents
      for key1,val1 in atts.iteritems():
        if isinstance(val1,dict): # flatten nested dicts
          for key2,val2 in val1.iteritems():
            newatts[key2] = val2 
        else: newatts[key1] = val1
      ncatts[var] = newatts      
    
    # create time dimensions and coordinate variables
    add_coord(outdata, 'time', data=np.arange(1,len(days_per_month)+1), dtype='i4', atts=ncatts['time'])
    add_var(outdata, 'length_of_month', ['time'], data=days_per_month, 
            atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
    # names of months (as char array)
    add_strvar(outdata, 'name_of_month', name_of_month, 'time', 
               atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
    
    # create new lat/lon dimensions and coordinate variables
    add_coord(outdata, 'lat', data=lat, atts=ncatts['lon'])
    add_coord(outdata, 'lon', data=lon, atts=ncatts['lat'])
    # create climatology variables  
    fillValue = -9999; axes = ('time','lat','lon')
    for name,field in data.iteritems():
      add_var(outdata, name, axes, data=field.filled(fillValue), atts=ncatts[name], fillValue=fillValue)
    # create land/sea/no-data mask
    mask = ma.getmaskarray(data['precip'])[0,:,:]
    atts = dict(name='datamask', units='', long_name='Mask for Climatology Fields', 
                description='where this mask is non-zero, no data is available')
    add_var(outdata, 'datamask', ['lat','lon'], data=mask, atts=atts)
    
    # dataset feedback and diagnostics
    # print dataset meta data
    print outdata
    # print dimensions meta data
    for dimobj in outdata.dimensions.values():
      print dimobj
    # print variable meta data
    for varobj in outdata.variables.values():
      print varobj
    
    # close netcdf files  
    outdata.close() # output
