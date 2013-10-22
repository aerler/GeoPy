'''
Created on 2013-10-21

Script to produce climatology files from monthly mean time-series' for all or a subset of available WRF experiments.

@author: Andre R. Erler, GPL v3
'''

# external
import numpy as np
import os
# internal
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.process import CentralProcessingUnit
from datasets.common import name_of_month, days_per_month
# WRF specific
from datasets.WRF import loadWRF_TS, fileclasses, root_folder
from plotting.ARB_settings import WRFname

#TODO: add date checks and option to overwrite or skip existing files
#TODO: wrap computation into a function for easy parallelization
#TODO: bundle experiments, filetypes, and maybe domains and parallelize using 'pool'

if __name__ == '__main__':
  
  # defaults
  varlist = None # ['precip', 'T2']
  experiments = [] # WRF experiment names (passed through WRFname
  period = (1979,1981) # averaging period
  domains = [1,2] # domains to be processed
  filetypes = ['srfc','xtrm','plev3d','hydro',] # filetypes to be processed
  grid = 'WRF' 
  # experiments
#   experiments = ['max']; period = (1979,1989)
#   experiments = ['max-2050']; period = (2045,2055)

  # inferred settings 
  startdate = period[0] # when the data record starts
#TODO: startdate should actually be read from the netcdf file!
  if len(experiments) > 0: experiments = [WRFname[exp] for exp in experiments]
  else: experiments = [exp for exp in WRFname.values()]
  # determine coordinate arrays
  if grid != 'WRF':    
    if grid == '025': dlon = dlat = 0.25 # resolution
    elif grid == '05': dlon = dlat = 0.5
    elif grid == '10': dlon = dlat = 1.0
    elif grid == '25': dlon = dlat = 2.5 
    slon, slat, elon, elat = -179.75, 3.25, -69.75, 85.75
    assert (elon-slon) % dlon == 0 
    lon = np.linspace(slon+dlon/2,elon-dlon/2,(elon-slon)/dlon)
    assert (elat-slat) % dlat == 0
    lat = np.linspace(slat+dlat/2,elat-dlat/2,(elat-slat)/dlat)
    # add new geographic coordinate axes for projected map
    xlon = Axis(coord=lon, atts=dict(name='lon', long_name='longitude', units='deg E'))
    ylat = Axis(coord=lat, atts=dict(name='lat', long_name='latitude', units='deg N'))
  else:
    xlon = None; ylat = None
  
  ## loop over experiments and domains
  for experiment in experiments:
    
    # loop over file types
    for filetype in filetypes:    

      fileclass = fileclasses[filetype] # used for target file name
      
      # in case anything goes wrong, just go to the next one...
      try:
        sources = loadWRF_TS(experiment=experiment, filetypes=[filetype], domains=domains) # comes out as a tuple...
        
        # effectively, loop over domains
        for domain,source in zip(domains,sources):
            
          # begin (loop over files)
          periodstr = '%4i-%4i'%period
          avgfolder = root_folder + experiment + '/'
          print('\n')
          print('   ***   Processing Grid %s from %s   ***   '%(grid,periodstr))              
          
          # load source
          print('       Source: \'%s\'\n'%fileclass.tsfile.format(domain))
          print(source)
          print('\n')
          # prepare sink
          gridstr = '' if grid is 'WRF' else '_'+grid
          filename = fileclass.climfile.format(domain,gridstr,'_'+periodstr)
          if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
          assert os.path.exists(avgfolder)
          sink = DatasetNetCDF(name='WRF Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
          sink.atts.period = periodstr 
          
          # determine averaging interval
          offset = source.time.getIndex(period[0]-startdate)/12 # origin of monthly time-series is at January 1979 
          # initialize processing
          CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=True) # no need for lat/lon
          
          # start processing climatology
          CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
          
          # reproject and resample (regrid) dataset
          if xlon is not None and ylat is not None:
            CPU.Regrid(xlon=xlon, ylat=ylat, flush=False)
            print('    ---   (%3.2f,  %3i x %3i)   ---   '%(dlon, len(lon), len(lat)))
          
          
          # sync temporary storage with output
          CPU.sync(flush=True)
      
          # make new masks
          #sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=True, merge=False)
      
          # add names and length of months
          sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                              atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
          #print '   ===   month   ===   '
          sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                        atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
          
          # close... and write results to file
          print('\n Writing to: \'%s\'\n'%filename)
          sink.sync()
          sink.close()
          # print dataset
          print('')
          print(sink)     
      
      # for try at line 65
      except: pass