'''
Created on 2013-10-21

Script to produce climatology files from monthly mean time-series' for all or a subset of available WRF experiments.

@author: Andre R. Erler, GPL v3
'''

# external
import numpy as np
import os
import multiprocessing # parallelization
# internal
from geodata.base import Axis
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import GridDefinition
from geodata.process import CentralProcessingUnit, DateError
from geodata.misc import isInt
from datasets.common import name_of_month, days_per_month
# WRF specific
from datasets.WRF import loadWRF_TS, fileclasses, root_folder
from plotting.ARB_settings import WRFname


def computeClimatology(pid, experiment, filetype, domain, periods=None, offset=0, griddef=None):
  ''' worker function to compute climatologies for given file parameters. '''
  # input type checks
  if not isinstance(experiment,basestring): raise TypeError
  if not isinstance(filetype,basestring): raise TypeError
  if not isinstance(domain,(np.integer,int)): raise TypeError
  if periods is not None and not (isinstance(periods,(tuple,list)) and isInt(periods)): raise TypeError
  if not isinstance(offset,(np.integer,int)): raise TypeError  
  if griddef is not None and not isinstance(griddef,GridDefinition): raise TypeError
  # parallelism
  if not isinstance(pid,(np.integer,int)): raise TypeError
  pidstr = '' if pid < 0 else  '[proc%02i]'%pid # pid for parallel mode output
  
  # load source
  fileclass = fileclasses[filetype] # used for target file name
  print('\n%s   ***   Experiment: \'%s\'   ***   '%(pidstr,experiment) +
        '\n%s   ***   \'%s\'   ***   '%(pidstr,fileclass.tsfile.format(domain)))
  source = loadWRF_TS(experiment=experiment, filetypes=[filetype], domains=domain) # comes out as a tuple...
  if pidstr == '': 
    print(''); print(source); print('')
  
  # figure out start date
  filebegin = int(source.atts.begin_date.split('-')[0]) # first element is the year
  fileend = int(source.atts.end_date.split('-')[0]) # first element is the year
  begindate = offset + filebegin
  if not ( filebegin <= begindate <= fileend ): raise DateError  
  
  ## loop over periods
  if periods is None: periods = [begindate-fileend]
  for period in periods:       
            
    # figure out period
    enddate = begindate + period     
    if filebegin > enddate: raise DateError    
    if enddate > fileend: 
      print('\n%s   ---   Invalid Period: End Date %4i not in File!   ---   \n'%(pidstr,enddate))
      
    else:  
      ## begin actual computation
      periodstr = '%4i-%4i'%(begindate,enddate)
      avgfolder = root_folder + experiment + '/'
      print('\n%s   <<<   Processing Grid %s from %s   >>>   \n'%(pidstr,grid,periodstr))              
      
      # prepare sink
      gridstr = '' if griddef is None or griddef.name is 'WRF' else '_'+griddef.name
      filename = fileclass.climfile.format(domain,gridstr,'_'+periodstr)
      assert os.path.exists(avgfolder)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      sink = DatasetNetCDF(name='WRF Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
      sink.atts.period = periodstr 
      
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, varlist=varlist, tmp=True) # no need for lat/lon
      
      # start processing climatology
      CPU.Climatology(period=period, offset=offset, flush=False)
      
      # reproject and resample (regrid) dataset
      if griddef is not None:
        CPU.Regrid(griddef=griddef, flush=False)
        print('%s    ---   (%3.2f,  %3i x %3i)   ---   \n'%(pidstr, dlon, len(lon), len(lat)))      
      
      # sync temporary storage with output
      CPU.sync(flush=True)
      
      # make new masks
      #sink.mask(sink.landmask, maskSelf=False, varlist=['snow','snowh','zs'], invert=True, merge=False)
      
      # add names and length of months
      sink.axisAnnotation('name_of_month', name_of_month, 'time', 
                          atts=dict(name='name_of_month', units='', long_name='Name of the Month'))
      sink += VarNC(sink.dataset, name='length_of_month', units='days', axes=(sink.time,), data=days_per_month,
                    atts=dict(name='length_of_month',units='days',long_name='Length of Month'))
      
      # close... and write results to file
      print('\n%s Writing to: \'%s\'\n'%(pidstr,filename))
      sink.sync()
      sink.close()
      # print dataset
      if pidstr == '':
        print(''); print(sink); print('')     

if __name__ == '__main__':
  
  NP = 4
  
  # defaults
  varlist = None # ['precip', 'T2']
  experiments = [] # WRF experiment names (passed through WRFname)
  periods = [5,10] # averaging period
  domains = [1,2] # domains to be processed
  filetypes = ['srfc','xtrm','plev3d','hydro',] # filetypes to be processed
#   filetypes = ['hydro',] # filetypes to be processed
  grid = 'WRF' 
  # experiments
#   experiments = ['gulf']; periods = [1]

  # expand experiments 
  if len(experiments) > 0: experiments = [WRFname[exp] for exp in experiments]
  else: experiments = [exp for exp in WRFname.values()]    

  ## do some fancy regridding
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
    griddef = GridDefinition(name=grid, projection=None, xlon=xlon, ylat=ylat) # projection=None >> lat/lon
  else:
    griddef = None
  
  args = []
  # generate list of parameters
  for experiment in experiments:    
    # loop over file types
    for filetype in filetypes:                
      # effectively, loop over domains
      for domain in domains:
        # call worker function
        args.append((experiment, filetype, domain))

  ## loop over and process all job sets
  if NP is not None and NP == 1:
    # don't parallelize, if there is only one process: just loop over files    
    for pid,arg in enumerate(args): # negative pid means serial mode
      experiment, filetype, domain = arg
      computeClimatology(pid, experiment, filetype, domain, periods=periods, griddef=griddef)
  else:
    if NP is None: pool = multiprocessing.Pool() 
    else: pool = multiprocessing.Pool(processes=NP)
    # distribute tasks to workers
    for pid,arg in enumerate(args): # negative pid means serial mode
      experiment, filetype, domain = arg      
      pool.apply_async(computeClimatology, (pid, experiment, filetype, domain), dict(periods=periods, griddef=griddef))
    pool.close()
    pool.join()
  print('')
        
  # loop over jobs
