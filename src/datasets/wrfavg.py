'''
Created on 2013-10-21

Script to produce climatology files from monthly mean time-series' for all or a subset of available WRF experiments.

@author: Andre R. Erler, GPL v3
'''

# external
import numpy as np
import os
import multiprocessing # parallelization
import logging # used to control error output of sub-processes
from datetime import datetime
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


def computeClimatology(experiment, filetype, domain, lparallel=None, periods=None, offset=0, griddef=None):
  ''' worker function to compute climatologies for given file parameters. '''
  # input type checks
  if not isinstance(experiment,basestring): raise TypeError
  if not isinstance(filetype,basestring): raise TypeError
  if not isinstance(domain,(np.integer,int)): raise TypeError
  if periods is not None and not (isinstance(periods,(tuple,list)) and isInt(periods)): raise TypeError
  if not isinstance(offset,(np.integer,int)): raise TypeError  
  if griddef is not None and not isinstance(griddef,GridDefinition): raise TypeError
  if not isinstance(lparallel,(bool,np.bool)): raise TypeError 
  # parallelism
  if lparallel:
    pid = int(multiprocessing.current_process().name.split('-')[-1]) # start at 1
    pidstr = '[proc%02i]'%pid # pid for parallel mode output  
  else:
    pidstr = '' # don't print process ID, sicne there is only one
  
  ## start actual work: this is inclosed in a try-block, so errors don't 
  try:
    
    #if pid == 1: raise Exception # to test error handling
  
    # load source
    fileclass = fileclasses[filetype] # used for target file name
    print('\n\n{0:s}   ***   Processing Experiment {1:>10s}   ***   '.format(pidstr,"'%s'"%experiment) +
          '\n{0:s}   ***   {1:^32s}   ***   \n'.format(pidstr,"'%s'"%fileclass.tsfile.format(domain)))
    source = loadWRF_TS(experiment=experiment, filetypes=[filetype], domains=domain) # comes out as a tuple...
    if not lparallel: 
      print(''); print(source); print('')
    # determine age of oldest source file
    if not loverwrite:
      sourceage = datetime.today()
      for filename in source.filelist:
        age = datetime.fromtimestamp(os.path.getmtime(filename))
        sourceage = age if age < sourceage else sourceage
    
    # figure out start date
    filebegin = int(source.atts.begin_date.split('-')[0]) # first element is the year
    fileend = int(source.atts.end_date.split('-')[0]) # first element is the year
    begindate = offset + filebegin
    if not ( filebegin <= begindate <= fileend ): raise DateError  
    
    ## loop over periods
    if periods is None: periods = [begindate-fileend]
  #   periods.sort(reverse=True) # reverse, so that largest chunk is done first
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
        print('\n%s   <<<   Computing Climatology from %s on %s grid  >>>   \n'%(pidstr,periodstr,grid))              
  
        # determine if sink file already exists, and what to do about it      
        gridstr = '' if griddef is None or griddef.name is 'WRF' else '_'+griddef.name
        filename = fileclass.climfile.format(domain,gridstr,'_'+periodstr)
        if ldebug: filename = 'test_' + filename
        assert os.path.exists(avgfolder)
        lskip = False # else just go ahead
        if os.path.exists(avgfolder+filename): 
          if not loverwrite: 
            age = datetime.fromtimestamp(os.path.getmtime(avgfolder+filename))
            # if sink file is newer than source file, skip (do not recompute)
            if age > sourceage: lskip = True
            #print sourceage, age
          if not lskip: os.remove(avgfolder+filename) 
        
        # depending on last modification time of file or overwrite setting, start computation, or skip
        if lskip:        
          # print message
          print('%s   >>>   Skipping: File \'%s\' already exists and is newer than source file.   <<<   \n'%(pidstr,filename))              
        else:
           
          # prepare sink
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
          
          # sync temporary storage with output dataset (sink)
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
          if not lparallel:
            print(''); print(sink); print('')   
             
    # return exit code
    return 0 # everything OK
  except Exception:
    # an error occurred
    if ldebug: raise # raise error
    else: return 1 # report error 

if __name__ == '__main__':
  
  ## read arguments
  # number of processes NP 
  if os.environ.has_key('PYAVG_THREADS'): 
    NP = int(os.environ['PYAVG_THREADS'])
  else: NP = None
  # run script in debug mode
  if os.environ.has_key('PYAVG_DEBUG'): 
    ldebug =  os.environ['PYAVG_DEBUG'] == 'DEBUG' 
  else: ldebug = False # i.e. append
  # re-compute everything or just update 
  if os.environ.has_key('PYAVG_OVERWRITE'): 
    loverwrite =  os.environ['PYAVG_OVERWRITE'] == 'OVERWRITE' 
  else: loverwrite = ldebug # False means only update old files
  
  # default settings
  if ldebug:
    #ldebug = False
    NP = NP or 2
    #loverwrite = True
    varlist = ['precip', ]
    experiments = ['max']
    #experiments = ['max','gulf','new','noah'] 
    periods = [1,]
    domains = [1,2] # domains to be processed
    filetypes = ['srfc',] # filetypes to be processed
    grid = 'WRF' 
  else:
    NP = NP or 4
    #loverwrite = False
    varlist = None
    experiments = [] # WRF experiment names (passed through WRFname)
    periods = [5,10] # averaging period
    domains = [1,2] # domains to be processed
    filetypes = ['srfc','xtrm','plev3d','hydro',] # filetypes to be processed
    grid = 'WRF' 

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
    #slon, slat, elon, elat = -179.75, 3.25, -69.75, 85.75
    slon, slat, elon, elat = -160.25, 32.75, -90.25, 72.75
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
  
  args = [] # list of arguments for workers, i.e. "work packages"
  # generate list of parameters
  for experiment in experiments:    
    # loop over file types
    for filetype in filetypes:                
      # effectively, loop over domains
      for domain in domains:
        # call worker function
        args.append((experiment, filetype, domain))
        
  # print an announcement
  print(datetime.today())
  print('\n Computing Climatologies for WRF experiments:\n')
  print(experiments)
  print('\nTHREADS: %i, DEBUG: %s, OVERWRITE: %s'%(NP,ldebug,loverwrite))
  if grid != 'WRF': print('\nRegridding to \'%s\' grid.\n')

  ## loop over and process all job sets
  exitcodes = [] # list of results
  if NP is not None and NP == 1:
    # don't parallelize, if there is only one process: just loop over files    
    for arg in args: # negative pid means serial mode
      experiment, filetype, domain = arg
      exitcode = computeClimatology(experiment, filetype, domain, lparallel=False, # lparallel=False -> serial 
                                    periods=periods, griddef=griddef)
      exitcodes.append(exitcode)
    # evaluate exit codes    
    exitcode = 0
    for ec in exitcodes:
      assert ec >= 0 
      exitcode += ec
  else:
    # create pool of workers   
    if NP is None: pool = multiprocessing.Pool() 
    else: pool = multiprocessing.Pool(processes=NP)
    # add debuggin info
    if ldebug:
      multiprocessing.log_to_stderr()
      logger = multiprocessing.get_logger()
      logger.setLevel(logging.INFO)
    # distribute tasks to workers
    kwargs = dict(lparallel=True, periods=periods, griddef=griddef) # not job dependent
    for arg in args: # negative pid means serial mode
      experiment, filetype, domain = arg      
      exitcodes.append(pool.apply_async(computeClimatology, (experiment, filetype, domain), kwargs))
    # wait until pool and queue finish
    pool.close()
    pool.join() 
    #print('\n   ***   all processes joined   ***   \n')
    # evaluate exit codes    
    exitcode = 0
    for ec in exitcodes:
      ec = ec.get()
      if ec < 0: raise ValueError, 'Exit codes have to be zero or positive!' 
      exitcode += ec
  nop = len(args) - exitcode
  # print summary
  if exitcode == 0:
    print('\n   >>>   All {:d} operations completed successfully!!!   <<<   \n'.format(nop))
  else:
    print('\n   ===   {:2d} operations completed successfully!    ===   \n'.format(nop) +
          '\n   ###   {:2d} operations did not complete/failed!   ###   \n'.format(exitcode))
  print(datetime.today())
  exit(exitcode)