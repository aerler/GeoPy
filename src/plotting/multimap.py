'''
Created on 2012-11-05, adapted for PyGeoDat on 2013-10-10

A simple script that mushroomed into a complex module... reads a Datasets and displays them in a proper 
geographic projection.

@author: Andre R. Erler, GPL v3
'''

## includes
from copy import copy # to copy map projection objects
# matplotlib config: size etc.
import numpy as np
import numpy.ma as ma
import matplotlib.pylab as pyl
import matplotlib as mpl
mpl.rc('lines', linewidth=1.)
mpl.rc('font', size=10)
from mpl_toolkits.basemap import maskoceans # used for masking data
# PyGeoDat stuff
from geodata.base import DatasetError
from datasets.common import days_per_month, days_per_month_365 # for annotation
from datasets.common import loadDatasets
from plotting.settings import getFigureSettings, getVariableSettings
# ARB project related stuff
from plotting.ARB_settings import getARBsetup, arb_figure_folder, arb_map_folder, arb_shapefile

if __name__ == '__main__':
  
#  filename = 'wrfsrfc_d%02i_clim.nc' # domain substitution
#  CFSR = loadCFSR(filename='CFSRclimFineRes1979-2009.nc')
#  RRTMG = loadWRF(exp='rrtmg-arb1', filetypes=['wrfsrfc_d%02i_clim.nc'], domains=dom)
#  axtitles = ['CRU Climatology', 'WRF Control', 'WRF RRTMG', 'Polar WRF']


  ## general settings and shortcuts
  WRFfiletypes=['srfc','xtrm','hydro'] # WRF data source
#   WRFfiletypes = ['srfc','hydro','xtrm'] # WRF data source
  # figure directory
  folder = arb_figure_folder
  lpickle = False
  # period shortcuts
  H01 = '1979'; H02 = '1979-1981'; H03 = '1979-1982'; H30 = '1979-2009' # for tests 
  H05 = '1979-1984'; H10 = '1979-1989'; H15 = '1979-1994' # historical validation periods
  G10 = '1969-1979'; I10 = '1989-1999'; J10 = '1999-2009' # additional historical periods
  A03 = '2045-2048'; A05 = '2045-2050'; A10 = '2045-2055'; A15 = '2045-2060' # mid-21st century
  B03 = '2095-2098'; B05 = '2095-2100'; B10 = '2095-2105'; B15 = '2095-2110' # late 21st century  
  ltitle = True # plot/figure title
  lcontour = False # contour or pcolor plot
  lframe = True # draw domain boundary
  lstations = True
  lbasin = True
  cbo = None # default based on figure type
  resolution = None # only for GPCC (None = default/highest)
  exptitles = None
  reflist = None # an additional list of experiments, that can be used to compute differences
  grid = None
  lWRFnative = False
  ldiff = False # compute differences
  lfrac = False # compute fraction
  domain = (2,)
  ## case settings
  
  # observations
  lprint = True # write plots to disk using case as a name tag
  maptype = 'lcc-new'; lstations = True; lbasin = True
  grid = 'arb2_d02'; domain = (2,)
#   explist = ['Unity']; exptitles = ['Merged Observations: Precipitation [mm/day]']; 
#   period = H10; case = 'unity'; ltitle = False
#   ldiff = True; reflist = ['Unity']
#   lfrac = True; reflist = ['Unity']
#   explist = ['max','ctrl','new','NARR']; period = H10; case = 'val'
#   explist = ['CFSR','CRU','NARR','Unity']; period = H10; case = 'obs'
#   exptitles = [None,None,None,'GPCC (no data)']
#   explist = ['max','ctrl','noah','CRU']; period = H10; case = 'val'
  explist = ['max','ctrl','new','noah']; period = H10; case = 'hydro'
#   explist = ['max','ctrl','new','CRU']; period = H10; case = 'val'
#   explist = ['max','CRU','cfsr','ctrl']; period = H10; case = 'val'
#   explist = ['max','PRISM','max-A','max-B','cfsr','max-C']; period = H10; case = 'ens'
#   explist = ['max','CRU','max-A','max-B','NARR','max-C']; period = H10; case = 'ens'
#   explist = ['max','cfsr','new','ctrl']; period = H10; case = 'hydro'
#   explist = ['max','max-2050','gulf','seaice-2050']; period = [H10, A10, H10, A10]; case = 'mix'
#   explist = ['GPCC','PRISM','CRU','GPCC']; period = [None,None,H30,H30]
#   explist = ['max-A','max-B','max-C','max-A-2050','max-B-2050','max-C-2050']
#   period = ['1979-1987']*3+['2045-2053']*3; case = 'maxens'
#   period = [A05,H05]+[A05]*4
#   explist = ['PRISM','CRU','GPCC','NARR']
#   grid = 'ARB_small_05'
#   period = [None,H30,H30,H30]; case = 'obs'
#   explist = ['max']; period = H10; case = 'test'
#   explist = ['max','ctrl','new','noah']; reflist = ['Unity']; period = H10; case = 'val'; ldiff=True

#   case = 'bugaboo'; period = '1997-1998'  # name tag
#   maptype = 'lcc-coast'; lstations = False # 'lcc-new'  
#   explist = ['coast']; domain = (3,);
#   case = 'columbia'; 
#   maptype = 'lcc-arb'; lstations = True; lbasin = True # 'lcc-new'  
#   ldiff = True; reflist = ['Unity']
#   period = ['1979-1980']  # name tag
#   explist = ['columbia']; domain = [(3,)]; 
#   exptitles = ['Max 3km'] 
#   period = ['1979-1980','1979-1984','1979-1980','1979-1980']  # name tag
#   explist = ['columbia', 'cfsr','columbia','columbia']; domain = [(3,),(2,),(2,),(1,)]; 
#   exptitles = ['Max 3km', 'CFSR 10km','Max 9km','Max 27km'] 
#   domain = [(1,2,3),None,(1,2),(1,)]
#   explist = ['coast','PRISM','coast','coast',]
#   exptitles = ['WRF 1km (Bugaboo)', 'PRISM Climatology', 'WRF 5km (Bugaboo)', 'WRF 25km (Bugaboo)']
#   explist = ['coast','PRISM','CFSR','coast',]
#   exptitles = ['WRF 1km (Bugaboo)', 'PRISM Climatology', 'CFSR 1997-1998', 'WRF 5km (Bugaboo)']
 

  if not case: raise ValueError, 'Need to define a \'case\' name!'
  
  ## select variables and seasons
  varlist = []; seasons = []
  # variables
#   varlist += ['T2']
#   varlist += ['Tmin', 'Tmax']
#   varlist += ['precip']
#   varlist += ['waterflx']
  varlist += ['p-et']
#   varlist += ['precipnc', 'precipc']
#   varlist += ['Q2']
#   varlist += ['evap']
#   varlist += ['pet']
#   varlist += ['snow']
#   varlist += ['snowh']
#   varlist += ['GLW','OLR','qtfx']
#   varlist += ['SWDOWN','GLW','OLR']
#   varlist += ['hfx','lhfx']
#   varlist += ['qtfx','lhfr']
#   varlist += ['SST']
  # seasons
#   seasons = [ [ 9 ] ]
#   seasons = [ [i] for i in xrange(12) ] # monthly
#   seasons += ['annual']
#   seasons += ['summer']
#   seasons += ['winter']
  seasons += ['spring']    
#   seasons += ['fall']
  # special variable/season combinations
#   varlist = ['seaice']; seasons = [8] # September seaice
#  varlist = ['snowh'];  seasons = [8] # September snow height
#  varlist = ['stns']; seasons = ['annual']
#   varlist = ['lndcls']; seasons = [''] # static
#   varlist = ['zs']; seasons = [''] # static
#   varlist = ['zs']; seasons = ['hidef'] # static
  

  # setup projection and map
  mapSetup = getARBsetup(maptype, stations=lstations, lpickle=lpickle, folder=arb_map_folder)
  
  ## load data
  loadlist = set(varlist).union(('lon2D','lat2D'))
  exps, axtitles, nexps = loadDatasets(explist, n=None, varlist=loadlist, titles=exptitles, periods=period, domains=domain, 
                                       grids=grid, resolutions=resolution, filetypes=WRFfiletypes, lWRFnative=lWRFnative, ltuple=True)
  nlen = len(exps)
  print exps[-1][-1]
  # load reference list
  if reflist is not None:
    if not isinstance(reflist,(list,tuple)): raise TypeError
    if len(explist) > len(reflist):
      if len(reflist) == 1: reflist *= len(explist)  
      else: raise DatasetError 
    refs, a, b = loadDatasets(reflist, n=None, varlist=loadlist, titles=None, periods=period, domains=domain, grids=grid,
                              resolutions=resolution, filetypes=WRFfiletypes, lWRFnative=lWRFnative, ltuple=True)
    # merge lists
    if len(exps) != len(refs): raise DatasetError, 'Experiments and reference list need to have the same length!'
    for i in xrange(len(exps)):
      if not isinstance(exps[i],tuple): raise TypeError 
      if not isinstance(refs[i],tuple): raise TypeError
      if len(exps[i]) != len(refs[i]): DatasetError, 'Experiments and reference tuples need to have the same length!'
      exps[i] = exps[i] + refs[i] # merge lists/tuples
  
  
  # get figure settings
  sf, figformat, margins, caxpos, subplot, figsize, cbo = getFigureSettings(nlen, cbar=True, cbo=cbo)
  
  # get projections settings
  projection, grid, res = mapSetup.getProjectionSettings()
  
  ## loop over varlist and seasons
  maps = []; x = []; y = [] # projection objects and coordinate fields (only computed once)
  # start loop
  for var in varlist:
    oldvar = var
    for season in seasons:
      
      # get variable properties and additional settings
      clevs, clim, cbl, clbl, cmap, lmskocn, lmsklnd, plottype, month = getVariableSettings(
                                                      var, season, oldvar=var, ldiff=ldiff, lfrac=lfrac)
      
      # assemble plot title
      if ldiff: filename = '%s_diff_%s_%s.%s'%(var,season,case,figformat)
      elif lfrac: filename = '%s_frac_%s_%s.%s'%(var,season,case,figformat)
      else: filename = '%s_%s_%s.%s'%(var,season,case,figformat)
      #print exps[0][0].name
      plat = exps[0][0].variables[var].plot
      if lfrac: figtitle = '{0:s} {1:s} [%]'.format(plottype,plat['plottitle'])
      elif plat['plotunits']: figtitle = '%s %s [%s]'%(plottype,plat['plottitle'],plat['plotunits'])
      else: figtitle = '%s %s'%(plottype,plat['plottitle'])
      
      # feedback
      print('\n\n   ***  %s %s (%s)   ***   \n'%(plottype,plat['plottitle'],var))
      
      ## compute data
      data = []; lons = []; lats=[]  # list of data and coordinate fields to be plotted 
      # compute average WRF precip            
      print(' - loading data ({0:s})'.format(var))
      for exptpl in exps:
        lontpl = []; lattpl = []; datatpl = []                
        for exp in exptpl:
          expvar = exp.variables[var]
          if len(seasons) > 1: expvar.load()
          #print expvar.name, exp.name, expvar.masked
          print(exp.name)
          assert expvar.gdal
          # handle dimensions
          if expvar.isProjected: 
            assert (exp.lon2D.ndim == 2) and (exp.lat2D.ndim == 2), 'No coordinate fields found!'
            exp.lon2D.load(); exp.lat2D.load()
            lon = exp.lon2D.getArray(); lat = exp.lat2D.getArray()          
          else: 
            assert expvar.hasAxis('lon') and expvar.hasAxis('lat'), 'No geographic axes found!'
            lon, lat = np.meshgrid(expvar.lon.getArray(),expvar.lat.getArray())
          lontpl.append(lon); lattpl.append(lat) # append to data list
          # figure out calendar
          if 'WRF' in exp.atts.get('description',''): mon = days_per_month_365
          else: mon = days_per_month
          # extract data field
          vardata = ma.zeros(expvar.mapSize) # allocate masked array
          #np.zeros(expvar.mapSize) # allocate array
          # compute average over seasonal range
          days = 0
          if expvar.hasAxis('time'):
#             vardata = expvar.mean(time=(min(month),max(month)), asVar=False)
            vardata.set_fill_value(np.NaN)
            for m in month:
              n = m-1 
              tmp = expvar(time=exp.time[n])
              vardata += tmp * mon[n]
              days += mon[n]
            vardata /=  days # normalize 
            vardata.set_fill_value(np.NaN)
          else:
            vardata = expvar[:].squeeze()
          vardata = vardata * expvar.plot.get('scalefactor',1) # apply plot unit conversion          
          if lmskocn: 
            if exp.variables.has_key('lnd'): # CESM and CFSR 
              vardata[exp.lnd.get()<0.5] = -2. # use land fraction
            elif exp.variables.has_key('lndidx'): 
              mask = exp.lndidx.get()
              vardata[mask==16] = -2. # use land use index (ocean)  
              vardata[mask==24] = -2. # use land use index (lake)
            else : vardata = maskoceans(lon,lat,vardata,resolution=res,grid=grid)
          if lmsklnd: 
            if exp.variables.has_key('lnd'): # CESM and CFSR 
              vardata[exp.lnd.get()>0.5] = 0 # use land fraction
            elif exp.variables.has_key('lndidx'): # use land use index (ocean and lake)
              mask = exp.lndidx.get(); tmp = vardata.copy(); vardata[:] = 0.
              vardata[mask==16] = tmp[mask==16]; vardata[mask==24] = tmp[mask==24]
          datatpl.append(vardata) # append to data list
        ## compute differences, if desired
        if ldiff or lfrac:
          assert len(datatpl)%2 == 0, 'needs to be divisible by 2'
          ntpl = len(datatpl)/2 # assuming (exp1, exp2, ..., ref1, ref2, ...)
          for i in xrange(ntpl):
            if ldiff: datatpl[i] = datatpl[i] - datatpl[i+ntpl] # compute differences in place
            elif lfrac: datatpl[i] = (datatpl[i]/datatpl[i+ntpl]-1)*100 # compute fractions in place
          del datatpl[ntpl+1:] # delete the rest 
        # add tuples to master list
        lons.append(lontpl); lats.append(lattpl); data.append(datatpl)
        print('')
              
      ## setup projection
      #print(' - setting up figure\n') 
      nax = subplot[0]*subplot[1] # number of panels
      # make figure and axes
      f = pyl.figure(facecolor='white', figsize=figsize)
      ax = []
      for n in xrange(nax):
        ax.append(f.add_subplot(subplot[0],subplot[1],n+1))
      f.subplots_adjust(**margins) # hspace, wspace
      if not maps:
        print(' - setting up map projection\n') 
        #mastermap = Basemap(ax=ax[n],**projection) # make just one basemap with dummy axes handle
        mastermap = mapSetup.basemap
        for axi in ax: # replace dummy axes handle with correct axes handle
          tmp = copy(mastermap)
          tmp.ax = axi  
          maps.append(tmp) # one map for each panel!!  
      else:
        print(' - resetting map projection\n') 
        for n in xrange(nax):
          maps[n].ax=ax[n] # assign new axes to old projection
      # transform coordinates (on per-map basis)
      if not (x and y):
        print(' - transforming coordinate fields\n')
        for n in xrange(nax):
          xtpl = []; ytpl = []
          for m in xrange(nexps[n]):
            xx, yy = maps[n](lons[n][m],lats[n][m]) # convert to map-native coordinates
            xtpl.append(xx); ytpl.append(yy)
          x.append(xtpl); y.append(ytpl) 
        
      ## Plot data
      # draw boundaries of inner domain
      if lframe:
        print(' - drawing data frames\n')
        for n in xrange(nax):
          for m in xrange(nexps[n]):   
            bdy = ma.ones(data[n][m].shape); bdy[ma.getmaskarray(data[n][m])] = 0
            # N.B.: for some reason, using np.ones_like() causes a masked data array to fill with zeros  
            #print bdy.mean(), data[n][m].__class__.__name__, data[n][m].fill_value 
            bdy[0,:]=0; bdy[-1,:]=0; bdy[:,0]=0; bdy[:,-1]=0 # demarcate domain boundaries        
            maps[n].contour(x[n][m],y[n][m],bdy,[1,0,-1],ax=ax[n], colors='k', fill=False) # draw boundary of inner domain
      # draw data
      norm = mpl.colors.Normalize(vmin=min(clevs),vmax=max(clevs),clip=True) # for colormap
      cd = []
#       print(' - creating plots\n')  
      for n in xrange(nax): 
        for m in xrange(nexps[n]):
          print('panel %i: min %f / max %f / mean %f'%(n,data[n][m].min(),data[n][m].max(),data[n][m].mean()))
          if lcontour: cd.append(maps[n].contourf(x[n][m],y[n][m],data[n][m],clevs,ax=ax[n],cmap=cmap, norm=norm,extend='both'))  
          else: cd.append(maps[n].pcolormesh(x[n][m],y[n][m],data[n][m],cmap=cmap,shading='gouraud'))
      # add colorbar
      #TODO: use utils.sharedColorbar
      cax = f.add_axes(caxpos)
      for cn in cd: # [c1d1, c1d2, c2d2]:
        if clim: cn.set_clim(vmin=clim[0],vmax=clim[1])
        else: cn.set_clim(vmin=min(clevs),vmax=max(clevs))
      cbar = f.colorbar(cax=cax,mappable=cd[0],orientation=cbo,extend='both') # ,size='3%',pad='2%'       
      if cbl is None: cbl = np.linspace(min(clevs),max(clevs),6)
      cbar.set_ticks(cbl); cbar.set_ticklabels([clbl%(lev) for lev in cbl])
    
      ## Annotation
      #print('\n - annotating plots\n')      
      # add labels
      if ltitle: f.suptitle(figtitle,fontsize=12)
      # add a map scale to lower left axes
      msn = len(maps)/2 # place scale 
      mapSetup.drawScale(maps[msn])
      n = -1 # axes counter
      for i in xrange(subplot[0]):
        for j in xrange(subplot[1]):
          n += 1 # count up
          axn = ax[n] 
          axn.set_title(axtitles[n],fontsize=11) # axes title
          if j == 0 : left = True
          else: left = False 
          if i == subplot[0]-1: bottom = True
          else: bottom = False
          # begin annotation
          bmap = maps[n]
          # black-out continents, if we have no proper land mask 
          if lmsklnd and not (exps[n][0].variables.has_key('lndmsk') or exps[n][0].variables.has_key('lndidx')): 
            blklnd = True        
          else: blklnd = False                  
          # misc annotatiosn
          mapSetup.miscAnnotation(bmap, blklnd=blklnd)
          # add parallels and meridians
          mapSetup.drawGrid(bmap, left, bottom)
          # mark stations
          if lstations: mapSetup.markStations(ax[n], bmap)     
          # add ARB basin outline
          if lbasin: 
            bmap.readshapefile(arb_shapefile, 'ARB', ax=axn, drawbounds=True, linewidth=0.5, color='k')
            #print bmap.ARB_info                   
              
      # save figure to disk
      if lprint:
        print('\nSaving figure in '+filename)
        f.savefig(folder+filename, **sf) # save figure to pdf
        print(folder)
  
  ## show plots after all iterations
  pyl.show()

