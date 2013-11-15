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
from datasets.WRF import loadWRF
from datasets.WRF_experiments import exps as WRF_exps
# from datasets.CESM import loadCESM, CESMtitle
from datasets.CFSR import loadCFSR
from datasets.NARR import loadNARR
from datasets.GPCC import loadGPCC
from datasets.CRU import loadCRU
from datasets.PRISM import loadPRISM
from datasets.common import days_per_month, days_per_month_365 # for annotation
from plotting.settings import getFigureSettings, getVariableSettings
# ARB project related stuff
from plotting.ARB_settings import getARBsetup, arb_figure_folder, arb_map_folder

if __name__ == '__main__':
  
#  filename = 'wrfsrfc_d%02i_clim.nc' # domain substitution
#  CFSR = loadCFSR(filename='CFSRclimFineRes1979-2009.nc')
#  RRTMG = loadWRF(exp='rrtmg-arb1', filetypes=['wrfsrfc_d%02i_clim.nc'], domains=dom)
#  axtitles = ['CRU Climatology', 'WRF Control', 'WRF RRTMG', 'Polar WRF']


  ## general settings and shortcuts
  WRFfiletypes=['srfc'] # WRF data source
  # figure directory
  folder = arb_figure_folder
  # period shortcuts
  H01 = '1979'; H02 = '1979-1981'; H03 = '1979-1982'; H30 = '1979-2009' # for tests 
  H05 = '1979-1984'; H10 = '1979-1989'; H15 = '1979-1994' # historical validation periods
  G10 = '1969-1979'; I10 = '1989-1999'; J10 = '1999-2009' # additional historical periods
  A03 = '2045-2048'; A05 = '2045-2050'; A10 = '2045-2055'; A15 = '2045-2060' # mid-21st century
  B03 = '2095-2098'; B05 = '2095-2100'; B10 = '2095-2105'; B15 = '2095-2110' # late 21st century  
  ltitle = True # plot/figure title
  lcontour = False # contour or pcolor plot
  lframe = True # draw domain boundary
  cbo = None # default based on figure type
  resolution = None # only for GPCC (None = default/highest)
  exptitles = None
  grid = None
  domain = (1,2)
  ## case settings
  
  # observations
  case = 'maxens'; lprint = True # write plots to disk using case as a name tag
  maptype = 'lcc-new'; lstations = True
#   grid = 'arb2_d02'; 
  lexceptWRF = True; domain = (2,)
#   explist = ['GPCC','PRISM','CRU','GPCC']
#   period = [None,None,H30,H30]
  explist = ['max-2050','max','max-A-2050','max-B-2050','seaice-2050','max-C-2050']
  period = [A05,H05]+[A05]*4
  explist = ['max','new','max-A','max-B','noah','max-C']
  period = H05
#   explist = ['CRU']
#   explist = ['PRISM','CRU']
#   period = [None,H30]
#   explist = ['PRISM']
#   period = [None]
#   case = 'bugaboo'; period = '1997-1998'  # name tag
#   maptype = 'lcc-coast'; lstations = False # 'lcc-new'  
#   explist = ['coast']; domain = (2,)
#   domain = [(1,2,3),None,(1,2),(1,)]
#   explist = ['coast','PRISM','coast','coast',]
#   exptitles = ['WRF 1km (Bugaboo)', 'PRISM Climatology', 'WRF 5km (Bugaboo)', 'WRF 25km (Bugaboo)']
#   explist = ['coast','PRISM','CFSR','coast',]
#   exptitles = ['WRF 1km (Bugaboo)', 'PRISM Climatology', 'CFSR 1997-1998', 'WRF 5km (Bugaboo)']
  
  ## select variables and seasons
  varlist = []; seasons = []
  # variables
#   varlist += ['T2']
#   varlist += ['Tmin', 'Tmax']
#   varlist += ['precip']
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
#   seasons = [ [i] for i in xrange(12) ] # monthly
  seasons += ['annual']
  seasons += ['summer']
  seasons += ['winter']
  seasons += ['spring']    
  seasons += ['fall']
  # special variable/season combinations
#   varlist = ['seaice']; seasons = [8] # September seaice
#  varlist = ['snowh'];  seasons = [8] # September snow height
#  varlist = ['zs']; seasons = ['hidef']
#  varlist = ['stns']; seasons = ['annual']
#   varlist = ['lndcls']; seasons = [''] # static
  

  # setup projection and map
  mapSetup = getARBsetup(maptype, stations=lstations, lpickle=True, folder=arb_map_folder)
  
  ## load data   
  if not isinstance(exptitles,(tuple,list)): exptitles = (exptitles,)*len(explist)
  elif len(exptitles) == 0: exptitles = (None,)*len(explist) 
  if not isinstance(period,(tuple,list)): period = (period,)*len(explist)
  if not isinstance(domain[0],(tuple,list)): domain = (domain,)*len(explist)
  if not isinstance(grid,(tuple,list)): grid = (grid,)*len(explist)
  # add stuff to varlist
  loadlist = set(varlist).union(('lon2D','lat2D'))
  exps = []; axtitles = []
  for exp,tit,prd,dom,grd in zip(explist,exptitles,period,domain,grid): 
#     ext = exp; axt = ''
    if isinstance(exp,str):
      if exp[0].isupper():
        if exp == 'GPCC': ext = (loadGPCC(resolution=resolution, period=prd, grid=grd, varlist=loadlist),); axt = 'GPCC Observations'
        elif exp == 'CRU': ext = (loadCRU(period=prd, grid=grd, varlist=loadlist),); axt = 'CRU Observations' 
        elif exp == 'PRISM': # all PRISM derivatives
          if len(varlist) == 1 and varlist[0] == 'precip': 
            ext = (loadGPCC(grid=grd, varlist=loadlist), loadPRISM(grid=grd, varlist=loadlist),); axt = 'PRISM (and GPCC)'
            #  ext = (loadPRISM(),); axt = 'PRISM'
          else: ext = (loadCRU(period='1979-2009', grid=grd, varlist=loadlist), loadPRISM(grid=grd, varlist=loadlist)); axt = 'PRISM (and CRU)'
          # ext = (loadPRISM(),)          
        elif exp == 'CFSR': ext = (loadCFSR(period=prd, grid=grd, varlist=loadlist),); axt = 'CFSR Reanalysis' 
        elif exp == 'NARR': ext = (loadNARR(period=prd, grid=grd, varlist=loadlist),); axt = 'NARR Reanalysis'
        else: # all other uppercase names are CESM runs
          raise NotImplementedError, "CESM datasets are currently not supported."  
#           ext = (loadCESM(exp=exp, period=prd),)
#           axt = CESMtitle.get(exp,exp)
      else: # WRF runs are all in lower case
        exp = WRF_exps[exp]        
        if 'xtrm' in WRFfiletypes: varatts = dict(Tmean=dict(name='T2'))
        else: varatts = None
        if lexceptWRF: grd = None
        ext = loadWRF(experiment=exp.name, period=prd, grid=grd, domains=dom, filetypes=WRFfiletypes, 
                      varlist=loadlist, varatts=varatts)  
        axt = exp.title # defaults to name...
    exps.append(ext); axtitles.append(tit or axt)  
  print exps[-1][-1]
  # count experiment tuples (layers per panel)
  nexps = []; nlen = len(exps)
  for n in xrange(nlen):
    if not isinstance(exps[n],(tuple,list)): # should not be necessary
      exps[n] = (exps[n],)
    nexps.append(len(exps[n])) # layer counter for each panel
  
  # get figure settings
  sf, figformat, margins, caxpos, subplot, figsize, cbo = getFigureSettings(nlen, cbo=cbo)
  
  # get projections settings
  projection, grid, res = mapSetup.getProjectionSettings()
  
  ## loop over varlist and seasons
  maps = []; x = []; y = [] # projection objects and coordinate fields (only computed once)
  # start loop
  for var in varlist:
    oldvar = var
    for season in seasons:
      
      # get variable properties and additional settings
      clevs, clim, cbl, clbl, cmap, lmskocn, lmsklnd, plottype, month = getVariableSettings(var, season, oldvar='')
      
      # assemble plot title
      filename = '%s_%s_%s.%s'%(var,season,case,figformat)
      #print exps[0][0].name
      plat = exps[0][0].variables[var].plot
      if plat['plotunits']: figtitle = '%s %s [%s]'%(plottype,plat['plottitle'],plat['plotunits'])
      else: figtitle = '%s %s'%(plottype,plat['plottitle'])
      
      # feedback
      print('\n\n   ***  %s %s (%s)   ***   \n'%(plottype,plat['plottitle'],var))
      
      ## compute data
      data = []; lons = []; lats=[]  # list of data and coordinate fields to be plotted 
      # compute average WRF precip            
      print(' - loading data')
      for exptpl in exps:
        lontpl = []; lattpl = []; datatpl = []                
        for exp in exptpl:
          expvar = exp.variables[var]
          if len(seasons) > 1: expvar.load()
          print expvar.name, exp.name, expvar.masked
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
          ax[n].set_title(axtitles[n],fontsize=11) # axes title
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
              
      # save figure to disk
      if lprint:
        print('\nSaving figure in '+filename)
        f.savefig(folder+filename, **sf) # save figure to pdf
        print(folder)
  
  ## show plots after all iterations
  pyl.show()

