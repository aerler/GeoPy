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
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
# PyGeoDat stuff
from datasets.WRF import loadWRF #, WRFtitle
# from datasets.CESM import loadCESM, CESMtitle
from datasets.CFSR import loadCFSR
from datasets.NARR import loadNARR
from datasets.GPCC import loadGPCC
from datasets.CRU import loadCRU
from datasets.PRISM import loadPRISM
from datasets.common import days_per_month, days_per_month_365, name_of_month # for annotation
# ARB project related stuff
from plotting.ARB_settings import WRFname, WRFtitle, getProjectionSettings

## figure settings
def getFigureSettings(nexp, cbo):
  sf = dict(dpi=150) # print properties
  figformat = 'png'
  folder = '/home/me/Research/Dynamical Downscaling/Figures/' # figure directory
  # figure out colorbar placement
  if cbo == 'vertical':
    margins = dict(bottom=0.02, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
    caxpos = [0.91, 0.05, 0.03, 0.9]
  else:# 'horizontal'
    margins = dict(bottom=0.1, left=0.065, right=.9725, top=.925, hspace=0.05, wspace=0.05)
    caxpos = [0.05, 0.05, 0.9, 0.03]        
  # pane settings
  if nexp == 1:
    ## 1 panel
    subplot = (1,1)
    figsize = (3.75,3.75) #figsize = (6.25,6.25)  #figsize = (7,5.5)
    margins = dict(bottom=0.025, left=0.075, right=0.875, top=0.875, hspace=0.0, wspace=0.0)
#     margins = dict(bottom=0.12, left=0.075, right=.9725, top=.95, hspace=0.05, wspace=0.05)
#    margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
  elif nexp == 2:
    ## 2 panel
    subplot = (1,2)
    figsize = (6.25,5.5)
  elif nexp == 4:
    # 4 panel
    subplot = (2,2)
    figsize = (6.25,6.25)
    margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
  elif nexp == 4:
    # 4 panel
    subplot = (2,2)
    figsize = (6.25,6.25)
    margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
  elif nexp == 6:
    # 6 panel
    subplot = (2,3) # rows, columns
    figsize = (9.25,6.5) # width, height (inches)
    cbo = 'horizontal'
    margins = dict(bottom=0.09, left=0.05, right=.97, top=.92, hspace=0.1, wspace=0.05)
    caxpos = [0.05, 0.025, 0.9, 0.03]
  #    margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
  # return values
  return sf, figformat, folder, margins, caxpos, subplot, figsize, cbo


if __name__ == '__main__':
  
#  filename = 'wrfsrfc_d%02i_clim.nc' # domain substitution
#  CFSR = loadCFSR(filename='CFSRclimFineRes1979-2009.nc')
#  RRTMG = loadWRF(exp='rrtmg-arb1', filetypes=['wrfsrfc_d%02i_clim.nc'], domains=dom)
#  axtitles = ['CRU Climatology', 'WRF Control', 'WRF RRTMG', 'Polar WRF']


  ## general settings and shortcuts
  H01 = '1979'; H02 = '1979-1981'; H03 = '1979-1982'; H30 = '1979-2009' # for tests 
  H05 = '1979-1984'; H10 = '1979-1989'; H15 = '1979-1994' # historical validation periods
  G10 = '1969-1979'; I10 = '1989-1999'; J10 = '1999-2009' # additional historical periods
  A03 = '2045-2048'; A05 = '2045-2050'; A10 = '2045-2055'; A15 = '2045-2060' # mid-21st century
  B03 = '2095-2098'; B05 = '2095-2100'; B10 = '2095-2105'; B15 = '2095-2110' # late 21st century
  lprint = False # write plots to disk
  ltitle = True # plot/figure title
  lcontour = False # contour or pcolor plot
  lframe = True # draw domain boundary
  cbo = 'vertical' # vertical horizontal
  resolution=None # only for GPCC (None = default/highest)
 
  ## case settings
  
  # observations
  case = 'max' # name tag
  projtype = 'lcc-new' # 'lcc-new'  
  period = H10; dom = (2,)
  explist = ['PRISM']; period = [None]
#   explist = ['max','NARR','PRISM','new']
#   period = [H10, H10, A10, H10]
  
  ## select variables and seasons
#   varlist = ['precipnc', 'precipc', 'T2']
  varlist = ['precip']
#   varlist = ['evap']
#   varlist = ['snow']
#   varlist = ['precip', 'T2', 'p-et','evap']
#   varlist = ['p-et','precip','snow']
#   varlist = ['GLW','OLR','qtfx']
#   varlist = ['SWDOWN','GLW','OLR']
#   varlist = ['hfx','lhfx']
#   varlist = ['qtfx','lhfr']
#   varlist = ['precip','T2']
#   varlist = ['T2']
#  varlist = ['precip','T2','snow']
#   varlist = ['snow', 'snowh']
#  varlist = ['SST','T2','precip','snow','snowh']
#   seasons = [ [i] for i in xrange(12) ] # monthly
  seasons = ['annual']
#   seasons = ['summer']
#   seasons = ['winter']    
#   seasons = ['winter', 'summer', 'annual']
#   varlist = ['snow']; seasons = ['fall','winter','spring']
#   varlist = ['seaice']; seasons = [8] # September seaice
#  varlist = ['snowh'];  seasons = [8] # September snow height
#  varlist = ['precip']; seasons = ['annual']
#  varlist = ['zs']; seasons = ['hidef']
#  varlist = ['stns']; seasons = ['annual']
#   varlist = ['lndcls']; seasons = [''] # static
  

  ## load data 
  if not isinstance(period,(tuple,list)): period = (period,)*len(explist)
  exps = []; axtitles = []
  for exp,prd in zip(explist,period): 
    ext = exp; axt = ''
    if isinstance(exp,str):
      if exp[0].isupper():
        if exp == 'GPCC': ext = (loadGPCC(resolution=resolution,period=prd),); axt = 'GPCC Observations' # ,period=prd
        elif exp == 'CRU': ext = (loadCRU(period=prd),); axt = 'CRU Observations' 
        elif exp[0:5] == 'PRISM': # all PRISM derivatives
          if len(varlist) == 1 and varlist[0] == 'precip': 
            ext = (loadGPCC(), loadPRISM()); axt = 'PRISM (and GPCC)'
            #  ext = (loadPRISM(),); axt = 'PRISM'
          else: ext = (loadCRU(period='1979-2009'), loadPRISM()); axt = 'PRISM (and CRU)'
          # ext = (loadPRISM(),)          
        elif exp == 'CFSR': ext = (loadCFSR(period=prd),); axt = 'CFSR Reanalysis' 
        elif exp == 'NARR': ext = (loadNARR(period=prd),); axt = 'NARR Reanalysis'
        else: # all other uppercase names are CESM runs
          raise NotImplementedError, "CESM datasets are currently not supported."  
#           ext = (loadCESM(exp=exp, period=prd),)
#           axt = CESMtitle.get(exp,exp)
      else: # WRF runs are all in lower case
        ext = loadWRF(experiment=WRFname[exp], period=prd, domains=dom, filetypes=['const','srfc'])
        axt = WRFtitle.get(exp,exp)
    exps.append(ext); axtitles.append(axt)  
  print exps[-1][-1]
  # count experiment tuples (layers per panel)
  nexps = []; nlen = len(exps)
  for n in xrange(nlen):
    if not isinstance(exps[n],(tuple,list)): # should not be necessary
      exps[n] = (exps[n],)
    nexps.append(len(exps[n])) # layer counter for each panel
  
  # get figure settings
  sf, figformat, folder, margins, caxpos, subplot, figsize, cbo = getFigureSettings(nexp=nlen, cbo=cbo)
  
  # get projections settings
  projection, grid, res = getProjectionSettings(projtype=projtype)
  
  ## loop over varlist and seasons
  maps = []; x = []; y = [] # projection objects and coordinate fields (only computed once)
  # start loop
  for var in varlist:
    oldvar = var
    for season in seasons:
      
      ## settings
      # plot variable and averaging 
      cbl = None; clim = None       
      lmskocn = False; lmsklnd = False # mask ocean or land?
      # color maps and   scale (contour levels)
      cmap = mpl.cm.gist_ncar; cmap.set_over('white'); cmap.set_under('black')
      if var == 'snow': # snow (liquid water equivalent) 
        lmskocn = True; clbl = '%2.0f' # kg/m^2
        clevs = np.linspace(0,200,41)
      elif var == 'snowh': # snow (depth/height) 
        lmskocn = True; clbl = '%2.1f' # m
        clevs = np.linspace(0,2,41)
      elif var=='hfx' or var=='lhfx' or var=='qtfx': # heat fluxes (W / m^2)
        clevs = np.linspace(-20,100,41); clbl = '%03.0f'
        if var == 'qtfx': clevs = clevs * 2
        if season == 'winter': clevs = clevs - 30
        elif season == 'summer': clevs = clevs + 30
      elif var=='GLW': # heat fluxes (W / m^2)
        clevs = np.linspace(200,320,41); clbl = '%03.0f'
        if season == 'winter': clevs = clevs - 40
        elif season == 'summer': clevs = clevs + 40
      elif var=='OLR': # heat fluxes (W / m^2)
        clevs = np.linspace(190,240,31); clbl = '%03.0f'
        if season == 'winter': clevs = clevs - 20
        elif season == 'summer': clevs = clevs + 30
      elif var=='rfx': # heat fluxes (W / m^2)
        clevs = np.linspace(320,470,51); clbl = '%03.0f'
        if season == 'winter': clevs = clevs - 100
        elif season == 'summer': clevs = clevs + 80
      elif var=='SWDOWN' or var=='SWNORM': # heat fluxes (W / m^2)
        clevs = np.linspace(80,220,51); clbl = '%03.0f'
        if season == 'winter': clevs = clevs - 80
        elif season == 'summer': clevs = clevs + 120
      elif var == 'lhfr': # relative latent heat flux (fraction)        
        clevs = np.linspace(0,1,26); clbl = '%2.1f' # fraction
      elif var == 'evap': # moisture fluxes (kg /(m^2 s))
        clevs = np.linspace(-4,4,25); clbl = '%02.1f'
        cmap = mpl.cm.PuOr
      elif var == 'p-et': # moisture fluxes (kg /(m^2 s))
        # clevs = np.linspace(-3,22,51); clbl = '%02.1f'
        clevs = np.linspace(-2,2,25); cmap = mpl.cm.PuOr; clbl = '%02.1f'
      elif var == 'precip' or var == 'precipnc': # total precipitation 
        clevs = np.linspace(0,20,41); clbl = '%02.1f' # mm/day
      elif var == 'precipc': # convective precipitation 
        clevs = np.linspace(0,5,26); clbl = '%02.1f' # mm/day
      elif oldvar=='SST' or var=='SST': # skin temperature (SST)
        clevs = np.linspace(240,300,61); clbl = '%03.0f' # K
        var = 'Ts'; lmsklnd = True # mask land
      elif var=='T2' or var=='Ts': # 2m or skin temperature (SST)
        clevs = np.linspace(255,290,36); clbl = '%03.0f' # K
        if season == 'winter': clevs = clevs - 10
        elif season == 'summer': clevs = clevs + 10
      elif var == 'seaice': # sea ice fraction
        lmsklnd = True # mask land        
        clevs = np.linspace(0.04,1,25); clbl = '%2.1f' # fraction
        cmap.set_under('white')
      elif var == 'zs': # surface elevation / topography
        if season == 'topo':
          lmskocn = True; clim = (-1.,2.5); # nice geographic map feel
          clevs = np.hstack((np.array((-1.5,)), np.linspace(0,2.5,26))); clbl = '%02.1f' # km
          cmap = mpl.cm.gist_earth; cmap.set_over('white'); cmap.set_under('blue') # topography
        elif season == 'hidef': 
          lmskocn = True; clim = (-0.5,2.5); # good contrast for high elevation
          clevs = np.hstack((np.array((-.5,)), np.linspace(0,2.5,26))); clbl = '%02.1f' # km
          cmap = mpl.cm.gist_ncar; cmap.set_over('white'); cmap.set_under('blue')
        cbl = np.linspace(0,clim[-1],6)
      elif var=='stns': # station density
        clevs = np.linspace(0,5,6); clbl = '%2i' # stations per grid points  
        cmap.set_over('purple'); cmap.set_under('white')      
      elif var=='lndcls': # land use classes (works best with contour plot)
        clevs = np.linspace(0.5,24.5,25); cbl = np.linspace(4,24,6)  
        clbl = '%2i'; cmap.set_over('purple'); cmap.set_under('white')
      # time frame / season
      if isinstance(season,str):
        if season == 'annual':  # all month
          month = range(1,13); plottype = 'Annual Average'
        elif season == 'winter':# DJF
          month = [12, 1, 2]; plottype = 'Winter Average'
        elif season == 'spring': # MAM
          month = [3, 4, 5]; plottype = 'Spring Average'
        elif season == 'summer': # JJA
          month = [6, 7, 8]; plottype = 'Summer Average'
        elif season == 'fall': # SON
          month = [9, 10, 11]; plottype = 'Fall Average'
        else:
          plottype = '' # for static fields
          month = [1]
      else:                
        month = season      
        if len(season) == 1 and isinstance(season[0],int):
          plottype =  '%s Average'%name_of_month[season[0]].strip()
          season = '%02i'%(season[0]+1) # number of month, used for file name
        else: plottype = 'Average'
      # assemble plot title
      filename = '%s_%s_%s.%s'%(var,season,case,figformat)
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
          print expvar.name, exp.name
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
      # lat_1 is first standard parallel.
      # lat_2 is second standard parallel (defaults to lat_1).
      # lon_0,lat_0 is central point.
      # rsphere=(6378137.00,6356752.3142) specifies WGS4 ellipsoid
      # area_thresh=1000 means don't plot coastline features less
      # than 1000 km^2 in area.
      if not maps:
        print(' - setting up map projection\n') 
        mastermap = Basemap(ax=ax[n],**projection)
        for axi in ax:          
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
            print bdy.mean(), data[n][m].__class__.__name__, data[n][m].fill_value 
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
    #  ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
      msn = len(maps)/2 # place scale 
      if projtype == 'lcc-new':
        maps[msn].drawmapscale(-128, 48, -120, 55, 400, barstyle='fancy', 
                             fontsize=8, yoffset=0.01*(maps[n].ymax-maps[n].ymin))
      elif projtype == 'lcc-small':
        maps[msn].drawmapscale(-136, 49, -137, 57, 800, barstyle='fancy', yoffset=0.01*(maps[n].ymax-maps[n].ymin))
      elif projtype == 'lcc-large':
        maps[msn].drawmapscale(-171, 21, -137, 57, 2000, barstyle='fancy', yoffset=0.01*(maps[n].ymax-maps[n].ymin))
      n = -1 # axes counter
      for i in xrange(subplot[0]):
        for j in xrange(subplot[1]):
          n += 1 # count up
          ax[n].set_title(axtitles[n],fontsize=11) # axes title
          if j == 0 : Left = True
          else: Left = False 
          if i == subplot[0]-1: Bottom = True
          else: Bottom = False
          # land/sea mask
          maps[n].drawlsmask(ocean_color='blue', land_color='green',resolution=res,grid=grid)
          # black-out continents, if we have no proper land mask 
          if lmsklnd and not (exps[n][0].variables.has_key('lnd') or exps[n][0].variables.has_key('lndidx')): 
            maps[n].fillcontinents(color='black',lake_color='black') 
          # add maps stuff
          maps[n].drawcoastlines(linewidth=0.5)
          maps[n].drawcountries(linewidth=0.5)
          maps[n].drawmapboundary(fill_color='k',linewidth=2)
          # labels = [left,right,top,bottom]
          if projtype=='lcc-new':
            maps[n].drawparallels([40,50,60,70],linewidth=1, labels=[Left,False,False,False])
            maps[n].drawparallels([45,55,65],linewidth=0.5, labels=[Left,False,False,False])
            maps[n].drawmeridians([-180,-160,-140,-120,-100],linewidth=1, labels=[False,False,False,Bottom])
            maps[n].drawmeridians([-170,-150,-130,-110],linewidth=0.5, labels=[False,False,False,Bottom])          
          elif projtype=='lcc-fine' or projtype=='lcc-small' or projtype=='lcc-intermed':
            maps[n].drawparallels([45,65],linewidth=1, labels=[Left,False,False,False])
            maps[n].drawparallels([55,75],linewidth=0.5, labels=[Left,False,False,False])
            maps[n].drawmeridians([-180,-160,-140,-120,-100],linewidth=1, labels=[False,False,False,Bottom])
            maps[n].drawmeridians([-170,-150,-130,-110],linewidth=0.5, labels=[False,False,False,Bottom])
          elif projtype == 'lcc-large':
            maps[n].drawparallels(range(0,90,30),linewidth=1, labels=[Left,False,False,False])
            maps[n].drawparallels(range(15,90,30),linewidth=0.5, labels=[Left,False,False,False])
            maps[n].drawmeridians(range(-180,180,30),linewidth=1, labels=[False,False,False,Bottom])
            maps[n].drawmeridians(range(-165,180,30),linewidth=0.5, labels=[False,False,False,Bottom])
          elif projtype == 'ortho':
            maps[n].drawparallels(range(-90,90,30),linewidth=1)
            maps[n].drawmeridians(range(-180,180,30),linewidth=1)
        
      # mark stations
      # ST_LINA, WESTLOCK_LITKE, JASPER, FORT_MCMURRAY, SHINING_BANK
      sn = ['SL', 'WL', 'J', 'FM', 'SB']
      slon = [-111.45,-113.85,-118.07,-111.22,-115.97]
      slat = [54.3,54.15,52.88,56.65,53.85]
      for (axn,mapt) in zip(ax,maps):
        for (name,lon,lat) in zip(sn,slon,slat):
          xx,yy = mapt(lon, lat)
          mapt.plot(xx,yy,'ko',markersize=3)
          axn.text(xx+1.5e4,yy-1.5e4,name,ha='left',va='top',fontsize=8)
      
      # save figure to disk
      if lprint:
        print('\nSaving figure in '+filename)
        f.savefig(folder+filename, **sf) # save figure to pdf
        print(folder)
  
  ## show plots after all iterations
  pyl.show()

