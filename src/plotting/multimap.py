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
import matplotlib as mpl
# mpl.use('Agg') # enforce QT4
import matplotlib.pylab as pyl
mpl.rc('lines', linewidth=1.)
mpl.rc('font', size=10)
# prevent figures from closing: don't run in interactive mode, or plt.show() will not block
pyl.ioff()
from mpl_toolkits.basemap import maskoceans # used for masking data
# PyGeoDat stuff
from geodata.base import DatasetError
from datasets.common import days_per_month, days_per_month_365 # for annotation
from datasets.common import loadDatasets, checkItemList
from datasets.WSC import basins
from plotting.settings import getFigureSettings, getVariableSettings
# ARB project related stuff
from projects.ARB_settings import getARBsetup, figure_folder, map_folder

if __name__ == '__main__':
  
  ## general settings and shortcuts
  WRFfiletypes = [] # WRF data source
  #WRFfiletypes += ['hydro']
  #WRFfiletypes += ['lsm']
  WRFfiletypes += ['srfc']
  #WRFfiletypes += ['xtrm']
  #WRFfiletypes += ['plev3d']
  # figure directory
  folder = '/home/me/Research/Dynamical Downscaling/Report/JClim Paper 2014/figures/'
  #folder = '/home/me/Research/Thesis/Report/Progress Report 2014/figures/'
#   folder = figure_folder
  lpickle = True
  # period shortcuts
  H01 = '1979-1980'; H02 = '1979-1981'; H03 = '1979-1982'; H30 = '1979-2009' # for tests 
  H05 = '1979-1984'; H10 = '1979-1989'; H15 = '1979-1994'; H60 = '1949-2009' # historical validation periods
  G10 = '1969-1979'; I10 = '1989-1999'; J10 = '1999-2009' # additional historical periods
  A03 = '2045-2048'; A05 = '2045-2050'; A09 = '2045-2054'; A10 = '2045-2055'; A15 = '2045-2060' # mid-21st century
  B03 = '2085-2088'; B05 = '2085-2900'; B10 = '2085-2095'; B15 = '2085-2100' # late 21st century  
  ltitle = True # plot/figure title
  figtitles = None
  subplot = None # subplot layout (or defaults based on number of plots)
  lbackground = True
  lcontour = True # contour or pcolor plot
  lframe = True # draw domain boundary
  loutline = True # draw boundaries around valid (non-masked) data
  framewidths = 1
  cbn = None # colorbar levels
  figuretype = None
  lsamesize = True
  lminor = True # draw minor tick mark labels
  lstations = True; stations = 'cities'
  lbasins = True; basinlist = ('ARB','FRB'); subbasins = {} #dict(ARB=('WholeARB','UpperARB','LowerCentralARB'))
  cbo = None # default based on figure type
  resolution = None # only for GPCC (None = default/highest)
  exptitles = None
  grid = None
  lWRFnative = False
  reflist = None # an additional list of experiments, that can be used to compute differences
  refprd = None; refdom = None
  ldiff = False # compute differences
  lfrac = False # compute fraction
  domain = (2,)
  variable_settings = None
  season_settings = None
  
  ## select variables and seasons
  variables = [] # variables
#   variables += ['Ts']
  variables += ['T2']
#   variables += ['Tmin', 'Tmax']
  variables += ['precip']
#  variables += ['waterflx']
#   variables += ['p-et']
#   variables += ['precipnc', 'precipc']
#   variables += ['Q2']
#   variables += ['evap']
#   variables += ['pet']
#  variables += ['runoff']
#  variables += ['sfroff']
#  variables += ['ugroff']
#   variables += ['snwmlt']
#   variables += ['snow']
#   variables += ['snowh']
#   variables += ['GLW','OLR','qtfx']
#   variables += ['SWDOWN','GLW','OLR']
#   variables += ['hfx','lhfx']
#   variables += ['qtfx','lhfr']
#   variables += ['SST']
#   variables += ['lat2D','lon2D']
  seasons = [] # seasons
#   seasons += ['cold']
#   seasons += ['warm']
#   seasons += ['melt']
  seasons += ['annual']
  seasons += ['summer']
  seasons += ['winter']
#   seasons += ['spring']    
#   seasons += ['fall']
  # special variable/season combinations
#   variables = ['seaice']; seasons = [8] # September seaice
#  variables = ['snowh'];  seasons = [8] # September snow height
#  variables = ['stns']; seasons = ['annual']
#   variables = ['lndcls']; seasons = [''] # static
#   variables = ['zs']; seasons = ['topo']; lcontour = True; WRFfiletypes = ['const'] if grid is None else ['const','srfc'] # static
#   variables = ['zs']; seasons = ['hidef']; WRFfiletypes=['const']; lcontour = True # static
  
  ## case settings
  
  # observations
  lprint = True # write plots to disk using case as a name tag
  maptype = 'lcc-new'; lstations = False; lbasins = True

# Fig. 2  
#   explist = ['max-ens']; period = H15
#   explist = ['Ens', 'Unity', 'max-ens', 'max-ens']; period = H15; domain = [None, None, 1, 2]
#   exptitles = ['CESM (80 km)','Merged Observations (10 km)', 'Outer WRF Domain (30 km)', 'Inner WRF Domain (10 km)']
#   case = 'valobs'; lsamesize = True; grid = 'arb2_d02'

# Fig. 3/4  
#   explist = ['Ens']; period = H15; grid = ['cesm1x1']
#   explist = ['max-ens']*3+['Ens']*3; grid = ['arb2_d02']*3+['cesm1x1']*3
#   seasons = ['annual', 'summer', 'winter']*2; period = H15
#   exptitles = ['WRF, 10 km ({:s} Average)']*3+['CESM ({:s} Average)']*3
#   exptitles = [model.format(season.title()) for model,season in zip(exptitles,seasons)]
#   case = 'val'; lsamesize = True; cbo = 'horizontal'
#   ldiff = True; reflist = ['Unity']*6; refprd = H15
#   variables = ['T2','precip']
#   seasons = [seasons] # only make one plot with all seasons!

# Fig. 5
  #explist = ['max-ens']; period = H15
#   explist = ['max', 'max-A', 'Unity', 'max-B', 'max-C', 'NARR']; period = H15
#   exptitles = ['WRF-1', 'WRF-2','Merged Observations', 'WRF-3', 'WRF-4', 'NARR (Reanalysis)']
#   case = 'val-ens'; lsamesize = False; grid = 'arb2_d02'
#   variables = ['precip']; seasons = ['summer'] 

# Fig. 6/7  
#   explist = ['max-ens-2050']*3+['Ens-2050']*3; grid = ['arb2_d02']*3+['cesm1x1']*3
#   seasons = ['annual', 'summer', 'winter']*2; period = A15
#   exptitles = ['WRF, 10 km ({:s} Average)']*3+['CESM ({:s} Average)']*3
#   exptitles = [model.format(season.title()) for model,season in zip(exptitles,seasons)]
#   case = 'prj'; lbasins = True; lsamesize = False; cbo = 'horizontal'
#   ldiff = True; reflist = ['max-ens']*3+['Ens']*3; refprd = H15  
#   seasons = [seasons] # only make one plot with all seasons!
#   variables = ['T2','precip']; variable_settings = ['T2_prj', 'precip_prj'] 

# Fig. 8
#   case = 'hydro'; lsamesize = False; cbo = 'vertical'; ltitle = True
#   variables = ['p-et']; seasons = [['annual', 'summer']]
#   exptitles = [r'Annual Average', r'Summer Average']
# top row
#   figtitles = r'WRF Ensemble Mean Net Precipitation $(P - ET)$' 
#   explist = ['max-ens']*2; period = H15  
# Fig. 8  (bottom row)
#   figtitles = r'Change in Net Precipitation $\Delta(P - ET)$' 
#   explist = ['max-ens-2050']*2; period = A15
#   ldiff = True; reflist = ['max-ens']; refprd = H15

# Fig. 13
  maptype = 'robinson'; lstations = False; lbasins = False; lminor = False; locean = True  
  case = 'cvdp'; lsamesize = False; cbo = 'horizontal'; ltitle = True
  variables = ['PDO_eof']; seasons = [None]; subplot = (2,1)
  exptitles = [r'HadISST', r'CESM Ensemble']; figtitles = r'Pacific Decadal Oscillation SST Pattern' 
  explist = ['HadISST_CVDP','Ctrl-1_CVDP']; period = H15

#   case = '3km'; stations = 'cities'
#   maptype = 'lcc-col'; lstations = True; lbasins = True # 'lcc-new'  
#   period = [H01]*4; period[1] = H15 
#   domain = [3,2,1,2]; lbackground = False
#   ldiff = True; reflist = ['Unity']; refprd = H30
#   grid = ['col2_d03','arb2_d02','col2_d01','col2_d02'] 
#   explist = ['max-3km','max-ctrl','max-3km','max-3km'] 
#   exptitles = ['WRF 3km','WRF 10km (15 yrs)','WRF 30km','WRF 10km']

#   maptype = 'lcc-large'; figuretype = 'largemap'; lstations = False; lbasins = True
#   period = None; lWRFnative = True; loutline = False; period = H10
#   explist = ['max']; exptitles = ' '; domain = (0,1,2)
#   case = 'arb2_frb'; basins = ('FRB',)
#   maptype = 'lcc-new'; lstations = False; lbasins = True
#   case = 'arb'; period = None; lWRFnative = True; lframe = False; loutline = False
#   explist = ['columbia']; exptitles = ' '; domain = (2,3)
# #   case = 'frb'; basins = ('FRB',)
#   case = 'arb'; basins = ('ARB',)
    
  if not case: raise ValueError, 'Need to define a \'case\' name!'
  
  # setup projection and map
  mapSetup = getARBsetup(maptype, lpickle=lpickle, folder=map_folder)
  
  ## load data
  if reflist is not None:
    if not isinstance(reflist,(list,tuple)): raise TypeError
    if len(explist) > len(reflist):
      if len(reflist) == 1: reflist *= len(explist)  
      else: raise DatasetError 
    lref = True    
  else: lref = False
  lbackground = not lref and lbackground
    
  loadlist = set(variables).union(('lon2D','lat2D','landmask','landfrac')) # landfrac is needed for CESM landmask
  exps, axtitles, nexps = loadDatasets(explist, n=None, varlist=loadlist, titles=exptitles, periods=period, 
                                       domains=domain, grids=grid, resolutions=resolution, 
                                       filetypes=WRFfiletypes, lWRFnative=lWRFnative, ltuple=True, 
                                       lbackground=lbackground, lautoregrid=True)
  nlen = len(exps)
  print exps[-1][-1]
  # load reference list
  if lref:
    if refprd is None: refprd = period    
    if refdom is None: refdom = domain
    refs, a, b = loadDatasets(reflist, n=None, varlist=loadlist, titles=None, periods=refprd, domains=refdom, 
                              grids=grid, resolutions=resolution, filetypes=WRFfiletypes, lWRFnative=lWRFnative, 
                              ltuple=True, lbackground=lbackground)
    # merge lists
    if len(exps) != len(refs): raise DatasetError, 'Experiments and reference list need to have the same length!'
    for i in xrange(len(exps)):
      if not isinstance(exps[i],tuple): raise TypeError 
      if not isinstance(refs[i],tuple): raise TypeError
      if len(exps[i]) != len(refs[i]): DatasetError, 'Experiments and reference tuples need to have the same length!'
      exps[i] = exps[i] + refs[i] # merge lists/tuples
  
  
  # get figure settings
  subplot = subplot or nlen
  sf, figformat, margins, caxpos, subplot, figsize, cbo = getFigureSettings(subplot, cbar=True, cbo=cbo, 
                                                             figuretype=figuretype, sameSize=lsamesize)
  if not ltitle: margins['top'] += 0.05
  
  # get projections settings
  projection, grid, res = mapSetup.getProjectionSettings()
  
  ## loop over variables and seasons
  maps = []; x = []; y = [] # projection objects and coordinate fields (only computed once)
  fn = -1 # figure counter
  N = len(variables)*len(seasons) # number of figures
  figtitles = checkItemList(figtitles, N, basestring, default=None)
  variable_settings = checkItemList(variable_settings, N, basestring, default=None)
  season_settings = checkItemList(season_settings, N, basestring, default=None)
  
#   if figtitles is not None:
#     if not isinstance(figtitles,(tuple,list)): figtitles = (figtitles,)*N
#     elif len(figtitles) != N: raise ValueError
     
  # start loop
  for varlist in variables:
    
    for sealist in seasons:
      
      # increment counter
      fn += 1
      M = len(exps) # number of panels
      
      # expand variables
      if isinstance(varlist,basestring): varstr = varlist
      elif isinstance(varlist,(list,tuple)):
        if all([var==varlist[0] for var in varlist]): varstr = varlist[0]
        else: varstr = ''.join([s[0] for s in varlist])
      else: varstr = ''
      varlist = checkItemList(varlist, M, basestring)
      # expand seasons
      if isinstance(sealist,basestring): seastr = '_'+sealist
      elif isinstance(sealist,(list,tuple)): seastr = '_'+''.join([s[0] for s in sealist])
      else: seastr = ''
      sealist = checkItemList(sealist, M, basestring)
      
      # get smart defaults for variables and seasons
      varlist_settings = variable_settings[fn] or varlist[0] # default: first variable
      if season_settings[fn]: sealist_settings = season_settings[fn]
      elif all([sea==sealist[0] for sea in sealist]): sealist_settings = sealist[0]
      else: sealist_settings = ''      
      # get variable properties and additional settings
      clevs, clim, cbl, clbl, cmap, lmskocn, lmsklnd, plottype = getVariableSettings(
                                                      varlist_settings, sealist_settings, 
                                                      ldiff=ldiff, lfrac=lfrac)

      # assemble filename      
      filename = varstr
      if ldiff: filename += '_diff'
      elif lfrac: filename += '_frac'
      filename += '{:s}_{:s}.{:s}'.format(seastr,case,figformat)
      # assemble plot title
      plat = exps[0][0].variables[varlist[0]].plot
      figtitle = figtitles[fn]
      if figtitle is None:
        figtitle = plottype + ' ' + plat['plottitle']
        if lfrac: figtitle += ' Fractions'
        if ldiff: figtitle += ' Differences'
        if plat['plotunits']: 
          if lfrac: figtitle += ' [%]'
          else: figtitle += ' [{:s}]'.format(plat['plotunits']) 
      
      # feedback
      print('\n\n   ***  %s %s (%s)   ***   \n'%(plottype,plat['plottitle'],varstr))
      
      ## compute data
      data = []; lons = []; lats=[]  # list of data and coordinate fields to be plotted 
      # compute average WRF precip            
      print(' - loading data ({0:s})'.format(varstr))
      for var,season,exptpl in zip(varlist,sealist,exps):
        lontpl = []; lattpl = []; datatpl = []                
        for exp in exptpl:
          expvar = exp.variables[var]
          expvar.load()
          #print expvar.name, exp.name, expvar.masked
          print(exp.name)
          assert expvar.gdal
          # handle dimensions
          if expvar.isProjected: 
            assert (exp.lon2D.ndim == 2) and (exp.lat2D.ndim == 2), 'No coordinate fields found!'
            if not exp.lon2D.data: exp.lon2D.load()
            if not exp.lat2D.data: exp.lat2D.load()
            lon = exp.lon2D.getArray(); lat = exp.lat2D.getArray()          
          else: 
            assert expvar.hasAxis('lon') and expvar.hasAxis('lat'), 'No geographic axes found!'
            lon, lat = np.meshgrid(expvar.lon.getArray(),expvar.lat.getArray())
          lontpl.append(lon); lattpl.append(lat) # append to data list
          # extract data field
          if expvar.hasAxis('time'):
            vardata = expvar.seasonalMean(season, asVar=False)
          else:
            vardata = expvar[:].squeeze()
          if expvar.masked: vardata.set_fill_value(np.NaN) # fill with NaN
          vardata = vardata.squeeze() # make sure it is 2D
          if 'scalefactor' in expvar.plot:
            vardata = vardata * expvar.plot['scalefactor'] # apply plot unit conversion
          # figure out ocean mask          
          if lmskocn:
            if exp.variables.has_key('landmask') and False:
              vardata[exp.landmask.getArray()] = -2.
            elif exp.variables.has_key('landfrac'): # CESM mostly 
              vardata[exp.landfrac.getArray(unmask=True,fillValue=0)<0.75] = -2. # use land fraction
            elif exp.variables.has_key('lndidx'): 
              mask = exp.lndidx.getArray()
              vardata[mask==16] = -2. # use land use index (ocean)  
              vardata[mask==24] = -2. # use land use index (lake)
            else :
              vardata = maskoceans(lon,lat,vardata,resolution=res,grid=grid)
          # figure out land mask
          if lmsklnd: 
            if exp.variables.has_key('landfrac'): # CESM and CFSR 
              vardata[exp.lnd.getArray(unmask=True,fillValue=0)>0.75] = 0 # use land fraction
            elif exp.variables.has_key('lndidx'): # use land use index (ocean and lake)
              mask = exp.lndidx.getArray(); tmp = vardata.copy(); vardata[:] = 0.
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
        ax.append(f.add_subplot(subplot[0],subplot[1],n+1, axisbg='blue'))
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
      if loutline or lframe:
        print(' - drawing data frames\n')
        for n in xrange(nax):
          for m in xrange(nexps[n]):   
            if loutline:
              bdy = ma.ones(data[n][m].shape); bdy[ma.getmaskarray(data[n][m])] = 0
              # N.B.: for some reason, using np.ones_like() causes a masked data array to fill with zeros  
              #print bdy.mean(), data[n][m].__class__.__name__, data[n][m].fill_value 
              bdy[0,:]=0; bdy[-1,:]=0; bdy[:,0]=0; bdy[:,-1]=0 # demarcate domain boundaries        
              maps[n].contour(x[n][m],y[n][m],bdy,[1,0,-1],ax=ax[n], colors='k', linewidths=framewidths, fill=False) # draw boundary of data
            if lframe:
              if isinstance(domain,(tuple,list)) and not ( domain[0] == 0 and m == 0):
                bdy = ma.ones(x[n][m].shape)   
                bdy[0,:]=0; bdy[-1,:]=0; bdy[:,0]=0; bdy[:,-1]=0 # demarcate domain boundaries        
                maps[n].contour(x[n][m],y[n][m],bdy,[1,0,-1],ax=ax[n], colors='k', linewidths=framewidths, fill=False) # draw boundary of domain
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
      if cbl is None:
        if cbn is None:
          if ( cbo == 'horizontal' and subplot[1] == 1 ): cbn = 5
          elif ( cbo == 'vertical' and subplot[0] == 1 ): cbn = 7
          elif ( cbo == 'horizontal' and subplot[1] == 2 ): cbn = 7
          elif ( cbo == 'vertical' and subplot[0] == 2 ): cbn = 9
          else: cbn = 9
        cbl = np.linspace(min(clevs),max(clevs),cbn)
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
          kwargs = dict()
          # white-out continents, if we have no proper land mask 
          if locean or ( lmsklnd and not (exps[n][0].variables.has_key('lndmsk') ) or exps[n][0].variables.has_key('lndidx')): 
            kwargs['maskland'] = True          
          if ldiff or lfrac or locean: 
            kwargs['ocean_color'] = 'white' ; kwargs['land_color'] = 'white'
          # misc annotatiosn
          mapSetup.miscAnnotation(bmap, **kwargs)
          # add parallels and meridians
          mapSetup.drawGrid(bmap, left=left, bottom=bottom, minor=lminor)
          # mark stations
          if lstations: mapSetup.markPoints(ax[n], bmap, pointset=stations)     
          # add ARB basin outline
          if lbasins:
            shpargs = dict(linewidth = 0.75) 
#             shpargs = dict(linewidth = 1.)
            for basin in basinlist:      
              basininfo = basins[basin]
              if basin in subbasins:
                for subbasin in subbasins[basin]:		  
                  bmap.readshapefile(basininfo.shapefiles[subbasin][:-4], subbasin, ax=axn, drawbounds=True, color='k', **shpargs)            
              else:
                bmap.readshapefile(basininfo.shapefiles['Whole'+basin][:-4], basin, ax=axn, drawbounds=True, color='k', **shpargs)            
          #print bmap.ARB_info                   
              
      # save figure to disk
      if lprint:
        print('\nSaving figure in '+filename)
        f.savefig(folder+filename, **sf) # save figure to pdf
        print(folder)
  
  ## show plots after all iterations
  pyl.show()

