'''
Created on 2013-11-13

This module defines variable and figure properties commonly used in plots.      

@author: Andre R. Erler, GPL v3
'''

import matplotlib as mpl
import numpy as np
from geodata.base import VariableError
from datasets.common import name_of_month # for annotation; days_per_month, days_per_month_365, 

# my own colormap
cdict = dict()
cdict['red'] = ((0,0,0,),(0.5,1,1),(1,1,1))
cdict['blue'] = ((0,1,1,),(0.5,1,1),(1,0,0))
cdict['green'] = ((0,0,0,),(0.5,1,1),(1,0,0))
mycmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)

## variable settings and seasons
def getVariableSettings(var, season, oldvar='', ldiff=False, lfrac=False):
        ## settings
  # plot variable and averaging 
  cbl = None; clim = None       
  lmskocn = False; lmsklnd = False # mask ocean or land?
  # color maps and   scale (contour levels)
  if ldiff:
    cmap = mycmap; cmap.set_over('red'); cmap.set_under('blue')
    if var in ('T2','Ts','Tmin','Tmax','Tmean'):
      clevs = np.linspace(-5,5,21); clbl = '%3.1f' # K
    elif var in ('evap','pet','precip','precipc','precipnc'):
      clevs = np.linspace(-5,5,21); clbl = '%3.1f' # mm/day
    elif var in ('snwmlt', 'runoff', 'ugroff', 'sfroff','p-et','waterflx'): # moisture fluxes (kg /(m^2 s))
      clevs = np.linspace(-2,2,21); clbl = '%3.1f' # mm/day  
    elif var == 'zs':
      clevs = np.linspace(-0.5,0.5,21); clbl = '%3.1f' # mm/day
    else: 
      raise VariableError, 'No settings found for differencing variable \'{0:s}\' found!'.format(var)
  elif lfrac:
    cmap = mycmap; cmap.set_over('red'); cmap.set_under('blue')
    if var in ('T2','Ts','Tmin','Tmax','Tmean'):
      clevs = np.linspace(-3,3,21); clbl = '%3.0f\%' 
    elif var in ('evap','pet','p-et','precip','precipc','precipnc','waterflx'):
      clevs = np.linspace(-100,100,21); clbl = '%3.0f'  
    else: 
      clevs = np.linspace(-50,50,21); clbl = '%3.0f'  
  else:
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
    elif var == 'pet': # potential evaporation
      clevs = np.linspace(0,6,26); clbl = '%02.1f' # mm/day
      if season == 'winter': clevs -= 2
      elif season == 'summer': clevs += 2    
    elif var == 'p-et' or var == 'waterflx': # moisture fluxes (kg /(m^2 s))
      # clevs = np.linspace(-3,22,51); clbl = '%02.1f'
      clevs = np.linspace(-2,2,25); cmap = mpl.cm.PuOr; clbl = '%02.1f'
    elif var in ('snwmlt', 'runoff', 'ugroff', 'sfroff'): # moisture fluxes (kg /(m^2 s))
      # clevs = np.linspace(-3,22,51); clbl = '%02.1f'
      clevs = np.linspace(0,8,25); clbl = '%02.1f'; cmap = mpl.cm.YlGnBu
    elif var == 'precip' or var == 'precipnc': # total precipitation 
      clevs = np.linspace(0,20,41); clbl = '%02.1f' # mm/day
    elif var == 'precipc': # convective precipitation 
      clevs = np.linspace(0,5,26); clbl = '%02.1f' # mm/day
    elif var == 'Q2':
      clevs = np.linspace(0,15,31); clbl = '%02.1f' # mm/day
    elif oldvar=='SST' or var=='SST' or oldvar=='Ts' or var=='Ts': # skin temperature (SST)
      clevs = np.linspace(240,305,66); clbl = '%03.0f' # K
      if oldvar=='SST' or var=='SST': lmsklnd = True # mask land for SST
      var = 'Ts'; oldvar = var
    elif var=='T2' or var=='Tmin' or var=='Tmax' or var=='Tmean': # 2m or skin temperature (SST)
      clevs = np.linspace(255,290,36); clbl = '%03.0f' # K
      if season == 'winter': clevs -= 10
      elif season == 'summer': clevs += 10
      if var=='Tmin': clevs -= 5
      elif var=='Tmax': clevs += 5
    elif var == 'seaice': # sea ice fraction
      lmsklnd = True # mask land        
      clevs = np.linspace(0.04,1,25); clbl = '%2.1f' # fraction
      cmap.set_under('white')
    elif var == 'zs': # surface elevation / topography
      if season == 'topo':
        lmskocn = True; clim = (-1.,2.5); # nice geographic map feel
        clevs = np.hstack((np.array((-1.5,-1,-0.5)), np.linspace(-0,2.5,51))); clbl = '%02.1f' # km
        cmap = mpl.cm.gist_earth; cmap.set_over('white'); cmap.set_under('blue') # topography
      elif season == 'hidef': 
        lmskocn = True; clim = (-0.5,2.5); # good contrast for high elevation
        clevs = np.hstack((np.array((-.5,)), np.linspace(0,2.5,26))); clbl = '%02.1f' # km
        cmap = mpl.cm.gist_ncar; cmap.set_over('white'); cmap.set_under('blue')
      else: 
        raise ValueError, 'No map color scheme defined (use \'season\' to select color scheme).'
      cbl = np.linspace(0,clim[-1],6)
    elif var=='stns': # station density
      clevs = np.linspace(0,5,6); clbl = '%2i' # stations per grid points  
      cmap.set_over('purple'); cmap.set_under('white')      
    elif var=='lndcls': # land use classes (works best with contour plot)
      clevs = np.linspace(0.5,24.5,25); cbl = np.linspace(4,24,6)  
      clbl = '%2i'; cmap.set_over('purple'); cmap.set_under('white')
    elif var=='lon2D': 
      clevs = np.linspace(-130,-100,30); clbl = '%02.0d'
    elif var=='lat2D': 
      clevs = np.linspace(30,60,30); clbl = '%02.0d'            
    else: 
      raise VariableError, 'No settings for variable \'{0:s}\' found!'.format(var) 
  # time frame / season
  if isinstance(season,str):
    if season == 'annual':  # all month
      month = range(1,13); plottype = 'Annual'
    elif season == 'cold': # DJF
      month = [10, 11, 12, 1, 2, 3]; plottype = 'Cold Season'    
    elif season == 'warm': # DJF
      month = [4, 5, 6, 7, 8, 9]; plottype = 'Warm Season'
    elif season == 'melt': # AMJ
      month = [4, 5, 6]; plottype = 'Melt Season'          
    elif season == 'OND': # DJF
      month = [10, 11, 12]; plottype = 'Oct.-Dec.'    
    elif season == 'winter': # DJF
      month = [12, 1, 2]; plottype = 'Winter'
    elif season == 'spring': # MAM
      month = [3, 4, 5]; plottype = 'Spring'
    elif season == 'summer': # JJA
      month = [6, 7, 8]; plottype = 'Summer'
    elif season == 'fall': # SON
      month = [9, 10, 11]; plottype = 'Fall'
    else:
      plottype = '' # for static fields
      month = [1]
  else:                
    month = season      
    if len(season) == 1 and isinstance(season[0],int):
      plottype =  '%s Average'%name_of_month[season[0]].strip()
      season = '%02i'%(season[0]+1) # number of month, used for file name
    else: plottype = 'Average'    
  if plottype is not '':
    if ldiff: plottype += ' Difference'
    elif lfrac: plottype += ' Difference'
    else: plottype += ' Average'
  # return
  return clevs, clim, cbl, clbl, cmap, lmskocn, lmsklnd, plottype, month


## figure settings
def getFigureSettings(nexp, cbar=True, cbo=None, figuretype=None, sameSize=True):
  sf = dict(dpi=150) # print properties
  figformat = 'png'
  # some special cases 
  if figuretype is not None:
    if figuretype == 'largemap':
      # copied from 6 panel figure
      nexp = 0 # skip pane settings 
      subplot = (1,1) # rows, columns
      figsize = (9.25,6.5) # width, height (inches)
      if cbar:
        margins = dict(bottom=0.09, left=0.05, right=.97, top=.92, hspace=0.1, wspace=0.05)
        cbo = cbo or 'horizontal'; caxpos = [0.05, 0.025, 0.9, 0.03]
      else:
        margins = dict(bottom=0.025, left=0.05, right=.97, top=.92, hspace=0.1, wspace=0.05)
  # pane settings
  if nexp == 1:
    ## 1 panel
    subplot = (1,1)
    if sameSize: figsize = (6.25,6.25)
    else: figsize = (3.75,3.75) # figsize = (7,5.5)
    if cbar:
      margins = dict(bottom=0.125, left=0.1, right=0.95, top=0.925, hspace=0.0, wspace=0.0)
      caxpos = [0.05, 0.05, 0.9, 0.03]
#       caxpos = [0.91, 0.05, 0.03, 0.9]
      cbo = cbo or 'horizontal'
    else:
      margins = dict(bottom=0.085, left=0.13, right=0.975, top=0.94, hspace=0.0, wspace=0.0)
    #     margins = dict(bottom=0.12, left=0.075, right=.9725, top=.95, hspace=0.05, wspace=0.05)
    #    margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
  elif nexp == 2:
    ## 2 panel
    subplot = (1,2)
    if sameSize: figsize = (6.25,6.25)
    else: figsize = (6.25,5.5)    
    if cbar:
      margins = dict(bottom=0.1, left=0.085, right=.975, top=.95, hspace=0.05, wspace=0.05)
      caxpos = [0.05, 0.05, 0.9, 0.03]
      cbo = cbo or 'horizontal'
    else:
      margins = dict(bottom=0.055, left=0.085, right=.975, top=.95, hspace=0.05, wspace=0.05)
  elif nexp == 4:
    # 4 panel
    subplot = (2,2)
    figsize = (6.25,6.25)
    if cbar:
      margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
      caxpos = [0.91, 0.05, 0.03, 0.9]
      cbo = cbo or 'vertical'
    else:
      margins = dict(bottom=0.05, left=0.08, right=.985, top=.96, hspace=0.10, wspace=0.02)
  elif nexp == 6:
    # 6 panel
    subplot = (2,3) # rows, columns
    figsize = (9.25,6.5) # width, height (inches)
    if cbar:
      margins = dict(bottom=0.09, left=0.05, right=.97, top=.92, hspace=0.1, wspace=0.05)
      cbo = cbo or 'horizontal'
      #    margins = dict(bottom=0.025, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
      caxpos = [0.05, 0.025, 0.9, 0.03]
    else:
      margins = dict(bottom=0.025, left=0.05, right=.97, top=.92, hspace=0.1, wspace=0.05)
#   # figure out colorbar placement
#   if cbo == 'vertical':
#     margins = dict(bottom=0.02, left=0.065, right=.885, top=.925, hspace=0.05, wspace=0.05)
#     caxpos = [0.91, 0.05, 0.03, 0.9]
#   else: # 'horizontal'
#     margins = dict(bottom=0.1, left=0.065, right=.9725, top=.925, hspace=0.05, wspace=0.05)
#     caxpos = [0.05, 0.05, 0.9, 0.03]        
  # return values
  if cbar: 
    return sf, figformat, margins, caxpos, subplot, figsize, cbo
  else:
    return sf, figformat, margins, subplot, figsize
