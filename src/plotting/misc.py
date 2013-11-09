'''
Created on 2013-11-08

This module defines a MapSetup Class that carries parameters concerning map setup and annotation.
The class is intended for use the with plotting functions in this package.      

@author: Andre R. Erler, GPL v3
'''

rsphere = (6378137.00, 6356752.3142)

class MapSetup(object):
  ''' The MapSetup class that carries parameters concerning map setup and annotation and contains methods 
      to annotate map or axes objects with these data. '''
  
  def __init__(self, name=None, projection=None, resolution=None, grid=None, scale=None, point_markers=None, 
               lat_full=None, lat_half=None, lon_full=None, lon_half=None):
    ''' Construct a MapSetup instance from input parameters. '''
    self.name = name or 'N/A'
    # projection parameters
    if not isinstance(projection,dict): raise TypeError
    self.projection = projection
    if resolution is None and 'resolution' in projection: 
      self.resolution = projection['resolution']
    else: self.resolution = resolution
    self.grid = grid
    # map grid etc.
    self.lat_full = lat_full
    self.lat_half = lat_half
    self.lon_full = lon_full
    self.lon_half = lon_half
    self.scale = scale
    # more annotation
    self.point_markers = point_markers
     
  # get projection
  def getProjectionSettings(self):
    ''' return elements of the projection dict and a bit more; mostly legacy '''
    # return values
    return self.projection, self.grid, self.resolution

  # draw lat/lon grid
  def drawGrid(self, basemap, left=True, bottom=True):
    ''' add meridians and parallels; 'left' and 'bottom' indicate whether parallel and meridians are labeled '''
    # labels = [left,right,top,bottom]
    if self.lat_full is not None:
      basemap.drawparallels(self.lat_full,linewidth=1, labels=[left,False,False,False])
    if self.lat_half is not None:
      basemap.drawparallels(self.lat_half,linewidth=0.5, labels=[left,False,False,False])
    if self.lon_full is not None:
      basemap.drawmeridians(self.lon_full,linewidth=1, labels=[False,False,False,bottom])
    if self.lon_half is not None:
      basemap.drawmeridians(self.lon_half,linewidth=0.5, labels=[False,False,False,bottom])
      
  # draw map scale
  def drawScale(self, basemap):
    ''' add a map scale to the map axes '''
    if self.scale is not None:
      basemap.drawmapscale(*self.scale, barstyle='fancy', fontsize=8, yoffset=0.01*(basemap.ymax-basemap.ymin))
      
  # misc annotations that I usually do
  def miscAnnotation(self, basemap, blklnd=False):
    ''' add coastlines, countries, color ocean and background etc. '''
    # land/sea mask
    basemap.drawlsmask(ocean_color='blue', land_color='green',resolution=self.resolution,grid=self.grid)
    if blklnd: basemap.fillcontinents(color='black',lake_color='black') # mask land
    # add maps stuff
    basemap.drawcoastlines(linewidth=0.5)
    basemap.drawcountries(linewidth=0.5)
    basemap.drawmapboundary(fill_color='k',linewidth=2)    
      
  # mark stations
  def markStations(self, ax, basemap):
    ''' mark points and label them '''
    if self.point_markers is not None:
      for name,lon,lat in self.point_markers:
        xx,yy = basemap(lon, lat)
        basemap.plot(xx,yy,'ko',markersize=3)
        ax.text(xx+1.5e4,yy-1.5e4,name,ha='left',va='top',fontsize=8)


## figure settings
def getFigureSettings(nexp, cbo, folder):
  sf = dict(dpi=150) # print properties
  figformat = 'png'  
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
