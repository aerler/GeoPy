'''
Created on 2013-11-08

This module defines a MapSetup Class that carries parameters concerning map setup and annotation.
The class is intended for use the with plotting functions in this package.      

@author: Andre R. Erler, GPL v3
'''

import pickle, os
from mpl_toolkits.basemap import Basemap

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
    # initialize basemap object
    self.basemap = Basemap(**projection) # make just one basemap with dummy axes handle
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


## function that serves a MapSetup instance with complementary pickles
def getMapSetup(lpickle=False, folder=None, name=None, **kwargs):
  ''' function that serves a MapSetup instance with complementary pickles '''
  # handle pickling
  if lpickle:
    if not isinstance(folder,basestring): raise TypeError 
    if not os.path.exists(folder): raise IOError
    filename = '{0:s}/{1:s}.pickle'.format(folder,name)
    if os.path.exists(filename):
      # open existing MapSetup from pickle
      filehandle = open(filename, 'r')
      mapSetup = pickle.load(filehandle)
      filehandle.close()
    else:
      # create new MapSetup and also pickle it
      mapSetup = MapSetup(name=name, **kwargs)
      filehandle = open(filename, 'w')
      pickle.dump(mapSetup, filehandle)
      filehandle.close()
  else:
    # instantiate object
    mapSetup = MapSetup(name=name, **kwargs)
  # return MapSetup instance
  return mapSetup
