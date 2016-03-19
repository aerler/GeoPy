'''
Created on 2013-10-10

Meta data related to the Athabasca River Basin downscaling project; primarily map annotation. 

@author: Andre R. Erler, GPL v3
'''
import numpy as np
from plotting.mapsetup import getMapSetup
from datasets.WSC import basins_info

figure_folder = '/scratch/fengyi/data/Figures/'
map_folder = figure_folder + '.mapsetup/'
# actual Athabasca River Basin (shape file from Aquanty)
#ARB_shapefolder = grid_folder+'/ARB_Aquanty/' 
#ARB_shapefile = ARB_shapefolder+'ARB_Basins_Outline_WGS84'
shape_folder = '/data/WSC/'
# Athabasca River Basin (shape file from Atlas of Canada) 
GLB_Info = basins_info['GLB']
GLB_shapefolder = GLB_Info.folder
GLB_shapefiles = GLB_Info.shapefiles
# Fraser River Basin
GRW_Info = basins_info['GRW']
GRW_shapefolder = GRW_Info.folder
GRW_shapefiles = GRW_Info.shapefiles
# N.B.: basemap can only read shapefiles in geographic projection; use this GDAL command to convert:
#       $ogr2ogr -t_srs WGS84 new_shapefile_WGS84.shp old_shapefile_in_projected.shp
#    or $ogr2ogr -t_srs EPSG:4326 new_shapefile_WGS84.shp old_shapefile_in_projected.shp

## Annotation
station_dict = dict()
# ST_LINA, WESTLOCK_LITKE, JASPER, FORT_MCMURRAY, SHINING_BANK
#station_dict['ARB'] = [('SL',-111.45,54.3), ('WL',-113.85,54.15), ('J',-118.07,52.88), 
#            ('FM',-111.22,56.65), ('SB',-115.97,53.85)]
#station_dict['cities'] = [('J',-118.07,52.88),('V',-123.1,49.25),('C',-114.07,51.05),
#                      ('E',-113.5,53.53),('PG',-122.75,53.92)]
station_dict['default'] = [] #station_dict['cities'] # default point markers

# parallels and meridians
annotation_dict = dict()
## Lambert Conic Conformal - Great Lakes Basin
annotation_dict['lcc-glb'] = dict(scale=(-90.5, 39.4, -82.5, 46, 400), lat_full=[30,40,50,60], lat_half=[45,55,65], 
                             lon_full=[-70,-80,-90,-100], lon_half=[-75,-85,-95])
## Lambert Conic Conformal - Grand River Watershed
annotation_dict['lcc-grw'] = dict(scale=(-79.7, 44.3, -80.25, 43.5, 40), lat_full=range(40,45), lat_half=np.arange(40.5,45), 
                             lon_full=range(-75,-85,-1), lon_half=np.arange(-75.5,-85,-1))
## Lambert Conic Conformal - Large Domain
annotation_dict['lcc-large'] = dict(scale=(-140, 22, -120, 53, 2000), lat_full=[0,30,60,90], lat_half=[15,45,75], 
                               lon_full=[120,150,-180,-150,-120,-90,-60,-30], lon_half=[135,165,-165,-135,-105,-75,-45])
annotation_dict['lcc-can'] = dict(scale=(-85, 38, -100, 55, 800), 
                                  lat_full=[30,40,50,60,70,80], lat_half=[35,45,55,65,75], 
                                  lon_full=[-180,-160,-140,-120,-100,-80,-60,-40,-20,0], 
                                  lon_half=[-170,-150,-130,-110,-90,-70,-50,-30,-10])
## Lambert Azimuthal Equal Area
annotation_dict['laea'] = annotation_dict['lcc-large']   
## Orthographic Projection
annotation_dict['ortho-NA'] = dict(scale=None, lat_full=range(-90,90,30), lat_half=None, lon_full=range(-180,180,30), lon_half=None)
## Global Robinson Projection
annotation_dict['robinson'] = dict(scale=None, lat_full=range(-60,100,60), lon_full=range(-180,200,120),
                                               lat_half=range(-90,100,60), lon_half=range(-120,160,120))
 


## setup projection: lambert conformal
# lat_1 is first standard parallel.
# lat_2 is second standard parallel (defaults to lat_1).
# lon_0,lat_0 is central point.
# rsphere=(6378137.00,6356752.3142) specifies WGS4 ellipsoid
# area_thresh=1000 means don't plot coastline features less
# than 1000 km^2 in area.
# common variables
rsphere = (6378137.00,6356752.3142); grid = 10
# lon_0,lat_0 is central point. lat_ts is latitude of true scale.
projection_dict = dict()
## Lambert Conic Conformal - Great Lakes Basin
projection_dict['lcc-glb'] = dict(projection='lcc', lat_0=46, lon_0=-83, lat_1=45, rsphere=rsphere,
              width=180*10e3, height=160*10e3, area_thresh = 1e3, resolution='l')
## Lambert Conic Conformal - Grand River Watershed
projection_dict['lcc-grw'] = dict(projection='lcc', lat_0=43.5, lon_0=-80.25, lat_1=43.5, rsphere=rsphere,
              width=15*10e3, height=20*10e3, area_thresh = 1e3, resolution='i')
## Lambert Conic Conformal - Canada
projection_dict['lcc-can'] = dict(projection='lcc', lat_0=52.5, lon_0=-105, lat_1=52.5, rsphere=rsphere,
              width=5000e3, height=4500e3, area_thresh = 10e3, resolution='l')
## Lambert Conic Conformal - Large Domain
projection_dict['lcc-large'] = dict(projection='lcc', lat_0=50, lon_0=-130, lat_1=50, #rsphere=rsphere,
              width=11000e3, height=7500e3, area_thresh = 10e3, resolution='l')
## Lambert Azimuthal Equal Area
projection_dict['laea'] = dict(projection='laea', lat_0=57, lon_0=-137, lat_ts=53, resolution='l', #
              width=259*30e3, height=179*30e3, rsphere=rsphere, area_thresh = 1000.)  
## Orthographic Projection
projection_dict['ortho-NA'] = dict(projection='ortho', lat_0 = 50, lon_0 = -130, resolution = 'l', area_thresh = 1000.)
## Global Robinson Projection
projection_dict['robinson'] = dict(projection='robin', lat_0 = 0, lon_0 = 180, resolution = 'l', area_thresh = 100000.)


## function to actually get a MapSetup object for the ARB region
def getSetup(projection, annotation=None, stations=None, lpickle=False, folder=None, lrm=False):
  ''' return a MapSetup object with data for the chosen ARB setting '''
  # projection
  proj = projection_dict[projection]
  # annotation
  if annotation is None:
    if projection in annotation_dict: anno = annotation_dict[projection]
  else: anno = annotation_dict[annotation]
  # station markers
  if stations is None:
    stations = station_dict
  else:
    if not isinstance(stations,basestring): raise TypeError
    stations = station_dict[stations]
  mapSetup = getMapSetup(lpickle=lpickle, folder=folder, lrm=lrm, # pickle arguments; the rest is passed on to MapSetup 
                         name=projection, projection=proj, grid=10, point_markers=stations, **anno)
  # return object
  return mapSetup


# create pickles
if __name__ == '__main__':

#   proj_list = None
  proj_list = ['lcc-glb']

  if proj_list is None: proj_list = projection_dict.keys()    
  # loop over projections
  for name in proj_list:
    proj = projection_dict[name]
    
    # test retrieval function
    test = getSetup(name, lpickle=True, stations=None, folder=map_folder, lrm=True)
    print test.name
    print test
    print test.point_markers
      
