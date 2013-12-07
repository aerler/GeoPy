'''
Created on 2013-10-10

Meta data related to the Athabasca River Basin downscaling project; primarily map annotation. 

@author: Andre R. Erler, GPL v3
'''

from datasets.common import grid_folder
from plotting.mapsetup import getMapSetup

arb_figure_folder = '/home/me/Research/Dynamical Downscaling/Figures/'
arb_map_folder = arb_figure_folder + '.mapsetup/'
arb_shapefolder = grid_folder+'/Athabasca_River_Basin/'
arb_shapefile = arb_shapefolder+'ARB_Basins_Outline_WGS84' # basemap automatically appends '.shp'
# N.B.: basemap can only read shapefiles in geographic projection; use this GDAL command to convert:
#       $ogr2ogr -t_srs WGS84 new_shapefile_WGS84.shp old_shapefile_in_projected.shp
#    or $ogr2ogr -t_srs EPSG:4326 new_shapefile_WGS84.shp old_shapefile_in_projected.shp

## Annotation
station_dict = dict()
# ST_LINA, WESTLOCK_LITKE, JASPER, FORT_MCMURRAY, SHINING_BANK
station_dict['ARB'] = [('SL',-111.45,54.3), ('WL',-113.85,54.15), ('J',-118.07,52.88), 
            ('FM',-111.22,56.65), ('SB',-115.97,53.85)]
station_dict['cities'] = [('J',-118.07,52.88),('V',-123.1,49.25),('C',-114.07,51.05),
                      ('E',-113.5,53.53),('PG',-122.75,53.92)]
station_dict['default'] = station_dict['cities'] # default point markers

# parallels and meridians
annotation_dict = dict()
## Lambert Conic Conformal - Very High Resolution Coast Mountains
annotation_dict['lcc-coast'] = dict(scale=(-127, 49.25, -125, 51, 100), lat_full=[50,55], lat_half=[48,49,51,52,53,54], 
                             lon_full=[-125,-120], lon_half=[-127.5,-122.5,-117.5])
## Lambert Conic Conformal - Columbia Icefield
annotation_dict['lcc-col'] = dict(scale=(-116.75, 48.25, -120, 55, 400), lat_full=[40,50,60,70], lat_half=[45,55,65], 
                             lon_full=[-160,-140,-120,-100], lon_half=[-150,-130,-110])
## Lambert Conic Conformal - Athabasca River Basin
annotation_dict['lcc-arb'] = dict(scale=(-111, 52, -120, 55, 400), lat_full=[50,60], lat_half=[45,55,65], 
                             lon_full=[-120,-110], lon_half=[-125,-115,-105])
## Lambert Conic Conformal - New Fine Domain
annotation_dict['lcc-new'] = dict(scale=(-128, 48, -120, 55, 400), lat_full=[40,50,60,70], lat_half=[45,55,65], 
                             lon_full=[-180,-160,-140,-120,-100], lon_half=[-170,-150,-130,-110])
## Lambert Conic Conformal - Fine Domain
annotation_dict['lcc-fine'] = dict(scale=(-136, 49, -137, 57, 800), lat_full=[45,65], lat_half=[55,75], 
                              lon_full=[-180,-160,-140,-120,-100], lon_half=[-170,-150,-130,-110])
## Lambert Conic Conformal - Small Domain
annotation_dict['lcc-small'] = annotation_dict['lcc-fine']
## Lambert Conic Conformal - Intermed Domain
annotation_dict['lcc-intermed'] = annotation_dict['lcc-fine']
## Lambert Conic Conformal - Large Domain
annotation_dict['lcc-large'] = dict(scale=(-140, 22, -120, 53, 2000), lat_full=[0,30,60,90], lat_half=[15,45,75], 
                               lon_full=[120,150,-180,-150,-120,-90,-60,-30], lon_half=[135,165,-165,-135,-105,-75,-45])
## Lambert Azimuthal Equal Area
annotation_dict['laea'] = annotation_dict['lcc-large']   
## Orthographic Projection
annotation_dict['ortho-NA'] = dict(scale=None, lat_full=range(-90,90,30), lat_half=None, lon_full=range(-180,180,30), lon_half=None)


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
## Lambert Conic Conformal - Very High Resolution Coast Mountains
projection_dict['lcc-coast'] = dict(projection='lcc', lat_0=51, lon_0=-125, lat_1=51, rsphere=rsphere,
              width=50*10e3, height=50*10e3, area_thresh = 500., resolution='i')
## Lambert Conic Conformal - Columbia Icefield
projection_dict['lcc-col'] = dict(projection='lcc', lat_0=52., lon_0=-120., lat_1=53, rsphere=rsphere,
              width=100*10e3, height=100*10e3, area_thresh = 500., resolution='l')
## Lambert Conic Conformal - Athabasca River Basin
projection_dict['lcc-arb'] = dict(projection='lcc', lat_0=55.5, lon_0=-114.5, lat_1=55, rsphere=rsphere,
              width=110*10e3, height=110*10e3, area_thresh = 500., resolution='l')
## Lambert Conic Conformal - New Fine Domain
projection_dict['lcc-new'] = dict(projection='lcc', lat_0=55, lon_0=-120, lat_1=52, rsphere=rsphere,
              width=180*10e3, height=180*10e3, area_thresh = 1000., resolution='l')
## Lambert Conic Conformal - Fine Domain
projection_dict['lcc-fine'] = dict(projection='lcc', lat_0=58, lon_0=-132, lat_1=53, rsphere=rsphere,
              width=200*10e3, height=300*10e3, area_thresh = 1000., resolution='l')
## Lambert Conic Conformal - Small Domain
projection_dict['lcc-small'] = dict(projection='lcc', lat_0=56, lon_0=-130, lat_1=53, rsphere=rsphere,
              width=2500e3, height=2650e3, area_thresh = 1000., resolution='l')
## Lambert Conic Conformal - Intermed Domain
projection_dict['lcc-intermed'] = dict(projection='lcc', lat_0=57, lon_0=-140, lat_1=53, rsphere=rsphere,
              width=4000e3, height=3400e3, area_thresh = 1000., resolution='l')
## Lambert Conic Conformal - Large Domain
projection_dict['lcc-large'] = dict(projection='lcc', lat_0=50, lon_0=-130, lat_1=50, #rsphere=rsphere,
              width=11000e3, height=7500e3, area_thresh = 10e3, resolution='l')
## Lambert Azimuthal Equal Area
projection_dict['laea'] = dict(projection='laea', lat_0=57, lon_0=-137, lat_ts=53, resolution='l', #
              width=259*30e3, height=179*30e3, rsphere=rsphere, area_thresh = 1000.)  
## Orthographic Projection
projection_dict['ortho-NA'] = dict(projection='ortho', lat_0 = 50, lon_0 = -130, resolution = 'l', area_thresh = 1000.)


## function to actually get a MapSetup object for the ARB region
def getARBsetup(projection, annotation=None, stations=None, lpickle=False, folder=None, lrm=False):
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

  proj_list = None
#   proj_list = ['lcc-coast']

  if proj_list is None: proj_list = projection_dict.keys()    
  # loop over projections
  for name in proj_list:
    proj = projection_dict[name]
    
    # test retrieval function
    test = getARBsetup(name, lpickle=True, stations=None, folder=arb_map_folder, lrm=True)
    print test.name
    print test
    print test.point_markers
      