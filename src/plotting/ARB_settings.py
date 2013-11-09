'''
Created on 2013-10-10

Meta data related to the Athabasca River Basin downscaling project; primarily map annotation. 

@author: Andre R. Erler, GPL v3
'''

from plotting.misc import MapSetup

## Annotation

# ST_LINA, WESTLOCK_LITKE, JASPER, FORT_MCMURRAY, SHINING_BANK
station_list = [('SL',-111.45,54.3), ('WL',-113.85,54.15), ('J',-118.07,52.88), 
            ('FM',-111.22,56.65), ('SB',-115.97,53.85)]

# parallels and meridians
annotation_dict = dict()
## Lambert Conic Conformal - Very High Resolution Coast Mountains
annotation_dict['lcc-coast'] = dict()
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
annotation_dict['lcc-large'] = dict(scale=(-171, 21, -137, 57, 2000), lat_full=[0,30,60,90], lat_half=[15,45,75], 
                               lon_full=[-180,-150,-120,-90,-60], lon_half=[-165,-135,-105,-75])
## Lambert Azimuthal Equal Area
annotation_dict['laea'] = annotation_dict['lcc-large']   
## Orthographic Projection
annotation_dict['ortho-NA'] = dict(scale=None, lat_full=range(-90,90,30), lat_half=None, lon_full=range(-90,90,30), lon_half=None)


## setup projection: lambert conformal
# common variables
rsphere = (6378137.00,6356752.3142); grid = 10
# lon_0,lat_0 is central point. lat_ts is latitude of true scale.
projection_dict = dict()
## Lambert Conic Conformal - Very High Resolution Coast Mountains
projection_dict['lcc-coast'] = dict(projection='lcc', lat_0=51, lon_0=-125, lat_1=51, rsphere=rsphere,
              width=50*10e3, height=50*10e3, area_thresh = 500., resolution='i')
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
projection_dict['lcc-large'] = dict(projection='lcc', lat_0=54.5, lon_0=-140, lat_1=53, #rsphere=rsphere,
              width=11000e3, height=7500e3, area_thresh = 10e3, resolution='l')
## Lambert Azimuthal Equal Area
projection_dict['laea'] = dict(projection='laea', lat_0=57, lon_0=-137, lat_ts=53, resolution='l', #
              width=259*30e3, height=179*30e3, rsphere=rsphere, area_thresh = 1000.)  
## Orthographic Projection
projection_dict['ortho-NA'] = dict(projection='ortho', lat_0 = 75, lon_0 = -137, resolution = 'l', area_thresh = 1000.)


## function to actuall get a MapSetup object for the ARB region
def getMapSetup(projection, annotation=None, stations=True):
  ''' return a MapSetup object with data for the chosen ARB setting '''
  # projection
  proj = projection_dict[projection]
  # annotation
  if annotation is None:
    if projection in annotation_dict: anno = annotation_dict[projection]
  else: anno = annotation_dict[annotation]
  # stations
  if stations: stat = station_list
  else: stat = None
  # instantiate object
  mapSetup = MapSetup(name=projection, projection=proj, grid=10, point_markers=stat, **anno)
  # return object
  return mapSetup
  

# create pickles
if __name__ == '__main__':

  

  #TODO: pull creation of projection object into this function
  #TODO: add optional loading from pickled object    
  # loop over projections
  for name,proj in projection_dict.items():
    
    # test retrieval function
    test = getMapSetup(name)
    assert test.projection is proj
    print test.name
    
    #TODO: generate projection object
    
    #TODO: pickle object
    pass
  