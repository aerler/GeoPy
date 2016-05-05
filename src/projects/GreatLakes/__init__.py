'''
Created on 2016-04-13

A package that contains settings for the GreatLakes region projects for use with the geodata package.

@author: Andre R. Erler, GPL v3
'''

# import figure settings
from figure_settings import getVariableSettings, getFigureSettings, figure_folder

# import map projection settings (basemap)
try: 
  from map_settings import getSetup, map_folder
except ImportError: 
  print("Error importing map settings - 'basemap' is likely not installed.")

## import load functions with GreatLakes experiments into local namespace

# import relevant WRF experiments
from WRF_experiments import WRF_exps, WRF_ens
# import WRF load functions
from WRF_experiments import loadWRF, loadWRF_Shp, loadWRF_Stn, loadWRF_TS, loadWRF_ShpTS, loadWRF_StnTS, loadWRF_Ensemble, loadWRF_ShpEns, loadWRF_StnEns

# also load CESM experiments and functions
from projects.CESM_experiments import CESM_exps, CESM_ens
# import CESM load functions
from projects.CESM_experiments import loadCESM, loadCESM_Shp, loadCESM_Stn, loadCESM_TS, loadCESM_ShpTS, loadCESM_StnTS, loadCESM_Ensemble, loadCESM_ShpEns, loadCESM_StnEns

# add relevant experiments to general load functions
from datasets.common import loadDataset, loadClim, loadShpTS, loadStnTS, loadEnsembleTS, addLoadFcts
# modify functions (wont affect modified WRF/CESM functions)
addLoadFcts(locals(), locals(), WRF_exps=WRF_exps, WRF_ens=WRF_ens, CESM_exps=CESM_exps, CESM_ens=CESM_ens)
