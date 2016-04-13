'''
Created on 2016-04-13

A package that contains settings for the GreatLakes region projects for use with the geodata package.

@author: Andre R. Erler, GPL v3
'''

# import figure settings
from figure_settings import getVariableSettings, getFigureSettings

# import map projection settings (basemap)
#from map_settings import getSetup

## import load functions with GreatLakes experiments into local namespace

# import relevant WRF experiments
from WRF_experiments import WRF_exps, WRF_ens
# import WRF load functions
from WRF_experiments import loadWRF, loadWRF_Shp, loadWRF_Stn, loadWRF_TS, loadWRF_ShpTS, loadWRF_StnTS, loadWRF_Ensemble, loadWRF_ShpEns, loadWRF_StnEns

# also load CESM experiments and functions


# add relevant experiments to general load functions
from datasets.common import loadDataset, loadClim, loadShpTS, loadStnTS, loadEnsembleTS, addLoadFcts
# modify functions (wont affect modified WRF/CESM functions)
addLoadFcts(locals(), locals(), WRF_exps=WRF_exps, WRF_ens=WRF_ens)
