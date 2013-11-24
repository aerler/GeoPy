'''
Created on 2013-09-08

A package that provides access to a variety of datasets for use with the geodata package.
The modules contain package itself exposes functions to load certain datasets, while the 
submodules also contain meta data and projection parameters. 

@author: Andre R. Erler, GPL v3
'''

dataset_list = ['NARR','CFSR','GPCC','CRU','PRISM']

from datasets.NARR import loadNARR_LTM, loadNARR_TS, loadNARR
from datasets.CFSR import loadCFSR_TS, loadCFSR
from datasets.GPCC import loadGPCC_LTM, loadGPCC_TS, loadGPCC
from datasets.CRU import loadCRU_TS, loadCRU
from datasets.PRISM import loadPRISM
from datasets.Unity import loadUnity

from datasets.common import loadDatasets
