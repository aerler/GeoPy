'''
Created on 2013-09-08

A package that provides access to a variety of datasets for use with the geodata package.
The modules contain package itself exposes functions to load certain datasets, while the 
submodules also contain meta data and projection parameters. 

@author: Andre R. Erler, GPL v3
'''

from datasets.NARR import loadNARRLTM, loadNARRTS
from datasets.GPCC import loadGPCCLTM, loadGPCCTS
