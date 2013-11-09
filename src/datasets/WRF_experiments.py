'''
Created on 2013-11-08

This module contains meta data for all available WRF experiments. 

@author: Andre R. Erler, GPL v3
'''

# data root folder
from datasets.WRF import root_folder, avgfolder, outfolder

class Exp(object):
  ''' class of objects that contain meta data for WRF experiments '''
  # experiment parameter definition (class property)
  parameters = dict()
  parameters['name'] = dict(type=basestring,req=True) # name
  parameters['shortname'] = dict(type=basestring,req=False) # short name
  parameters['title'] = dict(type=basestring,req=False) # title used in plots
  parameters['begindate'] = dict(type=basestring,req=True) # simulation start date
  parameters['enddate'] = dict(type=basestring,req=False) # simulation end date (if it already finished)
  parameters['avgfolder'] = dict(type=basestring,req=True) # folder for monthly averages
  parameters['outfolder'] = dict(type=basestring,req=False) # folder for direct WRF averages
  # default values (functions)
  defaults = dict()
  defaults['shortname'] = lambda atts: atts['name']
  defaults['title'] = lambda atts: atts['name'] # need lambda, because parameters are not set yet  
  defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/'.format(avgfolder,atts['name'])
  defaults['begindate'] = '1979-01-01'
  
  def __init__(self, **kwargs):
    ''' initialize values from arguments '''
    # loop over simulation parameters
    for argname,argatt in self.parameters.items():
      if argname in kwargs:
        # assign argument based on keyword
        arg = kwargs[argname]
        if not isinstance(arg,argatt['type']): 
          raise TypeError, "Argument '{0:s}' must be of type '{1:s}'.".format(argname,argatt['type'].__name__)
        self.__dict__[argname] = arg
      elif argname in self.defaults:
        # assign some default values, if necessary
        if callable(self.defaults[argname]): 
          self.__dict__[argname] = self.defaults[argname](self.__dict__)
        else: self.__dict__[argname] = self.defaults[argname]
      elif argatt['req']:
        # if the argument is required and there is no default, raise error
        raise ValueError, "Argument '{0:s}' for experiment '{1:s}' required.".format(argname,self.name)
      else:
        # if the argument is not required, just assign None 
        self.__dict__[argname] = None    
    

## list of experiments
# N.B.: This is the reference list, with unambiguous, unique keys and no aliases/duplicate entries  
experiments = dict() # dictionary of experiments
# list of experiments
experiments['coast-brian'] = Exp(shortname='coast', name='coast-brian', begindate='1997-09-01', enddate='1998-09-01')
#print experiments['coast-brian'].__dict__
# hitop
experiments['hitop-ctrl'] = Exp(shortname='hitop', name='hitop-ctrl', avgfolder=avgfolder+'/hitop-ctrl/')
experiments['nofdda-ctrl'] = Exp(name='nofdda-ctrl', avgfolder=avgfolder+'/hitop-test/nofdda-ctrl/')
experiments['nofdda-hitop'] = Exp(name='nofdda-hitop', avgfolder=avgfolder+'/hitop-test/nofdda-hitop/')
experiments['hitop-old'] = Exp(name='hitop-old', avgfolder=avgfolder+'/hitop-test/hitop-ctrl/')
## shorthands for common experiments
# these are all based on the "new" configuration (ARB3 domain)
experiments['new-ctrl'] = Exp(shortname='new', name='new-ctrl', title='Thompson, Tiedtke, Noah-MP')
experiments['new-nogulf'] = Exp(shortname='nogulf', name='new-nogulf', title='New-1 (no Gulf)') # ARB2 domain
experiments['new-noah'] = Exp(shortname='noah', name='new-noah', title='Thompson & Tiedtke') # ARB2 domain
experiments['v35-noah'] = Exp(shortname='noah35', name='v35-noah', title='V35 & Noah (New)') # ARB2 domain
experiments['cfsr-new'] = Exp(shortname='cfsr-new', name='cfsr-new', title='New-1 (CFSR)')
# these are all based on the "max" configuration (ARB2 domain)
experiments['max-ctrl'] = Exp(shortname='max', name='max-ctrl', title='Morrison & Grell-3')
experiments['max-gulf'] = Exp(shortname='gulf', name='max-gulf', title='Max-1 (Gulf)') # ARB3 domain
experiments['max-ens-A'] = Exp(shortname='max-A', name='max-ens-A', title='Max-A')
experiments['max-ens-B'] = Exp(shortname='max-B', name='max-ens-B', title='Max-B')
experiments['max-ens-C'] = Exp(shortname='max-C', name='max-ens-C', title='Max-C')
experiments['max-ctrl-2050'] = Exp(shortname='max-2050', name='max-ctrl-2050', title='Max-1 (2050)')
experiments['max-ens-A-2050'] = Exp(shortname='max-A-2050', name='max-ens-A-2050', title='Max-A (2050)')
experiments['max-ens-B-2050'] = Exp(shortname='max-B-2050', name='max-ens-B-2050', title='Max-B (2050)')
experiments['max-ens-C-2050'] = Exp(shortname='max-C-2050', name='max-ens-C-2050', title='Max-C (2050)')
experiments['cfsr-max'] = Exp(shortname='cfsr-max', name='cfsr-max')
# these are all based on the old configuration (original + RRTMG, ARB2)
experiments['ctrl-1'] = Exp(shortname='ctrl-1', name='ctrl-1', title='Ctrl-1 (Old)')
experiments['tiedtke-ctrl'] = Exp(shortname='tiedt', name='tiedtke-ctrl', title='Tiedtke (Old)') 
experiments['tom-ctrl'] = Exp(shortname='tom', name='tom-ctrl', title='Thompson (Old)')
experiments['wdm6-ctrl'] = Exp(shortname='wdm6', name='wdm6-ctrl', title='WDM6 (Old)')
experiments['milbrandt-ctrl'] = Exp(shortname='milb', name='milbrandt-ctrl', title='Milbrandt-Yau (Old)')
experiments['nmpnew-ctrl'] = Exp(shortname='nmpnew', name='nmpnew-ctrl', title='New (Noah-MP, Old)')
experiments['nmpbar-ctrl'] = Exp(shortname='nmpbar', name='nmpbar-ctrl', title='Barlage (Noah-MP, Old)')
experiments['nmpsnw-ctrl'] = Exp(shortname='nmpsnw', name='nmpsnw-ctrl', title='Snow (Noah-MP, Old)')
# these are all based on the original configuration (mostly ARB1 domain)
experiments['cam-ctrl'] = Exp(shortname='cam3', name='cam-ctrl', title='Ctrl-1 (CAM)') 
experiments['pbl1-arb1'] = Exp(shortname='pbl4', name='pbl1-arb1', title='PBL-1 (CAM)')
experiments['grell3-arb1'] = Exp(shortname='grell', name='grell3-arb1', title='Grell-3 (CAM)')
experiments['moris-ctrl'] = Exp(shortname='moris', name='moris-ctrl', title='Morrison (CAM)')
experiments['noahmp-arb1'] = Exp(shortname='nmpdef', name='noahmp-arb1', title='Noah-MP (CAM)')
experiments['v35-clm'] = Exp(shortname='clm4', name='v35-clm', title='CLM (V35, CAM)')
experiments['polar-arb1'] = Exp(shortname='pwrf', name='polar-arb1', title='PolarWRF (CAM)')
experiments['modis-arb1'] = Exp(shortname='modis', name='modis-arb1', title='Modis (CAM)')
experiments['cfsr-cam'] = Exp(shortname='cfsr', name='cfsr-cam', title='CAM-1 (CFSR)')
experiments['cam-ens-A'] = Exp(shortname='cam-A', name='cam-ens-A', title='Ens-A (CAM)')
experiments['cam-ens-B'] = Exp(shortname='cam-B', name='cam-ens-B', title='Ens-B (CAM)')
experiments['cam-ens-C'] = Exp(shortname='cam-C', name='cam-ens-C', title='Ens-C (CAM)')
experiments['cam-ctrl-1-2050'] = Exp(shortname='cam-1-2050', name='cam-ctrl-1-2050', title='CAM-1 2050') #'ctrl-2-2050'
experiments['cam-ctrl-2-2100'] = Exp(shortname='cam-1-2100', name='cam-ctrl-2-2100', title='CAM-2 2100')
experiments['cam-ctrl-2-2050'] = Exp(shortname='cam-2-2050', name='cam-ctrl-2-2050', title='CAM-2 2050') #'ctrl-arb1-2050'

## an alternate dictionary using short names and aliases for referencing
exps = dict()
# use short names where availalbe, normal names otherwise
for key,item in experiments.items():
  if item.shortname is not None: 
    exps[item.shortname] = item
  else: exps[item.name] = item
# add aliases here


if __name__ == '__main__':
    pass