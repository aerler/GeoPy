'''
Created on 2013-11-08

This module contains meta data for all available WRF experiments. 

@author: Andre R. Erler, GPL v3
'''

# data root folder
from datasets.WRF import avgfolder #, root_folder, outfolder

class Exp(object):
  ''' class of objects that contain meta data for WRF experiments '''
  # experiment parameter definition (class property)
  parameters = dict()
  parameters['name'] = dict(type=basestring,req=True) # name
  parameters['shortname'] = dict(type=basestring,req=False) # short name
  parameters['title'] = dict(type=basestring,req=False) # title used in plots
  parameters['grid'] = dict(type=basestring,req=True) # name
  parameters['parent'] = dict(type=basestring,req=True) # driving dataset
  parameters['begindate'] = dict(type=basestring,req=True) # simulation start date
  parameters['beginyear'] = dict(type=int,req=True) # simulation start year
  parameters['enddate'] = dict(type=basestring,req=False) # simulation end date (if it already finished)
  parameters['endyear'] = dict(type=int,req=False) # simulation end year
  parameters['avgfolder'] = dict(type=basestring,req=True) # folder for monthly averages
  parameters['outfolder'] = dict(type=basestring,req=False) # folder for direct WRF averages
  # default values (functions)
  defaults = dict()
  defaults['shortname'] = lambda atts: atts['name']
  defaults['title'] = lambda atts: atts['name'] # need lambda, because parameters are not set yet
  defaults['parent'] = 'Ctrl' # CESM simulations that is driving most of the WRF runs   
  defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/'.format(avgfolder,atts['name'])
  defaults['begindate'] = '1979-01-01'
  defaults['beginyear'] = lambda atts: int(atts['begindate'].split('-')[0]) # first field
  
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
experiments['coast-brian'] = Exp(shortname='coast', name='coast-brian', begindate='1979-09-01', enddate='1979-09-01', grid='coast1', parent='CFSR')
experiments['columbia'] = Exp(shortname='columbia', name='columbia', title='Max 3km (CFSR)', begindate='1979-09-01', grid='col1', parent='CFSR')
# these are all based on the "new" configuration (ARB3 domain)
experiments['new-ctrl'] = Exp(shortname='new', name='new-ctrl', title='New-1 (Noah-MP)', begindate='1979-01-01', grid='arb3')
experiments['new-grell'] = Exp(shortname='grell', name='new-grell', title='New-1 (Grell)', begindate='1979-01-01', grid='arb3')
experiments['new-nogulf'] = Exp(shortname='nogulf', name='new-nogulf', title='New-1 (no Gulf)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['new-noah'] = Exp(shortname='noah', name='new-noah', title='New (Noah)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['v35-noah'] = Exp(shortname='noah35', name='v35-noah', title='V35 & Noah (New)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['cfsr-new'] = Exp(shortname='cfsr-new', name='cfsr-new', title='New (CFSR)', begindate='1979-01-01', grid='arb3', parent='CFSR')
# these are all based on the "max" configuration (ARB2 domain)
experiments['max-diff'] = Exp(shortname='diff', name='max-diff', title='Max-1 (diff)', begindate='1979-01-01', grid='arb2')
experiments['max-clm'] = Exp(shortname='max-clm', name='max-clm', title='Max-1 (CLM)', begindate='1979-01-01', grid='arb2')
experiments['max-kf'] = Exp(shortname='max-kf', name='max-kf', title='Max-1 (KF)', begindate='1979-01-01', grid='arb2')
experiments['max-nosub'] = Exp(shortname='nosub', name='max-nosub', title='Max-1 (nosub)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp'] = Exp(shortname='max-nmp', name='max-nmp', title='Max-1 (Noah-MP)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp-old'] = Exp(shortname='max-nmp-old', name='max-nmp-old', title='Max-1 (old NMP)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp'] = Exp(shortname='max-nmp', name='max-nmp', title='Max-1 (Noah-MP)', begindate='1979-01-01', grid='arb2')
experiments['max-nofdda'] = Exp(shortname='max-nofdda', name='max-nofdda', title='Max-1 (No Nudging)', begindate='1979-01-01', grid='arb2')
experiments['max-fdda'] = Exp(shortname='max-fdda', name='max-fdda', title='Max-1 (Nudging++)', begindate='1979-01-01', grid='arb2')
experiments['max-hilev'] = Exp(shortname='hilev', name='max-hilev', title='Max-1 (hilev)', begindate='1979-01-01', grid='arb2')
experiments['max-lowres'] = Exp(shortname='lowres', name='max-lowres', title='Max-1 (lowres)', begindate='1979-01-01', grid='arb2')
experiments['max-gulf'] = Exp(shortname='gulf', name='max-gulf', title='Max-1 (Gulf)', begindate='1979-01-01', grid='arb3') # ARB3 domain
experiments['max-ensemble'] = Exp(shortname='max-ens', name='max-ensemble', title='Max Ensemble Mean', begindate='1979-01-01', grid='arb2', parent='CESM')
experiments['max-ctrl'] = Exp(shortname='max', name='max-ctrl', title='Max-1', begindate='1979-01-01', grid='arb2')
experiments['max-ens-A'] = Exp(shortname='max-A', name='max-ens-A', title='Max-A', begindate='1979-01-01', grid='arb2', parent='Ens-A')
experiments['max-ens-B'] = Exp(shortname='max-B', name='max-ens-B', title='Max-B', begindate='1979-01-01', grid='arb2', parent='Ens-B')
experiments['max-ens-C'] = Exp(shortname='max-C', name='max-ens-C', title='Max-C', begindate='1979-01-01', grid='arb2', parent='Ens-C')
experiments['max-ctrl-2050'] = Exp(shortname='max-2050', name='max-ctrl-2050', title='Max-1 (2050)', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050')
experiments['max-ensemble-2050'] = Exp(shortname='max-ens-2050', name='max-ensemble-2050', title='Max Ensemble Mean (2050)', begindate='2045-01-01', grid='arb2')
experiments['max-ctrl-2100'] = Exp(shortname='max-2100', name='max-ctrl-2100', title='Max-1 (2100)', begindate='2085-01-01', grid='arb2', parent='Ctrl-2100')
experiments['max-ens-A-2050'] = Exp(shortname='max-A-2050', name='max-ens-A-2050', title='Max-A (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-A-2050')
experiments['max-ens-B-2050'] = Exp(shortname='max-B-2050', name='max-ens-B-2050', title='Max-B (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-B-2050')
experiments['max-ens-C-2050'] = Exp(shortname='max-C-2050', name='max-ens-C-2050', title='Max-C (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-C-2050')
experiments['max-seaice-2050'] = Exp(shortname='seaice-2050', name='max-seaice-2050', title='Seaice (2050)', begindate='2045-01-01', grid='arb2', parent='Seaice-2050')
experiments['cfsr-max'] = Exp(shortname='cfsr', name='cfsr-max', title='Max (CFSR)', begindate='1979-01-01', grid='arb2', parent='CFSR')
# these are all based on the old configuration (original + RRTMG, ARB2)
experiments['ctrl-1'] = Exp(shortname='ctrl', name='ctrl-1', title='Ctrl-1', begindate='1979-01-01', grid='arb2')
experiments['tiedtke-ctrl'] = Exp(shortname='tiedt', name='tiedtke-ctrl', title='Tiedtke (Ctrl)', begindate='1979-01-01', grid='arb2') 
experiments['tom-ctrl'] = Exp(shortname='tom', name='tom-ctrl', title='Thompson (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['wdm6-ctrl'] = Exp(shortname='wdm6', name='wdm6-ctrl', title='WDM6 (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['milbrandt-ctrl'] = Exp(shortname='milb', name='milbrandt-ctrl', title='Milbrandt-Yau (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['epssm-ctrl'] = Exp(shortname='epssm', name='epssm-ctrl', title='Wave Damping (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpnew-ctrl'] = Exp(shortname='nmpnew', name='nmpnew-ctrl', title='New (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpbar-ctrl'] = Exp(shortname='nmpbar', name='nmpbar-ctrl', title='Barlage (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpsnw-ctrl'] = Exp(shortname='nmpsnw', name='nmpsnw-ctrl', title='Snow (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
# these are all based on the original configuration (mostly ARB1 domain)
experiments['2way-arb1'] = Exp(shortname='2way', name='2way-arb1', title='2-way Nest (CAM)', begindate='1979-01-01', grid='arb1')
experiments['pbl1-arb1'] = Exp(shortname='pbl4', name='pbl1-arb1', title='PBL-1 (CAM)', begindate='1979-01-01', grid='arb1')
experiments['grell3-arb1'] = Exp(shortname='grell', name='grell3-arb1', title='Grell-3 (CAM)', begindate='1979-01-01', grid='arb1')
experiments['noahmp-arb1'] = Exp(shortname='nmpdef', name='noahmp-arb1', title='Noah-MP (CAM)', begindate='1979-01-01', grid='arb1')
experiments['rrtmg-arb1'] = Exp(shortname='rrtmg', name='rrtmg-arb1', title='RRTMG (Very Old)', begindate='1979-01-01', grid='arb1')
experiments['hitop-arb1'] = Exp(shortname='hitop', name='hitop-arb1', begindate='1979-01-01', grid='arb1')
experiments['polar-arb1'] = Exp(shortname='polar', name='polar-arb1', title='PolarWRF (CAM)', begindate='1979-01-01', grid='arb1')
experiments['modis-arb1'] = Exp(shortname='modis', name='modis-arb1', title='Modis (CAM)', begindate='1979-01-01', grid='arb1')
experiments['moris-ctrl'] = Exp(shortname='moris', name='moris-ctrl', title='Morrison (CAM)', begindate='1979-01-01', grid='arb2')
experiments['v35-clm'] = Exp(shortname='clm4', name='v35-clm', title='CLM (V35, CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ctrl'] = Exp(shortname='cam3', name='cam-ctrl', title='Ctrl (CAM)', begindate='1979-01-01', grid='arb2') 
experiments['cfsr-cam'] = Exp(shortname='cfsr-cam', name='cfsr-cam', title='CAM-1 (CFSR)', begindate='1979-01-01', grid='arb2', parent='CFSR')
experiments['cam-ens-A'] = Exp(shortname='cam-A', name='cam-ens-A', title='Ens-A (CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ens-B'] = Exp(shortname='cam-B', name='cam-ens-B', title='Ens-B (CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ens-C'] = Exp(shortname='cam-C', name='cam-ens-C', title='Ens-C (CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ctrl-1-2050'] = Exp(shortname='cam-1-2050', name='cam-ctrl-1-2050', title='CAM-1 2050', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050')
experiments['cam-ctrl-2-2050'] = Exp(shortname='cam-2-2050', name='cam-ctrl-2-2050', title='CAM-2 2050', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050')
experiments['cam-ctrl-2-2100'] = Exp(shortname='cam-1-2100', name='cam-ctrl-2-2100', title='CAM-2 2100', begindate='2095-01-01', grid='arb2', parent='Ctrl-2100')
# other hitop/fdda
# experiments['nofdda-ctrl'] = Exp(name='nofdda-ctrl', avgfolder=avgfolder+'/hitop-test/nofdda-ctrl/', begindate='1979-01-01', grid='arb1')
# experiments['nofdda-hitop'] = Exp(name='nofdda-hitop', avgfolder=avgfolder+'/hitop-test/nofdda-hitop/', begindate='1979-01-01', grid='arb1')
# experiments['hitop-old'] = Exp(name='hitop-old', avgfolder=avgfolder+'/hitop-test/hitop-ctrl/', begindate='1979-01-01', grid='arb1')

## an alternate dictionary using short names and aliases for referencing
exps = dict()
# use short names where availalbe, normal names otherwise
for key,item in experiments.iteritems():
  exps[item.name] = item
  if item.shortname is not None: 
    exps[item.shortname] = item
  # both, short and long name are added to list
# add aliases here
WRF_exps = exps # alias for whole dict
WRF_experiments = experiments # alias for whole dict

if __name__ == '__main__':
    pass