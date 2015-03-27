'''
Created on 2013-11-08

This module contains meta data for all available WRF experiments. 

@author: Andre R. Erler, GPL v3
'''

from collections import OrderedDict

# the averaging folder needs to be repeated here, or it causes problems with circular imports
from datasets.common import data_root
avgfolder = data_root + 'WRF/' + 'wrfavg/' # long-term mean folder

## class that defines experiments
class Exp(object):
  ''' class of objects that contain meta data for WRF experiments '''
  # experiment parameter definition (class property)
  parameters = OrderedDict() # order matters, because parameters can depend on one another for defaults
  parameters['name'] = dict(type=basestring,req=True) # name
  parameters['shortname'] = dict(type=basestring,req=False) # short name
  parameters['title'] = dict(type=basestring,req=False) # title used in plots
  parameters['grid'] = dict(type=basestring,req=True) # name of the grid layout
  parameters['domains'] = dict(type=int,req=True) # number of domains
  parameters['parent'] = dict(type=basestring,req=True) # driving dataset
  parameters['ensemble'] = dict(type=basestring,req=False) # ensemble this run is a member of
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
  defaults['parent'] = 'Ctrl-1' # CESM simulations that is driving most of the WRF runs   
  defaults['domains'] = 2 # most WRF runs have two domains
  defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/'.format(avgfolder,atts['name'])
  defaults['begindate'] = '1979-01-01'
  defaults['beginyear'] = lambda atts: int(atts['begindate'].split('-')[0])  if atts['begindate'] else None # first field
  defaults['endyear'] = lambda atts: int(atts['enddate'].split('-')[0]) if atts['enddate'] else None # first field
  
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
experiments = OrderedDict() # dictionary of experiments
# very high resolution experiments
experiments['erai-wc2-2013'] = Exp(shortname='wc2-2013', name='erai-wc2-2013', title='ERA-I 1km (2013)', begindate='2013-08-01', grid='wc2', parent='ERA-I')
experiments['erai-wc2-2010'] = Exp(shortname='wc2-2010', name='erai-wc2-2010', title='ERA-I 1km (2010)', begindate='2010-08-01', grid='wc2', parent='ERA-I')
experiments['coast-brian'] = Exp(shortname='coast', name='coast-brian', begindate='1979-09-01', enddate='1979-09-01', grid='coast1', parent='CFSR')
experiments['col1-ctrl'] = Exp(shortname='col1', name='col1-ctrl', title='Max 3km (CFSR)', begindate='1979-09-01', grid='col1', domains=3, parent='CFSR')
experiments['max-3km'] = Exp(shortname='max-3km', name='max-3km', title='Max 3km (CESM)', begindate='1979-09-01', grid='col2', domains=3, parent='Ctrl')
experiments['erai-3km'] = Exp(shortname='erai-3km', name='erai-3km', title='Max 3km (ERA-I)', begindate='1979-09-01', grid='col2', domains=3, parent='ERA-I')
# Marc's experiments
experiments['marc-g'] = Exp(shortname='marc-g', name='marc-g', title='Marc-G', begindate='1979-01-01', grid='grb1')
experiments['marc-g-2050'] = Exp(shortname='marc-g-2050', name='marc-g-2050', title='Marc-G (2050)', begindate='2045-01-01', grid='grb1')
experiments['marc-gg'] = Exp(shortname='marc-gg', name='marc-gg', title='Marc-g (N-MP)', begindate='1979-01-01', grid='grb1')
experiments['marc-gg-2050'] = Exp(shortname='marc-gg-2050', name='marc-gg-2050', title='Marc-g (2050, N-MP)', begindate='2045-01-01', grid='grb1')
experiments['marc-m'] = Exp(shortname='marc-m', name='marc-m', title='Marc-M', begindate='1979-01-01', grid='grb1')
experiments['marc-m-2050'] = Exp(shortname='marc-m-2050', name='marc-m-2050', title='Marc-M (2050)', begindate='2045-01-01', grid='grb1')
experiments['marc-mm'] = Exp(shortname='marc-mm', name='marc-mm', title='Marc-m (N-MP)', begindate='1979-01-01', grid='grb1')
experiments['marc-mm-2050'] = Exp(shortname='marc-mm-2050', name='marc-mm-2050', title='Marc-m (2050, N-MP)', begindate='2045-01-01', grid='grb1')
experiments['marc-t'] = Exp(shortname='marc-t', name='marc-t', title='Marc-T', begindate='1979-01-01', grid='grb1')
experiments['marc-t-2050'] = Exp(shortname='marc-t-2050', name='marc-t-2050', title='Marc-T (2050)', begindate='2045-01-01', grid='grb1')
experiments['marc-ensemble'] = Exp(shortname='marc-ens', name='marc-ensemble', title="Marc's Ensemble", begindate='1979-01-01', grid='grb1', parent='Ens')
experiments['marc-ensemble-2050'] = Exp(shortname='marc-ens-2050', name='marc-ensemble-2050', title="Marc's Ensemble (2050)", begindate='2045-01-01', grid='grb1', parent='Ens-2050')
# some new experiments using WRF V3.6 or V3.6.1 and the new configuration 
experiments['erai-v361-ctrl'] = Exp(shortname='erai-v361', name='erai-v361-ctrl', title='ERA-I (New, V3.6.1)', begindate='1979-01-01', grid='arb3', parent='ERA-I')
experiments['erai-v361-noah'] = Exp(shortname='erai-v361-noah', name='erai-v361-noah', title='ERA-I (Noah, V3.6.1)', begindate='1979-01-01', grid='arb3', parent='ERA-I')
experiments['new-v361-ctrl'] = Exp(shortname='new-v361', name='new-v361-ctrl', title='New (V3.6.1)', begindate='1979-01-01', grid='arb3')
experiments['new-v361-ctrl-2050'] = Exp(shortname='new-v361-2050', name='new-v361-ctrl-2050', title='New (2050, V3.6.1)', begindate='2045-01-01', grid='arb3')
experiments['new-v361-ctrl-2100'] = Exp(shortname='new-v361-2100', name='new-v361-ctrl-2100', title='New (2100, V3.6.1)', begindate='2085-01-01', grid='arb3')
experiments['new-v36-nmp'] = Exp(shortname='new-nmp', name='new-v36-nmp', title='New (N-MP, V3.6)', begindate='1979-01-01', grid='arb3')
experiments['new-v36-noah'] = Exp(shortname='new-noah', name='new-v36-noah', title='New (Noah, V3.6)', begindate='1979-01-01', grid='arb3')
experiments['new-v36-clm'] = Exp(shortname='new-clm', name='new-v36-clm', title='New (CLM, V3.6)', begindate='1979-01-01', grid='arb3')
experiments['erai-v36-nmp'] = Exp(shortname='erai-nmp', name='erai-v36-nmp', title='ERA-I (N-MP, V3.6)', begindate='1979-01-01', grid='arb3', parent='ERA-I')
experiments['erai-v36-noah'] = Exp(shortname='erai-noah', name='erai-v36-noah', title='ERA-I (Noah, V3.6)', begindate='1979-01-01', grid='arb3', parent='ERA-I')
# these are all based on the "new" configuration (ARB3 domain)
experiments['new-ctrl'] = Exp(shortname='new', name='new-ctrl', title='New (V3.4)', begindate='1979-01-01', grid='arb3')
experiments['new-ctrl-2050'] = Exp(shortname='new-2050', name='new-ctrl-2050', title='New (2050)', begindate='2045-01-01', grid='arb3')
experiments['new-ctrl-2100'] = Exp(shortname='new-2100', name='new-ctrl-2100', title='New (2100)', begindate='2085-01-01', grid='arb3')
experiments['new-grell'] = Exp(shortname='grell', name='new-grell', title='New (Grell)', begindate='1979-01-01', grid='arb3')
experiments['new-grell-old'] = Exp(shortname='new-grell-old', name='new-grell-old', title='New (Grell, old NMP)', begindate='1979-01-01', grid='arb3')
experiments['new-nogulf'] = Exp(shortname='nogulf', name='new-nogulf', title='New (no Gulf)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['new-noah'] = Exp(shortname='noah', name='new-noah', title='New (Noah)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['v35-noah'] = Exp(shortname='noah35', name='v35-noah', title='V35 & Noah (New)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['cfsr-new'] = Exp(shortname='cfsr-new', name='cfsr-new', title='New (CFSR)', begindate='1979-01-01', grid='arb3', parent='CFSR')
# these are all based on the "max" configuration (ARB2 domain)
experiments['max-ctrl-dry'] = Exp(shortname='max-dry', name='max-ctrl-dry', title='Max-1 (dry)', begindate='1979-01-01', grid='arb2')
experiments['max-grass'] = Exp(shortname='grass', name='max-grass', title='Deforest (Max)', begindate='1979-01-01', domains=1, grid='arb2')
experiments['max-lowres'] = Exp(shortname='lowres', name='max-lowres', title='Max (low-res)', begindate='1979-01-01', domains=1, grid='arb2-120km')
experiments['max-diff'] = Exp(shortname='diff', name='max-diff', title='Max-1 (diff)', begindate='1979-01-01', grid='arb2')
experiments['max-clm'] = Exp(shortname='max-clm', name='max-clm', title='Max-1 (CLM)', begindate='1979-01-01', grid='arb2')
experiments['max-kf'] = Exp(shortname='max-kf', name='max-kf', title='Max-1 (KF)', begindate='1979-01-01', grid='arb2')
experiments['max-nosub'] = Exp(shortname='nosub', name='max-nosub', title='Max-1 (nosub)', begindate='1979-01-01', grid='arb2')
experiments['max-nosc'] = Exp(shortname='nosc', name='max-nosc', title='Max-1 (nosc)', begindate='1979-01-01', grid='arb2')
experiments['max-noflake'] = Exp(shortname='noflake', name='max-noflake', title='Max-1 (no Flake)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp'] = Exp(shortname='max-nmp', name='max-nmp', title='Max-1 (Noah-MP)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp-2050'] = Exp(shortname='max-nmp-2050', name='max-nmp-2050', title='Max-1 (Noah-MP, 2050)', begindate='2045-01-01', grid='arb2')
experiments['max-nmp-2100'] = Exp(shortname='max-nmp-2100', name='max-nmp-2100', title='Max-1 (Noah-MP, 2100)', begindate='2085-01-01', grid='arb2')
experiments['max-nmp-old'] = Exp(shortname='max-nmp-old', name='max-nmp-old', title='Max-1 (old NMP)', begindate='1979-01-01', grid='arb2')
experiments['max-nofdda'] = Exp(shortname='max-nofdda', name='max-nofdda', title='Max-1 (No Nudging)', begindate='1979-01-01', grid='arb2')
experiments['max-fdda'] = Exp(shortname='max-fdda', name='max-fdda', title='Max-1 (Nudging++)', begindate='1979-01-01', grid='arb2')
experiments['max-hilev'] = Exp(shortname='hilev', name='max-hilev', title='Max-1 (hilev)', begindate='1979-01-01', grid='arb2')
experiments['max-1deg'] = Exp(shortname='1deg', name='max-1deg', title='Max-1 (1 deg.)', begindate='1979-01-01', grid='arb2')
experiments['max-1deg-2050'] = Exp(shortname='1deg-2050', name='max-1deg-2050', title='Max-1 (1 deg., 2050)', begindate='2045-01-01', grid='arb2')
experiments['max-1deg-2100'] = Exp(shortname='1deg-2100', name='max-1deg-2100', title='Max-1 (1 deg., 2100)', begindate='2085-01-01', grid='arb2')
experiments['max-cg'] = Exp(shortname='cg', name='max-cg', title='Max-1 (CG)', begindate='1979-01-01', grid='arb2')
experiments['max-gulf'] = Exp(shortname='gulf', name='max-gulf', title='Max-1 (Gulf)', begindate='1979-01-01', grid='arb3') # ARB3 domain
experiments['max-ensemble'] = Exp(shortname='max-ens', name='max-ensemble', title='Max Ensemble Mean', begindate='1979-01-01', grid='arb2', parent='Ens')
experiments['mex-ensemble'] = Exp(shortname='mex-ens', name='mex-ensemble', title='Max Ens Mean (ext.)', begindate='1979-01-01', grid='arb2', parent='Ens')
experiments['phys-ensemble'] = Exp(shortname='phys-ens', name='phys-ensemble', title='Physics Ens Mean', begindate='1979-01-01', grid='arb2', parent='Ctrl')
experiments['grand-ensemble'] = Exp(shortname='all-ens', name='grand-ensemble', title='Grand Ensemble Mean', begindate='1979-01-01', grid='arb2', parent='Ens')
experiments['max-ctrl'] = Exp(shortname='max', name='max-ctrl', title='Max-1', begindate='1979-01-01', grid='arb2', ensemble='max-ensemble')
experiments['max-ens-A'] = Exp(shortname='max-A', name='max-ens-A', title='Max-A', begindate='1979-01-01', grid='arb2', parent='Ens-A', ensemble='max-ensemble')
experiments['max-ens-B'] = Exp(shortname='max-B', name='max-ens-B', title='Max-B', begindate='1979-01-01', grid='arb2', parent='Ens-B', ensemble='max-ensemble')
experiments['max-ens-C'] = Exp(shortname='max-C', name='max-ens-C', title='Max-C', begindate='1979-01-01', grid='arb2', parent='Ens-C', ensemble='max-ensemble')
experiments['max-ensemble-2050'] = Exp(shortname='max-ens-2050', name='max-ensemble-2050', title='Max Ensemble Mean (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-2050')
experiments['mex-ensemble-2050'] = Exp(shortname='mex-ens-2050', name='mex-ensemble-2050', title='Max Ens Mean (ext., 2050)', begindate='2045-01-01', grid='arb2', parent='Ens-2050')
experiments['phys-ensemble-2050'] = Exp(shortname='phys-ens-2050', name='phys-ensemble-2050', title='Physics Ens Mean (2050)', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050')
experiments['grand-ensemble-2050'] = Exp(shortname='all-ens-2050', name='grand-ensemble-2050', title='Grand Ensemble Mean (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-2050')
experiments['max-ctrl-2050'] = Exp(shortname='max-2050', name='max-ctrl-2050', title='Max-1 (2050)', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050', ensemble='max-ensemble-2050')
experiments['max-ens-A-2050'] = Exp(shortname='max-A-2050', name='max-ens-A-2050', title='Max-A (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-A-2050', ensemble='max-ensemble-2050')
experiments['max-ens-B-2050'] = Exp(shortname='max-B-2050', name='max-ens-B-2050', title='Max-B (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-B-2050', ensemble='max-ensemble-2050')
experiments['max-ens-C-2050'] = Exp(shortname='max-C-2050', name='max-ens-C-2050', title='Max-C (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-C-2050', ensemble='max-ensemble-2050')
experiments['max-seaice-2050'] = Exp(shortname='seaice-2050', name='max-seaice-2050', title='Seaice (2050)', begindate='2045-01-01', grid='arb2', parent='Seaice-2050')
experiments['max-ensemble-2100'] = Exp(shortname='max-ens-2100', name='max-ensemble-2100', title='Max Ensemble Mean (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-2100')
experiments['mex-ensemble-2100'] = Exp(shortname='mex-ens-2100', name='mex-ensemble-2100', title='Max Ens Mean (ext., 2100)', begindate='2085-01-01', grid='arb2', parent='Ens-2100')
experiments['phys-ensemble-2100'] = Exp(shortname='phys-ens-2100', name='phys-ensemble-2100', title='Physics Ens Mean (2100)', begindate='2085-01-01', grid='arb2', parent='Ctrl-2100')
experiments['grand-ensemble-2100'] = Exp(shortname='all-ens-2100', name='grand-ensemble-2100', title='Grand Ensemble Mean (2050)', begindate='2085-01-01', grid='arb2', parent='Ens-2100')
experiments['max-ctrl-2100'] = Exp(shortname='max-2100', name='max-ctrl-2100', title='Max-1 (2100)', begindate='2085-01-01', grid='arb2', parent='Ctrl-2100', ensemble='max-ensemble-2100')
experiments['max-ens-A-2100'] = Exp(shortname='max-A-2100', name='max-ens-A-2100', title='Max-A (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-A-2100', ensemble='max-ensemble-2100')
experiments['max-ens-B-2100'] = Exp(shortname='max-B-2100', name='max-ens-B-2100', title='Max-B (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-B-2100', ensemble='max-ensemble-2100')
experiments['max-ens-C-2100'] = Exp(shortname='max-C-2100', name='max-ens-C-2100', title='Max-C (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-C-2100', ensemble='max-ensemble-2100')
experiments['max-seaice-2100'] = Exp(shortname='seaice-2100', name='max-seaice-2100', title='Seaice (2100)', begindate='2085-01-01', grid='arb2', parent='Seaice-2100')
experiments['cfsr-lowres'] = Exp(shortname='lowres', name='cfsr-lowres', title='Max (CFSR, low res.)', begindate='1979-01-01', grid='arb2')
experiments['cfsr-max'] = Exp(shortname='cfsr', name='cfsr-max', title='Max (CFSR)', begindate='1979-01-01', grid='arb2', parent='CFSR')
experiments['erai-max'] = Exp(shortname='erai', name='erai-max', title='Max (ERA-I)', begindate='1979-01-01', grid='arb2', parent='ERA-I')
# these are all based on the old configuration (original + RRTMG, ARB2)
experiments['ctrl-1'] = Exp(shortname='ctrl', name='ctrl-1', title='Ctrl-1', begindate='1979-01-01', grid='arb2')
experiments['ctrl-2050'] = Exp(shortname='ctrl-2050', name='ctrl-2050', title='Ctrl-1 (2050)', begindate='2045-01-01', grid='arb2')
experiments['ctrl-2100'] = Exp(shortname='ctrl-2100', name='ctrl-2100', title='Ctrl-1 (2100)', begindate='2085-01-01', grid='arb2')
experiments['tiedtke-ctrl'] = Exp(shortname='tiedt', name='tiedtke-ctrl', title='Tiedtke (Ctrl)', begindate='1979-01-01', grid='arb2') 
experiments['tom-ctrl'] = Exp(shortname='tom', name='tom-ctrl', title='Thompson (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['wdm6-ctrl'] = Exp(shortname='wdm6', name='wdm6-ctrl', title='WDM6 (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['milbrandt-ctrl'] = Exp(shortname='milb', name='milbrandt-ctrl', title='Milbrandt-Yau (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['epssm-ctrl'] = Exp(shortname='epssm', name='epssm-ctrl', title='Wave Damping (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpnew-ctrl'] = Exp(shortname='nmpnew', name='nmpnew-ctrl', title='New (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpbar-ctrl'] = Exp(shortname='nmpbar', name='nmpbar-ctrl', title='Barlage (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpsnw-ctrl'] = Exp(shortname='nmpsnw', name='nmpsnw-ctrl', title='Snow (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
# some new experiments based on the old CAM configuration 
experiments['old-ctrl'] = Exp(shortname='old', name='old-ctrl', title='Old', begindate='1979-01-01', grid='arb2')
experiments['old-ctrl-2050'] = Exp(shortname='old-2050', name='old-ctrl-2050', title='Old-2050', begindate='2045-01-01', grid='arb2')
experiments['old-ctrl-2100'] = Exp(shortname='old-2100', name='old-ctrl-2100', title='Old-2100', begindate='2085-01-01', grid='arb2')
# these are all based on the original configuration (mostly ARB1 domain and CAM)
experiments['ctrl-1-arb1'] = Exp(shortname='ctrl-arb1', name='ctrl-1-arb1', title='Ctrl-ARB1', begindate='1979-01-01', grid='arb1')
experiments['ctrl-arb1-2050'] = Exp(shortname='ctrl-arb1-2050', name='ctrl-arb1-2050', title='Ctrl-ARB1 (2050)', begindate='2045-01-01', grid='arb1')
experiments['ctrl-2-arb1'] = Exp(shortname='ctrl-2-arb1', name='ctrl-2-arb1', title='Ctrl-2 (CAM)', begindate='1979-01-01', grid='arb1')
experiments['2way-arb1'] = Exp(shortname='2way', name='2way-arb1', title='2-way Nest (CAM)', begindate='1979-01-01', grid='arb1')
experiments['pbl1-arb1'] = Exp(shortname='pbl4', name='pbl1-arb1', title='PBL-1 (CAM)', begindate='1979-01-01', grid='arb1')
experiments['grell3-arb1'] = Exp(shortname='grell', name='grell3-arb1', title='Grell-3 (CAM)', begindate='1979-01-01', grid='arb1')
experiments['noahmp-arb1'] = Exp(shortname='nmpdef', name='noahmp-arb1', title='Noah-MP (CAM)', begindate='1979-01-01', grid='arb1')
experiments['rrtmg-arb1'] = Exp(shortname='rrtmg', name='rrtmg-arb1', title='RRTMG (Very Old)', begindate='1979-01-01', grid='arb1')
experiments['hitop-arb1'] = Exp(shortname='hitop', name='hitop-arb1', begindate='1979-01-01', grid='arb1')
experiments['polar-arb1'] = Exp(shortname='polar', name='polar-arb1', title='PolarWRF (CAM)', begindate='1979-01-01', grid='arb1')
experiments['modis-arb1'] = Exp(shortname='modis', name='modis-arb1', title='Modis (CAM)', begindate='1979-01-01', grid='arb1')
experiments['moris-ctrl'] = Exp(shortname='moris', name='moris-ctrl', title='Morrison (CAM)', begindate='1979-01-01', grid='arb2')
experiments['v35-clm'] = Exp(shortname='clm', name='v35-clm', title='CLM (V35, CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ctrl'] = Exp(shortname='cam', name='cam-ctrl', title='Ctrl (CAM)', begindate='1979-01-01', grid='arb2') 
experiments['cfsr-cam'] = Exp(shortname='cfsr-cam', name='cfsr-cam', title='CAM-1 (CFSR)', begindate='1979-01-01', grid='arb2', parent='CFSR')
experiments['cam-ens-A'] = Exp(shortname='cam-A', name='cam-ens-A', title='Ens-A (CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ens-B'] = Exp(shortname='cam-B', name='cam-ens-B', title='Ens-B (CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ens-C'] = Exp(shortname='cam-C', name='cam-ens-C', title='Ens-C (CAM)', begindate='1979-01-01', grid='arb2')
experiments['cam-ctrl-1-2050'] = Exp(shortname='cam-1-2050', name='cam-ctrl-1-2050', title='CAM-1 2050', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050')
experiments['cam-ctrl-2-2050'] = Exp(shortname='cam-2050', name='cam-ctrl-2-2050', title='CAM-2050', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050')
experiments['cam-ctrl-2-2100'] = Exp(shortname='cam-2100', name='cam-ctrl-2-2100', title='CAM-2100', begindate='2095-01-01', grid='arb2', parent='Ctrl-2100')
# other hitop/fdda
# experiments['nofdda-ctrl'] = Exp(name='nofdda-ctrl', avgfolder=avgfolder+'/hitop-test/nofdda-ctrl/', begindate='1979-01-01', grid='arb1')
# experiments['nofdda-hitop'] = Exp(name='nofdda-hitop', avgfolder=avgfolder+'/hitop-test/nofdda-hitop/', begindate='1979-01-01', grid='arb1')
# experiments['hitop-old'] = Exp(name='hitop-old', avgfolder=avgfolder+'/hitop-test/hitop-ctrl/', begindate='1979-01-01', grid='arb1')

## an alternate dictionary using short names and aliases for referencing
exps = OrderedDict()
# use short names where availalbe, normal names otherwise
for key,item in experiments.iteritems():
  exps[item.name] = item
  if item.shortname is not None: 
    exps[item.shortname] = item
  # both, short and long name are added to list
# add aliases here
WRF_exps = exps # alias for whole dict
WRF_experiments = experiments # alias for whole dict

## dict of ensembles
ensembles = OrderedDict()
# initial condition ensemble
ensembles['max-ens'] = ('max-ctrl', 'max-ens-A', 'max-ens-B', 'max-ens-C')
ensembles['phys-ens'] = ('old-ctrl','ctrl-1','max-ctrl')
# Marc's simulations
ensembles['marc-ens'] = ('marc-g','marc-gg', 'marc-m','marc-mm', 'marc-t')
# add future versions
for ensname,enslist in ensembles.items():
  for suffix in '2050','2100':
    suffix = '-'+suffix
    ensembles[ensname+suffix] = tuple(expname[:-2]+suffix if expname[-2:] == '-1' else expname+suffix for expname in enslist)
# extended ensemble
ensembles['mex-ens'] = ensembles['max-ens'] + ('erai-max',)
ensembles['mex-ens-2050'] = ensembles['max-ens-2050'] + ('max-seaice-2050',)
ensembles['mex-ens-2100'] = ensembles['max-ens-2100'] + ('max-seaice-2100',)
# construct grand ensemble
for suffix in '','2050','2100':
  suffix = '-'+suffix if suffix else ''
  ensembles['all-ens'+suffix] = ensembles['phys-ens'+suffix] + ensembles['mex-ens'+suffix][1:]
  # N.B.: omit max-ctrl the second time
  
# replace names with experiment instances
for ensname,enslist in ensembles.iteritems():
  ensembles[ensname] = tuple(experiments[expname] for expname in enslist)
# make sorted copy
WRF_ens = OrderedDict()
name_list = ensembles.keys(); name_list.sort()
for key in name_list:
  WRF_ens[key] = ensembles[key]
ensembles = WRF_ens


if __name__ == '__main__':
    
  ## view/test ensembles
  for name,members in WRF_ens.iteritems():
    s = '  {:s}: '.format(name)
    for member in members: s += ' {:s},'.format(member.name)
    print(s)