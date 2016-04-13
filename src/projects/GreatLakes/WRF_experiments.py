'''
Created on 2013-11-08

This module contains meta data for WRF simulations for the Great Lakes region. 

@author: Andre R. Erler, GPL v3
'''

from collections import OrderedDict
from datasets.common import addLoadFcts
from datasets.WRF import Exp as WRF_Exp

## EXP class with specific default values
class Exp(WRF_Exp): 
  parameters = WRF_Exp.parameters.copy()
  defaults = WRF_Exp.defaults.copy()
  # set some project specific defaults
  defaults['parent'] = 'Ctrl-1' # CESM simulations that is driving most of the WRF runs   
  defaults['project'] = 'GreatLakes' # most WRF runs so far are from this project
  defaults['domains'] = 2 # most WRF runs have two domains
  defaults['begindate'] = '1979-01-01' # most WRF runs start in 1979
  defaults['grid'] = 'glb1' # Marc's Great Lakes domain
  
## list of experiments
# N.B.: This is the reference list, with unambiguous, unique keys and no aliases/duplicate entries  
experiments = OrderedDict() # dictionary of experiments
## Great Lakes experiments
# T-Ensemble
experiments['erai-t'] = Exp(shortname='erai-t', name='erai-t', title='T (ERA-I)')
experiments['t-ctrl'] = Exp(shortname='t-ctrl', name='t-ctrl', title='T-Ctrl', ensemble='t-ensemble-2100')
experiments['t-ctrl-2050'] = Exp(shortname='t-ctrl-2050', name='t-ctrl-2050', title='T-Ctrl (2050)', begindate='2045-01-01', ensemble='t-ensemble-2050')
experiments['t-ctrl-2100'] = Exp(shortname='t-ctrl-2100', name='t-ctrl-2100', title='T-Ctrl (2100)', begindate='2085-01-01', ensemble='t-ensemble-2100')
experiments['t-ens-A'] = Exp(shortname='t-ens-A', name='t-ens-A', title='T-Ens-A', parent='Ens-A', ensemble='t-ensemble')
experiments['t-ens-A-2050'] = Exp(shortname='t-ens-A-2050', name='t-ens-A-2050', title='T-Ens-A (2050)', parent='Ens-A-2050', ensemble='t-ensemble-2050', begindate='2045-01-01')
experiments['t-ens-A-2100'] = Exp(shortname='t-ens-A-2100', name='t-ens-A-2100', title='T-Ens-A (2100)', parent='Ens-A-2100', ensemble='t-ensemble-2100', begindate='2085-01-01')
experiments['t-ens-B'] = Exp(shortname='t-ens-B', name='t-ens-B', title='T-Ens-B', parent='Ens-B', ensemble='t-ensemble')
experiments['t-ens-B-2050'] = Exp(shortname='t-ens-B-2050', name='t-ens-B-2050', title='T-Ens-B (2050)', begindate='2045-01-01', parent='Ens-B-2050', ensemble='t-ensemble-2050')
experiments['t-ens-B-2100'] = Exp(shortname='t-ens-B-2100', name='t-ens-B-2100', title='T-Ens-B (2100)', begindate='2085-01-01', parent='Ens-B-2100', ensemble='t-ensemble-2100')
experiments['t-ens-C'] = Exp(shortname='t-ens-C', name='t-ens-C', title='T-Ens-C', parent='Ens-C', ensemble='t-ensemble')
experiments['t-ens-C-2050'] = Exp(shortname='t-ens-C-2050', name='t-ens-C-2050', title='T-Ens-C (2050)', begindate='2045-01-01', parent='Ens-C-2050', ensemble='t-ensemble-2050')
experiments['t-ens-C-2100'] = Exp(shortname='t-ens-C-2100', name='t-ens-C-2100', title='T-Ens-C (2100)', begindate='2085-01-01', parent='Ens-C-2100', ensemble='t-ensemble-2100')
experiments['t-ensemble'] = Exp(shortname='t-ens', name='t-ensemble', title="T-Ensemble", parent='Ens')
experiments['t-ensemble-2050'] = Exp(shortname='t-ens-2050', name='t-ensemble-2050', title="T-Ens. (2050)", begindate='2045-01-01', parent='Ens-2050')
experiments['t-ensemble-2100'] = Exp(shortname='t-ens-2100', name='t-ensemble-2100', title="T-Ens. (2100)", begindate='2085-01-01', parent='Ens-2050')
# G-Ensemble
experiments['erai-g'] = Exp(shortname='erai-g', name='erai-g', title='G (ERA-I)')
experiments['g-ctrl'] = Exp(shortname='g-ctrl', name='g-ctrl', title='G-Ctrl', ensemble='g-ensemble')
experiments['g-ctrl-2050'] = Exp(shortname='g-ctrl-2050', name='g-ctrl-2050', title='G-Ctrl (2050)', begindate='2045-01-01', ensemble='g-ensemble-2050')
experiments['g-ctrl-2100'] = Exp(shortname='g-ctrl-2100', name='g-ctrl-2100', title='G-Ctrl (2100)', begindate='2085-01-01', ensemble='g-ensemble-2050')
experiments['g-ens-A'] = Exp(shortname='g-ens-A', name='g-ens-A', title='G-Ens-A', parent='Ens-A', ensemble='g-ensemble')
experiments['g-ens-A-2050'] = Exp(shortname='g-ens-A-2050', name='g-ens-A-2050', title='G-Ens-A (2050)', parent='Ens-A-2050', ensemble='g-ensemble-2050', begindate='2045-01-01')
experiments['g-ens-A-2100'] = Exp(shortname='g-ens-A-2100', name='g-ens-A-2100', title='G-Ens-A (2100)', parent='Ens-A-2100', ensemble='g-ensemble-2100', begindate='2085-01-01')
experiments['g-ens-B'] = Exp(shortname='g-ens-B', name='g-ens-B', title='G-Ens-B', parent='Ens-B', ensemble='g-ensemble')
experiments['g-ens-B-2050'] = Exp(shortname='g-ens-B-2050', name='g-ens-B-2050', title='G-Ens-B (2050)', begindate='2045-01-01', parent='Ens-B-2050', ensemble='g-ensemble-2050')
experiments['g-ens-B-2100'] = Exp(shortname='g-ens-B-2100', name='g-ens-B-2100', title='G-Ens-B (2100)', begindate='2085-01-01', parent='Ens-B-2100', ensemble='g-ensemble-2100')
experiments['g-ens-C'] = Exp(shortname='g-ens-C', name='g-ens-C', title='G-Ens-C', parent='Ens-C', ensemble='g-ensemble')
experiments['g-ens-C-2050'] = Exp(shortname='g-ens-C-2050', name='g-ens-C-2050', title='G-Ens-C (2050)', begindate='2045-01-01', parent='Ens-C-2050', ensemble='g-ensemble-2050')
experiments['g-ens-C-2100'] = Exp(shortname='g-ens-C-2100', name='g-ens-C-2100', title='G-Ens-C (2100)', begindate='2085-01-01', parent='Ens-C-2100', ensemble='g-ensemble-2100')
experiments['g-ensemble'] = Exp(shortname='g-ens', name='g-ensemble', title="G-Ensemble", parent='Ens')
experiments['g-ensemble-2050'] = Exp(shortname='g-ens-2050', name='g-ensemble-2050', title="G-Ens. (2050)", begindate='2045-01-01', parent='Ens-2050')
experiments['g-ensemble-2100'] = Exp(shortname='g-ens-2100', name='g-ensemble-2100', title="G-Ens. (2100)", begindate='2085-01-01', parent='Ens-2050')
# sensitivity experiments
experiments['gg-ctrl'] = Exp(shortname='gg-ctrl', name='gg-ctrl', title='G-Ctrl (N-MP)')
experiments['gg-ctrl-2050'] = Exp(shortname='gg-ctrl-2050', name='gg-ctrl-2050', title='G-Ctrl (2050, N-MP)', begindate='2045-01-01')
experiments['gg-ctrl-2100'] = Exp(shortname='gg-ctrl-2100', name='gg-ctrl-2100', title='G-Ctrl (2100, N-MP)', begindate='2085-01-01')
experiments['m-ctrl'] = Exp(shortname='m-ctrl', name='m-ctrl', title='M-Ctrl')
experiments['m-ctrl-2050'] = Exp(shortname='m-ctrl-2050', name='m-ctrl-2050', title='M-Ctrl (2050)', begindate='2045-01-01')
experiments['m-ctrl-2100'] = Exp(shortname='m-ctrl-2100', name='m-ctrl-2100', title='M-Ctrl (2100)', begindate='2085-01-01')
experiments['mm-ctrl'] = Exp(shortname='mm-ctrl', name='mm-ctrl', title='M-Ctrl (N-MP)')
experiments['mm-ctrl-2050'] = Exp(shortname='mm-ctrl-2050', name='mm-ctrl-2050', title='M-Ctrl (2050, N-MP)', begindate='2045-01-01')
experiments['mm-ctrl-2100'] = Exp(shortname='mm-ctrl-2100', name='mm-ctrl-2100', title='M-Ctrl (2100, N-MP)', begindate='2085-01-01')
# Marc's Physics Ensemble
experiments['physics-ensemble'] = Exp(shortname='physics-ens', name='physics-ensemble', title="Physics Ensemble", parent='Ens')
experiments['physics-ensemble-2050'] = Exp(shortname='physics-ens-2050', name='physics-ensemble-2050', title="Phys. Ens. (2050)", begindate='2045-01-01', parent='Ens-2050')
experiments['physics-ensemble-2100'] = Exp(shortname='physics-ens-2100', name='physics-ensemble-2100', title="Phys. Ens. (2100)", begindate='2085-01-01', parent='Ens-2050')

## an alternate dictionary using short names and aliases for referencing
exps = OrderedDict()
# use short names where available, normal names otherwise
for key,item in experiments.iteritems():
  exps[key] = item # this prevents name collisions between regions
  if item.shortname is not None: 
    exps[item.shortname] = item
  # both, short and long name are added to list
# add aliases here
WRF_exps = exps # alias for whole dict (short and proper names)
WRF_experiments = experiments # alias for dict with proper names

## dict of ensembles
ensembles = OrderedDict()
# Great Lakes ensembles
ensembles['marc-ens'] = ('g-ctrl','gg-ctrl', 'm-ctrl','mm-ctrl', 't-ctrl')
ensembles['g-ens'] = ('g-ctrl', 'g-ens-A', 'g-ens-B', 'g-ens-C')
ensembles['t-ens'] = ('t-ctrl', 't-ens-A', 't-ens-B', 't-ens-C')
# N.B.: static & meta data for the ensemble is copied from the first-listed member;
#       this includes station attributes, such as the elevation error 
# add future versions
for ensname,enslist in ensembles.items():
  for suffix in '2050','2100':
    suffix = '-'+suffix
    ensembles[ensname+suffix] = tuple(expname[:-2]+suffix if expname[-2:] == '-1' else expname+suffix for expname in enslist)
# replace names with experiment instances
for ensname,enslist in ensembles.iteritems():
  ensembles[ensname] = tuple(experiments[expname] for expname in enslist)
# make sorted copy
WRF_ens = OrderedDict()
name_list = ensembles.keys(); name_list.sort()
for key in name_list:
  WRF_ens[key] = ensembles[key]
ensembles = WRF_ens


## generate loadWRF* versions with these experiments
from datasets.WRF import loadWRF, loadWRF_Shp, loadWRF_Stn, loadWRF_TS, loadWRF_ShpTS, loadWRF_StnTS, loadWRF_Ensemble, loadWRF_ShpEns, loadWRF_StnEns
addLoadFcts(locals(), locals(), exps=WRF_exps, enses=WRF_ens)


if __name__ == '__main__':
    
  ## view/test ensembles
  for name,members in WRF_ens.iteritems():
    s = '  {:s}: '.format(name)
    for member in members: s += ' {:s},'.format(member.name)
    print(s)
