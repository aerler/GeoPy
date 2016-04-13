# -*- coding: utf-8 -*-
"""
Created on Apr 13 2016

This module contains meta data for all available CESM experiments. 

@author: Andre R. Erler, GPL v3
"""

from collections import OrderedDict
from datasets.common import addLoadFcts
from datasets.CESM import Exp as CESM_Exp

## EXP class with specific default values
class Exp(CESM_Exp): 
  parameters = CESM_Exp.parameters.copy()
  defaults = CESM_Exp.defaults.copy()
  # set some project specific defaults
  defaults['project'] = 'Downscaling' # CESM runs are for downscaling with WRF
  defaults['begindate'] = '1979-01-01' # historical runs start in 1979
  defaults['grid'] = 'cesm1x1' # CESM 1 degree standard grid


# list of experiments
# N.B.: This is the reference list, with unambiguous, unique keys and no aliases/duplicate entries  
experiments = OrderedDict() # dictionary of experiments
# historical 
# N.B.: the extnded ensemble end data is necessary for CVDP
experiments['ens20trcn1x1']  = Exp(shortname='Ens',    name='ens20trcn1x1',  title='CESM Ensemble Mean', begindate='1979-01-01', enddate='2039-01-01')
experiments['mens20trcn1x1'] = Exp(shortname='MEns',   name='mens20trcn1x1', title='CESM Mini Ensemble', begindate='1979-01-01', enddate='2024-01-01')
experiments['tb20trcn1x1']   = Exp(shortname='Ctrl-1', name='tb20trcn1x1',   title='Exp 1 (CESM)', begindate='1979-01-01', enddate='1994-01-01', ensemble='ens20trcn1x1')
experiments['hab20trcn1x1']  = Exp(shortname='Ctrl-A', name='hab20trcn1x1',  title='Exp 2 (CESM)', begindate='1979-01-01', enddate='1994-01-01', ensemble='ens20trcn1x1')
experiments['hbb20trcn1x1']  = Exp(shortname='Ctrl-B', name='hbb20trcn1x1',  title='Exp 3 (CESM)', begindate='1979-01-01', enddate='1994-01-01', ensemble='ens20trcn1x1')
experiments['hcb20trcn1x1']  = Exp(shortname='Ctrl-C', name='hcb20trcn1x1',  title='Exp 4 (CESM)', begindate='1979-01-01', enddate='1994-01-01', ensemble='ens20trcn1x1')
# mid-21st century
experiments['ensrcp85cn1x1']  = Exp(shortname='Ens-2050',    name='ensrcp85cn1x1',  title='CESM Ensemble Mean (2050)', begindate='2045-01-01', enddate='2105-01-01')
experiments['mensrcp85cn1x1'] = Exp(shortname='MEns-2050',   name='mensrcp85cn1x1', title='CESM Mini Ensemble (2050)', begindate='2045-01-01', enddate='2090-01-01')
experiments['seaice-5r-hf']   = Exp(shortname='Seaice-2050', name='seaice-5r-hf',   title='Seaice (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01')
experiments['htbrcp85cn1x1']  = Exp(shortname='Ctrl-1-2050', name='htbrcp85cn1x1',  title='Exp 1 (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', ensemble='ensrcp85cn1x1')
experiments['habrcp85cn1x1']  = Exp(shortname='Ctrl-A-2050', name='habrcp85cn1x1',  title='Exp 2 (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', ensemble='ensrcp85cn1x1')
experiments['hbbrcp85cn1x1']  = Exp(shortname='Ctrl-B-2050', name='hbbrcp85cn1x1',  title='Exp 3 (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', ensemble='ensrcp85cn1x1')
experiments['hcbrcp85cn1x1']  = Exp(shortname='Ctrl-C-2050', name='hcbrcp85cn1x1',  title='Exp 4 (CESM, 2050)', begindate='2045-01-01', enddate='2060-01-01', ensemble='ensrcp85cn1x1')
# mid-21st century
experiments['ensrcp85cn1x1d']  = Exp(shortname='Ens-2100',    name='ensrcp85cn1x1d',  title='CESM Ensemble Mean (2100)', begindate='2085-01-01', enddate='2145-01-01')
experiments['mensrcp85cn1x1d'] = Exp(shortname='MEns-2100',   name='mensrcp85cn1x1d', title='CESM Mini Ensemble (2100)', begindate='2085-01-01', enddate='2130-01-01')
experiments['seaice-5r-hfd']   = Exp(shortname='Seaice-2100', name='seaice-5r-hfd',   title='Seaice (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01')
experiments['htbrcp85cn1x1d']  = Exp(shortname='Ctrl-1-2100', name='htbrcp85cn1x1d',  title='Exp 1 (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', ensemble='ensrcp85cn1x1d')
experiments['habrcp85cn1x1d']  = Exp(shortname='Ctrl-A-2100', name='habrcp85cn1x1d',  title='Exp 2 (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', ensemble='ensrcp85cn1x1d')
experiments['hbbrcp85cn1x1d']  = Exp(shortname='Ctrl-B-2100', name='hbbrcp85cn1x1d',  title='Exp 3 (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', ensemble='ensrcp85cn1x1d')
experiments['hcbrcp85cn1x1d']  = Exp(shortname='Ctrl-C-2100', name='hcbrcp85cn1x1d',  title='Exp 4 (CESM, 2100)', begindate='2085-01-01', enddate='2100-01-01', ensemble='ensrcp85cn1x1d')
## an alternate dictionary using short names and aliases for referencing
exps = OrderedDict()
# use short names where available, normal names otherwise
for key,item in experiments.iteritems():
  exps[item.name] = item
  if item.shortname is not None: 
    exps[item.shortname] = item
  # both, short and long name are added to list
# add aliases here
CESM_exps = exps # alias for whole dict
CESM_experiments = experiments # alias for whole dict

## dict of ensembles
ensembles = CESM_ens = OrderedDict()
# initial condition ensemble
ensembles['ens20trcn1x1'] =   [tag+'20trcn1x1' for tag in 'tb', 'hab', 'hab', 'hab']
ensembles['ensrcp85cn1x1'] =  [tag+'rcp85cn1x1' for tag in 'htb', 'hab', 'hab', 'hab']
ensembles['ensrcp85cn1x1d'] = [tag+'rcp85cn1x1d' for tag in 'htb', 'hab', 'hab', 'hab']
ensembles['mens20trcn1x1'] =   [tag+'20trcn1x1' for tag in 'hab', 'hab', 'hab']
ensembles['mensrcp85cn1x1'] =  [tag+'rcp85cn1x1' for tag in 'hab', 'hab', 'hab']
ensembles['mensrcp85cn1x1d'] = [tag+'rcp85cn1x1d' for tag in 'hab', 'hab', 'hab']

# N.B.: static & meta data for the ensemble is copied from the first-listed member;
#       this includes station attributes, such as the elevation error 
# replace names with experiment instances
for ensname,enslist in ensembles.items(): # don't use iter, because we chagne the dict!
  members = tuple([experiments[expname] for expname in enslist])
  ensembles[ensname] = members
  ensembles[experiments[ensname].shortname] = members

# ## dict of ensembles
# ensembles = CESM_ens = OrderedDict()
# ensemble_list = list(set([exp.ensemble for exp in experiments.values() if exp.ensemble]))
# # ensemble_list.sort()
# for ensemble in ensemble_list:
#   #print ensemble, experiments[ensemble].shortname
#   members = [exp for exp in experiments.values() if exp.ensemble and exp.ensemble == ensemble]
# #   members.sort()
#   ensembles[experiments[ensemble].shortname] = members


## generate loadCESM* versions with these experiments
from datasets.CESM import loadCESM, loadCESM_Shp, loadCESM_Stn, loadCESM_TS, loadCESM_ShpTS, loadCESM_StnTS, loadCESM_Ensemble, loadCESM_ShpEns, loadCESM_StnEns
addLoadFcts(locals(), locals(), exps=CESM_exps, enses=CESM_ens)


if __name__ == '__main__':
    
  ## view/test ensembles
  for name,members in CESM_ens.iteritems():
    s = '  {:s}: '.format(name)
    for member in members: s += ' {:s},'.format(member.name)
    print(s)