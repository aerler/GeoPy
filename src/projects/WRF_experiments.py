'''
Created on 2013-11-08, revised 2016-04-15

This module contains meta data for all available WRF experiments. 

@author: Andre R. Erler, GPL v3
'''

from importlib import import_module
from collections import OrderedDict
from datasets.common import addLoadFcts
from datasets.WRF import Exp

# list of projects to merge
project_list = OrderedDict(GreatLakes='GL',WesternCanada='WC')
# N.B.: the key is the project name and the value is the list abbreviation

## import experiment and ensemble lists
mod_list = [import_module('projects.{0:s}.WRF_experiments'.format(proj)) for proj in project_list.iterkeys()]

## merge experiment dictionaries
experiments = OrderedDict()
ensembles = OrderedDict()
# loop over projects
for abbr,mod in zip(project_list.values(),mod_list):
  tag = abbr.lower() # only lower case names for WRF
  # loop over ensembles in project
  for name,ens in mod.ensembles.iteritems():
    ensembles['{:s}_{:s}'.format(name,tag)] = ens
  # loop over experiments in project
  for name,exp in mod.experiments.iteritems():
    experiments['{:s}_{:s}'.format(name,tag)] = exp

# add aliases here
WRF_exps = experiments # alias for experiments
WRF_ens = ensembles # alias for ensembles


## generate loadWRF* versions with these experiments
# import datasets.WRF as dataset
from datasets.WRF import loadWRF, loadWRF_Shp, loadWRF_Stn, loadWRF_TS, loadWRF_ShpTS, loadWRF_StnTS, loadWRF_Ensemble, loadWRF_ShpEns, loadWRF_StnEns
addLoadFcts(locals(), locals(), exps=WRF_exps, enses=WRF_ens)


if __name__ == '__main__':
    
  ## view/test ensembles
  print('\n')
  proj = ''; s = ''
  for name,members in WRF_ens.iteritems():
    if proj != members[0].project:
      print('\n{:s}:'.format(members[0].project))
      proj = members[0].project
    s = '  {:s}: '.format(name)
    for member in members: s += ' {:s},'.format(member.name)
    print(s)

  ## view/test experiments
  proj = ''; s = ''
  for name,exp in WRF_exps.iteritems():
    if proj != exp.project:
      print(s)
      s = '\n{:s}:\n  '.format(exp.project)
      k = len(s)+30
      proj = exp.project
    s += '{:s}, '.format(name)
    # add line breaks
    if len(s)-k > 0: 
      s += '\n  '
      k += 30
  print(s) # print last one
  