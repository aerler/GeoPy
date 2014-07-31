'''
Created on 2013-11-08

This module contains meta data for all available WRF experiments. 

@author: Andre R. Erler, GPL v3
'''

from datasets.WRF import Exp   
from collections import OrderedDict

## list of experiments
# N.B.: This is the reference list, with unambiguous, unique keys and no aliases/duplicate entries  
experiments = OrderedDict() # dictionary of experiments
# list of experiments
experiments['coast-brian'] = Exp(shortname='coast', name='coast-brian', begindate='1979-09-01', enddate='1979-09-01', grid='coast1', parent='CFSR')
experiments['col1-ctrl'] = Exp(shortname='col1', name='col1-ctrl', title='Max 3km (CFSR)', begindate='1979-09-01', grid='col1', parent='CFSR')
experiments['max-3km'] = Exp(shortname='max-3km', name='max-3km', title='Max 3km (Ctrl)', begindate='1979-09-01', grid='col2', parent='Ctrl')
experiments['erai-3km'] = Exp(shortname='erai-3km', name='erai-3km', title='Max 3km (ERA-I)', begindate='1979-09-01', grid='col2', parent='ERA-I')
# these are all based on the "new" configuration (ARB3 domain)
experiments['new-ctrl'] = Exp(shortname='new', name='new-ctrl', title='New-1 (Noah-MP)', begindate='1979-01-01', grid='arb3')
experiments['new-ctrl-2050'] = Exp(shortname='new-2050', name='new-ctrl-2050', title='New-1 (2050)', begindate='2045-01-01', grid='arb3')
experiments['new-grell'] = Exp(shortname='grell', name='new-grell', title='New-1 (Grell)', begindate='1979-01-01', grid='arb3')
experiments['new-grell-old'] = Exp(shortname='new-grell-old', name='new-grell-old', title='New-1 (Grell, old NMP)', begindate='1979-01-01', grid='arb3')
experiments['new-nogulf'] = Exp(shortname='nogulf', name='new-nogulf', title='New-1 (no Gulf)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['new-noah'] = Exp(shortname='noah', name='new-noah', title='New (Noah)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['v35-noah'] = Exp(shortname='noah35', name='v35-noah', title='V35 & Noah (New)', begindate='1979-01-01', grid='arb2') # ARB2 domain
experiments['cfsr-new'] = Exp(shortname='cfsr-new', name='cfsr-new', title='New (CFSR)', begindate='1979-01-01', grid='arb3', parent='CFSR')
# these are all based on the "max" configuration (ARB2 domain)
experiments['max-diff'] = Exp(shortname='diff', name='max-diff', title='Max-1 (diff)', begindate='1979-01-01', grid='arb2')
experiments['max-clm'] = Exp(shortname='max-clm', name='max-clm', title='Max-1 (CLM)', begindate='1979-01-01', grid='arb2')
experiments['max-kf'] = Exp(shortname='max-kf', name='max-kf', title='Max-1 (KF)', begindate='1979-01-01', grid='arb2')
experiments['max-nosub'] = Exp(shortname='nosub', name='max-nosub', title='Max-1 (nosub)', begindate='1979-01-01', grid='arb2')
experiments['max-nosc'] = Exp(shortname='nosc', name='max-nosc', title='Max-1 (nosc)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp'] = Exp(shortname='max-nmp', name='max-nmp', title='Max-1 (Noah-MP)', begindate='1979-01-01', grid='arb2')
experiments['max-nmp-2050'] = Exp(shortname='max-nmp-2050', name='max-nmp-2050', title='Max-1 (Noah-MP, 2050)', begindate='2045-01-01', grid='arb2')
experiments['max-nmp-old'] = Exp(shortname='max-nmp-old', name='max-nmp-old', title='Max-1 (old NMP)', begindate='1979-01-01', grid='arb2')
experiments['max-nofdda'] = Exp(shortname='max-nofdda', name='max-nofdda', title='Max-1 (No Nudging)', begindate='1979-01-01', grid='arb2')
experiments['max-fdda'] = Exp(shortname='max-fdda', name='max-fdda', title='Max-1 (Nudging++)', begindate='1979-01-01', grid='arb2')
experiments['max-hilev'] = Exp(shortname='hilev', name='max-hilev', title='Max-1 (hilev)', begindate='1979-01-01', grid='arb2')
experiments['max-1deg'] = Exp(shortname='1deg', name='max-1deg', title='Max-1 (1 deg.)', begindate='1979-01-01', grid='arb2')
experiments['max-1deg-2050'] = Exp(shortname='1deg-2050', name='max-1deg-2050', title='Max-1 (1 deg., 2050)', begindate='2045-01-01', grid='arb2')
experiments['max-1deg-2100'] = Exp(shortname='1deg-2100', name='max-1deg-2100', title='Max-1 (1 deg., 2100)', begindate='2085-01-01', grid='arb2')
experiments['max-cg'] = Exp(shortname='cg', name='max-cg', title='Max-1 (CG)', begindate='1979-01-01', grid='arb2')
experiments['max-gulf'] = Exp(shortname='gulf', name='max-gulf', title='Max-1 (Gulf)', begindate='1979-01-01', grid='arb3') # ARB3 domain
experiments['max-ensemble'] = Exp(shortname='max-ens', name='max-ensemble', title='Max Ensemble Mean', begindate='1979-01-01', grid='arb2', parent='CESM')
experiments['max-ctrl'] = Exp(shortname='max', name='max-ctrl', title='Max-1', begindate='1979-01-01', grid='arb2', ensemble='max-ensemble')
experiments['max-ens-A'] = Exp(shortname='max-A', name='max-ens-A', title='Max-A', begindate='1979-01-01', grid='arb2', parent='Ens-A', ensemble='max-ensemble')
experiments['max-ens-B'] = Exp(shortname='max-B', name='max-ens-B', title='Max-B', begindate='1979-01-01', grid='arb2', parent='Ens-B', ensemble='max-ensemble')
experiments['max-ens-C'] = Exp(shortname='max-C', name='max-ens-C', title='Max-C', begindate='1979-01-01', grid='arb2', parent='Ens-C', ensemble='max-ensemble')
experiments['max-ensemble-2050'] = Exp(shortname='max-ens-2050', name='max-ensemble-2050', title='Max Ensemble Mean (2050)', begindate='2045-01-01', grid='arb2')
experiments['max-ctrl-2050'] = Exp(shortname='max-2050', name='max-ctrl-2050', title='Max-1 (2050)', begindate='2045-01-01', grid='arb2', parent='Ctrl-2050', ensemble='max-ensemble-2050')
experiments['max-ens-A-2050'] = Exp(shortname='max-A-2050', name='max-ens-A-2050', title='Max-A (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-A-2050', ensemble='max-ensemble-2050')
experiments['max-ens-B-2050'] = Exp(shortname='max-B-2050', name='max-ens-B-2050', title='Max-B (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-B-2050', ensemble='max-ensemble-2050')
experiments['max-ens-C-2050'] = Exp(shortname='max-C-2050', name='max-ens-C-2050', title='Max-C (2050)', begindate='2045-01-01', grid='arb2', parent='Ens-C-2050', ensemble='max-ensemble-2050')
experiments['max-seaice-2050'] = Exp(shortname='seaice-2050', name='max-seaice-2050', title='Seaice (2050)', begindate='2045-01-01', grid='arb2', parent='Seaice-2050')
experiments['max-ensemble-2100'] = Exp(shortname='max-ens-2100', name='max-ensemble-2100', title='Max Ensemble Mean (2100)', begindate='2085-01-01', grid='arb2')
experiments['max-ctrl-2100'] = Exp(shortname='max-2100', name='max-ctrl-2100', title='Max-1 (2100)', begindate='2085-01-01', grid='arb2', parent='Ctrl-2100', ensemble='max-ensemble-2100')
experiments['max-ens-A-2100'] = Exp(shortname='max-A-2100', name='max-ens-A-2100', title='Max-A (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-A-2100', ensemble='max-ensemble-2100')
experiments['max-ens-B-2100'] = Exp(shortname='max-B-2100', name='max-ens-B-2100', title='Max-B (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-B-2100', ensemble='max-ensemble-2100')
experiments['max-ens-C-2100'] = Exp(shortname='max-C-2100', name='max-ens-C-2100', title='Max-C (2100)', begindate='2085-01-01', grid='arb2', parent='Ens-C-2100', ensemble='max-ensemble-2100')
experiments['max-seaice-2100'] = Exp(shortname='seaice-2100', name='max-seaice-2100', title='Seaice (2100)', begindate='2085-01-01', grid='arb2', parent='Seaice-2100')
experiments['cfsr-max'] = Exp(shortname='cfsr', name='cfsr-max', title='Max (CFSR)', begindate='1979-01-01', grid='arb2', parent='CFSR')
experiments['erai-max'] = Exp(shortname='erai', name='erai-max', title='Max (ERA-I)', begindate='1979-01-01', grid='arb2', parent='ERA-I')
# these are all based on the old configuration (original + RRTMG, ARB2)
experiments['ctrl-1'] = Exp(shortname='ctrl', name='ctrl-1', title='Ctrl-1', begindate='1979-01-01', grid='arb2')
experiments['ctrl-2050'] = Exp(shortname='ctrl-2050', name='ctrl-2050', title='Ctrl-1 (2050)', begindate='2045-01-01', grid='arb2')
experiments['tiedtke-ctrl'] = Exp(shortname='tiedt', name='tiedtke-ctrl', title='Tiedtke (Ctrl)', begindate='1979-01-01', grid='arb2') 
experiments['tom-ctrl'] = Exp(shortname='tom', name='tom-ctrl', title='Thompson (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['wdm6-ctrl'] = Exp(shortname='wdm6', name='wdm6-ctrl', title='WDM6 (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['milbrandt-ctrl'] = Exp(shortname='milb', name='milbrandt-ctrl', title='Milbrandt-Yau (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['epssm-ctrl'] = Exp(shortname='epssm', name='epssm-ctrl', title='Wave Damping (Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpnew-ctrl'] = Exp(shortname='nmpnew', name='nmpnew-ctrl', title='New (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpbar-ctrl'] = Exp(shortname='nmpbar', name='nmpbar-ctrl', title='Barlage (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
experiments['nmpsnw-ctrl'] = Exp(shortname='nmpsnw', name='nmpsnw-ctrl', title='Snow (Noah-MP, Ctrl)', begindate='1979-01-01', grid='arb2')
# these are all based on the original configuration (mostly ARB1 domain)
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
ensembles = WRF_ens = OrderedDict()
ensemble_list = list(set([exp.ensemble for exp in experiments.values() if exp.ensemble]))
ensemble_list.sort()
for ensemble in ensemble_list:
  #print ensemble, experiments[ensemble].shortname
  members = [exp for exp in experiments.values() if exp.ensemble and exp.ensemble == ensemble]
  members.sort()
  ensembles[experiments[ensemble].shortname] = members

if __name__ == '__main__':
    
  ## view/test ensembles
  for name,members in WRF_ens.iteritems():
    s = '  {:s}: '.format(name)
    for member in members: s += ' {:s},'.format(member.name)
    print(s)