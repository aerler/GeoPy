'''
Created on 2013-09-09

Some tools and data that are used by many datasets, but not much beyond that.

@author: Andre R. Erler, GPL v3
'''

import numpy as np

# days per month
days_per_month = np.array([31,28.2425,31,30,31,30,31,31,30,31,30,31]) # 97 leap days every 400 years
# N.B.: the Gregorian calendar repeats every 400 years
days_per_month_365 = np.array([31,28,31,30,31,30,31,31,30,31,30,31]) # no leap day
# human-readable names
name_of_month = ['January  ', 'February ', 'March    ', 'April    ', 'May      ', 'June     ', #
                 'July     ', 'August   ', 'September', 'October  ', 'November ', 'December ']


def translateVarNames(varlist, varatts):
  ''' Simple function to replace names in a variable list with their original names as inferred from the 
      attributes dictionary. Note that this requires the dictionary to have the field 'name'. '''
  if not isinstance(varlist,list) or not isinstance(varatts,dict): raise TypeError  
  # cycle over names in variable attributes (i.e. final names, not original names)  
  for key,atts in varatts.iteritems():
    if 'name' in atts and atts['name'] in varlist: 
      varlist[varlist.index(atts['name'])] = key # original name is used as key in the attributes dict
  # return varlist with final names replaced by original names
  return varlist
  
# data root folder
import socket
hostname = socket.gethostname()
if hostname=='komputer':
  data_root = '/home/DATA/DATA/'
#  root = '/media/tmp/' # RAM disk for development
else:
  data_root = '/home/me/DATA/PRISM/'
