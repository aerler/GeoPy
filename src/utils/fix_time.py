#!/usr/local/bin/python3.6
# encoding: utf-8
'''
A simple script to fix the time coordinate/index in concatenated NetCDF files; the script replaces the time 
coordinate with the number of month since a reference time (1979-01) by default; it also reads the NetCDF
attribute 'begin_date' and changes 'end_date' based on the length of the time axis (assimung monthly steps).

@author:     Andre R. Erler

@copyright:  2019 Aquanty Inc. All rights reserved.

@license:    GPL v3

@contact:    aerler@aquanty.com
@deffield    updated: 30/05/2019
'''

import os, sys
import numpy as np
import netCDF4 as nc
import pandas as pd

# find reference date
ref_date = os.getenv('NC_REFERENCE_DATE', '1979-01')
print("Using reference date: "+ref_date)
ref_dt = pd.to_datetime(ref_date)

# read start date option
master_start_date = os.getenv('NC_REFERENCE_DATE', None)
if master_start_date is not None:
    master_start_dt = pd.to_datetime(master_start_date)

# get file list
file_list = sys.argv[1:] # first is script name
# print("Looping over file list:")
# print(file_list)
# print("")

# loop over file list
for ncfile in file_list:
  
    if not os.path.exists(ncfile):
        raise IOError(ncfile)

    print("Opening file: '{}'".format(ncfile))
    # open file
    ds = nc.Dataset(ncfile,'a')
    if master_start_date is None: 
        start_date = ds.getncattr('begin_date')
        print("  Start date ('begin_date'): "+start_date)
        start_dt = pd.to_datetime(start_date)
    else:
        start_date = master_start_date
        start_dt = master_start_dt
        ds.setncattr('begin_date',start_date)
    
    # compute offset to reference
    start_month = (start_dt.year - ref_dt.year)*12 + (start_dt.month - ref_dt.month)
    
    # fix time axis
    tax = ds['time']
    tax_len = len(tax)
    print('  New time index: {} - {}'.format(start_month,start_month+len(tax)))
    tax[:] = np.arange(start_month,start_month+tax_len, dtype=tax.dtype)
    # change time units
    tax.setncattr('units','month since '+ref_date)
    
    # compute and set end date
    end_year = start_dt.year + (start_dt.month + tax_len -1)//12
    end_month = (start_dt.month + tax_len -1)%12
    end_date = '{YEAR:04d}-{MON:02d}'.format(YEAR=end_year,MON=end_month)
    print("  End date ('end_date'): "+end_date)
    ds.setncattr('end_date',end_date)
    
    # save and close file
    ds.sync(); ds.close()
    