'''
Created on July 6, 2020

A module to load data from various station datasets as time series and convert to NetCDF

@author: Andre R. Erler, GPL v3
'''



# external imports
import datetime as dt
import pandas as pd
import os.path as osp
import numpy as np
import netCDF4 as nc # netCDF4-python module
import xarray as xr
# internal imports
from datasets.common import getRootFolder
from geodata.netcdf import DatasetNetCDF
from processing.newvars import e_sat, computeNetRadiation, computePotEvapPM, toa_rad, clearsky_rad,\
  net_longwave_radiation, computePotEvapHog, computePotEvapHar, computePotEvapTh,\
  computePotEvapPT
# for georeferencing
from geospatial.netcdf_tools import addTimeStamps
from geospatial.xarray_tools import loadXArray, updateVariableAttrs, computeNormals

## Meta-vardata

dataset_name = 'ClimateStations'
root_folder = getRootFolder(dataset_name=dataset_name, fallback_name='HGS') # get dataset root folder based on environment variables

# attributes of variables in final collection
varatts = dict(precip   = dict(name='precip', units='kg/m^2/s', long_name='Total Precipitation', positive=1, limits=(0,0.0025)),
               MaxPrecip_1h  = dict(name='MaxPrecip_1h', units='kg/m^2/s', long_name='Maximum Hourly Precipitation', positive=1, limits=(0,0.01)),
               MaxLiqprec_1h = dict(name='MaxLiqprec_1h', units='kg/m^2/s', long_name='Maximum Hourly Rain', positive=1, limits=(0,0.01)),
               liqprec  = dict(name='liqprec', units='kg/m^2/s', long_name='Rainfall (tipping bucket)', positive=1, limits=(0,0.0025)),
               snowfall = dict(name='snowfall', units='cm', long_name='Snowfall Depth', positive=1, limits=(0,0.0025)),
               snowh    = dict(name='snowh', units='m', long_name='Snow Depth', positive=1, limits=(0,2)),
               pet      = dict(name='pet', units='kg/m^2/s', long_name='PET (Penman-Monteith)'),
               pet_sol  = dict(name='pet_sol', units='kg/m^2/s', long_name='PET (solar radiation only)'),
               pet_dgu  = dict(name='pet_dgu', units='Pa/K', long_name='PET Denominator'),
               pet_rad  = dict(name='pet_rad', units='kg/m^2/s', long_name='PET Radiation Term'),
               pet_wnd  = dict(name='pet_wnd', units='kg/m^2/s', long_name='PET Wind Term'),
               pet_pt   = dict(name='pet_pt', units='kg/m^2/s', long_name='PET (Priestley-Taylor)'),
               pet_pts  = dict(name='pet_pts', units='kg/m^2/s', long_name='PET (Priestley-Taylor, approx. LW)'),
               pet_hog  = dict(name='pet_hog', units='kg/m^2/s', long_name='PET (Hogg 1997)'),
               pet_har  = dict(name='pet_har', units='kg/m^2/s', long_name='PET (Hargeaves)'),
               pet_haa  = dict(name='pet_haa', units='kg/m^2/s', long_name='PET (Hargeaves-Allen)'),
               pet_th   = dict(name='pet_th', units='kg/m^2/s', long_name='PET (Thornthwaite)'),
               pmsl     = dict(name='pmsl', units='Pa', long_name='Mean Sea-level Pressure', positive=1, limits=(8e4,11e4)), # sea-level pressure
               ps       = dict(name='ps', units='Pa', long_name='Surface Air Pressure', positive=1, limits=(7e4,11e4)), # surface pressure
               Ts       = dict(name='Ts', units='K', long_name='Skin Temperature', positive=1, limits=(200,350)), # average skin temperature
               TSmin    = dict(name='TSmin', units='K', long_name='Minimum Skin Temperature', positive=1, limits=(200,350)), # minimum skin temperature
               TSmax    = dict(name='TSmax', units='K', long_name='Maximum Skin Temperature', positive=1, limits=(200,350)), # maximum skin temperature
               T2       = dict(name='T2', units='K', long_name='2m Temperature', positive=1, limits=(220,340)), # 2m average temperature
               Tmin     = dict(name='Tmin', units='K', long_name='Minimum 2m Temperature', positive=1, limits=(220,340)), # 2m minimum temperature
               Tmax     = dict(name='Tmax', units='K', long_name='Maximum 2m Temperature', positive=1, limits=(220,340)), # 2m maximum temperature
               Q2       = dict(name='Q2', units='Pa', long_name='Water Vapor Pressure (derived)', positive=1, limits=(0,5e3)), # 2m water vapor pressure
               Q2max    = dict(name='Q2max', units='Pa', long_name='Maximum Water Vapor Pressure', positive=1, limits=(0,5e3)), # maximum diurnal water vapor pressure
               Q2min    = dict(name='Q2min', units='Pa', long_name='minimum Water Vapor Pressure', positive=1, limits=(0,5e3)), # minimum diurnal water vapor pressure
               RH       = dict(name='RH', units='', long_name='Relative Humidity', positive=1, limits=(0,1.1)), # 2m relative humidity
               RHmax    = dict(name='RHmax', units='', long_name='Maximum Relative Humidity', positive=1, limits=(0,1.1)), # 2m diurnal maximum relative humidity
               RHmin    = dict(name='RHmin', units='', long_name='Minimum Relative Humidity', positive=1, limits=(0,1.1)), # 2m diurnal minimum relative humidity
               U2       = dict(name='U2', units='m/s', long_name='2m Wind Speed', positive=1, limits=(0,50)), # 2m wind speed
               U2_dir   = dict(name='U2_dir', units='deg', long_name='2m Wind Direction', positive=1, limits=(0,360)), # 2m wind direction
               U2max    = dict(name='U2max', units='m/s', long_name='2m Maximum Wind Speed', positive=1, limits=(0,50)), # 2m maximum diurnal wind speed
               U10      = dict(name='U10', units='m/s', long_name='10m Wind Speed', positive=1, limits=(0,100)), # 2m wind speed
               U10_dir  = dict(name='U10_dir', units='deg', long_name='10m Wind Direction', positive=1, limits=(0,360)), # 10m wind direction
               U10max   = dict(name='U10max', units='m/s', long_name='10m Maximum Wind Speed', positive=1, limits=(0,150)), # 10m maximum diurnal wind speed
               DNSW     = dict(name='DNSW', units='W/m^2', long_name='Downward Shortwave Radiation', positive=1, limits=(0,500)),
               UPSW     = dict(name='UPSW', units='W/m^2', long_name='Upward Shortwave Radiation', positive=1, limits=(0,200)),
               DNLW     = dict(name='DNLW', units='W/m^2', long_name='Downward Longwave Radiation', positive=1, limits=(100,500)),
               UPLW     = dict(name='UPLW', units='W/m^2', long_name='Upward Longwave Radiation', positive=1, limits=(100,500)),
               netrad   = dict(name='netrad', units='W/m^2', long_name='Net Downward Radiation', limits=(-100,400)), # radiation absorbed by the ground
               netrad_lw  = dict(name='netrad_lw', units='W/m^2', long_name='Net Longwave Radiation'), # net LW radiation absorbed by the ground
               DNLW_raw   = dict(name='DNLW_raw', units='W/m^2', long_name='Downward Longwave Radiation (uncorrected)'),
               UPLW_raw   = dict(name='UPLW_raw', units='W/m^2', long_name='Upward Longwave Radiation (uncorrected)'),
               DNSW_alt   = dict(name='DNSW_alt', units='W/m^2', long_name='Downward Shortwave Radiation (alternate)', positive=1, limits=(0,500)),
               netrad_raw = dict(name='netrad_raw', units='W/m^2', long_name='Net Downward Radiation (uncorrected)'), # radiation absorbed by the ground
               Ra       = dict(name='Ra', units='W/m^2', long_name='Extraterrestrial Solar Radiation'), # ToA radiation
               Rs0      = dict(name='Rs0', units='W/m^2', long_name='Clear-sky Solar Radiation'), # at the surface, based on elevation
               sunfrac  = dict(name='sunfrac', units='\%', long_name='Fraction of Clear Sky', positive=1, limits=(0,100)), # direct observation... not sure how...
               d_gw     = dict(name='d_gw', units='m', long_name='Depth to Groundwater Table', positive=1, limits=(0,10)), # Elora station...
               # axes
               time    = dict(name='time', units='days', long_name='Time in Days'), # time coordinate
               )
varlist = varatts.keys()
ignore_list = []

## station meta data
class StationMeta(object):
    name = None
    title = None
    region = None
    lat = None
    lon = None
    zs = None
    folder = None
    filelist = None
    file_fmt = None
    testfile = None
    readargs = None
    varatts = None
    minmax = None
    sampling = 'h'
    
    def __init__(self, name=None, title=None, region=None, lat=None, lon=None, zs=None, filelist=None, filename=None, 
                 testfile=None, file_fmt=None, folder=None, readargs=None, varatts=None, minmax=None, sampling='h'):
        ''' assign some values with smart defaults '''
        self.name = name
        self.title = title if title else name
        self.region = region
        self.lat = lat; self.lon = lon; self.zs = zs
        self.folder = folder if folder else osp.join(root_folder,region,'source',name) # default folder
        if filename is not None: # generate filelist
            if filelist is not None: raise ValueError(filelist)
            filelist = [filename]
        self.filelist = filelist
        if file_fmt is None: # auto-detect file format
            if all(fn.lower().endswith(('.xls','.xlsx')) for fn in filelist): file_fmt = 'xls'
            elif all(fn.lower().endswith('.csv') for fn in filelist): file_fmt = 'csv'
            else:
                raise NotImplementedError('Cannot determine source format:'.format(file_fmt))
        self.file_fmt = file_fmt
        self.testfile = testfile
        self.readargs = readargs if readargs else dict()
        self.varatts = varatts if varatts else dict()
        self.minmax = minmax if minmax else dict()
        self.sampling = sampling

      
# Ontario stations
ontario_station_list = dict()
# UTMMS station
# Outages - Radiation Balance:
#  2016-07-27 - 2018-04-05
#  2012-12-01 - 2013-01-01
#             - 2007-10-01
# Outages - Wind:
#  2009-02-26 - 2009-03-31
#  2006-06-02 - 2006-07-04
#  2000-11-09 - 2000-11-28
# Outages - RelHum:
#  2005-02-01 - 2005-02-09
#  2002-03-28 - 2002-04-17
# Outages - all: 
#  2017-08-31 - 2017-11-21
#  2006-06-26 - 2006-07-04
#  2004-12-10 - 2004-12-14
#  2003-12-19 - 2004-01-09
#  2003-09-09 - 2003-10-16
#  2000-05-02 - 2000-05-17
stn_varatts = dict(temp_cel = dict(name='T2', offset=273.15),
                   rel_hum_pct = dict(name='RH', scalefactor=0.01),
                   wind_spd_ms = dict(name='U2',),
                   wind_dir_deg = dict(name='U2_dir',),
                   precip_mm = dict(name='precip', scalefactor=24./86400), # convert hourly accumulation to daily
                   glb_rad_wm2 = dict(name='DNSW_alt',), # most likely downwelling solar radiation
                   cnr1_net_rad_total = dict(name='netrad_raw'),
                   cnr1_sw_in = dict(name='DNSW'),
                   cnr1_sw_out = dict(name='UPSW'),
                   cnr1_lw_in_cor = dict(name='DNLW'),
                   cnr1_lw_out_cor = dict(name='UPLW'),
                   cnr1_lw_in_raw = dict(name='DNLW_raw'),
                   cnr1_lw_out_raw = dict(name='UPLW_raw'),
                   cnr1_temp_c = dict(name='Ts', offset=273.15), )
stn_readargs = dict(header=0, index_col=0, usecols=['timestamp_est'], parse_dates=True, na_values=['*','no data'])
minmax_vars = dict(T2=('Tmin','Tmax'), Ts=('TSmin','TSmax'), RH=('RHmin','RHmax'), Q2=('Q2min','Q2max'), 
                   precip=(None,'MaxPrecip_1h'), U2=(None,'U2max'))
meta = StationMeta(name='UTM', title='University of Toronto, Mississauga', region='Ontario',
                   lat=43.55, lon=-79.66, zs=112,
                   filename='UTMMS Full Data Jan 1 2000 to Sept 26 2018.xlsx', testfile='UTM_test.xlsx',
                   readargs=stn_readargs, varatts=stn_varatts, minmax=minmax_vars, sampling='h')
ontario_station_list[meta.name] = meta

# Elora Research Station (ERS) (University of Guelph)
stn_varatts = dict(PRESS = dict(name='ps', scalefactor=1000),
                   ATEMP_AV = dict(name='T2', offset=273.15),
                   RH_AV = dict(name='RH', scalefactor=0.01,),
                   # WS10_P5S, WS10_PT, and WS10_PDR refer to peak wind speed
                   WS10_AV = dict(name='U10',),
                   WS10_AVD = dict(name='U10_dir',),
                   WS2_AV = dict(name='U2',),
                   PRECIP = dict(name='precip', scalefactor=24./86400), # convert hourly accumulation to daily
                   TBRG = dict(name='liqprec', scalefactor=24./86400), # tipping bucket rain gauge
                   SNOWFALL = dict(name='snowfall', scalefactor=24./86400), # hourly snow fall in cm
                   SNOW_GR = dict(name='snowh', scalefactor=0.01), # snow depth on the ground
                   SOL_RAD = dict(name='DNSW', scalefactor=1e3/3.6), # downwelling solar radiation
                   LW_RAD = dict(name='DNLW', scalefactor=1e3/3.6), # most likely downwelling LW radiation
                   SUNSHINE = dict(name='sunfrac'),
                   WATER = dict(name='d_gw', scalefactor=0.01), )
def eloraDateParser(arg):
    args = arg.split()
    year = int(args[0]); day = int(args[1]); hour = int(args[2])//100
    datetime = dt.datetime(year, 1, 1,) + dt.timedelta(days=day-1, hours=hour)
    # N.B.: for now we are ignoring timezones...
    #print(arg, datetime)
    return datetime
stn_readargs = dict(header=0, index_col=0, usecols=['YEAR','JD','TIME'], na_values=['9999','6999'],
                    parse_dates=[['YEAR','JD','TIME']], date_parser=eloraDateParser,)
minmax_vars = dict(T2=('Tmin','Tmax'), RH=('RHmin','RHmax'), Q2=('Q2min','Q2max'), liqprec=(None,'MaxLiqprec_1h'),
                   precip=(None,'MaxPrecip_1h'), U2=(None,'U2max'),U10=(None,'U10max'))
meta = StationMeta(name='Elora', title='Elora Research Station, Univ. Guelph', region='Ontario',
                   lat=43.64, lon=-80.4, zs=374,# elevation as per https://www.mapcoordinates.net/en
#                    testfile='ERS_weather_data_hourly_2003.csv',
                   testfile=['ERS_weather_data_hourly_{:04d}.csv'.format(year) for year in range(2010,2011)],
                   filelist=['ERS_weather_data_hourly_{:04d}.csv'.format(year) for year in range(2003,2019)], 
                   readargs=stn_readargs, varatts=stn_varatts, minmax=minmax_vars, sampling='h')
ontario_station_list[meta.name] = meta


def getFolderFileName(station=None, region='Ontario', period=None, mode='daily'):
    ''' return folder and file name in standard format '''
    mode = mode.lower()
    mode_str = mode
    mode_folder = 'stnavg'
    if mode == 'daily': mode_folder = 'station_daily'
    elif mode in ('monthly','month'): pass # defaults
    elif mode in ('clim','climatology','normals'):
        if isinstance(period,str): prdstr = period
        elif isinstance(period,(tuple,list)):
            assert len(period) == 2
            prdstr = '{:04d}-{:04d}'.format(*period)
        else:
            raise ValueError("To load climatologies, a period (string) has to be defined! (period={})".format(period))
        mode_str = 'clim_'+prdstr
    else: raise NotImplementedError(mode)
    folder = '{:s}/{:s}/{:s}/'.format(root_folder,region,mode_folder)
    filename = "{:s}_{:s}.nc".format(station,mode_str).lower()
    # return
    return folder,filename


def _createStationSample(name=None, station_list=None):
    ''' create an empty sample dataset from a list of stations '''
    # select station list
    n = len(station_list)
    # create coordinate/meta data arrays
    stn_names = []; stn_titles = []
    stn_lat = np.empty((n,), dtype='float64'); stn_lon = np.empty((n,), dtype='float64'); stn_zs = np.empty((n,), dtype='float64')
    i = 0 # iterate over station list and fill arrays
    for key,meta in station_list.items():
        stn_names.append(key); stn_titles.append(meta.title)
        stn_lat[i] = meta.lat; stn_lon[i] = meta.lon; stn_zs[i] = meta.zs
        i += 1
    # creat dataset
    from geodata.base import Axis, Variable, Dataset
    ds = Dataset(name=name, title=name+' Stations')
    stnax =  Axis(name='station', coord=np.arange(1,n+1), units='#')
    ds += Variable(name='station_name', data=np.array(stn_names), units='', axes=(stnax,))
    ds += Variable(name='station_title', data=np.array(stn_titles), units='', axes=(stnax,))
    ds += Variable(name='stn_lat', data=np.array(stn_lat), units='deg N', axes=(stnax,))
    ds += Variable(name='stn_lon', data=np.array(stn_lon), units='deg E', axes=(stnax,))
    ds += Variable(name='stn_zs', data=np.array(stn_zs), units='m', axes=(stnax,))
    # return template dataset
    return ds
  
def loadStationSample(name=None, filetype=None):
    ''' load an empty sample dataset, based in meta data '''
    # select station list
    if filetype.lower() == 'ontario':
        station_list = ontario_station_list
    ds = _createStationSample(name=filetype if name is None else name, station_list=station_list)
    # return template dataset
    return ds


## functions to load station data from source files


def loadStation_Src(station, region='Ontario', station_list=None, ldebug=False, varatts=varatts, **kwargs):
    ''' load station data from original source into pandas dataframe '''
    # get station meta data
    if station_list is None:
        station_list = globals()[region.lower()+'_station_list']
    station = station_list[station]
    # figure out read parameters
    readargs = dict() # default args
    readargs.update(station.readargs); readargs.update(kwargs)
    # add column/variables
    if 'usecols' in readargs: readargs['usecols'].extend(station.varatts.keys())
    else: readargs['usecols'] = station.varatts.keys()
    ## load file(s) in Pandas
    if ldebug:
        if isinstance(station.testfile, str): filelist = [station.testfile]
        else: filelist = station.testfile
    else: filelist = station.filelist 
    df_list = []
    for filename in filelist:
        filepath = osp.join(station.folder,filename)
        if station.file_fmt == 'xls':
            if ldebug: print(readargs)
            df_list.append(pd.read_excel(filepath, **readargs))
        elif station.file_fmt == 'csv':
            if ldebug: print(readargs)
            df_list.append(pd.read_csv(filepath, **readargs))
        else:
            raise NotImplementedError(station.file_fmt)
    # join dataframes
    if len(df_list) == 1: 
        df = df_list[0]
    else:
        # remove overlapping dates between files
        df = df_list[0]
        for tdf in df_list[1:]:
            enddate = df.index[-1]
            i = 0 # find first date that is past the previous end date
            while enddate >= tdf.index[i]: i += 1
            if ldebug: print(enddate,tdf.index[i]) # diagnostic
            tdf = tdf[i:]; tdf = tdf[~tdf.index.duplicated(keep='first')]
            df = pd.concat([df,tdf], verify_integrity=True) # concat, but still check for duplicates
    # rename columns
    stn_varatts = station.varatts.copy()
    df = df.rename(columns={col:atts['name'] for col,atts in stn_varatts.items()}) # rename variables/columns
    df = df.rename_axis("time", axis="index") # rename axis/index to time
    ravmap = {atts['name']:col for col,atts in stn_varatts.items()}
    # compute water vapor pressure (non-linear, hence before aggregation)
    varlist = df.columns
    if ldebug: print(varlist)
    if 'Q2' not in varlist and 'T2' in varlist and 'RH' in varlist:
        lKelvin = stn_varatts[ravmap['T2']].get('offset',0) == 0
        RH_scale = stn_varatts[ravmap['RH']].get('scalefactor',1) 
        df['Q2'] = e_sat(df['T2'], lKelvin=lKelvin) * (df['RH']*RH_scale).clip(0,1)
    ## aggregate to daily
    if station.sampling != 'D':
        rdf = df.resample('1D',)
        df = rdf.mean()
        # add min/max
        for var0,minmax in station.minmax.items():
            if var0 in df.columns:
                for mvar,mode in zip(minmax,('min','max')):
                    if mvar: # could be None if either min or max is not required
                        df[mvar] = getattr(rdf[var0],mode)() # compute min/max
                        # add new attributes (same as master var)
                        atts = stn_varatts[ravmap[var0]].copy() if var0 in ravmap else dict() 
                        atts['name'] = mvar
                        stn_varatts[mvar] = atts
    ## format dataframe
    for atts in stn_varatts.values():
        varname = atts['name']; sf = atts.get('scalefactor',1); of = atts.get('offset',0)
        if sf != 1: df[varname] = df[varname] * sf
        if of != 0: df[varname] = df[varname] + of
    # clip data
    for varname,column in df.items():
        #print(varname)
        #print((column.values < 0).sum())
        if varname in varatts:
            atts = varatts[varname]
            if atts.get('positive',False):
                if ldebug: print(varname, (df[varname].values < 0).sum())
                column = column.clip(0, None)
            if 'limits' in atts:
                amin,amax = atts['limits']
                if amin is not None: column[column < amin] = np.NaN
                if amax is not None: column[column > amax] = np.NaN
            df[varname] = column
    # compute net radiation (linear, hence after aggregation)
    if 'netrad' not in varlist and all(radvar in varlist for radvar in ('DNSW','UPSW','DNLW','UPLW')):
        df['netrad'] = df['DNSW'] + df['DNLW'] - df['UPSW'] - df['UPLW']        
    # convert to xarray and add attributes
    xds = df.to_xarray()
    for varname,variable in xds.data_vars.items():
        if varname in varatts:
            variable.attrs.update(varatts[varname])
    xds.attrs['name'] = station.name; xds.attrs['title'] = station.title; xds.attrs['region'] = station.region
    xds.attrs['lat'] = station.lat; xds.attrs['lon'] = station.lon; xds.attrs['zs'] = station.zs
    ## add complex variables related to FAO PET
    # compute Penman-Monteith PET (only works with xarray)
    pet,pet_rad,pet_wnd = computePotEvapPM(xds, lterms=True, lrad=True, lA=False, lem=False, 
                                           lnetlw=False, lgrdflx=False, lpmsl=False, lxarray=True)
    xds['pet'] = pet; xds['pet_rad'] = pet_rad; xds['pet_wnd'] = pet_wnd
    # compute ToA and approximate clear-sky solar radiation
    Ra = toa_rad(time=xds['time'].data, lat=xds.attrs['lat'], lmonth=False, l365=False, time_offset=0, ldeg=True)
    xds['Ra'] = xr.DataArray(coords=(xds.coords['time'],), data=Ra, name='Ra', attrs=varatts['Ra'])
    Rs0 = clearsky_rad(Ra=Ra, zs=xds.attrs['zs'])
    xds['Rs0'] = xr.DataArray(coords=(xds.coords['time'],), data=Rs0, name='Rs0', attrs=varatts['Rs0']) 
    # compute PET using only direct solar radiation (and estimated longwave radiation)
    if 'DNSW_alt' in xds:
        tmp_ds = xds.drop(['netrad','DNSW','DNLW','UPSW','UPLW'], errors='ignore').rename(dict(DNSW_alt='DNSW'))
    else:
        tmp_ds = xds.drop(['netrad','DNLW','UPSW','UPLW'], errors='ignore')
    print(tmp_ds)
    # compute net longwave radiation and solar radiation-based PET
    netrad_lw = net_longwave_radiation(Tmin=tmp_ds['Tmin'], Tmax=tmp_ds['Tmax'], ea=tmp_ds['Q2'], Rs=tmp_ds['DNSW'], Rs0=tmp_ds['Rs0'])
    tmp_ds['netrad_lw'] = xr.DataArray(coords=(tmp_ds.coords['time'],), data=netrad_lw, name='netrad_lw', attrs=varatts['netrad_lw'])
    xds['netrad_lw'] = xr.DataArray(coords=(xds.coords['time'],), data=netrad_lw, name='netrad_lw', attrs=varatts['netrad_lw']) 
    if 'netrad' not in xds: 
        netrad = (1-0.23)*xds['DNSW'] + xds['netrad_lw']
        netrad.name = 'netrad'; netrad.attrs.update(varatts['netrad'])
        netrad.attrs['long_name'] = netrad.attrs['long_name'] + ' (estimated)'
        xds['netrad'] = netrad
    xds['pet_sol'] = computePotEvapPM(tmp_ds, lterms=False, lrad=False, lA=False, lnetlw=True, lgrdflx=False, lpmsl=False, lxarray=True)
    # compute PET based on Priestly-Taylor
    xds['pet_pt'] = computePotEvapPT(xds, lrad=True, lnetlw=False, lA=False, lem=False, lgrdflx=False, lpmsl=False, lxarray=True)
    xds['pet_pts'] = computePotEvapPT(tmp_ds, lrad=False, lnetlw=True, lA=False, lgrdflx=False, lpmsl=False, lxarray=True)
    # compute Hogg PET (based on Tmin & Tmax)
    xds['pet_hog'] = computePotEvapHog(xds, lmeans=False, lq2=False, zs='zs', lxarray=True)
    # compute Hargreaves PET (based on Tmin/Tmax and ToA/astronomical radiation)
    xds['pet_har'] = computePotEvapHar(xds, lat='lat', lmeans=False, l365=False, time_offset=0, lAllen=False, lxarray=True)
    xds['pet_haa'] = computePotEvapHar(xds, lat='lat', lmeans=False, l365=False, time_offset=0, lAllen=True, lxarray=True)
    # compute Thonthwaite PET, based on climatology and temperature
    if not ldebug: # requires a full year for climatology
        tmp_ds['climT2'] = computeNormals(xds['T2'], aggregation='month', time_stamp=False, lresample=True, time_name='month')
        xds['pet_th'] = computePotEvapTh(tmp_ds, climT2='climT2', lat='lat', l365=False, time_offset=0, p='center', lxarray=True)
    # return properly formatted dataset
    return xds


## functions to load station data (from daily NetCDF files)

def loadClimateStation(station, varlist=None, region='Ontario', name=None, time_slice=None, period=None, mode=None, 
                       lload=True, lxarray=False, varatts=varatts, **kwargs):
    ''' function to load formatted data from climate stations into xarray '''
    # determine folder and filename
    folder, filename = getFolderFileName(station=station, region=region, period=period, mode=mode)
    if name is None: name = station
    # load data as xarray or GeoPy dataset
    if lxarray:
        ## load dataset into xarray
        xr_args = dict(decode_cf=True, mask_and_scale=True, decode_times=True, autoclose=True)
        xr_args.update(kwargs)
        if lload:
            xds = xr.load_dataset(folder+filename, **xr_args) # load entire dataset into memory dirctly
        else:
            xds = xr.open_dataset(folder+filename, **xr_args) # open file and lazily load data into memory
        # update varatts and prune
        xds = updateVariableAttrs(xds, varatts=varatts, varmap=None, varlist=varlist)
        # apply time slice
        if time_slice: xds = xds.loc[{'time':slice(*time_slice),}] # slice time
        # some attributes
        xds.attrs['name'] = name
        # load time stamps (like coordinate variables)
        if 'time_stamp' in xds: xds['time_stamp'].load()
        dataset = xds
    else:
        ## load as GeoPy dataset
        dataset = DatasetNetCDF(name=name, filelist=[folder+filename], varlist=varlist, multifile=False, **kwargs)
        if lload: dataset.load()
    # return dataset
    return dataset


def loadClimStn_Daily(station, region='Ontario', time_slice=None, **kwargs):
    ''' wrapper to load daily station data '''
    return loadClimateStation(station, region=region, time_slice=time_slice, mode='daily', **kwargs)

def loadClimStn_TS(station, region='Ontario', time_slice=None, **kwargs):
    ''' wrapper to load monthly transient station data '''
    return loadClimateStation(station, region=region, time_slice=time_slice, mode='monthly', **kwargs)

def loadClimStn(station, region='Ontario', period=None, **kwargs):
    ''' wrapper to load monthly transient station data '''
    return loadClimateStation(station, region=region, period=period, mode='clim', **kwargs)


if __name__ == '__main__':
  
    import time
    print('pandas version:',pd.__version__)
  
    work_list = []
    work_list += ['convert_stations']
#     work_list += ['load_Daily']
    work_list += ['compute_monthly']
#     work_list += ['load_source']
#     work_list += ['load_Monthly']
    work_list += ['load_Normals']
    
    # settings
#     station = 'UTM'
    station = 'Elora'
    region = 'Ontario'
    time_slice = ('2011-01-01','2017-12-31')
    lxarray = True

    # loop over workloads
    for mode in work_list:
      
        if mode == 'load_Normals':
            
            xds = loadClimStn(station=station, region='Ontario', period=(2011,2018), lxarray=lxarray)
            
            print(xds)
            if lxarray: print(xds.attrs)
            print()
            
            if lxarray: var0 = next(iter(xds.data_vars.values()))
            else: var0 = next(iter(xds.variables.values()))
            print(var0)
            if lxarray: print(var0.attrs)
            
        elif mode == 'load_Monthly':
            
            xds = loadClimStn_TS(station=station, region='Ontario', time_slice=time_slice, lxarray=lxarray)
            
            print(xds)
            if lxarray: print(xds.attrs)
            print()
            
            if lxarray: var0 = next(iter(xds.data_vars.values()))
            else: var0 = next(iter(xds.variables.values()))
            print(var0)
            if lxarray: print(var0.attrs)
            
        elif mode == 'compute_monthly':
            
            # load data (into memory)
            xds = loadClimStn_Daily(station=station, region='Ontario', time_slice=None, lxarray=True)
            
            # aggregate month
            rds = xds.resample(time='MS',skipna=True,).mean()
            print(rds)
            print('')
            
            ## save monthly timeseries
            # define destination file
            nc_folder, nc_filename = getFolderFileName(station=station, region=region, mode='monthly')
            nc_filepath = nc_folder + nc_filename
            print("\nExporting Monthly Timeseries to NetCDF-4 file:\n '{}'".format(nc_filepath))
            # write to NetCDF
            rds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=['time'], engine='netcdf4')
            # update time information
            print("\nAdding human-readable time-stamp variable ('time_stamp')\n")
            ncds = nc.Dataset(nc_filepath, mode='a')
            ncts = addTimeStamps(ncds, units='month') # add time-stamps
            rds['time_stamp'] = xr.DataArray(data=ncts[:], coords=(rds.coords['time'],),)
            ncds.close()              
            
            ## aggregate to normals and save
            # trim
            rds = rds.loc[{'time':slice(*time_slice),}]
            # compute normals
            cds = computeNormals(rds, aggregation='month', time_stamp='time_stamp', lresample=False, time_name='time')
            print(cds)
            print('')
    
            # save normals
            nc_folder, nc_filename = getFolderFileName(station=station, region=region, mode='clim', period=cds.period) # define destination file
            nc_filepath = nc_folder + nc_filename
            print("\nExporting Monthly Normals to NetCDF-4 file:\n '{}'".format(nc_filepath))
            # write to NetCDF
            cds.to_netcdf(nc_filepath, mode='w', format='NETCDF4', unlimited_dims=None, engine='netcdf4')
            
    
        elif mode == 'load_Daily':
            
            xds = loadClimStn_Daily(station=station, region='Ontario', time_slice=time_slice, lload=True, lxarray=lxarray)
            
            print(xds)
            if lxarray: print(xds.attrs)
            print()
            
            if lxarray: var0 = next(iter(xds.data_vars.values()))
            else: var0 = next(iter(xds.variables.values()))
            print(var0)
            if lxarray: print(var0.attrs)
            
        elif mode == 'load_source':
            
            xds = loadStation_Src(station=station, region=region, ldebug=True,)
            
            print(xds)
            print(xds.attrs)
            print()

            tax = xds.time
            print(tax)
            print(tax.attrs)
            print(tax.data[100:130])
            
    #         var0 = next(iter(xds.data_vars.values()))
    #         print(var0)
    #         print(var0.attrs)
            
        elif mode == 'convert_stations':
    
            # start operation
            start = time.time()
            
            # load data        
            print("\nLoading time-varying data from source file\n")
            xds = loadStation_Src(station=station, region=region, ldebug=False)
            print(xds)
            
            # write NetCDF
            nc_filepath = osp.join(*getFolderFileName(station=xds.attrs['name'], region=xds.attrs['region'], mode='daily'))
            xds.to_netcdf(nc_filepath)
            # add timestamp
            print("\nAdding human-readable time-stamp variable ('time_stamp')\n")
            ncds = nc.Dataset(nc_filepath, mode='a')
            ncts = addTimeStamps(ncds, units='day') # add time-stamps
            ncds.close()
            # print timing
            end = time.time()
            print(('\n   Required time:   {:.0f} seconds\n'.format(end-start)))
     