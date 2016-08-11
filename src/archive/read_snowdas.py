import numpy as np
from osgeo import gdal
from pylab import *
from ftplib import FTP
import os as os
import subprocess
from matplotlib.colors import LogNorm

#import matplotlib as mpl
#mpl.rcParams['text.usetex']=True
#mpl.rcParams['text.dvipnghack'] = True 




## change working directory
processing_dir = '/Users/menounos/snowdas/tmp/'

## polygon file
poly_file = '/Users/menounos/snowdas/shapefiles/totalmica_finalboundary.shp'


os.chdir(processing_dir)   # Change current working directory


## clean out old files 
os.system('rm ' + processing_dir + '*')



gdal_config     = '--config GDAL_DATA /Library/Frameworks/GDAL.framework/Versions/1.10/Resources/gdal/ '

path = 'DATASETS/NOAA/G02158/unmasked/'

day   = '15'
month = '08_Aug'
year  = '2014'

tar_file ='SNODAS_unmasked_' + year + month[0:2] + day+ '.tar'

product = 'zz_ssmv11034tS__T0001TTNATS' + year + month[0:2] + day + '05HP001.dat'

print tar_file
print product

snowdas_path = path + '/' + year + '/' + month 


ftp_site = 'sidads.colorado.edu'

ftp = FTP(ftp_site)     # connect to host, default port
ftp.login('anonymous', 'anonymous@')

ftp.cwd(snowdas_path)
ftp.retrlines('LIST')

ftp.retrbinary("RETR " + tar_file, open(tar_file, 'wb').write)

print 'Done downloading file', tar_file, '...'


## untar file 

print processing_dir + tar_file

os.system('ls -l /Users/menounos/snowdas/tmp')

os.system('tar -xvf ' + processing_dir + tar_file)

os.system('gunzip *')

files = os.listdir(processing_dir)  

## file file of interest and then write appropriate header file 


for file in files:
    if file == product:
        print 'writing header for file', file, 'now'
        
        f_out = open(processing_dir + file[:-4] + '.hdr', 'wb')
        f_out.write('byteorder M')
        f_out.write('\n')
        f_out.write('layout bil')
        f_out.write('\n')
        f_out.write('nbands 1')
        f_out.write('\n')
        f_out.write('nbits 16')
        f_out.write('\n')
        f_out.write('ncols 8192')
        f_out.write('\n')
        f_out.write('nrows 4096')
        f_out.write('\n')
        f_out.write('ulxmap -130.517083333332')
        f_out.write('\n')
        f_out.write('ulymap 58.232916666654')
        f_out.write('\n')
        f_out.write('xdim 0.0083333333333')
        f_out.write('\n')
        f_out.write('ydim 0.0083333333333')
        f_out.write('\n')
        
        f_out.close()
        

print 'warping and clipping'


## project grid to BC Albers (EPSG:3005)
## WGS84 lat lon EPSG:4326


epsg_wgs84      = ' -s_srs EPSG:4326 '

epsg_bcalb      = ' -t_srs EPSG:3005 '

bcalb_resample  = ' -r near '

f_name     = processing_dir + product

f_name2    = processing_dir + 'SWE_unclipped.tif'


os.system('/Users/menounos/anaconda/bin/gdalwarp ' + \
          epsg_wgs84 + gdal_config + epsg_bcalb + bcalb_resample + f_name + " " + f_name2)


print 'here!'

## now do clipping 

os.system('/Users/menounos/snowdas/clip.sh')

#subprocess.call(['./Users/menounos/snowdas/clip'])

print 'here!!!'

ds = gdal.Open(processing_dir + 'swe_clip.tif', gdal.GA_ReadOnly)


band = ds.GetRasterBand(1)
print band.GetNoDataValue()
# None ## normally this would have a finite value, e.g. 1e+20
ar = band.ReadAsArray()
print np.isnan(ar).all()
# False
print '%.1f%% masked' % (np.isnan(ar).sum() * 100.0 / ar.size)
# 43.0% masked

print ar.max()
print ar.min()

ar[ar>20000] = 0 


print ar.sum()/1e6


masked_array=np.ma.masked_where(ar == 0, ar)
cmap = matplotlib.cm.jet
cmap.set_bad('w',1.)

#cmap = plt.get_cmap('jet', 5)

plt.figure()
#plt.rc( 'text', usetex=True )


plt.contourf(np.flipud(masked_array/1000.0), cmap=cmap, vmin=0.1, vmax=4)
plt.title(r'Snow Water Equivalent (m): ' + year + '/' + month[0:2] + '/' + day + '\n' +
          'Total Volume (km$^{3}$): ' + `round(ar.sum()/1e6, 2)`)
plt.colorbar()

plt.savefig('/Users/menounos/snowdas/Snowdas_' + year + '_' + month[0:2] + '_' + day + '.jpg', dpi=600)

plt.show()

