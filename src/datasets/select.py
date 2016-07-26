# coding: utf-8
import numpy as np
import codecs
folder      = '/scratch/huoyilin/data/GHCN/' # root folder for station data: interval and datatype (source folder)
stationfile = 'ghcnd-stations.txt' # file that contains station meta data (to load station records)
stationfile1 = 'ghcnd-stations_Tibet.txt'
stationfile = '{:s}/{:s}'.format(folder,stationfile)
stationfile1 = '{:s}/{:s}'.format(folder,stationfile1)
f = codecs.open(stationfile, 'r', 'UTF-8')
f1= codecs.open(stationfile1, 'w', 'UTF-8')
stationnum=0
for line in f:
      collist = line.split()
      lat=float(collist[1])
      lon=float(collist[2])
      if lat>5 and lat<49 and lon>55 and lon<135:
         f1.writelines(line)
         stationnum+=1
f.close()
f1.close()
print 'stationnum',stationnum
