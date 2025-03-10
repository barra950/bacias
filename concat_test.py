import sys,warnings,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset,num2date
#from mpl_toolkits.basemap import Basemap,cm
from datetime import datetime
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
from scipy import stats
from cartopy import config
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy as cart
import cmocean
import numpy.ma
import cartopy.io.shapereader as shpreader # Import shapefiles
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


counter = 0
for k in range(1940,2024):
    rootgrp = Dataset(f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br/santos_{k}.nc','r')

    dims = rootgrp.dimensions

    vars = rootgrp.variables
    
    attrs = rootgrp.ncattrs

    ndims = len(dims)

    print ('number of dimensions = ' + str(ndims))
    
    wndspeed = np.sqrt((vars['u100'][:])**2   +  (vars['v100'][:])**2)
    wndspeed = wndspeed * (np.log(150/vars['fsr'][:]) / np.log(100/vars['fsr'][:]))
    time=vars['time'][:]

    if counter == 0:
    
        arr1w = wndspeed
        arr1t = time
    
    else:
        
        arr2w = np.concatenate((arr1w, wndspeed), axis=0)
        
        arr1w = arr2w

        arr2t = np.concatenate((arr1t, time), axis=0)

        arr1t = arr2t
        
    counter =counter +1

print(arr1t.shape,arr1w.shape)

# Get the time information for the file.
timeUnits = vars['time'].units

# Make dates for the file.
Date = num2date(arr1t,timeUnits,calendar='standard')
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])


print(Date,Date.shape)

lon=vars['longitude'][:]
lat=vars['latitude'][:]

maxwnd_avg = np.nanmean(arr1w,axis=(0))
print(lat.shape)

print(maxwnd_avg[np.where(maxwnd_avg==np.nanmax(maxwnd_avg))[0],np.where(maxwnd_avg==np.nanmax(maxwnd_avg))[1]],"Max average wind")
print(lat[np.where(maxwnd_avg==np.nanmax(maxwnd_avg))[0]],"Latitude of max average wind")
print(lon[np.where(maxwnd_avg==np.nanmax(maxwnd_avg))[1]],"Longitude of max average wind")

print(lat)
print(lon)
