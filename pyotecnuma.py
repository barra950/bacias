import pickle
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


with open('/home/numa23/copernicus/theta1000m.pickle', 'rb') as handle:
    theta1000m = pickle.load(handle)



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




rootgrp = Dataset('/home/numa23/copernicus/otec_brasil_2014_2023_0.5m.nc','r')

dims = rootgrp.dimensions

vars = rootgrp.variables

attrs = rootgrp.ncattrs

ndims = len(dims)

print ('number of dimensions = ' + str(ndims))

for key in dims:
    print ('dimension['+key+'] = ' +str(len(dims[key])))

gattrs = rootgrp.ncattrs()
ngattrs = len(gattrs)

print ('number of global attributes = ' + str(ngattrs))

for key in gattrs:
    print ('global attribute['+key+']=' + str(getattr(rootgrp,key)))

vars = rootgrp.variables
nvars = len(vars)
print ('number of variables = ' + str(nvars))

for var in vars:
    print ('---------------------- variable '+var+'----------------')
    print ('shape = ' + str(vars[var].shape))
    vdims = vars[var].dimensions
    for vd in vdims:
        print ('dimension['+vd+']=' + str(len(dims[vd])))




lat = vars['latitude'][:]
lon = vars['longitude'][:]
depth = vars['depth'][:]
time = vars['time'][:]
theta = vars['thetao'][:]
bottomT = vars['bottomT'][:]


# Get the time information for the file.
timeUnits = vars['time'].units

# Make dates for the file.
Date = num2date(time,timeUnits,calendar='standard')
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])


print(Date[len(Date)-1])
print(Date[0])
print(theta[0,0,0,0])

#Desenhando a bacia de Santos e Campos

#Plotting
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-52, -37, -18, -30], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-52,-36,2) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,-18,2) # define latitude ticks  #Bacia de Campos
# lontick = np.arange(-51,-45,0.5) # define longitude ticks  #Foz do amazonas
# lattick = np.arange(2,7,0.5) # define latitude ticks  #Foz do amazonas
# lontick = np.arange(-39,-31,1) # define longitude ticks    #Bacia de potiguar
# lattick = np.arange(-6,-1,1) # define latitude ticks    #Bacia de potiguar
grl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color='k',alpha=0.7,linewidth=1.2)
grl.xlabels_top = False
grl.ylabels_right = False
grl.xlocator = mticker.FixedLocator(lontick)
grl.ylocator = mticker.FixedLocator(lattick)

grl.xformatter = LONGITUDE_FORMATTER
grl.yformatter = LATITUDE_FORMATTER


plt.plot([-42.25],[-24.5],markersize=20,marker="*",label="Bloco Búzios",transform = ccrs.PlateCarree()) #Buzios

plt.plot([-40],[-22],markersize=20,marker="*",color="red",label="Bloco Albacora",transform = ccrs.PlateCarree()) #Albacora

plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos


plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos


# for k in theta[0,0,237,:]:
#     if k is numpy.ma.masked:
#         print("pozzi",k)
#     else:
#         print(k)

for k in range(0,theta1000m.shape[0]):
    for y in range(0,theta1000m.shape[1]):
        for t in range(theta1000m.shape[2]):
            for r in range(theta1000m.shape[3]):
                if theta1000m[k,y,t,r] is numpy.ma.masked:
                    theta1000m[k,y,t,r] = bottomT[k,t,r]
                    #print(k)



Pg = 30*1025*4000*3*0.85*(45/30)*(abs(theta-theta1000m)**2) /  (16*(1+(45/30))*(theta+273.15))  / 10**6  #in MW
Pgmean = np.nanmean(Pg,axis=(0))


#30*1025*4000*3*0.85*(45/30)*((20)**2) /  (16*(1+(45/30))*(25+273))  / 10**6


#Plotar diferenca de temperatura media
dT = theta-theta1000m
dTmean = np.nanmean(dT,axis=(0))
cmap = cmocean.cm.thermal
plt.contourf(lon, lat ,dTmean[0],np.arange(0,32,1),transform = ccrs.PlateCarree(),colors="none",levels=[20,50],hatches=['.'],zorder=1)
plt.contourf(lon, lat ,dTmean[0],np.arange(0,32,1),transform = ccrs.PlateCarree(),color='k',cmap=cmap,zorder=0)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Diferença de temperaura ($\rm^{o}C$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


#Plotar power gross mean
#cmap = cmocean.cm.speed
#plt.contourf(lon, lat ,Pgmean[0],np.arange(0,26,1),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
#cbar = plt.colorbar()  #pad=0.1
#cbar.set_label(r'Power gross mean (mW)',size=20)
# # print(pdmean_f[30,39],"Buzios")
# # print(pdmean_f[20,48],"Albacore")



plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
plt.legend(loc='lower right')

nameoffigure = "bacias_otec1" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()
