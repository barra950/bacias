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




rootgrp = Dataset('/home/owner/Documents/copernicus/download2022_2023_mn.nc','r')

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

olat = vars['latitude'][:]
olon = vars['longitude'][:]
#mslp = vars['msl'][:]
time = vars['time'][:]
uwnd = vars['u100'][:]
vwnd = vars['v100'][:]
ssrd = vars['ssrd'][:]



# Get the time information for the file.
timeUnits = vars['time'].units
    
# Make dates for the file.
Date = num2date(time,timeUnits,calendar='standard')
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])

#%% 

#Velocidade de vento bacia de campos

#Plotting
fig=plt.figure(figsize=(8,8))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-47, -37, -18, -28], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-47,-36,0.5) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-28,-18,0.5) # define latitude ticks  #Bacia de Campos
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



#plt.plot([-50.25,-50,-50,-50.25,-50.25],[5.5,5.5,5.25,5.25,5.5],markersize=3,transform = ccrs.PlateCarree()) #Foz do amazonas 
#plt.plot([-37.25,-37.0,-37.0,-37.25,-37,-37,-36.75,-36.75,-36.5,-36.5],[-3.75,-3.75,-4,-4,-4.25,-4.5,-4.5,-4.25,-4.25,-4.5],markersize=3,transform = ccrs.PlateCarree())   #Bacia de potiguar

plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 


plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos


wndspeed = np.sqrt(vwnd**2   +  uwnd**2)
wndmean = np.nanmean(wndspeed,axis=(0))

plt.contourf(olon, olat ,wndmean,np.arange(0,20.5,0.5),transform = ccrs.PlateCarree(),color='k',cmap='CMRmap')
cbar = plt.colorbar()
cbar.set_label(r'Velocidade média do vento a 100 m (m $\rms^{-1}$)',size=20)
#cbar.ax.tick_params(length=5, width=2,labelsize=16)

# q = ax.quiver(olon, olat, uwnd[0], vwnd[0], transform=ccrs.PlateCarree())
# ax.quiverkey(q, 1.03, 1.03, 20, label='20m/s')

#ax.streamplot(olon, olat, uwnd[0], vwnd[0], transform=ccrs.PlateCarree(),linewidth=1, density=2, color='crimson')

plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.15, right=0.99, left=0.0)
plt.show() 


#%%

#Desenhando a bacia de Santos e Campos

#Plotting
fig=plt.figure(figsize=(15,15))
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

wndspeed = np.sqrt(vwnd**2   +  uwnd**2)
pd = 0.5*1.185*wndspeed**3
pdmean = np.nanmean(pd,axis=(0))
#wndmean = np.nanmean(wndspeed,axis=(0))
ssrdmean = np.nanmean(ssrd,axis=(0))/3600


# #Plot velocidade media do vento
# plt.contourf(olon, olat ,wndmean,np.arange(0,20.5,0.5),transform = ccrs.PlateCarree(),color='k',cmap='CMRmap')
# cbar = plt.colorbar()
# cbar.set_label(r'Velocidade média do vento a 100 m (m $\rms^{-1}$)',size=20)
# #cbar.ax.tick_params(length=5, width=2,labelsize=16)


#Plot power density
cmap = cmocean.cm.speed
plt.contourf(olon, olat ,pdmean_f,np.arange(0,850,50),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar()  #pad=0.1
cbar.set_label(r'Densidade de potência do vento a 100 m (W $\rmm^{-2}$)',size=20)
print(pdmean_f[30,39],"Buzios")
print(pdmean_f[20,48],"Albacore")
#cbar.ax.tick_params(length=5, width=2,labelsize=16)

# #Plot radiacao solar
# cmap = cmocean.cm.solar
# plt.contourf(olon, olat ,ssrdmean_f,np.arange(150,275,5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
# cbar = plt.colorbar()  #pad=0.1
# cbar.set_label(r'Radiação de onda curta (W $\rmm^{-2}$)',size=20)
# print(ssrdmean_f[30,39],"Buzios")
# print(ssrdmean_f[20,48],"Albacore")
# #cbar.ax.tick_params(length=5, width=2,labelsize=16)

# q = ax.quiver(olon, olat, uwnd[0], vwnd[0], transform=ccrs.PlateCarree())
# ax.quiverkey(q, 1.03, 1.03, 20, label='20m/s')

#ax.streamplot(olon, olat, uwnd[0], vwnd[0], transform=ccrs.PlateCarree(),linewidth=1, density=2, color='crimson')

plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.99, left=0.1)
plt.legend(loc='lower right')

nameoffigure = "bacias_solar1" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/owner/Documents/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()

plt.show() 


#%%

#Desenhando as bacias 

#Plotting
fig=plt.figure(figsize=(8,8))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-55, -30, 10, -10], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-54,-30,2) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-10,10,2) # define latitude ticks  #Bacia de Campos
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



plt.plot([-50],[5.5],markersize=20,marker="*",label="Bloco FZA-M",transform = ccrs.PlateCarree()) #FZA-M

plt.plot([-37],[-4],markersize=20,marker="*",color="red",label="Pitu Oeste",transform = ccrs.PlateCarree()) #Pitu Oeste


plt.plot([-51.5182,-49.409,-48.260,-47.795,-46.801,-45.559,-47.919],[4.442,7.018,5.745,4.255,2.764,2.391,-0.621],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

plt.plot([-45.559,-44.720,-43.602,-42.547,-44.472],[2.391,2.112,1.801,1.056,-2.236],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

plt.plot([-42.547,-41.894,-41.087,-41.894],[1.056,0.683,0.559,-2.764],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

plt.plot([-41.087,-39.037,-37.267,-38.416],[0.559,0.404,-0.590,-3.758],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

plt.plot([-37.267,-35.559,-33.354,-32.422,-35.342],[-0.590,-1.801,-2.453,-3.665,-5.248],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 



# wndspeed = np.sqrt(vwnd**2   +  uwnd**2)
# pd = 0.5*1.185*wndspeed**3
# pdmean = np.nanmean(pd,axis=(0))
# #wndmean = np.nanmean(wndspeed,axis=(0))
# ssrdmean = np.nanmean(ssrd,axis=(0))/3600


# #Plot velocidade media do vento
# plt.contourf(olon, olat ,wndmean,np.arange(0,20.5,0.5),transform = ccrs.PlateCarree(),color='k',cmap='CMRmap')
# cbar = plt.colorbar()
# cbar.set_label(r'Velocidade média do vento a 100 m (m $\rms^{-1}$)',size=20)
# #cbar.ax.tick_params(length=5, width=2,labelsize=16)


# #Plot power density
# cmap = cmocean.cm.speed
# plt.contourf(olon, olat ,pdmean_f,np.arange(0,850,50),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
# cbar = plt.colorbar()  #pad=0.1
# cbar.set_label(r'Densidade de potência do vento a 100 m (W $\rmm^{-2}$)',size=20)
# print(pdmean_f[22,24],"Bloco FZA-M")
# print(pdmean_f[60,76],"Pitu Oeste")
# #cbar.ax.tick_params(length=5, width=2,labelsize=16)

#Plot radiacao solar
cmap = cmocean.cm.solar
plt.contourf(olon, olat ,ssrdmean_f,np.arange(150,275,5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar()  #pad=0.1
cbar.set_label(r'Radiação de onda curta (W $\rmm^{-2}$)',size=20)
print(ssrdmean_f[22,24],"Bloco FZA-M")
print(ssrdmean_f[60,76],"Pitu Oeste")
#cbar.ax.tick_params(length=5, width=2,labelsize=16)

# q = ax.quiver(olon, olat, uwnd[0], vwnd[0], transform=ccrs.PlateCarree())
# ax.quiverkey(q, 1.03, 1.03, 20, label='20m/s')

#ax.streamplot(olon, olat, uwnd[0], vwnd[0], transform=ccrs.PlateCarree(),linewidth=1, density=2, color='crimson')

plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.99, left=0.1)
plt.legend(loc='upper right')

nameoffigure = "bacias_solar1" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/owner/Documents/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()

plt.show() 


#%%
import pickle
import mpmath
from mpmath import *
from mpmath import mp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)




with open('/home/owner/Documents/copernicus/variables/pdmean.pickle', 'wb') as handle:
    pickle.dump(pdmean, handle, protocol=pickle.HIGHEST_PROTOCOL)


    


with open('/home/owner/Documents/copernicus/variables/ssrdmean.pickle', 'wb') as handle:
    pickle.dump(ssrdmean, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
    

#%%    
import pickle
import mpmath
from mpmath import *
from mpmath import mp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


with open('/home/owner/Documents/copernicus/variables/ssrdmean_sc_2014_2015.pickle', 'rb') as handle:
    ssrd1 = pickle.load(handle)
      
    
with open('/home/owner/Documents/copernicus/variables/ssrdmean_sc_2016_2017.pickle', 'rb') as handle:
    ssrd2 = pickle.load(handle)
    
    
with open('/home/owner/Documents/copernicus/variables/ssrdmean_sc_2018_2019.pickle', 'rb') as handle:
    ssrd3 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/ssrdmean_sc_2020_2021.pickle', 'rb') as handle:
    ssrd4 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/ssrdmean_sc_2022_2023.pickle', 'rb') as handle:
    ssrd5 = pickle.load(handle)
    

ssrdmean_f = (ssrd1 + ssrd2 + ssrd3 + ssrd4 + ssrd5)/5


####################################################################################################################################


with open('/home/owner/Documents/copernicus/variables/pdmean_sc_2014_2015.pickle', 'rb') as handle:
    pd1 = pickle.load(handle)
      
    
with open('/home/owner/Documents/copernicus/variables/pdmean_sc_2016_2017.pickle', 'rb') as handle:
    pd2 = pickle.load(handle)
    
    
with open('/home/owner/Documents/copernicus/variables/pdmean_sc_2018_2019.pickle', 'rb') as handle:
    pd3 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/pdmean_sc_2020_2021.pickle', 'rb') as handle:
    pd4 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/pdmean_sc_2022_2023.pickle', 'rb') as handle:
    pd5 = pickle.load(handle)
    

pdmean_f = (pd1 + pd2 + pd3 + pd4 + pd5)/5




#%%    
import pickle
import mpmath
from mpmath import *
from mpmath import mp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


with open('/home/owner/Documents/copernicus/variables/ssrdmean_mn_2014_2015.pickle', 'rb') as handle:
    ssrd1 = pickle.load(handle)
      
    
with open('/home/owner/Documents/copernicus/variables/ssrdmean_mn_2016_2017.pickle', 'rb') as handle:
    ssrd2 = pickle.load(handle)
    
    
with open('/home/owner/Documents/copernicus/variables/ssrdmean_mn_2018_2019.pickle', 'rb') as handle:
    ssrd3 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/ssrdmean_mn_2020_2021.pickle', 'rb') as handle:
    ssrd4 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/ssrdmean_mn_2022_2023.pickle', 'rb') as handle:
    ssrd5 = pickle.load(handle)
    

ssrdmean_f = (ssrd1 + ssrd2 + ssrd3 + ssrd4 + ssrd5)/5


####################################################################################################################################


with open('/home/owner/Documents/copernicus/variables/pdmean_mn_2014_2015.pickle', 'rb') as handle:
    pd1 = pickle.load(handle)
      
    
with open('/home/owner/Documents/copernicus/variables/pdmean_mn_2016_2017.pickle', 'rb') as handle:
    pd2 = pickle.load(handle)
    
    
with open('/home/owner/Documents/copernicus/variables/pdmean_mn_2018_2019.pickle', 'rb') as handle:
    pd3 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/pdmean_mn_2020_2021.pickle', 'rb') as handle:
    pd4 = pickle.load(handle)
    

with open('/home/owner/Documents/copernicus/variables/pdmean_mn_2022_2023.pickle', 'rb') as handle:
    pd5 = pickle.load(handle)
    

pdmean_f = (pd1 + pd2 + pd3 + pd4 + pd5)/5












