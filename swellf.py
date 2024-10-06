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


rootgrp = Dataset('/home/numa23/copernicus/ondas_brasil_2014_2023.nc','r')

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
Hm0 = vars['VHM0'][:]
time = vars['time'][:]
T10 = vars['VTM10'][:]


# Get the time information for the file.
timeUnits = vars['time'].units

# Make dates for the file.
Date = num2date(time,timeUnits,calendar='standard')
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])



rootgrp = Dataset('/home/numa23/copernicus/bathy.nc','r')

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

latb = vars['latitude'][:]
lonb = vars['longitude'][:]
depthb = vars['deptho'][:]




print(Date[len(Date)-1])
print(Date[0])
#print(theta[0,0,0,0])

#Desenhando a bacia de Santos e Campos

#Plotting
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-52, -37, -18, -30], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
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


#plt.plot([-42.25],[-24.5],markersize=20,marker="*",label="Bloco Búzios",transform = ccrs.PlateCarree()) #Buzios

#plt.plot([-40],[-22],markersize=20,marker="*",color="red",label="Bloco Albacora",transform = ccrs.PlateCarree()) #Albacora

#plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos


#plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos

plt.plot([-46.25,-42.0,-42.0,-46.25,-46.25],[-26.25,-26.25,-22.75,-22.75,-26.25],markersize=3,color="k",transform = ccrs.PlateCarree()) #Santos
plt.text(-46.4, -27.3, 'Santos', fontsize = 25,color="k")

plt.plot([-41.5,-39.75,-39.75,-41.5,-41.5],[-22.75,-22.75,-21.25,-21.25,-22.75],markersize=3,color="k",transform = ccrs.PlateCarree()) #Campos
plt.text(-41.6, -23.8, 'Campos', fontsize = 25,color="k")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
#shape_feature = ShapelyFeature(Reader(fname).geometries(),
#                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
#ax.add_feature(shape_feature)

#shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
#                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
#ax.add_feature(shape_feature2)


P = 1025*(9.806**2)*T10*(Hm0**2)/(64*np.pi)
Pmean = np.nanmean(P,axis=0)
print(np.nanmin(Pmean),np.nanmax(Pmean))

#Plotando ondas
#dTmean = np.nanmean(dT,axis=0)
cmap = cmocean.cm.thermal
manual_locations = [(-48, -26),(-47.5, -26.5),(-47,-27),(-47,-27.9),(-46.6,-27.2)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=10,manual=manual_locations)
plt.contourf(lon, lat ,Pmean/1000,np.arange(0,40,5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Fluxo de energia (kW $\rmm^{-1}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")



plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_otec_ondas1" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()



#Plotting campos
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-41.5, -39.75, -21.25, -22.75], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


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


#plt.plot([-42.25],[-24.5],markersize=20,marker="*",label="Bloco Búzios",transform = ccrs.PlateCarree()) #Buzios

#plt.plot([-40],[-22],markersize=20,marker="*",color="red",label="Bloco Albacora",transform = ccrs.PlateCarree()) #Albacora

#plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos


#plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos

#plt.plot([-46.25,-42.0,-42.0,-46.25,-46.25],[-26.25,-26.25,-22.75,-22.75,-26.25],markersize=3,color="k",transform = ccrs.PlateCarree()) #Santos
#plt.text(-46.4, -27.3, 'Santos', fontsize = 25,color="k")

#plt.plot([-41.5,-39.75,-39.75,-41.5,-41.5],[-22.75,-22.75,-21.25,-21.25,-22.75],markersize=3,color="k",transform = ccrs.PlateCarree()) #Campos
#plt.text(-41.6, -23.8, 'Campos', fontsize = 25,color="k")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature2)


P = 1025*(9.806**2)*T10*(Hm0**2)/(64*np.pi)
Pmean = np.nanmean(P,axis=0)
print(np.nanmin(Pmean),np.nanmax(Pmean))

#Plotando ondas
#dTmean = np.nanmean(dT,axis=0)
cmap = cmocean.cm.thermal
manual_locations = [(-41, -22.5),(-40.8, -22.5),(-40.5,-22.5),(-40.4,-22.5),(-40.1,-22.5)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=20,manual=manual_locations)
plt.contourf(lon, lat ,Pmean/1000,np.arange(0,40,5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Fluxo de energia (kW $\rmm^{-1}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")



plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_otec_ondas_campos" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()


#Plotting santos
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-46.25, -42.0, -26.25, -22.75], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


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


#plt.plot([-42.25],[-24.5],markersize=20,marker="*",label="Bloco Búzios",transform = ccrs.PlateCarree()) #Buzios

#plt.plot([-40],[-22],markersize=20,marker="*",color="red",label="Bloco Albacora",transform = ccrs.PlateCarree()) #Albacora

#plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos


#plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos

#plt.plot([-46.25,-42.0,-42.0,-46.25,-46.25],[-26.25,-26.25,-22.75,-22.75,-26.25],markersize=3,color="k",transform = ccrs.PlateCarree()) #Santos
#plt.text(-46.4, -27.3, 'Santos', fontsize = 25,color="k")

#plt.plot([-41.5,-39.75,-39.75,-41.5,-41.5],[-22.75,-22.75,-21.25,-21.25,-22.75],markersize=3,color="k",transform = ccrs.PlateCarree()) #Campos
#plt.text(-41.6, -23.8, 'Campos', fontsize = 25,color="k")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature2)


P = 1025*(9.806**2)*T10*(Hm0**2)/(64*np.pi)
Pmean = np.nanmean(P,axis=0)
print(np.nanmin(Pmean),np.nanmax(Pmean))

#Plotando ondas
#dTmean = np.nanmean(dT,axis=0)
cmap = cmocean.cm.thermal
manual_locations = [(-43.5, -23),(-43.5, -23.6),(-43.5,-24.1),(-43.77,-24.4),(-43.5,-24.5)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=20,manual=manual_locations)
plt.contourf(lon, lat ,Pmean/1000,np.arange(0,40,5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Fluxo de energia (kW $\rmm^{-1}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")



plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_otec_ondas_santos" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()



#Plotting para margen norte
fig=plt.figure(figsize=(9,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-54, -42, 9, -4], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')
#ax.add_feature(cart.feature.OCEAN, zorder=0, edgecolor='k')

# # add grid ticks
lontick = np.arange(-54,-42,2) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-4,9,2) # define latitude ticks  #Bacia de Campos
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

#plt.plot([-50],[5.5],markersize=20,marker="*",label="Bloco FZA-M",transform = ccrs.PlateCarree()) #FZA-M

#plt.plot([-37],[-4],markersize=20,marker="*",color="red",label="Pitu Oeste",transform = ccrs.PlateCarree()) #Pitu Oeste


#plt.plot([-51.5182,-49.409,-48.260,-47.795,-46.801,-45.559,-47.919],[4.442,7.018,5.745,4.255,2.764,2.391,-0.621],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

#plt.plot([-45.559,-44.720,-43.602,-42.547,-44.472],[2.391,2.112,1.801,1.056,-2.236],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

#plt.plot([-42.547,-41.894,-41.087,-41.894],[1.056,0.683,0.559,-2.764],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

#plt.plot([-41.087,-39.037,-37.267,-38.416],[0.559,0.404,-0.590,-3.758],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

#plt.plot([-37.267,-35.559,-33.354,-32.422,-35.342],[-0.590,-1.801,-2.453,-3.665,-5.248],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 

plt.plot([-53,-44.25,-44.25,-53,-53],[-2.5,-2.5,7,7,-2.5],markersize=3,color="k",transform = ccrs.PlateCarree()) #Margem Norte
plt.text(-53, 7.5, 'Margem Norte', fontsize = 25,color="k")


#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature2)



P = 1025*(9.806**2)*T10*(Hm0**2)/(64*np.pi)
Pmean = np.nanmean(P,axis=0)
print(np.nanmin(Pmean),np.nanmax(Pmean))

#Plotando ondas
#dTmean = np.nanmean(dT,axis=0)
cmap = cmocean.cm.thermal
manual_locations = [(-48, -26),(-47.5, -26.5),(-47,-27),(-47,-27.9),(-46.6,-27.2)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=10,manual=manual_locations)
plt.contourf(lon, lat ,Pmean/1000,np.arange(0,40,5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal",pad=0.1)  #pad=0.1
cbar.set_label(r'Fluxo de energia (kW $\rmm^{-1}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")



plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.02, top=0.95, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='upper right')

nameoffigure = "bacias_otec_ondas2" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()

