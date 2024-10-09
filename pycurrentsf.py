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



rootgrp = Dataset('/home/numa23/copernicus/correntes_brasil_2014_2023_50m.nc','r')

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
uo = vars['uo'][:]
vo = vars['vo'][:]



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
print(depth)
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



cspeed = np.sqrt(vo**2   +  uo**2)
Pc = 0.5*1025*cspeed**3
Pcmean = np.nanmean(Pc,axis=(0))

print(np.nanmin(Pcmean),np.nanmax(Pcmean))

#Plotar potencia da corrent
cmap = cmocean.cm.thermal
manual_locations = [(-48, -26),(-47.5, -26.5),(-47,-27),(-47,-27.9),(-46.6,-27.2)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=10,manual=manual_locations)
plt.contourf(lon, lat ,Pcmean[0],np.arange(0,160,10),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Densidade de potência da corrente a 50m (W m$\rm^{-2}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_correntes_50m_1" #+ "0"
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



cspeed = np.sqrt(vo**2   +  uo**2)
Pc = 0.5*1025*cspeed**3
Pcmean = np.nanmean(Pc,axis=(0))

print(np.nanmin(Pcmean),np.nanmax(Pcmean))

#Plotar potencia da corrent
cmap = cmocean.cm.thermal
manual_locations = [(-41, -22.5),(-40.8, -22.5),(-40.5,-22.5),(-40.4,-22.5),(-40.1,-22.5)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=20,manual=manual_locations)
plt.contourf(lon, lat ,Pcmean[0],np.arange(0,160,10),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Densidade de potência da corrente a 50m (W m$\rm^{-2}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_correntes_50m_campos" #+ "0"
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



cspeed = np.sqrt(vo**2   +  uo**2)
Pc = 0.5*1025*cspeed**3
Pcmean = np.nanmean(Pc,axis=(0))

print(np.nanmin(Pcmean),np.nanmax(Pcmean))

#Plotar potencia da corrent
cmap = cmocean.cm.thermal
manual_locations = [(-43.5, -23),(-43.5, -23.6),(-43.5,-24.1),(-43.77,-24.4),(-43.5,-24.5)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=20,manual=manual_locations)
plt.contourf(lon, lat ,Pcmean[0],np.arange(0,160,10),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Densidade de potência da corrente a 50m (W m$\rm^{-2}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_correntes_50m_santos" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()



#Plotting para margem norte
fig=plt.figure(figsize=(9,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-54, -42, 9, -4], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


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


#plt.plot([-42.25],[-24.5],markersize=20,marker="*",label="Bloco Búzios",transform = ccrs.PlateCarree()) #Buzios

#plt.plot([-40],[-22],markersize=20,marker="*",color="red",label="Bloco Albacora",transform = ccrs.PlateCarree()) #Albacora

#plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos


#plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos

plt.plot([-53,-44.25,-44.25,-53,-53],[-2.5,-2.5,7,7,-2.5],markersize=3,color="k",transform = ccrs.PlateCarree()) #Margem Norte
plt.text(-53, 7.5, 'Margem Norte', fontsize = 25,color="k")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature2)



cspeed = np.sqrt(vo**2   +  uo**2)
Pc = 0.5*1025*cspeed**3
Pcmean = np.nanmean(Pc,axis=(0))

print(np.nanmin(Pcmean),np.nanmax(Pcmean))

#Plotar potencia da corrent
cmap = cmocean.cm.thermal
manual_locations = [(-48, -26),(-47.5, -26.5),(-47,-27),(-47,-27.9),(-46.6,-27.2)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=10,manual=manual_locations)
plt.contourf(lon, lat ,Pcmean[0],np.arange(0,1300,100),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Densidade de potência da corrente a 50m (W m$\rm^{-2}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_correntes_50m_2" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()





#Plotting hotspot1
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-50, -47.0, 2, 5], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


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



cspeed = np.sqrt(vo**2   +  uo**2)
Pc = 0.5*1025*cspeed**3
Pcmean = np.nanmean(Pc,axis=(0))

print(np.nanmin(Pcmean),np.nanmax(Pcmean))

#Plotar potencia da corrent
cmap = cmocean.cm.thermal
manual_locations = [(-43.5, -23),(-43.5, -23.6),(-43.5,-24.1),(-43.77,-24.4),(-43.5,-24.5)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=20,manual=manual_locations)
plt.contourf(lon, lat ,Pcmean[0],np.arange(0,1300,100),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Densidade de potência da corrente a 50m (W m$\rm^{-2}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_correntes_50m_hotspot1" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()




#Plotting hotspot2
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-47, -43.0, -1, 3], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


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



cspeed = np.sqrt(vo**2   +  uo**2)
Pc = 0.5*1025*cspeed**3
Pcmean = np.nanmean(Pc,axis=(0))

print(np.nanmin(Pcmean),np.nanmax(Pcmean))

#Plotar potencia da corrent
cmap = cmocean.cm.thermal
manual_locations = [(-43.5, -23),(-43.5, -23.6),(-43.5,-24.1),(-43.77,-24.4),(-43.5,-24.5)]
cp = plt.contour(lonb, latb ,depthb,[50, 100, 200, 500, 1000],transform = ccrs.PlateCarree(),colors='k')
plt.clabel(cp,inline=True, fontsize=20,manual=manual_locations)
plt.contourf(lon, lat ,Pcmean[0],np.arange(0,1300,100),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation="horizontal")  #pad=0.1
cbar.set_label(r'Densidade de potência da corrente a 50m (W m$\rm^{-2}$)',size=20)
## print(pdmean_f[30,39],"Buzios")
## print(pdmean_f[20,48],"Albacore")


plt.tick_params('both', length=15, width=2, which='major')


plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, right=0.95, left=0.05)
#plt.legend(loc='lower right')

nameoffigure = "bacias_correntes_50m_hotspot2" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()
