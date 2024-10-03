#Realiza as importações e define os diretórios dos dados e das figuras

import geopandas as gpd
import rioxarray as rio
from shapely.geometry import mapping

def shape_clip_dataarray(dataarray, shapefile_path, projection = 'epsg:4326', x_dim = 'longitude', y_dim ='latitude', invert= False, all_touched=True):
    """Clip a DataArray using a shapefile."""
    shapefile = gpd.read_file(shapefile_path)
    dataarray = dataarray.rio.write_crs(projection)
    dataarray = dataarray.rio.set_spatial_dims(x_dim, y_dim)
    return dataarray.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop = True, invert = invert, all_touched = all_touched)



import xarray as xr
import os
import metpy as met
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
import cartopy.io.shapereader as shpreader # Import shapefiles
#import shape_mask_dataset


path_figuras = '/home/numa23/copernicus/figures/'
filelocation = '/home/numa23/copernicus/br2/'
shapefile_br = "/home/numa23/copernicus/Lim_america_do_sul_IBGE_2021" #America do sul

#Cria máscaras para cada área

########### Nordeste ###########

filepattern = 'brasil_tot_'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))

teste = np.ones([101,177])
data_xr = xr.DataArray(teste, 
coords={'longitude': ds.longitude,'latitude': ds.latitude}, 
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray(data_xr, shapefile_br, invert = True, all_touched = False)

mask_ne = ds2.T




# Carrega os arquivos
# Transforma u e v em magnitude, aplica a máscara, faz a média espacial e a média diária


#Área Nordeste

filepattern = 'brasil_tot_'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]

count = 0
for arq in arqs:
    ds = xr.open_dataset(os.path.join(filelocation,arq))
    ds_temp = (0.5*1.179*((met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))))**3).resample(time = '1D').mean(dim = 'time')*mask_ne
    ds_ne=ds
    #ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
    if count == 0:
        vento100_ne = ds_temp
        count += 1
    else:
        vento100_ne = xr.concat([vento100_ne, ds_temp],dim='time')
    # vento100_ne = vento_ne.mean(dim = ('latitude', 'longitude')).resample(time = '1D').mean(dim = 'time')

vento100_ne



vento150_mean = np.nanmean(vento100_ne,axis=0)



import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy as cart
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cmocean


#Plotting
fig=plt.figure(figsize=(6,10))
plt.rcParams.update({"font.size": 18})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
# ax.set_extent([-52, -37, -18, -30], crs=ccrs.PlateCarree())  #Bacia de santos e campos
ax.set_extent([-55, -30, 9, -35], crs=ccrs.PlateCarree())  #Bacia de santos e campos
#ax.set_extent([-41.5, -39.75, -21.25, -22.75], crs=ccrs.PlateCarree())  #Bacia de campos
#ax.set_extent([-46.25, -42.0, -22.75, -26.25], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-30,2) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-36,9,2) # define latitude ticks  #Bacia de Campos
# lontick = np.arange(-51,-45,0.5) # define longitude ticks  #Foz do amazonas
# lattick = np.arange(2,7,0.5) # define latitude ticks  #Foz do amazonas
# lontick = np.arange(-39,-31,1) # define longitude ticks    #Bacia de potiguar
# lattick = np.arange(-6,-1,1) # define latitude ticks    #Bacia de potiguar
grl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color='k',alpha=0.7,linewidth=1.2,linestyle="--")
grl.top_labels = False
grl.right_labels = False
grl.xlocator = mticker.FixedLocator(lontick)
grl.ylocator = mticker.FixedLocator(lattick)

grl.xformatter = LONGITUDE_FORMATTER
grl.yformatter = LATITUDE_FORMATTER


plt.plot([-46.25,-42.0,-42.0,-46.25,-46.25],[-26.25,-26.25,-22.75,-22.75,-26.25],markersize=3,color="red",transform = ccrs.PlateCarree()) #Santos 
plt.text(-46.4, -27.3, 'Santos', fontsize = 15,color="red")

plt.plot([-41.5,-39.75,-39.75,-41.5,-41.5],[-22.75,-22.75,-21.25,-21.25,-22.75],markersize=3,color="red",transform = ccrs.PlateCarree()) #Campos
plt.text(-39.37, -22.35, 'Campos', fontsize = 15,color="red")


plt.plot([-53,-44.25,-44.25,-53,-53],[-2.5,-2.5,7,7,-2.5],markersize=3,color="red",transform = ccrs.PlateCarree()) #Margem Norte
plt.text(-53, 7.5, 'Margem Norte', fontsize = 15,color="red")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
ax.add_feature(shape_feature2)



#Plotar com mask
cmap = cmocean.cm.speed
plt.contourf(ds_ne.longitude, ds_ne.latitude ,vento150_mean,np.arange(0,1050,50),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation = 'horizontal',fraction=0.03,pad=0.05) 
cbar.set_label('Densidade de potência do vento a 150 m \n(W m$^{-2}$)',size=17)

plt.subplots_adjust(bottom=0.1, top=0.98, hspace=0.5, right=0.99, left=0.01)

nameoffigure = "Bacias_santos_campos" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()


#Plotting
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 18})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
# ax.set_extent([-52, -37, -18, -30], crs=ccrs.PlateCarree())  #Bacia de santos e campos
ax.set_extent([-55, -30, 9, -12], crs=ccrs.PlateCarree())  #Bacia de santos e campos
#ax.set_extent([-41.5, -39.75, -21.25, -22.75], crs=ccrs.PlateCarree())  #Bacia de campos
#ax.set_extent([-46.25, -42.0, -22.75, -26.25], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-30,2) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-12,9,2) # define latitude ticks  #Bacia de Campos
# lontick = np.arange(-51,-45,0.5) # define longitude ticks  #Foz do amazonas
# lattick = np.arange(2,7,0.5) # define latitude ticks  #Foz do amazonas
# lontick = np.arange(-39,-31,1) # define longitude ticks    #Bacia de potiguar
# lattick = np.arange(-6,-1,1) # define latitude ticks    #Bacia de potiguar
grl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color='k',alpha=0.7,linewidth=1.2,linestyle="--")
grl.top_labels = False
grl.right_labels = False
grl.xlocator = mticker.FixedLocator(lontick)
grl.ylocator = mticker.FixedLocator(lattick)

grl.xformatter = LONGITUDE_FORMATTER
grl.yformatter = LATITUDE_FORMATTER




plt.plot([-53,-44.25,-44.25,-53,-53],[-2.5,-2.5,7,7,-2.5],markersize=3,color="k",transform = ccrs.PlateCarree()) #Margem Norte
plt.text(-53, 7.5, 'Margem Norte', fontsize = 25,color="k")

plt.plot([-44.29,-44.29,-37.71,-37.71,-44.29],[-5.115,0.3207,0.3207,-5.115,-5.115],markersize=3,color="red",transform = ccrs.PlateCarree()) #Regiaod02
plt.text(-41.5, 0.5, 'd02', fontsize = 18,color="red")

plt.plot([-42.1,-42.1,-39.9,-39.9,-42.1],[-3.306,-1.493,-1.493,-3.306,-3.306],markersize=3,color="red",transform = ccrs.PlateCarree()) #Regiaod03
plt.text(-41.5, -1.25, 'd03', fontsize = 18,color="red")

plt.plot([-51,-51,-31,-31,-51],[-11,6,6,-11,-11],markersize=3,color="red",transform = ccrs.PlateCarree()) #Regiaod01
plt.text(-41.5, 6.25, 'd01', fontsize = 18,color="red")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
#ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
#ax.add_feature(shape_feature2)



#Plotar com mask
cmap = cmocean.cm.speed
plt.contourf(ds_ne.longitude, ds_ne.latitude ,vento150_mean,np.arange(0,750,50),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation = 'horizontal',fraction=0.03,pad=0.05) 
cbar.set_label('Densidade de potência do vento a 150 m \n(W m$^{-2}$)',size=17)

plt.subplots_adjust(bottom=0.1, top=0.98, hspace=0.5, right=0.99, left=0.1)

nameoffigure = "Bacias_margem_norte" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()



#Plotting
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 18})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-59, -25, -8, -38], crs=ccrs.PlateCarree())  #Bacia de santos e campos
#ax.set_extent([-41.5, -39.75, -21.25, -22.75], crs=ccrs.PlateCarree())  #Bacia de campos
#ax.set_extent([-46.25, -42.0, -22.75, -26.25], crs=ccrs.PlateCarree())  #Bacia de santos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-60,-23,2) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-38,-8,2) # define latitude ticks  #Bacia de Campos
# lontick = np.arange(-51,-45,0.5) # define longitude ticks  #Foz do amazonas
# lattick = np.arange(2,7,0.5) # define latitude ticks  #Foz do amazonas
# lontick = np.arange(-39,-31,1) # define longitude ticks    #Bacia de potiguar
# lattick = np.arange(-6,-1,1) # define latitude ticks    #Bacia de potiguar
grl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color='k',alpha=0.7,linewidth=1.2,linestyle="--")
grl.top_labels = False
grl.right_labels = False
grl.xlocator = mticker.FixedLocator(lontick)
grl.ylocator = mticker.FixedLocator(lattick)

grl.xformatter = LONGITUDE_FORMATTER
grl.yformatter = LATITUDE_FORMATTER


plt.plot([-46.25,-42.0,-42.0,-46.25,-46.25],[-26.25,-26.25,-22.75,-22.75,-26.25],markersize=3,color="k",transform = ccrs.PlateCarree()) #Santos
plt.text(-46.4, -27.3, 'Santos', fontsize = 25,color="k")

plt.plot([-41.5,-39.75,-39.75,-41.5,-41.5],[-22.75,-22.75,-21.25,-21.25,-22.75],markersize=3,color="k",transform = ccrs.PlateCarree()) #Campos
plt.text(-39.37, -22.35, 'Campos', fontsize = 25,color="k")

plt.plot([-58,-58,-26,-26,-58],[-36,-9,-9,-36,-36],markersize=3,color="red",transform = ccrs.PlateCarree()) #d03
plt.text(-42, -21, 'd03', fontsize = 18,color="red")


plt.plot([-42.5,-42.5,-39.46,-39.46,-42.5],[-24.02,-21.2,-21.2,-24.02,-24.02],markersize=3,color="red",transform = ccrs.PlateCarree()) #d01
plt.text(-42, -10, 'd01', fontsize = 18,color="red")


plt.plot([-47.33,-47.33,-36.67,-36.67,-47.33],[-28.01,-18.83,-18.83,-28.01,-28.01],markersize=3,color="red",transform = ccrs.PlateCarree()) #d02
plt.text(-42, -18.5, 'd02', fontsize = 18,color="red")


fname = '/home/numa23/copernicus/shapes_Corbiniano/Blocos/Histórico_de_Blocos_Ofertados_(Modalidade,_Rodada).shp'

fname2 = '/home/numa23/copernicus/shapes_Corbiniano/Campos-zip (1)/campos_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
#ax.add_feature(shape_feature)

shape_feature2 = ShapelyFeature(Reader(fname2).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
#ax.add_feature(shape_feature2)



#Plotar com mask
cmap = cmocean.cm.speed
plt.contourf(ds_ne.longitude, ds_ne.latitude ,vento150_mean,np.arange(0,1050,50),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation = 'horizontal',fraction=0.03,pad=0.05)
cbar.set_label('Densidade de potência do vento a 150 m \n(W m$^{-2}$)',size=17)

plt.subplots_adjust(bottom=0.1, top=0.98, hspace=0.5, right=0.99, left=0.1)

nameoffigure = "Bacias_santos_campos_tds" #+ "0"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/copernicus/figures/"+string_in_string,dpi=300)
#plt.close()
