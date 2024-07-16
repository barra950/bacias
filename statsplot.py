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


path_figuras = '/home/owner/Documents/copernicus/figures/figures2/'
filelocation = '/home/owner/Documents/copernicus/'
shapefile_br = "/home/owner/Documents/copernicus/Lim_america_do_sul_IBGE_2021" #Estados brasileiros

#Cria máscaras para cada área

########### Nordeste ###########

filepattern = 'mn'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))

teste = np.ones([36,39])
data_xr = xr.DataArray(teste, 
coords={'longitude': ds.longitude,'latitude': ds.latitude}, 
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray(data_xr, shapefile_br, invert = True, all_touched = False)

mask_ne = ds2.T



########### Sudeste ###########

filepattern = 'santos20'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))
ds

teste = np.ones([18,15])
data_xr = xr.DataArray(teste, 
coords={'longitude': ds.longitude,'latitude': ds.latitude}, 
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray(data_xr, shapefile_br, invert = True, all_touched = False)

mask_se = ds2.T



########### Sul ###########

filepattern = 'campos20'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))
ds

teste = np.ones([8,7])
data_xr = xr.DataArray(teste, 
coords={'longitude': ds.longitude,'latitude': ds.latitude}, 
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray(data_xr, shapefile_br, invert = True, all_touched = False)

mask_s = ds2.T


# Carrega os arquivos
# Transforma u e v em magnitude, aplica a máscara, faz a média espacial e a média diária


#Área Nordeste

filepattern = 'mn'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]

count = 0
for arq in arqs:
    ds = xr.open_dataset(os.path.join(filelocation,arq))
    ds_temp = (met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))).resample(time = '1D').mean(dim = 'time')*mask_ne
    ds_ne=ds
    #ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
    if count == 0:
        vento100_ne = ds_temp
        count += 1
    else:
        vento100_ne = xr.concat([vento100_ne, ds_temp],dim='time')
    # vento100_ne = vento_ne.mean(dim = ('latitude', 'longitude')).resample(time = '1D').mean(dim = 'time')

vento100_ne


#Área Sudeste

filepattern = 'santos20'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]

count = 0
for arq in arqs:
    ds = xr.open_dataset(os.path.join(filelocation,arq))
    ds_temp = (met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))).resample(time = '1D').mean(dim = 'time')*mask_se
    ds_se=ds
    #ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
    if count == 0:
        vento100_se = ds_temp
        count += 1
    else:
        vento100_se = xr.concat([vento100_se, ds_temp],dim='time')

vento100_se

#Área Sul

filepattern = 'campos20'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]

count = 0
for arq in arqs:
    ds = xr.open_dataset(os.path.join(filelocation,arq))
    ds_temp = (met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))).resample(time = '1D').mean(dim = 'time')*mask_s
    ds_s=ds
    #ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
    if count == 0:
        vento100_s = ds_temp
        count += 1
    else:
        vento100_s = xr.concat([vento100_s, ds_temp],dim='time')

vento100_s

vento150_ne_mean = np.nanmean(vento100_ne,axis=0)
vento150_se_mean = np.nanmean(vento100_se,axis=0)
vento150_s_mean = np.nanmean(vento100_s,axis=0)

#%%

import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy as cart
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cmocean


#Plotting
fig=plt.figure(figsize=(10,10))
plt.rcParams.update({"font.size": 18})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-52, -37, -18, -30], crs=ccrs.PlateCarree())  #Bacia de santos e campos
#ax.set_extent([-41.5, -39.75, -21.25, -22.75], crs=ccrs.PlateCarree())  #Bacia de campos
#ax.set_extent([-46.25, -42.0, -22.75, -26.25], crs=ccrs.PlateCarree())  #Bacia de santos
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


# plt.plot([-42.25],[-24.5],markersize=20,marker="*",label="Bloco Búzios",transform = ccrs.PlateCarree()) #Buzios 

# plt.plot([-40],[-22],markersize=20,marker="*",color="red",label="Bloco Albacora",transform = ccrs.PlateCarree()) #Albacora

# plt.plot([-48.56,-44.62,-44.57,-44.26,-42.72,-42.43,-40.22,-42.001],[-27.85,-27.85,-27.34,-27.19,-26.43,-26.33,-25.91,-23.006],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de santos 


# plt.plot([-40.22,-38.807,-37.445,-37.109,-40.319],[-25.91,-24.672,-22.638,-22.033,-20.453],markersize=3,transform = ccrs.PlateCarree(),color='red') #Bacia de campos



fname = '/home/owner/Documents/copernicus/bacias_gishub_db.shp'

#ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
ax.add_feature(shape_feature)


#Plotar com mask
cmap = cmocean.cm.thermal
plt.contourf(ds_s.longitude, ds_s.latitude ,vento150_s_mean,np.arange(0,10.5,0.5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
plt.contourf(ds_se.longitude, ds_se.latitude[1:] ,vento150_se_mean,np.arange(0,10.5,0.5),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation = 'horizontal') 




# #Plotar sem mask
# cmap = cmocean.cm.thermal
# plt.contourf(ds.longitude, ds.latitude ,ds.u100[0],np.arange(-10,11-1),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
# cbar = plt.colorbar(orientation = 'horizontal') 
