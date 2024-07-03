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


path_figuras = '/home/owner/Documents/copernicus/figures/'
filelocation = '/home/owner/Documents/copernicus/'
shapefile_br = "/home/owner/Documents/copernicus/shape_br" #Estados brasileiros


########### Nordeste ###########

filepattern = 'ts2014_2022.nc'

arqs = [f for f in os.listdir(filelocation) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))

teste = np.ones([7,5])
data_xr = xr.DataArray(teste, 
coords={'longitude': ds.longitude,'latitude': ds.latitude}, 
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray(data_xr, shapefile_br, invert = True, all_touched = False)

mask_ne = ds2.T





arqs = [f for f in os.listdir(filelocation) if f.startswith(filepattern)]

count = 0
for arq in arqs:
    ds = xr.open_dataset(os.path.join(filelocation,arq))
    ds2 = met.calc.wind_speed(ds.u100, ds.v100).resample(time = '1D').mean(dim = 'time')*mask_ne
    ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
    if count == 0:
        vento100_ne = ds_temp
        count += 1
    else:
        vento100_ne = xr.concat([vento100_ne, ds_temp],dim='time')
    # vento100_ne = vento_ne.mean(dim = ('latitude', 'longitude')).resample(time = '1D').mean(dim = 'time')

vento100_ne


# Transformando num pandas dataframe

# initialize data of lists.
data = {'NE': vento100_ne.values}
  
# Create DataFrame
vento100 = pd.DataFrame(data, vento100_ne.time, columns=['NE']).stack()

vento100


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
plt.contourf(ds.longitude, ds.latitude[1:] ,ds.u100[0,1:,:]*mask_ne,np.arange(-10,11-1),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
cbar = plt.colorbar(orientation = 'horizontal') 


# #Plotar sem mask
# cmap = cmocean.cm.thermal
# plt.contourf(ds.longitude, ds.latitude ,ds.u100[0],np.arange(-10,11-1),transform = ccrs.PlateCarree(),color='k',cmap=cmap)
# cbar = plt.colorbar(orientation = 'horizontal') 










# mean
mean_ne = vento100_ne.mean()
mean = vento100.mean()

# variance
var_ne = vento100_ne.var()
var = vento100.var()

# standard deviation
std_ne = vento100_ne.std()
std = vento100.std()

# skewness
skew = stats.skew(vento100, nan_policy = 'omit')

print(skew)


# Figura dispersão + histograma
# create the bins (x axis) for the data
mag = pd.DataFrame(vento100, columns=['mean'])

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(vento100_ne.time, vento100_ne, alpha=0.6, color="b")
# ax.set_xlabel("Tempo")
# ax.set_title("Dispersão do vento")
# ax.set_ylabel("Magnitude do vento diário médio \n(ms$^{-1}$)")

ax.set_xlabel("Time")
ax.set_title("Wind scatter")
ax.set_ylabel("Average daily wind magnitude \n(ms$^{-1}$)")
ax.grid(True)

# Format the x axis
ax.xaxis.set_major_formatter(DateFormatter("%Y"))

ax2 = fig.add_subplot(1, 2, 2, sharey = ax)
bins = np.arange(0, vento100.max(), .5)

# make the histogram
sns.histplot(data=mag, y="mean", bins=bins)

# set limits and labels
# ax2.set_ylim(bins[0], bins[-1])
# ax2.set_xlabel("Frequência")
ax2.set_ylabel('')
# ax2.set_title("Histograma")

ax2.set_xlabel("Frequency")
ax2.set_title("Histogram")

#fig.savefig(path_figuras + '\dispersao_areas+hist.png', format='png', dpi=300, bbox_inches='tight')


# add PDF
x_r100 = np.arange(0, 100, 1)

# take 1000 records of 100 samples
random_samples = np.random.normal(mean, std, size=[100, 1000])

# create placeholder for pdfs
pdfs = np.zeros([x_r100.size, 1000])

# loop through all 1000 records and create a pdf of each sample
for i in range(1000):
    # find pdfs
    pdfi = stats.norm.pdf(
        x_r100, random_samples[:, i].mean(), random_samples[:, i].std()
    )

    # add to array
    pdfs[:, i] = pdfi


fig, ax = plt.subplots()
bins = np.arange(0, vento100.max(), .5)

# make histogram
_ = sns.histplot(vento100, bins=bins, stat="density", ax=ax)

# set x limits
_ = ax.set_xlim(bins[0], bins[-1])

# get y lims for plotting mean line
ylim = ax.get_ylim()

# add vertical line with mean
_ = ax.vlines(mean, ymin=ylim[0], ymax=ylim[1], color="C3", lw=3)

# plot pdf
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), c="k", lw=3)

# plot 95th percentile
_ = ax.plot(x_r100, np.quantile(pdfs, 0.95, axis=1), "--", lw=2, color="k")

# plot 5th percentile
_ = ax.plot(x_r100, np.quantile(pdfs, 0.05, axis=1), "--", lw=2, color="k")

# set limits and labels
ax.set_xlim(bins[0], bins[-1])
# ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
# ax.set_ylabel("Frequência")
# ax.set_title("Distribuição normal da magnitude do vento")

ax.set_xlabel("Average daily wind magnitude \n(ms$^{-1}$)")
ax.set_ylabel("Frequency")
ax.set_title("Normal distribution of wind magnitude")


ax.annotate(
    f"Skew = ${skew:0.3f}$",
    xy=(0.05, 0.90),
    xycoords='axes fraction'
)

# ax.legend(['Média', 'Dist. Normal'])
ax.legend(['Mean', 'Normal Dist.'])

fig.savefig(path_figuras + '\distribuicao_media+skew+percentis.png', format='png', dpi=300, bbox_inches='tight')



dt = 2
intervalos = [ '2014-2016', '2017-2019', '2020-2022']
skew = []

fig, ax = plt.subplots()

for tt in range(2014, 2021, 3):
    #vento100.loc['2020':'2021']  #essa linha muito provavelmente nao faz nada
    exec(f'mag100_{tt}_{tt+dt} = vento100.loc[str({tt}):str({tt+dt})]')
    exec(f'mag_intervalo = vento100.loc[str({tt}):str({tt+dt})]')
    
    # Computa as variaveis estatisticas
    # mean
    mean = mag_intervalo.mean()
    # variance
    var = mag_intervalo.var()
    # standard deviation
    std = mag_intervalo.std()
    # calculate the skewness of our precip data
    skew.append(stats.skew(mag_intervalo, nan_policy = 'omit'))
    
    # plot pdf
    print(mag_intervalo.max())
    bins = np.arange(0, mag_intervalo.max(), .5)
    _ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)

# set limits and labels
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
ax.set_ylabel("Frequência")
ax.set_title("Distribuições normais das climatologias")
ax.legend(intervalos, ncol=1)


fig.savefig(path_figuras + '\distribuicao_media_clim30y_w10y.png', format='png', dpi=300, bbox_inches='tight')



intervalos = [ '2014-2016', '2017-2019', '2020-2022']

fig, ax = plt.subplots()

vento_clim = vento100.loc['2014':'2016']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)
bins_pad = bins

vento_clim = vento100.loc['2017':'2019']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)


vento_clim = vento100.loc['2020':'2022']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)





# # set limits and labels
ax.set_xlim(bins_pad[0], bins_pad[-1])
# ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
# ax.set_ylabel("Frequência")
# ax.set_title("Distribuições normais das climatologias")

ax.set_xlabel("Average daily wind magnitude \n(ms$^{-1}$)")
ax.set_ylabel("Frequency")
ax.set_title("Normal distributions of climatologies")
ax.legend(intervalos, ncol=1)


fig.savefig(path_figuras + '\distribuicao_media_clim_ultimos30y_w10y.png', format='png', dpi=300, bbox_inches='tight')




