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
    ds2 = (met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))).resample(time = '1D').mean(dim = 'time')*mask_ne
    ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
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
    ds2 = (met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))).resample(time = '1D').mean(dim = 'time')*mask_se
    ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
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
    ds2 = (met.calc.wind_speed(ds.u100, ds.v100)*(np.log(150/ds.fsr)/np.log(100/ds.fsr))).resample(time = '1D').mean(dim = 'time')*mask_s
    ds_temp = ds2.mean(dim = ('latitude', 'longitude'))
    if count == 0:
        vento100_s = ds_temp
        count += 1
    else:
        vento100_s = xr.concat([vento100_s, ds_temp],dim='time')

vento100_s

#Faz a agregação das 3 séries

# Transformando num pandas dataframe

# initialize data of lists.
data = {'NE': vento100_ne.values,
        'SE':vento100_se.values,
        'S':vento100_s.values}
  
# Create DataFrame
vento100 = pd.DataFrame(data, vento100_ne.time, columns=['NE', 'SE', 'S']).stack()

vento100

#Computa as variáveis estatísticas

# mean
mean_ne = vento100_ne.mean()
mean_se = vento100_se.mean()
mean_s = vento100_s.mean()
mean = vento100.mean()

# variance
var_ne = vento100_ne.var()
var_se = vento100_se.var()
var_s = vento100_s.var()
var = vento100.var()

# standard deviation
std_ne = vento100_ne.std()
std_se = vento100_se.std()
std_s = vento100_s.std()
std = vento100.std()

# skewness
skew = stats.skew(vento100, nan_policy = 'omit')

print(skew)

#Figuras dispersão + histogramas de todas as áreas juntas

# Figura dispersão + histograma
# create the bins (x axis) for the data
mag = pd.DataFrame(vento100, columns=['mean'])

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(vento100_ne.time, vento100_ne, alpha=0.6, color="b")
ax.scatter(vento100_se.time, vento100_se, alpha=0.6, color="b")
ax.scatter(vento100_s.time, vento100_s, alpha=0.6, color="b")
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

fig.savefig(path_figuras + '\dispersao_areas+hist.png', format='png', dpi=300, bbox_inches='tight')

#Figura histograma com linha da média e distribuição normal + incerteza

# add PDF
x_r100 = np.arange(0, 100, 1)

# Todas as áreas juntas

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


#Todas as áreas juntas e cada área separadamente

intervalos = ['Todas as áreas', 'Nordeste', 'Sudeste', 'Sul']

fig, ax = plt.subplots()

# Computa as variaveis estatisticas
# mean
mean = vento100.mean()
# variance
var = vento100.var()
# standard deviation
std = vento100.std()
    
# plot pdf
bins = np.arange(0, vento100.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2, color = 'k')
bins_pad = bins


# mean
mean = vento100_ne.mean()
# variance
var = vento100_ne.var()
# standard deviation
std = vento100_ne.std()
# plot pdf
bins = np.arange(0, vento100_ne.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)


# mean
mean = vento100_se.mean()
# variance
var = vento100_se.var()
# standard deviation
std = vento100_se.std()
# plot pdf
bins = np.arange(0, vento100_se.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)



# mean
mean = vento100_s.mean()
# variance
var = vento100_s.var()
# standard deviation
std = vento100_s.std()
# plot pdf
bins = np.arange(0, vento100_s.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)



# set limits and labels
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
ax.set_ylabel("Frequência")
ax.set_title("Distribuições Normais")
ax.legend(intervalos, ncol=1)


fig.savefig(path_figuras + '\distribuicoes_normais_areas.png', format='png', dpi=300, bbox_inches='tight')


#Figura com a distribuição de densidade média da climatologia - janela de 10 anos

#Todas as áreas

dt = 2
intervalos = ['2014-2015', '2016-2017','2018-2019', '2020-2021','2022-2023']
skew = []

fig, ax = plt.subplots()

for tt in range(2014, 2024, 2):
    #vento100.loc['1940':'1941']
    exec(f'mag100_{tt}_{tt+dt} = vento100.loc[str({tt}):str({tt+dt})]')
    exec(f'mag_intervalo = vento100.loc[str({tt}):str({tt+dt})]')
    
    # Computa as variaveis estatisticas
    # mean
    mean = mag_intervalo.mean()
    print(mean)
    # variance
    var = mag_intervalo.var()
    # standard deviation
    std = mag_intervalo.std()
    # calculate the skewness of our precip data
    skew.append(stats.skew(mag_intervalo, nan_policy = 'omit'))
    
    # plot pdf
    bins = np.arange(0, mag_intervalo.max(), .5)
    _ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)

# set limits and labels
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
ax.set_ylabel("Frequência")
ax.set_title("Distribuições normais das climatologias")
ax.legend(intervalos, ncol=1)


fig.savefig(path_figuras + '\distribuicao_media_clim30y_w10y.png', format='png', dpi=300, bbox_inches='tight')

#%%

intervalos = ['2014-2018', '2019-2023']

fig, ax = plt.subplots()

vento_clim = vento100.loc['2014':'2019'][:-3]
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

vento_clim = vento100.loc['2018':'2024']
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

#%%

#Figura com a distribuição de densidade média dos últimos 30 anos e janelas de 10 anos
#Todas as áreas


intervalos = ['2014-2015', '2016-2017','2018-2019', '2020-2021','2022-2023']

fig, ax = plt.subplots()

vento_clim = vento100.loc['2014':'2016'][:-3]
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2, color = 'k')
bins_pad = bins

vento_clim = vento100.loc['2016':'2018'][:-3]
# mean
mean = vento_clim.mean()
print(mean,'mean')
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)


vento_clim = vento100.loc['2018':'2020'][:-3]
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)


vento_clim = vento100.loc['2020':'2022'][:-3]
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2)


vento_clim = vento100.loc['2022':'2024']
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


fig.savefig(path_figuras + '\distribuicao_media_clim_1993-2022_w10y.png', format='png', dpi=300, bbox_inches='tight')

#%%

#Figura da diferença dos últimos 30 anos em relação à média dos últimos 30 anos.
#Todas as áreas

mag_ultimos30 = vento100.loc['2014':'2024']

diff_mag_ultimos30 = []
for tt in range(2014, 2024):
    exec(f'diff_mag_ultimos30.append(mag_ultimos30.loc[str({tt})].mean() - mag_ultimos30.mean())')

time_30y = np.linspace(2014, 2023, 10)

fig = plt.figure(figsize=(12, 6)) 
ax = fig.add_subplot(111) 
_ = ax.plot(time_30y, diff_mag_ultimos30, color = 'k', ls = '-', marker = '*')
_ = ax.axhline(0, ls = '--', color = 'k')
_ = plt.xticks(time_30y, rotation = 45)
_ = ax.legend(['Diferença'])

# Add gridlines
ax.grid(True)

# Add titles
ax.set_title('Diferença de cada ano em relação a média')
ax.set_ylabel('Magnitude do vento diário médio \n(ms$^{-1}$)')


fig.savefig(path_figuras + '\distancia_media_clim30y.png', format='png', dpi=300, bbox_inches='tight')


#%%

#Figura da diferença das distribuições de densidade anual dos últimos 30 anos em relação aos últimos 30 anos.
#Todas as áreas

# intervalos = ['1993', '1994', '1995', '2013-2022']
intervalos = list(range(2014,2024,1))

vento_clim = vento100.loc['2014':'2024']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
clim_ultimos30 = stats.norm.pdf(x_r100, mean, std)

fig, ax = plt.subplots()

for tt in range(2014, 2024, 1):
    exec(f'mag_intervalo = vento100.loc[str({tt})]')
    
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
    bins = np.arange(0, mag_intervalo.max(), .5)
    _ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std) - clim_ultimos30, lw=2)

# set limits and labels
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
ax.set_ylabel("Frequência")
ax.set_title("Distribuições normais das climatologias")
ax.legend(intervalos, ncol=5)


fig.savefig(path_figuras + '\distribuicoes_anuais_ultimos30y.png', format='png', dpi=300, bbox_inches='tight')


#%%

#Série temporal da amplitude (valor máximo - valor mínimo) da diferença das distribuições dos últimos 30 anos e de cada ano nesse intervalo

vento_clim = vento100.loc['2014':'2024']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
clim_ultimos30 = stats.norm.pdf(x_r100, mean, std)

serie_amp = []
for tt in range(2014, 2024, 1):
    exec(f'mag_intervalo = vento100.loc[str({tt})]')
    
    # Computa as variaveis estatisticas
    # mean
    mean = mag_intervalo.mean()
    # variance
    var = mag_intervalo.var()
    # standard deviation
    std = mag_intervalo.std()
    
    # plot pdf
    bins = np.arange(0, mag_intervalo.max(), .5)
    dif = stats.norm.pdf(x_r100, mean, std) - clim_ultimos30
    serie_amp.append(max(dif) - min(dif)) # gera a amplitude (max - min)
    

time_30y = np.linspace(2014, 2023, 10)

fig = plt.figure(figsize=(12, 6)) 
ax = fig.add_subplot(111) 
_ = ax.plot(time_30y, serie_amp, color = 'k', ls = '-', marker = '*')
_ = ax.plot(time_30y[serie_amp.index(min(serie_amp))], min(serie_amp), color = 'r', ls = '-', marker = 'o')
# _ = ax.axhline(0, ls = '--', color = 'k')
_ = plt.xticks(time_30y, rotation = 45)
_ = ax.legend(['Diferença','Valor mínimo'])

# Add gridlines
ax.grid(True)

# Add titles
ax.set_title('Série da amplitude do bias')
ax.set_ylabel('Magnitude do vento diário médio \n(ms$^{-1}$)')


fig.savefig(path_figuras + '\serie_amplitude_bias_30y.png', format='png', dpi=300, bbox_inches='tight')

#%%


teste = dict(zip([int(x) for x in time_30y],serie_amp))
print(teste[2014])
print(teste[2015])
print(teste[2015]-teste[2014])

#Mesma análise anterior apenas para os anos de 2011 e 2002

intervalos = ['2014', '2023']
# intervalos = list(range(1993,2022,1))

vento_clim = vento100.loc['2014':'2024']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
clim_ultimos30 = stats.norm.pdf(x_r100, mean, std)


fig, ax = plt.subplots()

vento_clim = vento100.loc['2014']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std) - clim_ultimos30, lw=2)


vento_clim = vento100.loc['2023']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std) - clim_ultimos30, lw=2)

# set limits and labels
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Magnitude do vento diário médio \n(ms$^{-1}$)")
ax.set_ylabel("Frequência")
ax.set_title("Distribuições normais das climatologias")
ax.legend(intervalos, ncol=1)


# fig.savefig(path_figuras + '\distribuicao_media_clim30y_w10y.png', format='png', dpi=300, bbox_inches='tight')


#%%

#Série temporal da média do bias absoluto da diferença das distribuições dos últimos 30 anos e de cada ano nesse intervalo

vento_clim = vento100.loc['2014':'2024']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
clim_ultimos30 = stats.norm.pdf(x_r100, mean, std)

serie_abs = []
for tt in range(2014, 2024, 1):
    exec(f'mag_intervalo = vento100.loc[str({tt})]')
    
    # Computa as variaveis estatisticas
    # mean
    mean = mag_intervalo.mean()
    # variance
    var = mag_intervalo.var()
    # standard deviation
    std = mag_intervalo.std()
    
    # plot pdf
    bins = np.arange(0, mag_intervalo.max(), .5)
    dif_abs = abs(stats.norm.pdf(x_r100, mean, std) - clim_ultimos30)
    serie_abs.append(np.mean(dif_abs)) # gera a amplitude (max - min)
    

time_30y = np.linspace(2014, 2023, 10)

fig = plt.figure(figsize=(12, 6)) 
ax = fig.add_subplot(111) 
_ = ax.plot(time_30y, serie_abs, color = 'k', ls = '-', marker = '*')
# _ = ax.plot(time_30y[serie_abs.index(min(serie_abs))], min(serie_abs), color = 'r', ls = '-', marker = 'o')
# _ = ax.axhline(0, ls = '--', color = 'k')
_ = plt.xticks(time_30y, rotation = 45)
# _ = ax.legend(['Diferença','Valor mínimo'])
_ = ax.legend(['Difference'])

# Add gridlines
ax.grid(True)

# Add titles
# ax.set_title('Série da média do bias absoluto')
# ax.set_ylabel('Magnitude do vento diário médio \n(ms$^{-1}$)')

ax.set_title('Absolute bias mean time series')
# ax.set_ylabel('Average daily wind magnitude \n(ms$^{-1}$)')
ax.set_ylabel('Absolute bias frequency')


fig.savefig(path_figuras + '\serie_media_bias_absoluto_30y.png', format='png', dpi=300, bbox_inches='tight')

#%%

# np.mean(serie_abs)
serie_abs[-1]

#Figura com a distribuição de densidade média dos últimos 30 e para os anos de 2002 e 2020

#Figura com a distribuição de densidade média dos últimos 30 e para os ano de 2022

intervalos = ['2014-2023', '2023']

fig, ax = plt.subplots()

vento_clim = vento100.loc['2014':'2024']
# mean
mean = vento_clim.mean()
# variance
var = vento_clim.var()
# standard deviation
std = vento_clim.std()
# plot pdf
bins = np.arange(0, vento_clim.max(), .5)
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean, std), lw=2, color = 'k')
bins_pad = bins


vento_clim = vento100.loc['2023']
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

fig.savefig(path_figuras + '\distribuicao_media_clim_ultimos30y_2022.png', format='png', dpi=300, bbox_inches='tight')

















