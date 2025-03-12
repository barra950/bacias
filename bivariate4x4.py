import numpy as np


import matplotlib.pyplot as plt
import numpy as np



# Define the 4x4 grid for temperature and precipitation
# Inverted logic: High values become low, and low values become high
temperature = np.array([
    [0.8, 0.6, 0.4, 0.2],  # High, Medium-High, Medium-Low, Low temperature (rows)
    [0.8, 0.6, 0.4, 0.2],
    [0.8, 0.6, 0.4, 0.2],
    [0.8, 0.6, 0.4, 0.2]
])

precipitation = np.array([
    [0.2, 0.2, 0.2, 0.2],  # Low precipitation (columns)
    [0.4, 0.4, 0.4, 0.4],  # Medium-Low precipitation
    [0.6, 0.6, 0.6, 0.6],  # Medium-High precipitation
    [0.8, 0.8, 0.8, 0.8]   # High precipitation
])

# Combine temperature and precipitation into RGB values
# Temperature is represented by the red channel (orange), precipitation by the blue channel
# Green channel is kept low to avoid mixing colors
red = temperature  # Orange (temperature)
green = temperature * 1  # Reduce green to make it more orange
blue = precipitation  # Blue (precipitation)

# Stack the channels to create an RGB image
rgb_image = np.stack([red, green, blue], axis=-1)

# Create the plot
fig = plt.figure()
plt.imshow(rgb_image, extent=[0, 3, 0, 3])  # 3x3 grid
plt.xticks(np.arange(0.5, 3.5, 1), labels=['Low', 'Medium', 'High'])  # Label x-axis (precipitation)
plt.yticks(np.arange(0.5, 3.5, 1), labels=['low', 'Medium', 'high'])  # Label y-axis (temperature, inverted)
# plt.xlabel("Precipitation")
# plt.ylabel("Temperature")
# plt.title("Inverted Temperature and Precipitation Grid")
plt.xlabel("Precipitacao")
plt.ylabel("Temperatura")
nameoffigure = "bivariate_colorbar44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)


def classify_distribution(values):
    """
    Classifies each value in the input array into:
    - 'Bottom 1/4'
    - 'Lower-Middle 1/4'
    - 'Upper-Middle 1/4'
    - 'Top 1/4'
    based on their position in the sorted distribution.
    Works for arrays of any dimension.
    """
    # Flatten the array to handle any dimension
    flattened_values = values.flatten()
    
    # Sort the flattened values to determine thresholds
    sorted_values = np.sort(flattened_values)
    
    # Calculate the indices for the quartiles
    n = len(sorted_values)
    q1 = sorted_values[n // 4]  # 1/4 of the data
    q2 = sorted_values[n // 2]  # 2/4 (median) of the data
    q3 = sorted_values[3 * n // 4]  # 3/4 of the data
    
    # Classify each value
    classification = np.where(flattened_values < q1, '1',
                              np.where(flattened_values < q2, '2',
                                      np.where(flattened_values < q3, '3', '4')))
    
    # Reshape the classification result to match the original input shape
    return classification.reshape(values.shape)


#path_figuras = '/home/numa23/copernicus/figures/'
#filelocation = '/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/'
#shapefile_br = "/home/numa23/copernicus/Lim_america_do_sul_IBGE_2021" #America do sul

#filepattern = 'precip_'


#arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]

#count = 0
#for arq in arqs:
#    ds = xr.open_dataset(os.path.join(filelocation,arq))
#    ds_temp = ds.tp
#    #ds_ne=ds
#    if count == 0:
#        vento100_ne = ds_temp
#        count += 1
#    else:
#        vento100_ne = xr.concat([vento100_ne, ds_temp],dim='time')
#    # vento100_ne = vento_ne.mean(dim = ('latitude', 'longitude')).resample(time = '1D').mean(dim = 'time')


#print(vento100_ne)
#print(vento100_ne.shape)

#print(vento100_ne[0,0,0])
#print(vento100_ne[1,0,0])


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
for k in range(2016,2024):
    rootgrp = Dataset(f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/temp_{k}.nc','r')

    dims = rootgrp.dimensions

    vars = rootgrp.variables
    
    attrs = rootgrp.ncattrs

    ndims = len(dims)

    print ('number of dimensions = ' + str(ndims))
    
    temp = vars['t2m'][:]
    time=vars['valid_time'][:]

    if counter == 0:
    
        arr1w = temp
        arr1t = time
    
    else:
        
        arr2w = np.concatenate((arr1w, temp), axis=0)
        
        arr1w = arr2w

        arr2t = np.concatenate((arr1t, time), axis=0)

        arr1t = arr2t
        
    counter =counter +1

temp = np.nanmean(arr1w,axis=(0))

classification_result_temp = classify_distribution(temp)
print(arr1w.shape)

# Get the time information for the file.
timeUnits = vars['valid_time'].units

# Make dates for the file.
Date = num2date(arr1t,timeUnits,calendar='standard')
Day = np.asarray([d.day for d in Date])
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])


veraotemp = []
outonotemp = []
invernotemp = []
primaveratemp = []
for k in range(0,len(arr1w)):
    if Month[k] == 1:
        veraotemp.append(arr1w[k])
    if Month[k] == 2:
        veraotemp.append(arr1w[k])
    if Month[k] == 3:
        if Day[k] > 21:
            outonotemp.append(arr1w[k])
        else:
            veraotemp.append(arr1w[k])
    if Month[k] == 4:
        outonotemp.append(arr1w[k])
    if Month[k] == 5:
        outonotemp.append(arr1w[k])
    if Month[k] == 6:
        if Day[k] > 21:
            invernotemp.append(arr1w[k])
        else:
            outonotemp.append(arr1w[k])
    if Month[k] == 7:
        invernotemp.append(arr1w[k])
    if Month[k] == 8:
        invernotemp.append(arr1w[k])
    if Month[k] == 9:
        if Day[k] > 21:
            primaveratemp.append(arr1w[k])
        else:
            invernotemp.append(arr1w[k])
    if Month[k] == 10:
        primaveratemp.append(arr1w[k])
    if Month[k] == 11:
        primaveratemp.append(arr1w[k])
    if Month[k] == 12:
        if Day[k] > 21:
            veraotemp.append(arr1w[k])
        else:
            primaveratemp.append(arr1w[k])

veraotemp = np.array(veraotemp)
outonotemp = np.array(outonotemp)
invernotemp = np.array(invernotemp)
primaveratemp = np.array(primaveratemp)


counter = 0
for k in range(2016,2024):
    rootgrp = Dataset(f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/precip_{k}.nc','r')

    dims = rootgrp.dimensions

    vars = rootgrp.variables
    
    attrs = rootgrp.ncattrs

    ndims = len(dims)

    print ('number of dimensions = ' + str(ndims))
    
    humi = vars['tp'][:]
    time=vars['valid_time'][:]

    if counter == 0:
    
        arr1w = humi
        arr1t = time
    
    else:
        
        arr2w = np.concatenate((arr1w, humi), axis=0)
        
        arr1w = arr2w

        arr2t = np.concatenate((arr1t, time), axis=0)

        arr1t = arr2t
        
    counter =counter +1


# Get the time information for the file.
timeUnits = vars['valid_time'].units

# Make dates for the file.
Date = num2date(arr1t,timeUnits,calendar='standard')
Day = np.asarray([d.day for d in Date])
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])


lon=vars['longitude'][:]
lat=vars['latitude'][:]

print(arr1w.shape)

humi = np.nanmean(arr1w,axis=(0))

classification_result_humi = classify_distribution(humi)

temphumi = np.char.add(classification_result_temp, classification_result_humi)

veraohumi = []
outonohumi = []
invernohumi = []
primaverahumi = []
for k in range(0,len(arr1w)):
    if Month[k] == 1:
        veraohumi.append(arr1w[k])
    if Month[k] == 2:
        veraohumi.append(arr1w[k])
    if Month[k] == 3:
        if Day[k] > 21:
            outonohumi.append(arr1w[k])
        else:
            veraohumi.append(arr1w[k])
    if Month[k] == 4:
        outonohumi.append(arr1w[k])
    if Month[k] == 5:
        outonohumi.append(arr1w[k])
    if Month[k] == 6:
        if Day[k] > 21:
            invernohumi.append(arr1w[k])
        else:
            outonohumi.append(arr1w[k])
    if Month[k] == 7:
        invernohumi.append(arr1w[k])
    if Month[k] == 8:
        invernohumi.append(arr1w[k])
    if Month[k] == 9:
        if Day[k] > 21:
            primaverahumi.append(arr1w[k])
        else:
            invernohumi.append(arr1w[k])
    if Month[k] == 10:
        primaverahumi.append(arr1w[k])
    if Month[k] == 11:
        primaverahumi.append(arr1w[k])
    if Month[k] == 12:
        if Day[k] > 21:
            veraohumi.append(arr1w[k])
        else:
            primaverahumi.append(arr1w[k])

veraohumi = np.array(veraohumi)
outonohumi = np.array(outonohumi)
invernohumi = np.array(invernohumi)
primaverahumi = np.array(primaverahumi)



colors_blend = np.zeros((len(temphumi), len(temphumi[0]),3))
for k in range(0,len(temphumi)):
    for t in range(0,len(temphumi[0])):
        if temphumi[k,t] == '11':
            colors_blend[k,t,0] = rgb_image[3,0,0]
            colors_blend[k,t,1] = rgb_image[3,0,1]
            colors_blend[k,t,2] = rgb_image[3,0,2]
        if temphumi[k,t] == '12':
            colors_blend[k,t,0] = rgb_image[3,1,0]
            colors_blend[k,t,1] = rgb_image[3,1,1]
            colors_blend[k,t,2] = rgb_image[3,1,2]
        if temphumi[k,t] == '13':
            colors_blend[k,t,0] = rgb_image[3,2,0]
            colors_blend[k,t,1] = rgb_image[3,2,1]
            colors_blend[k,t,2] = rgb_image[3,2,2]
        if temphumi[k,t] == '14':
            colors_blend[k,t,0] = rgb_image[3,3,0]
            colors_blend[k,t,1] = rgb_image[3,3,1]
            colors_blend[k,t,2] = rgb_image[3,3,2]       
        if temphumi[k,t] == '21':
            colors_blend[k,t,0] = rgb_image[2,0,0]
            colors_blend[k,t,1] = rgb_image[2,0,1]
            colors_blend[k,t,2] = rgb_image[2,0,2]
        if temphumi[k,t] == '22':
            colors_blend[k,t,0] = rgb_image[2,1,0]
            colors_blend[k,t,1] = rgb_image[2,1,1]
            colors_blend[k,t,2] = rgb_image[2,1,2]
        if temphumi[k,t] == '23':
            colors_blend[k,t,0] = rgb_image[2,2,0]
            colors_blend[k,t,1] = rgb_image[2,2,1]
            colors_blend[k,t,2] = rgb_image[2,2,2]
        if temphumi[k,t] == '24':
            colors_blend[k,t,0] = rgb_image[2,3,0]
            colors_blend[k,t,1] = rgb_image[2,3,1]
            colors_blend[k,t,2] = rgb_image[2,3,2]            
        if temphumi[k,t] == '31':
            colors_blend[k,t,0] = rgb_image[1,0,0]
            colors_blend[k,t,1] = rgb_image[1,0,1]
            colors_blend[k,t,2] = rgb_image[1,0,2]
        if temphumi[k,t] == '32':
            colors_blend[k,t,0] = rgb_image[1,1,0]
            colors_blend[k,t,1] = rgb_image[1,1,1]
            colors_blend[k,t,2] = rgb_image[1,1,2]
        if temphumi[k,t] == '33':
            colors_blend[k,t,0] = rgb_image[1,2,0]
            colors_blend[k,t,1] = rgb_image[1,2,1]
            colors_blend[k,t,2] = rgb_image[1,2,2]
        if temphumi[k,t] == '34':
            colors_blend[k,t,0] = rgb_image[1,3,0]
            colors_blend[k,t,1] = rgb_image[1,3,1]
            colors_blend[k,t,2] = rgb_image[1,3,2]
        if temphumi[k,t] == '41':
            colors_blend[k,t,0] = rgb_image[0,0,0]
            colors_blend[k,t,1] = rgb_image[0,0,1]
            colors_blend[k,t,2] = rgb_image[0,0,2]
        if temphumi[k,t] == '42':
            colors_blend[k,t,0] = rgb_image[0,1,0]
            colors_blend[k,t,1] = rgb_image[0,1,1]
            colors_blend[k,t,2] = rgb_image[0,1,2]
        if temphumi[k,t] == '43':
            colors_blend[k,t,0] = rgb_image[0,2,0]
            colors_blend[k,t,1] = rgb_image[0,2,1]
            colors_blend[k,t,2] = rgb_image[0,2,2]
        if temphumi[k,t] == '44':
            colors_blend[k,t,0] = rgb_image[0,3,0]
            colors_blend[k,t,1] = rgb_image[0,3,1]
            colors_blend[k,t,2] = rgb_image[0,3,2]



# Create the plot
fig = plt.figure()
plt.imshow(colors_blend)  # 3x3 grid
#plt.xticks(np.arange(0.5, 3.5, 1), labels=['Low', 'Medium', 'High'])  # Label x-axis (precipitation)
#plt.yticks(np.arange(0.5, 3.5, 1), labels=['low', 'Medium', 'high'])  # Label y-axis (temperature, inverted)
plt.xlabel("Precipitation")
plt.ylabel("Temperature")
plt.title("Inverted Temperature and Precipitation Grid")
nameoffigure = "Inverted_grid_plot44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)


# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
im = ax.imshow(colors_blend[::-1,:], extent=(-55, -35, -30, 9), transform=ccrs.PlateCarree(), origin='lower')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)



# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
cmap = cmocean.cm.speed
plt.contourf(lon, lat ,humi,transform = ccrs.PlateCarree(),color='k',cmap=cmap)
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil_precip"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)


#%%

# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
plt.contourf(lon, lat ,temp,transform = ccrs.PlateCarree(),color='k',cmap='CMRmap')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil_temp"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)


#Agora, plotando as figuras para verao outono inverno e primavera

veraotemp = np.nanmean(veraotemp,axis=(0))

classification_result_veraotemp = classify_distribution(veraotemp)

veraohumi = np.nanmean(veraohumi,axis=(0))

classification_result_veraohumi = classify_distribution(veraohumi)

veraotemphumi = np.char.add(classification_result_veraotemp, classification_result_veraohumi)


outonotemp = np.nanmean(outonotemp,axis=(0))

classification_result_outonotemp = classify_distribution(outonotemp)

outonohumi = np.nanmean(outonohumi,axis=(0))

classification_result_outonohumi = classify_distribution(outonohumi)

outonotemphumi = np.char.add(classification_result_outonotemp, classification_result_outonohumi)


invernotemp = np.nanmean(invernotemp,axis=(0))

classification_result_invernotemp = classify_distribution(invernotemp)

invernohumi = np.nanmean(invernohumi,axis=(0))

classification_result_invernohumi = classify_distribution(invernohumi)

invernotemphumi = np.char.add(classification_result_invernotemp, classification_result_invernohumi)


primaveratemp = np.nanmean(primaveratemp,axis=(0))

classification_result_primaveratemp = classify_distribution(primaveratemp)

primaverahumi = np.nanmean(primaverahumi,axis=(0))

classification_result_primaverahumi = classify_distribution(primaverahumi)

primaveratemphumi = np.char.add(classification_result_primaveratemp, classification_result_primaverahumi)


#Parte do verao
colors_blend = np.zeros((len(veraotemphumi), len(veraotemphumi[0]),3))
for k in range(0,len(veraotemphumi)):
    for t in range(0,len(veraotemphumi[0])):
        if veraotemphumi[k,t] == '11':
            colors_blend[k,t,0] = rgb_image[3,0,0]
            colors_blend[k,t,1] = rgb_image[3,0,1]
            colors_blend[k,t,2] = rgb_image[3,0,2]
        if veraotemphumi[k,t] == '12':
            colors_blend[k,t,0] = rgb_image[3,1,0]
            colors_blend[k,t,1] = rgb_image[3,1,1]
            colors_blend[k,t,2] = rgb_image[3,1,2]
        if veraotemphumi[k,t] == '13':
            colors_blend[k,t,0] = rgb_image[3,2,0]
            colors_blend[k,t,1] = rgb_image[3,2,1]
            colors_blend[k,t,2] = rgb_image[3,2,2]
        if veraotemphumi[k,t] == '14':
            colors_blend[k,t,0] = rgb_image[3,3,0]
            colors_blend[k,t,1] = rgb_image[3,3,1]
            colors_blend[k,t,2] = rgb_image[3,3,2]       
        if veraotemphumi[k,t] == '21':
            colors_blend[k,t,0] = rgb_image[2,0,0]
            colors_blend[k,t,1] = rgb_image[2,0,1]
            colors_blend[k,t,2] = rgb_image[2,0,2]
        if veraotemphumi[k,t] == '22':
            colors_blend[k,t,0] = rgb_image[2,1,0]
            colors_blend[k,t,1] = rgb_image[2,1,1]
            colors_blend[k,t,2] = rgb_image[2,1,2]
        if veraotemphumi[k,t] == '23':
            colors_blend[k,t,0] = rgb_image[2,2,0]
            colors_blend[k,t,1] = rgb_image[2,2,1]
            colors_blend[k,t,2] = rgb_image[2,2,2]
        if veraotemphumi[k,t] == '24':
            colors_blend[k,t,0] = rgb_image[2,3,0]
            colors_blend[k,t,1] = rgb_image[2,3,1]
            colors_blend[k,t,2] = rgb_image[2,3,2]            
        if veraotemphumi[k,t] == '31':
            colors_blend[k,t,0] = rgb_image[1,0,0]
            colors_blend[k,t,1] = rgb_image[1,0,1]
            colors_blend[k,t,2] = rgb_image[1,0,2]
        if veraotemphumi[k,t] == '32':
            colors_blend[k,t,0] = rgb_image[1,1,0]
            colors_blend[k,t,1] = rgb_image[1,1,1]
            colors_blend[k,t,2] = rgb_image[1,1,2]
        if veraotemphumi[k,t] == '33':
            colors_blend[k,t,0] = rgb_image[1,2,0]
            colors_blend[k,t,1] = rgb_image[1,2,1]
            colors_blend[k,t,2] = rgb_image[1,2,2]
        if veraotemphumi[k,t] == '34':
            colors_blend[k,t,0] = rgb_image[1,3,0]
            colors_blend[k,t,1] = rgb_image[1,3,1]
            colors_blend[k,t,2] = rgb_image[1,3,2]
        if veraotemphumi[k,t] == '41':
            colors_blend[k,t,0] = rgb_image[0,0,0]
            colors_blend[k,t,1] = rgb_image[0,0,1]
            colors_blend[k,t,2] = rgb_image[0,0,2]
        if veraotemphumi[k,t] == '42':
            colors_blend[k,t,0] = rgb_image[0,1,0]
            colors_blend[k,t,1] = rgb_image[0,1,1]
            colors_blend[k,t,2] = rgb_image[0,1,2]
        if veraotemphumi[k,t] == '43':
            colors_blend[k,t,0] = rgb_image[0,2,0]
            colors_blend[k,t,1] = rgb_image[0,2,1]
            colors_blend[k,t,2] = rgb_image[0,2,2]
        if veraotemphumi[k,t] == '44':
            colors_blend[k,t,0] = rgb_image[0,3,0]
            colors_blend[k,t,1] = rgb_image[0,3,1]
            colors_blend[k,t,2] = rgb_image[0,3,2]


# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
im = ax.imshow(colors_blend[::-1,:], extent=(-55, -35, -30, 9), transform=ccrs.PlateCarree(), origin='lower')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil_verao44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)


#Parte do outono
colors_blend = np.zeros((len(outonotemphumi), len(outonotemphumi[0]),3))
for k in range(0,len(outonotemphumi)):
    for t in range(0,len(outonotemphumi[0])):
        if outonotemphumi[k,t] == '11':
            colors_blend[k,t,0] = rgb_image[3,0,0]
            colors_blend[k,t,1] = rgb_image[3,0,1]
            colors_blend[k,t,2] = rgb_image[3,0,2]
        if outonotemphumi[k,t] == '12':
            colors_blend[k,t,0] = rgb_image[3,1,0]
            colors_blend[k,t,1] = rgb_image[3,1,1]
            colors_blend[k,t,2] = rgb_image[3,1,2]
        if outonotemphumi[k,t] == '13':
            colors_blend[k,t,0] = rgb_image[3,2,0]
            colors_blend[k,t,1] = rgb_image[3,2,1]
            colors_blend[k,t,2] = rgb_image[3,2,2]
        if outonotemphumi[k,t] == '14':
            colors_blend[k,t,0] = rgb_image[3,3,0]
            colors_blend[k,t,1] = rgb_image[3,3,1]
            colors_blend[k,t,2] = rgb_image[3,3,2]       
        if outonotemphumi[k,t] == '21':
            colors_blend[k,t,0] = rgb_image[2,0,0]
            colors_blend[k,t,1] = rgb_image[2,0,1]
            colors_blend[k,t,2] = rgb_image[2,0,2]
        if outonotemphumi[k,t] == '22':
            colors_blend[k,t,0] = rgb_image[2,1,0]
            colors_blend[k,t,1] = rgb_image[2,1,1]
            colors_blend[k,t,2] = rgb_image[2,1,2]
        if outonotemphumi[k,t] == '23':
            colors_blend[k,t,0] = rgb_image[2,2,0]
            colors_blend[k,t,1] = rgb_image[2,2,1]
            colors_blend[k,t,2] = rgb_image[2,2,2]
        if outonotemphumi[k,t] == '24':
            colors_blend[k,t,0] = rgb_image[2,3,0]
            colors_blend[k,t,1] = rgb_image[2,3,1]
            colors_blend[k,t,2] = rgb_image[2,3,2]            
        if outonotemphumi[k,t] == '31':
            colors_blend[k,t,0] = rgb_image[1,0,0]
            colors_blend[k,t,1] = rgb_image[1,0,1]
            colors_blend[k,t,2] = rgb_image[1,0,2]
        if outonotemphumi[k,t] == '32':
            colors_blend[k,t,0] = rgb_image[1,1,0]
            colors_blend[k,t,1] = rgb_image[1,1,1]
            colors_blend[k,t,2] = rgb_image[1,1,2]
        if outonotemphumi[k,t] == '33':
            colors_blend[k,t,0] = rgb_image[1,2,0]
            colors_blend[k,t,1] = rgb_image[1,2,1]
            colors_blend[k,t,2] = rgb_image[1,2,2]
        if outonotemphumi[k,t] == '34':
            colors_blend[k,t,0] = rgb_image[1,3,0]
            colors_blend[k,t,1] = rgb_image[1,3,1]
            colors_blend[k,t,2] = rgb_image[1,3,2]
        if outonotemphumi[k,t] == '41':
            colors_blend[k,t,0] = rgb_image[0,0,0]
            colors_blend[k,t,1] = rgb_image[0,0,1]
            colors_blend[k,t,2] = rgb_image[0,0,2]
        if outonotemphumi[k,t] == '42':
            colors_blend[k,t,0] = rgb_image[0,1,0]
            colors_blend[k,t,1] = rgb_image[0,1,1]
            colors_blend[k,t,2] = rgb_image[0,1,2]
        if outonotemphumi[k,t] == '43':
            colors_blend[k,t,0] = rgb_image[0,2,0]
            colors_blend[k,t,1] = rgb_image[0,2,1]
            colors_blend[k,t,2] = rgb_image[0,2,2]
        if outonotemphumi[k,t] == '44':
            colors_blend[k,t,0] = rgb_image[0,3,0]
            colors_blend[k,t,1] = rgb_image[0,3,1]
            colors_blend[k,t,2] = rgb_image[0,3,2]


# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
im = ax.imshow(colors_blend[::-1,:], extent=(-55, -35, -30, 9), transform=ccrs.PlateCarree(), origin='lower')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil_outono44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)



#Parte do inverno
colors_blend = np.zeros((len(invernotemphumi), len(invernotemphumi[0]),3))
for k in range(0,len(invernotemphumi)):
    for t in range(0,len(invernotemphumi[0])):
        if invernotemphumi[k,t] == '11':
            colors_blend[k,t,0] = rgb_image[3,0,0]
            colors_blend[k,t,1] = rgb_image[3,0,1]
            colors_blend[k,t,2] = rgb_image[3,0,2]
        if invernotemphumi[k,t] == '12':
            colors_blend[k,t,0] = rgb_image[3,1,0]
            colors_blend[k,t,1] = rgb_image[3,1,1]
            colors_blend[k,t,2] = rgb_image[3,1,2]
        if invernotemphumi[k,t] == '13':
            colors_blend[k,t,0] = rgb_image[3,2,0]
            colors_blend[k,t,1] = rgb_image[3,2,1]
            colors_blend[k,t,2] = rgb_image[3,2,2]
        if invernotemphumi[k,t] == '14':
            colors_blend[k,t,0] = rgb_image[3,3,0]
            colors_blend[k,t,1] = rgb_image[3,3,1]
            colors_blend[k,t,2] = rgb_image[3,3,2]       
        if invernotemphumi[k,t] == '21':
            colors_blend[k,t,0] = rgb_image[2,0,0]
            colors_blend[k,t,1] = rgb_image[2,0,1]
            colors_blend[k,t,2] = rgb_image[2,0,2]
        if invernotemphumi[k,t] == '22':
            colors_blend[k,t,0] = rgb_image[2,1,0]
            colors_blend[k,t,1] = rgb_image[2,1,1]
            colors_blend[k,t,2] = rgb_image[2,1,2]
        if invernotemphumi[k,t] == '23':
            colors_blend[k,t,0] = rgb_image[2,2,0]
            colors_blend[k,t,1] = rgb_image[2,2,1]
            colors_blend[k,t,2] = rgb_image[2,2,2]
        if invernotemphumi[k,t] == '24':
            colors_blend[k,t,0] = rgb_image[2,3,0]
            colors_blend[k,t,1] = rgb_image[2,3,1]
            colors_blend[k,t,2] = rgb_image[2,3,2]            
        if invernotemphumi[k,t] == '31':
            colors_blend[k,t,0] = rgb_image[1,0,0]
            colors_blend[k,t,1] = rgb_image[1,0,1]
            colors_blend[k,t,2] = rgb_image[1,0,2]
        if invernotemphumi[k,t] == '32':
            colors_blend[k,t,0] = rgb_image[1,1,0]
            colors_blend[k,t,1] = rgb_image[1,1,1]
            colors_blend[k,t,2] = rgb_image[1,1,2]
        if invernotemphumi[k,t] == '33':
            colors_blend[k,t,0] = rgb_image[1,2,0]
            colors_blend[k,t,1] = rgb_image[1,2,1]
            colors_blend[k,t,2] = rgb_image[1,2,2]
        if invernotemphumi[k,t] == '34':
            colors_blend[k,t,0] = rgb_image[1,3,0]
            colors_blend[k,t,1] = rgb_image[1,3,1]
            colors_blend[k,t,2] = rgb_image[1,3,2]
        if invernotemphumi[k,t] == '41':
            colors_blend[k,t,0] = rgb_image[0,0,0]
            colors_blend[k,t,1] = rgb_image[0,0,1]
            colors_blend[k,t,2] = rgb_image[0,0,2]
        if invernotemphumi[k,t] == '42':
            colors_blend[k,t,0] = rgb_image[0,1,0]
            colors_blend[k,t,1] = rgb_image[0,1,1]
            colors_blend[k,t,2] = rgb_image[0,1,2]
        if invernotemphumi[k,t] == '43':
            colors_blend[k,t,0] = rgb_image[0,2,0]
            colors_blend[k,t,1] = rgb_image[0,2,1]
            colors_blend[k,t,2] = rgb_image[0,2,2]
        if invernotemphumi[k,t] == '44':
            colors_blend[k,t,0] = rgb_image[0,3,0]
            colors_blend[k,t,1] = rgb_image[0,3,1]
            colors_blend[k,t,2] = rgb_image[0,3,2]


# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
im = ax.imshow(colors_blend[::-1,:], extent=(-55, -35, -30, 9), transform=ccrs.PlateCarree(), origin='lower')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil_inverno44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)


#Parte da primavera
colors_blend = np.zeros((len(primaveratemphumi), len(primaveratemphumi[0]),3))
for k in range(0,len(primaveratemphumi)):
    for t in range(0,len(primaveratemphumi[0])):
        if primaveratemphumi[k,t] == '11':
            colors_blend[k,t,0] = rgb_image[3,0,0]
            colors_blend[k,t,1] = rgb_image[3,0,1]
            colors_blend[k,t,2] = rgb_image[3,0,2]
        if primaveratemphumi[k,t] == '12':
            colors_blend[k,t,0] = rgb_image[3,1,0]
            colors_blend[k,t,1] = rgb_image[3,1,1]
            colors_blend[k,t,2] = rgb_image[3,1,2]
        if primaveratemphumi[k,t] == '13':
            colors_blend[k,t,0] = rgb_image[3,2,0]
            colors_blend[k,t,1] = rgb_image[3,2,1]
            colors_blend[k,t,2] = rgb_image[3,2,2]
        if primaveratemphumi[k,t] == '14':
            colors_blend[k,t,0] = rgb_image[3,3,0]
            colors_blend[k,t,1] = rgb_image[3,3,1]
            colors_blend[k,t,2] = rgb_image[3,3,2]       
        if primaveratemphumi[k,t] == '21':
            colors_blend[k,t,0] = rgb_image[2,0,0]
            colors_blend[k,t,1] = rgb_image[2,0,1]
            colors_blend[k,t,2] = rgb_image[2,0,2]
        if primaveratemphumi[k,t] == '22':
            colors_blend[k,t,0] = rgb_image[2,1,0]
            colors_blend[k,t,1] = rgb_image[2,1,1]
            colors_blend[k,t,2] = rgb_image[2,1,2]
        if primaveratemphumi[k,t] == '23':
            colors_blend[k,t,0] = rgb_image[2,2,0]
            colors_blend[k,t,1] = rgb_image[2,2,1]
            colors_blend[k,t,2] = rgb_image[2,2,2]
        if primaveratemphumi[k,t] == '24':
            colors_blend[k,t,0] = rgb_image[2,3,0]
            colors_blend[k,t,1] = rgb_image[2,3,1]
            colors_blend[k,t,2] = rgb_image[2,3,2]            
        if primaveratemphumi[k,t] == '31':
            colors_blend[k,t,0] = rgb_image[1,0,0]
            colors_blend[k,t,1] = rgb_image[1,0,1]
            colors_blend[k,t,2] = rgb_image[1,0,2]
        if primaveratemphumi[k,t] == '32':
            colors_blend[k,t,0] = rgb_image[1,1,0]
            colors_blend[k,t,1] = rgb_image[1,1,1]
            colors_blend[k,t,2] = rgb_image[1,1,2]
        if primaveratemphumi[k,t] == '33':
            colors_blend[k,t,0] = rgb_image[1,2,0]
            colors_blend[k,t,1] = rgb_image[1,2,1]
            colors_blend[k,t,2] = rgb_image[1,2,2]
        if primaveratemphumi[k,t] == '34':
            colors_blend[k,t,0] = rgb_image[1,3,0]
            colors_blend[k,t,1] = rgb_image[1,3,1]
            colors_blend[k,t,2] = rgb_image[1,3,2]
        if primaveratemphumi[k,t] == '41':
            colors_blend[k,t,0] = rgb_image[0,0,0]
            colors_blend[k,t,1] = rgb_image[0,0,1]
            colors_blend[k,t,2] = rgb_image[0,0,2]
        if primaveratemphumi[k,t] == '42':
            colors_blend[k,t,0] = rgb_image[0,1,0]
            colors_blend[k,t,1] = rgb_image[0,1,1]
            colors_blend[k,t,2] = rgb_image[0,1,2]
        if primaveratemphumi[k,t] == '43':
            colors_blend[k,t,0] = rgb_image[0,2,0]
            colors_blend[k,t,1] = rgb_image[0,2,1]
            colors_blend[k,t,2] = rgb_image[0,2,2]
        if primaveratemphumi[k,t] == '44':
            colors_blend[k,t,0] = rgb_image[0,3,0]
            colors_blend[k,t,1] = rgb_image[0,3,1]
            colors_blend[k,t,2] = rgb_image[0,3,2]


# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-55, -35, 9, -30], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-55,-35,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-30,9,1) # define latitude ticks  #Bacia de Campos
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

# Plot the bivariate data
im = ax.imshow(colors_blend[::-1,:], extent=(-55, -35, -30, 9), transform=ccrs.PlateCarree(), origin='lower')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "plot_fig_brasil_primavera44"
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"+string_in_string,dpi=100)
