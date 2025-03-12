import numpy as np


import matplotlib.pyplot as plt
import numpy as np

# Define the 3x3 grid for temperature and precipitation
# Inverted logic: High values become low, and low values become high
temperature = np.array([
    [0.8, 0.5, 0.2],  # High, Medium, Low temperature (rows)
    [0.8, 0.5, 0.2],
    [0.8, 0.5, 0.2]
])

precipitation = np.array([
    [0.2, 0.2, 0.2],  # Low precipitation (columns)
    [0.5, 0.5, 0.5],  # Medium precipitation
    [0.8, 0.8, 0.8]   # High precipitation
])

# Combine temperature and precipitation into RGB values
# Temperature is represented by the red channel (orange), precipitation by the blue channel
# Green channel is kept low to avoid mixing colors
red = temperature  # Orange (temperature)
green = temperature * 1#0.5  # Reduce green to make it more orange
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
plt.xlabel("Vento")
plt.ylabel("Radiacao solar")
plt.show()


def classify_distribution(values):
    """
    Classifies each value in the input array into:
    - 'Bottom 1/3'
    - 'Middle 1/3'
    - 'Top 1/3'
    based on their position in the sorted distribution.
    Works for arrays of any dimension.
    """
    # Flatten the array to handle any dimension
    flattened_values = values.flatten()
    
    # Sort the flattened values to determine thresholds
    sorted_values = np.sort(flattened_values)
    
    # Calculate the indices for the 1/3 and 2/3 percentiles
    lower_threshold = sorted_values[len(sorted_values) // 3]  # 1/3 of the data
    upper_threshold = sorted_values[2 * len(sorted_values) // 3]  # 2/3 of the data
    
    # Classify each value
    classification = np.where(flattened_values < lower_threshold, '1',
                              np.where(flattened_values < upper_threshold, '2', '3'))
    
    # Reshape the classification result to match the original input shape
    return classification.reshape(values.shape)


# def classify_distribution(values):
#     """
#     Classifies each value in the input array into:
#     - 'Bottom 1/3'
#     - 'Middle 1/3'
#     - 'Top 1/3'
#     based on their position relative to the range between the minimum and maximum values.
#     Works for arrays of any dimension and with any values (positive, negative, or mixed).
#     """
#     # Flatten the array to handle any dimension
#     flattened_values = values.flatten()
    
#     # Find the minimum and maximum values
#     min_value = np.min(flattened_values)
#     max_value = np.max(flattened_values)
    
#     # Calculate the range and thresholds
#     range_values = max_value - min_value
#     lower_threshold = min_value + (range_values / 3)  # 1/3 of the range
#     upper_threshold = min_value + (2 * range_values / 3)  # 2/3 of the range
    
#     # Classify each value
#     classification = np.where(flattened_values < lower_threshold, '1',
#                               np.where(flattened_values < upper_threshold, '2', '3'))
    
#     # Reshape the classification result to match the original input shape
#     return classification.reshape(values.shape)

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




rootgrp = Dataset('/home/owner/Documents/copernicus/download2023sc.nc','r')

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
#mslp = vars['msl'][:]
time = vars['time'][:]
uwnd = vars['u100'][:]
vwnd = vars['v100'][:]
ssrd = vars['ssrd'][:]


classification_result_temp = classify_distribution(np.nanmean(ssrd,axis=(0)))

wndspeed = np.sqrt(vwnd**2   +  uwnd**2)
classification_result_humi = classify_distribution(np.nanmean(wndspeed,axis=(0)))

temphumi = np.char.add(classification_result_temp, classification_result_humi)

#%%



colors_blend = np.zeros((len(temphumi), len(temphumi[0]),3))
for k in range(0,len(temphumi)):
    for t in range(0,len(temphumi[0])):
        if temphumi[k,t] == '11':
            colors_blend[k,t,0] = rgb_image[2,0,0]
            colors_blend[k,t,1] = rgb_image[2,0,1]
            colors_blend[k,t,2] = rgb_image[2,0,2]
        if temphumi[k,t] == '12':
            colors_blend[k,t,0] = rgb_image[2,1,0]
            colors_blend[k,t,1] = rgb_image[2,1,1]
            colors_blend[k,t,2] = rgb_image[2,1,2]
        if temphumi[k,t] == '13':
            colors_blend[k,t,0] = rgb_image[2,2,0]
            colors_blend[k,t,1] = rgb_image[2,2,1]
            colors_blend[k,t,2] = rgb_image[2,2,2]
        if temphumi[k,t] == '21':
            colors_blend[k,t,0] = rgb_image[1,0,0]
            colors_blend[k,t,1] = rgb_image[1,0,1]
            colors_blend[k,t,2] = rgb_image[1,0,2]
        if temphumi[k,t] == '22':
            colors_blend[k,t,0] = rgb_image[1,1,0]
            colors_blend[k,t,1] = rgb_image[1,1,1]
            colors_blend[k,t,2] = rgb_image[1,1,2]
        if temphumi[k,t] == '23':
            colors_blend[k,t,0] = rgb_image[1,2,0]
            colors_blend[k,t,1] = rgb_image[1,2,1]
            colors_blend[k,t,2] = rgb_image[1,2,2]
        if temphumi[k,t] == '31':
            colors_blend[k,t,0] = rgb_image[0,0,0]
            colors_blend[k,t,1] = rgb_image[0,0,1]
            colors_blend[k,t,2] = rgb_image[0,0,2]
        if temphumi[k,t] == '32':
            colors_blend[k,t,0] = rgb_image[0,1,0]
            colors_blend[k,t,1] = rgb_image[0,1,1]
            colors_blend[k,t,2] = rgb_image[0,1,2]
        if temphumi[k,t] == '33':
            colors_blend[k,t,0] = rgb_image[0,2,0]
            colors_blend[k,t,1] = rgb_image[0,2,1]
            colors_blend[k,t,2] = rgb_image[0,2,2]
            
            
# Create the plot
fig = plt.figure()
plt.imshow(colors_blend)  # 3x3 grid
#plt.xticks(np.arange(0.5, 3.5, 1), labels=['Low', 'Medium', 'High'])  # Label x-axis (precipitation)
#plt.yticks(np.arange(0.5, 3.5, 1), labels=['low', 'Medium', 'high'])  # Label y-axis (temperature, inverted)
plt.xlabel("Precipitation")
plt.ylabel("Temperature")
plt.title("Inverted Temperature and Precipitation Grid")
plt.show()

#%%

# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,8))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-52, -36, -17, -31], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-52,-36,0.5) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-31,-17,0.5) # define latitude ticks  #Bacia de Campos
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
im = ax.imshow(colors_blend[::-1,:], extent=(-52, -36, -31, -17), transform=ccrs.PlateCarree(), origin='lower')


#%%

# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,8))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-52, -36, -17, -31], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-52,-36,0.5) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-31,-17,0.5) # define latitude ticks  #Bacia de Campos
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
plt.contourf(lon, lat ,np.nanmean(wndspeed,axis=(0)),transform = ccrs.PlateCarree(),color='k',cmap=cmap)

#%%

# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,8))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-52, -36, -17, -31], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-52,-36,0.5) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-31,-17,0.5) # define latitude ticks  #Bacia de Campos
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
plt.contourf(lon, lat ,np.nanmean(ssrd,axis=(0)),transform = ccrs.PlateCarree(),color='k',cmap='CMRmap')







