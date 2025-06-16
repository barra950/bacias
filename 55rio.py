import pandas as pd
import matplotlib.pyplot as plt #if using matplotlib
#import plotly.express as px #if using plotly
import geopandas as gpd
import pyproj
import numpy as np
import mapclassify
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex
import rasterio
from rasterio.windows import Window
import xarray as xr
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
import pandas as pd
import pickle
import geopy.distance
from math import sin, cos, sqrt, atan2, radians

import geopandas as gpd
import rioxarray as rio
from shapely.geometry import mapping

def shape_clip_dataarray(dataarray, shapefile_path, state_index=None, projection='epsg:4326',
                         x_dim='longitude', y_dim='latitude', invert=False, all_touched=True):
    """Clip a DataArray using a shapefile, optionally selecting a specific state by index."""
    shapefile = gpd.read_file(shapefile_path)

    # If state_index is specified, select only that state's geometry
    if state_index is not None:
        shapefile = shapefile.iloc[[state_index]]  # Use double brackets to keep as GeoDataFrame

    dataarray = dataarray.rio.write_crs(projection)
    dataarray = dataarray.rio.set_spatial_dims(x_dim, y_dim)
    return dataarray.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs,
                             drop=True, invert=invert, all_touched=all_touched)


def shape_clip_dataarray_total(dataarray, shapefile_path, projection = 'epsg:4326', x_dim = 'longitude', y_dim ='latitude', invert= False, all_touched=True):
    """Clip a DataArray using a shapefile."""
    shapefile = gpd.read_file(shapefile_path)
    dataarray = dataarray.rio.write_crs(projection)
    dataarray = dataarray.rio.set_spatial_dims(x_dim, y_dim)
    return dataarray.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop = True, invert = invert, all_touched = all_touched)



def classify_distribution(values):
    """
    Classifies each value in the input array into:
    - '0' for nan values
    - 'Bottom 1/5'
    - 'Lower Middle 1/5'
    - 'Middle 1/5'
    - 'Upper Middle 1/5'
    - 'Top 1/5'
    based on their position relative to the range between the minimum and maximum values.
    Works for arrays of any dimension and with any values (positive, negative, or mixed).
    """
    # Flatten the array to handle any dimension
    flattened_values = values.flatten()
    
    # Create an output array filled with '0' (will be overwritten for non-nan values)
    classification = np.full_like(flattened_values, '0', dtype='U1')
    
    # Find non-nan values
    non_nan_mask = ~np.isnan(flattened_values)
    non_nan_values = flattened_values[non_nan_mask]
    
    if len(non_nan_values) > 0:
        # Find the minimum and maximum values (ignoring nan)
        min_value = np.min(non_nan_values)
        max_value = np.max(non_nan_values)
        
        # Calculate the range and thresholds
        range_values = max_value - min_value
        threshold1 = min_value + (range_values / 5)  # 1/5 of the range
        threshold2 = min_value + (2 * range_values / 5)  # 2/5 of the range
        threshold3 = min_value + (3 * range_values / 5)  # 3/5 of the range
        threshold4 = min_value + (4 * range_values / 5)  # 4/5 of the range
        
        # Classify non-nan values
        classification[non_nan_mask] = np.where(non_nan_values < threshold1, '1',
                                              np.where(non_nan_values < threshold2, '2',
                                                      np.where(non_nan_values < threshold3, '3',
                                                              np.where(non_nan_values < threshold4, '4', '5'))))
    
    # Reshape the classification result to match the original input shape
    return classification.reshape(values.shape)


def classify_distribution2(basearray1,values):
    """
    Classifies each value in the input array into:
    - '0' for nan values
    - 'Bottom 1/5'
    - 'Lower Middle 1/5'
    - 'Middle 1/5'
    - 'Upper Middle 1/5'
    - 'Top 1/5'
    based on their position relative to the range between the minimum and maximum values.
    Works for arrays of any dimension and with any values (positive, negative, or mixed).
    """
    # Flatten the array to handle any dimension
    flattened_values = values.flatten()
    
    # Create an output array filled with '0' (will be overwritten for non-nan values)
    classification = np.full_like(flattened_values, '0', dtype='U1')
    
    # Find non-nan values
    non_nan_mask = ~np.isnan(flattened_values)
    non_nan_values = flattened_values[non_nan_mask]
    
    if len(non_nan_values) > 0:
        # Find the minimum and maximum values (ignoring nan)
        min_value = np.nanmin(basearray1)
        max_value = np.nanmax(basearray1)
        
        # Calculate the range and thresholds
        range_values = max_value - min_value
        threshold1 = min_value + (range_values / 5)  # 1/5 of the range
        threshold2 = min_value + (2 * range_values / 5)  # 2/5 of the range
        threshold3 = min_value + (3 * range_values / 5)  # 3/5 of the range
        threshold4 = min_value + (4 * range_values / 5)  # 4/5 of the range
        
        # Classify non-nan values
        classification[non_nan_mask] = np.where(non_nan_values < threshold1, '1',
                                              np.where(non_nan_values < threshold2, '2',
                                                      np.where(non_nan_values < threshold3, '3',
                                                              np.where(non_nan_values < threshold4, '4', '5'))))
    
    # Reshape the classification result to match the original input shape
    return classification.reshape(values.shape)



figsdir = "/home/numa23/Public/aesop/figures/"
filelocation = '/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/'
shapefile_br="/home/numa23/Public/Projeto_BR_OTEC/copernicus/shapefile_rj"


#Lendo os dados do era5-land
counter = 0
for k in range(2016,2025):
    for m in range(1,13):
        rootgrp = Dataset(f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/temp_precip_rio_{m}{k}.nc','r')
    
        dims = rootgrp.dimensions
    
        vars = rootgrp.variables
        
        attrs = rootgrp.ncattrs
    
        ndims = len(dims)
    
        print ('number of dimensions = ' + str(ndims))
        
        temp = vars['t2m'][:]
        humi = vars['tp'][:]
        time=vars['valid_time'][:]
    
        if counter == 0:
        
            arr1w = temp
            arr1p = humi
            arr1t = time
        
        else:
            
            arr2w = np.concatenate((arr1w, temp), axis=0)
            arr2p = np.concatenate((arr1p, humi), axis=0)
            
            arr1w = arr2w
            arr1p = arr2p
    
            arr2t = np.concatenate((arr1t, time), axis=0)
    
            arr1t = arr2t
            
        counter =counter +1

arr1w = arr1w - 273.15
temp = arr1w
humi = arr1p

lon=vars['longitude'][:]
lat=vars['latitude'][:]

# Get the time information for the file.
timeUnits = vars['valid_time'].units

# Make dates for the file.
Date = num2date(arr1t,timeUnits,calendar='standard')
Day = np.asarray([d.day for d in Date])
Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])

print(Date[0],Date[len(Date)-1])


meantemp = np.nanmean(temp[51:],axis=(0))
meanhumi = np.nanmean(humi[51:],axis=(0))

meanhumi = np.ma.masked_values(meanhumi,-99789)
meantemp = np.ma.masked_values(meantemp,-99789)


filepattern = 'temp_precip_rio'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))

teste = np.ones([46, 36])
data_xr = xr.DataArray(teste,
coords={'longitude': ds.longitude,'latitude': ds.latitude},
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray_total(data_xr, shapefile_br, invert = True, all_touched = True)

mask_total = ds2.T

for k in range(0,mask_total.shape[0]):
    for y in range(0,mask_total.shape[1]):
        if mask_total[k,y] == 1.0:
            mask_total[k,y] = np.nan
        else:
            mask_total[k,y] = 1.0

meantemp = meantemp*mask_total
meanhumi = meanhumi*mask_total



# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-45, -40.5, -20.5, -24], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
lontick = np.arange(-75,-32,1) # define longitude ticks  #Bacia de Campos
lattick = np.arange(-35,9,1) # define latitude ticks  #Bacia de Campos
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

fname = "/home/numa23/Public/Projeto_BR_OTEC/copernicus/shapefile_rj/RJ_Municipios_2022.shp"


shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature)



# Plot the bivariate data
cmap = cmocean.cm.speed
plt.pcolormesh(lon, lat ,meantemp,transform = ccrs.PlateCarree(),color='k',cmap=cmap)
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "teste_shapefile_rio"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)






#set up the file path and read the shapefile data

fp = "/home/numa23/Public/Projeto_BR_OTEC/copernicus/shapefile_rj/RJ_Municipios_2022.shx"

map_df = gpd.read_file(fp)
map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

def csv_to_2d_array_pandas(file_path, delimiter=';'):
    """
    Reads CSV using pandas and returns as proper 2D list.
    
    Args:
        file_path (str): Path to the CSV file
        delimiter (str): Character that separates values (default: ';')
        
    Returns:
        list: 2D array (list of lists)
        None: If file can't be read
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df.values.tolist()  # Convert to 2D Python list
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def non_zero_elements_equal(arr):
    non_zero = arr[arr != 0]  # Extract non-zero elements
    return np.all(non_zero == non_zero[0]) if len(non_zero) > 0 else True

datacsv = csv_to_2d_array_pandas('/home/numa23/Public/aesop/datadadosnormalizadosatuais(corrigido)2025.csv')
datacsv = np.array(datacsv)

casos = datacsv[0:418] #excluindo numero de semanas de 2025
print(casos.shape)

casos = casos[:,1:].astype(float)

casosmean = casos[1,:]


casosmean = np.nanmean(casos,axis=0) 

#see what the map looks like
fig=plt.figure(figsize=(8,12))
map_df.plot(figsize=(20, 10))

nameoffigure = "colorplethmap1"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)

print(map_df.head(93))


# Your custom RGB color matrix (5x5x3)
temperature = np.array([
    [1.0, 0.8, 0.6, 0.4, 0.2],  # High to Low temperature (rows)
    [1.0, 0.8, 0.6, 0.4, 0.2],
    [1.0, 0.8, 0.6, 0.4, 0.2],
    [1.0, 0.8, 0.6, 0.4, 0.2],
    [1.0, 0.8, 0.6, 0.4, 0.2]
])

precipitation = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2],  # Low precipitation (columns)
    [0.4, 0.4, 0.4, 0.4, 0.4],  # Medium-Low
    [0.6, 0.6, 0.6, 0.6, 0.6],  # Medium
    [0.8, 0.8, 0.8, 0.8, 0.8],  # Medium-High
    [1.0, 1.0, 1.0, 1.0, 1.0]   # High
])

# Combine into RGB (Red=Temperature, Blue=Precipitation, Green=Temperature)
red = temperature
green = temperature * 1  # Adjust if needed
blue = precipitation

# Stack to form RGB (shape: 5x5x3)
rgb_image = np.stack([red, green, blue], axis=-1)

# Flatten into 25 RGB colors (each row is an RGB triplet)
colors_25 = rgb_image.reshape(-1, 3)  # Shape: (25, 3)

# Convert to hex (optional, but works with geopandas.plot)
hex_colors = [
    '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
    for r, g, b in colors_25
]


# Convert each RGB triplet to hex (keeping 5x5 structure)
hex_colors2 = np.empty((5, 5), dtype=object)  # Create empty 5x5 array for strings

for i in range(5):
    for j in range(5):
        r, g, b = rgb_image[i, j]
        hex_colors2[i, j] = '#%02x%02x%02x' % (
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )

# Now `hex_colors2` is a 5x5 array of hex strings

# Classify your territories into 1-25 (replace this with your actual classification)
# Example: Assuming `casosmean` is your data (normalized to 0-1)
normalized_data = (casosmean - casosmean.min()) / (casosmean.max() - casosmean.min())
classified = (normalized_data * 24).astype(int)  # 0-24 (for 25 classes)

# Assign colors based on classification
colors_for_territories = [hex_colors[c] for c in classified]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
map_df.plot(color=colors_for_territories, linewidth=1, ax=ax, edgecolor='black')
ax.axis('off')

# Save
plt.savefig(figsdir + "temperature_precipitation_choropleth.png", dpi=300, bbox_inches='tight')
plt.close()


tempclas = []
humiclas = []
for t in range(0,len(casosmean)):
#for t in range(0,2):
   
    #Cria máscaras para cada área

    ########### Brasil ###########

    filepattern = 'temp_precip_rio'

    arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
    ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))

    # Create your test DataArray
    teste = np.ones([46, 36])
    data_xr = xr.DataArray(teste,
                      coords={'longitude': ds.longitude, 'latitude': ds.latitude},
                      dims=["longitude", "latitude"])

    # Create mask for the 5th state (index 4 if zero-based)
    ds2 = shape_clip_dataarray(data_xr, shapefile_br, state_index=t, invert=True, all_touched=True)
    mask = ds2.T    

    for ky in range(0,mask.shape[0]):
        for y in range(0,mask.shape[1]):
            if mask[ky,y] == 1.0:
                mask[ky,y] = np.nan
            else:
                mask[ky,y] = 1.0
    
    ctemp = meantemp*mask
    chumi = meanhumi*mask
       
    tempclas.append(np.nanmean(ctemp))
    humiclas.append(np.nanmean(chumi))


tempclas = np.array(tempclas)
humiclas = np.array(humiclas)

classification_result_temp = classify_distribution(tempclas)
classification_result_casos = classify_distribution(casosmean)
tempcasos = np.char.add(classification_result_temp, classification_result_casos)

print(tempcasos)
