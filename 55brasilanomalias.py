import pandas as pd
import matplotlib.pyplot as plt #if using matplotlib
#import plotly.express as px #if using plotly
import geopandas as gpd
import pyproj
import numpy as np
import mapclassify
import os
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
import datetime as dt
from datetime import timedelta
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
import math
from shapely.geometry import Polygon, MultiPolygon

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


def classify_anomalies(values):
    """
    Classifies each value in the input array into:
    - '0' for nan values
    - '1', '2', '3' for negative values (three equal intervals from min to 0)
    - '4', '5', '6' for positive values (three equal intervals from 0 to max)
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
        
        # Separate negative and positive values
        negative_mask = non_nan_values < 0
        positive_mask = non_nan_values >= 0
        
        # Classify negative values (1, 2, 3)
        if np.any(negative_mask):
            negative_values = non_nan_values[negative_mask]
            
            # Calculate thresholds for negative range (min_value to 0)
            negative_range = 0 - min_value
            neg_threshold1 = min_value + (negative_range / 3)  # 1/3 of negative range
            neg_threshold2 = min_value + (2 * negative_range / 3)  # 2/3 of negative range
            
            # Get indices in the non_nan_values array for negative values
            neg_indices_in_non_nan = np.where(negative_mask)[0]
            
            # Classify negative values
            negative_classification = np.where(negative_values < neg_threshold1, '1',
                                             np.where(negative_values < neg_threshold2, '2', '3'))
            
            # Map back to the original non_nan_mask indices
            non_nan_negative_indices = np.where(non_nan_mask)[0][neg_indices_in_non_nan]
            classification[non_nan_negative_indices] = negative_classification
        
        # Classify positive values (4, 5, 6)
        if np.any(positive_mask):
            positive_values = non_nan_values[positive_mask]
            
            # Calculate thresholds for positive range (0 to max_value)
            positive_range = max_value - 0
            pos_threshold1 = 0 + (positive_range / 3)  # 1/3 of positive range
            pos_threshold2 = 0 + (2 * positive_range / 3)  # 2/3 of positive range
            
            # Get indices in the non_nan_values array for positive values
            pos_indices_in_non_nan = np.where(positive_mask)[0]
            
            # Classify positive values
            positive_classification = np.where(positive_values < pos_threshold1, '4',
                                             np.where(positive_values < pos_threshold2, '5', '6'))
            
            # Map back to the original non_nan_mask indices
            non_nan_positive_indices = np.where(non_nan_mask)[0][pos_indices_in_non_nan]
            classification[non_nan_positive_indices] = positive_classification
    
    # Reshape the classification result to match the original input shape
    return classification.reshape(values.shape)


def get_state_coordinates(shapefile_path, state_index=None):
    """
    Get the latitudes and longitudes of state boundaries from a shapefile.
    Handles both Polygon and MultiPolygon geometries.

    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile containing state boundaries
    state_index : int or None, optional
        If specified, returns coordinates for only this state index

    Returns:
    --------
    dict or list
        - If state_index=None: dict with state indices as keys and lists of coordinate arrays as values
        - If state_index specified: list of coordinate arrays for that state
    """
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    def get_coords(geom):
        """Helper function to get coordinates from either Polygon or MultiPolygon"""
        if isinstance(geom, Polygon):
            return [list(geom.exterior.coords)]
        elif isinstance(geom, MultiPolygon):
            return [list(poly.exterior.coords) for poly in geom.geoms]
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom)}")

    # If state_index is specified, select only that state
    if state_index is not None:
        geometry = shapefile.iloc[state_index].geometry
        return get_coords(geometry)

    # Otherwise, return coordinates for all states
    state_coordinates = {}
    for idx, row in shapefile.iterrows():
        state_coordinates[idx] = get_coords(row.geometry)

    return state_coordinates



def find_nearest_grid_point(lons, lats, target_point):
    """
    Find the closest grid point to a target coordinate.

    Parameters:
    -----------
    lons : numpy.ndarray
        1D or 2D array of longitude values
    lats : numpy.ndarray
        1D or 2D array of latitude values
    target_point : tuple or list or numpy.ndarray
        Target coordinate as (longitude, latitude)

    Returns:
    --------
    tuple
        (closest_lon, closest_lat, index_tuple)
        where index_tuple is (i,) for 1D arrays or (i,j) for 2D arrays
    """
    target_lon, target_lat = target_point
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    # Case 1: Both lons and lats are 1D (regular grid)
    if lons.ndim == 1 and lats.ndim == 1:
        # Calculate all pairwise distances using broadcasting
        lat_diff = np.radians(lats) - np.radians(target_lat)
        lon_diff = np.radians(lons) - np.radians(target_lon)

        # Haversine formula (fixed parentheses)
        a = (np.sin(lat_diff[:, np.newaxis]/2)**2 +
             np.cos(np.radians(target_lat)) * np.cos(np.radians(lats)[:, np.newaxis]) *
             (np.sin(lon_diff/2)**2))
        distances = 6371 * 2 * np.arcsin(np.sqrt(a))

        j, i = np.unravel_index(np.argmin(distances), distances.shape)
        return (lons[i], lats[j], (j, i))

    # Case 2: Both lons and lats are 2D (irregular grid)
    elif lons.ndim == 2 and lats.ndim == 2 and lons.shape == lats.shape:
        # Calculate distances directly for each point
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        target_lat_rad = np.radians(target_lat)
        target_lon_rad = np.radians(target_lon)

        dlat = lat_rad - target_lat_rad
        dlon = lon_rad - target_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(target_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        distances = 6371 * 2 * np.arcsin(np.sqrt(a))

        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        return (lons[i,j], lats[i,j], (i,j))

    else:
        raise ValueError("Input arrays must be either both 1D or both 2D with matching shapes")



figsdir = "/home/numa23/Public/aesop/figures/"
filelocation = '/home/numa23/Public/aesop/era5_land_data/'
shapefile_br="/home/numa23/Public/aesop/shapefile_br_municipios"

# #Lendo os dados do era5-land dewpoint
# counter = 0
# for k in range(2017,2025):
#     for m in range(1,13):
#         rootgrp = Dataset(f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/dewp_rio_{m}{k}.nc','r')
    
#         dims = rootgrp.dimensions
    
#         vars = rootgrp.variables
        
#         attrs = rootgrp.ncattrs
    
#         ndims = len(dims)
    
#         print ('number of dimensions = ' + str(ndims))
        
#         dewp = vars['d2m'][:]
        
    
#         if counter == 0:
        
#             arr1w = dewp
        
#         else:
            
#             arr2w = np.concatenate((arr1w, dewp), axis=0)
            
#             arr1w = arr2w
    
            
#         counter =counter +1

# arr1w = arr1w - 273.15
# dewp = arr1w



#Lendo os dados do era5-land temperatura e precipitacao

datei = dt.datetime(2025,7,9)
datef = dt.datetime(2025,7,16)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


for data in daterange(datei, datef):
    year = str(data.year)
    month = str("{:02d}".format(data.month))
    day = str("{:02d}".format(data.day))
    print(year,month,day)


counter = 0
for data in daterange(datei, datef):
    year = str(data.year)
    month = str("{:02d}".format(data.month))
    day = str("{:02d}".format(data.day))
    rootgrp = Dataset('/home/numa23/Public/aesop/era5_land_data/era5land_' + year + month + day + '.nc','r')

    dims = rootgrp.dimensions

    vars = rootgrp.variables

    attrs = rootgrp.ncattrs

    ndims = len(dims)

    print ('number of dimensions = ' + str(ndims))

    temp = vars['t2m'][:]
    time=vars['valid_time'][:]
    humi = vars['tp'][:]
    dewp = vars['d2m'][:]

    if counter == 0:

        arr1w = temp
        arr1p = humi
        arr1d = dewp
        arr1t = time

    else:

        arr2w = np.concatenate((arr1w, temp), axis=0)
        arr2p = np.concatenate((arr1p, humi), axis=0)
        arr2d = np.concatenate((arr1d, dewp), axis=0)

        arr1w = arr2w
        arr1p = arr2p
        arr1d = arr2d

        arr2t = np.concatenate((arr1t, time), axis=0)

        arr1t = arr2t

    counter =counter + 1



arr1w = arr1w - 273.15
arr1d = arr1d - 273.15
dewp = arr1d
temp = arr1w
humi = arr1p

rh = 100 * np.exp(17.625 * dewp/(243.04 + dewp))/np.exp(17.625 * temp/(243.04 + temp))




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

meantemp = np.nanmean(temp[:],axis=(0))
meanhumi = np.nanmean(humi[:],axis=(0))*24*100
meanrh = np.nanmean(rh[:],axis=(0))

meanhumi = np.ma.masked_values(meanhumi,-99789)
meantemp = np.ma.masked_values(meantemp,-99789)
meanrh = np.ma.masked_values(meanrh,-99789)

filepattern = 'era5'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))

teste = np.ones([411,401])
data_xr = xr.DataArray(teste,
coords={'longitude': ds.longitude,'latitude': ds.latitude},
dims=["longitude", "latitude"])
ds2 = shape_clip_dataarray_total(data_xr, shapefile_br, invert = True, all_touched = False)

mask_total = ds2.T

for k in range(0,mask_total.shape[0]):
    for y in range(0,mask_total.shape[1]):
        if mask_total[k,y] == 1.0:
            mask_total[k,y] = np.nan
        else:
            mask_total[k,y] = 1.0

meantemp = meantemp*mask_total
meanrh = meanrh*mask_total
meanhumi = meanhumi*mask_total

#temp_inteiro = np.array([-5,40])
#rh_inteiro = np.array([25,100])
#humi_inteiro = np.array([0,250])


# Create a figure and axis with a Plate Carree projection
#Plotting
fig=plt.figure(figsize=(8,12))
plt.rcParams.update({"font.size": 16})
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
#ax.set_title('Sea Level Pressure and surface flow at 2005-10-28 18Z')
# ax.set_extent([-51, -45, 6, 2], crs=ccrs.PlateCarree())  #Foz do amazonas
# ax.set_extent([-39, -31, -6, -1], crs=ccrs.PlateCarree()) #Bacia de potiguar
ax.set_extent([-65, -30.5, 7.5, -35], crs=ccrs.PlateCarree())  #Bacia de campos
ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor="white")
#ax.add_feature(cart.feature.STATES, zorder=100, edgecolor='k')


# # add grid ticks
#lontick = np.arange(-75,-32,5) # define longitude ticks  #Bacia de Campos
#lattick = np.arange(-35,9,5) # define latitude ticks  #Bacia de Campos
# lontick = np.arange(-51,-45,0.5) # define longitude ticks  #Foz do amazonas
# lattick = np.arange(2,7,0.5) # define latitude ticks  #Foz do amazonas
# lontick = np.arange(-39,-31,1) # define longitude ticks    #Bacia de potiguar
# lattick = np.arange(-6,-1,1) # define latitude ticks    #Bacia de potiguar
grl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color='k',alpha=0.7,linewidth=1.2)
grl.xlabels_top = False
grl.ylabels_right = False
#grl.xlocator = mticker.FixedLocator(lontick)
#grl.ylocator = mticker.FixedLocator(lattick)

grl.xformatter = LONGITUDE_FORMATTER
grl.yformatter = LATITUDE_FORMATTER

fname = "/home/numa23/Public/aesop/shapefile_br_municipios/BR_Municipios_2022.shp"


shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none',edgecolor="red")
ax.add_feature(shape_feature)



# Plot the bivariate data
cmap = cmocean.cm.speed
plt.pcolormesh(lon, lat ,meantemp,transform = ccrs.PlateCarree(),color='k',cmap=cmap,linewidth=0)
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "teste_shapefile_brasil"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)

#set up the file path and read the shapefile data

fp = "/home/numa23/Public/aesop/shapefile_br_municipios/BR_Municipios_2022.shx"

map_df = gpd.read_file(fp)
map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

map_df = map_df[map_df['CD_MUN'] != '4300002']
map_df = map_df[map_df['CD_MUN'] != '4300001']

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

#datacsv = csv_to_2d_array_pandas('/home/numa23/Public/aesop/tabelafinal_arbov.csv')
datacsv = csv_to_2d_array_pandas('/home/numa23/Public/aesop/tabelafinal.csv')
datacsv = np.array(datacsv)

#casos_inteiro = np.array(datacsv[:,1:], dtype=float)

casos = datacsv[len(datacsv)-1] #pegar ultimo valor
casostotal = datacsv[:,1:]
print(casos.shape)
print(casos)

casos = casos[1:].astype(float)
casostotal = casostotal.astype(float)
casostotalmean = np.nanmean(casostotal[:],axis=(0))
#print(casos[1,2],99999999999999999999999999)


casosmean = casos
casosmean = casosmean - casostotalmean


#see what the map looks like
fig=plt.figure(figsize=(8,12))
map_df.plot(figsize=(20, 10))

nameoffigure = "colorplethmap1_br"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)

print(map_df.head(4840))


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
map_df.plot(color=colors_for_territories, linewidth=1, ax=ax, edgecolor='k')
ax.axis('off')

# Save
plt.savefig(figsdir + "temperature_precipitation_choropleth.png", dpi=300, bbox_inches='tight')
plt.close()

tempclas = []
humiclas = []
rhclas = []
for t in range(0,len(casosmean)+2): # +2 por causa dos dois municipios extras que precisam ser retirados
#for t in range(0,1):
    if t != 4606 and t != 4607:  #essas sao as posicoes dos municipios que precisam ser retirados
   
        #Cria máscaras para cada área
    
        ########### Brasil ###########
    
        filepattern = 'era5'
    
        arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]
        ds = xr.open_dataset(os.path.join(filelocation,arqs[0]))
    
        # Create your test DataArray
        teste = np.ones([411,401])
        data_xr = xr.DataArray(teste,
                          coords={'longitude': ds.longitude, 'latitude': ds.latitude},
                          dims=["longitude", "latitude"])
    
        # Create mask for the 5th state (index 4 if zero-based)
        ds2 = shape_clip_dataarray(data_xr, shapefile_br, state_index=t, invert=True, all_touched=False)
        mask = ds2.T    
    
        for ky in range(0,mask.shape[0]):
            for y in range(0,mask.shape[1]):
                if mask[ky,y] == 1.0:
                    mask[ky,y] = np.nan
                else:
                    mask[ky,y] = 1.0
        
        ctemp = meantemp*mask
        chumi = meanhumi*mask
        crh = meanrh*mask
        
        if math.isnan(np.nanmean(ctemp)) == True:
            ds2 = shape_clip_dataarray(data_xr, shapefile_br, state_index=t, invert=True, all_touched=True)
            mask = ds2.T    
        
            for ky in range(0,mask.shape[0]):
                for y in range(0,mask.shape[1]):
                    if mask[ky,y] == 1.0:
                        mask[ky,y] = np.nan
                    else:
                        mask[ky,y] = 1.0    
            ctemp = meantemp*mask
            crh = meanrh*mask
            chumi = meanhumi*mask
            print("sem dados")
        
        if t == 2066:
            ctemp = meantemp[182,365]
            crh = meanrh[182,365]
            chumi = meanhumi[182,365]
        if t == 2017:
            ctemp = meantemp[185,364]
            crh = meanrh[185,364]
            chumi = meanhumi[185,364]


        tempclas.append(np.nanmean(ctemp))
        humiclas.append(np.nanmean(chumi))
        rhclas.append(np.nanmean(crh))
        
tempclas = np.array(tempclas)
humiclas = np.array(humiclas)
rhclas = np.array(rhclas)
        

with open('/home/numa23/Public/aesop/tempmunicipios2.pickle', 'rb') as handle:
    tempmunicipios = pickle.load(handle)

with open('/home/numa23/Public/aesop/humimunicipios2.pickle', 'rb') as handle:
    humimunicipios = pickle.load(handle)

with open('/home/numa23/Public/aesop/rhmunicipios2.pickle', 'rb') as handle:
    rhmunicipios = pickle.load(handle)


tempmunicipios = np.array(tempmunicipios)
humimunicipios = np.array(humimunicipios)
rhmunicipios = np.array(rhmunicipios)


tempclas = tempclas - tempmunicipios
humiclas = humiclas - humimunicipios
rhclas = rhclas - rhmunicipios
        


classification_result_temp = classify_anomalies(tempclas)
classification_result_casos = classify_anomalies(casosmean)
tempcasos = np.char.add(classification_result_temp, classification_result_casos)

classification_result_humi = classify_anomalies(humiclas)
humicasos = np.char.add(classification_result_humi, classification_result_casos)

classification_result_rh = classify_anomalies(rhclas)
rhcasos = np.char.add(classification_result_rh, classification_result_casos)

print("parando por aqui")

# PLotando a colorbar de temperatura
min_valuetemp = np.nanmin(tempclas)
max_valuetemp = np.nanmax(tempclas)
negative_rangetemp = abs(min_valuetemp)
positive_rangetemp = max_valuetemp 

min_valuecasos = np.nanmin(casosmean)
max_valuecasos = np.nanmax(casosmean)
negative_rangecasos = abs(min_valuecasos)
positive_rangecasos = max_valuecasos

fig = plt.figure()
plt.imshow(rgb_image, extent=[0, 6, 0, 6])  # 5x5 grid
plt.xticks(np.arange(0, 7, 1), labels=[np.round(min_valuecasos,2),np.round(min_valuecasos + (negative_rangecasos / 3),2),np.round(min_valuecasos + (2 * negative_rangecasos / 3),2),0,np.round((positive_rangecasos / 3),2),np.round((2 * positive_rangecasos / 3),2),np.round(positive_rangecasos,2)])  # Label x-axis (casos)
plt.yticks(np.arange(0, 7, 1), labels=[np.round(min_valuetemp,2),np.round(min_valuetemp + (negative_rangetemp / 3),2),np.round(min_valuetemp + (2 * negative_rangetemp / 3),2),0,np.round((positive_rangetemp / 3),2),np.round((2 * positive_rangetemp / 3),2),np.round(positive_rangetemp,2)])  # Label y-axis (temperature, inverted)
plt.xlabel("Casos (?)")
plt.ylabel("Temperatura (C)")
#nameoffigure = "bivariate_colorbar55_temp_arbov_anom"
nameoffigure = "bivariate_colorbar55_temp_ivas_anom"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)


# PLotando a colorbar de umidade relativa
min_valuerh = np.nanmin(rhclas)
max_valuerh = np.nanmax(rhclas)
range_valuesrh = max_valuerh - min_valuerh

min_valuecasos = np.nanmin(casosmean)
max_valuecasos = np.nanmax(casosmean)
range_valuescasos = max_valuecasos - min_valuecasos

fig = plt.figure()
plt.imshow(rgb_image, extent=[0, 5, 0, 5])  # 5x5 grid
plt.xticks(np.arange(1, 5, 1), labels=[np.round(min_valuecasos + (range_valuescasos / 5),2),np.round(min_valuecasos + (2*range_valuescasos / 5),2),np.round(min_valuecasos + 3*(range_valuescasos / 5),2),np.round(min_valuecasos + 4*(range_valuescasos / 5),2)])  # Label x-axis (casos)
plt.yticks(np.arange(1, 5, 1), labels=[np.round(min_valuerh + (range_valuesrh / 5),2), np.round(min_valuerh + (2*range_valuesrh / 5),2),np.round(min_valuerh + 3*(range_valuesrh / 5),2),np.round(min_valuerh + 4*(range_valuesrh / 5),2)])  # Label y-axis (rel humidity, inverted)
plt.xlabel("Casos (?)")
plt.ylabel("Umidade relativa (%)")
#nameoffigure = "bivariate_colorbar55_rh_arbov_anom"
nameoffigure = "bivariate_colorbar55_rh_ivas_anom"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)


# PLotando a colorbar de precipitacao
min_valuehumi = np.nanmin(humiclas)
max_valuehumi = np.nanmax(humiclas)
range_valueshumi = max_valuehumi - min_valuehumi

min_valuecasos = np.nanmin(casosmean)
max_valuecasos = np.nanmax(casosmean)
range_valuescasos = max_valuecasos - min_valuecasos

fig = plt.figure()
plt.imshow(rgb_image, extent=[0, 5, 0, 5])  # 5x5 grid
plt.xticks(np.arange(1, 5, 1), labels=[np.round(min_valuecasos + (range_valuescasos / 5),2),np.round(min_valuecasos + (2*range_valuescasos / 5),2),np.round(min_valuecasos + 3*(range_valuescasos / 5),2),np.round(min_valuecasos + 4*(range_valuescasos / 5),2)])  # Label x-axis (casos)
plt.yticks(np.arange(1, 5, 1), labels=[np.round(min_valuehumi + (range_valueshumi / 5),2), np.round(min_valuehumi + (2*range_valueshumi / 5),2),np.round(min_valuehumi + 3*(range_valueshumi / 5),2),np.round(min_valuehumi + 4*(range_valueshumi / 5),2)])  # Label y-axis (precip, inverted)
plt.xlabel("Casos (?min_valuecasos min_valuecasos )")
plt.ylabel("Precipitacao (mm)")
#nameoffigure = "bivariate_colorbar55_precip_arbov_anom"
nameoffigure = "bivariate_colorbar55_precip_ivas_anom"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)


#Gerando o array de cores que sera usado para plot de temperatura
shape = (len(tempcasos))
colors_blend = np.full(shape, '', dtype=object)
for k in range(0,len(tempcasos)):
    if tempcasos[k] == '11':
        colors_blend[k] = hex_colors2[4,0]
    if tempcasos[k] == '12':
        colors_blend[k] = hex_colors2[4,1]
    if tempcasos[k] == '13':
        colors_blend[k] = hex_colors2[4,2]
    if tempcasos[k] == '14':
        colors_blend[k] = hex_colors2[4,3]
    if tempcasos[k] == '15':
        colors_blend[k] = hex_colors2[4,4]
    if tempcasos[k] == '21':
        colors_blend[k] = hex_colors2[3,0]
    if tempcasos[k] == '22':
        colors_blend[k] = hex_colors2[3,1]
    if tempcasos[k] == '23':
        colors_blend[k] = hex_colors2[3,2]
    if tempcasos[k] == '24':
        colors_blend[k] = hex_colors2[3,3]
    if tempcasos[k] == '25':
        colors_blend[k] = hex_colors2[3,4]
    if tempcasos[k] == '31':
        colors_blend[k] = hex_colors2[2,0]
    if tempcasos[k] == '32':
        colors_blend[k] = hex_colors2[2,1]
    if tempcasos[k] == '33':
        colors_blend[k] = hex_colors2[2,2]
    if tempcasos[k] == '34':
        colors_blend[k] = hex_colors2[2,3]
    if tempcasos[k] == '35':
        colors_blend[k] = hex_colors2[2,4]
    if tempcasos[k] == '41':
        colors_blend[k] = hex_colors2[1,0]
    if tempcasos[k] == '42':
        colors_blend[k] = hex_colors2[1,1]
    if tempcasos[k] == '43':
        colors_blend[k] = hex_colors2[1,2]
    if tempcasos[k] == '44':
        colors_blend[k] = hex_colors2[1,3]
    if tempcasos[k] == '45':
        colors_blend[k] = hex_colors2[1,4]
    if tempcasos[k] == '51':
        colors_blend[k] = hex_colors2[0,0]
    if tempcasos[k] == '52':
        colors_blend[k] = hex_colors2[0,1]
    if tempcasos[k] == '53':
        colors_blend[k] = hex_colors2[0,2]
    if tempcasos[k] == '54':
        colors_blend[k] = hex_colors2[0,3]
    if tempcasos[k] == '55':
        colors_blend[k] = hex_colors2[0,4]
    if '0' in tempcasos[k]:
        colors_blend[k] = '#dc143c'


print(colors_blend)

with open('/home/numa23/Public/aesop/colors_blend_temp.pickle', 'wb') as handle:
    pickle.dump(colors_blend, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/numa23/Public/aesop/tempclas.pickle', 'wb') as handle:
    pickle.dump(tempclas, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/numa23/Public/aesop/classification_result_temp.pickle', 'wb') as handle:
    pickle.dump(classification_result_temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/numa23/Public/aesop/classification_result_casos.pickle', 'wb') as handle:
    pickle.dump(classification_result_casos, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/numa23/Public/aesop/tempcasos.pickle', 'wb') as handle:
    pickle.dump(tempcasos, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/numa23/Public/aesop/hex_colors2.pickle', 'wb') as handle:
    pickle.dump(hex_colors2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot the casos vs temp plot
fig, ax = plt.subplots(figsize=(10, 6))
map_df.plot(color=colors_blend, linewidth=0.1, ax=ax, edgecolor='k')
ax.axis('off')

# Save
#plt.savefig(figsdir + "55casos_temp_brasil_arbov_anom.png", dpi=700, bbox_inches='tight')
plt.savefig(figsdir + "55casos_temp_brasil_ivas_anom.png", dpi=700, bbox_inches='tight')
plt.close()



#Gerando o array de cores que sera usado para plot de umidade relativa
shape = (len(rhcasos))
colors_blend = np.full(shape, '', dtype=object)
for k in range(0,len(rhcasos)):
    if rhcasos[k] == '11':
        colors_blend[k] = hex_colors2[4,0]
    if rhcasos[k] == '12':
        colors_blend[k] = hex_colors2[4,1]
    if rhcasos[k] == '13':
        colors_blend[k] = hex_colors2[4,2]
    if rhcasos[k] == '14':
        colors_blend[k] = hex_colors2[4,3]
    if rhcasos[k] == '15':
        colors_blend[k] = hex_colors2[4,4]
    if rhcasos[k] == '21':
        colors_blend[k] = hex_colors2[3,0]
    if rhcasos[k] == '22':
        colors_blend[k] = hex_colors2[3,1]
    if rhcasos[k] == '23':
        colors_blend[k] = hex_colors2[3,2]
    if rhcasos[k] == '24':
        colors_blend[k] = hex_colors2[3,3]
    if rhcasos[k] == '25':
        colors_blend[k] = hex_colors2[3,4]
    if rhcasos[k] == '31':
        colors_blend[k] = hex_colors2[2,0]
    if rhcasos[k] == '32':
        colors_blend[k] = hex_colors2[2,1]
    if rhcasos[k] == '33':
        colors_blend[k] = hex_colors2[2,2]
    if rhcasos[k] == '34':
        colors_blend[k] = hex_colors2[2,3]
    if rhcasos[k] == '35':
        colors_blend[k] = hex_colors2[2,4]
    if rhcasos[k] == '41':
        colors_blend[k] = hex_colors2[1,0]
    if rhcasos[k] == '42':
        colors_blend[k] = hex_colors2[1,1]
    if rhcasos[k] == '43':
        colors_blend[k] = hex_colors2[1,2]
    if rhcasos[k] == '44':
        colors_blend[k] = hex_colors2[1,3]
    if rhcasos[k] == '45':
        colors_blend[k] = hex_colors2[1,4]
    if rhcasos[k] == '51':
        colors_blend[k] = hex_colors2[0,0]
    if rhcasos[k] == '52':
        colors_blend[k] = hex_colors2[0,1]
    if rhcasos[k] == '53':
        colors_blend[k] = hex_colors2[0,2]
    if rhcasos[k] == '54':
        colors_blend[k] = hex_colors2[0,3]
    if rhcasos[k] == '55':
        colors_blend[k] = hex_colors2[0,4]
    if '0' in rhcasos[k]:
        colors_blend[k] = '#dc143c'


with open('/home/numa23/Public/aesop/colors_blend_rh.pickle', 'wb') as handle:
    pickle.dump(colors_blend, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Plot the casos vs umidade relativa plot
fig, ax = plt.subplots(figsize=(10, 6))
map_df.plot(color=colors_blend, linewidth=0.1, ax=ax, edgecolor='k')
ax.axis('off')

# Save
#plt.savefig(figsdir + "55casos_rh_brasil_arbov_anom.png", dpi=700, bbox_inches='tight')
plt.savefig(figsdir + "55casos_rh_brasil_ivas_anom.png", dpi=700, bbox_inches='tight')
plt.close()



#Gerando o array de cores que sera usado para plot de precipitacao
shape = (len(humicasos))
colors_blend = np.full(shape, '', dtype=object)
for k in range(0,len(humicasos)):
    if humicasos[k] == '11':
        colors_blend[k] = hex_colors2[4,0]
    if humicasos[k] == '12':
        colors_blend[k] = hex_colors2[4,1]
    if humicasos[k] == '13':
        colors_blend[k] = hex_colors2[4,2]
    if humicasos[k] == '14':
        colors_blend[k] = hex_colors2[4,3]
    if humicasos[k] == '15':
        colors_blend[k] = hex_colors2[4,4]
    if humicasos[k] == '21':
        colors_blend[k] = hex_colors2[3,0]
    if humicasos[k] == '22':
        colors_blend[k] = hex_colors2[3,1]
    if humicasos[k] == '23':
        colors_blend[k] = hex_colors2[3,2]
    if humicasos[k] == '24':
        colors_blend[k] = hex_colors2[3,3]
    if humicasos[k] == '25':
        colors_blend[k] = hex_colors2[3,4]
    if humicasos[k] == '31':
        colors_blend[k] = hex_colors2[2,0]
    if humicasos[k] == '32':
        colors_blend[k] = hex_colors2[2,1]
    if humicasos[k] == '33':
        colors_blend[k] = hex_colors2[2,2]
    if humicasos[k] == '34':
        colors_blend[k] = hex_colors2[2,3]
    if humicasos[k] == '35':
        colors_blend[k] = hex_colors2[2,4]
    if humicasos[k] == '41':
        colors_blend[k] = hex_colors2[1,0]
    if humicasos[k] == '42':
        colors_blend[k] = hex_colors2[1,1]
    if humicasos[k] == '43':
        colors_blend[k] = hex_colors2[1,2]
    if humicasos[k] == '44':
        colors_blend[k] = hex_colors2[1,3]
    if humicasos[k] == '45':
        colors_blend[k] = hex_colors2[1,4]
    if humicasos[k] == '51':
        colors_blend[k] = hex_colors2[0,0]
    if humicasos[k] == '52':
        colors_blend[k] = hex_colors2[0,1]
    if humicasos[k] == '53':
        colors_blend[k] = hex_colors2[0,2]
    if humicasos[k] == '54':
        colors_blend[k] = hex_colors2[0,3]
    if humicasos[k] == '55':
        colors_blend[k] = hex_colors2[0,4]
    if '0' in humicasos[k]:
        colors_blend[k] = '#dc143c'


with open('/home/numa23/Public/aesop/colors_blend_precip.pickle', 'wb') as handle:
    pickle.dump(colors_blend, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot the casos vs precip
fig, ax = plt.subplots(figsize=(10, 6))
map_df.plot(color=colors_blend, linewidth=0.1, ax=ax, edgecolor='k')
ax.axis('off')

# Save
#plt.savefig(figsdir + "55casos_precip_brasil_arbov_anom.png", dpi=700, bbox_inches='tight')
plt.savefig(figsdir + "55casos_precip_brasil_ivas_anom.png", dpi=700, bbox_inches='tight')
plt.close()

print("acabou")
