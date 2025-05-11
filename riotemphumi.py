#Realiza as importações e define os diretórios dos dados e das figuras

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

figsdir = "/home/numa23/Public/aesop/figures/"
filelocation = '/home/numa23/Public/Projeto_BR_OTEC/copernicus/br7/'
shapefile_br="/home/numa23/Public/Projeto_BR_OTEC/copernicus/shapefile_rj"

#Lendo a landuse file

with rasterio.open('/home/numa23/Public/aesop/brasil_coverage_2023.tif') as src:
    chunk_size = 30000  # pixels
    
    # Get the coordinate reference system
    crs = src.crs
    
    # Iterate through the image in chunks
    for i in range(100000, 130000,chunk_size):
        for j in range(100000, 130000,chunk_size):
            # Define window to read
            window = Window(j, i, 
                          min(chunk_size, src.width - j), 
                          min(chunk_size, src.height - i))
            
            # Read the land-use data
            landuse_array = src.read(1, window=window)
            rows, cols = landuse_array.shape

            # Create coordinate arrays using vectorized operations
            row_indices, col_indices = np.indices((rows, cols))
            global_rows = window.row_off + row_indices
            global_cols = window.col_off + col_indices
            
            # Get coordinates in original CRS
            x_coords, y_coords = src.xy(global_rows, global_cols)
            
            # Transform to lat/lon if needed
            if not crs.is_geographic:
                from rasterio.warp import transform
                lons, lats = transform(
                    src.crs, 'EPSG:4326', 
                    x_coords.flatten(), y_coords.flatten()
                )
                coord_array = np.dstack((
                    np.array(lats).reshape(rows, cols),
                    np.array(lons).reshape(rows, cols)
                ))
            else:
                coord_array = np.dstack((y_coords, x_coords))
            
            # Print comprehensive information for this chunk
            print(f"\n=== Chunk at window position ({i},{j}) ===")
            print(f"Pixel dimensions: {rows} rows × {cols} columns")
            print(f"Coordinate array shape: {coord_array.shape}")
            
            # Top-left coordinates
            tl_lat, tl_lon = coord_array[0, 0]
            print(f"\nTop-Left Coordinates: {tl_lat:.6f}°N, {tl_lon:.6f}°E")
            
            # Bottom-right coordinates
            br_lat, br_lon = coord_array[-1, -1]
            print(f"Bottom-Right Coordinates: {br_lat:.6f}°N, {br_lon:.6f}°E")
            
            # Center coordinates (calculated from corners)
            center_lat = (tl_lat + br_lat) / 2
            center_lon = (tl_lon + br_lon) / 2
            print(f"Approx Center: {center_lat:.6f}°N, {center_lon:.6f}°E")
            
            # Land-use information
            unique_values = np.unique(landuse_array)
            print(f"\nLand-use values present: {unique_values}")
            print(f"Dominant land-use: {np.bincount(landuse_array.flatten()).argmax()}")
            print(f"Land-use array shape: {landuse_array.shape}")
            
            # Optional: Save both arrays for this chunk
            np.savez(f'chunk_{i}_{j}.npz',
                    coords=coord_array,
                    landuse=landuse_array)
            
            print("=" * 60)

latitude = []
longitude = []
for k in coord_array:
    latitude.append(k[0,0])

for k in coord_array[0]:
    longitude.append(k[1])
latitude = np.array(latitude)
longitude = np.array(longitude)
print(latitude,"laaaaaaaaaaaaaaaaat",latitude.shape)
print(longitude,"looooooooooooooon",longitude.shape)
print(landuse_array.shape,"laaaaaanduse")

landuse_array = np.where(landuse_array == 24, landuse_array, np.nan)
print(landuse_array.shape,"laaaaaanduse")

locals_with_city = []
for k in range(0,len(landuse_array)):
    for t in range(0,len(landuse_array[0])):
        if landuse_array[k,t] == 24:
            locals_with_city.append([latitude[k],longitude[t]])




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


weekmeantemp = []
weekmeanhumi = []
for k in range(0,len(humi),(7*24)):
    mtemp = np.nanmean(temp[k:k+(7*24)],axis=(0))
    mhumi = np.nanmean(humi[k:k+(7*24)],axis=(0))
    weekmeantemp.append(mtemp)
    weekmeanhumi.append(mhumi)

weekmeanhumi = np.ma.masked_values(weekmeanhumi,-99789)
weekmeantemp = np.ma.masked_values(weekmeantemp,-99789)

weekmeanhumi = weekmeanhumi[51:]
weekmeantemp = weekmeantemp[51:]
print(weekmeantemp.shape)

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

weekmeantemp = weekmeantemp*mask_total
weekmeanhumi = weekmeanhumi*mask_total


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
plt.pcolormesh(lon, lat ,weekmeantemp[0],transform = ccrs.PlateCarree(),color='k',cmap=cmap)
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "teste_shapefile_rio"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)

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

print(casos.shape,weekmeanhumi.shape)
print(np.nanmin(weekmeanhumi),np.nanmax(weekmeanhumi),'fefefefefefefef')
print(np.nanmin(weekmeantemp),np.nanmax(weekmeantemp),'rgrgrgrgrgrgrgrg')

pozzi = classify_distribution2(weekmeanhumi,weekmeanhumi)
pozzi2 = classify_distribution(weekmeanhumi)

print(np.array_equal(pozzi, pozzi2))

#Criando um array que mostra quantidade de ponto com landusetype urban infrastructure a cada ponto de lat/lon do era5 land

#arrcidade = np.ones_like(weekmeantemp[0])*0.0
#for k in range(0,len(arrcidade)):
#    for t in range(0,len(arrcidade[0])):
#        for x in locals_with_city:
#            if abs(lat[k] - x[0]) < 0.05 and abs(lon[t] - x[1]) < 0.05:
#                arrcidade[k,t] = arrcidade[k,t] + 1
#                #print("foi")
#            #else:
#                #print("nao foi")
#
#print(arrcidade.shape,"arrcidaedeeeeeeeeeee")
#
#
#with open('/home/numa23/Public/aesop/arrcidade.pickle', 'wb') as handle:
#    pickle.dump(arrcidade, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/numa23/Public/aesop/arrcidade.pickle','rb') as handle:
    arrcidade = pickle.load(handle)


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
plt.pcolormesh(lon, lat ,arrcidade,transform = ccrs.PlateCarree(),color='k',edgecolor='none')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "teste_cidade_era5"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)



st1 = []
st2 = []
total1 = []
counter = 0
classificacao_casos = np.zeros((56, 1))
norm_class = np.zeros((56, 1))
#for k in range(0,len(casos)):
for k in range(0,1):    
    #for t in range(0,len(casos[0])-1):
    for t in range(0,2):
       
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
        
        cweektemp = weekmeantemp[k]*mask
        cweekhumi = weekmeanhumi[k]*mask
        
        classification_result_temp = classify_distribution2(weekmeantemp,cweektemp)
        classification_result_humi = classify_distribution2(weekmeanhumi,cweekhumi)
        temphumi = np.char.add(classification_result_temp, classification_result_humi)
        
        if non_zero_elements_equal(temphumi.astype(int)) == True:
            st1.append(np.nanmax(temphumi.astype(int)))
            st1.append(np.nanmin(temphumi.astype(int)))
        if non_zero_elements_equal(temphumi.astype(int)) == False:
            st2.append(np.nanmax(temphumi.astype(int)))
            st2.append(np.nanmin(temphumi.astype(int)))

        for kk in temphumi:
            for tt in kk:
                counter = counter + int(tt)
                total1.append(int(tt)) 
                if tt == "11":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '12':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "13":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '14':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "15":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "21":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '22':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "23":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '24':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "25":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "31":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '32':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "33":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '34':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "35":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "41":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '42':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "43":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '44':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "45":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "51":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '52':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "53":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == '54':
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1
                if tt == "55":
                    classificacao_casos[int(tt)] = classificacao_casos[int(tt)] + float(casos[k,t+1])
                    norm_class[int(tt)] = norm_class[int(tt)] + 1

print(classificacao_casos/norm_class)
print(np.nanmin(classificacao_casos),np.nanmax(classificacao_casos))
#for k in temphumi:
#    for t in k:
#        print(t)
print(counter,"counter")
print(np.nanmin(np.array(total1)),np.nanmax(np.array(total1)))
print("st1",st1)
print("st2",st2)

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
plt.pcolormesh(lon, lat ,cweektemp,transform = ccrs.PlateCarree(),color='k',cmap=cmap)
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "teste_shapefile_rio2"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)
