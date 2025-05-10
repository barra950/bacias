import rasterio
from rasterio.windows import Window
import numpy as np
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

figsdir = "/home/numa23/Public/aesop/figures/"

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

#Replace all values in the landuse array that are not 24 with nan
landuse_array = np.where(landuse_array == 24, landuse_array, np.nan)
print(np.nanmin(landuse_array),np.nanmax(landuse_array))

# Plot the bivariate data
cmap = cmocean.cm.speed
plt.pcolormesh(longitude[::10], latitude[::10] ,landuse_array[::10,::10],transform = ccrs.PlateCarree(),color='k',edgecolor='none')
plt.subplots_adjust(bottom=0.05, top=0.96, hspace=0.4,right=0.99,left=0.01)
nameoffigure = "teste_raster"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)
