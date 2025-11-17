import pandas as pd
import matplotlib.pyplot as plt #if using matplotlib
#import plotly.express as px #if using plotly
import geopandas as gpd
import pyproj
import numpy as np
##import mapclassify
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
from datetime import datetime
import datetime as dt
from datetime import timedelta
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
from scipy import stats
import matplotlib.ticker as mticker
import numpy.ma
import pandas as pd
import pickle
import geopy.distance
from math import sin, cos, sqrt, atan2, radians
import math
from shapely.geometry import Polygon, MultiPolygon
from netCDF4 import Dataset,num2date


##import geopandas as gpd
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



filelocation = '/disco3/matheus/era5/ERA5_land/'
shapefile_br="/home/numa19/matheus/shapefile_br_municipios"




#Lendo os dados do era5-land temperatura e precipitacao

filepattern = '.nc'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.endswith(filepattern)]

# Open all files at once (more efficient)
ds = xr.open_mfdataset([os.path.join(filelocation, arq) for arq in arqs],combine="nested",concat_dim='step')

dstemp = ds.t2m - 273.15
dsdewp = ds.d2m - 273.15
dshumi = ds.tp
dsrh =100 * np.exp(17.625 * (dsdewp)/(243.04 + (dsdewp)))/np.exp(17.625 * (dstemp)/(243.04 + (dstemp)))


stdtemp = dstemp.std(dim='step')

stdhumi = (dshumi*24*100*365).std(dim='step')
stdrh = dsrh.std(dim='step')


lon=ds.longitude
lat=ds.latitude

# # Get the time information for the file.
# timeUnits = vars['valid_time'].units

# # Make dates for the file.
# Date = num2date(arr1t,timeUnits,calendar='standard')
# Day = np.asarray([d.day for d in Date])
# Month = np.asarray([d.month for d in Date])
# Year = np.asarray([d.year for d in Date])

# print(Date[0],Date[len(Date)-1])

# meantemp = np.nanmean(temp[:],axis=(0))
# meanhumi = np.nanmean(humi[:],axis=(0))*24*100*365
# meanrh = np.nanmean(rh[:],axis=(0))

stdhumi = np.ma.masked_values(stdhumi,-99789)
stdtemp = np.ma.masked_values(stdtemp,-99789)
stdrh = np.ma.masked_values(stdrh,-99789)


print(stdrh.shape,np.nanmean(stdtemp),np.nanmean(stdrh),np.nanmean(stdhumi))



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

stdtemp = stdtemp*mask_total
stdrh = stdrh*mask_total
stdhumi = stdhumi*mask_total





tempclas = []
humiclas = []
rhclas = []
for t in range(0,5570+2): # +2 por causa dos dois municipios extras que precisam ser retirados. 5570 é o numero de municipios
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
        
        ctemp = stdtemp*mask
        chumi = stdhumi*mask
        crh = stdrh*mask
        
        if math.isnan(np.nanmean(ctemp)) == True:
            ds2 = shape_clip_dataarray(data_xr, shapefile_br, state_index=t, invert=True, all_touched=True)
            mask = ds2.T    
        
            for ky in range(0,mask.shape[0]):
                for y in range(0,mask.shape[1]):
                    if mask[ky,y] == 1.0:
                        mask[ky,y] = np.nan
                    else:
                        mask[ky,y] = 1.0    
            ctemp = stdtemp*mask
            crh = stdrh*mask
            chumi = stdhumi*mask
            print("sem dados")
        
        if t == 2066:
            ctemp = stdtemp[182,365]
            crh = stdrh[182,365]
            chumi = stdhumi[182,365]
        if t == 2017:
            ctemp = stdtemp[185,364]
            crh = stdrh[185,364]
            chumi = stdhumi[185,364]


        tempclas.append(np.nanmean(ctemp))
        humiclas.append(np.nanmean(chumi))
        rhclas.append(np.nanmean(crh))
        
        

print(np.nanmean(humiclas),np.nanmean(tempclas),np.nanmean(rhclas))



with open('/home/numa19/matheus/aesop/stdtempmunicipios.pickle', 'wb') as handle:
    pickle.dump(tempclas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/home/numa19/matheus/aesop/stdhumimunicipios.pickle', 'wb') as handle:
    pickle.dump(humiclas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/home/numa19/matheus/aesop/stdrhmunicipios.pickle', 'wb') as handle:
    pickle.dump(rhclas, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("acabou")
