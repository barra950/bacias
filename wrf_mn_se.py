import pandas as pd
import numpy as np
import os
import xarray as xr
import sys,warnings,os
import numpy as np
from netCDF4 import Dataset,num2date
from datetime import datetime
import datetime as dt
from datetime import timedelta
from scipy import stats
import numpy.ma
import pickle
from math import sin, cos, sqrt, atan2, radians
import math
from wrf import getvar, extract_times, ALL_TIMES, destagger

filelocation = '/mnt/Destino2/wrfout/OTEC_hindcast/margem_norte/'
filepattern = 'wrfout_OTECmn-d02-2022-'
filepattern2 = 'wrfout_OTECmn-d02-2022-10'

#filelocation = '/mnt/Destino2/wrfout/OTEC_hindcast/sudeste/'
#filepattern = 'wrfout_OTECse-d02-2022-'
#filepattern2 = 'wrfout_OTECse-d02-2022-10'

arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern)]

#arqs = [f for f in sorted(os.listdir(filelocation)) if f.startswith(filepattern) or f.startswith(filepattern2)]

counter = 0
for arq in arqs:
    print(arq)
    

    rootgrp = Dataset('/mnt/Destino2/wrfout/OTEC_hindcast/margem_norte/' + arq,'r')

    dims = rootgrp.dimensions

    vars = rootgrp.variables

    attrs = rootgrp.ncattrs

    ndims = len(dims)

    print ('number of dimensions = ' + str(ndims))


    # Get the START_DATE attribute
    start_date_str = getattr(rootgrp, 'START_DATE', 'Not found')
    #print(f"START_DATE string: {start_date_str}")

    # Convert to datetime object
    if start_date_str != 'Not found':
        start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d_%H:%M:%S')
        print(f"START_DATE datetime: {start_date_dt}")
    else:
        start_date_dt = None
        print("START_DATE not found")

    time_minutes = vars['XTIME'][12:]  # XTIME in minutes
    
    # Convert XTIME (minutes) to datetime array
    if start_date_dt is not None:
        # Convert minutes to timedelta and add to start_date_dt
        time_datetime = [start_date_dt + timedelta(minutes=float(minutes)) 
                        for minutes in time_minutes]
        
        # Convert to numpy array for easier handling
        time_datetime = np.array(time_datetime)
    time = time_datetime
    print("indoooo")
    
    wspd = getvar(rootgrp, "wspd", timeidx=ALL_TIMES, units="m/s", meta=False)[12:,3,:,:]
    
#    # Get staggered U and V
#    u_stag = vars['U'][12:, 2, :, :]
#    v_stag = vars['V'][12:, 2, :, :]
#
#    # Destagger to mass points
#    u_destag = destagger(u_stag, stagger_dim=1)  # stagger_dim=1 for x-stagger (west-east)
#    v_destag = destagger(v_stag, stagger_dim=0)  # stagger_dim=0 for y-stagger (south-north)
#
#    # Now calculate windspeed
#    wspd = (u_destag**2 + v_destag**2)**0.5

    swdown = vars['SWDOWN'][12:]
    #time=vars['XTIME'][12:]
    #u10 = vars['U10'][12:]
    #v10 = vars['V10'][12:]
    #wspd = (u10**2 + v10**2)**(0.5)
    if counter == 0:

        arr1w = swdown
        arr1p = wspd
        #arr1d = dewp
        arr1t = time

    else:

        arr2w = np.concatenate((arr1w, swdown), axis=0)
        arr2p = np.concatenate((arr1p, wspd), axis=0)
        #arr2d = np.concatenate((arr1d, dewp), axis=0)

        arr1w = arr2w
        arr1p = arr2p
        #arr1d = arr2d

        arr2t = np.concatenate((arr1t, time), axis=0)

        arr1t = arr2t

    counter =counter + 1


#removing the last 13 times to avoid getting data from 2023 (works for both regions, but only to be used if the last file has data from 2023. Otherwise comment)
arr1w = arr1w[:-13]
arr1p = arr1p[:-13]
arr1t = arr1t[:-13]

print(arr1t)
print(arr1p)


def create_month_mask(arr1t, target_month, arr1w_shape):
    """
    Create a mask array with same shape as arr1w where values are 1 for a specific month, NaN otherwise.
    
    Parameters:
    -----------
    arr1t : numpy.ndarray
        Array of datetime objects with shape (time,)
    target_month : int
        Target month (1=January, 2=February, ..., 12=December)
    arr1w_shape : tuple
        Shape of the target array (time, lat, lon)
    
    Returns:
    --------
    mask_array : numpy.ndarray
        Array with shape (time, lat, lon) where values are 1 for target month, NaN otherwise
    """
    # Extract months from datetime array
    months = np.array([t.month for t in arr1t])
    
    # Create boolean mask for target month
    month_mask = (months == target_month)
    
    # Initialize output array with NaNs
    mask_array = np.full(arr1w_shape, np.nan)
    
    # Set values to 1 where month matches target month
    # We need to broadcast the time dimension mask to the spatial dimensions
    for i in range(arr1w_shape[0]):
        if month_mask[i]:
            mask_array[i, :, :] = 1
    
    return mask_array

#test_mask = create_month_mask(arr1t, 4, arr1w.shape)

from netCDF4 import Dataset
import numpy as np
lat1 = vars["XLAT"][0, :, 0]
lon1 = vars["XLONG"][0, 0, :]
landmask = vars['LANDMASK'][0,:,:]

#Tem que deletar or arquivos .nc que comecam com msm nome antes pra n dar error de permissao
for mnth in range(1,13): #for all months
    
    # Create a new dataset.
    nome_dado = os.path.join('/home/numa23/Public/Projeto_BR_OTEC/copernicus/dados_wrf/media_' + str(mnth) + '_mn.nc')
    dataset = Dataset(nome_dado, 'w', format='NETCDF4')
    
    # Create dimensions.
    #time = dataset.createDimension('time', None) # None is for unlimited dimension
    latitude = dataset.createDimension('latitude', None)
    longitude = dataset.createDimension('longitude', None)
    solarfinal = dataset.createDimension('solarfinal', None)
    wspdfinal = dataset.createDimension('wspdfinal', None)
    solarfullmean = dataset.createDimension('solarfullmean', None)
    wspdfullmean = dataset.createDimension('wspdfullmean', None)
    lndmask = dataset.createDimension('lndmask', None)
    
    # Create variables.
    latitudes = dataset.createVariable('latitude', 'f4', ('latitude',))
    longitudes = dataset.createVariable('longitude', 'f4', ('longitude',))
    solarfinal = dataset.createVariable('solarfinal', 'f8', ('latitude', 'longitude'))
    wspdfinal = dataset.createVariable('wspdfinal', 'f8', ('latitude', 'longitude'))
    solarfullmean = dataset.createVariable('solarfullmean', 'f8', ('latitude', 'longitude'))
    wspdfullmean = dataset.createVariable('wspdfullmean', 'f8', ('latitude', 'longitude'))
    lndmask = dataset.createVariable('lndmask', 'f8', ('latitude', 'longitude'))
    
    # Assign data to variables.
    latitudes[:] = lat1
    longitudes[:] = lon1
    
    test_mask = create_month_mask(arr1t, mnth, arr1w.shape)
    solarfinal[:] = np.nanmean((arr1w*test_mask),axis=0)
    
    test_mask = create_month_mask(arr1t, mnth, arr1p.shape)
    wspdfinal[:] = np.nanmean((arr1p*test_mask),axis=0)

    solarfullmean[:] = np.nanmean(arr1w,axis=0)

    wspdfullmean[:] = np.nanmean(arr1p,axis=0)

    lndmask[:] = landmask
    # Add global attributes.
    dataset.description = "Example NetCDF file created with Python."
    dataset.author = "Your Name"
    
    # Close the dataset.
    dataset.close()

print("acabou")
