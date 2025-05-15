# -*- coding: utf-8 -*-
# AUTOR: Douglas Medeiros Nehme
# CONTACT: medeiros.douglas3@gmail.com
# CRIATION: apr/2024
# OBJECTIVE: Calculate 150m wind turbulence
#            intensity return level using ERA5
#            data for the three areas of CNOOC
#            Project using the cumulative
#            distribuition function method

import os
import sys
import metpy
import pickle

import numpy as np
import xesmf as xe
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from scipy.stats import norm as normal
from scipy.stats import genextreme as gev
from matplotlib.ticker import ScalarFormatter

sys.path.insert(
    0, os.path.expanduser('/home/numa23/Public/Projeto_BR_OTEC/copernicus/airsea')
)

import airsea

###############################################
rootdir = (
    # '/home/douglas/Dropbox/profissional/LAMCE/cnooc'
    '/home/numa23/Public/Projeto_BR_OTEC/copernicus'
)

areas = ['mn', 'campos', 'santos']
var   = 'wti'
varname = 'Wind Turbulence Intensity'
unidade = ''

# in percentage
minimum_of_valid_data = 80

cardinal_info = {
    'cardinal': {
        'level': 1,
        'dirs' : [
        'N', 'E', 'S', 'W'
    ]},
    'intercardinal': {
        'level': 2,
        'dirs' :  [
        'N', 'NE', 'E', 'SE',
        'S', 'SW', 'W', 'NW'
    ]},
    'sec-intercardinal': {
        'level': 3,
        'dirs' : [
        'N', 'NNE', 'NE', 'ENE',
        'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW',
        'W', 'WNW', 'NW', 'NNW'
    ]}
}

card_level, card_dirs = cardinal_info[
    'intercardinal'
].values()

col_names = [
    'dir_count_total', 'dir_percent_total',
    'valid_count_yearly', 'valid_percent_yearly',
    'mean', 'std',
    1.01, 10, 20, 30, 50, 100
    ]

tabela = pd.DataFrame(
    columns=col_names,
    index=card_dirs,
    data=np.nan
)
statistics_by_dir = {
    'mn': tabela.copy(),
    'campos': tabela.copy(),
    'santos' : tabela.copy()
}

for area in areas:
# for area in ['SE']:
    print(
        f'GEV Return Level of {var} for {area}'
    )
    ds = xr.open_mfdataset(
        f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br/{area}_*.nc',
        combine='nested',
        concat_dim='time'
    )
    

    ds['wspd100'], _ = airsea.cart2pol_wind(
        ds.u100,
        ds.v100,
        rnd=3,
        blowing_from=True
    )

    z = 150
    zref = 100
    roughness = ds.fsr.as_numpy()

    ds['wspd150'] = ds['wspd100'] * (np.log(z/roughness) / np.log(zref/roughness))

    # hourly = xr.merge([
    #     wspd150,
    #     ds.wdir100
    # ])
    # hourly = hourly.rename({
    #     'wspd100': 'wspd150'
    # })

    if area == 'mn':
        # i1, j1 = 14, 26
        i1, j1 = -0.25, -44.25
    if area == 'campos':
        # i1, j1 = 9, 10
        i1, j1 = -22.75, -40.25
    if area == 'santos':
        # i1, j1 = 6, 18
        i1, j1 = -23.75, -42.0

    hourly = ds.sel(
        latitude=i1,
        longitude=j1,
        method='nearest'
    )

    # Estimating Wind Turbulence Intensity
    # Turk & Emeis (2010) utilizam a razão
    # entre o desvio padrão e a média da
    # velocidade do vento medida a cada 10min
    # para estimar a turbulência do vento.
    # Porém, como aqui estou trabalhando com
    # valores extremos, logo que não seguem
    # uma distribuição normal, fiquei em
    # dúvida se deveria adaptar a estimativa
    # e usar a mediana.
    # Depois de pensar melhor, percebi que
    # não faz sentido usar a mediana nessa
    # parte do código, porque ainda não estou
    # tratando de valores extremos, visto que
    # isso só ocorrerá a partir do momento em
    # que fizer o resample para valores anuais,
    # notadamente a partir do momento em que
    # trabalhar com a variável yearly
    wti_daily = (
        hourly.wspd150.resample(time='D').std() /
        hourly.wspd150.resample(time='D').mean()
    )
    wti_daily = xr.merge([
        wti_daily,
        hourly.u100.resample(time='D').mean(),
        hourly.v100.resample(time='D').mean()
    ])
    wti_daily = wti_daily.rename({
        'wspd150': 'wti150'
    })
    
    _, wti_daily['wdir100'] = airsea.cart2pol_wind(
        wti_daily.u100,
        wti_daily.v100,
        rnd=3,
        blowing_from=True
    )
    wcard = metpy.calc.angle_to_direction(
        wti_daily['wdir100'] * metpy.units.units.deg,
        level=card_level
    )
    wti_daily['wcard'] = xr.DataArray(
        data=wcard,
        dims='time'
    )

    for direction in card_dirs:

        try:
            print(
                f'....Working on {direction} dir'
            )

            wti_daily_card = wti_daily.where(
                wti_daily.wcard == direction
            ).wti150

            yearly = wti_daily_card.resample(
                time='1Y'
            ).max()

            mean = np.round(
                np.mean(wti_daily_card), 3
            )
            std = np.round(
                wti_daily_card.std(), 3
            )
            dir_count_total = wti_daily_card.count().values

            dir_percent_total = (
                dir_count_total / wti_daily.wti150.size
            ) * 100

            statistics_by_dir[area].loc[
                direction, 'mean'
            ] = mean

            statistics_by_dir[area].loc[
                direction, 'std'
            ] = std

            statistics_by_dir[area].loc[
                direction, 'dir_count_total'
            ] = dir_count_total

            statistics_by_dir[area].loc[
                direction, 'dir_percent_total'
            ] = np.round(dir_percent_total, 1)

            ###############################################
            valid_count_yearly = yearly.count().values

            valid_percent_yearly = (
                valid_count_yearly / yearly.size
            ) * 100
            
            valid_percent_yearly = np.round(
                valid_percent_yearly, 0
            )

            statistics_by_dir[area].loc[
                direction, 'valid_count_yearly'
            ] = valid_count_yearly

            statistics_by_dir[area].loc[
                direction, 'valid_percent_yearly'
            ] = valid_percent_yearly

            if any(yearly.isnull()):
                if valid_percent_yearly >= minimum_of_valid_data:
                    yearly = yearly.fillna(
                        yearly.median()
                    )
                
            data = yearly

            data.name = 'wti150'
            data = data.round(3)
            
            ###############################################
            # Bootstraping to aloow calculate confidence
            # intervals

            # Step is 1/12 to force the calculations
            # consider monthly steps. So, we have the periods
            # var with a shape of 2388, which represents 12
            # values for each year, from year 1 to year 200
            step = 1/12.
            periods = np.arange(1.01, 200 + step, step)

            nsamples = 1000
            # generate 1000 samples by resampling original
            # data with replacement
            params = []
            for i in range(nsamples):
                params.append(
                    gev.fit(
                        np.random.choice(
                            yearly,
                            size=yearly.size,
                            replace=True
                )))
            # calculate return levels for each of the 1000 samples
            levels = []
            for i in range(nsamples):
                levels.append(
                    gev.ppf(
                        1 - 1 / periods,
                        *params[i]
                ))
            levels = np.array(levels)

            for prd in tabela.columns[6:]:
                statistics_by_dir[area].loc[direction, prd] = np.round(
                    np.median(levels[:, np.absolute(periods - prd).argmin()]),
                    3
                )

        except ValueError:
            print(
                f"....we've only {valid_percent_yearly}% of valid data, when the minimum is {minimum_of_valid_data}%"
            )



filepath = os.path.join(
    rootdir,
    'statistics',
    f'statistics_by_dir_{var}.pkl'
)

with open(filepath, 'wb') as handle:
    pickle.dump(
        statistics_by_dir,
        handle
    )
handle.close()
