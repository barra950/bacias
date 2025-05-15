# -*- coding: utf-8 -*-
# AUTOR: Douglas Medeiros Nehme
# CONTACT: medeiros.douglas3@gmail.com
# CRIATION: apr/2024
# OBJECTIVE: Calculate wave peak period
#            return level using ERA5 data for
#            the three areas of CNOOC Project
#            using the cumulative distribuition
#            function method

import os
import metpy
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from copy import deepcopy as dcopy
from scipy.stats import genextreme as gev

###############################################
rootdir = (
    # '/home/douglas/Dropbox/profissional/LAMCE/cnooc'
    '/home/numa23/Public/Projeto_BR_OTEC/copernicus'
)

areas = ['mn', 'campos', 'santos']
var   = 'hs_tp'
# varname = 'Peak Wave Period'
# unidade = 's'

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
    # 'cardinal'
].values()

sea_states = ['Ambos', 'Unimodal', 'Bimodal']

columns_names = [
    'dir_count_total', 'dir_percent_total',
    'valid_count_yearly', 'valid_percent_yearly',    
    'mean', 'std',
    1.01, 10, 20, 30, 50, 100
]

counts_hs_tp = {
    'mn': dict(zip(sea_states, [dict(), dict(), dict()])),
    'campos': dict(zip(sea_states, [dict(), dict(), dict()])),
    'santos' : dict(zip(sea_states, [dict(), dict(), dict()]))
}

tp_bins_all = np.arange(2, 24, 1)

hs_bins = {
    'mn': np.arange(0, 4.5, 0.5),
    'campos': np.arange(0, 6.5, 0.5),
    'santos': np.arange(0, 10.0, 0.5),
}

tabela_stats = pd.DataFrame(
    data=np.nan,
    columns=columns_names,
    index=tp_bins_all,
)

# Inside each area key (mn, campos, santos) we have
# a dataframe for each card_dirs value
dictdir = dict([
    (key, tabela_stats.copy()) for key in card_dirs
])
statistics_by_dir = {
    'mn': dict(zip(sea_states, [dcopy(dictdir), dcopy(dictdir), dcopy(dictdir)])),
    'campos': dict(zip(sea_states, [dcopy(dictdir), dcopy(dictdir), dcopy(dictdir)])),
    'santos' : dict(zip(sea_states, [dcopy(dictdir), dcopy(dictdir), dcopy(dictdir)]))
}

for area in areas:

    print(
        f'GEV Return Level of {var} for {area}'
    )
    ds1 = xr.open_mfdataset(
        f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br4/waves_{area}_*.nc',
        combine='nested',
        concat_dim='valid_time'
    )


    ds2 = xr.open_mfdataset(
        f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br4/{area}_*_partitioned_directions.nc',
        combine='nested',
        concat_dim='valid_time'
    )
    

    ds1 = ds1.drop_vars(
        ['number', 'expver', 'tmax', 'hmax', 'wmb']
    )
    ds2 = ds2.drop_vars(
        ['number', 'expver']
    )

    ds = xr.merge(
        [ds1, ds2]
    )
    del ds1, ds2

    if area == 'mn':
        i1, j1 = -0.25, -44.25
    if area == 'campos':
        i1, j1 = -22.75, -40.25
    if area == 'santos':
        # i1, j1 = -30, -49
        i1, j1 = -23.75, -42.0
    
    hourly = ds.sel(
        latitude=i1,
        longitude=j1,
        method='nearest'
    )

    for wave_dir in ['mwd', 'mdww', 'p140122', 'p140125', 'p140128']:
        card = metpy.calc.angle_to_direction(
            hourly[wave_dir] * metpy.units.units.deg,
            level=card_level
        )
        hourly[f'{wave_dir}_card'] = xr.DataArray(
            data=card,
            dims='valid_time'
        )

    for sea_state in sea_states:
        
        print(f'..{sea_state}')

        if sea_state == 'Ambos':
            hourly_seastate = hourly

        if sea_state == 'Unimodal':
            hourly_seastate = hourly.where(
                hourly.mdww_card == hourly.p140122_card,
                drop=True
            )

        if sea_state == 'Bimodal':
            hourly_seastate = hourly.where(
                hourly.mdww_card != hourly.p140122_card,
                drop=True
            )

        # Contagem de ocorrências Hs x Tp
        counts, _, _ = np.histogram2d(
            hourly_seastate.swh,
            hourly_seastate.pp1d,
            bins=(
                hs_bins[area],
                tp_bins_all
        ))
        counts_hs_tp[area][sea_state] = pd.DataFrame(
            columns=tp_bins_all[:-1],
            index=hs_bins[area][:-1],
            data=counts,
            dtype=int
        )
        
        for direction in card_dirs:

            print(f'....{direction} dir')

            hourly_seastate_card = hourly_seastate.where(
                hourly_seastate.mwd_card == direction
            )

            hs = hourly_seastate_card.swh.to_pandas()

            tp_binned = pd.cut(
                hourly_seastate_card.pp1d,
                tp_bins_all
            )

            hs_grouped_tp = hs.groupby(
                tp_binned
            )

            base_series = pd.Series(
                index=pd.DatetimeIndex([
                    '1940-01-01 00:00:00',
                    '2023-01-01 00:00:00'
                ]),
                data=np.nan
            )

            for key, value in hs_grouped_tp.groups.items():

                tp = key.left

                try:
                    print(
                        f'......Working on {tp}s data'
                    )

                    yearly = hs[value].resample('YS').max()
                    data = yearly.round(1)

                    if data.size > 0:
                        if data.index[0] > base_series.index[0]:
                            data = pd.concat([
                                base_series.head(1),
                                data
                            ])
                        
                        if data.index[-1] < base_series.index[-1]:
                            data = pd.concat([
                                data,
                                base_series.tail(1)
                            ])
                        
                        data = data.resample('YS').asfreq()
                    
                    else:
                        data = base_series.resample('YS').asfreq()

                    statistics_by_dir[area][sea_state][direction].loc[
                        tp, 'mean'
                    ] = np.round(data.mean(), 1)

                    statistics_by_dir[area][sea_state][direction].loc[
                        tp, 'std'
                    ] = np.round(data.std(), 1)

                    dir_count_total = hs[value].count()

                    dir_percent_total = (
                        dir_count_total / hs.count()
                    ) * 100

                    statistics_by_dir[area][sea_state][direction].loc[
                        tp, 'dir_count_total'
                    ] = dir_count_total

                    statistics_by_dir[area][sea_state][direction].loc[
                        tp, 'dir_percent_total'
                    ] = dir_percent_total

                    statistics_by_dir[area][sea_state][direction].loc[
                        tp, 'valid_count_yearly'
                    ] = data.count()

                    valid_percent_yearly = np.round(
                        (data.count() / data.size) * 100, 0
                    )

                    statistics_by_dir[area][sea_state][direction].loc[
                        tp, 'valid_percent_yearly'
                    ] = valid_percent_yearly

                    if valid_percent_yearly >= minimum_of_valid_data:
                        data = data.fillna(
                            data.median()
                        )

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
                                    data,
                                    size=data.size,
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

                    for prd in statistics_by_dir[area][sea_state][direction].columns[6:]:
                            statistics_by_dir[area][sea_state][direction].loc[tp, prd] = np.round(
                                np.median(levels[:, np.absolute(periods - prd).argmin()]),
                                2
                            )
                    
                except ValueError:
                    print(
                        f"......we've only {valid_percent_yearly}% of valid data, when the minimum is {minimum_of_valid_data}%"
                    )

filepath = os.path.join(
    rootdir,
    'statistics',
    f'counts_{var}.pkl'
)
with open(filepath, 'wb') as handle:
    pickle.dump(
        counts_hs_tp,
        handle
    )
handle.close()


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
