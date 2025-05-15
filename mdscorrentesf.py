# -*- coding: utf-8 -*-
# AUTOR: Douglas Medeiros Nehme
# CONTACT: medeiros.douglas3@gmail.com
# CRIATION: oct/2023
# OBJECTIVE: Calculate current speed return
#            level using CFSR and CFSv2
#            data for SE Brazil
#
# Nesta rotina, os resultados de nível de retorno
# e direção da corrente são calculados para todas
# as profundidades de acordo com a direção da
# corrente na superfície

import os
import sys
import metpy
import pickle
import functools

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from scipy.stats import norm as normal
from scipy.stats import genextreme as gev
from matplotlib.ticker import ScalarFormatter

#sys.path.insert(
#    0, os.path.expanduser('~/Dropbox/airsea')
#)

import airsea

def slice_depth(ds, depth):
    """
    Get just the first depth of each file to speed up
    the opening process and load less data
    """
    return ds.sel(depthBelowSea=depth)

###############################################
rootdir = (
    # '/home/douglas/Dropbox/profissional/LAMCE/cnooc'
    '/home/numa23/Public/Projeto_BR_OTEC/copernicus'
)

areas = ['MN', 'campos', 'santos']
var   = 'cspd'
varname = 'Ocean Current Speed'
unidade = '$\\mathdefault{m.s^{-1}}$'

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

depths = [
    5, 15, 25, 35,
    45, 55, 65, 75,
    85, 95, 105, 115
]
columns_names = [
    'dir_count_total', 'dir_percent_total',
    'valid_count_yearly', 'valid_percent_yearly',    
    'mean', 'std',
    1.01, 10, 20, 30, 50, 100, 'dir'
]

tabelaMN_stats = pd.DataFrame(
    data=np.nan,
    columns=columns_names,
    index=depths[:-1],
)
tabelacampos_stats = pd.DataFrame(
    data=np.nan,
    columns=columns_names,
    index=depths[:-2],
)
tabelasantos_stats = pd.DataFrame(
    data=np.nan,
    columns=columns_names,
    index=depths,
)
# Inside each area key (MN, campos, santos) we have
# a dataset for each card_dirs value
statistics_by_dir = {
    'MN': dict([(key, tabelaMN_stats.copy()) for key in card_dirs]),
    'campos': dict([(key, tabelacampos_stats.copy()) for key in card_dirs]),
    'santos' : dict([(key, tabelasantos_stats.copy())  for key in card_dirs])
}

tabelaMN_count = pd.DataFrame(
    data=np.nan,
    columns=card_dirs,
    index=depths[:-1],
)
tabelacampos_count = pd.DataFrame(
    data=np.nan,
    columns=card_dirs,
    index=depths[:-2],
)
tabelasantos_count = pd.DataFrame(
    data=np.nan,
    columns=card_dirs,
    index=depths,
)
# Dict to store the occurence of each
# direction by each depth for all the
# possible directions of surface cspd
counts_hourly_by_depth = {
    'MN': dict([(key, tabelaMN_count.copy()) for key in card_dirs]),
    'campos': dict([(key, tabelacampos_count.copy()) for key in card_dirs]),
    'santos' : dict([(key, tabelasantos_count.copy())  for key in card_dirs])
}

# for area in areas:
for area in ["MN","campos",'santos']:

    print(
        f'GEV Return Level of {var} for {area}'
    )
    cfsr = xr.open_mfdataset(
        f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br56_teste/data/currents_cfsr_*_{area}_alldepths.nc',
        combine='nested',
        concat_dim='time',
        # drop_variables=['depthBelowSea'],
        # parallel=True,
        autoclose=True,
        # preprocess=functools.partial(
        #     slice_depth,
        #     depth=depth)
    )
    cfsv2 = xr.open_mfdataset(
        f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br56_teste/data/currents_cfsv2_*_{area}_alldepths.nc',
        combine='nested',
        concat_dim='time',
        # drop_variables=['depthBelowSea'],
        # parallel=True,
        autoclose=True,
        # preprocess=functools.partial(
        #     slice_depth,
        #     depth=depth)
    )
    ds = xr.concat(
        [cfsr, cfsv2],
        dim='time'
    )

    del cfsr, cfsv2

    # Create a new variable (bathymetry) to retain the last non-nan value
    # of depthBelowSea for each grid point
    ds['bathymetry'] = xr.open_dataarray(
        f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/br56_teste/bathymetry_{area}_cfsr_cfsv2.nc'
    )

    # Transforming lon from [0/360] to [-180/180]
    ds.coords['longitude'] = (ds.coords['longitude'] + 180) % 360 - 180
    ds = ds.sortby(
        ds.longitude
    )

    ds = ds.where(
        (ds.bathymetry >= 60) &
        (ds.bathymetry <= 150)
    )
    ###############################################
    # Como a batemetria da grade oceânica do ERA5 é diferente da grade oceânica
    # do CFSR/CFSv2, minha prioridade ao escolher o ponto de análise (i1, j1) do
    # segundo produto foi a célula com a batemtria mais próxima de 100 metros,
    # pois mesmo as análises sendo realizadas entre 60 e 150 de profundidade, 100
    # metros é o foco do projeto.

    if area == 'MN':
        # indeces usados no ERA5
        # i1, j1 = -4.5, -36.5
        # indices usados no CFSR/CFSv2
        
        #i1, j1 = -0.25, -44.25
        i1, j1 = -0.75, -44.75
    if area == 'campos':
        # indeces usados no ERA5
        # i1, j1 = -23.25, -41.5
        # indices usados no CFSR/CFSv2
       
        #i1, j1 = -22.75, -40.25
        i1, j1 = -22.25, -41.25
    if area == 'santos':
        # indeces usados no ERA5
        # i1, j1 = -30, -48.5
        # indices usados no CFSR/CFSv2
        
        #i1, j1 = -23.75, -42.0
        i1, j1 = -23.25, -42.0

    synoptic = ds.sel(
        latitude=i1,
        longitude=j1,
        method='nearest'
    )
    print(synoptic)
   
    # Calculate ocean current speed
    synoptic['cspd'], synoptic['cdir'] = airsea.cart2pol_wind(
        synoptic.uoe,
        synoptic.von,
        rnd=3,
        blowing_from=False
    )
    ccard = metpy.calc.angle_to_direction(
        synoptic['cdir'] * metpy.units.units.deg,
        level=card_level
    )
    synoptic['ccard'] = xr.DataArray(
        data=ccard,
        dims=['time', 'depthBelowSea']
    )

    for sfc_direction in card_dirs:

        print(
            f'..Working on {sfc_direction} dir'
        )

        # Get the times where the surface current
        # is aligned with sfc_direction value
        surface_cspd_mask = synoptic.ccard.sel(
            depthBelowSea=5
        ) == sfc_direction

        # Drop depthBelowSea coordinate, that had
        # just the 5m value, to be possible to
        # replicate this to the original shape
        surface_cspd_mask = surface_cspd_mask.drop_vars(
            'depthBelowSea'
        )
        # Replicate the mask var to the original
        # shape, so mask had just a time dimension
        # with 65482 size above and now is a 2D of
        # (65482, 40) shape
        surface_cspd_mask = surface_cspd_mask.expand_dims(
            dim={'depthBelowSea': synoptic.depthBelowSea},
            axis=1
        )
        synoptic_card = synoptic.where(
            surface_cspd_mask
        )
        # Transform nan values from ccard variable
        # to 'UND' forcing it to have str type
        synoptic_card['ccard'] = synoptic_card.ccard.where(
            synoptic_card.ccard.notnull(),
            'UND'
        )

        for depth in statistics_by_dir[area][sfc_direction].index:

            try:
                print(
                    f'....Working on {depth}m data'
                )

                synoptic_card_depth = synoptic_card.sel(
                    depthBelowSea=depth
                )
                # Get the number of occurences of each
                # direction to calculate the return
                # level from the most common one
                ccard_dirs, ccard_counts = np.unique(
                    synoptic_card_depth.ccard,
                    return_counts=True
                )
                # Delete from the two arrays above the
                # values related to UND
                ccard_counts = np.delete(
                    ccard_counts,
                    np.argwhere(
                        ccard_dirs == 'UND'
                ))
                ccard_dirs = np.delete(
                    ccard_dirs,
                    np.argwhere(
                        ccard_dirs == 'UND'
                ))

                # Populating count_by_depth
                for dirs, counts in zip(ccard_dirs, ccard_counts):
                    counts_hourly_by_depth[area][sfc_direction].loc[
                        depth, dirs
                    ] = counts
                print(232323232323232323232)
                print(ccard_dirs)
                most_common_dir = ccard_dirs[
                    ccard_counts.argmax()
                ]

                # Storing the info of the most common dir
                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'dir'
                ] = most_common_dir

                # Filter the dataset by the most
                # common direction on that depth
                synoptic_card_depth_common = synoptic_card_depth.where(
                    synoptic_card_depth.ccard == most_common_dir
                ).cspd

                yearly = synoptic_card_depth_common.resample(
                    time='1Y'
                ).max()

                mean = np.round(
                    np.mean(synoptic_card_depth_common), 3
                )
                std = np.round(
                    synoptic_card_depth_common.std().values, 3
                )
                dir_count_total = synoptic_card_depth_common.count().values

                dir_percent_total = (
                    dir_count_total / synoptic_card_depth.cspd.size
                ) * 100

                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'mean'
                ] = mean
                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'std'
                ] = std

                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'dir_count_total'
                ] = dir_count_total

                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'dir_percent_total'
                ] = np.round(dir_percent_total, 0)
                
                ###############################################
                valid_count_yearly = yearly.count().values

                valid_percent_yearly = (
                    valid_count_yearly / yearly.size
                ) * 100

                valid_percent_yearly = np.round(
                    valid_percent_yearly, 0
                )

                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'valid_count_yearly'
                ] = valid_count_yearly

                statistics_by_dir[area][sfc_direction].loc[
                    depth, 'valid_percent_yearly'
                ] = valid_percent_yearly
               
                if any(yearly.isnull()):
                    if valid_percent_yearly >= minimum_of_valid_data:
                        yearly = yearly.fillna(
                            yearly.median()
                        )

                data = yearly.to_pandas()
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

                for prd in statistics_by_dir[area][sfc_direction].columns[6:-1]:
                    statistics_by_dir[area][sfc_direction].loc[depth, prd] = np.round(
                        np.median(levels[:, np.absolute(periods - prd).argmin()]),
                        3
                    )
           
            except ValueError:
                print(
                    f"......we've only {valid_percent_yearly}% of valid data, when the minimum is {minimum_of_valid_data}%"
                )

    # # create a data frame from yearly wind speed data
    # df = pd.DataFrame(
    #     index=np.arange(
    #         data.size
    # ))
    # # Add info of year
    # df['year'] = data.index.year.values
    # # sort the data in descending way
    # df['sorted'] = data.values
    # df.sort_values(
    #     'sorted',
    #     ascending=False,
    #     inplace=True
    # )
    # # assign the data ranks
    # df['ranks'] = np.arange(
    #     data.size
    # )
    # # rank via scipy rankdata instead to deal with duplicate values
    # # With this function, duplicated wspd100 data hava the same rank
    # if len(df['sorted']) != len(np.unique(df['sorted'])):
    #     df['ranks_sp'] = np.sort(
    #         rankdata(
    #             -data
    #     ))
    # # find exceedance probability
    # df['exceedance'] = df['ranks_sp'] / (data.size + 1)
    # # find return period
    # df['period'] = 1 / df['exceedance']

    # df.to_csv(
    #     os.path.join(
    #         rootdir,
    #         'statistics',
    #         f'return_levels_empiric_{var}_{area}.csv'
    #     ),
    #     na_rep=-9999
    # )

    # ###############################################

    # if area == 'NE':
    #     min_bin = 0
    #     max_bin = 3
    #     step = 0.1
    #     xlim = (0, 3)
   
    # if area == 'SE':
    #     min_bin = 0
    #     max_bin = 3
    #     step = 0.1
    #     xlim = (0, 3)

    # if area == 'S':
    #     min_bin = 0
    #     max_bin = 3
    #     step = 0.1
    #     xlim = (0, 3)

    # bins_yearly = np.arange(
    #     min_bin,
    #     yearly.max() + step,
    #     step
    # )
    # fig, ax = plt.subplots(
    #     figsize=(8, 8)
    # )
    # # make the histogram
    # ax.hist(
    #     yearly.values.flatten(),
    #     bins=bins_yearly,
    #     density=True,
    #     color='tab:gray',
    #     edgecolor='k',
    #     alpha=0.35,
    #     label='Annual Maximums'
    # );
    # x = np.arange(
    #     min_bin,
    #     max_bin,
    #     step
    # )
    # ax.plot(
    #     x,
    #     normal.pdf(
    #         x,
    #         yearly.mean(),
    #         yearly.std()
    #     ),
    #     c='tab:gray',
    #     lw=2,
    #     label='Normal Distribuition'
    # )
    # shape_yearly, loc_yearly, scale_yearly = gev.fit(
    #     yearly.values, 0
    # )
    # ax.plot(
    #     x,
    #     gev.pdf(
    #         x,
    #         shape_yearly,
    #         loc=loc_yearly,
    #         scale=scale_yearly
    #     ),
    #     c='tab:gray',
    #     lw=2,
    #     linestyle='--',
    #     label='GEV Distribuition'
    # )
    # ax.legend(
    #     loc='upper right'
    # )
    # ax.set_xlim(xlim);
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # ax.set_yticklabels(
    #     ax.get_yticklabels(),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # ax.set_ylabel(
    #     'Occurrence Frequency',
    #     fontsize=12,
    #     fontweight='bold'
    # )
    # ax.set_xlabel(
    #     f'{varname} ({unidade})',
    #     fontsize=12,
    #     fontweight='bold'
    # );

    # fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         rootdir,
    #         'figure',
    #         f'histogram_{var}_{area}.png'
    # ))

    ###############################################

    # # plot the results
    # fig, [ax1, ax2] = plt.subplots(
    #     1, 2,
    #     figsize=(15, 7),
    #     sharey=True,
    #     # sharex=True,
    #     gridspec_kw={
    #         'wspace': 0.1
    # })
    # ax1.scatter(
    #     df['period'],
    #     df['sorted'],
    #     s=60,
    #     marker='o'
    # )
    # ax1.set_xlabel(
    #     'Período de Retorno (Anos)',
    #     fontsize=14,
    #     fontweight='bold'
    # )
    # ax1.set_ylabel(
    #     f'Nível de Retorno ({unidade})',
    #     fontsize=14,
    #     fontweight='bold'
    # )
    # ax1.set_xscale('linear')  # notice the xscale
    # # ax1.set_xlim([1, 40])
    # # ax1.set_ylim([20, 28])
    # # ax1.set_yticklabels(
    # #     ax1.get_yticklabels(),
    # #     fontsize=11,
    # #     fontweight='bold'
    # # )
    # # ax1.set_xticklabels(
    # #     ax1.get_xticklabels(),
    # #     fontsize=11,
    # #     fontweight='bold'
    # # )
    # ax1.grid()
    # ax2.scatter(
    #     df['period'],
    #     df['sorted'],
    #     s=60,
    #     marker='o'
    # )
    # ax2.set_xlabel(
    #     'Período de Retorno (Anos)',
    #     fontsize=14,
    #     fontweight='bold'
    # )
    # ax2.set_xscale('log')  # notice the xscale
    # # ax2.set_xlim([1, 40])
    # # ax2.set_xticks(
    # #     [1, 2, 3, 4, 5, 10, 20, 30, 40]
    # # );
    # # ax2.set_xticklabels([
    # #     '$\\mathdefault{10^{0}}$', 2, 3, 4, 5,
    # #     '$\\mathdefault{10^{1}}$', 20, 30, 40],
    # #     fontsize=11,
    # #     fontweight='bold'
    # # );
    # ax2.grid()
    # fig.savefig(
    #     '/home/numa06/douglas/cnooc/figure/cspd_SE_return_level_0.png'
    # )
    # print('Grafico 1 feito')
    # ###############################################

    # xlim = (0.9, 225)

    # if area == 'NE':
    #     ylim = (0.7, 1.2)
    # if area == 'SE':
    #     ylim = (0.6, 1.3)
    # if area == 'S':
    #     ylim = (0.75, 2)

    # ###############################################

    # return_levels = pd.DataFrame(
    #     columns=[
    #         'RL2.5_GEVbootstrap',
    #         'RLmedian_GEVbootstrap',
    #         'RL97.5_GEVbootstrap',
    #         'RL_GEVunico'],
    #     index=[1.01, 10, 20, 50, 100],
    #     data=np.ones((5, 4)) * np.nan
    # )
    # return_levels.index.name = 'return_periods_years'

    # confidence = 0.95

    # # Calculo do Niveis de retorno associados
    # # aos quantis de 2,5% e 97,5% do intervalo
    # # de confiança, ou seja, para um intervalo
    # # de 95%. Para isso são usados os resultados
    # # produzidos pelo bootstraping. Por fim,
    # # também calculo o nível de retorno para cada
    # # período definido no index (em anos) do
    # # dataframe return_levels a partir da média
    # # dos resultados do bootstraping, assim como
    # # para os valores iniciais obtidos diretamente
    # # das simulações globais.
    # for prd in return_levels.index:
    #     lvl = gev.ppf(
    #         1 - 1/prd,
    #         shape_yearly,
    #         loc=loc_yearly,
    #         scale=scale_yearly
    #     )
    #     lvl = np.round(lvl, 3)

    #     return_levels.loc[prd, 'RL_GEVunico'] = lvl

    #     idx = np.absolute(periods - prd).argmin()
       
    #     lvl_min_confidence = np.quantile(
    #         levels,
    #         [(1 - confidence) / 2.],
    #         axis=0
    #     ).T.squeeze()[idx]
       
    #     lvl_max_confidence = np.quantile(
    #         levels,
    #         [confidence + (1 - confidence) / 2.],
    #         axis=0
    #     ).T.squeeze()[idx]

    #     lvl_min_confidence = np.round(
    #         lvl_min_confidence,
    #         3
    #     )
    #     lvl_max_confidence = np.round(
    #         lvl_max_confidence,
    #         3
    #     )
    #     return_levels.loc[prd, 'RL2.5_GEVbootstrap']  = lvl_min_confidence
    #     return_levels.loc[prd, 'RL97.5_GEVbootstrap'] = lvl_max_confidence
    #     return_levels.loc[prd, 'RLmedian_GEVbootstrap'] = np.round(
    #         np.median(levels[:, np.absolute(periods - prd).argmin()]),
    #         3
    #     )
   
    # return_levels.to_csv(
    #     os.path.join(
    #         rootdir,
    #         'statistics',
    #         f'return_levels_GEV_{var}_{area}.csv'
    #     ),
    #     na_rep=-9999
    # )

    # ###############################################

    # fig, ax = plt.subplots()
    # ax.scatter(
    #     df['period'],
    #     df['sorted'],
    #     s=9,
    #     c='k',
    #     marker='o',
    #     label='Annual Maximums (1940-2023)',
    #     zorder=9,
    #     alpha=0.5
    # )
    # # plot return levels curve
    # ax.plot(
    #     periods,
    #     # Média dos Níveis de retorno a partir do bootstraping que
    #     # calcula distribuições GEV quantas vezes forem definidas
    #     # em nsamples
    #     np.median(levels, axis=0),
    #     # # Representação dos Níveis de retorno a partir da Distribuição GEV
    #     # levels_yearly,
    #     label='Estimate by GEV Distribuition',
    #     color='k',
    #     zorder=8
    # )
    # # plot confidence intervals
    # ax.plot(
    #     periods,
    #     np.quantile(
    #         levels,
    #         [confidence + (1 - confidence) / 2.],
    #         axis=0).T,
    #         "k--",
    #         label='95% Confidence Interval'
    # )
    # ax.plot(
    #     periods,
    #     np.quantile(
    #         levels,
    #         [(1 - confidence) / 2.],
    #         axis=0).T,
    #         "k--"
    # )
    # ax.set_xscale("log")
    # ax.set_ylim(ylim)
    # ax.set_xlim(xlim)
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # ax.set_yticklabels(
    #     ax.get_yticklabels(),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # ax.xaxis.set_major_formatter(
    #     ScalarFormatter()
    # )
    # ax.set_ylabel(
    #     f'Return Level ({unidade})',
    #     fontsize=12,
    #     fontweight='bold'
    # )
    # ax.set_xlabel(
    #     'Return Period (Years)',
    #     fontsize=12,
    #     fontweight='bold'
    # )
    # ax.legend(
    #     loc='upper left'
    # )
    # ax.grid()

    # fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         rootdir,
    #         'figure',
    #         f'return_level_2_{var}_{area}.png'
    # ))

    # ###############################################

    # fig, ax = plt.subplots()
    # empirico = ax.scatter(
    #     df['period'],
    #     df['sorted'],
    #     s=9,
    #     c=df['year'],
    #     marker='o',
    #     cmap='turbo',
    #     label='Annual Maximums (1940-2023)',
    #     zorder=9,
    #     alpha=0.5
    # )
    # # plot return levels curve
    # ax.plot(
    #     periods,
    #     # Média dos Níveis de retorno a partir do bootstraping que
    #     # calcula distribuições GEV quantas vezes forem definidas
    #     # em nsamples
    #     np.median(levels, axis=0),
    #     # # Representação dos Níveis de retorno a partir da Distribuição GEV
    #     # levels_yearly,
    #     label='Estimate by GEV Distribuition',
    #     color='k',
    #     zorder=8
    # )
    # # plot confidence intervals
    # ax.plot(
    #     periods,
    #     np.quantile(
    #         levels,
    #         [confidence + (1 - confidence) / 2.],
    #         axis=0).T,
    #         "k--",
    #         label='95% Confidence Interval'
    # )
    # ax.plot(
    #     periods,
    #     np.quantile(
    #         levels,
    #         [(1 - confidence) / 2.],
    #         axis=0).T,
    #         "k--"
    # )
    # ax.set_xscale("log")
    # ax.set_ylim(ylim)
    # ax.set_xlim(xlim)
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # ax.set_yticklabels(
    #     ax.get_yticklabels(),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # ax.xaxis.set_major_formatter(
    #     ScalarFormatter()
    # )
    # ax.set_ylabel(
    #     f'Return Level ({unidade})',
    #     fontsize=12,
    #     fontweight='bold'
    # )
    # ax.set_xlabel(
    #     'Return Period (Years)',
    #     fontsize=12,
    #     fontweight='bold'
    # )
    # ax.legend(
    #     loc='upper left'
    # )
    # ax.grid()
    # cbar = fig.colorbar(empirico)
    # cbar.set_label(
    #     'Years',
    #     fontsize=10,
    #     fontweight='bold'
    # )
    # fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         rootdir,
    #         'figure',
    #         f'return_level_3_{var}_{area}.png'
    # ))

    # plt.close('all')

filepath = os.path.join(
    rootdir,
    'statistics',
    f'statistics_by_dir_{var}_V2.pkl'
)

with open(filepath, 'wb') as handle:
    pickle.dump(
        statistics_by_dir,
        handle
    )
handle.close()
