# -*- coding: utf-8 -*-
"""Aesop_indicesclinicos
"""
#%% Importação de Bibliotecas
import os
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from unidecode import unidecode


#%% Carregamento dados de população por município
def carregar_dados_municipios(caminho_csv):
    dados = pd.read_csv(caminho_csv)
    dados.columns = ["Variável","Cod_Mun","Nm_Mun","Populacao"]
    dados["Cod_Mun"] = dados["Cod_Mun"].astype(str).str[:-1]#.astype(int)
    #dados = pd.read_csv(caminho_csv, sep=",", usecols=["Variável","Cod_Mun","Nm_Mun","População"])
    print("Dados de municípios carregados.")
    return dados


caminho_municipios = "/home/numa23/Public/aesop/tabela4709.csv"
dados_municipios = carregar_dados_municipios(caminho_municipios)

print(dados_municipios)


#%% Leitura e pré-processamento da base de atendimentos
def carregar_base_atendimentos(caminho_csv):
    #df = pd.read_csv(caminho_csv, sep=";", usecols=[
    #    'municipio',"co_ibge", 'ano',"epiweek","atend_totais", 'atend_ivas', 'atend_arbov', 'pc_cobertura_sf',"pc_cobertura_ab","cod_rgimedlata","nome_rgi","cod_rgint", 'nome_rgint'
    #])
    df = pd.read_csv(caminho_csv, sep=";", usecols=[
        'municipio', "co_ibge",'ano', 'atend_ivas', 'atend_arbov', 'epiweek', 'nome_rgint'
    ],dtype={"co_ibge": "str"})
    df.index = pd.to_datetime(df['ano'], format='%Y') + df['epiweek'].apply(lambda x: pd.Timedelta(x * 7, 'days'))
    df = df.drop(['ano', 'epiweek'], axis=1)
    df['data'] = df.index
    df = df.groupby(['municipio', 'data', 'nome_rgint']).sum().reset_index()
    df = df.set_index('data')
    print("base de atendimentos carregada e processada.")
    return df

caminho_base = "/home/numa23/Public/aesop/output.csv"
base_atendimentos = carregar_base_atendimentos(caminho_base)

print(base_atendimentos)

#%% Normalização dos dados por população
def normalizar_dados_por_populacao(dados_municipios, df_atendimentos):
    dados_normalizados = {}
    for local in dados_municipios["Cod_Mun"]:
        print(f"Normalizando dados para: {local}")
        try:
            populacao = dados_municipios.loc[dados_municipios["Cod_Mun"] == local, "Populacao"].values[0]
            #print(populacao)
            atendimentos = df_atendimentos.loc[df_atendimentos["co_ibge"] == local, "atend_ivas"]
            print(atendimentos)
            #print(local,type(local),99999999999)
            #print(df_atendimentos["cod_rgimediata"].dtypes,888888888)
            dados_normalizados[local] = atendimentos * 100 / populacao
        except IndexError:
            print(f"Aviso: Município '{local}' não encontrado na base de atendimentos.")
    print("Dados normalizados por população.")
    return pd.DataFrame.from_dict(dados_normalizados)

dados_normalizados = normalizar_dados_por_populacao(dados_municipios, base_atendimentos)

print(dados_normalizados)

#%% Exportação dos dados normalizados
def exportar_dados(dados_normalizados, caminho_saida):
    dados_normalizados.to_csv(caminho_saida, sep=';', decimal='.')
    print(f"Dados exportados para {caminho_saida}")

caminho_saida_csv = "datadadosnormalizadosatuais_br(corrigido)2025.csv"
exportar_dados(dados_normalizados, caminho_saida_csv)
