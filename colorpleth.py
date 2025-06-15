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
from pandas import date_range

figsdir = "/home/numa23/Public/aesop/figures/"

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

datas = pd.to_datetime(casos[:,0])

casos = casos[:,1:].astype(float)

verao = []
outono = []
inverno = []
primavera = []
for k in range(0,len(datas)):
    if datas.strftime('%m')[k] == '01':
        verao.append(casos[k,:])
    if datas.strftime('%m')[k] == '02':
        verao.append(casos[k,:])
    if datas.strftime('%m')[k] == '03':
        if float(datas.strftime('%d')[k]) > 21:
            outono.append(casos[k,:])
        else:
            verao.append(casos[k,:])
    if datas.strftime('%m')[k] == '04':
        outono.append(casos[k,:])
    if datas.strftime('%m')[k] == '05':
        outono.append(casos[k,:])
    if datas.strftime('%m')[k] == '06':
        if float(datas.strftime('%d')[k]) > 21:
            inverno.append(casos[k,:])
        else:
            outono.append(casos[k,:])
    if datas.strftime('%m')[k] == '07':
        inverno.append(casos[k,:])
    if datas.strftime('%m')[k] == '08':
        inverno.append(casos[k,:])
    if datas.strftime('%m')[k] == '09':
        if float(datas.strftime('%d')[k]) > 21:
            primavera.append(casos[k,:])
        else:
            inverno.append(casos[k,:])
    if datas.strftime('%m')[k] == '10':
        primavera.append(casos[k,:])
    if datas.strftime('%m')[k] == '11':
        primavera.append(casos[k,:])
    if datas.strftime('%m')[k] == '12':
        if float(datas.strftime('%d')[k]) > 21:
            verao.append(casos[k,:])
        else:
            primavera.append(casos[k,:])

verao = np.array(verao)
outono = np.array(outono)
inverno = np.array(inverno)
primavera = np.array(primavera)

#Fazendo a media dos casos

#Media do ano inteiro
casosmean = np.nanmean(casos,axis=0) 

#Media das estacoes
casosmeanverao = np.nanmean(verao,axis=0)
casosmeanoutono = np.nanmean(outono,axis=0)
casosmeaninverno = np.nanmean(inverno,axis=0)
casosmeanprimavera = np.nanmean(primavera,axis=0)

#see what the map looks like
fig=plt.figure(figsize=(8,12))
map_df.plot(figsize=(20, 10))

nameoffigure = "colorplethmap1"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=100)

print(map_df.head(93))

#Plotando ano inteiro
# Define how many discrete colors you want
num_colors = 25  # Change this to your desired number

# Create bins and normalize
custom_bins = np.linspace(casosmean.min(), casosmean.max(), num_colors + 1)
cmap = plt.get_cmap('Blues')
norm = BoundaryNorm(custom_bins, cmap.N)

fig, ax = plt.subplots(1, figsize=(10,6))
map_df.plot(casosmean,cmap = cmap, norm=norm, linewidth=1, ax=ax, edgecolor='0.2', legend = False)
ax.axis('off')


# Add custom colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=custom_bins)
#cbar.set_ticklabels([f"{custom_bins[i]:.2f}–{custom_bins[i+1]:.2f}" for i in range(len(custom_bins)-1)])

nameoffigure = "colorplethmap_ano_inteiro"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=300)


#Plotando verao
# Define how many discrete colors you want
num_colors = 25  # Change this to your desired number

# Create bins and normalize
custom_bins = np.linspace(casosmeanverao.min(), casosmeanverao.max(), num_colors + 1)
cmap = plt.get_cmap('Blues')
norm = BoundaryNorm(custom_bins, cmap.N)

fig, ax = plt.subplots(1, figsize=(10,6))
map_df.plot(casosmeanverao,cmap = cmap, norm=norm, linewidth=1, ax=ax, edgecolor='0.2', legend = False)
ax.axis('off')


# Add custom colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=custom_bins)
#cbar.set_ticklabels([f"{custom_bins[i]:.2f}–{custom_bins[i+1]:.2f}" for i in range(len(custom_bins)-1)])

nameoffigure = "colorplethmap_verao"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=300)


#Plotando outono
# Define how many discrete colors you want
num_colors = 25  # Change this to your desired number

# Create bins and normalize
custom_bins = np.linspace(casosmeanoutono.min(), casosmeanoutono.max(), num_colors + 1)
cmap = plt.get_cmap('Blues')
norm = BoundaryNorm(custom_bins, cmap.N)

fig, ax = plt.subplots(1, figsize=(10,6))
map_df.plot(casosmeanoutono,cmap = cmap, norm=norm, linewidth=1, ax=ax, edgecolor='0.2', legend = False)
ax.axis('off')


# Add custom colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=custom_bins)
#cbar.set_ticklabels([f"{custom_bins[i]:.2f}–{custom_bins[i+1]:.2f}" for i in range(len(custom_bins)-1)])

nameoffigure = "colorplethmap_outono"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=300)


#Plotando inverno
# Define how many discrete colors you want
num_colors = 25  # Change this to your desired number

# Create bins and normalize
custom_bins = np.linspace(casosmeaninverno.min(), casosmeaninverno.max(), num_colors + 1)
cmap = plt.get_cmap('Blues')
norm = BoundaryNorm(custom_bins, cmap.N)

fig, ax = plt.subplots(1, figsize=(10,6))
map_df.plot(casosmeaninverno,cmap = cmap, norm=norm, linewidth=1, ax=ax, edgecolor='0.2', legend = False)
ax.axis('off')


# Add custom colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=custom_bins)
#cbar.set_ticklabels([f"{custom_bins[i]:.2f}–{custom_bins[i+1]:.2f}" for i in range(len(custom_bins)-1)])

nameoffigure = "colorplethmap_inverno"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=300)


#Plotando primavera
# Define how many discrete colors you want
num_colors = 25  # Change this to your desired number

# Create bins and normalize
custom_bins = np.linspace(casosmeanprimavera.min(), casosmeanprimavera.max(), num_colors + 1)
cmap = plt.get_cmap('Blues')
norm = BoundaryNorm(custom_bins, cmap.N)

fig, ax = plt.subplots(1, figsize=(10,6))
map_df.plot(casosmeanprimavera,cmap = cmap, norm=norm, linewidth=1, ax=ax, edgecolor='0.2', legend = False)
ax.axis('off')


# Add custom colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=custom_bins)
#cbar.set_ticklabels([f"{custom_bins[i]:.2f}–{custom_bins[i+1]:.2f}" for i in range(len(custom_bins)-1)])

nameoffigure = "colorplethmap_primavera"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir+string_in_string,dpi=300)
