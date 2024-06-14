import nctoolkit as nc
ds = nc.open_data(['/home/numa23/copernicus/otec_brasil_2014_2021_0.5m.nc', '/home/numa23/copernicus/otec_brasil_2021_2023_0.5m.nc'])
ds.merge("time")


ds.to_nc('otec_brasil_2014_2023_0.5m.nc')
