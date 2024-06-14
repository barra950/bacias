import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
  variables=["thetao","bottomT"],
  minimum_longitude=-54,
  maximum_longitude=-27,
  minimum_latitude=-40,
  maximum_latitude=11,
  start_datetime="2021-06-30T00:00:00",
  end_datetime="2023-12-31T23:59:59",
  minimum_depth=1000,
  maximum_depth=1100,
  output_filename = "pozzi2.nc",
  output_directory = "copernicus-data"
)
