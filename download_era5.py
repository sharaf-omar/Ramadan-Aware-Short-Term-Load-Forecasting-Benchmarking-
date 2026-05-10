import cdsapi

c = cdsapi.Client()

years = range(2018, 2026)

variables = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "ssrd": "surface_solar_radiation_downwards"
}

area = [42.5, 25.5, 35.8, 45.8] #standard bounding box maximum/minimum latitudes and longitudes

for var_name, var_code in variables.items():
    for year in years:
        print(f"Downloading {var_name} - {year}")

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': var_code,
                'year': str(year),
                'month': [f"{m:02d}" for m in range(1, 13)],
                'day': [f"{d:02d}" for d in range(1, 32)],
                'time': [f"{h:02d}:00" for h in range(24)],
                'area': area,
            },
            f'{var_name}_{year}.nc' 
        )

print("Done.")