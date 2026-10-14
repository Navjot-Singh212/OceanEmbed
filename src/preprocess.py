import xarray as xr
import numpy as np
import os
import glob
import json


target_lat=np.arange(5.0,30.25,0.25)
target_long=np.arange(45.0,105.25,0.25)


file_configs = {
    "ssh_2013_2024_combined.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["sla"]},
    "sss_2013_2024_combined.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["sos"]},
    "sst_2013_2024_combined.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["analysed_sst"]},
    "bathymetry.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["mask","deptho"]},
    "glory_yearly/glory_target_*.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["thetao"]},
    "ccmp_yearly/ccmp_winds_*.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["uwnd", "vwnd"]}, 
    "oscar_yearly/oscar_*.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": ["u", "v"]} 
}

def regrid_dataset(input_path,output_path,config):
    print(f"Loading raw data from {input_path}")
    ds=xr.open_mfdataset(input_path,combine="by_coords",chunks={"time":5},engine="h5netcdf",parallel=False)
    ds = ds.sel(time=slice("2022-01-01", "2024-06-30"))

    # lat_label=config["lat_name"]
    # lon_label=config["lon_name"]
    
    # rename_dict={}
    # if lat_label!="lat":
    #     rename_dict[lat_label]="lat"
    # if lon_label!="lon":
    #     rename_dict[lon_label]="lon"

    if "latitude" in ds.dims or "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
        ds=ds.set_index(lat="lat")
    if "longitude" in ds.dims or "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
        ds=ds.set_index(lon="lon")

    if "depth" in ds.dims:
        if ds.sizes["depth"] == 1:
            ds = ds.squeeze(dim="depth", drop=True)
        else:
            print(f"Keeping depth dimension: {ds.sizes['depth']} vertical layers detected.")
        
    
    if "lat" not in ds.dims and "lat" not in ds.coords:
        
        print("Available dims:", ds.dims)

    #some precautions
    ds=ds.sel(lat=slice(5.0,30.0),lon=slice(45.0,105.0))

    if "uwnd" in config["var"]:
        print("Downsampling 6-hourly wind to daily to save disk space...")
        ds = ds.resample(time="1D").mean()


    ds_regridded=ds.interp(lat=target_lat,lon=target_long,method="linear")
    ds_regridded=ds_regridded.fillna(0.0)


    #save
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    ds_regridded.to_netcdf(output_path)
    print(f"File successfully saved to {output_path}")
    ds.close()
    ds_regridded.close()




#temporal preprocessing
def average_wind_to_daily(input_path,output_path):
    print(f"Loading wind data from {input_path}")
    ds=xr.open_dataset(input_path,chunks={"time":5})
    print("Averaging 6-hour data to daily~")
    ds_daily=ds.resample(time="1D").mean()

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    ds_daily.to_netcdf(output_path)
    print(f"Saved aligned wind data to: {output_path}")

    ds.close()
    ds_daily.close()

#normalisation
scaler_dict={}
def normalize_file(input_path,output_path,var_name):
    print(f"Normalising {var_name}...")
    ds=xr.open_dataset(input_path,chunks={"time": 5, "lat": 50, "lon": 50})

    if isinstance(var_name, str):
        var_name = [var_name]

    for v in var_name:
        print(f"Normalising {v}....")
        var_mean=float(ds[v].mean(skipna=True).compute())
        var_std=float(ds[v].std(skipna=True).compute())

        global scaler_dict
        scaler_dict[v]={"mean":var_mean,"std":var_std}
        ds[v]=(ds[v]-var_mean)/var_std

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()
    print(f"Successfully normalised {var_name} to {output_path}")    

if __name__=="__main__":
    base_dir = r"/content/drive/MyDrive/OceanEmbed/data"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    normalized_dir = os.path.join(base_dir, "normalized")

    for pattern, config in file_configs.items():
        input_pattern=os.path.join(raw_dir,pattern)
        base_name = pattern.split('/')[0].replace(".nc", "") if '/' in pattern else pattern.replace(".nc", "")        
        output_file=os.path.join(processed_dir,f"{base_name}_processed.nc")

        if os.path.exists(output_file):
            print(f"Already processed : {output_file}! skipping this file~")
            continue

        if glob.glob(input_pattern):
            regrid_dataset(input_pattern,output_file,config)
        else:
            print(f"SKIPPING: no files found for {pattern}")

    # wind_input_path = os.path.join(processed_dir, "ccmp_yearly_processed.nc")
    # wind_output_path = os.path.join(processed_dir, "ccmp_daily.nc")
    
    # if os.path.exists(wind_output_path):
    #     print(f"Already time-aligned. Skipping {wind_output_path}")
    # elif os.path.exists(wind_input_path):
    #     average_wind_to_daily(wind_input_path,wind_output_path)
    # else:
    #     print(f"Processed wind data not found. Check the input path.")
    




    files_to_normalize = {
        "ssh_2013_2024_combined_processed.nc": ["sla"],
        "sss_2013_2024_combined_processed.nc": ["sos"],
        "sst_2013_2024_combined_processed.nc": ["analysed_sst"],
        "glory_yearly_processed.nc": ["thetao"],
        "ccmp_yearly_processed.nc": ["uwnd", "vwnd"],          
        "oscar_yearly_processed.nc": ["u", "v"],
        "bathymetry_processed.nc": ["deptho"] 
    }

    for filename,var_name in files_to_normalize.items():
        input_path=os.path.join(processed_dir,filename)

        # clean_names=filename.replace("Winds/","").replace("Oscar_currents/","")
        output_path=os.path.join(normalized_dir,filename)

        if os.path.exists(input_path):
            if not os.path.exists(output_path):
                # if isinstance(var_name, list):
                #     print(f"Normalising multiple variables in {filename}...")
                #     ds = xr.open_dataset(input_path)
                #     for v in var_name:
                #         var_mean = float(ds[v].mean(skipna=True))
                #         var_std = float(ds[v].std(skipna=True))
                #         scaler_dict[v] = {"mean": var_mean, "std": var_std}
                #         ds[v] = (ds[v] - var_mean) / var_std
                #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
                #     ds.to_netcdf(output_path)
                #     ds.close()
                #     print(f"Successfully normalised variables {var_name} to {output_path}")
                    
                normalize_file(input_path,output_path,var_name)
            else:
                print(f"Already normalized: {filename}. Skipping...")
        else:
            print(f"file not found: {input_path}")

    json_path=os.path.join(normalized_dir,"scalers.json")
    with open(json_path,"w") as f:
        json.dump(scaler_dict,f,indent=4)
    print(f"\nSaved Scaling factors to: {json_path}")



