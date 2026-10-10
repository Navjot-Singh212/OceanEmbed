import xarray as xr
import numpy as np
import os
import glob
import json


target_lat=np.arange(5.0,30.25,0.25)
target_long=np.arange(45.0,105.25,0.25)


file_configs = {
    "SSH.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": "sla"},
    "SSS.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": "sos"},
    "SST.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": "analysed_sst"},
    "bathymetry.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": "mask"},
    "glory_target.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": "thetao"},
    "Winds/CCMP_Wind_Analysis_*.nc": {"lat_name": "latitude", "lon_name": "longitude", "var": "ws"}, 
    "Oscar_currents/oscar_currents_final_*.nc": {"lat_name": "lat", "lon_name": "lon", "var": "u"} 
}

def regrid_dataset(input_path,output_path,config):
    print(f"Loading raw data from {input_path}")
    ds=xr.open_mfdataset(input_path,combine="by_coords")

    lat_label=config["lat_name"]
    lon_label=config["lon_name"]

    # rename_dict={}
    # if lat_label!="lat":
    #     rename_dict[lat_label]="lat"
    # if lon_label!="lon":
    #     rename_dict[lon_label]="lon"

    if "latitude" in ds.dims or "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.dims or "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
        
    
    if "lat" not in ds.dims and "lat" not in ds.coords:
        
        print("Available dims:", ds.dims)

    #some precautions
    ds=ds.sel(lat=slice(5.0,30.0),lon=slice(45.0,105.0))
    ds_regridded=ds.interp(lat=target_lat,lon=target_long,method="linear")
    ds_regridded=ds_regridded.fillna(0.0)


    #save
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    ds_regridded.to_netcdf(output_path)
    print(f"File successfully saved to {output_path}")
    ds.close()
    ds_regridded.close()

if __name__=="__main__":
    raw_dir=r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\raw"
    processed_dir=r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\processed"

    for pattern, config in file_configs.items():
        input_pattern=os.path.join(raw_dir,pattern)
        base_name=pattern.replace("_*.nc","").replace(".nc","")
        output_file=os.path.join(processed_dir,f"{base_name}_processed.nc")

        if os.path.exists(output_file):
            print(f"Already processed : {output_file}! skipping this file~")
            continue

        if glob.glob(input_pattern):
            regrid_dataset(input_pattern,output_file,config)
        else:
            print(f"SKIPPING: no files found for {pattern}")



#temporal preprocessing
def average_wind_to_daily(input_path,output_path):
    print(f"Loading wind data from {input_path}")
    ds=xr.open_dataset(input_path)
    print("Averaging 6-hour data to daily~")
    ds_daily=ds.resample(time="1D").mean()

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    ds_daily.to_netcdf(output_path)
    print(f"Saved aligned wind data to: {output_path}")

    ds.close()
    ds_daily.close()
if __name__=="__main__":
    wind_input_path=r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\processed\Winds\CCMP_Wind_Analysis_processed.nc"
    wind_output_path=r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\processed\Winds\CCMP_Wind_Analysis_daily.nc"

    if os.path.exists(wind_output_path):
        print(f"Already time-aligned. Skipping {wind_output_path}")
    elif os.path.exists(wind_input_path):
        average_wind_to_daily(wind_input_path,wind_output_path)
    else:
        print(f"Processed wind data not found. Check the input path.")

#normalisation
scaler_dict={}
def normalize_file(input_path,output_path,var_name):
    print(f"Normalising {var_name}...")
    ds=xr.open_dataset(input_path)

    if isinstance(var_name, str):
        var_name = [var_name]

    for v in var_name:
        print(f"Normalising {v}....")
        var_mean=float(ds[v].mean(skipna=True))
        var_std=float(ds[v].std(skipna=True))

        scaler_dict[v]={"mean":var_mean,"std":var_std}
        ds[v]=(ds[v]-var_mean)/var_std

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()
    print(f"Successfully normalised {var_name} to {output_path}")    

if __name__=="__main__":
    processed_dir=r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\processed"
    normalized_dir = r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\normalized"

    files_to_normalize = {
        "SSH_processed.nc": "sla",
        "SSS_processed.nc": "sos",
        "SST_processed.nc": "analysed_sst",
        "glory_target_processed.nc": "thetao",
        "Winds/CCMP_Wind_Analysis_daily.nc": ["uwnd", "vwnd"],          
        "Oscar_currents/oscar_currents_final_processed.nc": ["u", "v"] 
    }

    for filename,var_name in files_to_normalize.items():
        input_path=os.path.join(processed_dir,filename)

        clean_names=filename.replace("Winds/","").replace("Oscar_currents/","")
        output_path=os.path.join(normalized_dir,clean_names)

        if os.path.exists(input_path):
            if not os.path.exists(output_path):
                if isinstance(var_name, list):
                    print(f"Normalising multiple variables in {filename}...")
                    ds = xr.open_dataset(input_path)
                    for v in var_name:
                        var_mean = float(ds[v].mean(skipna=True))
                        var_std = float(ds[v].std(skipna=True))
                        scaler_dict[v] = {"mean": var_mean, "std": var_std}
                        ds[v] = (ds[v] - var_mean) / var_std
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    ds.to_netcdf(output_path)
                    ds.close()
                    print(f"Successfully normalised variables {var_name} to {output_path}")
                
                normalize_file(input_path,output_path,var_name)
            else:
                print(f"Already normalized: {clean_names}. Skipping...")
        else:
            print(f"file not found: {input_path}")

    json_path=os.path.join(normalized_dir,"scalers.json")
    with open(json_path,"w") as f:
        json.dump(scaler_dict,f,indent=4)
    print(f"\nSaved Scaling factors to: {json_path}")



