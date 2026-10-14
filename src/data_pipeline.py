import torch
import json
import numpy as np
import xarray as xr
import pandas as pd
import os
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent

def load_normalization_stats(json_path=None):
    if json_path is None:
        json_path = BASE_DIR / "data" / "normalized" / "scaler.json"

    with open(json_path, 'r') as f:
        return json.load(f)

def process_for_inference(raw_data_dict, stats):
   
    ordered_vars = ['analysed_sst', 'sla', 'sos', 'u', 'v', 'uwnd', 'vwnd']
    
    normalized_channels = []
    
    for var in ordered_vars:
        mean_val = stats[var]["mean"]
        std_val = stats[var]["std"]
        
        raw_array = raw_data_dict[var]
        
        normalized_array = (raw_array - mean_val) / std_val
        
        clean_array = np.nan_to_num(normalized_array, nan=0.0)
        
        normalized_channels.append(clean_array)
        
    x_data = np.stack(normalized_channels, axis=0)
    
    x_data = np.swapaxes(x_data, 0, 1)
    
    final_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0)
    
    return final_tensor

def fetch_local_test_data(target_date, raw_dir=None):
    if raw_dir is None:
        raw_dir = BASE_DIR / "data" / "raw"

    end_dt = pd.to_datetime(target_date)
    start_dt = end_dt - pd.Timedelta(days=6)
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    file_map = {
        "analysed_sst": "sst_2013_2024_combined.nc",
        "sla": "ssh_2013_2024_combined.nc",
        "sos": "sss_2013_2024_combined.nc",
        "u": "oscar_yearly/*.nc",
        "v": "oscar_yearly/*.nc",
        "uwnd": "ccmp_yearly/*.nc",
        "vwnd": "ccmp_yearly/*.nc"
    }
    
    raw_data_dict = {}
    
    ref_path = os.path.join(raw_dir, file_map["analysed_sst"])
    ref_ds = xr.open_dataset(ref_path).sel(time=slice(start_str, end_str))
    
    ref_lat_key = 'latitude' if 'latitude' in ref_ds.dims else 'lat'
    ref_lon_key = 'longitude' if 'longitude' in ref_ds.dims else 'lon'

    print("\nProcessing Variable Shapes")
    
    for var, path_suffix in file_map.items():
        full_path = os.path.join(raw_dir, path_suffix)
        
        if "*" in path_suffix:
            ds = xr.open_mfdataset(full_path, combine='by_coords')
        else:
            ds = xr.open_dataset(full_path)
            
        if "depth" in ds.dims:
            ds = ds.squeeze(dim="depth", drop=True)
            
        sliced_ds = ds.sel(time=slice(start_str, end_str))
        
        if len(sliced_ds.time) > 7:
            sliced_ds = sliced_ds.resample(time='1D').mean()
            
        if var != "analysed_sst":
            orig_lat = 'latitude' if 'latitude' in sliced_ds.dims else 'lat'
            orig_lon = 'longitude' if 'longitude' in sliced_ds.dims else 'lon'
            sliced_ds = sliced_ds.interp({orig_lat: ref_ds[ref_lat_key], orig_lon: ref_ds[ref_lon_key]}, method="nearest")
        
        da = sliced_ds[var]

        final_lat = 'latitude' if 'latitude' in da.dims else 'lat'
        final_lon = 'longitude' if 'longitude' in da.dims else 'lon'
        
        da = da.transpose("time", final_lat, final_lon)
        
        if len(da.time) != 7:
            ds.close()
            ref_ds.close()
            raise ValueError(f"Found {len(da.time)} days for {var} instead of 7.")
            
        arr = da.values
        raw_data_dict[var] = arr
        print(f"  {var:12s} -> shape: {arr.shape}")
        ds.close()
        
    ref_ds.close()
    return raw_data_dict

if __name__ == "__main__":
    test_date = "2024-05-15" 
    stats_file =BASE_DIR / "data" / "normalized" / "scaler.json" 
    
    print("Loading stats...")
    stats = load_normalization_stats(stats_file)
    
    print(f"Fetching 7 days of raw data ending on {test_date}...")
    raw_data = fetch_local_test_data(test_date)
    
    print("Normalizing and stacking tensor...")
    inference_tensor = process_for_inference(raw_data, stats)
    
    print(f"Success! Model-ready tensor shape: {inference_tensor.shape}")