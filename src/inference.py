import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

from data_pipeline import fetch_local_test_data, load_normalization_stats, process_for_inference

from model import OceanForecaster 


BASE_DIR = Path(__file__).resolve().parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(weights_path):
    model = OceanForecaster(input_channels=7, hidden_channels=32, kernel_size=3,output_channels=15)
    model.load_state_dict(torch.load(weights_path,map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def reverse_normalization(prediction_tensor, stats_file=None):
    if stats_file is None:
        stats_file = BASE_DIR / "data" / "normalized" / "scaler.json"

    stats = load_normalization_stats(stats_file)
    thetao_mean = stats["thetao"]["mean"]
    thetao_std = stats["thetao"]["std"]
    
    pred_array = prediction_tensor.detach().cpu().numpy().squeeze()
    
    celsius_array = (pred_array * thetao_std) + thetao_mean
    return celsius_array

def apply_land_mask(
    celsius_array, 
    bathy_path=None,
    sst_path=None
):
    if bathy_path is None:
        bathy_path = BASE_DIR / "data" / "raw" / "bathymetry.nc"
    if sst_path is None:
        sst_path = BASE_DIR / "data" / "raw" / "sst_2013_2024_combined.nc"

    bathy_ds = xr.open_dataset(bathy_path)
    ref_ds = xr.open_dataset(sst_path)

    if "depth" in bathy_ds.dims:
        bathy_ds = bathy_ds.squeeze(dim="depth", drop=True)

    b_lat = 'latitude' if 'latitude' in bathy_ds.dims else 'lat'
    b_lon = 'longitude' if 'longitude' in bathy_ds.dims else 'lon'
    r_lat = 'latitude' if 'latitude' in ref_ds.dims else 'lat'
    r_lon = 'longitude' if 'longitude' in ref_ds.dims else 'lon'

    bathy_ds = bathy_ds.interp({b_lat: ref_ds[r_lat], b_lon: ref_ds[r_lon]}, method="nearest")

    bathy_var = 'deptho' if 'deptho' in bathy_ds.data_vars else list(bathy_ds.data_vars)[0]
    depth_grid = bathy_ds[bathy_var].values.squeeze()

    land_mask = np.isnan(depth_grid) | (depth_grid == 0)

    masked_array = np.where(land_mask, np.nan, celsius_array)

    bathy_ds.close()
    ref_ds.close()

    return masked_array

def run_prediction(target_date, weights_path):
    print(f"Starting Inference for {target_date}")
    
    stats_file =BASE_DIR / "data" / "normalized" / "scaler.json"
    stats = load_normalization_stats(stats_file)
    
    print("Fetching satellite data...")
    raw_data = fetch_local_test_data(target_date)
    
    print("Formatting tensor...")
    input_tensor = process_for_inference(raw_data, stats)
    
    input_tensor = input_tensor.float().to(DEVICE)    
    model = load_trained_model(weights_path)
    
    print("Running ConvLSTM forward pass...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    print("Reversing normalization to Celsius...")
    celsius_prediction = reverse_normalization(output_tensor, stats_file)
    
    print("Applying bathymetry land mask...")
    final_map = apply_land_mask(celsius_prediction)
    
    print("\nSuccess! Final prediction shape:", final_map.shape)
    
    #Visualization
    # selected_depths = [0, 4, 8, 14]

    fig, axes = plt.subplots(3, 5, figsize=(22, 12))  # 3 rows x 5 cols = 15 subplots
    axes = axes.flatten()

    vmin = np.nanmin(final_map)
    vmax = np.nanmax(final_map)

    for depth_idx in range(15):
        im = axes[depth_idx].imshow(
            final_map[depth_idx], 
            cmap='jet', 
            origin='lower',
            vmin=vmin, 
            vmax=vmax
        )
        axes[depth_idx].set_title(f"Depth Layer {depth_idx}", fontsize=10)
        axes[depth_idx].axis('off')


    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.015, pad=0.02)
    cbar.set_label("Ocean Temperature (°C)", fontsize=12)

    plt.suptitle(f"OceanEmbed 15-Layer 3D Temperature Profile ({target_date})", fontsize=16, y=0.95)
    
    output_path = BASE_DIR / "data" / "results" / f"ocean_prediction_15_depths_{target_date}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"15-depth map successfully saved as '{output_path}'!")
    
    plt.close()
    
    plt.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OceanEmbed ConvLSTM Inference")
    parser.add_argument(
        "--date", 
        type=str, 
        default="2024-01-15", 
        help="Target date for prediction (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default=str(BASE_DIR / "data" / "ocean_model.pth"), 
        help="Path to the saved .pth model weights"
    )
    
    args = parser.parse_args()
    
    run_prediction(args.date, args.weights)