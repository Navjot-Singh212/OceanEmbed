import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

class OceanDataset(Dataset):
    def __init__(self,normalised_dir,sequence_length=7,lazy=True):
        self.normalised_dir=normalised_dir
        self.sequence_length=sequence_length
        self.lazy=lazy

        self.file_map={
           "sst": ("SST_processed.nc", "analysed_sst"),
            "ssh": ("SSH_processed.nc", "sla"),
            "sss": ("SSS_processed.nc", "sos"),
            "winds": ("CCMP_Wind_Analysis_daily.nc", None),
            "currents": ("oscar_currents_final_processed.nc", None),
            "target": ("glory_target_processed.nc", "thetao")
        }
        ref_path=os.path.join(self.normalised_dir,self.file_map["sst"][0])
        with xr.open_dataset(ref_path) as ds:
            self.total_days=ds.sizes["time"]

        self.ds_handles=None

        if not self.lazy:
            self.load_all_to_memory()

    def _init_lazy_handles(self):
        self.ds_handles={}
        for key, (fname,var_name) in self.file_map.items():
            file_path=os.path.join(self.normalised_dir,fname)
            if var_name is None:
                self.ds_handles[key] = xr.open_dataset(file_path)
            else:
                self.ds_handles[key]=xr.open_dataset(file_path)[var_name]

    def load_all_to_memory(self):
        #this is just for the sample test case btw cuz if i use the actual large dataset it will give memory error
        print("Pre-loading entire data into RAM...")
        self.data_in_memory={}
        for key, (fname,var_name) in self.file_map.items():
            file_path=os.path.join(self.normalised_dir,fname)
            if var_name is None:
                self.data_in_memory[key] = xr.open_dataset(file_path)
            else:
                self.data_in_memory[key]=xr.open_dataset(file_path)[var_name].values

    def __len__(self):
        #gives the number of sequences
        return self.total_days-self.sequence_length

    def __getitem__(self, idx):
        time_slice=slice(idx,idx+self.sequence_length)
        target_idx=idx+self.sequence_length

        if self.lazy:
            if self.ds_handles is None:
                self._init_lazy_handles()

            past_sst = self.ds_handles["sst"].isel(time=time_slice).values
            past_ssh = self.ds_handles["ssh"].isel(time=time_slice).values
            past_sss = self.ds_handles["sss"].isel(time=time_slice).values.squeeze()
            past_wind_u = self.ds_handles["winds"]["uwnd"].isel(time=time_slice).values
            past_wind_v = self.ds_handles["winds"]["vwnd"].isel(time=time_slice).values
            past_curr_u = self.ds_handles["currents"]["u"].isel(time=time_slice).values
            past_curr_v = self.ds_handles["currents"]["v"].isel(time=time_slice).values

            target_thetao = self.ds_handles["target"].isel(time=target_idx).values
        else:
            past_sst = self.data_in_memory["sst"][time_slice]
            past_ssh = self.data_in_memory["ssh"][time_slice]
            past_sss = self.data_in_memory["sss"][time_slice].squeeze()
            past_wind_u = self.data_in_memory["winds"]["uwnd"].isel(time=time_slice).values
            past_wind_v = self.data_in_memory["winds"]["vwnd"].isel(time=time_slice).values
            past_curr_u = self.data_in_memory["currents"]["u"].isel(time=time_slice).values
            past_curr_v = self.data_in_memory["currents"]["v"].isel(time=time_slice).values

            target_thetao = self.data_in_memory["target"][target_idx]
            
        if past_curr_u.shape != past_sst.shape:
            past_curr_u = np.swapaxes(past_curr_u, 1, 2)
            past_curr_v = np.swapaxes(past_curr_v, 1, 2)

        depth_indices = [0, 3, 7, 11, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35]  #extracted directly from the Copernicus design for ocean depth levels from 0.49m to 1062.44m    
        target_thetao=target_thetao[depth_indices,:,:]


        x_data=np.stack([past_sst, past_ssh, past_sss, past_curr_u, past_curr_v, past_wind_u, past_wind_v],axis=0)
        x_data=np.swapaxes(x_data,0,1)

        if len(target_thetao.shape)==2:
            target_thetao=np.expand_axis(target_thetao,axis=0) if hasattr(np,"expand_axis") else np.expand_dims(target_thetao,axis=0)

        x_tensor=torch.tensor(x_data,dtype=torch.float32)
        y_tensor=torch.tensor(target_thetao,dtype=torch.float32)

        return x_tensor,y_tensor

if __name__=="__main__":
    norm_dir = r"D:\Coding and programming\Hackathon Projects\OceanEmbed\data\normalized"
    dataset=OceanDataset(normalised_dir=norm_dir,sequence_length=7,lazy=False)
    print(f"Total trainable sequence in dataset: {len(dataset)}")

    #lets just pull one sample sequence kinda wanna see it's shape and whats in it
    x_sample,y_sample=dataset[0]
    print("TENSOR SHAPE VERIFICATION~")
    print(f"Input (X) shape: {x_sample.shape} -> (Time, Channels, Height, Width)")
    print(f"Target Tensor (y) shape : {y_sample.shape}  -> (Channels, Height, Width)")

    loader=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0)
    batch_x,batch_y=next(iter(loader))

    print("DATA LOADER SHAPE VERIFICATION")
    print(f"Batch Input (X) shape  : {batch_x.shape} -> (Batch, Time, Channels, Height, Width)")
    print(f"Batch Target (Y) shape : {batch_y.shape} -> (Batch, Channels, Height, Width)")
        

                

        

