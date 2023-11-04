import os
import numpy as np
import time
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F


class hdf5_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_file, resolution, shapes_per_epoch=1):
        self.data_dir = data_dir
        self.data_file = data_file
        self.resolution = resolution

        data_hdf5_name = self.data_dir+'/'+self.data_file+'.hdf5'

        data_dict = h5py.File(data_hdf5_name, 'r')
        self.data_len = len(data_dict['voxels'])
        self.data_points = data_dict['points_'+str(self.resolution)][:]
        self.data_values = data_dict['values_'+str(self.resolution)][:]
        self.data_voxels = data_dict['voxels'][:,:,:,:,0].astype(np.uint8)
        data_dict.close()

        #fix hollow shapes by filling invisible voxels
        #only do this for "02773838_bag" and "02954340_cap" and "03948459_pistol" to save time
        voxel_idxs = np.array(range(resolution), np.int64)
        voxel_x, voxel_y, voxel_z = np.meshgrid(voxel_idxs,voxel_idxs,voxel_idxs, sparse=False, indexing='ij')
        if "02773838" in data_hdf5_name or "02954340" in data_hdf5_name or "03948459" in data_hdf5_name:
            for idx in range(len(self.data_voxels)):
                voxels = self.data_voxels[idx]
                points = self.data_points[idx]
                values = self.data_values[idx]

                #downsample if resolution!=64
                if resolution<64:
                    voxels = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(voxels[0::2,0::2,0::2],voxels[1::2,0::2,0::2]),voxels[0::2,1::2,0::2]),voxels[1::2,1::2,0::2]),voxels[0::2,0::2,1::2]),voxels[1::2,0::2,1::2]),voxels[0::2,1::2,1::2]),voxels[1::2,1::2,1::2])
                if resolution<32:
                    voxels = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(voxels[0::2,0::2,0::2],voxels[1::2,0::2,0::2]),voxels[0::2,1::2,0::2]),voxels[1::2,1::2,0::2]),voxels[0::2,0::2,1::2]),voxels[1::2,0::2,1::2]),voxels[0::2,1::2,1::2]),voxels[1::2,1::2,1::2])

                mask_x = 1-np.max(voxels,0).astype(np.int64)
                depth_x0 = np.argmax(voxels,0) + 65536*mask_x
                voxels_ = np.ascontiguousarray(voxels[::-1,:,:])
                depth_x1 = resolution-1-np.argmax(voxels_,0) - 65536*mask_x

                mask_z = 1-np.max(voxels,2).astype(np.int64)
                depth_z0 = np.argmax(voxels,2) + 65536*mask_z
                voxels_ = np.ascontiguousarray(voxels[:,:,::-1])
                depth_z1 = resolution-1-np.argmax(voxels_,2) - 65536*mask_z

                filled_voxels = (voxel_x>=depth_x0[None,...]) & (voxel_x<=depth_x1[None,...]) & (voxel_z>=depth_z0[...,None]) & (voxel_z<=depth_z1[...,None])
                filled_voxels = filled_voxels.astype(np.uint8)

                values = filled_voxels[points[:,0],points[:,1],points[:,2]]

                self.data_values[idx] = values[...,None]

        #convert to float
        self.data_points = (self.data_points.astype(np.float32)+0.5)/self.resolution-0.5 #[-0.5,0.5]
        self.data_values = self.data_values.astype(np.float32)

        #set the number of shapes to shapes_per_epoch or larger
        self.repeat_times = (shapes_per_epoch+self.data_len-1)//self.data_len
        self.faked_data_len = self.data_len*self.repeat_times


    def __len__(self):
        return self.faked_data_len

    def __getitem__(self, index):
        index = index%self.data_len
        if self.resolution==64:
            ridx = np.random.randint(4)
            points = self.data_points[index,ridx::4]
            values = self.data_values[index,ridx::4]
        else:
            points = self.data_points[index]
            values = self.data_values[index]
        voxels = self.data_voxels[index,None,...]
        return points, values, voxels
