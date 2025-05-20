#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:22:31 2023

@author: cssc
"""
# merging of 30 individual data sets 
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx 
import os
import glob
import warnings
import scipy.stats as stats
import scipy.io
import cartopy.crs as ccrs
import cartopy.feature as cf
import sys
from itertools import product
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
warnings.filterwarnings("ignore")
import time

# Record the starting time
start_time = time.time()

###############################################################################

ds=xr.open_dataset('tas_day_EC-Earth3_historical_r6i1p1f1_gr_19850101-20141231_JJA.nc')         #change

print(ds)
ds.time.values

da = ds.tas[:,:,:].values 
ntime, nlat, nlon = da.shape
da2d = da.reshape(ntime, (nlat*nlon), order = "F")
###############################################################################
lat = ds.tas['lat'].values   # latitude
lon = ds.tas['lon'].values   # longitude

latlonpoints=[]

for i in range(nlon):
    for j in range(nlat):
        idx=[lon[i],lat[j]]
        latlonpoints.append(idx)

nodes=np.array(latlonpoints); print(nodes.shape)

lat=nodes[:,1]
lon=nodes[:,0]

vcrd = np.vstack((lat,lon)).T

# Set the region 
ctr_lat0, ctr_lat1, ctr_lon0, ctr_lon1 = -18, 5,  360-75, 360-45  #(Amazon), -12, 18, 40, 80 (IO), -5, 5, 360-170, 360-120 (Nino)                

lat0, lat1, lon0, lon1 = np.min(vcrd[:, 0]), np.max(vcrd[:, 0]), np.min(vcrd[:, 1]), np.max(vcrd[:, 1])
bm_lac, bm_loc = (lat0 + lat1) / 2, (lon0 + lon1) / 2

##### Region or box selecting by us
reg_ctr = np.where((vcrd[:, 0] >= ctr_lat0) & (vcrd[:, 0] <= ctr_lat1) & (vcrd[:, 1] >= ctr_lon0) & (vcrd[:, 1] <= ctr_lon1))[0]

###############################################################################

cc_mat=np.load('tas_day_EC-Earth3_historical_r6i1p1f1_adjacency.npy')               #change

A=np.zeros((nlat, nlon))
for i in range(0,nlat):
#    print(arr1[i])
#    A[i].append([])
    for j in range(0,nlon):
#        print(arr2[j])
        A[i][j] = np.cos(np.deg2rad(lat[i]) )       

A1 = np.reshape(A, nlat*nlon, order='F') 
cc_mat = cc_mat * A1                                   # Create weighted matrix

###############################################################################
# weighted adjacency matrix
net = cc_mat 
edg = np.empty((0, 2), dtype=np.int32)
nbr = np.array([], dtype=np.int32)

for v in reg_ctr:
    adj_nbr = np.where(net[v] != 0)[0]
    nbr = np.concatenate((nbr, adj_nbr))

unq, cnt = np.unique(nbr, return_counts=True)   # Unique Values with Counts
rd = np.zeros(nodes.shape[0], dtype=np.int32)
rd[unq] = cnt.astype(np.int32)

#ax=plt.figure().add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
#lat = np.arange(lat0, lat1 + 0.01, 1.0)        # Change
#lon = np.arange(lon0, lon1 + 0.01, 1.0)        # Change

lat = np.arange(lat0, lat1 + 0.01, 2.0)        # Change here to 2.0 for 2 degree * 2 degree 
lon = np.arange(lon0, lon1 + 0.01, 2.0)        # Change here to 2.0 for 2 degree * 2 degree

crd = np.array(list(product(lat, lon))).astype(np.float32)
data = np.zeros(crd.shape[0], dtype=np.int32)

for i, v in enumerate(crd):
    idx = np.where((vcrd[:, 0] == v[0]) & (vcrd[:, 1] == v[1]))[0]
    if idx.size != 0:
        data[i] = rd[idx[0]]
        
    #print(np.max(data))

############################################################################### 
cmap='Reds'
matplotlib.rcParams['font.size'] = 28    
#fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, figsize=(8,6))
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, figsize=(8,6))

for spine in ax.spines.values():
        spine.set_linewidth(4)
lo, la = np.meshgrid(lon, lat)
#levels=np.arange(0,max(data))
levels=np.arange(0,190,10)
cf_rd = ax.contourf(lo, la, data.reshape((lat.shape[0], lon.shape[0])), extend='max', transform=ccrs.PlateCarree(), cmap=cmap, levels=levels)
cb=plt.colorbar(cf_rd,ax=ax,orientation="horizontal",pad=0.1,aspect=35, drawedges = False)
cb.set_label('Partial degree', fontsize=28, fontweight='bold') 
cb.ax.tick_params(labelsize=28) 
cb.set_ticks([0, 30, 60, 90, 120, 150, 180])


cb.outline.set_linewidth(4)
for label in cb.ax.get_xticklabels():
    label.set_fontweight('bold') 
ax.coastlines()
ax.add_feature(cf.STATES)
ax.add_feature(cf.BORDERS)
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.0)
gl.right_labels=False
gl.top_labels=False
gl.xlabel_style = {'size': 28, 'weight': 'bold'}
gl.ylabel_style = {'size': 28, 'weight': 'bold'}

patches = []
reg_ctr = np.array([[ctr_lon0, ctr_lat0],
                    [ctr_lon0, ctr_lat1],
                    [ctr_lon1, ctr_lat1],
                    [ctr_lon1, ctr_lat0]])

patches.append(Polygon(reg_ctr, closed=True))
ax.add_collection(PatchCollection(patches, facecolor='None', edgecolor='k', linewidths=4.0, zorder=2.0, transform=ccrs.PlateCarree()))
idx=np.nonzero(data);print(data[idx])
plt.savefig(f"pl_Amz_r6.png",dpi=600,bbox_inches='tight')  # change
###############################################################################
D=np.load('distance_lat_lon.npy') 
cc_mat=np.load('tas_day_EC-Earth3_historical_r6i1p1f1_adjacency.npy')               #change   
cc_mat1 = np.zeros_like(cc_mat)

    # Preserve the specified rows and columns
for i in reg_ctr:
    cc_mat1[i, :] = cc_mat[i, :]  # Preserve the entire row
    cc_mat1[:, i] = cc_mat[:, i]  # Preserve the entire column

result = D * cc_mat1
np_result = np.array(result)

count1 = np.sum((np_result < 5000) & (np_result > 0), axis=1)
count2 = np.sum(np_result > 10000, axis=1)
sum_count1 = sum(count1)
sum_count2 = sum(count2)
count3 = sum_count2/sum_count1
count3 = round(count3, 3)
print(count3)  # Calculation of CR

end_time = time.time()

# Calculate total time taken
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")

