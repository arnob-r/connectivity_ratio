# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 02:07:50 2025

@author: ar.ray
"""
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cf
import time
from scipy import stats
from scipy.sparse.csgraph import dijkstra
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
start_time = time.time()

###############################################################################
# Degree
ds = xr.open_dataset('tas_day_EC-Earth3_historical_r6i1p1f1_gr_19850101-20141231_JJA.nc')                               
da = ds.tas[:,:,:].values
ntime, nlat, nlon = da.shape
da2d = da.reshape(ntime, (nlat*nlon), order = "F")
Mat = np.transpose(da2d)
cc_mat = np.corrcoef(Mat)   

# Significance testing using Beta distribution
n = 90
dist = stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
pvals = 2 * dist.cdf(-abs(cc_mat[np.triu_indices(cc_mat.shape[0], k=1)]))
_, corrected_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

# Thresholding
P = np.zeros_like(cc_mat)
P[np.triu_indices(cc_mat.shape[0], k=1)] = corrected_pvals < 0.05
P = P + P.T
cc_mat_new = cc_mat * P
M = cc_mat_new - np.eye(cc_mat_new.shape[0])
corr_mat = abs(M)
threshold = 0.5
adjacency_summer = np.where(corr_mat > threshold, 1, 0)

np.save("tas_day_EC-Earth3_historical_r6i1p1f1_adjacency.npy", adjacency_summer)  

# Latitude weights
arr1 = ds.tas['lat'].values
arr2 = ds.tas['lon'].values
A = np.zeros((nlat, nlon))
for i in range(0, nlat):
    A[i, :] = np.cos(np.deg2rad(arr1[i]))       
A1 = A.reshape(nlat*nlon, order='F')
sqrt_weights = np.sqrt(A1)
adjacency_summer_weighted = adjacency_summer * sqrt_weights[:, None] * sqrt_weights[None, :]

def node_degrees(adjacency_matrix):
    return np.sum(adjacency_matrix, axis=1)

degree=node_degrees(adjacency_summer_weighted)
# reshaping degee
c1=degree
c1=c1.reshape((nlat,nlon),order='F')

cmap="Reds"
matplotlib.rcParams['font.size'] = 28      
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, figsize=(8,6))

for spine in ax.spines.values():
        spine.set_linewidth(4)  # Set your desired linewidth here
# Set the extent of the map (longitude and latitude range)
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
levels=np.arange(0,2100,100)
im=ax.contourf(ds.lon, ds.lat, c1, levels = levels,
            cmap=cmap,transform=ccrs.PlateCarree(), extend="max", add_colorbar=True)
ax.coastlines()
#cb=plt.colorbar(im,ax=ax,orientation="horizontal",shrink=0.95, pad=0.1,aspect=50,
#                drawedges=False,label="Degree")
cb=plt.colorbar(im,ax=ax,orientation="horizontal", pad=0.1, aspect=35, drawedges=False)
cb.set_label('Degree', fontsize=28, fontweight='bold')
cb.set_ticks([0, 400, 800, 1200, 1600, 2000])

cb.outline.set_linewidth(4)
cb.ax.tick_params(labelsize=28) 
for label in cb.ax.get_xticklabels():
    label.set_fontweight('bold')       
ax.add_feature(cf.LAND, edgecolor='black', facecolor='white')
ax.add_feature(cf.OCEAN, edgecolor='none', facecolor='white')
ax.add_feature(cf.LAKES, edgecolor='black', facecolor='lightblue') 
#ax.add_feature(cf.RIVERS)
ax.add_feature(cf.STATES)
ax.add_feature(cf.BORDERS)
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0) 
gl.right_labels=False
gl.top_labels=False
gl.xlabel_style = {'size': 28, 'weight': 'bold'}
gl.ylabel_style = {'size': 28, 'weight': 'bold'}
plt.savefig(f"r6_deg.png",dpi=600,bbox_inches='tight') 

############################################################################### 
D=np.load('distance_lat_lon.npy')
result = D * adjacency_summer
###############################################################################
# Number of spatial short-range connectivity
np_result = np.array(result)
#max_values = np.max(np_result, axis=1)
count1 = np.sum((np_result < 5000) & (np_result > 0), axis=1)
#   print(count1)    
M2 = np.reshape(count1, nlat*nlon, order='F')    
M2 = M2.reshape((nlat,nlon),order='F') 
cmap="Reds"
matplotlib.rcParams['font.size'] = 28  
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, figsize=(8,6))

for spine in ax.spines.values():
        spine.set_linewidth(4)  # Set your desired linewidth here
# Set the extent of the map (longitude and latitude range)
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
levels=np.arange(0,1050,50)
im=ax.contourf(ds.lon, ds.lat, M2, levels=levels, 
            cmap=cmap,transform=ccrs.PlateCarree(),extend="max", add_colorbar=True)
ax.coastlines()
#cb=plt.colorbar(im,ax=ax,orientation="horizontal",shrink=0.95, pad=0.1,aspect=50,
#                drawedges=False,label="Degree")
cb=plt.colorbar(im,ax=ax,orientation="horizontal", pad=0.1, aspect=35, drawedges=False)
cb.set_label(f'No. of link lengths less than 5000 km', fontsize=24, fontweight='bold') 
cb.set_ticks([0, 200, 400, 600, 800, 1000])
cb.outline.set_linewidth(4)
cb.ax.tick_params(labelsize=28) 
for label in cb.ax.get_xticklabels():
    label.set_fontweight('bold') 
ax.add_feature(cf.LAND, edgecolor='black', facecolor='white')
ax.add_feature(cf.OCEAN, edgecolor='none', facecolor='white')
ax.add_feature(cf.LAKES, edgecolor='black', facecolor='lightblue') 
#ax.add_feature(cf.RIVERS)
ax.add_feature(cf.STATES)
ax.add_feature(cf.BORDERS)
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0) 
gl.right_labels=False
gl.top_labels=False
gl.xlabel_style = {'size': 28, 'weight': 'bold'}
gl.ylabel_style = {'size': 28, 'weight': 'bold'} 
plt.savefig(f"r6_5000.png",dpi=600,bbox_inches='tight')

###############################################################################
# Number of spatial long-range connectivity
np_result = np.array(result)
max_values = np.max(np_result, axis=1)
count2 = np.sum(np_result > 10000, axis=1)
#   print(count)    
M1 = np.reshape(count2, nlat*nlon, order='F')    
M1 = M1.reshape((nlat,nlon),order='F') 
cmap="Reds"
#cmap="YlOrBr"
matplotlib.rcParams['font.size'] = 28   
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, figsize=(8,6))

for spine in ax.spines.values():
        spine.set_linewidth(4)  # Set your desired linewidth here
# Set the extent of the map (longitude and latitude range)
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
levels=np.arange(0,1050,50)
im=ax.contourf(ds.lon, ds.lat, M1, levels=levels,
            cmap=cmap,transform=ccrs.PlateCarree(), extend="max", add_colorbar=True)
ax.coastlines()
#cb=plt.colorbar(im,ax=ax,orientation="horizontal",shrink=0.95, pad=0.1,aspect=50,
#                drawedges=False,label="Degree")
cb=plt.colorbar(im,ax=ax,orientation="horizontal", pad=0.1, aspect=35, drawedges=False)
cb.set_label(f'No. of link lengths exceeding 10000 km', fontsize=23, fontweight='bold') 
cb.set_ticks([0, 200, 400, 600, 800, 1000])
cb.outline.set_linewidth(4)
cb.ax.tick_params(labelsize=28) 
for label in cb.ax.get_xticklabels():
    label.set_fontweight('bold') 
ax.add_feature(cf.LAND, edgecolor='black', facecolor='white')
ax.add_feature(cf.OCEAN, edgecolor='none', facecolor='white')
ax.add_feature(cf.LAKES, edgecolor='black', facecolor='lightblue') 
#ax.add_feature(cf.RIVERS)
ax.add_feature(cf.STATES)
ax.add_feature(cf.BORDERS)
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0) 
gl.right_labels=False 
gl.top_labels=False 
gl.xlabel_style = {'size': 28, 'weight': 'bold'}
gl.ylabel_style = {'size': 28, 'weight': 'bold'}

plt.savefig(f"r6_10000.png",dpi=600,bbox_inches='tight')

###############################################################################
# CR calculation
sum_count1 = sum(count1)
sum_count2 = sum(count2)
count3 = sum_count2/sum_count1
count3 = round(count3, 3)
print(count3)
############################################################################### 
# Normalized frequency plot
upper_part_elements = [result[i, j] for i in range(result.shape[0]) for j in range(i+1, result.shape[1])]
filtered_list = [x for x in upper_part_elements if x != 0]
data = filtered_list
hist, bins = np.histogram(data, bins=50, density=True)
bin_centers = (bins[1:] + bins[:-1]) * 0.5
# Normalize histogram
hist = hist/hist.sum()
width=bins[1] - bins[0]

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
fig.subplots_adjust(top=0.85)
plt.bar(bin_centers, hist, width=bins[1] - bins[0], alpha=1.0, edgecolor='black', facecolor='None', linewidth=2)
#plt.plot(hist, color = 'blue')

for spine in plt.gca().spines.values():
    spine.set_linewidth(4)
plt.xlabel('Distance(km)', fontsize=30, fontweight='bold')
plt.ylabel('Link density', fontsize=30, fontweight='bold')
plt.xticks(ticks=[0, 10000, 20000], labels=['0', '10000', '20000'],fontsize=30, fontweight='bold')
plt.yticks(ticks=[0, 0.1, 0.2, 0.3], labels=['0', '0.1', '0.2', '0.3'], fontsize=30, fontweight='bold')

#plt.xticks(fontsize=14)
txt = f'$\mathit{{CR}}$ = {count3}' 
plt.text(0.95, 0.95, txt, transform=plt.gca().transAxes, fontsize=30, color='black', ha='right', va='top', fontweight='bold')

plt.savefig(f"r6_hist.png",dpi=600,bbox_inches='tight')

###############################################################################

end_time = time.time()

# Calculate total time taken
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
