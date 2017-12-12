## Load Packages
import iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
import gsw
import iris.quickplot as qplt
import iris.analysis.stats
sys.path.append('/RESEARCH/paper_ocean_heat_carbon/code/python')
import colormaps as cmaps
import numpy.ma as ma
import scipy.interpolate as interp
import cartopy.crs as ccrs
import seaborn as sns
import cartopy.feature as cfeature

## Define Functions

def find_sigma_depth(sigma, sigma_level, depth_array, sigma_depth):
    for y in range(0, len(lons)):
        for x in range(0, len(lats)):
            f = interp.interp1d(sigma[:,x,y], depth_array, bounds_error=False)
            sigma_depth[x,y] = f(sigma_level)
    return sigma_depth

def var_on_isopycnal(var, depth, sigma_depth, var_isopycnal):
    for y in range(0, len(lons)):
        for x in range(0, len(lats)):
            f = interp.interp1d(depth, var[:,x,y], bounds_error=False)
            var_isopycnal[x,y] = f(sigma_depth[x,y])
    return var_isopycnal

## Load Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'
o2 = iris.load_cube(PATH+'o2.nc')
o2 = o2[-500:]

age = iris.load_cube(PATH+'residency_age_surface.nc')
age.coord('Time').rename('time')

neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')


## Restrict Domain to North Atlantic Ocean
constraint = iris.Constraint(latitude=lambda y: 0 < y < 90)
neutral_rho = neutral_rho.extract(constraint)
age = age.extract(constraint)
o2 = o2.extract(constraint)

constraint = iris.Constraint(longitude=lambda x: -80 < x < 5)
neutral_rho = neutral_rho.extract(constraint)
age = age.extract(constraint)
o2 = o2.extract(constraint)

# Find the average depth of the 27 neutral density surface
lons = neutral_rho.coord('longitude').points
lats = neutral_rho.coord('latitude').points
depth = neutral_rho.coord('tcell pstar').points

neutral_rho_mean = neutral_rho.collapsed('time', iris.analysis.MEAN)

depth27_mean = np.ones((42,29)) * np.nan
depth27_mean = find_sigma_depth(neutral_rho_mean.data-1000, 27, depth, depth27_mean)

# Find the depth of the 27 neutral density surface as a function of time
depth27 = np.ones((500,42,29)) * np.nan
depth27_anomalies = np.ones((500,42,29)) * np.nan
sigma_depth = np.ones((42,29)) * np.nan

for t in range(0,500):
    depth27[t,:,:] = find_sigma_depth(neutral_rho[t,:,:,:].data-1000, 27, depth, sigma_depth)
    # Find the anomaly
    depth27_anomalies[t,:,:] = depth27[t] - depth27_mean


mean = np.nanmean(depth27_anomalies, axis=0)
std = np.nanstd(depth27_anomalies, axis=0)

## Figure
fig = plt.figure(figsize=(12,10))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
plt.contourf(lons, lats, mean, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Mean', fontsize = 14)

ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
plt.contourf(lons, lats, std, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Standard Deviation', fontsize = 14)
plt.show()
