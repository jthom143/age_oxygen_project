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
import cartopy.feature as cfeature
import iris.analysis.calculus

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

## Restrict analysis to the top 2000 dbars
o2 = o2[:,:21,:,:]
age = age[:,:21,:,:]
neutral_rho = neutral_rho[:,:21,:,:]

## Average age and oxygen over entire time series
o2_clim = o2.collapsed('time', iris.analysis.MEAN)
age_clim = age.collapsed('time', iris.analysis.MEAN)
neutral_rho_clim = neutral_rho.collapsed('time', iris.analysis.MEAN)

## Find maximum in age and minimum in oxygen
o2_min = o2_clim.collapsed('tcell pstar', iris.analysis.MIN)
age_max = age_clim.collapsed('tcell pstar', iris.analysis.MAX)

## Find depth of maximum and minimum in oxygen
lats = age_clim.coord('latitude').points
lons = age_clim.coord('longitude').points
depth = age_clim.coord('tcell pstar').points

o2_min_depth = np.ones((len(lats), len(lons)))*np.nan
age_max_depth = np.ones((len(lats), len(lons)))*np.nan

o2_min_sigma = np.ones((len(lats), len(lons)))*np.nan
age_max_sigma = np.ones((len(lats), len(lons)))*np.nan

for x in range(0,len(lons)):
    for y in range(0,len(lats)):
        o2_min_ind = np.argmin(o2_clim.data, axis=0)
        age_max_ind = np.argmax(age_clim.data, axis=0)

        o2_min_depth[y,x] = depth[o2_min_ind[y,x]]
        age_max_depth[y,x] = depth[age_max_ind[y,x]]

        o2_min_sigma[y,x] = neutral_rho_clim[:,y,x].data[o2_min_ind[y,x]]
        age_max_sigma[y,x] = neutral_rho_clim[:,y,x].data[age_max_ind[y,x]]


## Figure
fig = plt.figure(figsize=(12,10))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(0, 325, 25)
plt.contourf(lons, lats, age_max.data, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Age Maximum', fontsize = 14)

ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(0, 350, 25)
plt.contourf(lons, lats, o2_min.data*1e6, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Oxygen Minimum', fontsize = 14)




fig = plt.figure(figsize=(12,10))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(0, 2150, 150)
plt.contourf(lons, lats, age_max_depth, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Depth of Age Maximum', fontsize = 14)

ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_min_depth,clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Depth of Oxygen Minimum', fontsize = 12)


fig = plt.figure()
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(-800, 900, 100)
plt.contourf(lons, lats, age_max_depth - o2_min_depth, clevs, cmap = 'RdBu_r', extend = 'both')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('Difference', fontsize = 12)

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(22, 28.1, .1)
plt.contourf(lons, lats, age_max_sigma-1000, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Neutral Density of Age Maximum', fontsize = 14)

ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_min_sigma-1000, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Neutral Density of Oxygen Minimum', fontsize = 14)

fig = plt.figure()
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
plt.contourf(lons, lats, age_max_sigma - o2_min_sigma, clevs, cmap = 'RdBu_r', extend = 'both')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('Difference', fontsize = 14)


plt.show()
