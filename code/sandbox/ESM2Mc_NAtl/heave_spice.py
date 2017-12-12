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
import iris.analysis.calculus

## Define Functions
def find_sigma_depth(sigma, sigma_level, depth_array, sigma_depth, lons, lats):
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
temp = iris.load_cube(PATH+'temp.nc')
neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')

## Restrict Domain to North Atlantic Ocean
constraint = iris.Constraint(latitude=lambda y: 0 < y < 90)
neutral_rho = neutral_rho.extract(constraint)
temp = temp.extract(constraint)

constraint = iris.Constraint(longitude=lambda x: -80 < x < 5)
neutral_rho = neutral_rho.extract(constraint)
temp = temp.extract(constraint)

## Find the Depth of the 27 surface and interpolate temperature on to that surface
# Initalize depth and variable
depth_dens     = np.ones((len(temp.coord('time').points),
                         len(temp.coord('latitude').points),
                         len(temp.coord('longitude').points))) * np.nan
temp_dens      = np.ones((len(temp.coord('latitude').points),
                         len(temp.coord('longitude').points))) * np.nan
temp_dens_time = np.ones((len(temp.coord('time').points),
                         len(temp.coord('latitude').points),
                         len(temp.coord('longitude').points))) * np.nan

# Define old coordinate arrays
lons = temp.coord('longitude').points
lats = temp.coord('latitude').points
depth = temp.coord('tcell pstar').points

for t in range(0,500):
    depth_dens[t,:,:] = find_sigma_depth(neutral_rho[t].data-1000, 27.0, depth, depth_dens[t,:,:],
                                    lons, lats)

    temp_dens = var_on_isopycnal(temp[t].data, depth, depth_dens[t,:,:], temp_dens)
    temp_dens_time[t,:,:] = temp_dens

## Calculate the derivative of temperature with time for temp on density surface
dt = 1 # year
temp_density_grad = np.gradient(temp_dens_time, dt) # Deg C / yr
temp_density_dt = temp_density_grad[0]

dt = 1 # year
depth_dens_grad = np.gradient(depth_dens, dt) # Deg C / yr
depth_dens_dt = depth_dens[0]

## Plot time = 0
plt.figure()
clevs = np.arange(-1.2, 1.3, 0.1)
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, temp_density_dt[0,:,:], clevs, cmap = 'RdBu_r', extend = 'both')
plt.colorbar()
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.title('Spice')

plt.figure()
clevs = np.arange(-1.2, 1.3, 0.1)
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, depth_dens[0,:,:], cmap = cmaps.viridis)
plt.colorbar()
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.title('depth')



plt.show()
