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

### Define Functions

def find_sigma_depth(sigma, sigma_level, depth_array, sigma_depth):
    for y in range(0, len(distance)):
        f = interp.interp1d(sigma[:,y], depth_array, bounds_error=False)
        sigma_depth[y] = f(sigma_level)
    return sigma_depth

def var_on_isopycnal(var, depth, sigma_depth, var_isopycnal):
    for t in range(0, 500):
        for y in range(0, len(distance)):
            f = interp.interp1d(depth, var[t,:,y], bounds_error=False)
            var_isopycnal[t,y] = f(sigma_depth[y])
    return var_isopycnal




### Load Line W Data
o2_linew = iris.load_cube('o2_linew.nc')
age_linew = iris.load_cube('age_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')
time = np.arange(0, 500, 1)
### Line W Calculations
# Calculate Climatology of Neutral Density
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)
age_linew_clim = age_linew.collapsed('time', iris.analysis.MEAN)
o2_linew_clim = o2_linew.collapsed('time', iris.analysis.MEAN)

# Calculate the Correlation between age and oxygen
corr = iris.analysis.stats.pearsonr(o2_linew, age_linew, corr_coords=['time'])

# Calculate the depth of the 27 isopycnal Climatology
distance = age_linew.coord('distance').points
depth = age_linew.coord('tcell pstar').points

depth27 = np.ones((len(distance))) * np.nan
depth27 = find_sigma_depth(neutral_rho_linew_mean.data-1000, 27, depth, depth27)

age_linew_27 = np.ones((500,len(distance))) * np.nan
var = age_linew - age_linew_clim
age_linew_27 = var_on_isopycnal(var.data, depth, depth27, age_linew_27)

o2_linew_27 = np.ones((500,len(distance))) * np.nan
var = o2_linew - o2_linew_clim
o2_linew_27 = var_on_isopycnal(var.data, depth, depth27, o2_linew_27)


### Create Figure
plt.figure()
clevs = np.arange(-30, 31, 1)
ax = plt.subplot(1,2,1)
im = plt.contourf(distance, time, age_linew_27, clevs, cmap = 'RdBu_r')
plt.ylim([0,250])
ax.invert_yaxis()
plt.title('(a) Age')
plt.colorbar(im, orientation = 'horizontal')
plt.ylabel('Time [years]')
plt.xlabel('Distance [km]')

ax = plt.subplot(1,2,2)
im2 = plt.contourf(distance, time, o2_linew_27*1e6, clevs, cmap = 'RdBu_r')
plt.ylim([0,250])
ax.invert_yaxis()
plt.title('(b) Oxygen')
plt.colorbar(im2, orientation = 'horizontal')
plt.xlabel('Distance [km]')

plt.show()
