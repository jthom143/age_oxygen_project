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

################################################################################
#                        Correlation with Heave                                #
################################################################################

## Calculate Correlation between age and oxygen
age_o2_corr = iris.analysis.stats.pearsonr(o2, age, corr_coords=['time'])

# Calculate climatology of neutral density
neutral_rho_mean = neutral_rho.collapsed('time', iris.analysis.MEAN)
age_clim = age.collapsed('time', iris.analysis.MEAN)
o2_clim = o2.collapsed('time', iris.analysis.MEAN)

# Interpolate correlation onto neutral density surface 27 (climatology)
lons = neutral_rho.coord('longitude').points
lats = neutral_rho.coord('latitude').points
depth = neutral_rho.coord('tcell pstar').points

depth27 = np.ones((42,29)) * np.nan
depth27 = find_sigma_depth(neutral_rho_mean.data-1000, 27, depth, depth27)

corr27 = np.ones((42,29)) * np.nan
var = age_o2_corr.data
corr27 = var_on_isopycnal(var, depth, depth27, corr27)

## Calculate age and oxygen on average 27 depth
age27 = np.ones((500, 42,29)) * np.nan
o227 = np.ones((500, 42,29)) * np.nan

for t in range(0,500):
    age27 = var_on_isopycnal(age[t].data, depth, depth27, age27)

    o227 = var_on_isopycnal(o2[t].data, depth, depth27, o227)

time = o2.coord('time')
latitude = o2.coord('latitude')
longitude = o2.coord('longitude')
o2_27_no_heave = iris.cube.Cube(o227, long_name = 'Oxygen on neutral density 27',
                       dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])
age_27_no_heave = iris.cube.Cube(age27, long_name = 'Oxygen on neutral density 27',
                       dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])

################################################################################
#                     Correlation with no Heave                                #
################################################################################
depth27 = np.ones((42,29)) * np.nan
age27 = np.ones((42,29)) * np.nan
o227 = np.ones((42,29)) * np.nan

age27_t = np.ones((500,42,29)) * np.nan
o227_t = np.ones((500,42,29)) * np.nan

for t in range(0,500):
    depth27 = find_sigma_depth(neutral_rho[t].data-1000, 27, depth, depth27)

    age27 = var_on_isopycnal(age[t].data, depth, depth27, age27)
    age27_t[t,:,:] = age27

    o227 = var_on_isopycnal(o2[t].data, depth, depth27, o227)
    o227_t[t,:,:] = o227

time = o2.coord('time')
latitude = o2.coord('latitude')
longitude = o2.coord('longitude')
o2_27 = iris.cube.Cube(o227_t, long_name = 'Oxygen on neutral density 27',
                       dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])

age_27 = iris.cube.Cube(age27_t, long_name = 'Age on neutral density 27',
                       dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])

corr_age_o2_27 = iris.analysis.stats.pearsonr(o2_27, age_27, corr_coords=['time'])

o2_27_clim = o2_27.collapsed('time', iris.analysis.MEAN)

################################################################################
#                        Figure                                                #
################################################################################

# Line W:
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)

# Line 40N
constraint_lats = iris.Constraint(latitude=lambda y: 39 < y < 41)
constraint_lons = iris.Constraint(longitude=lambda x: -69 < x < -58)
o2_40N = o2.extract(constraint_lats)
o2_40N = o2_40N.extract(constraint_lons)
new_lons2 = o2_40N.coord('longitude').points
new_lats2 = np.ones((len(new_lons2), 1)) * 39.01


## Correlations
fig = plt.figure(figsize=(11,6))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
plt.contourf(lons, lats, corr27, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.plot(new_lons2, new_lats2, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) With Heave', fontsize = 12)


ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_age_o2_27.data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')


plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.plot(new_lons2, new_lats2, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Without Heave', fontsize = 12)

cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.04])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('Correlation', fontsize = 12)

plt.savefig('correlation_heave_27.png')
