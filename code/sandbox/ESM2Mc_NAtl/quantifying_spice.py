## Load Packages
import iris
import numpy as np
import matplotlib.pyplot as plt
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
#                     Correlation with no Heave                                #
################################################################################
depth27 = np.ones((42,29)) * np.nan
age27 = np.ones((42,29)) * np.nan
o227 = np.ones((42,29)) * np.nan

age27_t = np.ones((500,42,29)) * np.nan
o227_t = np.ones((500,42,29)) * np.nan

depth = o2.coord('tcell pstar').points
lons = o2.coord('longitude').points
lats = o2.coord('latitude').points

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


# Average over line w region
constraint_lats = iris.Constraint(latitude=lambda y: 32 < y < 34)
constraint_lons = iris.Constraint(longitude=lambda y: -66 < y < -64)

o2_27_linew_end = o2_27.extract(constraint_lats)
o2_27_linew_end = o2_27_linew_end.extract(constraint_lons)

age_27_linew_end = age_27.extract(constraint_lats)
age_27_linew_end = age_27_linew_end.extract(constraint_lons)

# Calculate Spice
age_spice = iris.analysis.calculus.differentiate(age_27, 'time')
o2_spice = iris.analysis.calculus.differentiate(o2_27, 'time')

age_spice_std = age_spice.collapsed('time', iris.analysis.STD_DEV)
o2_spice_std = o2_spice.collapsed('time', iris.analysis.STD_DEV)


# Isolate Line 40N End:
constraint_lats = iris.Constraint(latitude=lambda y: 39 < y < 41)
constraint_lons = iris.Constraint(longitude=lambda x: -59 < x < -58)

o2_27_line40N_end = o2_27.extract(constraint_lats)
o2_27_line40N_end = o2_27_line40N_end.extract(constraint_lons)

age_27_line40N_end = age_27.extract(constraint_lats)
age_27_line40N_end = age_27_line40N_end.extract(constraint_lons)

# Calculate Derivative
age_spice_linew = iris.analysis.calculus.differentiate(age_27_linew_end, 'time')
o2_spice_linew = iris.analysis.calculus.differentiate(o2_27_linew_end, 'time')

age_spice_line40N = iris.analysis.calculus.differentiate(age_27_line40N_end, 'time')
o2_spice_line40N = iris.analysis.calculus.differentiate(o2_27_line40N_end, 'time')

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
ax = plt.subplot(2,1,1)
plt.plot(age_spice_linew.data, color = 'k', lw = 1.5)
plt.plot(age_spice_line40N.data, color = 'g', lw = 1.5)
plt.title('(a) Age Spice')

ax = plt.subplot(2,1,2)
plt.plot(o2_spice_linew.data*1e6, color = 'k', lw = 1.5, label = 'Line W')
plt.plot(o2_spice_line40N.data*1e6, color = 'g', lw = 1.5, label = 'Line 40N')
plt.title('(b) Oxygen Spice')
plt.legend()






# Line W:
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)

# Line 40N
new_lons2 = o2_40N.coord('longitude').points
new_lats2 = np.ones((len(new_lons2), 1)) * 39.01

fig = plt.figure(figsize=(11,6))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(0, 0.05, 0.004)
im = plt.contourf(lons, lats, age_spice_std.data, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.plot(new_lons2, new_lats2, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Age Spice', fontsize = 12)

ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
im = plt.contourf(lons, lats, o2_spice_std.data*1e6, clevs,  cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.plot(new_lons2, new_lats2, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Oxygen Spice', fontsize = 12)

cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.04])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('years/year, umol/kg/year', fontsize = 12)

plt.show()
