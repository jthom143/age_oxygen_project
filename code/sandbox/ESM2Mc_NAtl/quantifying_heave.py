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
#                     Calculate Heave                                          #
################################################################################
z = np.ones((42,29)) * np.nan
depth27_t = np.ones((500,42,29)) * np.nan


depth = o2.coord('tcell pstar').points
lons = o2.coord('longitude').points
lats = o2.coord('latitude').points

for t in range(0,500):
    z = find_sigma_depth(neutral_rho[t].data-1000, 27, depth, z)
    depth27_t[t,:,:] = z
time = o2.coord('time')
latitude = o2.coord('latitude')
longitude = o2.coord('longitude')
depth27 = iris.cube.Cube(depth27_t, long_name = 'Depth of neutral density 27',
                       dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])

depth27_dt = iris.analysis.calculus.differentiate(depth27, 'time')

# Calculate Vertical Gradient Age
age_dz = iris.analysis.calculus.differentiate(age, 'tcell pstar')
age_dz_n = np.ones((500,42,29))*np.nan

o2_dz = iris.analysis.calculus.differentiate(o2, 'tcell pstar')
o2_dz_n = np.ones((500,42,29))*np.nan

depth = age_dz.coord('tcell pstar')
for t in range(0,500):
    for x in range(0,42):
        for y in range(0,29):
            ref_z = depth27[t,x,y].data
            if np.isnan(ref_z) == True:
                idepth = np.nan
            else:
                idepth = iris.analysis.interpolate.nearest_neighbour_indices(age_dz[t,:,x,y],
                [('tcell pstar', ref_z)])
                age_dz_n[t,x,y] = age_dz[t,idepth,x,y].data
                o2_dz_n[t,x,y] = o2_dz[t,idepth,x,y].data

age_heave = age_dz_n[:-1,:,:] * depth27_dt.data
age_heave_std = np.nanstd(age_heave, axis = 0)

o2_heave = o2_dz_n[:-1,:,:] * depth27_dt.data
o2_heave_std = np.nanstd(o2_heave, axis = 0)


depth27_dt_std = depth27_dt.collapsed('time', iris.analysis.STD_DEV)

age_dz_n_std_dev = np.nanstd(age_dz_n, axis = 0)

test = age_dz_n * o2_dz_n



# Line W:
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)


fig = plt.figure(figsize=(11,6))
ax = plt.subplot(1,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(0, 0.05, 0.004)
im = plt.contourf(lons, lats, age_heave_std, clevs, cmap = cmaps.viridis, extend = 'max')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Age Heave', fontsize = 12)

ax = plt.subplot(1,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
im = plt.contourf(lons, lats, o2_heave_std*1e6, clevs,  cmap = cmaps.viridis, extend = 'max')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Oxygen Heave', fontsize = 12)

cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.04])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('years/year, umol/kg/year', fontsize = 12)




fig = plt.figure(figsize=(11,6))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(-0.05, 0.051, 0.001)
im = plt.contourf(lons, lats, test[0]*1e6, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar()

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Age * O2', fontsize = 12)


plt.show()
