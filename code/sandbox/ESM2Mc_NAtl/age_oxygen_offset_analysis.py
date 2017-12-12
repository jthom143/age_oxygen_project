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


# Load Line W Data
age_linew = iris.load_cube('age_linew.nc')
o2_linew = iris.load_cube('o2_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')

# Climatology
age_linew_clim = age_linew.collapsed('time', iris.analysis.MEAN)
o2_linew_clim = o2_linew.collapsed('time', iris.analysis.MEAN)
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)

## Isolate age and oxygen at distance 600km
constraint = iris.Constraint(distance=lambda x: 600 < x < 700)
age_600 = age_linew.extract(constraint)
o2_600 = o2_linew.extract(constraint)

age_600_clim = age_600.collapsed('time', iris.analysis.MEAN)
o2_600_clim = o2_600.collapsed('time', iris.analysis.MEAN)


## Find minimum and maximum in age and o2
dist = o2_linew_clim[:22,:].coord('distance').points
depth = o2_linew_clim[:22,:].coord('tcell pstar').points

o2_min_depth = np.ones((len(dist)))*np.nan
age_max_depth = np.ones((len(dist)))*np.nan

for x in range(0,len(dist)):
    o2_min_ind = np.argmin(o2_linew_clim[:22,:].data, axis=0)
    age_max_ind = np.argmax(age_linew_clim[:22,:].data, axis=0)

    o2_min_depth[x] = depth[o2_min_ind[x]]
    age_max_depth[x] = depth[age_max_ind[x]]


# Plot figure
fig = plt.figure(figsize=(8.5,4.5))
plt.subplot(1,3,1)
o2_clim = o2_linew.collapsed('time', iris.analysis.MEAN)
clevs = np.arange(120, 330, 10)
ax = plt.gca()
plt.contourf(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, o2_linew_clim.data*1e6, clevs,
             cmap = cmaps.viridis, extend = 'both')
plt.plot(dist, o2_min_depth, ls = '--', color = 'k', lw = 3)
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,4000])
ax.invert_yaxis()
#cb.set_label('umol/kg', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 12)
plt.xlabel('Distance [km]', fontsize = 12)
plt.gcf().text(0.17, 0.97, '(a) Oxygen Climatology', fontsize=12)
plt.axvline(600, ls = '--', color = 'k')

plt.subplot(1,3,2)
age_clim = age_linew.collapsed('time', iris.analysis.MEAN)
ax = plt.gca()
clevs = np.arange(0, 210, 10)
im = plt.contourf(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, age_linew_clim.data, clevs, cmap = cmaps.viridis, extend = 'both')
plt.plot(dist, age_max_depth, ls = '--', color = 'k', lw = 3)
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.ylim([0,4000])
plt.clabel(CS, fontsize=9, inline=1)
ax.invert_yaxis()
#cb.set_label('years', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 12)
plt.gcf().text(0.45, 0.97, '(b) Age Climatology', fontsize=12)
plt.axvline(600, ls = '--', color = 'k')

ax1 = plt.subplot(1,3,3)
plt.plot(age_600_clim.data, age_600_clim.coord('tcell pstar').points, lw = 2)
#plt.plot(age_200[0,:].data, age_600.coord('tcell pstar').points, lw = 2, ls = '--', color = 'b')
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age (years)', color='b')
ax1.axhline(400, color = 'k', ls = '--')
ax1.axhline(700, color = 'k', ls = '--')
ax2 = ax1.twiny()
plt.plot(o2_600_clim.data*1e6, o2_600_clim.coord('tcell pstar').points, color = 'r', lw = 2)
#plt.plot(o2_200[0,:].data*1e6, age_600.coord('tcell pstar').points, color = 'r', lw = 2, ls = '--')
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg)', color='r')
plt.ylim([0,4000])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.gcf().text(0.72, 0.97, '(c) Vertical Profile', fontsize=12)


fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('umol/kg', fontsize = 12)


################################################################################
# Look at profiles directly east of cape cod... do we still see offset?
################################################################################

constraint_lats = iris.Constraint(latitude=lambda y: 39 < y < 41)
constraint_lons = iris.Constraint(longitude=lambda x: -69 < x < -58)
o2_40N = o2.extract(constraint_lats)
o2_40N = o2_40N.extract(constraint_lons)

age_40N = age.extract(constraint_lats)
age_40N = age_40N.extract(constraint_lons)

rho_40N = neutral_rho.extract(constraint_lats)
rho_40N = rho_40N.extract(constraint_lons)

o2_40N_clim = o2_40N.collapsed('time', iris.analysis.MEAN)
age_40N_clim = age_40N.collapsed('time', iris.analysis.MEAN)
rho_40N_clim = rho_40N.collapsed('time', iris.analysis.MEAN)

o2_40N_clim_vert = o2_40N_clim.collapsed('longitude', iris.analysis.MEAN)
age_40N_clim_vert = age_40N_clim.collapsed('longitude', iris.analysis.MEAN)


# Plot figure
fig = plt.figure(figsize=(12.5,6.5))
plt.subplot(1,3,1)
o2_clim = o2_linew.collapsed('time', iris.analysis.MEAN)
clevs = np.arange(120, 305, 5)
ax = plt.gca()
im = plt.contourf(o2_40N_clim.coord('longitude').points, o2_linew.coord('tcell pstar').points, o2_40N_clim.data*1e6, clevs,
             cmap = cmaps.viridis, extend = 'both')
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(o2_40N_clim.coord('longitude').points, o2_linew.coord('tcell pstar').points, rho_40N_clim.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
ax.invert_yaxis()
#cb.set_label('umol/kg', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 14)
plt.gcf().text(0.17, 0.97, '(a) Oxygen Climatology', fontsize=14)

plt.subplot(1,3,2)
age_clim = age_linew.collapsed('time', iris.analysis.MEAN)
ax = plt.gca()
clevs = np.arange(0, 105, 5)
plt.contourf(o2_40N_clim.coord('longitude').points, o2_linew.coord('tcell pstar').points, age_40N_clim.data, clevs, cmap = cmaps.viridis, extend = 'both')
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(o2_40N_clim.coord('longitude').points, o2_linew.coord('tcell pstar').points, rho_40N_clim.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.ylim([0,2000])
plt.clabel(CS, fontsize=9, inline=1)
ax.invert_yaxis()
#cb.set_label('years', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 14)
plt.gcf().text(0.45, 0.97, '(b) Age Climatology', fontsize=14)

ax1 = plt.subplot(1,3,3)
plt.plot(age_40N_clim_vert.data, o2_40N_clim_vert.coord('tcell pstar').points, lw = 2)
#plt.plot(age_200[0,:].data, age_600.coord('tcell pstar').points, lw = 2, ls = '--', color = 'b')
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age (years)', color='b')
ax2 = ax1.twiny()
plt.plot(o2_40N_clim_vert.data*1e6, o2_40N_clim_vert.coord('tcell pstar').points, color = 'r', lw = 2)
#plt.plot(o2_200[0,:].data*1e6, age_600.coord('tcell pstar').points, color = 'r', lw = 2, ls = '--')
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg)', color='r')
plt.ylim([0,4000])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.gcf().text(0.72, 0.97, '(c) Vertical Profile', fontsize=14)





















################################################################################
# Find maximum and Minimum
################################################################################

## Restrict analysis to the top 2000 dbars
o2 = o2[:,:22,:,:]
age = age[:,:22,:,:]
neutral_rho = neutral_rho[:,:22,:,:]

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

for x in range(0,len(lons)):
    for y in range(0,len(lats)):
        o2_min_ind = np.argmin(o2_clim.data, axis=0)
        age_max_ind = np.argmax(age_clim.data, axis=0)

        o2_min_depth[y,x] = depth[o2_min_ind[y,x]]
        age_max_depth[y,x] = depth[age_max_ind[y,x]]


### Plot the depth and value of the age maximum and o2 minimum

lats = o2_min.coord('latitude').points
lons = o2_min.coord('longitude').points

fig = plt.figure(figsize = (12,12))
ax = plt.subplot(2,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -40, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(0, 350, 25)
im = plt.contourf(lons, lats, o2_min.data*1e6, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')


ax = plt.subplot(2,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, -40, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(0, 325, 25)
plt.contourf(lons, lats, age_max.data, clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')



ax = plt.subplot(2,2,3, projection=ccrs.PlateCarree())
ax.set_extent((-85, -40, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(0, 1500, 100)
plt.contourf(lons, lats, o2_min_depth,clevs, cmap = cmaps.viridis, extend = 'max')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(c) Depth of Oxygen Minimum', fontsize = 12)

ax = plt.subplot(2,2,4, projection=ccrs.PlateCarree())
ax.set_extent((-85, -40, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(0, 1500, 100)
plt.contourf(lons, lats, age_max_depth, clevs, cmap = cmaps.viridis, extend = 'max')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(d) Depth of Age Maximum', fontsize = 12)

plt.figure(figsize=(8,8))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -40, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(-1000, 1100, 100)
plt.contourf(lons, lats, age_max_depth-o2_min_depth, clevs, cmap = 'RdBu_r', extend = 'both')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')
plt.colorbar(orientation = 'horizontal')
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('Depth Difference', fontsize = 12)




plt.show()
