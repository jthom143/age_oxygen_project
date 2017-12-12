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
o2 = iris.load_cube(PATH+'o2.nc')
o2_sat = iris.load_cube(PATH+'o2_sat.nc')
o2 = o2[-500:]
o2_sat = o2_sat[-500:]

age = iris.load_cube(PATH+'residency_age_surface.nc')
age.coord('Time').rename('time')

neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')
rho = iris.load_cube(PATH+'rho.nc')
aou = (o2_sat/rho) - o2
aou.rename('Apparent Oxygen Utilization')

## Restrict Domain to North Atlantic Ocean
constraint = iris.Constraint(latitude=lambda y: 0 < y < 90)
neutral_rho = neutral_rho.extract(constraint)
age = age.extract(constraint)
o2 = o2.extract(constraint)
aou = aou.extract(constraint)

constraint = iris.Constraint(longitude=lambda x: -80 < x < 5)
neutral_rho = neutral_rho.extract(constraint)
age = age.extract(constraint)
o2 = o2.extract(constraint)
aou = aou.extract(constraint)

################################################################################
#                     Interpolate to density surfaces                          #
################################################################################

ref_dens = {'26.5':  26.5,
            '26.75': 26.75,
            '27':    27.0,
            '27.25': 27.25}
variables = [aou]

o2_density = {}
age_density = {}
aou_density = {}

for var in variables:
    print var.long_name
    for key, i in ref_dens.iteritems():
        print i
        # Initalize depth and variable
        depth_dens    = np.ones((len(var.coord('latitude').points),
                                 len(var.coord('longitude').points))) * np.nan
        var_dens      = np.ones((len(var.coord('latitude').points),
                                 len(var.coord('longitude').points))) * np.nan
        var_dens_time = np.ones((len(var.coord('time').points),
                                 len(var.coord('latitude').points),
                                 len(var.coord('longitude').points))) * np.nan

        # Define old coordinate arrays
        lons = var.coord('longitude').points
        lats = var.coord('latitude').points
        depth = var.coord('tcell pstar').points


        for t in range(0,500):
            depth_dens = find_sigma_depth(neutral_rho[t].data-1000, i, depth, depth_dens,
                                          lons, lats)

            var_dens = var_on_isopycnal(var[t].data, depth, depth_dens, var_dens)
            var_dens_time[t,:,:] = var_dens

        # Convert to Cube:
        time = o2.coord('time')
        latitude = o2.coord('latitude')
        longitude = o2.coord('longitude')
        string = var.long_name + ' on density surface %10.2f' % i

        if var == o2:
            o2_density[key] = iris.cube.Cube(var_dens_time, long_name = string,
                dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])

        elif var == age:
            age_density[key] = iris.cube.Cube(var_dens_time, long_name = string,
                dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])

        elif var == aou:
            aou_density[key] = iris.cube.Cube(var_dens_time, long_name = string,
                dim_coords_and_dims = [(time,0), (latitude, 1), (longitude, 2)])
        else:
            print 'variable error'

################################################################################
#                        Figure                                                #
################################################################################

### OXYGEN
fig = plt.figure(figsize=(12,10))
clevs = np.arange(0, 375, 25)
## A
ax = plt.subplot(2,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_density['26.5'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) O2 on 26.5', fontsize = 14)

## B
ax = plt.subplot(2,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_density['26.75'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) O2 on 26.75', fontsize = 14)

## C
ax = plt.subplot(2,2,3, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_density['27'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(c) O2 on 27', fontsize = 14)

## D
ax = plt.subplot(2,2,4, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
im = plt.contourf(lons, lats, o2_density['27.25'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(d) O2 on 27.25', fontsize = 14)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('umol/kg', fontsize = 14)
plt.savefig('o2_isopycnals.png')

### Apparent Oxygen Utilization
fig = plt.figure(figsize=(12,10))
clevs = np.arange(0, 275, 25)
## A
ax = plt.subplot(2,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, aou_density['26.5'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) AOU on 26.5', fontsize = 14)

## B
ax = plt.subplot(2,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, aou_density['26.75'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) AOU on 26.75', fontsize = 14)

## C
ax = plt.subplot(2,2,3, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, aou_density['27'].collapsed('time', iris.analysis.MEAN).data*1e6,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(c) AOU on 27', fontsize = 14)

## D
ax = plt.subplot(2,2,4, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
im = plt.contourf(lons, lats, aou_density['27.25'].collapsed('time', iris.analysis.MEAN).data*1e6,
              clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(d) AOU on 27.25', fontsize = 14)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('umol/kg', fontsize = 14)
plt.savefig('aou_isopycnals.png')



### AGE
fig = plt.figure(figsize=(12,10))
clevs = np.arange(0, 220, 20)
## A
ax = plt.subplot(2,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, age_density['26.5'].collapsed('time', iris.analysis.MEAN).data,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Age on 26.5', fontsize = 14)

## B
ax = plt.subplot(2,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, age_density['26.75'].collapsed('time', iris.analysis.MEAN).data,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b) Age on 26.75', fontsize = 14)

## C
ax = plt.subplot(2,2,3, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
plt.contourf(lons, lats, age_density['27'].collapsed('time', iris.analysis.MEAN).data,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(c) Age on 27', fontsize = 14)

## D
ax = plt.subplot(2,2,4, projection=ccrs.PlateCarree())
ax.set_extent((-85, 0, 15, 80), crs=ccrs.PlateCarree())
im = plt.contourf(lons, lats, age_density['27.25'].collapsed('time', iris.analysis.MEAN).data,
             clevs, cmap = cmaps.viridis)
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(c) Age on 27.25', fontsize = 14)


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('years', fontsize = 14)
plt.savefig('age_isopycnals.png')

plt.show()
