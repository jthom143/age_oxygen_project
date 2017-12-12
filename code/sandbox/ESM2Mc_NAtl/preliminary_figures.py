################################################################################
# Script to make model figures for chapter 3
################################################################################

## Load Packages ###############################################################
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
from math import sin, cos, sqrt, atan2, radians

## Define Functions ############################################################
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

def interpolate_to_linew(cube, name):
    # Isolate Region of Interest to limit interpolation
    lat_constraint = iris.Constraint(latitude=lambda x: 25 <= x <= 50)
    lon_constraint = iris.Constraint(longitude=lambda y: -85 <= y <= -50)
    region = cube.extract(lat_constraint)
    region = region.extract(lon_constraint)

    # Interpolate onto 1deg x 1deg grid

    new_lats = np.arange(32, 41, 1)
    new_lons = np.linspace(-65, -69, num=9)

    latitude = iris.coords.DimCoord(new_lats, standard_name='latitude', units='degrees')
    longitude = iris.coords.DimCoord(new_lons, standard_name='longitude', units='degrees')
    new_cube = iris.cube.Cube(np.zeros((27, len(new_lats), len(new_lons)), np.float32),
                              dim_coords_and_dims=[(region.coord('neutral_density'), 0),
                                                   (latitude, 1), (longitude, 2)])

    # Regrid
    regrid = region.regrid(new_cube, iris.analysis.Linear())

    # Isolate relevant lat-lon pairs
    new = np.ones((27, len(new_lats))) * np.nan

    for k in range(0,len(regrid.coord('neutral_density').points)):
        for i in range(0,len(new_lats)):
            new[k,i] = regrid[k,i,i].data

    dist = calculate_distance_km(new_lats, new_lons)
    distance = iris.coords.DimCoord(dist, long_name='distance', units='km')
    model_linew = iris.cube.Cube(new, long_name=name, units=cube.units,
                                 dim_coords_and_dims=[(region.coord('neutral_density'), 0),
                                                      (distance, 1)])
    return model_linew


def calculate_distance_km(lat, lon):
    # Function to calculate distance from Cape Cod for line W data.
    # function calls for two Pandas data series (latitude and longitude)

    # approximate radius of earth in km
    R = 6373.0

    # convert series to array:
    #lat = lat.as_matrix()
    #lon = lon.as_matrix()

    lat1 = radians(40.012)
    lon1 = radians(-68.00)

    distance = np.ones([len(lat)])*np.nan

    for i in range(0, len(lat)):
        lat2 = radians(lat[i])
        lon2 = radians(lon[i])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance[i] = R * c

    return distance

    print("Result:", distance)


## Load Data ###################################################################

# General Model Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'

# Calculate AOU
o2 = iris.load_cube(PATH+'o2.nc')
o2_sat = iris.load_cube(PATH+'o2_sat.nc')

rho = iris.load_cube(PATH+'rho.nc')

o2_sat = o2_sat[-500:]
o2 = o2[-500:]

aou = (o2_sat/rho) - o2
aou.rename('Apparent Oxygen Utilization')

age = iris.load_cube(PATH+'residency_age_surface.nc')
age.coord('Time').rename('time')
neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')

# Load Line W Data
age_linew = iris.load_cube('age_linew.nc')
aou_linew = iris.load_cube('aou_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')

## Calculate Additional Line ###################################################
constraint_lats = iris.Constraint(latitude=lambda y: 39 < y < 41)
constraint_lons = iris.Constraint(longitude=lambda x: -69 < x < -58)

aou_line40N = aou.extract(constraint_lats)
aou_line40N = aou_line40N.extract(constraint_lons)

age_line40N = age.extract(constraint_lats)
age_line40N = age_line40N.extract(constraint_lons)

rho_line40N = neutral_rho.extract(constraint_lats)
rho_line40N = rho_line40N.extract(constraint_lons)

## Calculate Climatologies #####################################################
age_linew_clim = age_linew.collapsed('time', iris.analysis.MEAN)
aou_linew_clim = aou_linew.collapsed('time', iris.analysis.MEAN)
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)

aou_line40N_clim = aou_line40N.collapsed('time', iris.analysis.MEAN)
age_line40N_clim = age_line40N.collapsed('time', iris.analysis.MEAN)
rho_line40N_clim = rho_line40N.collapsed('time', iris.analysis.MEAN)

## Calculate Horizontal Average ################################################
aou_linew_clim_vert = aou_linew_clim[:,:6].collapsed('distance', iris.analysis.MEAN)
age_linew_clim_vert = age_linew_clim[:,:6].collapsed('distance', iris.analysis.MEAN)

aou_40N_clim_vert = aou_line40N_clim.collapsed('longitude', iris.analysis.MEAN)
age_40N_clim_vert = age_line40N_clim.collapsed('longitude', iris.analysis.MEAN)

## Calculate Age-Oxygen Correlation ############################################

corr_linew = iris.analysis.stats.pearsonr(aou_linew, age_linew,
             corr_coords=['time'])

corr_line40N = iris.analysis.stats.pearsonr(aou_line40N, age_line40N,
             corr_coords=['time'])

corr_linew_vert = np.nanmean(corr_linew[:,:6].data, axis = 1)
corr_line40N_vert1 = np.nanmean(corr_line40N.data, axis = 1)

## Calculate Age and Oxygen Vertical Gradient ##################################
aou_linew_vert_gradient = iris.analysis.calculus.differentiate(
                         aou_linew_clim[:,:6].collapsed('distance', iris.analysis.MEAN),
                          'tcell pstar')
age_linew_vert_gradient = iris.analysis.calculus.differentiate(
                          age_linew_clim[:,:6].collapsed('distance', iris.analysis.MEAN),
                          'tcell pstar')


aou_line40N_vert_gradient = iris.analysis.calculus.differentiate(aou_40N_clim_vert,
                          'tcell pstar')
age_line40N_vert_gradient = iris.analysis.calculus.differentiate(age_40N_clim_vert,
                          'tcell pstar')


## Calculate Standard Deviation of Neutral Density #############################

neutral_rho_linew_std =  neutral_rho_linew.collapsed('time', iris.analysis.STD_DEV)
neutral_rho_40N_std = rho_line40N.collapsed('time', iris.analysis.STD_DEV)

neutral_rho_linew_std_vert = neutral_rho_linew_std[:,:6].collapsed('distance', iris.analysis.MEAN)
neutral_rho_40N_std_vert = neutral_rho_40N_std.collapsed('longitude', iris.analysis.MEAN)


## Calculate the correlation of age and oxygen in density space and interpolate
## back to depth space #########################################################

# Load data
aou_density = iris.load_cube('/RESEARCH/chapter3/data/newCO2_control_800/derived/aou_density.nc')
age_density = iris.load_cube('/RESEARCH/chapter3/data/newCO2_control_800/derived/age_density.nc')

corr_density = iris.analysis.stats.pearsonr(aou_density, age_density, corr_coords=['time'])

# Interpolate to line w
corr_density_linew = interpolate_to_linew(corr_density, 'corr on line w')
corr_density_linew_avg = corr_density_linew[:,:6].collapsed('distance', iris.analysis.MEAN)

neutral_rho_linew_avg = neutral_rho_linew[:,:,:6].collapsed(['time', 'distance'],
                                                            iris.analysis.MEAN)

# Interpolate to find corr_density at each depth
f = interp.interp1d(corr_density_linew_avg.coord('neutral_density').points,
                    corr_density_linew_avg.data, bounds_error=False)
corr_depth_linew_avg = f(neutral_rho_linew_avg.data-1000)


## Calculate correlation between age and oxygen on AVERAGE depth of isopycnals #

iso = np.array((26, 26.5, 27, 27.5))
corr_heave = np.ones((len(iso),
                      len(aou.coord('latitude').points),
                      len(aou.coord('longitude').points))) * np.nan

neutral_rho_clim = neutral_rho.collapsed('time', iris.analysis.MEAN) - 1000
corr = iris.analysis.stats.pearsonr(aou, age, corr_coords=['time'])
lats = aou.coord('latitude').points
lons = aou.coord('longitude').points

for n in range(0,len(iso)):
    sigma_depth = np.ones((len(aou.coord('latitude').points),
                           len(aou.coord('longitude').points))) * np.nan
    sigma_depth = find_sigma_depth(neutral_rho_clim.data, iso[n],
                     neutral_rho_clim.coord('tcell pstar').points, sigma_depth)

    corr_heave[n,:,:] = var_on_isopycnal(corr.data,
                                         neutral_rho_clim.coord('tcell pstar').points,
                                         sigma_depth, corr_heave[n,:,:])



## Create Figures ##############################################################
dist = aou_linew_clim.coord('distance').points
depth = aou_line40N_clim.coord('tcell pstar').points

lons = aou_line40N_clim.coord('longitude').points


# Line W
fig = plt.figure(figsize=(16, 6))
plt.subplot(1,4,1)
clevs = np.arange(120, 330, 10)
ax = plt.gca()
im = plt.contourf(dist, depth, aou_linew_clim.data*1e6, cmap = cmaps.viridis,
             extend = 'both')
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(dist, depth, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,3000])
ax.invert_yaxis()
#cb.set_label('umol/kg', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 12)
plt.xlabel('Distance [km]', fontsize = 12)
plt.gcf().text(0.165, 0.98, '(a) AOU Climatology', fontsize=12)

plt.subplot(1,4,2)
clevs = np.arange(0, 210, 10)
ax = plt.gca()
im2 = plt.contourf(dist, depth, age_linew_clim.data, clevs, cmap = cmaps.viridis,
             extend = 'both')
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(dist, depth, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,3000])
ax.invert_yaxis()
#cb.set_label('umol/kg', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 12)
plt.gcf().text(0.365, 0.98, '(b) Age Climatology', fontsize=12)


plt.subplot(1,4,3)
clevs = np.arange(-1, 1.1, .1)
ax = plt.gca()
im3 = plt.contourf(dist, depth, corr_linew.data, clevs, cmap = 'RdBu_r',
             extend = 'both')
CS = plt.contour(dist, depth, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,3000])
plt.xlim([0,900])
ax.invert_yaxis()
plt.gcf().text(0.575, 0.98, '(c) Correlation', fontsize=12)

ax1 = plt.subplot(1,4,4)
plt.plot(age_linew_clim_vert.data, depth, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age (years)', color='b')
#ax1.axhline(400, color = 'k', ls = '--')
#ax1.axhline(700, color = 'k', ls = '--')
plt.axhspan(400,800 , facecolor='0.1', alpha=0.1)
ax2 = ax1.twiny()
plt.plot(aou_linew_clim_vert.data*1e6, depth, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg)', color='r')
plt.ylim([0,3000])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.gcf().text(0.775, 0.98, '(d) Vertical Profile', fontsize=12)

fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.135, 0.1, 0.15, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('umol/kg', fontsize = 12)

cbar_ax = fig.add_axes([0.335, 0.1, 0.15, 0.03])
cb = fig.colorbar(im2, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('years', fontsize = 12)

cbar_ax = fig.add_axes([0.54, 0.1, 0.15, 0.03])
cb = fig.colorbar(im3, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('correlation', fontsize = 12)


# Line 40N
fig = plt.figure(figsize=(16, 6))
plt.subplot(1,4,1)
clevs = np.arange(120, 305, 5)
ax = plt.gca()
im = plt.contourf(lons, depth, aou_line40N_clim.data*1e6, cmap = cmaps.viridis,
             extend = 'both')
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(lons, depth, rho_line40N_clim.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,3000])
ax.invert_yaxis()
#cb.set_label('umol/kg', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 12)
plt.xlabel('Longitude [degrees]', fontsize = 12)
plt.gcf().text(0.165, 0.98, '(a) AOU Climatology', fontsize=12)

plt.subplot(1,4,2)
clevs = np.arange(0, 105, 5)
ax = plt.gca()
im2 = plt.contourf(lons, depth, age_line40N_clim.data, clevs, cmap = cmaps.viridis,
             extend = 'both')
#cb = plt.colorbar(orientation='horizontal')
CS = plt.contour(lons, depth, rho_line40N_clim.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,3000])
ax.invert_yaxis()
#cb.set_label('umol/kg', fontsize = 14)
plt.xlabel('Longitude [degrees]', fontsize = 12)
plt.gcf().text(0.365, 0.98, '(b) Age Climatology', fontsize=12)


plt.subplot(1,4,3)
clevs = np.arange(-1, 1.1, .1)
ax = plt.gca()
im3 = plt.contourf(lons, depth, corr_line40N.data, clevs, cmap = 'RdBu_r',
             extend = 'both')
CS = plt.contour(lons, depth, rho_line40N_clim.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,3000])
ax.invert_yaxis()
plt.gcf().text(0.575, 0.98, '(c) Correlation', fontsize=12)


ax1 = plt.subplot(1,4,4)
plt.plot(age_40N_clim_vert.data, depth, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age (years)', color='b')

ax2 = ax1.twiny()
plt.plot(aou_40N_clim_vert.data*1e6, depth, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg)', color='r')
plt.ylim([0,3000])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.gcf().text(0.775, 0.98, '(d) Vertical Profile', fontsize=12)

fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.135, 0.1, 0.15, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('umol/kg', fontsize = 12)

cbar_ax = fig.add_axes([0.335, 0.1, 0.15, 0.03])
cb = fig.colorbar(im2, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('years', fontsize = 12)

cbar_ax = fig.add_axes([0.54, 0.1, 0.15, 0.03])
cb = fig.colorbar(im3, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('correlation', fontsize = 12)


plt.figure()
ax1 = plt.subplot(1,2,1)
plt.plot(age_linew_vert_gradient.data, aou_linew_vert_gradient.coord('tcell pstar').points, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age Gradient (years/dbar)', color='b')
ax1.set_xlim([-0.8, 0.8])

ax2 = ax1.twiny()
plt.plot(aou_linew_vert_gradient.data*1e6, aou_linew_vert_gradient.coord('tcell pstar').points, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen Gradient (umol/kg/dbar)', color='r')
plt.ylim([0,1500])
ax2.set_xlim([-0.35, 0.35])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.axvline(0, ls = '--', color = 'k', lw = 1.5)
plt.gcf().text(0.25, 0.98, '(a) Line W', fontsize=12)

ax1 = plt.subplot(1,2,2)
plt.plot(age_line40N_vert_gradient.data, aou_line40N_vert_gradient.coord('tcell pstar').points, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age Gradient (years/dbar)', color='b')
ax1.set_xlim([-0.8, 0.8])

ax2 = ax1.twiny()
plt.plot(aou_line40N_vert_gradient.data*1e6, aou_line40N_vert_gradient.coord('tcell pstar').points, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen Gradient (umol/kg/dbar)', color='r')
plt.ylim([0,1500])
ax2.set_xlim([-0.35, 0.35])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.axvline(0, ls = '--', color = 'k', lw = 1.5)
plt.gcf().text(0.65, 0.98, '(b) Line 40N', fontsize=12)



fig = plt.figure()
clevs = np.arange(0, 0.17, 0.01)


ax1 = plt.subplot(1,2,1)
plt.contourf(dist, depth, neutral_rho_linew_std.data, clevs, cmap = cmaps.viridis)
ax1.set_xlabel('Distance [km]')
plt.ylim([0,1500])
ax1.invert_yaxis()
ax1.set_ylabel('Depth [dbars]')
plt.gcf().text(0.25, 0.98, '(a) Line W', fontsize=12)

ax1 = plt.subplot(1,2,2)
im = plt.contourf(lons, depth, neutral_rho_40N_std.data, clevs, cmap = cmaps.viridis)
plt.xlabel('Longitude [degrees]', fontsize = 12)
plt.ylim([0,1500])
ax1.invert_yaxis()
plt.gcf().text(0.65, 0.98, '(b) Line 40N', fontsize=12)

fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('kg/m$^3$', fontsize = 12)





plt.figure()
ax1 = plt.subplot(1,4,1)
plt.plot(age_linew_vert_gradient.data, aou_linew_vert_gradient.coord('tcell pstar').points, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age Gradient (years/dbar)', color='b')
ax1.set_xlim([-0.4, 0.4])

ax2 = ax1.twiny()
plt.plot(aou_linew_vert_gradient.data*1e6, aou_linew_vert_gradient.coord('tcell pstar').points, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen Gradient (umol/kg/dbar)', color='r')
plt.ylim([0,1500])
ax2.set_xlim([-0.2, 0.2])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.axvline(0, ls = '--', color = 'k', lw = 1.5)
plt.gcf().text(0.25, 0.98, '(a) Line W', fontsize=12)

ax1 = plt.subplot(1,4,2)
plt.plot(age_line40N_vert_gradient.data, aou_line40N_vert_gradient.coord('tcell pstar').points, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age Gradient (years/dbar)', color='b')
ax1.set_xlim([-0.4, 0.4])

ax2 = ax1.twiny()
plt.plot(aou_line40N_vert_gradient.data*1e6, aou_line40N_vert_gradient.coord('tcell pstar').points, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen Gradient (umol/kg/dbar)', color='r')
plt.ylim([0,1500])
ax2.set_xlim([-0.2, 0.2])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
plt.axvline(0, ls = '--', color = 'k', lw = 1.5)
plt.gcf().text(0.65, 0.98, '(b) Line 40N', fontsize=12)


ax1 = plt.subplot(1,4,3)
plt.plot(neutral_rho_linew_std_vert.data, depth, color = 'k', lw = 1.5)
ax1.set_xlabel('Standard Deviation [kg m$^{-3}$]')
plt.ylim([0,1500])
ax1.invert_yaxis()
ax1.set_ylabel('Depth [dbars]')

plt.plot(neutral_rho_40N_std_vert.data, depth, color = 'g', lw = 1.5)
plt.ylim([0,1500])
ax1.invert_yaxis()
plt.title('(a) Neutral Density Standard Deviation')


ax1 = plt.subplot(1,4,4)
plt.plot(corr_linew_vert, depth, color = 'k', lw = 1.5)
plt.plot(corr_depth_linew_avg, depth, color = 'k', lw = 1.5, ls = '--')
ax1.set_xlabel('Correlation')
plt.ylim([0,1500])
ax1.invert_yaxis()
plt.title('(b) Correlation')

plt.plot(corr_line40N_vert1, depth, color = 'g', lw = 1.5)
plt.ylim([0,1500])
ax1.invert_yaxis()
plt.axvline(0, ls = '--', color = 'grey', lw = 1.5)












plt.figure(figsize = (10,10))
lats = corr_density.coord('latitude').points
lons = corr_density.coord('longitude').points

ax = plt.subplot(4,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_density[16,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b)', fontsize = 12)


ax = plt.subplot(4,2,4, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_density[18,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(d)', fontsize = 12)

ax = plt.subplot(4,2,6, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_density[20,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(f)', fontsize = 12)

ax = plt.subplot(4,2,8, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_density[22,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(h)', fontsize = 12)



lats = aou.coord('latitude').points
lons = aou.coord('longitude').points

ax = plt.subplot(4,2,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_heave[0,:,:], clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a)', fontsize = 12)


ax = plt.subplot(4,2,3, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_heave[1,:,:], clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(c)', fontsize = 12)

ax = plt.subplot(4,2,5, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_heave[2,:,:], clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(e)', fontsize = 12)

ax = plt.subplot(4,2,7, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, corr_heave[3,:,:], clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(g)', fontsize = 12)

plt.gcf().text(0.26, 0.95, 'With Heave', fontsize=14)
plt.gcf().text(0.665, 0.95, 'Without Heave', fontsize=14)

plt.gcf().text(0.02, 0.785, '$\gamma_n$ = 26.0', fontsize=14)
plt.gcf().text(0.02, 0.585, '$\gamma_n$ = 26.5', fontsize=14)
plt.gcf().text(0.02, 0.385, '$\gamma_n$ = 27.0', fontsize=14)
plt.gcf().text(0.02, 0.185, '$\gamma_n$ = 27.5', fontsize=14)


plt.show()
