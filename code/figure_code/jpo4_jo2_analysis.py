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
from math import sin, cos, sqrt, atan2, radians

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
    new_cube = iris.cube.Cube(np.zeros((14, len(new_lats), len(new_lons)), np.float32),
                              dim_coords_and_dims=[(region.coord('neutral_density'), 0),
                                                   (latitude, 1), (longitude, 2)])

    # Regrid
    regrid = region.regrid(new_cube, iris.analysis.Linear())

    # Isolate relevant lat-lon pairs
    new = np.ones((14, len(new_lats))) * np.nan

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


## Load Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'
jpo4 = iris.load_cube(PATH+'jpo4.nc')
jo2 = iris.load_cube(PATH+'jo2.nc')

jpo4 = jpo4[-500:]
jo2 = jo2[-500:]


neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')


## Restrict Domain to North Atlantic Ocean
constraint = iris.Constraint(latitude=lambda y: 0 < y < 90)
neutral_rho = neutral_rho.extract(constraint)
jpo4 = jpo4.extract(constraint)
jo2 = jo2.extract(constraint)

constraint = iris.Constraint(longitude=lambda x: -80 < x < 5)
neutral_rho = neutral_rho.extract(constraint)
jpo4 = jpo4.extract(constraint)
jo2 = jo2.extract(constraint)



################################################################################
## Convert age and oxygen data to neutral density vertical coordinates.
################################################################################

rho_scale = np.arange(22, 28.75, 0.25)
time = jo2.coord('time').points
lons = jo2.coord('longitude').points
lats = jo2.coord('latitude').points
depth = jo2.coord('tcell pstar').points

jpo4_density = np.ones((len(time), len(rho_scale), len(lats), len(lons))) * np.nan
jo2_density = np.ones((len(time), len(rho_scale), len(lats), len(lons))) * np.nan

for n in range(0,len(rho_scale)):
    rho = rho_scale[n]

    jpo4_on_sigma_time = np.ones((len(time), len(lats), len(lons))) * np.nan
    jo2_on_sigma_time = np.ones((len(time), len(lats), len(lons))) * np.nan

    for t in range(0,len(time)):
        sigma_depth = np.ones((len(lats), len(lons))) * np.nan
        jpo4_on_sigma = np.ones((len(lats), len(lons))) * np.nan
        jo2_on_sigma = np.ones((len(lats), len(lons))) * np.nan

        # Interpolate to find the depth of the isopycnal surface
        sigma_depth = find_sigma_depth(neutral_rho[t,:,:,:].data - 1000, rho, depth, sigma_depth)

        # Find the age and ozygen on that depth
        jpo4_on_sigma = var_on_isopycnal(jpo4[t,:,:,:].data, depth, sigma_depth, jpo4_on_sigma)
        jo2_on_sigma = var_on_isopycnal(jo2[t,:,:,:].data, depth, sigma_depth, jo2_on_sigma)

        # Put the 2D array in to a 3D array with time
        jpo4_on_sigma_time[t,:,:] = jpo4_on_sigma
        jo2_on_sigma_time[t,:,:] = jo2_on_sigma


    jpo4_density[:,n,:,:] =  jpo4_on_sigma_time
    jo2_density[:,n,:,:] =  jo2_on_sigma_time

# Create cubes
neutral_rho_dim = iris.coords.DimCoord(rho_scale,
                                       long_name='neutral_density',
                                       units=neutral_rho.units)

jpo4_density = iris.cube.Cube(jpo4_density, long_name = 'jpo4 on density surfaces',
                             units = jpo4.units,
                             dim_coords_and_dims=[(jpo4.coord('time')     , 0),
                                                  (neutral_rho_dim      , 1),
                                                  (jpo4.coord('latitude') , 2),
                                                  (jpo4.coord('longitude'), 3)])

jo2_density = iris.cube.Cube(jo2_density, long_name = 'JO2 on density surfaces',
                             units = jo2.units,
                             dim_coords_and_dims=[(jo2.coord('time')     , 0),
                                                  (neutral_rho_dim      , 1),
                                                  (jo2.coord('latitude') , 2),
                                                  (jo2.coord('longitude'), 3)])


iris.save(jpo4_density, '/RESEARCH/chapter3/data/newCO2_control_800/derived/jpo4_density.nc')
iris.save(jo2_density, '/RESEARCH/chapter3/data/newCO2_control_800/derived/jo2_density.nc')






























### Nov 30, 2017 Note: Need to time average JPO4 to make figure


## Create Figure

plt.figure(figsize = (10,10))
lats = jpo4_density.coord('latitude').points
lons = jpo4_density.coord('longitude').points

ax = plt.subplot(4,2,2, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, jpo4_density[16,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(b)', fontsize = 12)


ax = plt.subplot(4,2,4, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, jpo4_density[18,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(d)', fontsize = 12)

ax = plt.subplot(4,2,6, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, jpo4_density[20,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(f)', fontsize = 12)

ax = plt.subplot(4,2,8, projection=ccrs.PlateCarree())
ax.set_extent((-85, -10, 15, 55), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(lons, lats, jpo4_density[22,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

#plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(h)', fontsize = 12)


plt.gcf().text(0.26, 0.95, 'With Heave', fontsize=14)
plt.gcf().text(0.665, 0.95, 'Without Heave', fontsize=14)

plt.gcf().text(0.02, 0.785, '$\gamma_n$ = 26.0', fontsize=14)
plt.gcf().text(0.02, 0.585, '$\gamma_n$ = 26.5', fontsize=14)
plt.gcf().text(0.02, 0.385, '$\gamma_n$ = 27.0', fontsize=14)
plt.gcf().text(0.02, 0.185, '$\gamma_n$ = 27.5', fontsize=14)


plt.show()
