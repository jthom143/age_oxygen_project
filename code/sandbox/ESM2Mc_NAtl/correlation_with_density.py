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
## Convert age and oxygen data to neutral density vertical coordinates.
################################################################################

rho_scale = np.arange(22, 28.75, 0.25)
time = o2.coord('time').points
lons = o2.coord('longitude').points
lats = o2.coord('latitude').points
depth = o2.coord('tcell pstar').points

age_density = np.ones((len(time), len(rho_scale), len(lats), len(lons))) * np.nan
o2_density = np.ones((len(time), len(rho_scale), len(lats), len(lons))) * np.nan
aou_density = np.ones((len(time), len(rho_scale), len(lats), len(lons))) * np.nan

for n in range(0,len(rho_scale)):
    rho = rho_scale[n]

    age_on_sigma_time = np.ones((len(time), len(lats), len(lons))) * np.nan
    o2_on_sigma_time = np.ones((len(time), len(lats), len(lons))) * np.nan
    aou_on_sigma_time = np.ones((len(time), len(lats), len(lons))) * np.nan

    for t in range(0,len(time)):
        sigma_depth = np.ones((len(lats), len(lons))) * np.nan
        age_on_sigma = np.ones((len(lats), len(lons))) * np.nan
        o2_on_sigma = np.ones((len(lats), len(lons))) * np.nan
        aou_on_sigma = np.ones((len(lats), len(lons))) * np.nan

        # Interpolate to find the depth of the isopycnal surface
        sigma_depth = find_sigma_depth(neutral_rho[t,:,:,:].data - 1000, rho, depth, sigma_depth)

        # Find the age and ozygen on that depth
        age_on_sigma = var_on_isopycnal(age[t,:,:,:].data, depth, sigma_depth, age_on_sigma)
        o2_on_sigma = var_on_isopycnal(o2[t,:,:,:].data, depth, sigma_depth, o2_on_sigma)
        aou_on_sigma = var_on_isopycnal(aou[t,:,:,:].data, depth, sigma_depth, aou_on_sigma)

        # Put the 2D array in to a 3D array with time
        age_on_sigma_time[t,:,:] = age_on_sigma
        o2_on_sigma_time[t,:,:] = o2_on_sigma
        aou_on_sigma_time[t,:,:] = aou_on_sigma

    age_density[:,n,:,:] =  age_on_sigma_time
    o2_density[:,n,:,:] =  o2_on_sigma_time
    aou_density[:,n,:,:] =  aou_on_sigma_time

# Create cubes
neutral_rho_dim = iris.coords.DimCoord(rho_scale,
                                       long_name='neutral_density',
                                       units=neutral_rho.units)

age_density = iris.cube.Cube(age_density, long_name = 'age on density surfaces',
                             units = age.units,
                             dim_coords_and_dims=[(o2.coord('time')     , 0),
                                                  (neutral_rho_dim      , 1),
                                                  (o2.coord('latitude') , 2),
                                                  (o2.coord('longitude'), 3)])

o2_density = iris.cube.Cube(o2_density, long_name = 'oxygen on density surfaces',
                             units = o2.units,
                             dim_coords_and_dims=[(o2.coord('time')     , 0),
                                                  (neutral_rho_dim      , 1),
                                                  (o2.coord('latitude') , 2),
                                                  (o2.coord('longitude'), 3)])
aou_density = iris.cube.Cube(aou_density, long_name = 'oxygen on density surfaces',
                             units = aou.units,
                             dim_coords_and_dims=[(o2.coord('time')     , 0),
                                                  (neutral_rho_dim      , 1),
                                                  (o2.coord('latitude') , 2),
                                                  (o2.coord('longitude'), 3)])

iris.save(age_density, '/RESEARCH/chapter3/data/newCO2_control_800/derived/age_density.nc')
iris.save(o2_density, '/RESEARCH/chapter3/data/newCO2_control_800/derived/o2_density.nc')
iris.save(aou_density, '/RESEARCH/chapter3/data/newCO2_control_800/derived/aou_density.nc')

################################################################################
## Load N. Atl age and oxygen interpolated onto density surfaces
################################################################################





################################################################################
## Calculate Correlation between age and oxygen
################################################################################

corr = iris.analysis.stats.pearsonr(o2_density, age_density, corr_coords=['time'])

# Interpolate to line w
corr_linew = interpolate_to_linew(corr, 'corr on line w')
#corr_linew_avg = np.nanmean(corr_linew[:,:6].data, axis = 1)
corr_linew_avg = corr_linew[:,:6].collapsed('distance', iris.analysis.MEAN)

ax = plt.subplot(1,1,1)
plt.plot(corr_linew_avg.data, rho_scale)
ax.invert_yaxis()



new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)
fig = plt.figure(figsize=(11,6))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -20, 15, 60), crs=ccrs.PlateCarree())
clevs = np.arange(-1, 1.1, .1)
im = plt.contourf(lons, lats, corr[10,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines(resolution='50m', color='black', lw = 1.5)
ax.add_feature(cfeature.LAND, facecolor='0.75')

plt.plot(new_lons, new_lats, color = 'k', marker = '*', transform=ccrs.PlateCarree())
plt.title('(a) Corr on 27', fontsize = 12)
plt.colorbar()

plt.show()
