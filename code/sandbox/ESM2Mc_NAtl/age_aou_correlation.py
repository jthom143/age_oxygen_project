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

## Define Functions
# Function to calculate distance from Cape Cod:
def calculate_distance_km(lat, lon):
    from math import sin, cos, sqrt, atan2, radians
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

# Function to model variables to Line W
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
    new_cube = iris.cube.Cube(np.zeros((28, len(new_lats), len(new_lons)), np.float32),
                              dim_coords_and_dims=[(region.coord('tcell pstar'), 0),
                                                   (latitude, 1), (longitude, 2)])

    # Regrid
    regrid = region.regrid(new_cube, iris.analysis.Linear())

    # Isolate relevant lat-lon pairs
    new = np.ones((500, 28, len(new_lats))) * np.nan

    for t in range(0,len(regrid.coord('time').points)):
        for k in range(0,len(regrid.coord('tcell pstar').points)):
            for i in range(0,len(new_lats)):
                new[t,k,i] = regrid[t,k,i,i].data

    dist = calculate_distance_km(new_lats, new_lons)
    distance = iris.coords.DimCoord(dist, long_name='distance', units='km')
    model_linew = iris.cube.Cube(new, long_name=name, units=cube.units,
                                 dim_coords_and_dims=[(region.coord('time'), 0),
                                                      (region.coord('tcell pstar'), 1),
                                                      (distance, 2)])
    return model_linew

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
o2_sat = iris.load_cube(PATH+'o2_sat.nc')
o2 = o2[-500:]
o2_sat = o2_sat[-500:]

rho = iris.load_cube(PATH+'rho.nc')
age = iris.load_cube(PATH+'residency_age_surface.nc')
age.coord('Time').rename('time')

neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')
aou = (o2_sat/rho) - o2
aou.rename('Apparent Oxygen Utilization')

'''
## Interpolate model variables to Line W
aou_linew = interpolate_to_linew(aou, 'oxygen')
age_linew = interpolate_to_linew(age, 'age')
neutral_rho = interpolate_to_linew(neutral_rho, 'neutral_rho')

iris.save(aou_linew, 'aou_linew.nc')
iris.save(age_linew, 'age_linew.nc')
iris.save(neutral_rho, 'neutral_rho_linew.nc')
'''

### Load Line W Data
aou_linew = iris.load_cube('aou_linew.nc')
age_linew = iris.load_cube('age_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')

### Line W Calculations
# Calculate Climatology of Neutral Density
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)
age_clim = age.collapsed('time', iris.analysis.MEAN)
aou_clim = aou.collapsed('time', iris.analysis.MEAN)


# Calculate Correlation between age and oxygen
test = iris.analysis.stats.pearsonr(aou_linew, age_linew, corr_coords=['time'])

# Calculate climatology of neutral density
neutral_rho_mean = neutral_rho.collapsed('time', iris.analysis.MEAN)









######## Figures ##########
## Plot Age ang Oxygen on Line W
plt.figure(figsize=(16,10))
clevs = np.arange(-1, 1.1, 0.1)
ax = plt.subplot(2,2,1)
plt.contourf(aou_linew.coord('distance').points, aou_linew.coord('tcell pstar').points, test.data,clevs, cmap = 'RdBu_r')
cb = plt.colorbar()
cb.set_label('Pearson Correlation Coefficient', fontsize = 14)
CS = plt.contour(aou_linew.coord('distance').points, aou_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
ax.invert_yaxis()
plt.title('(a) Age-AOU Correlation', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 14)

plt.subplot(2,2,3)
aou_clim = aou_linew.collapsed('time', iris.analysis.MEAN)
clevs = np.arange(0, 130, 10)
ax = plt.gca()
plt.contourf(age_linew.coord('distance').points, aou_linew.coord('tcell pstar').points, aou_clim.data*1e6, clevs,
             cmap = cmaps.viridis, extend = 'both')
cb = plt.colorbar()
CS = plt.contour(age_linew.coord('distance').points, aou_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
ax.invert_yaxis()
cb.set_label('umol/kg', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 14)
plt.title('(b) AOU Climatology', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 14)

plt.subplot(2,2,4)
age_clim = age_linew.collapsed('time', iris.analysis.MEAN)
ax = plt.gca()
clevs = np.arange(0, 210, 10)
plt.contourf(aou_linew.coord('distance').points, aou_linew.coord('tcell pstar').points, age_clim.data, clevs, cmap = cmaps.viridis, extend = 'both')
cb = plt.colorbar()
CS = plt.contour(age_linew.coord('distance').points, aou_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.ylim([0,2000])
plt.clabel(CS, fontsize=9, inline=1)
ax.invert_yaxis()
plt.title('(c) Age Climatology', fontsize = 14)
cb.set_label('years', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 14)
plt.savefig('correlation_aou_clim_on_linew.pdf')

plt.show()
