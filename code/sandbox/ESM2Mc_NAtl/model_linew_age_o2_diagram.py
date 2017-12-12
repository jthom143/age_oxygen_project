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

def interpolate_to_line2(cube, name):
    # Isolate Region of Interest to limit interpolation
    lat_constraint = iris.Constraint(latitude=lambda x: 25 <= x <= 50)
    lon_constraint = iris.Constraint(longitude=lambda y: -85 <= y <= -40)
    region = cube.extract(lat_constraint)
    region = region.extract(lon_constraint)

    # Interpolate onto 1deg x 1deg grid
    new_lats = np.arange(37, 46, 1)
    new_lons = np.linspace(-50, -54, num=9)

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


def interpolate_to_line3(cube, name):
    # Isolate Region of Interest to limit interpolation
    lat_constraint = iris.Constraint(latitude=lambda x: 25 <= x <= 50)
    lon_constraint = iris.Constraint(longitude=lambda y: -85 <= y <= -40)
    region = cube.extract(lat_constraint)
    region = region.extract(lon_constraint)

    # Interpolate onto 1deg x 1deg grid
    new_lats = np.linspace(36, 37, num=9)
    new_lons = np.linspace(-15, -23, num=9)

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


PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'

# Load Data
age_linew = iris.load_cube('age_linew.nc')
o2_linew = iris.load_cube('o2_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')

# Climatology
age_linew_clim = age_linew.collapsed('time', iris.analysis.MEAN)
o2_linew_clim = o2_linew.collapsed('time', iris.analysis.MEAN)


# Create Figure
depth = age_linew.coord('tcell pstar').points
dist = age_linew.coord('distance').points

plt.figure()
plt.scatter(age_linew_clim[:,:].data.flatten(), o2_linew_clim[:,:].data.flatten()*1e6, color = 'b')
#plt.scatter(age_linew_clim[:24,:].data.flatten(), o2_linew_clim[:24,:].data.flatten()*1e6, color = 'r')
#plt.scatter(age_linew_clim[:21,:].data.flatten(), o2_linew_clim[:21,:].data.flatten()*1e6, color = 'g')
#plt.scatter(age_linew_clim[:18,:].data.flatten(), o2_linew_clim[:18,:].data.flatten()*1e6, color = 'c')
#plt.scatter(age_linew_clim[:15,:].data.flatten(), o2_linew_clim[:15,:].data.flatten()*1e6, color = 'k')
#plt.scatter(age_linew_clim[:10,:].data.flatten(), o2_linew_clim[:10,:].data.flatten()*1e6, color = 'orange')

plt.text(-47, 187, '100 m', fontsize = 14)
plt.text(100, 135, '500 m', fontsize = 14)
plt.text(121, 215, '1000 m', fontsize = 14)
plt.text( 75, 250, '2000 m', fontsize = 14)
plt.text( 73, 265, '3000 m', fontsize = 14)
#plt.text( 90, 265, '4000 m')

plt.plot(25, 193, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(92, 178, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(103, 215, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(58, 251, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(62, 260, marker = 'o', ls = '  ', color = 'k', ms = 8)

plt.xlabel('Age (years)')
plt.ylabel('Oxygen (umol/kg)')
plt.title('ESM2Mc Line-W Age-O2 Diagram')
plt.savefig('ESM2Mc_linew_age_o2.png')
plt.show()















'''
o2_line2 = interpolate_to_line2(o2, 'oxygen')
age_line2 = interpolate_to_line2(age, 'age')

age_line2_clim = age_line2.collapsed('time', iris.analysis.MEAN)
o2_line2_clim = o2_line2.collapsed('time', iris.analysis.MEAN)

plt.scatter(age_line2_clim.data.flatten(), o2_line2_clim.data.flatten()*1e6, color = 'r')
plt.scatter(age_linew_clim.data.flatten(), o2_linew_clim.data.flatten()*1e6)

plt.xlabel('Age (years)')
plt.ylabel('Oxygen (umol/kg)')
plt.title('ESM2Mc Age-O2 Diagram')
plt.savefig('ESM2Mc_2lines_age_o2.png')


#o2_line3 = interpolate_to_line3(o2, 'oxygen')
#age_line3 = interpolate_to_line3(age, 'age')

#age_line3_clim = age_line3.collapsed('time', iris.analysis.MEAN)
#o2_line3_clim = o2_line3.collapsed('time', iris.analysis.MEAN)

plt.figure()
plt.scatter(age_line3_clim.data.flatten(), o2_line3_clim.data.flatten()*1e6, color = 'g')
plt.xlabel('Age (years)')
plt.ylabel('Oxygen (umol/kg)')
plt.title('ESM2Mc Age-O2 Diagram')

# Plot Transects
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.figure(figsize=(12,10))
new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)

new_lats2 = np.arange(37, 46, 1)
new_lons2 = np.linspace(-50, -54, num=9)

new_lats3 = np.linspace(37, 37, num=9)
new_lons3 = np.linspace(-15, -23, num=9)


ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent((-90, 0, 0, 70), crs=ccrs.PlateCarree())
plt.plot(new_lons, new_lats, color = 'b', marker = '*', transform=ccrs.PlateCarree())
plt.plot(new_lons2, new_lats2, color = 'r', marker = '*', transform=ccrs.PlateCarree())
plt.plot(new_lons3, new_lats3, color = 'g', marker = '*', transform=ccrs.PlateCarree())

ax.stock_img()
ax.coastlines()
plt.show()
'''
