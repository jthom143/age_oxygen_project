## Oxygen Isopycnal Heave

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
    new_cube = iris.cube.Cube(np.zeros((27  , len(new_lats), len(new_lons)), np.float32),
                              dim_coords_and_dims=[(region.coord('tcell pstar'), 0),
                                                   (latitude, 1), (longitude, 2)])

    # Regrid
    regrid = region.regrid(new_cube, iris.analysis.Linear())

    # Isolate relevant lat-lon pairs
    new = np.ones((500, 27, len(new_lats))) * np.nan

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

def calculate_heave(var, density):
    # Calculate climatology
    var_clim = var.collapsed('time', iris.analysis.MEAN)
    density_clim = density.collapsed('time', iris.analysis.MEAN)

    # Calculate vertical gradient of average (time??) variable
    var_gradient = iris.analysis.calculus.differentiate(var_clim, 'tcell pstar')

    # Calculate vertical gradient of average (time??) density
    density_gradient = iris.analysis.calculus.differentiate(density_clim, 'tcell pstar')

    # Calculate density anomaly
    density_anomaly = density - density_clim

    # Interpolate density anomaly and variable to match vertical graident vertical coordinate
    density_anomaly = density_anomaly.interpolate([('tcell pstar', density_gradient.coord('tcell pstar').points)], iris.analysis.Linear())
    var = var.interpolate([('tcell pstar', density_gradient.coord('tcell pstar').points)], iris.analysis.Linear())

    # Calculate heave contribution
    heave = np.ones((500, 27, 80, 120))*np.nan
    for t in range(0,500):
        heave[t,:,:,:] = var[t,:,:,:].data - (var_gradient/density_gradient).data*density_anomaly[t,:,:,:].data

    heave = iris.cube.Cube(heave, long_name=var.name, units=var.units,
                                 dim_coords_and_dims=[(var.coord('time'), 0),
                                                      (var.coord('tcell pstar'), 1),
                                                      (var.coord('latitude'), 2),
                                                      (var.coord('longitude'), 3)])

    return heave

## Load Data
## Load Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'
o2 = iris.load_cube(PATH+'o2.nc')
o2 = o2[-500:]

age = iris.load_cube(PATH+'residency_age_surface.nc')
age.coord('Time').rename('time')

neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')

## Calculate Isopycnal Heave Component of Oxygen
o2_heave = calculate_heave(o2, neutral_rho)

## Interpolate to Line W and Save
o2_heave_linew = interpolate_to_linew(o2_heave, 'oxygen isopycnal heave')
iris.save(o2_heave_linew, 'o2_heave_linew.nc')
