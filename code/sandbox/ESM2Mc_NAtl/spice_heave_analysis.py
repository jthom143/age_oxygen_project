## Spice and Heave analysis
## Methods from Doney et al, 2006

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
def find_sigma_z(sigma, z):
    constraint = iris.Constraint(depth=lambda zz: z-10 < zz < z+10)
    sigma_z = sigma.extract(constraint)

    return sigma_z

def find_sigma_depth(sigma_mean, sigma_z, depth_array, sigma_depth, lons, lats):
    for t in range(0, 500):
        for y in range(0, len(lons)):
            for x in range(0, len(lats)):
                f = interp.interp1d(sigma_mean[:,x,y], depth_array, bounds_error=False)
                sigma_depth[t,x,y] = f(sigma_z[t,x,y].data)
    return sigma_depth

def var_on_isopycnal(var, depth, sigma_depth, var_isopycnal, lons, lats):
    for t in range(0, 500):
        for y in range(0, len(lons)):
            for x in range(0, len(lats)):
                f = interp.interp1d(depth, var[t,:,x,y], bounds_error=False)
                var_isopycnal[t,x,y] = f(sigma_depth[t,x,y])
    return var_isopycnal

def calc_spice_heave(var, z, density):
    # Find density surface at depth 513 for each year
    print 'calculating density surface at given depth'
    sigma_z = find_sigma_z(density, z)

    # Interpolate correlation onto neutral density surface (found above)
    lons = neutral_rho.coord('longitude').points
    lats = neutral_rho.coord('latitude').points
    depth = neutral_rho.coord('depth').points

    # Depth of density surface
    print 'interpolating to find depth of mean density surface over time'
    depth_sigma_z = np.ones((500,42,29)) * np.nan
    neutral_rho_clim = neutral_rho.collapsed('time', iris.analysis.MEAN)
    depth_sigma_z = find_sigma_depth(neutral_rho_clim.data, sigma_z, depth, depth_sigma_z, lons, lats)

    # Interpolate to find the value of the variable on the density surface
    print 'interpolating to find variable on density surface'
    var_sigma_z = np.ones((500,42,29)) * np.nan
    var_sigma_z = var_on_isopycnal(var.data, depth, depth_sigma_z, var_sigma_z, lons, lats)

    # Average to find the long-term mean of the variable on the chosen depth
    var_z = find_sigma_z(salt, z)
    var_z_mean = var_z.collapsed('time', iris.analysis.MEAN).data

    print 'Calculating Heave and Spice'
    # calculate the heave
    var_heave = var_sigma_z - var_z_mean

    # Calculate Spice
    var_z = find_sigma_z(var, 513)
    var_spice = var_z - var_sigma_z

    return var_heave, var_spice


## Load Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'
temp = iris.load_cube(PATH+'temp.nc')
salt = iris.load_cube(PATH+'salt.nc')
neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')

temp.coord('tcell pstar').rename('depth')
salt.coord('tcell pstar').rename('depth')
neutral_rho.coord('tcell pstar').rename('depth')

## Trim data to N. Atlantic
constraint = iris.Constraint(latitude=lambda y: 0 < y < 90)
neutral_rho = neutral_rho.extract(constraint)
temp = temp.extract(constraint)
salt = salt.extract(constraint)

constraint = iris.Constraint(longitude=lambda x: -80 < x < 5)
neutral_rho = neutral_rho.extract(constraint)
temp = temp.extract(constraint)
salt = salt.extract(constraint)

salt_heave, salt_spice = calc_spice_heave(salt, 513, neutral_rho-1000)
