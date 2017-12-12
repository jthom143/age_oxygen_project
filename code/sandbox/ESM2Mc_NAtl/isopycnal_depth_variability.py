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

## Define Functions  ###########################################################

def find_sigma_depth(sigma, sigma_level, depth_array, sigma_depth):
    for t in range(0,len(time)):
        for y in range(0, len(lons)):
            for x in range(0, len(lats)):
                f = interp.interp1d(sigma[t,:,x,y], depth_array, bounds_error=False)
                sigma_depth[t,x,y] = f(sigma_level)
    return sigma_depth

## Load Data  ##################################################################

PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'
o2 = iris.load_cube(PATH+'o2.nc')
o2 = o2[-500:]
age = iris.load_cube(PATH+'residency_age_surface.nc')
age.coord('Time').rename('time')
neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')

## Calculate the Standard Deviation of Neutral Density #########################

neutral_rho_std = neutral_rho.collapsed('time', iris.analysis.STD)
