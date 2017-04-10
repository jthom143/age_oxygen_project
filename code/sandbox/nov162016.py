###############################################################################
# Script plot along isopycnal temperature gradient
###############################################################################
import iris
import iris.analysis.cartography
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import numpy as np
from netCDF4 import Dataset
import gsw
import scipy.interpolate as interp


import sys # access system routines
sys.path.append('~/control_aredi800/python')
from calc_OHC_OCC import calc_occ, calc_ohc
import colormaps as cmaps

plt.ion()

def convection_composite(var):
    convect1 = var[123:128]
    convect2 = var[230:235]

    convect1 = convect1.collapsed('time', iris.analysis.MEAN)
    convect2 = convect2.collapsed('time', iris.analysis.MEAN)

    var_convect = (convect1+convect2)/2

    return var_convect

## Load Data
PATH='/datascope/gnana_esms/jthom143/data/newCO2_control_800/'
area = iris.load_cube(PATH+'area_t.nc')
rhodzt = iris.load_cube(PATH+'rho_dzt.nc')
temp = iris.load_cube(PATH+'temp.nc')
salt = iris.load_cube(PATH+'salt.nc')

# Calculate potential density
CT = temp.data
SA = salt.data

sigma0 = gsw.sigma0(SA, CT)
sigma0 = iris.cube.Cube(sigma0, long_name = 'Potential Density', units = 'kg m^-3',
    dim_coords_and_dims = [(temp.coord('time'), 0), (temp.coord('tcell pstar'), 1),
    (temp.coord('latitude'), 2), (temp.coord('longitude'), 3)])


## Zonal Average
temp_zonal = temp.collapsed('longitude', iris.analysis.MEAN)
sigma0_zonal = sigma0.collapsed('longitude', iris.analysis.MEAN)
## Calculate Climatologies
temp_zonal_climatology = temp_zonal.collapsed('time', iris.analysis.MEAN)
sigma0_zonal_climatology = sigma0_zonal.collapsed('time', iris.analysis.MEAN)

# Plot figure of temperature with depth and overlay potential density contour
cmap_fig1 = cmaps.inferno
fig1 = plt.figure(figsize = (8, 4))
clevs = np.arange(-2, 24, 2)
clevs_contour = np.arange(20, 29, 1)
lats = temp_zonal_climatology.coord('latitude').points
depth = temp_zonal_climatology.coord('tcell pstar').points
im1 = plt.contourf(lats, depth, temp_zonal_climatology.data, clevs, cmap = cmap_fig1, extend = 'both')
CS = plt.contour(lats, depth, sigma0_zonal_climatology.data, clevs_contour, colors='white')
plt.ylim([0,1000])
plt.xlim([-80,0])
plt.gca().invert_yaxis()
cbar = fig1.colorbar(im1, orientation='vertical')
cbar.set_label('$^{o}$ C')
manual_locations = [(-10, 805)]
plt.clabel(CS, inline=1, fontsize=10, manual = manual_locations)

# Interpolate to find the depth of the 27 isopycnal
depth1d = sigma0_zonal.coord('tcell pstar').points  # Create array of depth coordinates
depth = np.tile(depth1d, [80,1])                    # Create matrix of depth coordinates for each latitude
depth = depth.T                                     # Transpose matrix to match shape of latitude and sigma0

lat1d = sigma0_zonal.coord('latitude').points       # Create array of latitude coordinates
lat = np.tile(lat1d, [28,1])

result = np.zeros(500)
for t in range(0,500):
    y1 = 7  # Index of -60 deg
    y2 = 13 # Index of -50 deg

    #Interpolate to find the depth of the 27 isopycnal
    f = interp.interp2d(sigma0_zonal[t].data,lat, depth, kind='quintic')
    depth27 = f(27.00, lat1d)

    # Interpolate to get temperature on isopycnal
    isopycnal_temp = np.zeros((80,1))*np.nan
    f = interp.interp2d(depth, lat, temp_zonal[t].data, kind = 'linear')
    for y in range(0, 80):
        z = f(depth27[y],lat1d)
        isopycnal_temp[y] = z[y]

    result[t] = (isopycnal_temp[y2] - isopycnal_temp[y1])/(lats[y2]-lats[y1])


## Calculate temperature gradient
#dx = np.gradient(lat1d)
#gradient = np.gradient(isopycnal_temp[:,0],dx)

'''
## Calculate isopycnal temperature for convective composite
sigma0_convective = convection_composite(sigma0_zonal)
temp_convective = convection_composite(temp_zonal)


cmap_fig1 = cmaps.inferno
fig1 = plt.figure(figsize = (8, 4))
clevs = np.arange(-0.6, 0.61, 0.01)
clevs_contour = np.arange(20, 29, 1)
lats = temp_zonal_climatology.coord('latitude').points
depth = temp_zonal_climatology.coord('tcell pstar').points
im1 = plt.contourf(lats, depth, temp_convective.data - temp_zonal_climatology.data, clevs, cmap = 'RdBu_r', extend = 'both')
CS = plt.contour(lats, depth, sigma0_convective.data, clevs_contour, colors='k')
plt.ylim([0,1000])
plt.xlim([-80,0])
plt.gca().invert_yaxis()
cbar = fig1.colorbar(im1, orientation='vertical')
cbar.set_label('$^{o}$ C')
manual_locations = [(-10, 805)]
plt.clabel(CS, inline=1, fontsize=10, manual = manual_locations)
'''
