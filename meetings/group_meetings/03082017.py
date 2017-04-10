# Code to make figures shown in March 8, 2017 group meeting

# Load Packages
import iris
import numpy as np
import matplotlib.pyplot as plt
import iris.analysis.stats
import iris.quickplot as qplt
import seaborn as sns
import iris.analysis.cartography
import cartopy.crs as ccrs


import sys
sys.path.append('/RESEARCH/paper_ocean_heat_carbon/code/python')
import colormaps as cmaps


# Load Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'
o2 = iris.load_cube(PATH+'o2.nc')
age = iris.load_cube(PATH+'residency_age_surface.nc')

# Change coordinate name on age
age.coord('Time').rename('time')

# Trim ocean bling Data
o2 = o2[-500:]


# Calculate Correlation
correlation = iris.analysis.stats.pearsonr(age, o2, corr_coords='time')

'''
# Calculate Climatology
age_clim = age.collapsed('time', iris.analysis.MEAN)
o2_clim = o2.collapsed('time', iris.analysis.MEAN)

# Create figure of age and oxygen climatologies
lats = age.coord('latitude').points
lons = age.coord('longitude').points

clevs = np.arange(0, 501, 1)
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, age_clim[5,:,:].data, clevs, extend='max', cmap = cmaps.viridis)
ax.coastlines()
plt.title('Age ~60m')
plt.tight_layout()
plt.savefig('age1.pdf')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, age_clim[10,:,:].data,clevs, extend='max', cmap = cmaps.viridis)
ax.coastlines()
plt.title('Age ~140m')
plt.tight_layout()
plt.savefig('age2.pdf')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, age_clim[20,:,:].data,clevs, extend='max', cmap = cmaps.viridis)
ax.coastlines()
plt.title('Age ~2000m')
plt.colorbar(orientation='horizontal')
plt.tight_layout()
plt.savefig('age3.pdf')

clevs = np.arange(0, 0.0004, 0.00001)
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_clim[5,:,:].data,clevs, extend='max', cmap = cmaps.viridis)
ax.coastlines()
plt.title('Oxygen ~60m')
plt.tight_layout()
plt.savefig('o21.pdf')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_clim[10,:,:].data,clevs, extend='max', cmap = cmaps.viridis)
ax.coastlines()
plt.title('Oxygen ~140m')
plt.tight_layout()
plt.savefig('o22.pdf')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, o2_clim[20,:,:].data,clevs, extend='max', cmap = cmaps.viridis)
ax.coastlines()
plt.title('Oxygen ~2000m')
plt.colorbar(orientation='horizontal')
plt.tight_layout()
plt.savefig('o23.pdf')
plt.show()
'''
lats = age.coord('latitude').points
lons = age.coord('longitude').points

clevs = np.arange(-1, 1.05, 0.05)
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, correlation[5,:,:].data, clevs, cmap = 'RdBu_r')
ax.coastlines()
plt.title('Depth ~60', fontsize = 20)
plt.tight_layout()
plt.savefig('corr1.pdf')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, correlation[10,:,:].data,clevs, cmap = 'RdBu_r')
ax.coastlines()
plt.title('Depth ~140', fontsize = 20)
plt.tight_layout()
plt.savefig('corr2.pdf')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, correlation[20,:,:].data,clevs, cmap = 'RdBu_r')
ax.coastlines()
plt.title('Depth ~2000m', fontsize = 20)
plt.colorbar(orientation='horizontal')
plt.tight_layout()
plt.savefig('corr3.pdf')
