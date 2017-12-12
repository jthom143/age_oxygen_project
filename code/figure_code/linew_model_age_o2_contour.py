################################################################################
# Model Data
################################################################################
import iris
import iris.quickplot as qplt
import iris.analysis.stats
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/RESEARCH/chapter3/functions')
import colormaps as cmaps

# Load Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/derived/linew/'
o2_linew = iris.load_cube(PATH+'o2_linew.nc')
aou_linew = iris.load_cube(PATH+'aou_linew.nc')
age_linew = iris.load_cube(PATH+'age_linew.nc')
neutral_rho_linew = iris.load_cube(PATH+'neutral_rho_linew.nc')

# Calculate Climatology
o2_clim = o2_linew.collapsed('time', iris.analysis.MEAN)
age_clim = age_linew.collapsed('time', iris.analysis.MEAN)
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)

# Create Figure
fig = plt.figure(figsize = (12,4))
ax = plt.subplot(1,2,1)
clevs = np.arange(120, 310, 10)
plt.contourf(o2_clim.coord('distance').points, o2_clim.coord('tcell pstar').points,
             o2_clim.data*1e6, clevs, cmap = cmaps.viridis, extend = 'both')
cb = plt.colorbar()
cb.set_label('umol/kg', fontsize = 12)
CS = plt.contour(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,700])
ax.invert_yaxis()
plt.title('(a) Model Oxygen Climatology', fontsize = 13)
plt.xlabel('Distance (km)')
plt.ylabel('Depth (dbars)')

ax = plt.subplot(1,2,2)
clevs = np.arange(0, 200, 10)
plt.contourf(o2_clim.coord('distance').points, o2_clim.coord('tcell pstar').points,
             age_clim.data, clevs, cmap = cmaps.viridis, extend = 'max')
cb = plt.colorbar()
cb.set_label('years', fontsize = 12)
CS = plt.contour(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,700])
ax.invert_yaxis()
plt.title('(b) Model Age Climatology', fontsize = 13)
plt.xlabel('Distance (km)')
fig.subplots_adjust(bottom=0.2)
plt.savefig('age_oxygen_model_clim.png')
plt.savefig('/RESEARCH/chapter3/paper/brainstorming/figures/age_oxygen_model_clim.png')

plt.show()
