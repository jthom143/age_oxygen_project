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

### Load Line W Data
o2_linew = iris.load_cube('o2_linew.nc')
age_linew = iris.load_cube('age_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')

### Line W Calculations
# Calculate Climatology of Neutral Density
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)

# Calculate Correlation between age and oxygen
test = iris.analysis.stats.pearsonr(o2_linew, age_linew, corr_coords=['time'])

## Isolate age and oxygen at distance 600km
constraint = iris.Constraint(distance=lambda x: 600 < x < 700)
age_600 = age_linew.extract(constraint)
o2_600 = o2_linew.extract(constraint)

# Isolate age and oxygen at distance 200km
constraint = iris.Constraint(distance=lambda x: 150 < x < 250)
age_200 = age_linew.extract(constraint)
o2_200 = o2_linew.extract(constraint)

# Calculate the vertical gradient of age and oxygen at distance 600km
age_600_gradient = np.gradient(age_600[0,:].data)
o2_600_gradient = np.gradient(o2_600[0,:].data)

age_200_gradient = np.gradient(age_200[0,:].data)
o2_200_gradient = np.gradient(o2_200[0,:].data)

age_600_diff = iris.analysis.calculus.differentiate(age_600, 'tcell pstar')
o2_600_diff = iris.analysis.calculus.differentiate(o2_600, 'tcell pstar')


plt.figure(figsize=(16,10))
clevs = np.arange(-1, 1.1, 0.1)
ax = plt.subplot(2,2,1)
plt.contourf(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, test.data,clevs, cmap = 'RdBu_r')
cb = plt.colorbar()
cb.set_label('Pearson Correlation Coefficient', fontsize = 14)
CS = plt.contour(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
ax.invert_yaxis()
plt.title('(a) Age-Oxygen Correlation', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 14)

plt.subplot(2,2,3)
o2_clim = o2_linew.collapsed('time', iris.analysis.MEAN)
clevs = np.arange(120, 330, 10)
ax = plt.gca()
plt.contourf(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, o2_clim.data*1e6, clevs,
             cmap = cmaps.viridis, extend = 'both')
cb = plt.colorbar()
CS = plt.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
ax.invert_yaxis()
cb.set_label('umol/kg', fontsize = 14)
plt.ylabel('Depth [m]', fontsize = 14)
plt.title('(b) Oxygen Climatology', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 14)

plt.subplot(2,2,4)
age_clim = age_linew.collapsed('time', iris.analysis.MEAN)
ax = plt.gca()
clevs = np.arange(0, 210, 10)
plt.contourf(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, age_clim.data, clevs, cmap = cmaps.viridis, extend = 'both')
cb = plt.colorbar()
CS = plt.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                 levels = [26.0, 26.5, 27.0, 27.5])
plt.ylim([0,2000])
plt.clabel(CS, fontsize=9, inline=1)
ax.invert_yaxis()
plt.title('(c) Age Climatology', fontsize = 14)
cb.set_label('years', fontsize = 14)
plt.xlabel('Distance [km]', fontsize = 14)

fig = plt.figure()
ax1 = plt.subplot(1,2,1)
plt.plot(age_600[0,:].data, age_600.coord('tcell pstar').points, lw = 2)
#plt.plot(age_200[0,:].data, age_600.coord('tcell pstar').points, lw = 2, ls = '--', color = 'b')
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age (years)', color='b')
ax1.axhline(400, color = 'k', ls = '--')
ax1.axhline(700, color = 'k', ls = '--')
ax2 = ax1.twiny()
plt.plot(o2_600[0,:].data*1e6, age_600.coord('tcell pstar').points, color = 'r', lw = 2)
#plt.plot(o2_200[0,:].data*1e6, age_600.coord('tcell pstar').points, color = 'r', lw = 2, ls = '--')
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg)', color='r')
plt.ylim([0,2000])
ax2.invert_yaxis()
ax2.set_ylabel('Depth [dbars]')
#plt.title('(a) Vertical Profile')



ax1 = plt.subplot(1,2,2)
plt.plot(age_600_diff[0,:].data, age_600_diff.coord('tcell pstar').points, lw = 2)
ax1.tick_params('x', colors='b')
ax1.set_xlabel('Age (year/dbar)', color='b')
ax1.axhline(400, color = 'k', ls = '--')
ax1.axhline(700, color = 'k', ls = '--')
plt.xlim([-0.8, 0.8])
ax2 = ax1.twiny()
plt.plot(o2_600_diff[0,:].data*1e6, age_600_diff.coord('tcell pstar').points, color = 'r', lw = 2)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg/dbar)', color='r')
plt.xlim([-0.35, 0.35])
plt.ylim([0,2000])
ax2.invert_yaxis()
plt.subplots_adjust(top=0.8)
plt.gcf().text(0.61, 0.93, '(b) Vertical Gradeint', fontsize=14)
plt.gcf().text(0.21, 0.93, '(a) Vertical Profile', fontsize=14)
plt.gcf().text(0.04, 0.48, 'Depth [dbars]', fontsize=12, rotation = 'vertical')

plt.show()

'''
fig, ax1 = plt.subplots()
plt.plot(age_600[:,14].data - age_600[:,14].collapsed('time', iris.analysis.MEAN).data, lw = 2)
ax1.tick_params('y', colors='b')
ax1.set_ylabel('Age [years]', color='b')
plt.ylim([-30, 30])
ax1.axhline(0, color = 'k', ls = '--')
ax2 = ax1.twinx()
plt.plot((o2_600[:,14].data - o2_600[:,14].collapsed('time', iris.analysis.MEAN).data)*1e6, color = 'r', lw = 2)
plt.xlabel('Time [years]')
ax2.tick_params('y', colors='r')
ax2.set_ylabel('Oxygen [umol/kg]', color='r')
plt.ylim([-15, 15])
'''



## Plot the first 20 years
clevs = np.arange(-20, 21, 1)

fig = plt.figure(figsize = (16,10))

for i in range(0, 20):
    ax = plt.subplot(5,4,i+1)
    plt.contourf(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, (o2_linew[i,:,:].data-o2_clim.data)*1e6, clevs, cmap = 'RdBu_r')
    CS = plt.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                    levels = [26.0, 26.5, 27.0, 27.5])
    plt.clabel(CS, fontsize=9, inline=1)
    plt.ylim([0,2000])
    ax.invert_yaxis()


fig = plt.figure(figsize = (16,10))

for i in range(0, 20):
    ax = plt.subplot(5,4,i+1)
    plt.contourf(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, (age_linew[i,:,:].data-age_clim.data), clevs, cmap = 'RdBu_r', extend = 'both')
    CS = plt.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data - 1000, colors = 'k' ,
                    levels = [26.0, 26.5, 27.0, 27.5])
    plt.clabel(CS, fontsize=9, inline=1)
    plt.ylim([0,2000])
    ax.invert_yaxis()


'''
files = []
fig= plt.figure(figsize=(10, 5))
ax = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

for i in range(0, 20):  # 50 frames
    ax.cla()
    ax2.cla()
    ax.set_title('(a) Age')
    ax2.set_title('(b) Oxygen')
    ax.set_ylabel('Depth [dbars]')
    ax.set_xlabel('Distance [km]')
    ax2.set_xlabel('Distance [km]')
    im = ax.contourf(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, (age_linew[i,:,:].data-age_clim.data), clevs, cmap = 'RdBu_r', extend = 'both')
    #cb = plt.colorbar(im, ax, orientation = 'horizontal')
    #cb.set_label('years', fontsize = 12)
    CS = ax.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew[i,:,:].data - 1000, colors = 'k' ,
                levels = [26.0, 26.5, 27.0, 27.5])
    plt.clabel(CS, fontsize=9, inline=1)
    im2 = ax2.contourf(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, (o2_linew[i,:,:].data-o2_clim.data)*1e6, clevs, cmap = 'RdBu_r', extend = 'both')
    #cb = plt.colorbar(im2, ax2, orientation = 'horizontal')
    #cb.set_label('umol/kg', fontsize = 12)
    CS = ax2.contour(age_linew.coord('distance').points, o2_linew.coord('tcell pstar').points, neutral_rho_linew[i,:,:].data - 1000, colors = 'k' ,
                levels = [26.0, 26.5, 27.0, 27.5])
    plt.clabel(CS, fontsize=9, inline=1)
    ax.set_ylim([0,2000])
    ax2.set_ylim([0,2000])
    ax.invert_yaxis()
    ax2.invert_yaxis()
    fname = '_tmp%03d.png' % i
    print('Saving frame', fname)
    plt.savefig(fname)
    files.append(fname)
'''


plt.show()
