### Script to create figure of line_w age and o2/aou correlation for model data

# Created November 24, 2017

# Load python packages
import numpy as np
import pandas as pd
import gsw
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy.ma as ma
import glob
from math import sin, cos, sqrt, atan2, radians
import csv
import scipy.stats
import iris.analysis.stats
import iris
from scipy.stats import t as ttest

import sys
sys.path.append('/RESEARCH/chapter3/functions')
import colormaps as cmaps
from o2sat import o2sat



################################################################################
# Load Model Data
################################################################################
import iris
import iris.quickplot as qplt
import iris.analysis.stats

PATH = '/RESEARCH/chapter3/data/newCO2_control_800/derived/linew/'
o2_linew = iris.load_cube(PATH+'o2_linew.nc')
aou_linew = iris.load_cube(PATH+'aou_linew.nc')
age_linew = iris.load_cube(PATH+'age_linew.nc')
neutral_rho_linew = iris.load_cube(PATH+'neutral_rho_linew.nc')

# Calculate Correlations
correlation = iris.analysis.stats.pearsonr(o2_linew, age_linew,
                                           corr_coords=['time'])

correlation_aou = iris.analysis.stats.pearsonr(aou_linew, age_linew,
                                               corr_coords=['time'])

# Calculate climatology of neutral density
neutral_rho_linew_avg = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)


# Create figures
dist = correlation.coord('distance').points
depth = correlation.coord('tcell pstar').points

fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,2,1)
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(dist, depth, correlation.data, clevs, cmap = 'RdBu_r')
CS = plt.contour(dist, depth, neutral_rho_linew_avg.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,800])
ax.invert_yaxis()
plt.title('(a) Age vs Oxygen', fontsize = 13)
plt.ylabel('Depth (dbars)')
#plt.savefig('obs_age_o2_corr.png')
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.xlabel('Distance (km)')


ax = plt.subplot(1,2,2)
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(dist, depth, correlation_aou.data, clevs, cmap = 'RdBu_r')
CS = plt.contour(dist, depth, neutral_rho_linew_avg.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,800])
plt.yticks([0, 50, 1000, 1500, 2000],[' ', ' ', ' ', ' ', ' '])
ax.invert_yaxis()
plt.title('(b) Age vs AOU', fontsize = 13)
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.xlabel('Distance (km)')

fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('Correlation Coefficient', fontsize = 12)
plt.savefig('model_correlation.png')



################################################################################
# Scatter Plots
################################################################################

fig = plt.figure(figsize = (10,5))
ax1 = plt.subplot(1,2,1)
for i in range(0,500):
    plt.scatter(age_linew[i,:,0:6].data, o2_linew[i,:,0:6].data*1e6,
                c = correlation[:,0:6].data.flatten(), cmap = 'RdBu_r', s = 30, lw = 0.3, vmin=-1, vmax=1)
plt.xlim([-25, 275])
#plt.xlabel('Age (years)')
plt.ylabel('Oxygen (umol/kg)')
plt.xlabel('Age (years)')
plt.title('(a) Age vs Oxygen', fontsize = 13)

x = np.arange(0, 276, 1)


ax2 = plt.subplot(1,2,2)
plt.plot(x, 0.4*x, color = 'grey', ls = '--')
plt.plot(x, 0.8*x, color = 'grey', ls = '--')
plt.plot(x, 1.7*x, color = 'grey', ls = '--')
plt.plot(x, 100*x, color = 'grey', ls = '--')
plt.plot(x, 0.2*x, color = 'grey', ls = '--')
plt.plot(x, 0*x, color = 'grey', ls = '--')

for i in range(0,500):
     plt.scatter(age_linew[i,:,0:6].data, aou_linew[i,:,0:6].data*1e6,
                c = correlation_aou[:,0:6].data.flatten(), cmap = 'RdBu_r', s = 30, lw = 0.3, vmin=-1, vmax=1)


plt.xlim([-25, 275])
plt.ylim([-20, 120])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
#plt.xlabel('Age (years)')
plt.ylabel('AOU (umol/kg)')
plt.xlabel('Age (years)')
plt.title('(b) Age vs AOU', fontsize = 13)

fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('Correlation Coefficient', fontsize = 12)
plt.savefig('model_scatterplot.png')



plt.show()
