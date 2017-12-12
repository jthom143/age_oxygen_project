### Script to create figures for Anand's AMOC poster May 2017.

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

import sys
sys.path.append('/RESEARCH/chapter3/functions')
import colormaps as cmaps
from o2sat import o2sat

# Define Functions
def calculate_distance_km(lat, lon):
    # Function to calculate distance from Cape Cod for line W data.
    # function calls for two Pandas data series (latitude and longitude)

    # approximate radius of earth in km
    R = 6373.0

    # convert series to array:
    lat = lat.as_matrix()
    lon = lon.as_matrix()

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

def retrieve_old_grid(frame):
    depi = frame.press.values
    disti = frame.distance.values
    old_grid = (disti.flatten(), depi.flatten())

    return old_grid

## Load Data
PATH = '/RESEARCH/chapter3/data/GO_SHIP/linew/'
allFiles = glob.glob(PATH + "/*.csv")
raw_data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, header = 1, delim_whitespace = True, na_values = '-9.000')
    list_.append(df)
raw_data = pd.concat(list_)

data = raw_data[['Age', 'LAT', 'LON', 'CTDPRS', 'CTDTMP', 'CTDSAL', 'OXYGEN', 'pCFC-12', 'CFC12age', 'Mean']].copy()
data.columns = ['date', 'latitude', 'longitude', 'press', 'temp', 'salt', 'oxygen', 'pCFC12', 'CFC12age', 'mean_age']

## Calculate the distance from Cape Cod for all measurements
data['latitude'] = data['latitude'].astype(dtype=float)
data['longitude'] = data['longitude'].astype(dtype=float)
data['distance'] = calculate_distance_km(data.latitude, data.longitude)


# New Grid:
dist = np.arange(0,600,10)
depth = np.arange(0,5000,100)
XI, YI = np.meshgrid(dist,depth)
new_grid = (XI, YI)

# Create Array for Age and Oxygen Data Interpolated onto a grid:
years = np.unique(data.date)

o2_gridded  = np.ones((len(years), len(depth), len(dist))) * np.nan
age_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan

for i in range(0, len(years)):
    year = years[i]

    linew_year = data[data.date==year]
    old_grid = retrieve_old_grid(linew_year)
    age_grid = griddata(old_grid, linew_year.mean_age.values.flatten(), new_grid,
                        method='linear')
    o2_grid = griddata(old_grid, linew_year.oxygen.values.flatten(), new_grid,
                        method='linear')

    o2_gridded[i,:,:]  = o2_grid
    age_gridded[i,:,:] = age_grid



fig = plt.figure(figsize=(7,11))
clevs = np.arange(50, 270, 20)

ax = plt.subplot(7,2,1)
plt.contourf(XI, YI, age_gridded[9,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(a) Nov 2003')

ax = plt.subplot(7,2,3)
plt.contourf(XI, YI, age_gridded[10,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(b) May 2004')

ax = plt.subplot(7,2,5)
plt.contourf(XI, YI, age_gridded[11,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(c) Oct 2004')

ax = plt.subplot(7,2,7)
plt.contourf(XI, YI, age_gridded[12,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.ylabel('Depth (dbars)', fontsize = 14)
plt.title('(d) May 2005')

ax = plt.subplot(7,2,9)
plt.contourf(XI, YI, age_gridded[13,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(e) Oct 2005')

ax = plt.subplot(7,2,11)
plt.contourf(XI, YI, age_gridded[14,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(f) May 2006')


ax = plt.subplot(7,2,13)
plt.contourf(XI, YI, age_gridded[15,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
plt.yticks([0, 2000, 4000])
plt.xlabel('Distance (km)', fontsize = 14)
plt.title('(g) Oct 2006')

ax = plt.subplot(7,2,2)
plt.contourf(XI, YI, age_gridded[16,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(h) May 2007')

ax = plt.subplot(7,2,4)
plt.contourf(XI, YI, age_gridded[17,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(i) Oct 2007')

ax = plt.subplot(7,2,6)
plt.contourf(XI, YI, age_gridded[18,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(j) May 2008')

ax = plt.subplot(7,2,8)
plt.contourf(XI, YI, age_gridded[19,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(k) Sept 2009')

ax = plt.subplot(7,2,10)
plt.contourf(XI, YI, age_gridded[20,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(l) Oct 2010')

ax = plt.subplot(7,2,12)
plt.contourf(XI, YI, age_gridded[21,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(m) Jul 2011')

ax = plt.subplot(7,2,14)
im = plt.contourf(XI, YI, age_gridded[22,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.xlabel('Distance (km)', fontsize = 14)
plt.title('(n) Aug 2012')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('years', fontsize = 14)
plt.savefig('linew_age_obs.pdf')


clevs = np.arange(120, 320, 20)
fig = plt.figure(figsize=(7,11))

ax = plt.subplot(7,2,1)
plt.contourf(XI, YI, o2_gridded[9,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(a) Nov 2003')

ax = plt.subplot(7,2,3)
plt.contourf(XI, YI, o2_gridded[10,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(b) May 2004')

ax = plt.subplot(7,2,5)
plt.contourf(XI, YI, o2_gridded[11,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(c) Oct 2004')

ax = plt.subplot(7,2,7)
plt.contourf(XI, YI, o2_gridded[12,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.ylabel('Depth (dbars)', fontsize = 14)
plt.title('(d) May 2005')

ax = plt.subplot(7,2,9)
plt.contourf(XI, YI, o2_gridded[13,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(e) Oct 2005')

ax = plt.subplot(7,2,11)
plt.contourf(XI, YI, o2_gridded[14,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
plt.yticks([0, 2000, 4000])
plt.title('(f) May 2006')


ax = plt.subplot(7,2,13)
plt.contourf(XI, YI, o2_gridded[15,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
plt.yticks([0, 2000, 4000])
plt.xlabel('Distance (km)', fontsize = 14)
plt.title('(g) Oct 2006')

ax = plt.subplot(7,2,2)
plt.contourf(XI, YI, o2_gridded[16,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(h) May 2007')

ax = plt.subplot(7,2,4)
plt.contourf(XI, YI, o2_gridded[17,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(i) Oct 2007')

ax = plt.subplot(7,2,6)
plt.contourf(XI, YI, o2_gridded[18,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(j) May 2008')

ax = plt.subplot(7,2,8)
plt.contourf(XI, YI, o2_gridded[19,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(k) Sept 2009')

ax = plt.subplot(7,2,10)
plt.contourf(XI, YI, o2_gridded[20,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(l) Oct 2010')

ax = plt.subplot(7,2,12)
plt.contourf(XI, YI, o2_gridded[21,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('(m) Jul 2011')

ax = plt.subplot(7,2,14)
im = plt.contourf(XI, YI, o2_gridded[22,:,:], clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 5000])
ax.invert_yaxis()
ax.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.xlabel('Distance (km)', fontsize = 14)
plt.title('(n) Aug 2012')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('umol/kg', fontsize = 14)
plt.savefig('linew_o2_obs.pdf')

plt.show()
