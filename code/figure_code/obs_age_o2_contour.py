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

## Create Figure 4 from proposal using mean age instead of CFC12-age
linew_2003 = data[data.date=='2003nov']
linew_2012 = data[data.date=='2012aug']

# Interpolate data onto a grid:
dist = np.arange(0,600,10)
depth = np.arange(0,5000,100)
XI, YI = np.meshgrid(dist,depth)
new_grid = (XI, YI)

old_grid = retrieve_old_grid(linew_2003)
age_grid03   = griddata(old_grid, linew_2003.mean_age.values.flatten(), new_grid,
                      method='linear')
o2_grid03   = griddata(old_grid, linew_2003.oxygen.values.flatten(), new_grid,
                      method='linear')


old_grid = retrieve_old_grid(linew_2012)
age_grid12   = griddata(old_grid, linew_2012.mean_age.values.flatten(), new_grid,
                      method='linear')
o2_grid12   = griddata(old_grid, linew_2012.oxygen.values.flatten(), new_grid,
                      method='linear')

old_grid = retrieve_old_grid(linew_2012)
temp_grid12   = griddata(old_grid, linew_2012.temp.values.flatten(), new_grid,
                      method='linear')
salt_grid12   = griddata(old_grid, linew_2012.salt.values.flatten(), new_grid,
                      method='linear')


# Create Figure
plt.figure(figsize=(14,8))
ax = plt.subplot(2,2,1)
clevs = np.arange(120, 310, 10)
plt.contourf(XI, YI, o2_grid03, clevs, cmap = cmaps.viridis, extend = 'both')
cb = plt.colorbar()
#CS = plt.contour(XI, YI, sigma0_2003, colors = 'k', levels=[26.6, 27.0, 27.2, 27.6, 28.0])
#plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0, 2000])
ax.invert_yaxis()
cb.set_label('umol/kg', fontsize = 12)
plt.title('(a) Oxygen (Nov 2003)', fontsize = 14)
plt.ylabel('Pressure (dbars)', fontsize = 12)

ax = plt.subplot(2,2,2)
clevs = np.arange(0, 210, 10)
plt.contourf(XI, YI, age_grid03, clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 2000])
ax.invert_yaxis()
cb = plt.colorbar()
cb.set_label('years', fontsize = 12)
plt.title('(b) Mean Age (Nov 2003)', fontsize = 14)


ax = plt.subplot(2,2,3)
clevs = np.arange(120, 310, 10)
plt.contourf(XI, YI, o2_grid12, clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 2000])
ax.invert_yaxis()
cb = plt.colorbar()
cb.set_label('umol/kg', fontsize = 12)
plt.title('(c) Oxygen (Aug 2012)', fontsize = 14)
plt.xlabel('Distance (km)', fontsize = 12)
plt.ylabel('Pressure (dbars)', fontsize = 12)



ax = plt.subplot(2,2,4)
clevs = np.arange(0, 210, 10)
plt.contourf(XI, YI, age_grid12, clevs, cmap = cmaps.viridis, extend = 'both')
plt.ylim([0, 2000])
ax.invert_yaxis()
cb = plt.colorbar()
cb.set_label('years', fontsize = 12)
plt.title('(d) Mean Age (Aug 2012)', fontsize = 14)
plt.xlabel('Distance (km)', fontsize = 12)
plt.tight_layout()
plt.savefig('age_oxygen_figure1.png')



plt.show()
