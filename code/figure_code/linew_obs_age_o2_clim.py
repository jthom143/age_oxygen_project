### Script to create figure of line_w age and o2 for Observations

# Created November 16, 2017

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
#                               Functions                                      #
################################################################################
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

################################################################################
# Plot Observational Data
################################################################################

## Load Data
PATH = '/RESEARCH/chapter3/data/GO_SHIP/linew/'
allFiles = glob.glob(PATH + "/*.csv")
allDensityFiles = glob.glob(PATH + "processed/*.csv")
raw_data = pd.DataFrame()
list_ = []
i = 0
for file_ in allFiles:
    df = pd.read_csv(file_, header = 1, delim_whitespace = True,
                    na_values = '-9.000')
    df2 = pd.read_csv(allDensityFiles[i])
    df['neutral_dens'] = df2
    list_.append(df)
    i = i+1
raw_data = pd.concat(list_)

data = raw_data[['Age', 'LAT', 'LON', 'CTDPRS', 'CTDTMP', 'CTDSAL', 'OXYGEN',
                 'pCFC-12', 'CFC12age', 'Mean', 'neutral_dens']].copy()
data.columns = ['date', 'latitude', 'longitude', 'press', 'temp', 'salt',
                'oxygen', 'pCFC12', 'CFC12age', 'mean_age', 'neutral_dens']



# Calculate AOUdata['o2_sat'] = o2sat(data.salt, data.temp)
data['o2_sat'] = o2sat(data.salt, data.temp)
data['aou'] =   data.o2_sat - data.oxygen

# Change date column from string to DateTime
data['date'] = pd.to_datetime(data['date'], format='%Y%b')

## Calculate the distance from Cape Cod for all measurements
data['latitude'] = data['latitude'].astype(dtype=float)
data['longitude'] = data['longitude'].astype(dtype=float)
data['distance'] = calculate_distance_km(data.latitude, data.longitude)

mean_df = data.groupby(data['date'])['temp', 'salt', 'oxygen', 'mean_age', 'aou'].mean()
max_df =  data.groupby(data['date'])['distance'].max()
df = mean_df.join(max_df)
#mean_df.to_csv('output_means.csv')
#max_df.to_csv('output_max.csv')

## Trim the data to use only the years which have oxygen data
df_oxygen_no_nan = df[mean_df.oxygen.notnull()]
df_oxygen_no_nan = df_oxygen_no_nan[df_oxygen_no_nan['distance']>300]
years = df_oxygen_no_nan.index.values

## Grid The Age and Oxygen Data
# New Grid:
dist = np.arange(0,600,10)
depth = np.arange(0,5000,50)
XI, YI = np.meshgrid(dist,depth)
new_grid = (XI, YI)

# Create Array for Age and Oxygen Data Interpolated onto a grid:

o2_gridded  = np.ones((len(years), len(depth), len(dist))) * np.nan
age_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan
aou_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan
dens_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan

for i in range(0, len(years)):
    year = years[i]

    linew_year = data[data.date==year]
    old_grid = retrieve_old_grid(linew_year)
    age_grid = griddata(old_grid, linew_year.mean_age.values.flatten(), new_grid,
                        method='linear')
    o2_grid = griddata(old_grid, linew_year.oxygen.values.flatten(), new_grid,
                        method='linear')
    aou_grid = griddata(old_grid, linew_year.aou.values.flatten(), new_grid,
                        method='linear')
    dens_grid = griddata(old_grid, linew_year.neutral_dens.values.flatten(), new_grid,
                        method='linear')

    o2_gridded[i,:,:]  = o2_grid
    age_gridded[i,:,:] = age_grid
    aou_gridded[i,:,:] = aou_grid
    dens_gridded[i,:,:] = dens_grid

dens_mean = np.nanmean(dens_gridded,0)
age_mean = np.nanmean(age_gridded,0)
o2_mean = np.nanmean(o2_gridded,0)


fig = plt.figure(figsize = (12,4))
ax = plt.subplot(1,2,1)
clevs = np.arange(140, 301, 1)
im = plt.contourf(dist, depth, o2_mean, clevs, cmap = cmaps.viridis)
cb = plt.colorbar()
cb.set_label('umol/kg', fontsize = 12)
CS = plt.contour(dist[10:], depth, dens_mean[:,10:],
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
#plt.plot([20, 20, 250, 250, 20],[100, 600, 600, 100, 100], lw = 1.5, color = 'w')
#plt.plot([300, 300, 390, 390, 300],[650, 850, 850, 650, 650], lw = 1.5, color = 'w')
#plt.plot([200, 200, 400, 400, 200],[1500, 1900, 1900, 1500, 1500], lw = 1.5, color = 'w')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
ax.invert_yaxis()
plt.title('(a) Oxygen', fontsize = 13)
plt.ylabel('Depth (dbars)')
plt.xlabel('Distance (km)')

ax = plt.subplot(1,2,2)
clevs = np.arange(0, 121, 1)
im = plt.contourf(dist, depth, age_mean, clevs, cmap = cmaps.viridis)
#plt.plot([20, 20, 250, 250, 20],[100, 600, 600, 100, 100], lw = 1.5, color = 'w')
cb = plt.colorbar()
cb.set_label('years', fontsize = 12)
CS = plt.contour(dist[10:], depth, dens_mean[:,10:],
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.yticks([0, 50, 1000, 1500, 2000],[' ', ' ', ' ', ' ', ' '])
ax.invert_yaxis()
plt.title('(b) Age', fontsize = 13)
plt.xlabel('Distance (km)')

fig.subplots_adjust(bottom=0.15)


#plt.savefig('obs_age_o2_clim.png')



## Plot Timeseries

region1_age  = np.nanmean(np.nanmean(age_gridded[:,2:13, 2:26], 1) , 1)
region1_o2  = np.nanmean(np.nanmean(o2_gridded[:,2:13, 2:26], 1) , 1)

region2_age  = np.nanmean(np.nanmean(age_gridded[:,13:18, 30:40], 1) , 1)
region2_o2  = np.nanmean(np.nanmean(o2_gridded[:,13:18, 30:40], 1) , 1)

region3_age  = np.nanmean(np.nanmean(age_gridded[:,30:39, 20:41], 1) , 1)
region3_o2  = np.nanmean(np.nanmean(o2_gridded[:,30:39, 20:41], 1) , 1)

# Convert array to month units:
months = np.arange('2003-01', '2012-12', dtype='datetime64[M]')
region1_age_ts = np.ones((len(months)))*np.nan
region1_o2_ts = np.ones((len(months)))*np.nan
region2_age_ts = np.ones((len(months)))*np.nan
region2_o2_ts = np.ones((len(months)))*np.nan
region3_age_ts = np.ones((len(months)))*np.nan
region3_o2_ts = np.ones((len(months)))*np.nan

for t in range(0, len(years)):
    for i in range(0, len(months)):
        if years[t] == months [i]:
            region1_age_ts[i] = region1_age[t]
            region1_o2_ts[i]  = region1_o2[t]
            region2_age_ts[i] = region2_age[t]
            region2_o2_ts[i]  = region2_o2[t]
            region3_age_ts[i] = region3_age[t]
            region3_o2_ts[i]  = region3_o2[t]

# Mask neutral_dens
mask = np.isfinite(region1_age_ts)

# Create X array
x = np.arange(0, 119, 1)
xticks = np.arange(0, 119, 12)

fig = plt.figure(figsize = (12,6))
ax = plt.subplot(3,1,1)
plt.plot(x[mask], region1_age_ts[mask], ls = '-', marker = 'o', lw = 1.5)
ax2 = ax.twinx()
plt.plot(x[mask],region1_o2_ts[mask], ls = '-', marker = '*',lw = 1.5, color = 'r')
plt.xticks(xticks, [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' '])
ax.set_ylim([0,120])
ax2.set_ylim([140,285])

ax = plt.subplot(3,1,2)
plt.plot(x[mask],region2_age_ts[mask], ls = '-', marker = 'o', lw = 1.5)
ax2 = ax.twinx()
plt.plot(x[mask],region2_o2_ts[mask], ls = '-', marker = '*',lw = 1.5, color = 'r')
plt.xticks(xticks, [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',  ' '])
ax.set_ylim([0,120])
ax2.set_ylim([140,285])

ax = plt.subplot(3,1,3)
plt.plot(x[mask],region3_age_ts[mask], ls = '-', marker = 'o', lw = 1.5)
ax2 = ax.twinx()
plt.plot(x[mask],region3_o2_ts[mask], ls = '-', marker = '*', lw = 1.5, color = 'r')
ax.set_ylim([0,120])
ax2.set_ylim([140,285])

plt.xticks(xticks, ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',  '2012'])
plt.savefig('obs_age_o2_ts.png')



## Vertical Profile and Vertical Gradient:
age_mean_vert = np.nanmean(age_mean[:,30:41], 1)
o2_mean_vert = np.nanmean(o2_mean[:,30:41], 1)

age_mean_vert_gradient = np.gradient(age_mean_vert, 50)
o2_mean_vert_gradient = np.gradient(o2_mean_vert, 50)

plt.figure()
ax = plt.subplot(1,2,1)
plt.plot(age_mean_vert, depth, lw = 1.5)
ax.tick_params('x', colors='b')
ax.set_xlabel('Age (years)', color='b')
plt.ylabel('Depth (dbars)')
plt.xlim([0,150])
ax2 = ax.twiny()
plt.plot(o2_mean_vert, depth, color = 'r', lw = 1.5)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen (umol/kg)', color='r')
plt.xlim([175,280])
plt.ylim([0,2000])
ax.invert_yaxis()

ax = plt.subplot(1,2,2)
plt.plot(age_mean_vert_gradient, depth, lw = 1.5)
ax.tick_params('x', colors='b')
ax.set_xlabel('Age Gradient (years/dbar)', color='b')
plt.xlim([-0.8, 0.8])
ax2 = ax.twiny()
plt.plot(o2_mean_vert_gradient, depth, color = 'r', lw = 1.5)
ax2.tick_params('x', colors='r')
ax2.set_xlabel('Oxygen Gradient (umol/kg/dbar)', color='r')
ax2.set_xlim([-0.35, 0.35])
plt.ylim([0,2000])
ax.invert_yaxis()
plt.axvline(0, ls = '--', lw = 1.5, color = 'k')
plt.savefig('obs_vertical_profile_gradient.png')

plt.show()
