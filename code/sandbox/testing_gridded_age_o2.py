### Script to analyze observational data

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

from scipy.stats import t as ttest

import sys
sys.path.append('/RESEARCH/chapter3/functions')
import colormaps as cmaps
from o2sat import o2sat


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
    df = pd.read_csv(file_, header = 1, delim_whitespace = True,
                     na_values = '-9.000')
    list_.append(df)
raw_data = pd.concat(list_)

data = raw_data[['Age', 'LAT', 'LON', 'CTDPRS', 'CTDTMP', 'CTDSAL', 'OXYGEN',
                 'pCFC-12', 'CFC12age', 'Mean']].copy()
data.columns = ['date', 'latitude', 'longitude', 'press', 'temp', 'salt',
                'oxygen', 'pCFC12', 'CFC12age', 'mean_age']

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
depth = np.arange(0,5000,100)
XI, YI = np.meshgrid(dist,depth)
new_grid = (XI, YI)

# Create Array for Age and Oxygen Data Interpolated onto a grid:
o2_gridded  = np.ones((len(years), len(depth), len(dist))) * np.nan
age_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan
aou_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan

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

    o2_gridded[i,:,:]  = o2_grid
    age_gridded[i,:,:] = age_grid
    aou_gridded[i,:,:] = aou_grid


## Compare year gridded vs non-gridded at 500 dbars
data_500 = data[data['press']>490]
data_500 = data_500[data_500['press']<510]

data_plot = data_500[data_500.date=='2003-10-31T19:00:00.000000000-0500']

plt.plot(data_plot.distance, data_plot.oxygen, lw = 2)
plt.plot(dist, o2_gridded[0,5,:], lw = 2, color = 'r')


plt.figure()
data_plot = data[data.date=='2003-10-31T19:00:00.000000000-0500']

plt.subplot(1,2,1)
plt.scatter(age_gridded[0,:,:], o2_gridded[0,:,:])
plt.subplot(1,2,2)
plt.scatter(data_plot.mean_age, data_plot.oxygen)


# Calculate mean age and o2
age_gridded_mean = np.nanmean(age_gridded, 0)
o2_gridded_mean = np.nanmean(o2_gridded, 0)

# Calculate Correlation
corr = np.ones([len(depth), len(dist)]) * np.nan
o2_gridded = np.ma.masked_invalid(o2_gridded)
age_gridded = np.ma.masked_invalid(age_gridded)

for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = np.ma.corrcoef(o2_gridded[:,z,x], age_gridded[:,z,x])
        #print r
        corr[z,x] = r[0,1]

plt.figure()
plt.subplot(1,2,1)
plt.scatter(age_gridded_mean, o2_gridded_mean,
c = corr.flatten(), cmap = 'RdBu_r', s = 30, lw = 0.3, vmin=-1, vmax=1)
plt.xlim([-100, 500])
plt.subplot(1,2,2)
plt.scatter(data_plot.mean_age, data_plot.oxygen)


plt.show()
