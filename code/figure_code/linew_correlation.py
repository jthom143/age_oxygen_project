### Script to create figure of line_w age and o2/aou correlation for Observations
### and model data.

# Created Sept 28, 2017

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
def interpolate_to_linew(cube, name):
    # Isolate Region of Interest to limit interpolation
    lat_constraint = iris.Constraint(latitude=lambda x: 25 <= x <= 50)
    lon_constraint = iris.Constraint(longitude=lambda y: -85 <= y <= -50)
    region = cube.extract(lat_constraint)
    region = region.extract(lon_constraint)

    # Interpolate onto 1deg x 1deg grid
    new_lats = np.arange(22, 41, 1)
    new_lons = np.linspace(-60, -69, num=19)

    latitude = iris.coords.DimCoord(new_lats, standard_name='latitude', units='degrees')
    longitude = iris.coords.DimCoord(new_lons, standard_name='longitude', units='degrees')
    new_cube = iris.cube.Cube(np.zeros((28, len(new_lats), len(new_lons)), np.float32),
                              dim_coords_and_dims=[(region.coord('tcell pstar'), 0),
                                                   (latitude, 1), (longitude, 2)])

    # Regrid
    regrid = region.regrid(new_cube, iris.analysis.Linear())

    # Isolate relevant lat-lon pairs
    new = np.ones((500, 28, len(new_lats))) * np.nan

    for t in range(0,len(regrid.coord('time').points)):
        for k in range(0,len(regrid.coord('tcell pstar').points)):
            for i in range(0,len(new_lats)):
                new[t,k,i] = regrid[t,k,i,i].data

    dist = calculate_distance_km(new_lats, new_lons)
    distance = iris.coords.DimCoord(dist, long_name='distance', units='km')
    model_linew = iris.cube.Cube(new, long_name=name, units=cube.units,
                                 dim_coords_and_dims=[(region.coord('time'), 0),
                                                      (region.coord('tcell pstar'), 1),
                                                      (distance, 2)])
    return model_linew



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


# Calculate Correlations
corr = np.ones([len(depth), len(dist)]) * np.nan
o2_gridded = np.ma.masked_invalid(o2_gridded)
age_gridded = np.ma.masked_invalid(age_gridded)

for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = np.ma.corrcoef(o2_gridded[:,z,x], age_gridded[:,z,x])
        #print r
        corr[z,x] = r[0,1]

# Count the non-masked values
n_o2 = o2_gridded.count(axis=0)
n_age = age_gridded.count(axis=0)

new = age_gridded*o2_gridded
test = new.count(axis=0)

t = np.ones([len(depth), len(dist)]) * np.nan
t_star = np.ones([len(depth), len(dist)]) * np.nan
corr_sig = np.ones([len(depth), len(dist)]) * np.nan
for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = corr[z,x]
        n = test[z,x]

        t[z,x] = r*sqrt((n-2)/(1-r**2))

        t_star[z,x] = ttest.ppf(0.05, n-2) * -1

        if abs(t[z,x]) > t_star[z,x]:
            corr_sig[z,x] = r


fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,2,1)
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(dist, depth, corr, clevs, cmap = 'RdBu_r')
#cs = plt.contourf(dist, depth, corr_sig, 1, colors='none',
#                  hatches=['..'])
CS = plt.contour(dist[10:], depth, dens_mean[:,10:],
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
#cs = plt.contourf(dist, depth, corr_sig_o2_model, 1, colors='none',
                  #hatches=['..'])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,600])
ax.invert_yaxis()
plt.title('(a) Age vs Oxygen', fontsize = 13)
plt.ylabel('Depth (dbars)')
#plt.savefig('obs_age_o2_corr.png')
plt.xticks([0, 100, 200, 300, 400, 500, 600])
plt.xlabel('Distance (km)')


## Age vs AOU Correlation
# Calculate Correlations
corr_aou = np.ones([len(depth), len(dist)]) * np.nan
aou_gridded = np.ma.masked_invalid(aou_gridded)
age_gridded = np.ma.masked_invalid(age_gridded)

for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = np.ma.corrcoef(aou_gridded[:,z,x], age_gridded[:,z,x])
        #print r
        corr_aou[z,x] = r[0,1]


new = age_gridded*aou_gridded
test = new.count(axis=0)
t = np.ones([len(depth), len(dist)]) * np.nan
t_star = np.ones([len(depth), len(dist)]) * np.nan
corr_sig_aou = np.ones([len(depth), len(dist)]) * np.nan
for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = corr_aou[z,x]
        n = test[z,x]

        t[z,x] = r*sqrt((n-2)/(1-r**2))

        t_star[z,x] = ttest.ppf(0.05, n-2) * -1

        if abs(t[z,x]) > t_star[z,x]:
            corr_sig_aou[z,x] = r


ax = plt.subplot(1,2,2)
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(dist, depth, corr_aou, clevs, cmap = 'RdBu_r')
#cs = plt.contourf(dist, depth, corr_sig_aou, 1, colors='none',
                  #hatches=['..'])
CS = plt.contour(dist[10:], depth, dens_mean[:,10:],
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
#cs = plt.contourf(dist, depth, corr_sig_o2_model, 1, colors='none',
                  #hatches=['..'])
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,600])
plt.yticks([0, 50, 1000, 1500, 2000],[' ', ' ', ' ', ' ', ' '])
ax.invert_yaxis()
plt.title('(b) Age vs AOU', fontsize = 13)
plt.xticks([0, 100, 200, 300, 400, 500, 600])
plt.xlabel('Distance (km)')

fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
cb = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cb.set_label('Correlation Coefficient', fontsize = 12)
plt.savefig('obs_correlation.png')



################################################################################
# Scatter Plots
################################################################################

## Grid the Age, Oxygen, and AOU Data
# New Grid:
dist = np.arange(0,600,10)
depth = np.arange(0,5000,50)
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

# Calculate Climatologies
age_gridded_mean = np.nanmean(age_gridded, 0)
o2_gridded_mean = np.nanmean(o2_gridded, 0)
aou_gridded_mean = np.nanmean(aou_gridded, 0)

## Calculate Correlation between Age and O2/AOU

# Calculate Correlation
corr = np.ones([len(depth), len(dist)]) * np.nan
o2_gridded = np.ma.masked_invalid(o2_gridded)
age_gridded = np.ma.masked_invalid(age_gridded)

for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = np.ma.corrcoef(o2_gridded[:,z,x], age_gridded[:,z,x])
        #print r
        corr[z,x] = r[0,1]

corr_aou = np.ones([len(depth), len(dist)]) * np.nan
aou_gridded = np.ma.masked_invalid(aou_gridded)
age_gridded = np.ma.masked_invalid(age_gridded)

for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = np.ma.corrcoef(aou_gridded[:,z,x], age_gridded[:,z,x])
        #print r
        corr_aou[z,x] = r[0,1]






fig = plt.figure(figsize = (10,5))
ax1 = plt.subplot(1,2,1)
plt.scatter(age_gridded_mean[:,30:41], o2_gridded_mean[:,30:41],
            c = corr[:,30:41].flatten(), cmap = 'RdBu_r', s = 30, lw = 0.3, vmin=-1, vmax=1)
plt.xlim([-25, 275])
#plt.xlabel('Age (years)')
plt.ylabel('Oxygen (umol/kg)')
plt.xlabel('Age (years)')
plt.title('(a) Age vs Oxygen', fontsize = 13)

## Add Linear Lines
x = np.arange(0, 276, 1)
y = np.arange(0, 121, 1)


ax2 = plt.subplot(1,2,2)
plt.scatter(age_gridded_mean[:,30:41], aou_gridded_mean[:,30:41],
            c = corr_aou[:,30:41].flatten(), cmap = 'RdBu_r', s = 30, lw = 0.3, vmin=-1, vmax=1)
plt.plot(x, 0.4*x, color = 'grey', ls = '--')
plt.plot(x, 0.7*x, color = 'grey', ls = '--')
plt.plot(x, 1.7*x, color = 'grey', ls = '--')
plt.plot(x, 100*x, color = 'grey', ls = '--')
plt.plot(x, 0.2*x, color = 'grey', ls = '--')
plt.plot(x, 0*x, color = 'grey', ls = '--')


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
plt.savefig('obs_scatterplot.png')



plt.show()
