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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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

temp_gridded  = np.ones((len(years), len(depth), len(dist))) * np.nan
salt_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan
dens_gridded = np.ones((len(years), len(depth), len(dist))) * np.nan

for i in range(0, len(years)):
    year = years[i]

    linew_year = data[data.date==year]
    old_grid = retrieve_old_grid(linew_year)
    temp_grid = griddata(old_grid, linew_year.temp.values.flatten(), new_grid,
                        method='linear')
    salt_grid = griddata(old_grid, linew_year.salt.values.flatten(), new_grid,
                        method='linear')
    dens_grid = griddata(old_grid, linew_year.neutral_dens.values.flatten(), new_grid,
                        method='linear')


    temp_gridded[i,:,:]  = temp_grid
    salt_gridded[i,:,:] = salt_grid
    dens_gridded[i,:,:] = dens_grid

dens_mean = np.nanmean(dens_gridded,0)

# Mask NaN na_values
temp_gridded = np.ma.masked_invalid(temp_gridded)
salt_gridded = np.ma.masked_invalid(salt_gridded)
dens_gridded = np.ma.masked_invalid(dens_gridded)

# Average over time:
temp_gridded_mean = temp_gridded.mean(axis = 0)
salt_gridded_mean = salt_gridded.mean(axis = 0)
dens_gridded_mean = dens_gridded.mean(axis = 0)


# Create Figure
# Load Model Data
PATH = '/RESEARCH/chapter3/data/GO_SHIP/linew/'
fname = 'section12.dat.csv'
raw_data = pd.read_csv(PATH+fname, header = 1, delim_whitespace = True, na_values = '-9.000')
data = raw_data[['LAT', 'LON']].copy()
data.columns = ['latitude', 'longitude']

a = data.latitude.values
indexes = np.unique(a, return_index=True)[1]
linew_lats = [a[index] for index in sorted(indexes)]

b = data.longitude.values
indexes = np.unique(b, return_index=True)[1]
linew_lons = [b[index] for index in sorted(indexes)]

new_lats = np.arange(32, 41, 1)
new_lons = np.linspace(-65, -69, num=9)



plt.figure(figsize=(18,10))
# Plot transects
ax = plt.subplot(1,3,1, projection=ccrs.PlateCarree())
ax.set_extent((-85, -63, 20, 45), crs=ccrs.PlateCarree())
plt.plot(linew_lons, linew_lats, marker = '*', transform=ccrs.PlateCarree(), label = 'Observational Line W')
plt.plot(new_lons, new_lats, color = 'r', marker = '*', transform=ccrs.PlateCarree(), label = 'Model Interpolation')
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.05),  mode="expand", borderaxespad=0.)
ax.stock_img()
ax.coastlines()


ax = plt.subplot(2,3,2)
clevs_temp = np.arange(0, 32, 4)
plt.contourf(XI, YI, temp_gridded_mean, clevs_temp, cmap = cmaps.viridis)
cb = plt.colorbar()
CS = plt.contour(dist[5:], depth, dens_gridded_mean[:,5:],
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0, 2000])
plt.xlim([0, 700])
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700],[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
ax.invert_yaxis()
cb.set_label('Degrees C', fontsize = 12)
plt.title('(a) Line W Obs Temperature', fontsize = 12)
plt.ylabel('Depth (dbars)', fontsize = 12)

ax = plt.subplot(2,3,3)
clevs_salt = np.arange(34.4, 37.2, 0.4)
plt.contourf(XI, YI, salt_gridded_mean, clevs_salt, cmap = cmaps.viridis)
plt.ylim([0, 2000])
plt.xlim([0, 700])
plt.yticks([0, 50, 1000, 1500, 2000],[' ', ' ', ' ', ' ', ' '])
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700],[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
ax.invert_yaxis()
cb = plt.colorbar()
CS = plt.contour(dist[5:], depth, dens_gridded_mean[:,5:],
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
cb.set_label('PSU', fontsize = 12)
plt.title('(b) Line W Obs Salinity', fontsize = 12)


################################################################################
# Plot Model Data
################################################################################



################################################################################
# Plot Model Data
################################################################################


### Load Line W Data
PATH = '/RESEARCH/chapter3/data/newCO2_control_800/derived/linew/'
temp_linew = iris.load_cube(PATH+'temp_linew.nc')
salt_linew = iris.load_cube(PATH+'salt_linew.nc')
neutral_rho_linew = iris.load_cube(PATH+'neutral_rho_linew.nc')

### Line W Calculations
# Calculate Climatology of Neutral Density
neutral_rho_linew_mean = neutral_rho_linew.collapsed('time', iris.analysis.MEAN)
temp_clim = temp_linew.collapsed('time', iris.analysis.MEAN)
salt_clim = salt_linew.collapsed('time', iris.analysis.MEAN)


######## Figures ##########
## Plot Temp and Salinity on Line W
ax = plt.subplot(2,3,5)
plt.contourf(temp_linew.coord('distance').points, temp_linew.coord('tcell pstar').points, temp_clim.data, clevs_temp, cmap = cmaps.viridis)
cb = plt.colorbar()
CS = plt.contour(temp_linew.coord('distance').points, temp_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')
plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,700])
ax.invert_yaxis()
plt.title('(c) Line W Model Temperature', fontsize = 12)
plt.ylabel('Depth (dbars)')
cb.set_label('Degrees C', fontsize = 12)
plt.xlabel('Distance (km)')

ax = plt.subplot(2,3,6)
plt.contourf(temp_linew.coord('distance').points, temp_linew.coord('tcell pstar').points, salt_clim.data, clevs_salt, cmap = cmaps.viridis)
cb = plt.colorbar()
CS = plt.contour(temp_linew.coord('distance').points, temp_linew.coord('tcell pstar').points, neutral_rho_linew_mean.data-1000,
                 levels = [26.0, 26.5, 27.0, 27.5, 28.0], colors = 'k')

plt.clabel(CS, fontsize=9, inline=1)
plt.ylim([0,2000])
plt.xlim([0,700])
plt.yticks([0, 50, 1000, 1500, 2000],[' ', ' ', ' ', ' ', ' '])
ax.invert_yaxis()
plt.title('(c) Line W Model Salinity', fontsize = 12)
plt.xlabel('Distance (km)')
cb.set_label('PSU', fontsize = 12)
plt.savefig('linew_interpolation.png')


plt.show()




plt.show()
