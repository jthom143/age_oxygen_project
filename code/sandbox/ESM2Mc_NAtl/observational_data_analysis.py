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


# Plot age and Oxygen
fig, axes2d = plt.subplots(nrows=6, ncols=2,
                   sharex=True, sharey=True,figsize = (8,10))
clevs = np.arange(50, 160, 10)
n = 0
for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        im = cell.contourf(dist, depth, age_gridded[n,:,:], clevs, cmap = cmaps.viridis)
        plt.ylim([0,2000])
        cell.invert_yaxis()
        string = str(years[n])[:7]
        cell.set_title(string)
        n = n+1

plt.tight_layout()
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(left=0.13)
fig.subplots_adjust(bottom=0.06)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('years', fontsize = 14)

fig.text(0.47, 0.01, 'Distance (km)', ha='center', fontsize = 13)
fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical', fontsize = 13)

plt.savefig('age_contour.png')

## Oxygen
fig, axes2d = plt.subplots(nrows=6, ncols=2,
                   sharex=True, sharey=True,figsize = (8,10))
clevs = np.arange(120, 320, 20)
n = 0
for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        im = cell.contourf(dist, depth, o2_gridded[n,:,:], clevs, cmap = cmaps.viridis)
        plt.ylim([0,2000])
        cell.invert_yaxis()
        string = str(years[n])[:7]
        cell.set_title(string)
        n = n+1

plt.tight_layout()
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(left=0.13)
fig.subplots_adjust(bottom=0.06)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('umol/kg', fontsize = 14)

fig.text(0.47, 0.01, 'Distance (km)', ha='center', fontsize = 13)
fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical', fontsize = 13)

plt.savefig('o2_contour.png')


## Apparent Oxygen Utilization
fig, axes2d = plt.subplots(nrows=6, ncols=2,
                   sharex=True, sharey=True,figsize = (8,10))
clevs = np.arange(0, 180, 20)
n = 0
for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        im = cell.contourf(dist, depth, aou_gridded[n,:,:], clevs, cmap = cmaps.viridis)
        plt.ylim([0,2000])
        cell.invert_yaxis()
        string = str(years[n])[:7]
        cell.set_title(string)
        n = n+1

plt.tight_layout()
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(left=0.13)
fig.subplots_adjust(bottom=0.06)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('umol/kg', fontsize = 14)

fig.text(0.47, 0.01, 'Distance (km)', ha='center', fontsize = 13)
fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical', fontsize = 13)

plt.savefig('aou_contour.png')


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
fig = plt.figure()
levels = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ax = plt.subplot(1,1,1)
CS3 = plt.contourf(dist, depth, test, levels, cmap = 'YlGnBu')
CS3.cmap.set_under('white')
plt.ylim([0,2000])
ax.invert_yaxis()
plt.colorbar()
plt.title('Number of Observations')

plt.figure()
ax = plt.subplot(1,1,1)
clevs = np.arange(-1, 1.1, 0.1)
plt.contourf(dist, depth, corr, clevs, cmap = 'RdBu_r')
plt.ylim([0,2000])
ax.invert_yaxis()
plt.colorbar()
plt.title('Age-Oxygen Correlation')
plt.ylabel('Depth (m)')
plt.xlabel('Distance (km)')

t = np.ones([len(depth), len(dist)]) * np.nan
t_star = np.ones([len(depth), len(dist)]) * np.nan
corr_sig = np.ones([len(depth), len(dist)]) * np.nan
for x in range(0, len(dist)):
    for z in range(0, len(depth)):
        r = corr[z,x]
        n = test[z,x]

        t[z,x] = r*sqrt((n-2)/(1-r**2))

        t_star[z,x] = ttest.ppf(0.1, n-2) * -1

        if t[z,x] > t_star[z,x]:
            corr_sig[z,x] = r


plt.figure()
ax = plt.subplot(1,1,1)
levels = [0, 1, 2, 3, 4, 5, 6]
CS3 = plt.contourf(dist, depth, t, levels, cmap = 'YlGnBu')
CS3.cmap.set_under('white')
plt.ylim([0,2000])
ax.invert_yaxis()
plt.colorbar()
plt.title('t-stastic')
plt.ylabel('Depth (m)')
plt.xlabel('Distance (km)')

plt.figure()
ax = plt.subplot(1,1,1)
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(dist, depth, corr, clevs, cmap = 'RdBu_r')
cs = plt.contourf(dist, depth, corr_sig, 1, colors='none',
                  hatches=['..'])
plt.ylim([0,2000])
ax.invert_yaxis()
plt.colorbar(im)
plt.title('Age-Oxygen Correlation')
plt.ylabel('Depth (m)')
plt.xlabel('Distance (km)')
plt.savefig('obs_age_o2_corr.png')


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

        t_star[z,x] = ttest.ppf(0.1, n-2) * -1

        if t[z,x] > t_star[z,x]:
            corr_sig_aou[z,x] = r


plt.figure()
ax = plt.subplot(1,1,1)
clevs = np.arange(-1, 1.1, 0.1)
im = plt.contourf(dist, depth, corr_aou, clevs, cmap = 'RdBu_r')
cs = plt.contourf(dist, depth, corr_sig_aou, 1, colors='none',
                  hatches=['..'])
plt.ylim([0,2000])
ax.invert_yaxis()
plt.colorbar(im)
plt.title('Age-AOU Correlation')
plt.ylabel('Depth (m)')
plt.xlabel('Distance (km)')
plt.savefig('obs_age_aou_corr.png')



#### Figure Age vs Oxygen
plt.figure()
for i in range(0, len(years)):
    year = years[i]
    plt.scatter(data[data.date==year].mean_age, data[data.date==year].oxygen)
    plt.scatter(data[data.press<1000].mean_age, data[data.press<1000].oxygen)

plt.ylim([120, 320])
plt.xlim([-100, 500])
plt.ylabel('Age (years)')
plt.xlabel('Oxygen (umol/kg)')

plt.figure()
data_plot = data[data.date=='2003-10-31T19:00:00.000000000-0500']
plt.scatter(data_plot.mean_age, data_plot.oxygen)
#plt.scatter(data_plot[data_plot.press<4000].mean_age, data_plot[data_plot.press<4000].oxygen, color = 'b')
#plt.scatter(data_plot[data_plot.press<3000].mean_age, data_plot[data_plot.press<3000].oxygen, color = 'r')
#plt.scatter(data_plot[data_plot.press<2000].mean_age, data_plot[data_plot.press<2000].oxygen, color = 'g')
#plt.scatter(data_plot[data_plot.press<1000].mean_age, data_plot[data_plot.press<1000].oxygen, color = 'k')
#plt.scatter(data_plot[data_plot.press<500].mean_age, data_plot[data_plot.press<500].oxygen, color = 'c')
#plt.scatter(data_plot[data_plot.press<100].mean_age, data_plot[data_plot.press<100].oxygen, color = 'orange')
plt.text(-65, 175, '100 m', fontsize = 14)
plt.text(121, 174, '500 m', fontsize = 14)
plt.text(92, 243, '1000 m', fontsize = 14)
plt.text(100, 281, '2000 m', fontsize = 14)
#plt.text(178, 290, '3000 m', fontsize = 14)

plt.plot(17, 177, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(93, 178, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(68, 248, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(100, 267, marker = 'o', ls = '  ', color = 'k', ms = 8)

plt.ylim([120, 300])
plt.xlim([-100, 500])
plt.ylabel('Age (years)')
plt.xlabel('Oxygen (umol/kg)')
plt.savefig('linew_age_o2_plot.png')

#### Figure Age vs AOU
plt.figure()
data_plot = data[data.date=='2003-10-31T19:00:00.000000000-0500']
plt.scatter(data_plot.mean_age, data_plot.aou)

plt.text(-56, 32.5, '100 m', fontsize = 14)
plt.text(143, 122, '500 m', fontsize = 14)
plt.text(102, 72, '1000 m', fontsize = 14)
plt.text(92, 38, '2000 m', fontsize = 14)

plt.plot(4,  32.5, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(104, 122, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(64,   72, marker = 'o', ls = '  ', color = 'k', ms = 8)
plt.plot(92,   49, marker = 'o', ls = '  ', color = 'k', ms = 8)

#plt.ylim([120, 320])
#plt.xlim([-100, 500])
plt.ylabel('Age (years)')
plt.xlabel('AOU (umol/kg)')
plt.savefig('linew_age_aou_scatter.png')



### Figure: T-S Diagram
# Figure out boudaries (mins and maxs)
smin = 31
smax = 38
tmin = 0
tmax = 30

# Calculate how many gridcells we need in the x and y dimensions
xdim = round((smax-smin)/0.1+1,0)
ydim = round((tmax-tmin)+1,0)
# Create empty grid of zeros
dens = np.zeros((ydim,xdim))

# Create temp and salt vectors of appropiate dimensions
ti = np.linspace(1,ydim-1,ydim)+tmin
si = np.linspace(1,xdim-1,xdim)*0.1+smin

# Loop to fill in grid with densities
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        dens[j,i]=gsw.rho(si[i],ti[j],0)

# Substract 1000 to convert to sigma-t
dens = dens - 1000
plt.figure()
plt.scatter(data.salt, data.temp)
CS = plt.contour(si,ti,dens, linestyles='dashed', colors='k')
plt.clabel(CS, fontsize=12, inline=1, fmt='%1.1f') # Label every second level
plt.xlim([round(smin,1), round(smax,1)])
plt.ylim([round(tmin,1), round(tmax,1)])

plt.plot([35.1, 36.7, 36.7, 35.1, 35.1],[8, 8, 19, 19, 8], color = 'r')
plt.text(35.12, 19.2, 'North Atlantic Central Water', color = 'r')

plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (C)')
plt.title('Obs Line-W T-S Diagram')
plt.savefig('obs_linew_ts_diagram.png')


### Figure: T-S Diagram Number 2 - Made to match Le Bras et al
# Figure out boudaries (mins and maxs)
smin = 34.75
smax = 35.05
tmin = 2
tmax = 5

# Calculate how many gridcells we need in the x and y dimensions
xdim = round((smax-smin)/0.1+1,0)
ydim = round((tmax-tmin)+1,0)
# Create empty grid of zeros
dens = np.zeros((ydim,xdim))

# Create temp and salt vectors of appropiate dimensions
ti = np.linspace(tmin,tmax,ydim)
si = np.linspace(smin,smax,xdim)

# Loop to fill in grid with densities
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        dens[j,i]=gsw.rho(si[i],ti[j],0)

# Substract 1000 to convert to sigma-t
dens = dens - 1000
plt.figure()
plt.scatter(data.salt, data.temp)
plt.xlim([smin,smax])
plt.ylim([tmin,tmax])
CS = plt.contour(si,ti,dens, linestyles='dashed', colors='k')
plt.clabel(CS, CS.levels, inline=True, fontsize=10)


plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (C)')
plt.title('Obs Line-W T-S Diagram - Deep Ocean')
plt.savefig('obs_linew_ts_diagram_deep.png')



### Figure: T-S Diagram Number 3
# Figure out boudaries (mins and maxs)
smin = 34.75
smax = 36.95
tmin = 2
tmax = 30

# Calculate how many gridcells we need in the x and y dimensions
xdim = round((smax-smin)/0.1+1,0)
ydim = round((tmax-tmin)+1,0)
# Create empty grid of zeros
dens = np.zeros((ydim,xdim))

# Create temp and salt vectors of appropiate dimensions
ti = np.linspace(tmin,tmax,ydim)
si = np.linspace(smin,smax,xdim)

# Loop to fill in grid with densities
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        dens[j,i]=gsw.rho(si[i],ti[j],0)

# Substract 1000 to convert to sigma-t
dens = dens - 1000
plt.figure()
plt.scatter(data.salt, data.temp)
plt.xlim([smin,smax])
plt.ylim([tmin,tmax])
CS = plt.contour(si,ti,dens, linestyles='dashed', colors='k')
plt.clabel(CS, CS.levels, inline=True, fontsize=10)
#plt.plot([35.1, 36.7, 36.7, 35.1, 35.1],[8, 8, 19, 19, 8], color = 'r')
#plt.text(35.12, 19.2, 'North Atlantic Central Water', color = 'r')

plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (C)')
plt.title('Obs Line-W T-S Diagram')
plt.savefig('obs_linew_ts_diagram.png')

plt.show()
