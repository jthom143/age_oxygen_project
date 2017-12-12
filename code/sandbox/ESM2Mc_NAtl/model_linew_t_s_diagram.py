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

# Function to calculate distance from Cape Cod:
def calculate_distance_km(lat, lon):
    from math import sin, cos, sqrt, atan2, radians
    # Function to calculate distance from Cape Cod for line W data.
    # function calls for two Pandas data series (latitude and longitude)
    # approximate radius of earth in km
    R = 6373.0
    # convert series to array:
    #lat = lat.as_matrix()
    #lon = lon.as_matrix()

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

# Function to model variables to Line W
def interpolate_to_linew(cube, name):
    # Isolate Region of Interest to limit interpolation
    lat_constraint = iris.Constraint(latitude=lambda x: 25 <= x <= 50)
    lon_constraint = iris.Constraint(longitude=lambda y: -85 <= y <= -50)
    region = cube.extract(lat_constraint)
    region = region.extract(lon_constraint)

    # Interpolate onto 1deg x 1deg grid
    new_lats = np.arange(32, 41, 1)
    new_lons = np.linspace(-65, -69, num=9)

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


PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'

rho = iris.load_cube(PATH+'rho.nc')
temp = iris.load_cube(PATH+'temp.nc')
salt = iris.load_cube(PATH+'salt.nc')

neutral_rho = iris.load_cube(PATH+'neutral_rho.nc')

'''
## Interpolate model variables to Line W
temp_linew = interpolate_to_linew(temp, 'potential temperature')
salt_linew = interpolate_to_linew(salt, 'salinity')

iris.save(temp_linew, 'temp_linew.nc')
iris.save(salt_linew, 'salt_linew.nc')
'''

temp_linew = iris.load_cube('temp_linew.nc')
salt_linew = iris.load_cube('salt_linew.nc')
neutral_rho_linew = iris.load_cube('neutral_rho_linew.nc')
o2_linew = iris.load_cube('o2_linew.nc')
age_linew = iris.load_cube('age_linew.nc')

### Figure: T-S Diagram
# Figure out boudaries (mins and maxs)
smin = np.nanmin(salt_linew.data) - (0.01 * np.nanmin(salt_linew.data))
smax = np.nanmax(salt_linew.data) + (0.01 * np.nanmax(salt_linew.data))
tmin = np.nanmin(temp_linew.data) - (0.01 * np.nanmin(temp_linew.data))
tmax = np.nanmax(temp_linew.data) + (0.01 * np.nanmax(temp_linew.data))

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


salt_linew_clim = salt_linew.collapsed('time', iris.analysis.MEAN)
temp_linew_clim = temp_linew.collapsed('time', iris.analysis.MEAN)
o2_linew_clim = o2_linew.collapsed('time', iris.analysis.MEAN)
age_linew_clim = age_linew.collapsed('time', iris.analysis.MEAN)

t = age_linew_clim.data.flatten()
plt.scatter(salt_linew_clim.data.flatten(), temp_linew_clim.data.flatten(), c = t, cmap = cmaps.viridis, lw = 0, s = 35, vmin = 0, vmax = 150)
cb = plt.colorbar()
CS = plt.contour(si,ti,dens, linestyles='dashed', colors='k')
plt.clabel(CS, fontsize=12, inline=1, fmt='%1.1f') # Label every second level
plt.xlim([round(smin,1), round(smax,1)])
plt.ylim([round(tmin,1), round(tmax,1)])
cb.set_label('Age (years)')
plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (C)')
plt.title('ESM2Mc Line-W T-S Diagram')
plt.savefig('ESM2Mc_linew_ts_age_diagram.png')
plt.show()
'''
#plt.plot([35.1, 36.7, 36.7, 35.1, 35.1],[8, 8, 19, 19, 8], color = 'r')
#plt.text(35.12, 19.2, 'North Atlantic Central Water', color = 'r')

plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (C)')
plt.title('ESM2Mc Line-W T-S Diagram')
plt.savefig('ESM2Mc_linew_ts_diagram.png')

#plt.figure(figsize=(12,8))
#for i in range(0,2):
#    plt.scatter(salt_linew[i,:,0:4].data, temp_linew[i,:,0:4].data,
#                 s = 30, lw = 0.3, vmin=-1, vmax=1)
#plt.title('T-S Diagram', fontsize = 14)
#plt.ylabel('Oxygen [umol/kg]', fontsize = 14)
#plt.xlabel('Salinity (psu)')
#plt.ylabel('Temperature (C)')




## Entire North Atlantic
temp_clim = temp.collapsed('time', iris.analysis.MEAN)
salt_clim = salt.collapsed('time', iris.analysis.MEAN)

## Restrict Domain to North Atlantic Ocean
constraint = iris.Constraint(latitude=lambda y: 0 < y < 90)
temp_natl = temp_clim.extract(constraint)
salt_natl = salt_clim.extract(constraint)

constraint = iris.Constraint(longitude=lambda x: -80 < x < 5)
temp_natl = temp_natl.extract(constraint)
salt_natl = salt_natl.extract(constraint)


# Figure out boudaries (mins and maxs)
smin = 19
smax = 41
tmin = -5
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
plt.scatter(salt_natl.data.flatten(), temp_natl.data.flatten())
CS = plt.contour(si,ti,dens, linestyles='dashed', colors='k')
plt.clabel(CS, fontsize=12, inline=1, fmt='%1.1f') # Label every second level
plt.xlim([round(smin-1,1), round(smax+1,1)])
plt.ylim([round(tmin+1,1), round(tmax-1,1)])
plt.scatter(salt_linew_clim.data.flatten(), temp_linew_clim.data.flatten(), color = 'r')
plt.ylim([-5, 30])


plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (C)')
plt.title('ESM2Mc North Atlantic T-S Diagram')
plt.savefig('ESM2Mc_natl_ts_diagram.png')




from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


lons = temp_natl.coord('longitude').points
lats = temp_natl.coord('latitude').points
depth = temp_natl.coord('tcell pstar').points

plt.figure()
norm = MidpointNormalize(midpoint=500)

for x in range(0,len(lons)):
    for y in range(0,len(lats)):
        t = depth
        plt.scatter(salt_natl[:,y,x].data, temp_natl[:,y,x].data, c=t, norm = norm, cmap = cmaps.viridis,
                    s = 30, lw = 0.3)
plt.colorbar()


plt.figure()
norm = MidpointNormalize(midpoint=500)
t = depth
for tt in range(0,500):
    for d in range(0,9):
        plt.scatter(salt_linew[tt,:,d].data, temp_linew[tt,:,d].data, c=t, norm = norm, cmap = cmaps.viridis,
                    s = 30, lw = 0.3)
plt.colorbar()
plt.show()
'''
