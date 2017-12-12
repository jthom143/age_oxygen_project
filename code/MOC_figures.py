## Recreate MOC Figures from NOAA proposal

# Load Packages
import numpy as np
import iris
import iris.analysis.cartography
import matplotlib.pyplot as plt
import iris.analysis.stats
import cartopy.crs as ccrs

MOC = iris.load_cube('/RESEARCH/chapter3/data/newCO2_control_800/derived/MOC.nc')

constraint = iris.Constraint(latitude = lambda x: 28 <= x <=31)
MOC_30 = MOC.extract(constraint)

moc = MOC_30.collapsed('tcell pstar', iris.analysis.MAX)

plt.plot(moc.data[0:100], lw = 2)
plt.ylabel('Transport [Sv]', fontsize = 14)
plt.xlabel('Time [years]', fontsize = 14)
plt.title('AMOC Strength at 30N', fontsize = 14)
plt.ylim([15, 25])


o2 = iris.load_cube('/RESEARCH/chapter3/data/newCO2_control_800/o2.nc')
o2 = o2[-500:]

age = iris.load_cube('/RESEARCH/chapter3/data/newCO2_control_800/residency_age_surface.nc')
age.coord('Time').rename('time')

# Isolate age and oxygen at 900m depth
o2_900 = o2[:,16,:,:]
age_900 = age[:,16,:,:]


#corr_o2 = np.ones((len(o2.coord('latitude').points),
#                   len(o2.coord('longitude').points))) * np.nan
#for x in range(0,len(o2.coord('longitude').points)):
#    for y in range(0, len(o2.coord('latitude').points)):
#        a = iris.analysis.stats.pearsonr(moc, o2_900[:,y,x])
#        corr_o2[y, x] = a.data
#np.save('/RESEARCH/chapter3/data/newCO2_control_800/o2_moc_corr', corr_o2)
corr_o2 = np.load('/RESEARCH/chapter3/data/newCO2_control_800/o2_moc_corr.npy')

#corr_age = np.ones((len(age.coord('latitude').points),
#                   len(age.coord('longitude').points))) * np.nan
#for x in range(0,len(age.coord('longitude').points)):
#    for y in range(0, len(age.coord('latitude').points)):
#        a = iris.analysis.stats.pearsonr(moc, age_900[:,y,x])
#        corr_age[y, x] = a.data
#np.save('/RESEARCH/chapter3/data/newCO2_control_800/age_moc_corr', corr_age)
corr_age = np.load('/RESEARCH/chapter3/data/newCO2_control_800/age_moc_corr.npy')



plt.figure()
clevs = np.arange(-1, 1.1, 0.1)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent((-80, 0, 0, 60), crs=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(o2.coord('longitude').points, o2.coord('latitude').points, corr_o2,
             clevs, cmap = 'RdBu_r')
plt.title('(a) Correlation between MOC and O2 at 900m')
plt.colorbar()

plt.figure()
clevs = np.arange(-1, 1.1, 0.1)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent((-80, 0, 0, 60), crs=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(o2.coord('longitude').points, o2.coord('latitude').points, corr_age,
             clevs, cmap = 'RdBu_r')
plt.colorbar()
plt.title('(b) Correlation between MOC and age at 900m')


# Correlation along line w
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

def interpolate_to_linew(cube, name):
    # Isolate Region of Interest to limit interpolation
    lat_constraint = iris.Constraint(latitude=lambda x: 25 <= x <= 50)
    lon_constraint = iris.Constraint(longitude=lambda y: -85 <= y <= -50)
    region = cube.extract(lat_constraint)
    region = region.extract(lon_constraint)

    # Interpolate onto 1deg x 1deg grid
    new_lats = np.arange(30, 41, 1)
    new_lons = np.linspace(-64, -69, num=11)

    latitude = iris.coords.DimCoord(new_lats, standard_name='latitude', units='degrees')
    longitude = iris.coords.DimCoord(new_lons, standard_name='longitude', units='degrees')
    new_cube = iris.cube.Cube(np.zeros((28, len(new_lats), len(new_lons)), np.float32),
                              dim_coords_and_dims=[(region.coord('tcell pstar'), 0),(latitude, 1), (longitude, 2)])

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
    model_linew = iris.cube.Cube(new, long_name=name, units=cube.units,  dim_coords_and_dims=[(region.coord('time'), 0),
                                                                                              (region.coord('tcell pstar'), 1),
                                                                                              (distance, 2)])
    return model_linew

o2_linew = interpolate_to_linew(o2, 'oxygen')
age_linew = interpolate_to_linew(age, 'age')

corr_o2_linew = np.ones((len(o2_linew.coord('tcell pstar').points),
                         len(o2_linew.coord('distance').points))) * np.nan
for x in range(0,len(o2_linew.coord('tcell pstar').points)):
    for y in range(0, len(o2_linew.coord('distance').points)):
        a = iris.analysis.stats.pearsonr(moc, o2_linew[:,x,y])
        corr_o2_linew[x, y] = a.data
np.save('/RESEARCH/chapter3/data/newCO2_control_800/o2_moc_corr_linew', corr_o2_linew)
#corr_o2_linew = np.load('/RESEARCH/chapter3/data/newCO2_control_800/o2_moc_corr_linew.npy')

corr_age_linew = np.ones((len(age_linew.coord('tcell pstar').points),
                         len(age_linew.coord('distance').points))) * np.nan
for x in range(0,len(age_linew.coord('tcell pstar').points)):
    for y in range(0, len(age_linew.coord('distance').points)):
        a = iris.analysis.stats.pearsonr(moc, age_linew[:,x,y])
        corr_age_linew[x, y] = a.data
np.save('/RESEARCH/chapter3/data/newCO2_control_800/age_moc_corr_linew', corr_age_linew)
#corr_age_linew = np.load('/RESEARCH/chapter3/data/newCO2_control_800/age_moc_corr_linew.npy')

plt.figure()
ax1 = plt.subplot(1,2,1)
plt.contourf(o2_linew.coord('distance').points, o2_linew.coord('tcell pstar').points,
             corr_o2_linew, clevs, cmap = 'RdBu_r')
ax1.invert_yaxis()
plt.title('(a) MOC-O2 Correlation')


ax2 = plt.subplot(1,2,2)
plt.contourf(o2_linew.coord('distance').points, age_linew.coord('tcell pstar').points,
             corr_age_linew, clevs, cmap = 'RdBu_r')
ax2.invert_yaxis()
plt.title('(b) MOC-Age Correlation')

plt.show()
