# Script to zonally average the GFDL ESM2Mc model ideal age in each ocean basin
# Created: March 6, 2017
# Last Edited: March 6, 2017
import cartopy.crs as ccrs
from cartopy.io.shapereader import natural_earth, Reader
from cartopy.mpl.patch import geos_to_path
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
from matplotlib.path import Path


def get_geometries(country_names):
    """
    Get an iterable of Shapely geometries corrresponding to given countries.

    """
    # Using the Natural Earth feature interface provided by cartopy.
    # You could use a different source, all you need is the geometries.
    shape_records = Reader(natural_earth(resolution='110m',
                                         category='physical',
                                         name='ocean')).records()
    geoms = []
    for country in shape_records:
        if country.attributes['name_long'] in country_names:
            try:
                geoms += country.geometry
            except TypeError:
                geoms.append(country.geometry)
    return geoms, ccrs.PlateCarree()._as_mpl_transform


def pcolor_mask_geoms(cube, geoms, transform):
    path = Path.make_compound_path(*geos_to_path(geoms))
    im = iplt.pcolor(cube)
    im.set_clip_path(path, transform=transform)


# First plot the full map:
cube = iris.load_cube('/RESEARCH/paper_ocean_heat_carbon/data/newCO2_control_800/sst.nc')
cube = cube[0,:,:]
plt.figure(figsize=(12, 6))
ax1 = plt.axes(projection=ccrs.PlateCarree())
ax1.coastlines()
iplt.pcolor(cube)

'''
# Now plot just the required countries:
plt.figure(figsize=(12, 6))
ax2 = plt.axes(projection=ccrs.PlateCarree())
ax2.coastlines()
countries = [
    'United States',
    'United Kingdom',
    'Saudi Arabia',
    'South Africa',
    'Nigeria']
geoms, transform = get_geometries(countries)
pcolor_mask_geoms(cube, geoms, transform(ax2))
'''
plt.show()

ocean_shp = shpreader.natural_earth(resolution='110m',
                                    category='physical',
                                    name='ocean')
