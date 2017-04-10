import numpy as np
import iris
import shapely.geometry
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing


def basin_coords(basin_name):
    """ Determine the basins based on the basin name and create the polygon """
    if basin_name=='arctic':
       xp=[20.0,66.0,67.0,59.0,54,63.0,95.0,150.0,-175.0,-164,-110.0,-72.0] # ,75,190,235,300,345,20]
       yp=[80.0,80.0,76.5,75.0,72,69.0,66.0,66.0 , 66.5,65.5, 66.0,   80.0]  #70,68,70,83,80,80]
    elif basin_name=='atlantic':
       xp=[-65.0,-76.0,-77.0,-78.0,-82.0,-91.0,-101.0,-130.0,-164.0,-175.0,150.0,95.0,36.0,20.0,11.0, 9.0, 9.0,-5.5,-5.5,25.0, 19.5, 10.0,  0.0,-13.0,-20.0,-30.0,-40.0,-50.0,-60.0,-70.0,-60.0,-65.0]
       yp=[  9.0,  7.0,  8.5,  9.5,  9.0, 17.0,  21.0,  65.6,  65.5,  66.5, 66.0,66.0,62.0,68.0,60.0,55.0,49.0,36.0,35.0, 0.0,-34.5,-40.0,-45.0,-50.0,-50.0,-50.0,-50.0,-50.0,-50.0,-50.0,-20.0,  9.0]
    elif basin_name=='upwelling_equitorial_atlantic':
        yp=[-40, -13,  -9,  -8,  -5,  -3,   0,   5,   9,  31,  14,  12,   8,  4, 6, 3, -5, -12, -17, -22, -28, -40]
        xp=[ 20,  -1, -23, -35, -35, -40, -50, -52, -80, -10, -17, -17, -14, -8, 5, 9, 12,  13,  12,  14,  15,  20]


    return xp,yp

def basin_mask(cube,basin_name,is_basin=True,xp=None,yp=None):
    """ Find out whether lon, lat coordinated are inside a polygon """
    #if the basin polygon is not given find it based on the name
    if is_basin:
      xp,yp=basin_coords(basin_name)

    # Take latitude and longitude coordinates from basin and put in a list of tuples
    poly=[]
    for j in range(len(xp)):
        poly.append((yp[j],xp[j]))

    # Use Shapely package to create boarder and area of basin polygon
    ring=LinearRing(poly)
    area = Polygon(ring)

    if len(cube.shape) == 4:
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points
        depth = cube.coord('tcell pstar').points

        inside = np.ones((len(depth),len(lats), len(lons)))*np.nan
        for k in range(len(depth)):
            for i in range(len(lats)):
                for j in range(len(lons)):
                    coord = (lats[i],  lons[j])
                    inside[k, i, j] = area.contains(Point(coord))

    elif len(cube.shape) == 3:
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points

        inside = np.ones((len(lats), len(lons)))*np.nan
        for i in range(len(lats)):
            for j in range(len(lons)):
                coord = (lats[i],  lons[j])
                inside[i, j] = area.contains(Point(coord))

    # Apply mask to cube:
    a = cube.data
    b = np.logical_not(inside)
    array_masked = np.ma.array(a, mask=np.tile(b, (a.shape[0],1)))

    # Convert masked array to a cube
    cube_masked = iris.cube.Cube(array_masked, long_name=cube.long_name, units=cube.units,
                                               dim_coords_and_dims = [(cube.coord('time'), 0),
                                               (cube.coord('tcell pstar'), 1),
                                               (cube.coord('latitude'), 2),
                                               (cube.coord('longitude'), 3)],
                                                attributes=cube.attributes)
    return cube_masked
