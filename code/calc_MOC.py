"""
Script to calculate the Eulerian MOC
"""


import iris
import iris.analysis.cartography
import iris.coord_categorisation
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
import glob
from netcdftime import utime
from datetime import datetime
import time
import cartopy.crs as ccrs


def calc_MOC(v_cube):
    '''Calculates the meridional streamfunction from v wind.

    Parameters
    ----------
    v: an iris cube of the form v[pressure levs, latitudes]

    Returns
    -------
    psi: iris cube of meridional streamfunction
    '''
    pname = 'tcell pstar'
    radius = 6371229.
    g = 9.81
    coeff = radius

    v = v_cube.data
    psi = np.empty_like(v) # (depth, lats, lons)

    lats  = v_cube.coord('latitude').points
    lons  = v_cube.coord('longitude').points

    p = v_cube.coord(pname).points

    for ilon in range(lons.shape[0]):
        for ilat in range(lats.shape[0]):
            psi[0,ilat, ilon] = coeff*np.cos(np.deg2rad(lats[ilat])) *  v[0,ilat,ilon] * p[0]
            for ilev in range(p.shape[0])[1:]:
                psi[ilev,ilat,ilon] = psi[ilev-1,ilat,ilon] + coeff*np.cos(np.deg2rad(lats[ilat])) \
                                 * v[ilev,ilat,ilon] * (p[ilev]-p[ilev-1])

    psi_zm = np.zeros((p.shape[0],lats.shape[0]))

    for ilat in range(lats.shape[0]):
        for ilev in range(p.shape[0]):
            for ilon in range(lons.shape[0]):
                if psi.mask[ilev,ilat,ilon] == False:
                    psi_zm[ilev,ilat] += psi[ilev,ilat,ilon] * np.deg2rad(3.)

    psi_cube = iris.cube.Cube(psi_zm, var_name='psi', long_name='meridional_streamfunction')
    psi_cube.add_dim_coord(v_cube.coord(pname),0)
    psi_cube.add_dim_coord(v_cube.coord('latitude'),1)

    return psi_cube



def calc_MOC_from_T(T, vertical_integral=False):

    loncon = iris.Constraint(longitude = lambda x: -80 <= x <=0)
    T = T.extract(loncon)
    T = T.collapsed('longitude', iris.analysis.SUM)

    if vertical_integral:
        psi = np.ma.zeros(T.shape)
        psi[:,0,:] = T[:,0,:].data
        for iz in range(T.shape[1])[1:]:
            psi[:,iz,:] = psi[:,iz-1,:] + T[:,iz,:].data
    else:
        psi = T.data


    pname = 'tcell pstar'
    psi_cube = iris.cube.Cube(psi, var_name='psi', long_name='meridional_streamfunction')
    psi_cube.add_dim_coord(T.coord('time'),0)
    psi_cube.add_dim_coord(T.coord(pname),1)
    psi_cube.add_dim_coord(T.coord('latitude'),2)
    psi_cube.units = 'Sv'

    return psi_cube


PATH = '/RESEARCH/chapter3/data/newCO2_control_800/'


T_Eul = iris.load_cube(PATH+'ty_trans.nc')
T_GM = iris.load_cube(PATH+'ty_trans_gm.nc')
T_SM = iris.load_cube(PATH+'ty_trans_submeso.nc')


psi_Eul = calc_MOC_from_T(T_Eul, vertical_integral=True)
psi_GM = calc_MOC_from_T(T_GM)
psi_SM = calc_MOC_from_T(T_SM)


fig = plt.figure()
ax1 = fig.add_subplot(311)
qplt.pcolormesh(psi_Eul[0,:],vmin=-30,vmax=30)
ax1 = fig.add_subplot(312)
qplt.pcolormesh(psi_GM[0,:],vmin=-30,vmax=30)
ax1 = fig.add_subplot(313)
qplt.pcolormesh(psi_SM[0,:],vmin=-30,vmax=30)



fig = plt.figure()
qplt.pcolormesh(psi_Eul[0,:]+psi_GM[0,:]+psi_SM[0,:],vmin=-30,vmax=30)
qplt.show()

#qplt.pcolormesh(psi_cube[0,:],vmin=-30,vmax=30)
#qplt.show()


psi_cube = psi_Eul+psi_GM
iris.save(psi_cube, '/RESEARCH/chapter3/data/newCO2_control_800/derived/MOC.nc')
#iris.save(psi_cube, '/RESEARCH/chapter3/data/newCO2_control_800/derived/MOC.nc')

'''
#for iv in v:
#    iv.remove_coord(iv.aux_coords[0])
#    iv.remove_coord(iv.aux_coords[0])
#    iv.attributes = {}
#v = v.concatenate_cube()

#v = v.collapsed('longitude', iris.analysis.MEAN)

v_cube = v.collapsed('time', iris.analysis.MEAN)


iris.save(v_cube,'MOC_cntrl.nc')


loncon = iris.Constraint(longitude = lambda x: -80 <= x <=0)

v_cube = iris.load_cube('MOC_test.nc').extract(loncon)
v_cube_cntrl = iris.load_cube('MOC_cntrl.nc').extract(loncon)

psi_cube = calc_MOC(v_cube)
psi_cube_cntrl = calc_MOC(v_cube_cntrl)
'''
