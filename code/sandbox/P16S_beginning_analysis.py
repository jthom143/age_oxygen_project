import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

fname ='6_33RR20050106_hy1.csv'
path = '/RESEARCH/chapter3/data/GO_SHIP/P16/33RR200501/'
data = pd.read_csv(path+fname, header = 78, na_values='-999.000')

CFC_data = data[['DATE', 'LATITUDE', 'LONGITUDE', 'CTDPRS', 'CFC-12', 'CFC-12_FLAG_W']].copy()
CFC_data.columns = ['date', 'latitude', 'longitude', 'press', 'CFC_12', 'CFC_12_FLAG_W']
CFC_data=CFC_data.drop(data.index[[0]])

a = CFC_data.latitude.values
indexes = np.unique(a, return_index=True)[1]
line_lats = [a[index] for index in sorted(indexes)]

grid_z, grid_x = np.mgrid[0:5000:51j, line_lats[-2]:line_lats[0]:116j]

points = np.column_stack((CFC_data.press.values, CFC_data.latitude.values))
value = CFC_data.CFC_12.values
#b = griddata(points, value, (grid_z, grid_x), method = 'linear')
