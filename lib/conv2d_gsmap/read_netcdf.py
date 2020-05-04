from netCDF4 import Dataset
from datetime import datetime
import numpy as np

# input_nc = Dataset('data/conv2d_gsmap/tmp.nc', 'r')

# # check dataset
# for i in input_nc.variables:
#     print(i, input_nc.variables[i].shape)

# time = np.array(input_nc['time'][:])
# input_lon = np.array(input_nc['lon'][:])
# input_lat = np.array(input_nc['lat'][:])
# input_precip = np.array(input_nc['precip'][:])

output_nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')
output_time = np.array(output_nc['time'][:])
output_lon = np.array(output_nc['lon'][:])
output_lat = np.array(output_nc['lat'][:])
output_precip = np.array(output_nc['precip'][:])

print(output_lat)
print(output_lon)

print(len(output_precip))

# Test nearist grid point
print(output_lat[0], output_lon[1], output_lon[2])
print(output_lat[1], output_lon[1], output_lon[2])

# Neu tu 0.85 den 0.95, lay 0.9 thi no' thien ve 0.95
print(output_precip[2,0,1])
print(output_precip[2,0,2])
print("Goc 3: ",output_precip[0,1,1]) 
print("Goc 4: ",output_precip[9,1,2])
print(output_lon[60])
print(output_lat[92])
np.savez('data/npz/conv2d_gsmap.npz',
         time=time,
         input_lon=input_lon,
         input_lat=input_lat,
         input_precip=input_precip,
         output_lon=output_lon,
         output_lat=output_lat,
         output_precip=output_precip)

# for i in range(0, 5):
#     print(datetime.fromtimestamp(time[i]))