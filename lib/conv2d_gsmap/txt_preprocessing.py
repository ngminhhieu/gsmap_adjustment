import glob
import pandas as pd
from datetime import datetime
import re
import numpy as np

data_dir = 'data/conv2d_gsmap/raw_data_daily/'
data_paths = glob.glob(data_dir + '*.txt')


def satisfy(year, month, day):
    month_31 = [1, 3, 5, 7, 8, 10, 12]
    month_30 = [4, 6, 9, 11]
    if month in month_31:
        return True
    elif month in month_30:
        if day <= 30:
            return True
    else:
        if day <= 28:
            return True
        if day <= 29 and year % 4 == 0:
            return True
    return False


def process(file_pth):
    f = open(file_pth, "r")
    """Read name, lon, lat"""
    name = f.readline()
    name = re.sub(' +', ' ', name).strip()
    address = f.readline()
    address = re.sub(' +', ' ', address).strip().split()
    lon = address[0]
    lat = address[1]
    height = address[2]
    """Preprocessing time and precipitation"""
    time = []
    precipitation = []
    day_arr = []
    month_arr = []
    year_arr = []
    year = 2011
    while year < 2019:
        f.readline()
        for day in range(1, 32):
            data = f.readline()
            data = re.sub(' +', ' ', data).strip().split()
            for month in range(1, len(data)):
                data[month] = float(data[month])
                if satisfy(year, month, day):
                    time.append(datetime(year, month, day).timestamp())
                    day_arr.append(day)
                    month_arr.append(month)
                    year_arr.append(year)
                    if data[month] < 0:
                        data[month] = 0.0
                    precipitation.append(data[month])
        year += 1

    df = pd.DataFrame()
    df['time'] = time
    df['day'] = day_arr
    df['month'] = month_arr
    df['year'] = year_arr
    df['precipitation'] = precipitation
    df = df.sort_values('time', ascending=True)

    df.to_csv(f'./data/conv2d_gsmap/preprocessed_txt_data/{name}_{lon}_{lat}_{height}.csv',
              index=False)


# for file in data_paths:
#     process(file)


def test():
    original_nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')

    # check dataset
    gsmap_precip = np.array(original_nc['precip'][:])
    gsmap_precip = np.round(gsmap_precip, 1)

    abc = gsmap_precip.shape[1]*gsmap_precip.shape[2]
    rain_gsmap = np.zeros(shape=(1766, abc))
    for lat in range(gsmap_precip.shape[1]):
        for lon in range(gsmap_precip.shape[2]):
            rain_gsmap[:, lat*120+lon] = gsmap_precip[:, lat, lon]
    for i in range(abc):
        rain_gsmap[:, i] = gsmap_precip[]