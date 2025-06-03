import scipy
import matplotlib.pyplot as plt
import numpy
import pandas


def ele_calib(xs, ys):
    result = scipy.stats.linregress(xs, ys)
    return abs(result.slope), result.intercept


column_names = ['ASIC', 'Channel', 'TimeStamp', 'Adc', 'PileUp', 'OverFlow',
                'Gain', 'Thr', 'iCal', 'Trim', 'TS Asic', 'TS Fpga']
data_calib = pandas.read_csv('raw_data_ical_no_calib.dat', delimiter='\t', names=column_names, skiprows=2)
column_names = ['ASIC', 'Channel', 'TimeStamp', 'Adc', 'OverFlow', 'PileUp']
data = pandas.read_csv('GEM_HV_3930_2_AGH_plain_part_008.pcap_raw_hits.txt', delimiter='\t', names=column_names, skiprows=1, index_col=False)
grouped_data = {}
slope_data = {}


for (asic, channel), group in data_calib.groupby(['ASIC', 'Channel']):
    xs = group['iCal'].tolist()
    ys = group['Adc'].tolist()
    grouped_data[(asic, channel)] = (xs, ys)

for (asic, channel) in grouped_data:
    xs, ys = grouped_data[(asic, channel)]
    slope, intercept = ele_calib(xs, ys)
    slope_data[(asic, channel)] = (slope, intercept)


def calibrate_row(row):
    slope, intercept = slope_data[(row['ASIC'], row['Channel'])]
    return (row['Adc'] - intercept) / slope


data['Adc'] = data.apply(calibrate_row, axis=1)
#
# example = (91, 0)
# adcs = data['Adc'].where((data['ASIC'] == example[0]) & (data['Channel'] == example[1]))
# plt.figure()
# plt.hist(adcs, bins=100)
# plt.show()


data = data.sort_values(by='Timestamp').reset_index(drop=True)
time_diff = data['Timestamp'].diff().fillna(0)
new_group = (time_diff >= 100).cumsum()
data['GroupID'] = new_group
print(data['GroupID'].head(100))



