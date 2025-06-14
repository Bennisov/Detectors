import scipy
import matplotlib.pyplot as plt
import numpy
import pandas
import numba
import numpy as np


column_names = ['ASIC', 'Channel', 'TimeStamp', 'Adc', 'PileUp', 'OverFlow',
                'Gain', 'Thr', 'iCal', 'Trim', 'TS Asic', 'TS Fpga']
data_calib = pandas.read_csv('raw_data_ical_no_calib.dat', delimiter='\t', names=column_names, skiprows=2)
column_names = ['ASIC', 'Channel', 'TimeStamp', 'Adc', 'OverFlow', 'PileUp']
data = pandas.read_csv('GEM_HV_3930_2_AGH_plain_part_008.pcap_raw_hits.txt', delimiter='\t', names=column_names,
                       skiprows=1, index_col=False)
asic_calib = numpy.array(data_calib['ASIC'].tolist())
channel_calib = numpy.array(data_calib['Channel'].tolist())
adc_calib = numpy.array(data_calib['Adc'].tolist())
ical_calib = numpy.array(data_calib['iCal'].tolist())
asic_data = numpy.array(data['ASIC'].tolist())
channel_data = numpy.array(data['Channel'].tolist())
time_data = numpy.array(data['TimeStamp'].tolist())
adc_data = numpy.array(data['Adc'].tolist())


@numba.njit
def linregress_numba(x, y):
    n = x.shape[0]
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0

    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xx += x[i] * x[i]
        sum_xy += x[i] * y[i]

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 1.0, 0.0  # default slope and intercept to avoid division by zero

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return abs(slope), intercept


@numba.njit
def ele_calib(asic_data, channel_data, adc_data, asic_calib, channel_calib, adc_calib, ical_calib):
    n = asic_calib.shape[0]
    max_pairs = 200  # Adjust based on expected number of unique pairs
    slopes = np.zeros(max_pairs)
    intercepts = np.zeros(max_pairs)
    unique_a = np.full(max_pairs, -1)
    unique_c = np.full(max_pairs, -1)
    pair_count = 0

    # Build unique (asic, channel) list and compute slope/intercept
    for i in range(n):
        a = asic_calib[i]
        c = channel_calib[i]

        pair_index = -1
        for j in range(pair_count):
            if unique_a[j] == a and unique_c[j] == c:
                pair_index = j
                break

        if pair_index == -1:
            pair_index = pair_count
            unique_a[pair_index] = a
            unique_c[pair_index] = c
            pair_count += 1

    # Pre-allocate buffers
    temp_x = np.empty(n)
    temp_y = np.empty(n)

    for k in range(pair_count):
        a = unique_a[k]
        c = unique_c[k]
        count = 0

        # Collect matching data
        for i in range(n):
            if asic_calib[i] == a and channel_calib[i] == c:
                temp_x[count] = ical_calib[i]
                temp_y[count] = adc_calib[i]
                count += 1

        slope, intercept = linregress_numba(temp_x[:count], temp_y[:count])
        slopes[k] = slope
        intercepts[k] = intercept

    # Apply calibration
    m = asic_data.shape[0]
    for i in range(m):
        a = asic_data[i]
        c = channel_data[i]
        for j in range(pair_count):
            if unique_a[j] == a and unique_c[j] == c:
                slope = slopes[j]
                intercept = intercepts[j]
                adc_data[i] = (adc_data[i] - intercept) / slope
                break


ele_calib(asic_data, channel_data, adc_data, asic_calib, channel_calib, adc_calib, ical_calib)


@numba.njit
def map_events(asic_data, channel_data, adc_data, time_data):
    sort_indices = np.argsort(time_data)

    # Sort arrays
    asic_data = asic_data[sort_indices]
    channel_data = channel_data[sort_indices]
    adc_data = adc_data[sort_indices]
    time_data = time_data[sort_indices]

    # Compute time differences manually (no prepend)
    time_diff = np.empty_like(time_data)
    time_diff[0] = 0
    for i in range(1, len(time_data)):
        time_diff[i] = time_data[i] - time_data[i - 1]

    # Grouping
    group_id = np.empty_like(time_data)
    current_group = 0
    group_id[0] = current_group
    for i in range(1, len(time_data)):
        if time_diff[i] >= 100:
            current_group += 1
        group_id[i] = current_group

    # Geometry logic
    asic_adj = asic_data - 86
    coord = np.zeros_like(asic_adj)
    axis = np.empty_like(asic_adj)  # Use 0 for 'x', 1 for 'y'

    for i in range(len(asic_data)):
        if asic_adj[i] > 3:
            axis[i] = 1  # y-axis
            coord[i] = channel_data[i] + (3 - asic_adj[i]) * 32
        else:
            axis[i] = 0  # x-axis
            coord[i] = 31 - channel_data[i] + asic_adj[i] * 32

    # Get unique group count
    max_group_id = group_id[0]
    for i in range(1, len(group_id)):
        if group_id[i] > max_group_id:
            max_group_id = group_id[i]
    n_groups = max_group_id + 1

    event_x = np.full(n_groups, np.nan, dtype=np.float64)
    event_y = np.full(n_groups, np.nan, dtype=np.float64)
    event_values = np.zeros(n_groups, dtype=np.float64)

    for group_idx in range(n_groups):
        x_sum_weighted = 0.0
        x_sum_weights = 0.0
        y_sum_weighted = 0.0
        y_sum_weights = 0.0
        total_value = 0.0

        for i in range(len(group_id)):
            if group_id[i] == group_idx:
                total_value += adc_data[i]
                if axis[i] == 0:
                    x_sum_weighted += coord[i] * adc_data[i]
                    x_sum_weights += adc_data[i]
                else:
                    y_sum_weighted += coord[i] * adc_data[i]
                    y_sum_weights += adc_data[i]

        if x_sum_weights > 0:
            event_x[group_idx] = round(x_sum_weighted / x_sum_weights)
        if y_sum_weights > 0:
            event_y[group_idx] = round(y_sum_weighted / y_sum_weights)
        event_values[group_idx] = total_value

    return event_x, event_y, event_values


events_x, events_y, event_values = map_events(asic_data, channel_data, adc_data, time_data)
valid_mask = ~np.isnan(events_x) & ~np.isnan(events_y)
x = events_x[valid_mask]
y = events_y[valid_mask]
weights = event_values[valid_mask]

# Bin edges (adjust if needed)
xbins = ybins = np.arange(0, 96)

hist, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], weights=weights)

plt.figure(figsize=(8, 6))
plt.imshow(hist.T, origin='lower', cmap='inferno', extent=[0, 96, 0, 96], aspect='equal')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('ADC-Weighted Event Map')
plt.colorbar(label='Total ADC per Bin')
plt.grid(False)
plt.tight_layout()
plt.show()