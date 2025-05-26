import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

data1 = pd.read_csv(
    'RXPv3_b03PF500um_Fe55BS_R15_NO50000_2025-05-08_1526_ADCOUTchs_0.dat',
    delimiter='\t',
    header=None
)
data2 = pd.read_csv(
    'RXPv3_b03PF500um_Fe55FS_R15_NO100000_2025-05-08_1502_ADCOUTchs_0.dat',
    delimiter='\t',
    header=None
)
data3 = pd.read_csv(
    'RXPv3_b04PD1000um_Fe55FS_R15_NO100000_2025-05-08_1556_ADCOUTchs_2.dat',
    delimiter='\t',
    header=None
)
data1_np = data1.to_numpy()
data2_np = data2.to_numpy()
data3_np = data3.to_numpy()
data1_np = numpy.where(data1_np == 4095, 0, data1_np)
data2_np = numpy.where(data2_np == 4095, 0, data2_np)
data3_np = numpy.where(data3_np == 4095, 0, data3_np)


def mask_500(arr):
    out_arr = np.zeros((arr.shape[0], 6, 10))
    index_map = [
        (2, 0), (1, 0), (0, 0), (2, 1), (1, 1), (0, 1), (2, 2), (1, 2), (0, 2), (2, 3),
        (1, 3), (0, 3), (2, 4), (1, 4), (0, 4), (2, 5), (1, 5), (0, 5), (2, 6), (1, 6),
        (0, 6), (2, 7), (1, 7), (0, 7), (2, 8), (1, 8), (0, 8), (2, 9), (1, 9), (0, 9), (np.nan, np.nan), (np.nan, np.nan),
        (3, 9), (4, 9), (5, 9), (3, 8), (4, 8), (5, 8), (3, 7), (4, 7), (5, 7), (3, 6),
        (4, 6), (5, 6), (3, 5), (4, 5), (5, 5), (3, 4), (4, 4), (5, 4), (3, 3), (4, 3),
        (5, 3), (3, 2), (4, 2), (5, 2), (3, 1), (4, 1), (5, 1), (3, 0), (4, 0), (5, 0),
        (np.nan, np.nan), (np.nan, np.nan)
    ]
    for it in range(arr.shape[0]):
        for i in range(64):
            row, col = index_map[i]
            if np.isnan(row):
                continue
            out_arr[it, row, col] = arr[it, i]
    return out_arr


def mask_1000(arr):
    out_arr = np.zeros((arr.shape[0], 4, 5))
    index_map = (
            [(np.nan, np.nan), (1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)] +
            [(np.nan, np.nan)] * 20 +
            [(1, 3), (0, 3), (1, 4), (0, 4)] +
            [(np.nan, np.nan)] * 2 +
            [(2, 4), (3, 4), (2, 3), (3, 3), (2, 2), (3, 2)] +
            [(np.nan, np.nan)] * 20 +
            [(2, 1), (3, 1), (2, 0), (3, 0), (np.nan, np.nan)]
    )
    for it in range(arr.shape[0]):
        for i in range(64):
            row, col = index_map[i]
            if np.isnan(row):
                continue
            out_arr[it, row, col] = arr[it, i]
    return out_arr


mapped_data1 = mask_500(data1_np)
mapped_data2 = mask_500(data2_np)
mapped_data3 = mask_1000(data3_np)
print(mapped_data3[5])


def search(arr):
    distance_threshold = 2
    energy_threshold = 4095
    index_x, index_y = np.where(arr > 0)
    coords = list(zip(index_x, index_y))
    n_hits = len(coords)

    if n_hits == 0:
        return 0, []

    clustered = [False] * n_hits
    clusters = []

    for i in range(n_hits):
        if clustered[i]:
            continue
        cluster = [i]
        clustered[i] = True
        for j in range(i + 1, n_hits):
            if clustered[j]:
                continue
            dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            if dist <= distance_threshold:
                total_energy = arr[coords[i][0], coords[i][1]] + arr[coords[j][0], coords[j][1]]
                if total_energy < energy_threshold:
                    cluster.append(j)
                    clustered[j] = True
        clusters.append(cluster)

    energies = []
    for cluster in clusters:
        energy = sum(arr[coords[i][0], coords[i][1]] for i in cluster)
        energies.append(energy)

    return len(clusters), energies


labels1 = numpy.zeros(mapped_data1.shape[0])
labels2 = numpy.zeros(mapped_data1.shape[0])
labels3 = numpy.zeros(mapped_data1.shape[0])
energies1 = numpy.array([])
energies2 = numpy.array([])
energies3 = numpy.array([])
for i in range(mapped_data1.shape[0]):
    labels1[i], energy1 = search(mapped_data1[i])
    labels2[i], energy2 = search(mapped_data2[i])
    labels3[i], energy3 = search(mapped_data3[i])
    energies1 = np.append(energies1, energy1)
    energies2 = np.append(energies2, energy2)
    energies3 = np.append(energies3, energy3)
plt.figure()
plt.hist(energies1, bins=1000)
plt.yscale('log')
plt.show()
plt.figure()
plt.hist(energies2, bins=1000)
plt.yscale('log')
plt.show()
plt.figure()
plt.hist(energies3, bins=1000)
plt.yscale('log')
plt.show()