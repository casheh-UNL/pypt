# This script interpolates Fig. 8 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.023502
# to find the effective number of relativistic Standard Model degrees of freedom as a function of temperature.
# The points are saved in a CSV file named 'eff_DoF_points.csv' in the same folder as this script.
# WARNING: interpolation ranges from 1e-3 to ~1e2 GeV. For a wider range, see https://arxiv.org/pdf/1609.04979.

import csv
import os
from scipy.interpolate import interp1d

#############
# IMPORT DATA
#############

def load_points_from_csv(file_name):
    # search for CSV file in the same directory as this script
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, file_name)
    points = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # extract points and convert to floats
            point = (float(row[0]), float(row[1]))
            points.append(point)
    return points


file_path = 'eff_DoF_points.csv'
points = load_points_from_csv(file_path)

##################
# INTERPOLATE DATA
##################

# temperatures measured in GeV
Ts = [point[0] for point in points]
gs = [point[1] for point in points]

g = interp1d(Ts, gs, kind='linear')

def g_SM(T):
    if T < min(Ts):
        return min(gs)
    elif T > max(Ts):
        return 106.75
    else:
        return float(g(T))