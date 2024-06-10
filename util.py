import plotly.express as px
import plotly.graph_objects as go
import kaleido
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from folium import IFrame
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from branca.colormap import LinearColormap
import scienceplots as sp
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch
import matplotlib.path as mpath
import random
plt.style.use('science')

us_airport_loc = pd.read_csv('udata/us_airport_loc.csv')
flow = pd.read_csv('udata/_map_data.csv')
merged_data = pd.merge(us_airport_loc, flow, left_on='Name', right_on='Code', how='left')
u_mx = np.load('udata/udelay.npy')
od = np.load('udata/od_pair.npy')

arr = u_mx[:, :, 0].T
dep = u_mx[:, :, 1].T

df_arr = pd.DataFrame(arr, index=range(
    78912), columns=us_airport_loc['Name'].tolist())
df_dep = pd.DataFrame(dep, index=range(
    78912), columns=us_airport_loc['Name'].tolist())


class BezierCurve(FancyArrowPatch):
    def __init__(self, src, dst, color, linewidth, alpha):
        path = self._get_bezier_path(src, dst)
        FancyArrowPatch.__init__(self, path=path, color=color, linewidth=linewidth, alpha=alpha)

    def _get_bezier_path(self, src, dst):
        Path = mpath.Path
        verts = [src, ((src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2 + 0.1 * abs(src[0] - dst[0])), dst]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        return Path(verts, codes)


def get_delay_time(time_slot, airport_code):
    arr_delay = df_arr.loc[time_slot, :].fillna(0)
    dep_delay = df_dep.loc[time_slot, :].fillna(0)

    arr = arr_delay[airport_code]
    dep = dep_delay[airport_code]

    return arr, dep


def get_airpot_idx(airport_code):
    airport_idx = us_airport_loc[us_airport_loc['Name'] == airport_code].index[0]
    return airport_idx


def get_airport_name(airport_idx):
    airport_name = us_airport_loc.loc[airport_idx, 'Name']
    return airport_name


def get_od_delay(time_slot, num_node, dropout_rate=0.5):
    res = {}
    for src in range(num_node):
        src_code = get_airport_name(src)
        src_idx = get_airpot_idx(src_code)

        destinations = [dst for dst in range(num_node) if od[src_idx, get_airpot_idx(get_airport_name(dst))] != 0]
        num_to_drop = int(len(destinations) * dropout_rate)
        dropped_destinations = set(random.sample(destinations, num_to_drop))

        for dst in range(num_node):
            if dst in dropped_destinations:
                continue

            dst_code = get_airport_name(dst)
            dst_idx = get_airpot_idx(dst_code)

            if od[src_idx, dst_idx] == 0:
                continue

            arr_delay = df_arr.loc[time_slot, src_code]
            dep_delay = df_dep.loc[time_slot, dst_code]

            if pd.isna(arr_delay) or pd.isna(dep_delay):
                continue

            total_delay = arr_delay + dep_delay
            res[(src_code, dst_code)] = total_delay

    return res


def get_delay_time_all(airport_code):
    arr_delay = df_arr.loc[:, :]
    dep_delay = df_dep.loc[:, :]

    arr = arr_delay[airport_code]
    dep = dep_delay[airport_code]

    return arr, dep


def slot_to_time(slot):
    days_passed = slot // 36
    slot_in_day = slot % 36
    hour = 6 + slot_in_day // 2
    minute = (slot_in_day % 2) * 30
    date = datetime(2016, 1, 1) + timedelta(days=days_passed)
    date_str = date.strftime('%Y-%m-%d')

    return f"{date_str} {hour:02d}:{minute:02d}"


def get_airport_location(airport_code):
    airport = us_airport_loc[us_airport_loc['Name'] == airport_code]
    if airport.empty:
        print(f"No location data found for airport {airport_code}")
        return None
    else:
        lat = airport['LATITUDE'].values[0]
        lon = airport['LONGITUDE'].values[0]
        return lat, lon


def plot_edges(ax, map, od_delay):
    cmap = plt.get_cmap('jet')

    min_delay = min(od_delay.values())
    max_delay = max(od_delay.values())
    range_delay = max_delay - min_delay
    if range_delay == 0:
        range_delay = 1

    norm = plt.Normalize(min_delay, max_delay)

    for (src_code, dst_code), delay in od_delay.items():
        if delay == 0:
            continue

        src_loc = get_airport_location(src_code)
        dst_loc = get_airport_location(dst_code)

        if src_loc is None or dst_loc is None:
            continue

        src_x, src_y = map(src_loc[1], src_loc[0])
        dst_x, dst_y = map(dst_loc[1], dst_loc[0])

        # Filter out edges outside the specified range
        if not (-140 <= src_loc[1] <= -63 and 22 <= src_loc[0] <= 49 and
                -140 <= dst_loc[1] <= -63 and 22 <= dst_loc[0] <= 49):
            continue

        normalized_delay = (delay - min_delay) / range_delay
        color = cmap(normalized_delay)
        linewidth = 0.5 + 4 * normalized_delay
        alpha = min(1, max(0, 0.4 + 0.6 * normalized_delay))

        curve = BezierCurve((src_x, src_y), (dst_x, dst_y), color=color, linewidth=linewidth, alpha=alpha)
        ax.add_patch(curve)


def get_flow(airport_code):
    file_path = 'udata/_map_data.csv'
    df = pd.read_csv(file_path)
    row = df[df['Code'] == airport_code]
    if not row.empty:
        passengers = row['Passengers'].values[0]
        return passengers
    else:
        return None


def plot_airports(ax, map, s, time_slot, delay_type, dep=None, arr=None):
    size_min = 150
    size_max = 700

    arr_delay = df_arr.loc[time_slot, :].fillna(0)
    dep_delay = df_dep.loc[time_slot, :].fillna(0)

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(-10, 30)

    # Get passenger flows for all airports
    us_airport_loc['Passengers'] = us_airport_loc['Name'].apply(get_flow)

    # Normalize passenger flows to fit within size_min and size_max
    passenger_min = us_airport_loc['Passengers'].min()
    passenger_max = us_airport_loc['Passengers'].max()
    size_norm = Normalize(vmin=passenger_min, vmax=passenger_max)

    for airport in us_airport_loc['Name']:
        lat, lon = get_airport_location(airport)
        x, y = map(lon, lat)
        arr = arr_delay[airport]
        dep = dep_delay[airport]

        if delay_type == 'arr':
            color = cmap(norm(arr))
            data = arr
        elif delay_type == 'dep':
            color = cmap(norm(dep))
            data = dep

        # Get the size based on passenger flow
        passengers = get_flow(airport)
        if passengers is not None:
            normalized_size = size_norm(passengers)
            size = size_min + normalized_size * (size_max - size_min)
        else:
            size = size_min  # Default size if no passenger data

        res = []
        if data > 10:
            res.append([airport, data])

        map.scatter(x, y, s=size, color='gray', marker='o', edgecolor='black', linewidth=0.8, alpha=1)
