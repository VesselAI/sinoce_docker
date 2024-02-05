import os
from sklearn.cluster import DBSCAN
import glob
import numpy as np
import pandas as pd
from datetime import timedelta
import geopandas as gpd
import argparse
from shapely.geometry import LineString, Point, Polygon
import warnings



def read_ais(path):
    """
    Processes folder of csv files
    """
    start_path = os.getcwd()
    os.chdir(path)
    csv_files = glob.glob("*.{}".format("csv"))
    ais = []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, sep=";", na_values="\\N")
        df.dropna(inplace=True)
        df = df.astype(dtype={"mmsi": "int64", "imo_nr": "int64", "length": "int64"})
        df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
        df.rename(columns={"imo_nr": "imo", "date_time_utc": "ts"}, inplace=True)
        ais.append(df)

    ais = pd.concat(ais, axis=0)
    os.chdir(start_path)
    return ais

def clean_ais(ais: gpd.GeoDataFrame):
    """
    Remove unrealistic speed and course values
    """
    ais = ais[ais['imo'] > 0]  # Filter our unrealistic imo #s
    ais = ais[ais['sog'] < 40]
    ais = ais[ais['cog'] <= 360]
    ais.reset_index(inplace=True, drop=True)
    return ais

def interpolate_traj(df, freq='min'):
    """
    Interpolate a df with ais trajectory data at given frequency
    Drops other data
    """
    time_index = pd.date_range(start=df.iloc[0]['ts'], end=df.iloc[-1]['ts'], freq=freq)
    index = pd.DataFrame(list(df['ts']) + list(time_index))
    index = index.drop_duplicates(keep='first')
    #index.sort_values(by=0, inplace=True)
    index.reset_index(inplace=True, drop=True)
    index.sort_values(by=0, inplace=True)
    index = index.drop_duplicates(keep='first')

    df.set_index('ts', inplace=True, drop=True)
    df = df.reindex(index.loc[:, 0])
    df.interpolate(method='time', inplace=True, limit_direction='forward', axis=0)
    df = df[df.index.isin(time_index)]
    df['ts'] = df.index
    df.reset_index(inplace=True, drop=True)

    return df

def split_routes(ais, delta_t=10, delta_sog=0.2, interpolate=True):
    """
    Parameters
    ----------
    ais : df of ais data
    delta_t : integer [min]
        time between trajectories
    delta_sog: float [kn]
        filter threshold for stopping points

    Returns
    -------
    None.

    """
    ais = ais[ais["sog"] > delta_sog]  # Filter out stopping points
    ais = ais.sort_values(by=['imo', 'ts'])
    ais = ais.reset_index(drop=True)

    imo_bool = ais.imo.diff() > 0  # Find changes in imo
    time_bool = ais.ts.diff() > timedelta(minutes=delta_t)  # Find changes in time
    tot_bool = np.column_stack((imo_bool, time_bool)).any(axis=1)

    traj_index = ais.index[tot_bool]

    traj_list = []
    for i in range(len(traj_index) - 1):

        traj = ais[traj_index[i]:traj_index[i + 1]]

        if traj.shape[0] > 10:

            if interpolate:
                traj_list.append(interpolate_traj(traj))
            else:
                traj_list.append(traj)

    return traj_list

def gen_trips(traj_list):
    df = pd.DataFrame(columns=('mmsi', 'imo', 'length', 'start_lon', 'start_lat', 'stop_lon', 'stop_lat', 'start_loc',
                               'stop_loc', 'start_geom', 'stop_geom', 'start_time', 'stop_time', 'cog', 'avg_cog',
                               'sog', 'avg_sog', 'loc', 'ts'))

    for traj in traj_list:

        if traj.shape[0] > 2:
            line = LineString(list(zip(traj['lon'], traj['lat'])))
            start_loc = Point(traj.iloc[0]['lon'], traj.iloc[0]['lat'])
            stop_loc = Point(traj.iloc[-1]['lon'], traj.iloc[-1]['lat'])

            row = pd.DataFrame({'mmsi': [traj.iloc[0]['mmsi']], 'imo': [traj.iloc[0]['imo']],
                                'start_lon': [traj.iloc[0]['lon']],
                                'start_lat': [traj.iloc[0]['lat']], 'stop_lon': [traj.iloc[-1]['lon']],
                                'stop_lat': [traj.iloc[-1]['lat']], 'start_loc': start_loc.wkt,
                                'stop_loc': stop_loc.wkt, 'start_geom': start_loc, 'stop_geom': stop_loc,
                                'start_time': [traj.iloc[0]['ts']], 'stop_time': [traj.iloc[-1]['ts']],
                                'length': [traj.iloc[0]['length']],
                                'avg_cog': [np.mean(traj['cog'])], 'avg_sog': [np.mean(traj['sog'])],
                                'cog': [list(traj['cog'])], 'sog': [list(traj['sog'])], 'loc': line.wkt,
                                'line_geom': line, 'ts': [list(traj['ts'])]})

            df = pd.concat((df, row))
    gdf = gpd.GeoDataFrame(df, geometry='line_geom')
    gdf = gdf.set_crs('EPSG:4326')

    return gdf



def cluster_start_and_stop(gdf, return_points=False):
    """
    Identify clusters of starting and stopping points for trajectories
    """

    start_points = gpd.GeoDataFrame(gdf['start_geom'], geometry='start_geom')
    start_points.set_crs('EPSG:4326', inplace=True)
    start_points.to_crs('EPSG:3035', inplace=True) # Evaluating in meters

    stop_points = gpd.GeoDataFrame(gdf['stop_geom'], geometry='stop_geom')
    stop_points.set_crs('EPSG:4326', inplace=True)
    stop_points.to_crs('EPSG:3035', inplace=True)  # Evaluating in meters

    start = np.array([start_points['start_geom'].x, start_points['start_geom'].y]).T
    stop = np.array([stop_points['stop_geom'].x, stop_points['stop_geom'].y]).T

    points = np.vstack((start, stop))
    clusterer = DBSCAN(eps=600, min_samples=25).fit(points)
    labels = clusterer.labels_

    start_labels = labels[:start.shape[0]]
    stop_labels = labels[start.shape[0]:]

    gdf2 = gdf.copy()
    gdf2['start_label'] = start_labels
    gdf2['stop_label'] = stop_labels
    gdf2 = gdf2[gdf2['start_label'] != -1]
    gdf2 = gdf2[gdf2['stop_label'] != -1]
    gdf2 = gdf2.reset_index(drop=True)

    if return_points:
        return points, labels, gdf2
    else:
        return gdf2


def filter_SSN(gdf):
    """
    Filter out vessels under 45 [m]
    Vesels above 45 [m] must report to SSN
    """
    return gdf[gdf['length']>=45]



def get_location_mapping(routes, loc_path="./location_data.csv", threshold=2000):

    loc = pd.read_csv(loc_path, delimiter=',')
    loc['geom'] = gpd.points_from_xy(loc['lon'], loc['lat'])
    loc = gpd.GeoDataFrame(loc, geometry='geom', crs='EPSG:4326')
    loc.dropna(inplace=True)
    loc = loc.to_crs('EPSG:3035')

    #routes = routes.to_crs('EPSG:3035')

    routes['start_geom'] = gpd.points_from_xy(routes['start_lon'], routes['start_lat'], crs='EPSG:4326').to_crs('EPSG:3035')
    routes['stop_geom'] = gpd.points_from_xy(routes['stop_lon'], routes['stop_lat'], crs='EPSG:4326').to_crs('EPSG:3035')

    max_x = routes.start_geom.bounds['maxx'].max()
    min_x = routes.start_geom.bounds['minx'].min()
    max_y = routes.start_geom.bounds['maxy'].max()
    min_y = routes.start_geom.bounds['miny'].min()

    lines = [LineString([(min_x, max_y), (max_x, max_y)]),
            LineString([(min_x, min_y), (max_x, min_y)]),
            LineString([(max_x, min_y), (max_x, max_y)]),
            LineString([(min_x, min_y), (min_x, max_y)])]

    direction_list = ["NORTH", "SOUTH", "EAST", "WEST"]

    for label in np.unique(routes['start_label']):

            points = routes.loc[routes["start_label"] == label, 'start_geom']

            start_center = Polygon(points.unary_union).centroid
            distance = loc.distance(start_center)
            #if distance.min()> fix this
            #idx = np.argmin(np.array(distance))
            if distance.min()<threshold:
                closest = loc.loc[distance.idxmin(), :]
                routes.loc[routes["start_label"] == label, 'start_loc_name'] = closest['locode']
            else:
                min_distance = float('inf')
                for i, line in enumerate(lines):
                    distance = line.distance(start_center)
                    if distance < min_distance:
                        min_distance = distance
                        idx = i
                routes.loc[routes["start_label"] == label, 'start_loc_name'] = f"OUT_{direction_list[idx]}"



    for label in np.unique(routes['stop_label']):

            points = routes.loc[routes["stop_label"] == label, 'stop_geom']
            stop_center = Polygon(points.unary_union).centroid
            distance = loc.distance(stop_center)
            #if distance.min()> fix this
            if distance.min()<threshold:
                closest = loc.loc[distance.idxmin(), :]
                routes.loc[routes["stop_label"] == label, 'stop_loc_name'] = closest['locode']
            else:
                min_distance = float('inf')
                for i, line in enumerate(lines):
                    distance = line.distance(start_center)
                    if distance < min_distance:
                        min_distance = distance
                        idx = i
                routes.loc[routes["stop_label"] == label, 'stop_loc_name'] = f"OUT_{direction_list[idx]}"

    routes['start_geom'] = routes["start_geom"].to_crs("EPSG:4326")
    routes['stop_geom'] = routes["stop_geom"].to_crs("EPSG:4326")


    return routes


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Process CSV file(s) and save data to a parquet file.")
    parser.add_argument("input_folder", help="Path to folder the CSV file(s) for processing")
    parser.add_argument('-o', '--output_file', help='Path to the output file')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    print("Loading data")
    ais = read_ais(path=args.input_folder)
    ais = clean_ais(ais)
    print("Splitting routes")
    routes = split_routes(ais, delta_t=10, delta_sog=0.2, interpolate=True)
    print("Generating trajectories")
    trips = gen_trips(routes)
    print("Clustering starting and stopping points")
    trips = cluster_start_and_stop(trips)
    trips = filter_SSN(trips)
    print("Mapping LOCODE")
    trips = get_location_mapping(trips)
    print("Saving to parquet")
    trips.to_parquet(path= args.output_file+"ais_data.parquet")







