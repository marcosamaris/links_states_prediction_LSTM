import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import historical.readData.travels3   as tv
import networkx as nx


# Calculate the distance betwen bus stops
def get_distance_stops(distances, stop_buses_trip):

    # Distance between all the bus stops
    lat1 = stop_buses_trip.iloc[0, 7]
    lon1 = stop_buses_trip.iloc[0, 8]
    totalcal = [0.]
    dist = [0.]
    for lat2, lon2 in zip(stop_buses_trip.iloc[1:, 7], stop_buses_trip.iloc[1:, 8]):
        d = tv.haversine2(lat1, lon1, lat2, lon2)
        d = d * 1000
        dist.append(d)
        totalcal.append(totalcal[-1] + d)
        lat1 = lat2
        lon1 = lon2
    distance_all_stops = [stop_buses_trip.iloc[:,3], stop_buses_trip.iloc[:, 7], stop_buses_trip.iloc[:, 8], totalcal]

    return distance_all_stops


# Create the graph of a trip
def get_graph(select_lines, stop_buses_trip, distances):
    """

    :type stop_times: object
    """
    # Create a object Graph with the library networkX
    G = nx.Graph()

    # Add the nodes in the graph with each one of their attributes
    for node in range(len(stop_buses_trip)):
        G.add_node(node, id=stop_buses_trip.loc[node, "stop_id"], lat=stop_buses_trip.loc[node, "stop_lat"],
            lon=stop_buses_trip.loc[node, "stop_lon"], stop_seq=stop_buses_trip.loc[node, "stop_sequence"])

    distance_all_stops = get_distance_stops(distances, stop_buses_trip)

    # Add the edges with other attributes related to them
    for node in range(len(distance_all_stops[0])-1):
        G.add_edge(distance_all_stops[0][node], distance_all_stops[0][node + 1], distance=distance_all_stops[3][node])


    return G


if __name__ == '__main__':
    str_files_GTFS =  "../historical/readData/dados/gtfs/"

    trips = pd.read_csv(str_files_GTFS + 'trips.txt', sep=',')

    # select_lines = list(trips['trip_id'][0:20])
    #
    select_lines = ['8700-10-1', '8700-10-0', '1012-10-0', '1012-10-1' ]
    # select_lines = ['8700-10-1', '8700-10-0']

    stop_times = pd.read_csv(str_files_GTFS + 'stop_times.txt', sep=',')
    stops = pd.read_csv(str_files_GTFS + "stops.txt", sep=",")

    shapes = pd.read_csv(str_files_GTFS + "shapes.txt", sep=",")

    distances = tv.calcula_dist_shape(select_lines, trips, shapes)

    bus_network_graph = nx.MultiGraph
    array_graph_temporal = []

    for trip in zip(select_lines):
        stop_buses_line = stop_times.loc[stop_times['trip_id'].isin(trip)]
        stop_buses_line = pd.merge(stop_buses_line, stops, on="stop_id", how="left")

        graph_temporal = get_graph(trip[0], stop_buses_line, distances[trip[0]])

        array_graph_temporal.append(graph_temporal)

    bus_network_graph = nx.disjoint_union_all(array_graph_temporal)

print(nx.info(bus_network_graph))





