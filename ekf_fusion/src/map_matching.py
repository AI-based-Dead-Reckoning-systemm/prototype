import os
import osmnx as ox
import pandas as pd
import numpy as np

# Bounding box for Coventry, UK (SIH test area)
BBOX = {
    'north': 52.42,
    'south': 52.38,
    'east': -1.51,
    'west': -1.59
}

def load_or_download_graph(cache_path: str = "ekf_fusion/data/coventry_road_network.graphml"):
    """
    Loads the OSM road network graph from a cached .graphml file if it exists.
    Otherwise, downloads it via osmnx and caches it.
    """
    if os.path.exists(cache_path):
        print(f"Loading road network from cache: {cache_path}")
        G = ox.load_graphml(cache_path)
    else:
        print("Downloading road network via osmnx...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Use bbox as (north, south, east, west)
        G = ox.graph_from_bbox(
            BBOX['north'], BBOX['south'], BBOX['east'], BBOX['west'],
            network_type='drive',
            simplify=True
        )
        ox.save_graphml(G, cache_path)
        print(f"Saved road network to cache: {cache_path}")
    
    return G

def match_trajectory(fused_df: pd.DataFrame, G=None) -> pd.DataFrame:
    """
    Takes a DataFrame containing 'pos_lat' and 'pos_lon' and maps each point
    to the nearest edge on the road network graph G.
    Appends 'snapped_lat' and 'snapped_lon' to the returned DataFrame.
    """
    if G is None:
        G = load_or_download_graph()

    # Convert nodes and edges to GeoDataFrames
    # This allows for nearest edge snapping
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    latitudes = fused_df['pos_lat'].values
    longitudes = fused_df['pos_lon'].values

    snapped_lats = np.zeros_like(latitudes)
    snapped_lons = np.zeros_like(longitudes)

    # osmnx has a nearest_edges function that is vectorized
    # X corresponds to longitude, Y to latitude
    nearest_edges = ox.nearest_edges(G, longitudes, latitudes)

    # Now get the geometry for each nearest edge to find the exact snapped point.
    # A simpler approximation for large trajectories without full HMM map-matching:
    # Just project the point onto the LineString of the nearest edge.
    
    # Pre-build a dictionary or fast lookup for edge geometries
    # nearest_edges returns a tuple (u, v, key)
    # We iterate over points and their corresponding nearest edge
    
    import shapely.geometry as geom
    
    for i in range(len(fused_df)):
        edge_tuple = nearest_edges[i]
        # u, v, key = edge_tuple[0], edge_tuple[1], edge_tuple[2]
        
        # Access edge data
        try:
            edge_data = G.get_edge_data(edge_tuple[0], edge_tuple[1], edge_tuple[2])
            line = edge_data.get('geometry')
            
            if line is None:
                # If geometry is missing, it's a straight line between u and v
                u_node = G.nodes[edge_tuple[0]]
                v_node = G.nodes[edge_tuple[1]]
                line = geom.LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])

            point = geom.Point(longitudes[i], latitudes[i])
            
            # Project the point onto the line
            projected_dist = line.project(point)
            snapped_point = line.interpolate(projected_dist)
            
            snapped_lons[i] = snapped_point.x
            snapped_lats[i] = snapped_point.y
        except Exception as e:
            # Fallback to original point if mapping fails
            snapped_lats[i] = latitudes[i]
            snapped_lons[i] = longitudes[i]

    out_df = fused_df.copy()
    out_df['snapped_lat'] = snapped_lats
    out_df['snapped_lon'] = snapped_lons

    return out_df

if __name__ == "__main__":
    # Test the map-matching logic on the demo output
    import time
    
    df_path = "ekf_fusion/output/IDRFusedEstimate_S1_blackout_test.csv"
    if os.path.exists(df_path):
        print(f"Testing map matching on {df_path}")
        df = pd.read_csv(df_path)
        
        # Downsample for quick testing if needed, but let's do full
        # df = df.iloc[::10].reset_index(drop=True)
        
        start_t = time.time()
        G = load_or_download_graph()
        matched_df = match_trajectory(df, G)
        end_t = time.time()
        
        print(f"Map matching completed in {end_t - start_t:.2f} seconds.")
        print(matched_df[['timestamp_s', 'pos_lat', 'pos_lon', 'snapped_lat', 'snapped_lon']].head())
        
        out_path = df_path.replace(".csv", "_snapped.csv")
        matched_df.to_csv(out_path, index=False)
        print(f"Saved matched trajectory to {out_path}")
    else:
        print(f"File not found: {df_path}")
