import sys
import subprocess
import warnings
import os
import shutil
import random

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT BOOTSTRAPPER (AUTOMATED GEOSPATIAL STACK)
# ---------------------------------------------------------------------------
def bootstrap_geospatial_stack():
    dependencies = {
        "osmnx": "osmnx",
        "geopandas": "geopandas",
        "contextily": "contextily",
        "duckdb": "duckdb",
        "matplotlib": "matplotlib",
        "networkx": "networkx",
        "shapely": "shapely",
        "pandas": "pandas",
        "numpy": "numpy"
    }
    
    missing_packages = []
    for module_name, pip_name in dependencies.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        print(f"📦 Environment missing packages. Initializing automatic installation of: {missing_packages}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages, stdout=subprocess.DEVNULL)
            print("✅ Cycling Analytics stack successfully installed and loaded!\n")
        except Exception as e:
            print(f"⚠️ Automated installer warning: {e}")

bootstrap_geospatial_stack()

import osmnx as ox
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import duckdb
import pandas as pd
import numpy as np
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 2. GLOBAL SIMULATION ENGINE CONFIGURATION
# ---------------------------------------------------------------------------
LOCAL_RADIUS_METERS = 2000     # Expanded neighborhood envelope (~8 min ride)
COMMUTER_RADIUS_METERS = 6000  # Broad cross-city corridor envelope (~25 min ride)
PROJECTED_CRS = 3857           # Web Mercator for accurate metric routing
NUM_SIMULATION_TRIPS = 1000    # Boosted to 1000 to thoroughly illuminate "blank roads"

ox.settings.overpass_endpoint = "https://overpass.openstreetmap.fr/api/interpreter"
ox.settings.timeout = 600  
ox.settings.request_headers = {
    "User-Agent": "JaipurCyclingSuperchargedEngine/2.0 (Contact: urban_mobility_analytics@domain.edu)"
}

# India-Specific Level of Traffic Stress (LTS) Framework
LTS_CONFIG = {
    "motorway": {"lts": 4, "penalty": 10.0}, 
    "trunk":    {"lts": 4, "penalty": 5.0},  
    "primary":  {"lts": 4, "penalty": 3.5},  
    "secondary":{"lts": 3, "penalty": 2.0},  
    "tertiary": {"lts": 2, "penalty": 1.2},  
    "residential": {"lts": 1, "penalty": 1.0},
    "living_street": {"lts": 1, "penalty": 1.0},
    "service":  {"lts": 1, "penalty": 1.0},
    "cycleway": {"lts": 1, "penalty": 0.8}   
}

# ---------------------------------------------------------------------------
# 3. TARGETED MUNICIPAL BOUNDARY & STREET NETWORK INGESTION
# ---------------------------------------------------------------------------
MUNICIPAL_QUERY = "Jaipur Municipal Corporation, Rajasthan, India"
DISPLAY_TITLE = "Jaipur Nagar Nigam - Active Mobility Footprint"

try:
    print("🛰️  PHASE 1: ENFORCING JAIPUR NAGAR NIGAM MUNICIPAL BOUNDARY...")
    try:
        city_boundary_gdf = ox.geocode_to_gdf(MUNICIPAL_QUERY)
    except Exception:
        print("⚠️ Direct Nagar Nigam boundary lookup failed. Deploying urban core fallback...")
        city_boundary_gdf = ox.geocode_to_gdf("Jaipur, Jaipur Tehsil, Rajasthan, India")
        
    boundary_wgs84 = city_boundary_gdf.geometry.iloc[0]
    city_boundary_projected = city_boundary_gdf.to_crs(epsg=PROJECTED_CRS)
    minx, miny, maxx, maxy = boundary_wgs84.bounds
    
    print("🔗 PHASE 2: PARSING CYCLABLE STREET NETWORK TOPOLOGY...")
    G_raw = ox.graph_from_polygon(boundary_wgs84, network_type="all")
    G_proj = ox.project_graph(G_raw, to_crs=PROJECTED_CRS)
    G_un = ox.convert.to_undirected(G_proj)
    
    print("🚴‍♂️ PHASE 3: CALCULATING COALESCED TRAFFIC STRESS IMPEDANCE...")
    for u, v, k, data in G_un.edges(keys=True, data=True):
        highway_type = data.get("highway", "residential")
        if isinstance(highway_type, list): highway_type = highway_type[0]
            
        config = LTS_CONFIG.get(highway_type, {"lts": 2, "penalty": 1.2})
        data["lts_score"] = config["lts"]
        data["cycling_impedance"] = float(data.get("length", 1.0)) * config["penalty"]

    street_segments = ox.convert.graph_to_gdfs(G_un, nodes=False, edges=True)
    
    db = duckdb.connect()
    db.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    db.execute("SET s3_region='us-west-2';")
    
    # ---------------------------------------------------------------------------
    # 4. SUPERCHARGED CATCH-ALL POI HARVESTING ENGINE
    # ---------------------------------------------------------------------------
    # Using Boolean 'True' values extracts EVERY store, amenity, and office 
    # regardless of vague crowdsourced descriptions (e.g., shop=yes).
    OSM_LOCAL_TAGS = {
        "shop": True,              
        "amenity": True,           
        "leisure": True,           
        "public_transport": True,  
        "craft": True              
    }

    OSM_COMMUTER_TAGS = {
        "office": True,            
        "industrial": True,        
        "building": ["commercial", "office", "retail", "university", "college", "public", "hospital"]
    }
    
    OVERTURE_LOCAL = "('grocery', 'market', 'food_and_beverage', 'retail', 'health_and_medical', 'community_and_government', 'religion')"
    OVERTURE_COMMUTER = "('professional_services', 'business_and_b2b', 'government', 'higher_education', 'shopping_mall')"

    def harvest_pois(overture_filter, osm_tags, label):
        print(f"📥 Gathering SUPERCHARGED destinations for: [ {label} ]")
        list_gdfs = []
        
        # Channel A: Cloud Native Overture Parquet Engine
        query = f"""
            SELECT ST_X(ST_Point(bbox.xmin, bbox.ymin)) as lon, ST_Y(ST_Point(bbox.xmin, bbox.ymin)) as lat
            FROM read_parquet('s3://overturemaps-us-west-2/release/default/theme=places/type=place/*', hive_partitioning=1)
            WHERE bbox.xmin <= {maxx} AND bbox.xmax >= {minx}
              AND bbox.ymin <= {maxy} AND bbox.ymax >= {miny}
              AND categories.primary IN {overture_filter}
        """
        try:
            df_overture = db.execute(query).df()
            if not df_overture.empty:
                gdf_ov = gpd.GeoDataFrame(df_overture, geometry=gpd.points_from_xy(df_overture.lon, df_overture.lat), crs="EPSG:4326").to_crs(epsg=PROJECTED_CRS)
                gdf_ov = gdf_ov[gdf_ov.geometry.within(city_boundary_projected.geometry.iloc[0])]
                if not gdf_ov.empty: list_gdfs.append(gdf_ov)
        except Exception: pass
            
        # Channel B: OpenStreetMap Spatial Catch-All Engine
        try:
            osm_features = ox.features_from_polygon(boundary_wgs84, tags=osm_tags)
            osm_features = osm_features.to_crs(epsg=PROJECTED_CRS)
            
            # COLLAPSE METHOD: Converts large areas/polygons (like campuses) into clean routing centroids
            osm_features['geometry'] = osm_features.geometry.centroid
            if not osm_features.empty:
                list_gdfs.append(osm_features[["geometry"]].copy())
        except Exception: pass
            
        if list_gdfs:
            combined = pd.concat(list_gdfs, ignore_index=True)
            # Spatial Clustering: Snaps elements to a 10m grid to prevent commercial centers from overwhelming the engine
            combined['xy_str'] = combined.geometry.apply(lambda p: f"{round(p.x, -1)}_{round(p.y, -1)}")
            combined = combined.drop_duplicates(subset=['xy_str']).drop(columns=['xy_str'])
            print(f"   💥 Total High-Density Destination Nodes: {len(combined)}")
            return combined
        else:
            return gpd.GeoDataFrame()

    local_pois = harvest_pois(OVERTURE_LOCAL, OSM_LOCAL_TAGS, "LOCAL SCALE ERRANDS")
    commuter_pois = harvest_pois(OVERTURE_COMMUTER, OSM_COMMUTER_TAGS, "COMMUTER LONG-DISTANCE ANCHORS")

    # ---------------------------------------------------------------------------
    # 5. HIGH-DENSITY MONTE CARLO TRAFFIC PATH SIMULATOR
    # ---------------------------------------------------------------------------
    print("\n🎲 INITIALIZING HIGH-DENSITY TRAFFIC ROUTING ENGINE...")
    street_segments["pot_local"] = 0.0
    street_segments["pot_commute"] = 0.0
    
    network_nodes = list(G_un.nodes())
    sampled_origins = random.sample(network_nodes, min(len(network_nodes), NUM_SIMULATION_TRIPS))
    
    def get_nearest_node_list(gdf_pois):
        if gdf_pois.empty: return []
        px = [float(g.x) for g in gdf_pois.geometry]
        py = [float(g.y) for g in gdf_pois.geometry]
        return list(set(ox.distance.nearest_nodes(G_un, px, py)))

    local_target_nodes = get_nearest_node_list(local_pois)
    commuter_target_nodes = get_nearest_node_list(commuter_pois)

    edge_local_counts = {edge_idx: 0 for edge_idx in street_segments.index}
    edge_comm_counts = {edge_idx: 0 for edge_idx in street_segments.index}

    print(f"⏳ Processing {NUM_SIMULATION_TRIPS} Matrix Journeys Across Jaipur's Grid...")
    for origin in sampled_origins:
        # Route neighborhood utility trips
        local_lengths, local_paths = nx.single_source_dijkstra(G_un, source=origin, cutoff=LOCAL_RADIUS_METERS, weight='cycling_impedance')
        for target_node, path in local_paths.items():
            if target_node in local_target_nodes and len(path) > 1:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (u, v, 0) in edge_local_counts: edge_local_counts[(u, v, 0)] += 1
                    elif (v, u, 0) in edge_local_counts: edge_local_counts[(v, u, 0)] += 1

        # Route cross-city structural commutes
        comm_lengths, comm_paths = nx.single_source_dijkstra(G_un, source=origin, cutoff=COMMUTER_RADIUS_METERS, weight='cycling_impedance')
        for target_node, path in comm_paths.items():
            if target_node in commuter_target_nodes and len(path) > 1:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (u, v, 0) in edge_comm_counts: edge_comm_counts[(u, v, 0)] += 1
                    elif (v, u, 0) in edge_comm_counts: edge_comm_counts[(v, u, 0)] += 1

    street_segments["pot_local"] = street_segments.index.map(edge_local_counts).fillna(0.0)
    street_segments["pot_commute"] = street_segments.index.map(edge_comm_counts).fillna(0.0)
    street_segments["pot_total"] = street_segments["pot_local"] + street_segments["pot_commute"]

    # ---------------------------------------------------------------------------
    # 6. CARTOGRAPHIC PRODUCTION (CARTODB DARK MATTER MASK)
    # ---------------------------------------------------------------------------
    scales = {
        "pot_local": ("LOCAL UTILITY ERRAND CORRIDORS", "viridis"),
        "pot_commute": ("COMMUTER URBAN CYCLING BACKBONES", "plasma"),
        "pot_total": ("INTEGRATED SYNTHESIS MOVEMENT POTENTIAL MAP", "inferno")
    }

    for column_key, (map_title, cmap_choice) in scales.items():
        fig, ax = plt.subplots(figsize=(12, 12), facecolor="black")
        ax.set_facecolor("black")
        
        # Plot municipal boundary outline
        city_boundary_projected.plot(ax=ax, facecolor="none", edgecolor="#444444", linewidth=2.0, zorder=1)
        
        # Dynamic styling: roads with active flow are drawn wider and brighter
        max_val = street_segments[column_key].max() if street_segments[column_key].max() > 0 else 1.0
        linewidths = np.where(street_segments[column_key] > 0, 0.5 + (street_segments[column_key] / max_val) * 2.5, 0.3)
        alphas = np.where(street_segments[column_key] > 0, 0.9, 0.15)
        
        street_segments.plot(
            ax=ax, column=column_key, cmap=cmap_choice, 
            linewidth=linewidths, alpha=alphas, zorder=2
        )
        
        ctx.add_basemap(ax=ax, crs=city_boundary_projected.crs.to_string(), source=ctx.providers.CartoDB.DarkMatter, attribution="")
        
        # Invert mask to crop drawing strictly to Nagar Nigam territory boundaries
        view_box = box(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
        view_gdf = gpd.GeoDataFrame(geometry=[view_box], crs=city_boundary_projected.crs)
        inverse_mask = gpd.overlay(view_gdf, city_boundary_projected, how="difference")
        inverse_mask.plot(ax=ax, color="black", alpha=1.0, zorder=4)
        
        ax.set_axis_off()
        ax.set_title(f"{DISPLAY_TITLE.upper()}\n🔬 {map_title}", color="white", fontsize=12, fontweight="bold", pad=20, loc="left")
        
        legend_elements = [
            Line2D([0], [0], color=plt.get_cmap(cmap_choice)(0.2), linewidth=1, label="Low Potential"),
            Line2D([0], [0], color=plt.get_cmap(cmap_choice)(0.6), linewidth=2, label="Active Utility Network"),
            Line2D([0], [0], color=plt.get_cmap(cmap_choice)(0.95), linewidth=3, label="Critical Cycling Path")
        ]
        leg = ax.legend(handles=legend_elements, loc="lower left", facecolor="black", edgecolor="#222222", labelcolor="white", fontsize=9)
        leg.set_zorder(5)
        plt.show()
        plt.close()

    # ---------------------------------------------------------------------------
    # 7. METADATA WRITING & GIS COMPILATION EXPORT
    # ---------------------------------------------------------------------------
    print("\n🗄️  PHASE 6: COMPILING ENGINE OUTPUTS INTO SHAPEFILES...")
    gis_export_df = street_segments.copy().reset_index()
    shapefile_field_map = {
        "lts_score": "cyc_lts", 
        "cycling_impedance": "cyc_imped", 
        "pot_local": "pot_local", 
        "pot_commute": "pot_comm", 
        "pot_total": "pot_total"
    }
    gis_export_df = gis_export_df.rename(columns=shapefile_field_map)
    keep_cols = ["geometry"] + list(shapefile_field_map.values())
    gis_export_df = gis_export_df[[col for col in gis_export_df.columns if col in keep_cols]]
    
    for col in gis_export_df.columns:
        if col != 'geometry':
            gis_export_df[col] = gis_export_df[col].astype(int) if col == "cyc_lts" else gis_export_df[col].astype(float)
                
    export_folder = "jaipur_cycling_gis_outputs"
    os.makedirs(export_folder, exist_ok=True)
    gis_export_df.to_file(os.path.join(export_folder, "jaipur_cycling_potential.shp"), driver="ESRI Shapefile")
    shutil.make_archive("jaipur_cycling_potential_package", 'zip', export_folder)
    print("================================================================================")
    print("🎉 EXECUTION COMPLETE: High-density maps rendered and GIS packages compiled!")
    print("================================================================================")

except Exception as error:
    print(f"🚨 Pipeline critical failure: {error}")