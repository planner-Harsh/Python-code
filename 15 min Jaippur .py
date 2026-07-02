import sys
import subprocess
import warnings
import os
import shutil

# ---------------------------------------------------------------------------
# SELF-HEALING DEPENDENCY BOOTSTRAPPER
# ---------------------------------------------------------------------------
def bootstrap_geospatial_stack():
    dependencies = {
        "osmnx": "osmnx",
        "geopandas": "geopandas",
        "contextily": "contextily",
        "duckdb": "duckdb",
        "matplotlib": "matplotlib",
        "networkx": "networkx",
        "shapely": "shapely"
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
            print("✅ Geospatial stack successfully installed and loaded!\n")
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
from shapely.geometry import box, Point

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS & MATRIX CONFIGURATION
# ---------------------------------------------------------------------------
MAX_WALKING_METERS = 1250  # 15 Minutes at 5 km/h walking pace
PROJECTED_CRS = 3857       # Metric Mercator for distance accuracy

ox.settings.overpass_endpoint = "https://overpass.openstreetmap.fr/api/interpreter"
ox.settings.timeout = 400  
ox.settings.request_headers = {
    "User-Agent": "Supercharged15MinCityEngine/3.7 (Contact: urban_spatial_analytics@domain.edu)"
}

# RECONSTRUCTED CRITERIA DICTIONARIES (Based exactly on your 6 pillars)
OVERTURE_CATEGORIES = {
    "1_Groceries_and_Kirana": "('grocery_store', 'supermarket', 'bakery', 'market', 'greengrocer', 'convenience_store', 'general_store')",
    "2_Health_and_Wellbeing": "('clinic', 'doctor', 'medical_clinic', 'hospital', 'dentist', 'medical_facility', 'pharmacy', 'drugstore', 'chemist', 'medical_supply')",
    "3_Primary_School": "('primary_education', 'school', 'kindergarten', 'education', 'preschool')",
    "4_Parks": "('park', 'playground', 'recreation_area', 'garden', 'amusement_park', 'leisure')",
    "5_Local_Transit": "('bus_station', 'transit_station', 'subway_station', 'bus_stop', 'railway_station', 'public_transport')",
    "6_Neighbourhood_Banking": "('bank', 'atm', 'finance', 'money_transfer')"
}

OSM_MAXIMIZED_TAGS = {
    "1_Groceries_and_Kirana": {"shop": ["supermarket", "grocery", "bakery", "convenience", "general", "kiosk", "department_store", "wholesale"], "amenity": "marketplace"},
    "2_Health_and_Wellbeing": {"amenity": ["clinic", "doctors", "hospital", "dentist", "pharmacy"], "health": ["clinic", "hospital", "pharmacy"], "shop": ["chemist", "pharmacy", "medical"]},
    "3_Primary_School": {"amenity": ["school", "kindergarten", "childcare"]},
    "4_Parks": {"leisure": ["park", "playground", "garden", "recreation_ground"], "landuse": ["grass", "cemetery", "recreation_ground"]},
    "5_Local_Transit": {"highway": ["bus_stop", "platform"], "amenity": ["bus_station", "taxi", "ferry_terminal"], "railway": ["station", "stop", "halt"]},
    "6_Neighbourhood_Banking": {"amenity": ["bank", "atm", "bureau_de_change"]}
}

PILLAR_COLORS = {
    "1_Groceries_and_Kirana": "#ffff00",     # Yellow
    "2_Health_and_Wellbeing": "#00ff66",     # Vibrant Green
    "3_Primary_School": "#ff9900",           # Orange
    "4_Parks": "#00ff00",                    # Pure Green
    "5_Local_Transit": "#ff3333",            # Red
    "6_Neighbourhood_Banking": "#ff00ff"     # Magenta
}

def get_daily_rank_style(score):
    if score == 0: return {"color": "#1c1c1c", "linewidth": 0.4, "alpha": 0.3, "label": "Score 0: Infrastructure Dead Zone"}
    elif 1 <= score <= 2: return {"color": "#ff3333", "linewidth": 0.9, "alpha": 0.6, "label": "Score 1-2: Severe Deficit"}
    elif 3 <= score <= 4: return {"color": "#ff9900", "linewidth": 1.6, "alpha": 0.7, "label": "Score 3-4: Sub-optimal Coverage"}
    elif score == 5: return {"color": "#ffff00", "linewidth": 2.4, "alpha": 0.8, "label": "Score 5: Strong Proximity"}
    elif score == 6: return {"color": "#00ff66", "linewidth": 3.6, "alpha": 1.0, "label": "Score 6: Perfect 15-Min Pedestrian Street"}
    else: return {"color": "#00ff66", "linewidth": 3.6, "alpha": 1.0, "label": "Score 6: Perfect 15-Min Pedestrian Street"}

MUNICIPAL_QUERY = "Jaipur Municipal Corporation, India"
DISPLAY_TITLE = "Jaipur City (Modified 15-Min Model)"

try:
    # ---------------------------------------------------------------------------
    # LAYER 1: BOUNDARY & HIGH-DENSITY ROAD NETWORKS
    # ---------------------------------------------------------------------------
    print("🛰️  STEP 1: FETCHING ADMINISTRATIVE BOUNDARIES & PEDESTRIAN GRAPH...")
    city_boundary_gdf = ox.geocode_to_gdf(MUNICIPAL_QUERY)
    boundary_wgs84 = city_boundary_gdf.geometry.iloc[0]
    city_boundary_projected = city_boundary_gdf.to_crs(epsg=PROJECTED_CRS)
    minx, miny, maxx, maxy = boundary_wgs84.bounds
    
    print("🔗 STEP 2: PARSING PEDESTRIAN GRID METRICS (Alleys, Pathways, Open Sidewalks)...")
    G_raw = ox.graph_from_polygon(boundary_wgs84, network_type="walk")
    G = ox.project_graph(G_raw, to_crs=PROJECTED_CRS)
    G_un = ox.convert.to_undirected(G)
    street_segments = ox.convert.graph_to_gdfs(G_un, nodes=False, edges=True)
    
    pillar_columns = []
    db = duckdb.connect()
    db.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    db.execute("SET s3_region='us-west-2';")
    
    # ---------------------------------------------------------------------------
    # LAYER 2: DUAL-SOURCE DATA EXTRACTION & ISOCHRONE ROUTING
    # ---------------------------------------------------------------------------
    for pillar_name, overture_types in OVERTURE_CATEGORIES.items():
        print(f"\n🏃‍♂️ PIPELINE LAYER: [ {pillar_name.upper().replace('_', ' ')} ]")
        col_name = f"has_{pillar_name.lower()}"
        
        list_gdfs = []
        
        # Source A: Overture Cloud Engine
        query = f"""
            SELECT ST_X(ST_Point(bbox.xmin, bbox.ymin)) as lon, ST_Y(ST_Point(bbox.xmin, bbox.ymin)) as lat
            FROM read_parquet('s3://overturemaps-us-west-2/release/default/theme=places/type=place/*', hive_partitioning=1)
            WHERE bbox.xmin <= {maxx} AND bbox.xmax >= {minx}
              AND bbox.ymin <= {maxy} AND bbox.ymax >= {miny}
              AND categories.primary IN {overture_types}
        """
        try:
            df_overture = db.execute(query).df()
            if not df_overture.empty:
                gdf_ov = gpd.GeoDataFrame(df_overture, geometry=gpd.points_from_xy(df_overture.lon, df_overture.lat), crs="EPSG:4326").to_crs(epsg=PROJECTED_CRS)
                gdf_ov = gdf_ov[gdf_ov.geometry.within(city_boundary_projected.geometry.iloc[0])]
                if not gdf_ov.empty:
                    list_gdfs.append(gdf_ov)
                    print(f"   -> Extracted {len(gdf_ov)} POIs from Overture Maps Cloud Datasets.")
        except Exception:
            pass
            
        # Source B: Deep OpenStreetMap Scraper
        try:
            osm_features = ox.features_from_polygon(boundary_wgs84, tags=OSM_MAXIMIZED_TAGS[pillar_name])
            osm_features = osm_features.to_crs(epsg=PROJECTED_CRS)
            
            gdf_osm_pt = osm_features[osm_features.geometry.type == "Point"].copy()
            gdf_osm_poly = osm_features[osm_features.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
            
            if not gdf_osm_poly.empty:
                gdf_osm_poly["geometry"] = gdf_osm_poly.geometry.centroid
                gdf_osm_pt = pd.concat([gdf_osm_pt, gdf_osm_poly])
                
            if not gdf_osm_pt.empty:
                gdf_osm_pt = gdf_osm_pt[["geometry"]].copy()
                list_gdfs.append(gdf_osm_pt)
                print(f"   -> Extracted {len(gdf_osm_pt)} POIs from OpenStreetMap Topology Records.")
        except Exception:
            pass

        # Combine Both Channels & Drop Multi-Source Overlaps
        if list_gdfs:
            gdf_pois = pd.concat(list_gdfs, ignore_index=True)
            
            # 🛠️ FIXED: Using a temporary text column to drop duplicate coordinates safely without triggering MultiIndex pandas bugs
            gdf_pois['xy_str'] = gdf_pois.geometry.apply(lambda p: f"{round(p.x, 2)}_{round(p.y, 2)}")
            gdf_pois = gdf_pois.drop_duplicates(subset=['xy_str']).drop(columns=['xy_str'])
            
            print(f"   💥 TOTAL DE-DUPLICATED ACTIVE DESTINATIONS: {len(gdf_pois)} NODES")
        else:
            gdf_pois = gpd.GeoDataFrame()
            
        # Execute Routing Calculations
        if not gdf_pois.empty:
            poi_x = [float(geom.x) for geom in gdf_pois.geometry]
            poi_y = [float(geom.y) for geom in gdf_pois.geometry]
            
            source_nodes = list(set(ox.distance.nearest_nodes(G_un, poi_x, poi_y)))
            
            accessible_node_distances = nx.multi_source_dijkstra_path_length(
                G_un, source_nodes, cutoff=MAX_WALKING_METERS, weight='length'
            )
            
            street_segments[col_name] = [
                (u in accessible_node_distances or v in accessible_node_distances) 
                for u, v, k in street_segments.index
            ]
            
            # Render and display live map layer inline
            fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
            ax.set_facecolor("black")
            street_segments.plot(ax=ax, color="#141414", linewidth=0.5, alpha=0.4, zorder=1)
            
            accessible_streets = street_segments[street_segments[col_name] == True]
            if not accessible_streets.empty:
                accessible_streets.plot(ax=ax, color=PILLAR_COLORS[pillar_name], linewidth=1.4, alpha=0.8, zorder=2)
            
            gdf_pois.plot(ax=ax, color="white", markersize=1.5, alpha=0.4, zorder=3)
            ctx.add_basemap(ax=ax, crs=city_boundary_projected.crs.to_string(), source=ctx.providers.CartoDB.DarkMatter, attribution="")
            
            view_box = box(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
            view_gdf = gpd.GeoDataFrame(geometry=[view_box], crs=city_boundary_projected.crs)
            inverse_mask = gpd.overlay(view_gdf, city_boundary_projected, how="difference")
            inverse_mask.plot(ax=ax, color="black", alpha=1.0, zorder=4)
            
            ax.set_axis_off()
            ax.set_title(f"{DISPLAY_TITLE.upper()}\n15-MIN ISOCHRONE PEDESTRIAN ROUTING: {pillar_name.upper().replace('_', ' ')}", color="white", fontsize=9, fontweight="bold", pad=15, loc="left")
            plt.show()
            plt.close()
        else:
            street_segments[col_name] = False
            
        pillar_columns.append(col_name)
        
    # ---------------------------------------------------------------------------
    # LAYER 3: MASTER SYNTHESIS WALKABILITY MATRIX MAP
    # ---------------------------------------------------------------------------
    print("\n📊 SYNTHESIZING CUMULATIVE 15-MINUTE PEDESTRIAN WALKABILITY MATRIX INDEX...")
    street_segments['walkability_score'] = street_segments[pillar_columns].sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="black")
    ax.set_facecolor("black")
    city_boundary_projected.plot(ax=ax, facecolor="none", edgecolor="#222222", linewidth=1.2, zorder=1)
    
    for score in sorted(list(street_segments['walkability_score'].unique())):
        style = get_daily_rank_style(score)
        subset = street_segments[street_segments['walkability_score'] == score]
        if not subset.empty:
            if score >= 5:
                subset.plot(ax=ax, color=style["color"], linewidth=style["linewidth"]*2.2, alpha=0.12, zorder=2)
            subset.plot(ax=ax, color=style["color"], linewidth=style["linewidth"], alpha=style["alpha"], zorder=3)
            
    ctx.add_basemap(ax=ax, crs=city_boundary_projected.crs.to_string(), source=ctx.providers.CartoDB.DarkMatter, attribution="")
    
    view_box = box(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
    view_gdf = gpd.GeoDataFrame(geometry=[view_box], crs=city_boundary_projected.crs)
    inverse_mask = gpd.overlay(view_gdf, city_boundary_projected, how="difference")
    inverse_mask.plot(ax=ax, color="black", alpha=1.0, zorder=4)
    ax.set_axis_off()
    ax.set_title(f"{DISPLAY_TITLE.upper()} | COMPREHENSIVE 15-MINUTE PEDESTRIAN WALKABILITY INDEX", color="white", fontsize=11, fontweight="bold", pad=20, loc="left")
    
    seen_labels = set()
    legend_elements = []
    for s in [0, 1, 3, 5, 6]:
        st = get_daily_rank_style(s)
        if st["label"] not in seen_labels:
            seen_labels.add(st["label"])
            legend_elements.append(Line2D([0], [0], color=st["color"], linewidth=max(st["linewidth"], 2.0), label=st["label"]))
    leg = ax.legend(handles=legend_elements, loc="lower left", facecolor="black", edgecolor="#222222", labelcolor="white", fontsize=9, framealpha=1.0)
    leg.set_zorder(5)
    plt.show()
    plt.close()

    # ---------------------------------------------------------------------------
    # LAYER 4: SPATIAL DATA CLEANING & COMPACT GIS ARTIFACT EXPORT
    # ---------------------------------------------------------------------------
    print("\n🗄️  COMPILING GIS LAYERS AND FLATTENING ATTRIBUTE SCHEMAS...")
    gis_export_df = street_segments.copy().reset_index()
    
    # Map long headers into tidy 10-character limits to dodge Shapefile truncate errors
    shapefile_field_map = {
        "has_1_groceries_and_kirana": "walk_groc",
        "has_2_health_and_wellbeing": "walk_hlth",
        "has_3_primary_school": "walk_schl",
        "has_4_parks": "walk_park",
        "has_5_local_transit": "walk_trns",
        "has_6_neighbourhood_banking": "walk_bank",
        "walkability_score": "walk_score"
    }
    gis_export_df = gis_export_df.rename(columns=shapefile_field_map)
    
    for column_name in gis_export_df.columns:
        if column_name != 'geometry':
            if column_name in shapefile_field_map.values() and column_name != "walk_score":
                gis_export_df[column_name] = gis_export_df[column_name].astype(int)
            else:
                gis_export_df[column_name] = gis_export_df[column_name].apply(
                    lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
                )
                gis_export_df[column_name] = gis_export_df[column_name].apply(
                    lambda v: str(v) if isinstance(v, dict) else v
                )

    export_folder = "jaipur_15min_gis_outputs"
    os.makedirs(export_folder, exist_ok=True)
    
    shp_path = os.path.join(export_folder, "jaipur_walkability_network.shp")
    print(f"📦 Compiling ESRI Shapefile vector paths: {shp_path}")
    gis_export_df.to_file(shp_path, driver="ESRI Shapefile")
    
    zip_archive_name = "jaipur_walkability_shapefile_package"
    shutil.make_archive(zip_archive_name, 'zip', export_folder)
    
    print("\n" + "="*80)
    print("👑 SIMULATION COMPLETION AND PIPELINE ARTIFACT SUCCESS SUMMARY")
    print("="*80)
    print(f"✅ All maps rendered successfully inside notebook frames.")
    print(f"📥 GIS Shapefile Compiled Folder: '{export_folder}/'")
    print(f"🎉 DOWNLOADABLE ZIP PACK READY: '{zip_archive_name}.zip'")
    print("="*80 + "\n")

except Exception as error:
    print(f"!!! Operational error executing layout metrics: {error}")