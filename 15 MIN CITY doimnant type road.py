import warnings
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox
from shapely.geometry import box
import city2graph as c2g

# ---------------------------------------------------------------------------
# CORE CONFIGURATION & PILLARS OF A 15-MINUTE CITY
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

# Define target urban points of interest (POIs) categorized by spatial pillars
POI_CATEGORIES = {
    "Active Life": {"leisure": "sports_centre", "amenity": "gym"},
    "Daily Needs": {"shop": "supermarket", "amenity": "marketplace"},
    "Social Life": {"amenity": ["cafe", "restaurant", "bar", "pub"]},
    "Healthcare": {"amenity": ["pharmacy", "doctors", "hospital"]}
}

LAYER_COLORS = ["#00d2ff", "#ffff00", "#ff007f", "#00ff66"] 
WALKING_RADIUS = 1000  # 1,000 Meters constraint (~12 minute walk)
PROJECTED_CRS = 3857   # Web Mercator metric projection system

# ---------------------------------------------------------------------------
# MAP MAKER RENDERING ENGINE
# ---------------------------------------------------------------------------
def plot_city_connectivity(city_title, admin_polygon, poi_layers, graph_layers):
    """
    Assembles the final black-out cartographic layout with neon glowing edges,
    bounding canvas constraints, and an inverted background polygon mask.
    """
    print(f"-> Drawing spatial map layout canvas for {city_title}...")
    fig, ax = plt.subplots(figsize=(14, 14), facecolor="black")
    ax.set_facecolor("black")
    
    # Backdrop Layer: Display administrative city outline lines faintly
    admin_polygon.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=1.2, zorder=1)
    
    all_x_min, all_y_min, all_x_max, all_y_max = [], [], [], []
    
    # Process and render graph vectors layer-by-layer
    for idx, category in enumerate(POI_CATEGORIES.keys()):
        edges_gdf = graph_layers[category]
        nodes_gdf = poi_layers[category]
        color = LAYER_COLORS[idx]
        
        if not edges_gdf.empty:
            bounds = edges_gdf.total_bounds
            all_x_min.append(bounds[0]); all_y_min.append(bounds[1])
            all_x_max.append(bounds[2]); all_y_max.append(bounds[3])
            
            # --- NEON GLOW DESIGN ---
            # Broad base line providing soft luminescence
            edges_gdf.plot(ax=ax, color=color, linewidth=3.5, alpha=0.22, zorder=2)
            # Sharp core line focused perfectly over the base
            edges_gdf.plot(ax=ax, color=color, linewidth=0.8, alpha=0.9, zorder=3)
            
        if not nodes_gdf.empty:
            nodes_gdf.plot(ax=ax, color="white", markersize=14, edgecolor=color, linewidth=0.7, zorder=4)

    if all_x_min:
        min_x, min_y = min(all_x_min), min(all_y_min)
        max_x, max_y = max(all_x_max), max(all_y_max)
        x_span, y_span = max_x - min_x, max_y - min_y
        
        ax.set_xlim(min_x - (x_span * 0.05), max_x + (x_span * 0.05))
        ax.set_ylim(min_y - (y_span * 0.05), max_y + (y_span * 0.05))
    
    ctx.add_basemap(
        ax=ax, 
        crs=admin_polygon.crs.to_string(), 
        source=ctx.providers.CartoDB.DarkMatter, 
        attribution=""
    )
    
    # --- INVERSE MASK GENERATION ---
    view_box = box(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
    view_gdf = gpd.GeoDataFrame(geometry=[view_box], crs=admin_polygon.crs)
    inverse_mask = gpd.overlay(view_gdf, admin_polygon, how="difference")
    inverse_mask.plot(ax=ax, color="black", alpha=1.0, zorder=5)
    
    ax.set_axis_off()
    ax.set_title(
        f"{city_title.upper()} | 15-MINUTE WALKING GEOMETRY", 
        color="white", fontsize=15, fontweight="bold", pad=25, loc="left"
    )
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label=cat,
               markerfacecolor='white', markeredgecolor=LAYER_COLORS[i], markersize=9)
        for i, cat in enumerate(POI_CATEGORIES.keys())
    ]
    leg = ax.legend(
        handles=legend_elements, loc="lower left", facecolor="black", 
        edgecolor="#222222", labelcolor="white", fontsize=11, framealpha=1.0
    )
    leg.set_zorder(6)  
    
    clean_title = city_title.lower().replace(" ", "_").replace(",", "")
    output_path = f"walkability_map_{clean_title}.png"
    plt.savefig(output_path, facecolor="black", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"==> Exported completed visual asset: {output_path}")

# ---------------------------------------------------------------------------
# MAIN TRANSIT PROCESSING PIPELINE RUNNER
# ---------------------------------------------------------------------------
TARGET_CITIES = ["Budapest, Hungary", "Zurich, Switzerland"]

for selected_city in TARGET_CITIES:
    print(f"\n{'='*60}\nLaunching Network Processing Suite For: {selected_city}\n{'='*60}")
    
    try:
        city_boundary = ox.geocode_to_gdf(selected_city).to_crs(epsg=PROJECTED_CRS)
        
        print("-> Downloading driving infrastructure street grids...")
        raw_streets = ox.graph_from_place(selected_city, network_type="drive")
        _, street_segments = ox.graph_to_gdfs(raw_streets, nodes=True, edges=True)
        street_segments = street_segments.to_crs(epsg=PROJECTED_CRS)
        
        poi_storage = {}
        graph_storage = {}
        
        for pillar_name, query_tags in POI_CATEGORIES.items():
            print(f"-> Processing target layer subset: [{pillar_name}]")
            
            features = ox.features_from_place(selected_city, tags=query_tags)
            features = features[features.geometry.type == "Point"].to_crs(epsg=PROJECTED_CRS)
            poi_storage[pillar_name] = features
            
            if not features.empty:
                # FIXED: first argument passed positionally, 'radius' keyword updated to 'r0'
                nodes, edges = c2g.waxman_graph(
                    features, 
                    distance_metric="network", 
                    r0=WALKING_RADIUS, 
                    beta=0.5, 
                    network_gdf=street_segments
                )
                graph_storage[pillar_name] = edges
            else:
                graph_storage[pillar_name] = gpd.GeoDataFrame(crs=PROJECTED_CRS, geometry=[])
        
        plot_city_connectivity(selected_city, city_boundary, poi_storage, graph_storage)
        
    except Exception as error:
        print(f"!!! Error encountered parsing data pipeline for {selected_city}: {error}")
        
print("\nProcessing workflow sequence ended successfully.")