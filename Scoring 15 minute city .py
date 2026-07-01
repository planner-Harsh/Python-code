import warnings
import os
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox
from shapely.geometry import box

# ---------------------------------------------------------------------------
# CORE CONFIGURATION & COMPLETE 8-PILLAR OSM QUERY MATRIX
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

URBAN_PILLARS = {
    "Active Life": {"leisure": ["sports_centre", "fitness_centre"], "amenity": "gym"},
    "Food & Shopping": {"shop": ["supermarket", "convenience", "bakery"], "amenity": "marketplace"},
    "Education & Childcare": {"amenity": ["school", "kindergarten", "university", "college"]},
    "Healthcare": {"amenity": ["pharmacy", "doctors", "hospital", "clinic"]},
    "Public & Green Space": {"leisure": ["park", "garden", "playground"], "landuse": "recreation_ground"},
    "Daily Services": {"amenity": ["bank", "post_office", "atm", "townhall"]},
    "Culture & Community": {"amenity": ["library", "community_centre", "place_of_worship"], "tourism": "museum"},
    "Public Transit Stops": {"highway": "bus_stop", "railway": ["station", "tram_stop", "subway_entrance"]}
}

PILLAR_COLORS = {
    "Active Life": "#00d2ff",            # Cyan
    "Food & Shopping": "#ffff00",        # Yellow
    "Education & Childcare": "#ff9900",  # Orange
    "Healthcare": "#00ff66",             # Neon Green
    "Public & Green Space": "#33ff33",   # Deep Green
    "Daily Services": "#ff00ff",         # Magenta
    "Culture & Community": "#9933ff",    # Purple
    "Public Transit Stops": "#ff3333"    # Neon Red
}

# ---------------------------------------------------------------------------
# UPGRADED CUMULATIVE MASTER RANKING PALETTE (0 TO 8 SCORE)
# ---------------------------------------------------------------------------
RANK_STYLES = {
    0: {"color": "#222222", "linewidth": 0.4, "alpha": 0.4, "label": "0/8 Aspects - Complete Isolation"},
    1: {"color": "#ff3333", "linewidth": 0.6, "alpha": 0.6, "label": "1-2/8 Aspects - Severe Service Gaps"},
    2: {"color": "#ff3333", "linewidth": 0.6, "alpha": 0.6, "label": ""}, 
    3: {"color": "#ff9900", "linewidth": 1.0, "alpha": 0.7, "label": "3-4/8 Aspects - Limited Coverage"},
    4: {"color": "#ff9900", "linewidth": 1.0, "alpha": 0.7, "label": ""},
    5: {"color": "#ffff00", "linewidth": 1.4, "alpha": 0.8, "label": "5-6/8 Aspects - Strong Walkability"},
    6: {"color": "#ffff00", "linewidth": 1.4, "alpha": 0.8, "label": ""},
    7: {"color": "#00d2ff", "linewidth": 2.0, "alpha": 0.9, "label": "7/8 Aspects - Highly Integrated Area"},
    8: {"color": "#00ff66", "linewidth": 2.8, "alpha": 1.0, "label": "8/8 Aspects - The Perfect 15-Min Street!"}
}

WALKING_RADIUS = 1000  # 1km (~12 minute walk metric threshold)
PROJECTED_CRS = 3857   # Web Mercator metric projection

# ---------------------------------------------------------------------------
# CARTOGRAPHIC RENDERING FUNCTIONS
# ---------------------------------------------------------------------------
def render_isolated_pillar(city_name, admin_polygon, street_segments, pillar_name, coverage_buffer, poi_points):
    """Generates an individual atlas sheet for a single urban aspect."""
    clean_city = city_name.lower().replace(" ", "_").replace(",", "")
    clean_pillar = pillar_name.lower().replace(" ", "_").replace("&", "and")
    file_name = f"{clean_city}_1_pillar_{clean_pillar}.png"
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="black")
    ax.set_facecolor("black")
    
    street_segments.plot(ax=ax, color="#161616", linewidth=0.5, alpha=0.4, zorder=1)
    
    if not coverage_buffer.is_empty:
        accessible_streets = street_segments[street_segments.geometry.centroid.within(coverage_buffer)]
        if not accessible_streets.empty:
            color = PILLAR_COLORS[pillar_name]
            accessible_streets.plot(ax=ax, color=color, linewidth=2.5, alpha=0.15, zorder=2)
            accessible_streets.plot(ax=ax, color=color, linewidth=0.7, alpha=0.8, zorder=3)
            
    if not poi_points.empty:
        poi_points.plot(ax=ax, color="white", markersize=5, alpha=0.8, zorder=4)
        
    ctx.add_basemap(ax=ax, crs=admin_polygon.crs.to_string(), source=ctx.providers.CartoDB.DarkMatter, attribution="")
    
    view_box = box(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
    view_gdf = gpd.GeoDataFrame(geometry=[view_box], crs=admin_polygon.crs)
    inverse_mask = gpd.overlay(view_gdf, admin_polygon, how="difference")
    inverse_mask.plot(ax=ax, color="black", alpha=1.0, zorder=5)
    
    ax.set_axis_off()
    ax.set_title(f"{city_name.upper()}\n15-MIN WALK ASPECT: {pillar_name.upper()}", color="white", fontsize=12, fontweight="bold", pad=20, loc="left", linespacing=1.5)
    plt.savefig(file_name, facecolor="black", dpi=150, bbox_inches="tight")
    plt.close()

def render_master_index(city_name, admin_polygon, ranked_streets):
    """Generates the final comprehensive score map across all combined infrastructure layers."""
    clean_city = city_name.lower().replace(" ", "_").replace(",", "")
    file_name = f"{clean_city}_2_master_score_ranking.png"
    
    print(f"   ==> Computing and drawing Final Master Ranking Index map: [ {file_name} ]")
    fig, ax = plt.subplots(figsize=(14, 14), facecolor="black")
    ax.set_facecolor("black")
    
    admin_polygon.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.0, zorder=1)
    
    # Layer street sections progressively from low to high scores for sharp stacking visualization
    for score in sorted(RANK_STYLES.keys()):
        style = RANK_STYLES[score]
        subset = ranked_streets[ranked_streets['walkability_score'] == score]
        if not subset.empty:
            if score >= 7:  # Add an aesthetic underlying neon glow to highly complete regions
                subset.plot(ax=ax, color=style["color"], linewidth=style["linewidth"]*3, alpha=0.15, zorder=2)
            subset.plot(ax=ax, color=style["color"], linewidth=style["linewidth"], alpha=style["alpha"], zorder=3)
            
    ctx.add_basemap(ax=ax, crs=admin_polygon.crs.to_string(), source=ctx.providers.CartoDB.DarkMatter, attribution="")
    
    view_box = box(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
    view_gdf = gpd.GeoDataFrame(geometry=[view_box], crs=admin_polygon.crs)
    inverse_mask = gpd.overlay(view_gdf, admin_polygon, how="difference")
    inverse_mask.plot(ax=ax, color="black", alpha=1.0, zorder=4)
    
    ax.set_axis_off()
    ax.set_title(f"{city_name.upper()} | 15-MINUTE CITY CUMULATIVE STREET INDEX", color="white", fontsize=14, fontweight="bold", pad=25, loc="left")
    
    # Construct clean compiled index legend elements
    legend_elements = [
        Line2D([0], [0], color=style["color"], linewidth=max(style["linewidth"], 2.0), label=style["label"])
        for score, style in RANK_STYLES.items() if style["label"] != ""
    ]
    leg = ax.legend(handles=legend_elements, loc="lower left", facecolor="black", edgecolor="#222222", labelcolor="white", fontsize=11, framealpha=1.0)
    leg.set_zorder(5)
    
    plt.savefig(file_name, facecolor="black", dpi=250, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# PIPELINE EXECUTION ENGINE LOOP
# ---------------------------------------------------------------------------
TARGET_CITIES = ["Zurich, Switzerland", "Budapest, Hungary"]

for selected_city in TARGET_CITIES:
    print(f"\n{'='*75}\nRUNNING URBAN ANALYSIS SUITE FOR: {selected_city}\n{'='*75}")
    
    try:
        city_boundary = ox.geocode_to_gdf(selected_city).to_crs(epsg=PROJECTED_CRS)
        
        print("-> Downloading road structural framework...")
        raw_streets = ox.graph_from_place(selected_city, network_type="drive")
        _, street_segments = ox.graph_to_gdfs(raw_streets, nodes=True, edges=True)
        street_segments = street_segments.to_crs(epsg=PROJECTED_CRS)
        
        street_centroids = street_segments.copy()
        street_centroids['geometry'] = street_centroids.geometry.centroid
        
        pillar_columns = []
        
        # 1. Evaluate and extract independent layers
        for pillar_name, query_tags in URBAN_PILLARS.items():
            print(f"-> Processing aspect layer: {pillar_name}")
            col_name = f"has_{pillar_name.lower().replace(' ', '_').replace('&', 'and')}"
            
            try:
                features = ox.features_from_place(selected_city, tags=query_tags)
                features = features.to_crs(epsg=PROJECTED_CRS)
                poi_points = features[features.geometry.type == "Point"]
                
                if not features.empty:
                    unified_buffer = features.geometry.centroid.buffer(WALKING_RADIUS).unary_union
                    street_segments[col_name] = street_centroids.geometry.within(unified_buffer)
                    render_isolated_pillar(selected_city, city_boundary, street_segments, pillar_name, unified_buffer, poi_points)
                else:
                    street_segments[col_name] = False
                    render_isolated_pillar(selected_city, city_boundary, street_segments, pillar_name, box(0,0,0,0), gpd.GeoDataFrame())
            except Exception:
                street_segments[col_name] = False
                render_isolated_pillar(selected_city, city_boundary, street_segments, pillar_name, box(0,0,0,0), gpd.GeoDataFrame())
                
            pillar_columns.append(col_name)
            
        # 2. Aggregation: Add scores together to determine street rank
        street_segments['walkability_score'] = street_segments[pillar_columns].sum(axis=1)
        
        # 3. Build Final Cumulative Dashboard Image
        render_master_index(selected_city, city_boundary, street_segments)
        
    except Exception as error:
        print(f"!!! Operational crash processing city matrix chains: {error}")

print(f"\n{'='*75}\nExecution successful! All individual maps and master profiles are saved.\n{'='*75}")