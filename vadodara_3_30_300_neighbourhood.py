"""
=============================================================================
 3-30-300 RULE GEOSPATIAL ANALYSIS — VADODARA NEIGHBOURHOOD
=============================================================================
Evaluates building-level compliance with the 3-30-300 urban forestry rule
(Konijnendijk, 2021).

Pipeline order:
  1. BUILDINGS FIRST — loaded from your cleaned Vadodara_Buildings_Clean
     shapefile (buildings take priority over everything else). Saved + mapped
     on its own for verification before anything else runs.
  2. Canopy raster — already merged & clipped (/content/canopy_clipped.tif),
     thresholded to tree/non-tree (value > 3 = tree).
  3. Park / green-space data (OpenStreetMap).
  4. Rule 3 / 30 / 300 compliance analysis + maps.

Run this in the same environment where your files live (e.g. Colab, where
your files sit under /content/).

-----------------------------------------------------------------------------
INSTALL (run once, e.g. in a Colab cell with a leading '!'):
-----------------------------------------------------------------------------
pip install rasterio geopandas rasterstats shapely matplotlib \
            contextily mapclassify osmnx --quiet
=============================================================================
"""

import os
import gc
import glob
import shutil
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import Resampling
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION — edit these paths/values for your setup
# =============================================================================

# --- Building footprints: your cleaned shapefile ------------------------------
BUILDINGS_SHP_PATH = "/content/Vadodara_Buildings_Clean.shp"
# (the .cpg/.dbf/.fix/.prj/.shx sidecar files must sit alongside this .shp)

# --- Canopy raster: already merged & clipped ----------------------------------
CANOPY_TIF_IN = "/content/canopy_clipped.tif"

# --- Canopy classification rule -----------------------------------------------
# Only pixel values ABOVE this count as tree canopy. Set to None if the raster
# is ALREADY a binary 0/1 tree mask and no thresholding is needed.
CANOPY_ABOVE_VALUE = 3

# --- Output folder ------------------------------------------------------------
OUTPUT_DIR = "/content/outputs_3_30_300_neighbourhood"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Park/garden data: your Google My Maps CSV export -------------------------
PARK_CSV_PATH = "/content/Untitled map- Untitled layer.csv"
POINT_PARK_BUFFER_M = 15   # nominal radius for park features that are just points

# --- Rule thresholds -----------------------------------------------------------
BUFFER_M              = 30      # buffer distance (m) around building for Rule 3 / 30
CANOPY_PCT_THRESHOLD  = 30      # % canopy cover required for Rule 30
PARK_DIST_THRESHOLD   = 300     # metres for Rule 300
PARK_MIN_AREA_HA      = 0.5     # minimum park size to qualify ("high-quality" green space)
PARK_SEARCH_BUFFER_M  = 500     # search this far beyond the boundary for parks

# Local projected CRS for accurate metre-based buffers/distances (UTM 43N).
PROJECTED_CRS = "EPSG:32643"
GEOGRAPHIC_CRS = "EPSG:4326"

# --- Study area boundary ------------------------------------------------------
# Converted from your Earth Engine (JavaScript) definition — same [lon, lat]
# coordinate pairs, expressed as a plain Python list / Shapely polygon.
coords = [
    [73.17544490351595, 22.279195629165724],
    [73.17450076594271, 22.26021219450331],
    [73.20488482966341, 22.255525510949344],
    [73.21784526362337, 22.255366637571026],
    [73.2272866393558, 22.270299947006922],
    [73.2313206817142, 22.2803869722086],
    [73.2151567795104, 22.283775136625845],
    [73.20833323977651, 22.285760637695443],
    [73.2065307953185, 22.282782375517236],
    [73.19949267886342, 22.282702954324044],
    [73.18301318667592, 22.284172239095547],
    [73.18026660464467, 22.28508557049555],
    [73.17889331362905, 22.285720927950386],
    [73.17584632418813, 22.285641508426437],
    [73.17438720248403, 22.282861796665337],
]

boundary_geom = Polygon(coords)
boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_geom], crs=GEOGRAPHIC_CRS)
boundary_proj = boundary_gdf.to_crs(PROJECTED_CRS)
boundary_plot = boundary_proj.geometry.iloc[0]

print(f"Study area loaded: {boundary_proj.area.iloc[0] / 1e6:.3f} km^2 "
      f"({boundary_proj.area.iloc[0]:,.0f} m^2)")


def basemap_setup(ax, title):
    x, y = boundary_plot.exterior.xy
    ax.plot(x, y, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


# =============================================================================
# 1. BUILDING FOOTPRINTS — from your cleaned shapefile (FIRST STEP)
# =============================================================================

def load_buildings_from_shapefile(shp_path, boundary_wgs84):
    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Shapefile not found at {shp_path}. Make sure all sidecar files "
            f"(.dbf, .prj, .shx, .cpg) are uploaded alongside it."
        )
    gdf = gpd.read_file(shp_path)
    print(f"Raw features in shapefile: {len(gdf):,}  (CRS: {gdf.crs})")

    if gdf.crs is None:
        print("WARNING: shapefile has no CRS defined (missing/unreadable .prj) — "
              "assuming WGS84 (EPSG:4326). Verify this is correct.")
        gdf = gdf.set_crs(GEOGRAPHIC_CRS)

    gdf = gdf.to_crs(GEOGRAPHIC_CRS)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    print(f"Polygon/MultiPolygon features: {len(gdf):,}")

    gdf = gpd.clip(gdf, boundary_wgs84)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.reset_index(drop=True)
    print(f"Buildings after clipping to neighbourhood boundary: {len(gdf):,}")

    return gdf


buildings_wgs84_raw = load_buildings_from_shapefile(BUILDINGS_SHP_PATH, boundary_gdf)

buildings_proj = buildings_wgs84_raw.to_crs(PROJECTED_CRS)
buildings_proj["building_id"] = buildings_proj.index
buildings_proj["buffer_geom"] = buildings_proj.geometry.buffer(BUFFER_M)

# Save + map the building layer NOW, before canopy/park processing, so you can
# verify completeness before the rest of the pipeline runs.
buildings_proj.drop(columns=["buffer_geom"]).to_crs(GEOGRAPHIC_CRS).to_file(
    os.path.join(OUTPUT_DIR, "buildings_clean.geojson"), driver="GeoJSON"
)

fig, ax = plt.subplots(figsize=(9, 9))
buildings_proj.plot(ax=ax, color="#4575b4", edgecolor="none")
basemap_setup(ax, f"Building Footprints — Vadodara_Buildings_Clean ({len(buildings_proj):,} buildings)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map0_all_buildings.png"), dpi=250)
plt.show()
plt.close()
print(f"\nSTEP 1 DONE — {len(buildings_proj):,} buildings loaded. "
      f"Check {OUTPUT_DIR}/map0_all_buildings.png before continuing.\n")


# =============================================================================
# 2. CANOPY RASTER — already merged & clipped; threshold to binary (memory-safe)
# =============================================================================

def threshold_canopy_raster(in_path, out_path, canopy_above_value):
    """
    Reads the already-clipped canopy raster and writes a compact binary
    (uint8: 1=canopy, 0=non-canopy, 255=nodata) version to disk.
    Processes block-by-block (windowed read/write) so memory use stays small
    regardless of raster size — avoids the earlier RAM-crash pattern.
    If canopy_above_value is None, assumes the input is already a binary
    0/1 mask and just copies/recasts it to uint8 with nodata preserved.
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Canopy raster not found at {in_path}.")

    with rasterio.open(in_path) as src:
        src_nodata = src.nodata
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="lzw", nodata=255)

        with rasterio.open(out_path, "w", **profile) as dst:
            canopy_pixel_total = 0
            valid_pixel_total = 0
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)

                if src_nodata is not None:
                    nodata_mask = data == src_nodata
                else:
                    nodata_mask = np.zeros_like(data, dtype=bool)

                if canopy_above_value is None:
                    binary = (data == 1).astype(np.uint8)
                else:
                    binary = (data > canopy_above_value).astype(np.uint8)

                out_block = np.where(nodata_mask, 255, binary).astype(np.uint8)
                dst.write(out_block, 1, window=window)

                canopy_pixel_total += int((out_block == 1).sum())
                valid_pixel_total += int((out_block != 255).sum())

        transform = dst.transform
        crs = dst.crs
        shape = (dst.height, dst.width)

    return transform, crs, shape, canopy_pixel_total, valid_pixel_total


CANOPY_TIF_OUT = os.path.join(OUTPUT_DIR, "canopy_binary.tif")
(canopy_transform, canopy_crs, canopy_shape,
 canopy_pixel_count, valid_pixel_count) = threshold_canopy_raster(
    CANOPY_TIF_IN, CANOPY_TIF_OUT, CANOPY_ABOVE_VALUE
)
pct_canopy_overall = (canopy_pixel_count / valid_pixel_count * 100) if valid_pixel_count else 0.0
print(f"STEP 2 DONE — Canopy raster thresholded: {CANOPY_TIF_OUT}  shape={canopy_shape}")
print(f"  Canopy pixels: {canopy_pixel_count:,} / {valid_pixel_count:,} valid "
      f"({pct_canopy_overall:.1f}% overall canopy cover)\n")


# =============================================================================
# 3. PARK / GREEN SPACE DATA — your uploaded CSV (Google My Maps) + OSM
# =============================================================================

def load_parks_from_csv(csv_path, min_area_ha, point_buffer_m):
    """
    Parses a Google My Maps CSV export (columns typically include a WKT
    geometry column, e.g. 'WKT', plus 'name'/'description'). Polygon features
    are filtered by min_area_ha like any other park source. Point features
    (a pin dropped on a garden with no drawn boundary) have no area to check —
    since you manually curated this layer, they're trusted as real park/garden
    locations and given a small nominal buffer instead of being filtered out.
    """
    if not os.path.exists(csv_path):
        print(f"WARNING: park CSV not found at {csv_path}; skipping this source.")
        return gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)

    df = pd.read_csv(csv_path)
    print(f"Park/garden CSV columns: {list(df.columns)}")

    wkt_col = next((c for c in df.columns if c.strip().lower() in ("wkt", "geometry")), None)
    if wkt_col is None:
        raise ValueError(
            f"No WKT/geometry column found in {csv_path}. "
            f"Columns present: {list(df.columns)}. Update the parsing logic "
            f"if your export uses separate lat/lon columns instead."
        )

    df = df[df[wkt_col].notna()].copy()
    df["geometry"] = df[wkt_col].apply(wkt_loads)
    gdf = gpd.GeoDataFrame(df.drop(columns=[wkt_col]), geometry="geometry", crs=GEOGRAPHIC_CRS)

    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    points = gdf[gdf.geometry.type == "Point"].copy()
    other_count = len(gdf) - len(polys) - len(points)
    if other_count:
        print(f"  (ignoring {other_count} non-point/polygon feature(s), e.g. lines)")

    polys_proj = polys.to_crs(PROJECTED_CRS)
    polys_proj["area_ha"] = polys_proj.geometry.area / 10_000
    qualifying_polys = polys_proj[polys_proj["area_ha"] >= min_area_ha]
    dropped = len(polys_proj) - len(qualifying_polys)
    print(f"  {len(polys_proj)} polygon feature(s) "
          f"({len(qualifying_polys)} >= {min_area_ha} ha, {dropped} excluded as too small)")

    points_proj = points.to_crs(PROJECTED_CRS)
    points_proj["geometry"] = points_proj.geometry.buffer(point_buffer_m)
    points_proj["area_ha"] = points_proj.geometry.area / 10_000
    print(f"  {len(points_proj)} point feature(s) (buffered to {point_buffer_m} m radius, "
          f"kept regardless of size — manually curated)")

    combined = gpd.GeoDataFrame(
        pd.concat(
            [qualifying_polys[["geometry", "area_ha"]], points_proj[["geometry", "area_ha"]]],
            ignore_index=True,
        ),
        crs=PROJECTED_CRS,
    )
    combined["source"] = "user CSV (Google My Maps)"
    return combined


def fetch_parks_osm(boundary_wgs84, min_area_ha):
    """Supplementary source: OpenStreetMap parks/green spaces."""
    import osmnx as ox

    tag_queries = [
        {"leisure": ["park", "garden", "nature_reserve", "common"]},
        {"landuse": ["recreation_ground", "forest", "village_green"]},
        {"boundary": "protected_area"},
    ]

    parks_list = []
    for tags in tag_queries:
        try:
            gdf = ox.features_from_polygon(boundary_wgs84.geometry.iloc[0], tags)
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if len(gdf):
                parks_list.append(gdf.reset_index()[["geometry"]])
        except Exception as e:
            print(f"  (skipped tag set {tags}: {e})")

    if not parks_list:
        print("  No park/green-space features returned from OSM for this area.")
        return gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)

    parks = gpd.GeoDataFrame(pd.concat(parks_list, ignore_index=True), crs=GEOGRAPHIC_CRS)
    parks_proj = parks.to_crs(PROJECTED_CRS)
    parks_proj["area_ha"] = parks_proj.geometry.area / 10_000
    parks_proj = parks_proj[parks_proj["area_ha"] >= min_area_ha].reset_index(drop=True)
    parks_proj["source"] = "OSM"
    return parks_proj


park_search_area_wgs84 = gpd.GeoDataFrame(
    geometry=[boundary_proj.geometry.iloc[0].buffer(PARK_SEARCH_BUFFER_M)],
    crs=PROJECTED_CRS,
).to_crs(GEOGRAPHIC_CRS)

print("Loading park/garden data from your uploaded CSV...")
csv_parks_proj = load_parks_from_csv(PARK_CSV_PATH, PARK_MIN_AREA_HA, POINT_PARK_BUFFER_M)
print(f"  Total qualifying features from CSV: {len(csv_parks_proj)}")

print("Fetching supplementary park/green-space data from OpenStreetMap "
      f"(boundary + {PARK_SEARCH_BUFFER_M} m search buffer)...")
osm_parks_proj = fetch_parks_osm(park_search_area_wgs84, PARK_MIN_AREA_HA)
print(f"  Qualifying parks from OSM: {len(osm_parks_proj)}")

_park_frames = [f for f in [csv_parks_proj, osm_parks_proj] if len(f)]
if _park_frames:
    parks_proj = gpd.GeoDataFrame(pd.concat(_park_frames, ignore_index=True), crs=PROJECTED_CRS)
else:
    parks_proj = gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)

print(f"STEP 3 DONE — Total qualifying parks/gardens (CSV + OSM): {len(parks_proj)}\n")


# =============================================================================
# 4. ZONAL STATS — canopy cover within each building's 30 m buffer (vectorized)
# =============================================================================
from rasterstats import zonal_stats

print(f"Computing canopy cover for {len(buildings_proj):,} buildings (vectorized)...")

buffers_in_raster_crs = gpd.GeoSeries(
    buildings_proj["buffer_geom"].values, crs=PROJECTED_CRS
).to_crs(canopy_crs)

stats = zonal_stats(
    buffers_in_raster_crs,
    CANOPY_TIF_OUT,
    band=1,
    stats=["mean"],
    nodata=255,
    all_touched=True,
)

buildings_proj["canopy_pct"] = [
    (s["mean"] * 100) if s["mean"] is not None else 0.0 for s in stats
]

print("Zonal stats done.")


# =============================================================================
# 5. RULE COMPLIANCE
# =============================================================================

buildings_proj["rule3_pass"] = buildings_proj["canopy_pct"] > 0
buildings_proj["rule30_pass"] = buildings_proj["canopy_pct"] >= CANOPY_PCT_THRESHOLD

if len(parks_proj) > 0:
    park_union = parks_proj.geometry.unary_union
    buildings_proj["dist_to_park_m"] = buildings_proj.geometry.centroid.distance(park_union)
else:
    buildings_proj["dist_to_park_m"] = np.inf

buildings_proj["rule300_pass"] = buildings_proj["dist_to_park_m"] <= PARK_DIST_THRESHOLD

buildings_proj["rules_met"] = (
    buildings_proj["rule3_pass"].astype(int)
    + buildings_proj["rule30_pass"].astype(int)
    + buildings_proj["rule300_pass"].astype(int)
)

n = len(buildings_proj)
print("\n--- COMPLIANCE SUMMARY ---")
print(f"Total buildings analysed: {n:,}")
print(f"Rule 3   pass: {buildings_proj['rule3_pass'].sum():,} / {n:,} "
      f"({buildings_proj['rule3_pass'].mean()*100:.1f}%)")
print(f"Rule 30  pass: {buildings_proj['rule30_pass'].sum():,} / {n:,} "
      f"({buildings_proj['rule30_pass'].mean()*100:.1f}%)")
print(f"Rule 300 pass: {buildings_proj['rule300_pass'].sum():,} / {n:,} "
      f"({buildings_proj['rule300_pass'].mean()*100:.1f}%)")
print(f"All 3 rules  : {(buildings_proj['rules_met']==3).sum():,} / {n:,} "
      f"({(buildings_proj['rules_met']==3).mean()*100:.1f}%)")

buildings_wgs84 = buildings_proj.drop(columns=["buffer_geom"]).to_crs(GEOGRAPHIC_CRS)
buildings_wgs84.to_file(os.path.join(OUTPUT_DIR, "buildings_3_30_300_results.geojson"),
                         driver="GeoJSON")
if len(parks_proj):
    parks_proj.to_crs(GEOGRAPHIC_CRS).to_file(
        os.path.join(OUTPUT_DIR, "qualifying_parks.geojson"), driver="GeoJSON")
print(f"\nResults saved to {OUTPUT_DIR}")


# =============================================================================
# 6. MAPS
# =============================================================================

PASS_FAIL_CMAP = ListedColormap(["#d73027", "#1a9850"])   # red = fail, green = pass
COMBINED_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#1a9850"]  # 0,1,2,3 rules met
COMBINED_CMAP = ListedColormap(COMBINED_COLORS)

# --- Map 1: Canopy cover raster ---------------------------------------------
with rasterio.open(CANOPY_TIF_OUT) as src:
    max_dim = 3000
    scale = min(1.0, max_dim / max(src.height, src.width))
    out_h, out_w = max(1, int(src.height * scale)), max(1, int(src.width * scale))
    canopy_preview = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    preview_bounds = src.bounds

fig, ax = plt.subplots(figsize=(9, 9))
extent = [preview_bounds.left, preview_bounds.right, preview_bounds.bottom, preview_bounds.top]
canopy_display_cmap = ListedColormap(["#e0e0e0", "#1a9850"])
canopy_preview_masked = np.where(canopy_preview == 255, 0, canopy_preview)
ax.imshow(canopy_preview_masked, cmap=canopy_display_cmap, extent=extent, origin="upper")
canopy_title = (f"Tree Canopy Cover (value > {CANOPY_ABOVE_VALUE})"
                if CANOPY_ABOVE_VALUE is not None else "Tree Canopy Cover")
basemap_setup(ax, canopy_title)
legend_elems = [
    mpatches.Patch(color="#1a9850", label="Tree canopy"),
    mpatches.Patch(color="#e0e0e0", label="No canopy"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map1_canopy_cover.png"), dpi=250)
plt.show()
plt.close()

# --- Map 2: Rule 3 compliance -----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))
buildings_proj.plot(ax=ax, column="rule3_pass", cmap=PASS_FAIL_CMAP, categorical=True, linewidth=0)
basemap_setup(ax, "Rule 3 — Tree Visibility Compliance")
legend_elems = [
    mpatches.Patch(color="#1a9850", label="Pass (canopy present within 30 m)"),
    mpatches.Patch(color="#d73027", label="Fail"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map2_rule3_visibility.png"), dpi=250)
plt.show()
plt.close()

# --- Map 3: Rule 30 compliance ----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))
buildings_proj.plot(ax=ax, column="rule30_pass", cmap=PASS_FAIL_CMAP, categorical=True, linewidth=0)
basemap_setup(ax, "Rule 30 — Canopy Density Compliance (>=30%)")
legend_elems = [
    mpatches.Patch(color="#1a9850", label="Pass (>=30% canopy in 30 m buffer)"),
    mpatches.Patch(color="#d73027", label="Fail"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map3_rule30_density.png"), dpi=250)
plt.show()
plt.close()

# --- Map 4: Rule 300 compliance ----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))
buildings_proj.plot(ax=ax, column="rule300_pass", cmap=PASS_FAIL_CMAP, categorical=True, linewidth=0)
if len(parks_proj):
    parks_proj.plot(ax=ax, color="#005a32", alpha=0.6, edgecolor="black", linewidth=0.3)
basemap_setup(ax, "Rule 300 — Proximity to Green Space (<=300 m)")
legend_elems = [
    mpatches.Patch(color="#1a9850", label="Pass (<=300 m to qualifying park)"),
    mpatches.Patch(color="#d73027", label="Fail"),
    mpatches.Patch(color="#005a32", label="Qualifying park (>=0.5 ha)"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map4_rule300_proximity.png"), dpi=250)
plt.show()
plt.close()

# --- Map 5: Combined compliance (0-3 rules met) -------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, COMBINED_CMAP.N)
buildings_proj.plot(ax=ax, column="rules_met", cmap=COMBINED_CMAP, norm=norm, linewidth=0)
if len(parks_proj):
    parks_proj.plot(ax=ax, color="#005a32", alpha=0.5, edgecolor="black", linewidth=0.3)
basemap_setup(ax, "Combined 3-30-300 Rule Compliance — Vadodara Neighbourhood")
legend_elems = [
    mpatches.Patch(color=COMBINED_COLORS[3], label="All 3 rules met"),
    mpatches.Patch(color=COMBINED_COLORS[2], label="2 rules met"),
    mpatches.Patch(color=COMBINED_COLORS[1], label="1 rule met"),
    mpatches.Patch(color=COMBINED_COLORS[0], label="0 rules met"),
    mpatches.Patch(color="#005a32", label="Qualifying park"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map5_combined_compliance.png"), dpi=250)
plt.show()
plt.close()

print("\nAll 6 maps displayed above and saved to:", OUTPUT_DIR)
print(" - map0_all_buildings.png       (STEP 1)")
print(" - map1_canopy_cover.png        (STEP 2)")
print(" - map2_rule3_visibility.png")
print(" - map3_rule30_density.png")
print(" - map4_rule300_proximity.png")
print(" - map5_combined_compliance.png")


# =============================================================================
# 7. EXPORT SHAPEFILES FOR DOWNLOAD (buildings, parks, boundary, canopy, maps)
# =============================================================================

SHP_EXPORT_DIR = os.path.join(OUTPUT_DIR, "shapefiles")
os.makedirs(SHP_EXPORT_DIR, exist_ok=True)

print("\nExporting shapefiles...")

# Buildings with all rule-compliance attributes
buildings_wgs84.to_file(os.path.join(SHP_EXPORT_DIR, "buildings_3_30_300_results.shp"))
print(f"  buildings_3_30_300_results.shp  ({len(buildings_wgs84):,} features)")

# Qualifying parks/gardens (CSV + OSM, merged)
if len(parks_proj):
    parks_wgs84 = parks_proj.to_crs(GEOGRAPHIC_CRS)
    parks_wgs84.to_file(os.path.join(SHP_EXPORT_DIR, "qualifying_parks.shp"))
    print(f"  qualifying_parks.shp  ({len(parks_wgs84):,} features)")

# Study area boundary
boundary_gdf.to_file(os.path.join(SHP_EXPORT_DIR, "study_area_boundary.shp"))
print("  study_area_boundary.shp")

# Canopy raster (kept as GeoTIFF — shapefiles are vector-only, not for rasters)
shutil.copy(CANOPY_TIF_OUT, os.path.join(SHP_EXPORT_DIR, "canopy_binary.tif"))

# All 6 PNG maps, for convenience in the same download
for png in glob.glob(os.path.join(OUTPUT_DIR, "*.png")):
    shutil.copy(png, SHP_EXPORT_DIR)

# Zip everything into one downloadable file
zip_base = os.path.join(OUTPUT_DIR, "vadodara_3_30_300_all_data")
zip_path = zip_base + ".zip"
if os.path.exists(zip_path):
    os.remove(zip_path)
shutil.make_archive(zip_base, "zip", SHP_EXPORT_DIR)
print(f"\nAll shapefiles + canopy raster + maps zipped at: {zip_path}")

try:
    from google.colab import files
    files.download(zip_path)
    print("Download triggered in your browser.")
except Exception as e:
    print(f"(Auto-download not available here: {e}. "
          f"Download manually from {zip_path} via the Colab file browser sidebar.)")

print("\nDone.")
