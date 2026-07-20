import os
import zipfile
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from io import BytesIO
import shutil

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_IND_shp.zip"
DATA_DIR = "gadm_data"
EXTRACT_DIR = os.path.join(DATA_DIR, "extracted")
SHAPEFILE_PATH = os.path.join(EXTRACT_DIR, "gadm41_IND_2.shp")  # level 2 = districts

STATE_NAME = "Gujarat"
DISTRICT_NAME = "Vadodara"

PNG_OUTPUT = "gujarat_vadodara_highlight.png"
SHAPEFILE_OUTPUT_DIR = "gujarat_highlight_shapefile"
ZIP_OUTPUT = "gujarat_highlight_shapefile.zip"

# ------------------------------------------------------------
# 1. Download and extract GADM India shapefile
# ------------------------------------------------------------
def download_and_extract():
    """Download the GADM zip for India and extract it."""
    if not os.path.exists(SHAPEFILE_PATH):
        print("Downloading India shapefile from GADM...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        response = requests.get(GADM_URL)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(EXTRACT_DIR)
        print("Download and extraction complete.")
    else:
        print("Shapefile already exists, skipping download.")

download_and_extract()

# ------------------------------------------------------------
# 2. Load districts and filter Gujarat
# ------------------------------------------------------------
print("Loading districts shapefile...")
gdf_india = gpd.read_file(SHAPEFILE_PATH)

# Filter for Gujarat state (NAME_1 column in GADM)
gdf_gujarat = gdf_india[gdf_india['NAME_1'] == STATE_NAME].copy()
print(f"Found {len(gdf_gujarat)} districts in {STATE_NAME}.")

# ------------------------------------------------------------
# 3. Create the map with Vadodara highlighted (PNG FIRST)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# Plot all Gujarat districts in a neutral color
gdf_gujarat.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)

# Highlight Vadodara district
vadodara = gdf_gujarat[gdf_gujarat['NAME_2'] == DISTRICT_NAME]
if vadodara.empty:
    raise ValueError(f"District '{DISTRICT_NAME}' not found in Gujarat. Check spelling.")

vadodara.plot(ax=ax, color='red', edgecolor='black', linewidth=1.2)

# Add title and clean up axes
ax.set_title(f"Gujarat Districts – {DISTRICT_NAME} Highlighted", fontsize=16)
ax.axis('off')

# Save the PNG
plt.savefig(PNG_OUTPUT, dpi=200, bbox_inches='tight')
plt.close()
print(f"PNG map saved as: {PNG_OUTPUT}")

# ------------------------------------------------------------
# 4. Make the shapefile downloadable
# ------------------------------------------------------------
# Add a 'highlighted' column (1 for Vadodara, 0 for others)
gdf_gujarat['highlighted'] = (gdf_gujarat['NAME_2'] == DISTRICT_NAME).astype(int)

# Create output directory for shapefile
if os.path.exists(SHAPEFILE_OUTPUT_DIR):
    shutil.rmtree(SHAPEFILE_OUTPUT_DIR)
os.makedirs(SHAPEFILE_OUTPUT_DIR)

# Save the shapefile (all required sidecar files will be created)
shapefile_path = os.path.join(SHAPEFILE_OUTPUT_DIR, "gujarat_districts.shp")
gdf_gujarat.to_file(shapefile_path)

# Zip the whole folder for easy download
with zipfile.ZipFile(ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SHAPEFILE_OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, start=SHAPEFILE_OUTPUT_DIR)
            zf.write(full_path, arcname)

print(f"Shapefile archive ready for download: {ZIP_OUTPUT}")
print("Done.")