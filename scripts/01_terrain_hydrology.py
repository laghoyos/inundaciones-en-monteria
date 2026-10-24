"""Stage 1: terrain hydrology derivatives with WhiteboxTools.

Inputs : dem.tif (1 m bare-earth DTM, EPSG:9377)
Outputs: outputs/*.tif
"""
import os
import sys
import time

import whitebox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

wbt = whitebox.WhiteboxTools()
wbt.set_working_dir(OUT)
wbt.set_verbose_mode(False)
wbt.set_compress_rasters(True)

DEM = os.path.join(ROOT, "dem.tif")

# Flow-path thresholds, in contributing cells (1 cell = 1 m2)
MINOR_PATHS_M2 = 5_000      # 0.5 ha: street/lot-scale overland flow paths
MAJOR_PATHS_M2 = 250_000    # 25 ha: main drainage lines (caños, channels)


def run(name, fn, *args, **kwargs):
    t = time.time()
    rc = fn(*args, **kwargs)
    print(f"{name:<32} {time.time() - t:7.1f}s  rc={rc}", flush=True)
    if rc != 0:
        sys.exit(f"{name} failed")


# 1. Depressions: complete fill (no flat fixing) so filled - dem = ponding depth
if not os.path.exists(os.path.join(OUT, "depression_depth.tif")):
    run("fill_depressions", wbt.fill_depressions, DEM, "dem_filled.tif", fix_flats=False)
    run("depth_in_sink", wbt.depth_in_sink, DEM, "depression_depth.tif", zero_background=True)

# 2. Hydrologically conditioned DEM for routing: constrained breaching (fast Lindsay
#    algorithm; trenches at most 1 m deep / 100 m long, which mimics culverts and
#    channels under roads), then fill whatever remains with flat resolution.
run("fill_single_cell_pits", wbt.fill_single_cell_pits, DEM, "dem_nopits.tif")
run("breach_depressions", wbt.breach_depressions, "dem_nopits.tif", "dem_breached_partial.tif",
    max_depth=1.0, max_length=100)
run("fill_depressions (flats)", wbt.fill_depressions, "dem_breached_partial.tif", "dem_breached.tif",
    fix_flats=True)

# 3. Flow direction / accumulation
run("d8_pointer", wbt.d8_pointer, "dem_breached.tif", "d8_pointer.tif")
run("d8_flow_accumulation", wbt.d8_flow_accumulation, "d8_pointer.tif", "flow_acc_cells.tif",
    out_type="cells", pntr=True)
# Dispersive (FD8) specific contributing area for the wetness index
run("fd8_flow_accumulation", wbt.fd8_flow_accumulation, "dem_breached.tif", "sca_fd8.tif",
    out_type="specific contributing area", exponent=1.1)

# 4. Flow-path networks
run("extract_streams minor", wbt.extract_streams, "flow_acc_cells.tif", "flowpaths_minor.tif",
    threshold=MINOR_PATHS_M2, zero_background=False)
run("extract_streams major", wbt.extract_streams, "flow_acc_cells.tif", "flowpaths_major.tif",
    threshold=MAJOR_PATHS_M2, zero_background=False)
run("strahler_order", wbt.strahler_stream_order, "d8_pointer.tif", "flowpaths_minor.tif",
    "flowpaths_strahler.tif")

# 5. HAND: height above nearest major drainage line
run("elevation_above_stream (HAND)", wbt.elevation_above_stream,
    "dem_breached.tif", "flowpaths_major.tif", "hand.tif")

# 6. Topographic wetness index
run("slope", wbt.slope, "dem_breached.tif", "slope_deg.tif", units="degrees")
run("wetness_index", wbt.wetness_index, "sca_fd8.tif", "slope_deg.tif", "twi.tif")

# 7. Hillshade for map backgrounds
run("hillshade", wbt.multidirectional_hillshade, DEM, "hillshade.tif", altitude=35)

print("done")
