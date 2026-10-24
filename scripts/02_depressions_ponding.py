"""Stage 2: depression inventory and rain-driven fill-and-spill ponding.

For every significant terrain depression (closed low area where runoff cannot
leave by gravity) we compute:
  - area, max depth, storage volume, spill elevation
  - direct catchment (cells whose D8 flow path reaches it first)
  - the downstream depression it spills into (or 0 = leaves the area / reaches drainage)
Then, for each rain scenario, runoff = P * C * catchment area is routed through the
depression cascade (upstream spill feeds downstream depressions) and each depression
is filled from the bottom up, giving a water level and a ponding-depth raster.
"""
import csv
import os

import numpy as np
import rasterio
import whitebox
from rasterio.warp import transform as warp_transform
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

MIN_CELL_DEPTH = 0.02   # m, cells shallower than this are not depression
MIN_AREA = 50           # m2
MIN_MAX_DEPTH = 0.10    # m
MIN_VOLUME = 5          # m3
RUNOFF_COEF = 0.6       # mixed urban surface; change for sensitivity
SCENARIOS_MM = [30, 60, 100]

# Whitebox D8 pointer encoding -> (drow, dcol)
D8 = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0), 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}


def read(name, dtype=None):
    with rasterio.open(os.path.join(OUT, name) if not os.path.isabs(name) else name) as s:
        a = s.read(1)
        return (a.astype(dtype) if dtype else a), s.profile


def write(name, arr, profile, dtype, nodata):
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
             predictor=2 if dtype.startswith("float") else 1, BIGTIFF="IF_SAFER")
    with rasterio.open(os.path.join(OUT, name), "w", **p) as d:
        d.write(arr.astype(dtype), 1)


dem, prof = read(os.path.join(ROOT, "dem.tif"))
valid = dem != prof["nodata"]
depth, _ = read("depression_depth.tif")
depth = np.where(valid & (depth > MIN_CELL_DEPTH), depth, 0).astype(np.float32)
H, W = dem.shape

# ---------------------------------------------------------------- inventory
lab, n = ndimage.label(depth > 0, structure=np.ones((3, 3)))
print("raw depressions:", n)
flat = lab.ravel()
idx = np.flatnonzero(flat)                      # depression cells
L = flat[idx]
area = np.bincount(L, minlength=n + 1).astype(np.float64)
vol = np.bincount(L, weights=depth.ravel()[idx], minlength=n + 1)
maxd = np.zeros(n + 1, np.float32)
np.maximum.at(maxd, L, depth.ravel()[idx])
keep = (area >= MIN_AREA) & (maxd >= MIN_MAX_DEPTH) & (vol >= MIN_VOLUME)
keep[0] = False
newid = np.zeros(n + 1, np.int32)
newid[keep] = np.arange(1, keep.sum() + 1)
lab = newid[lab]
m = int(keep.sum())
area, vol, maxd = area[keep], vol[keep], maxd[keep]
print("significant depressions:", m)
write("depression_id.tif", lab, prof, "int32", 0)
del flat, idx, L

# ---------------------------------------------------------------- catchments
wbt = whitebox.WhiteboxTools()
wbt.set_working_dir(OUT)
wbt.set_verbose_mode(False)
assert wbt.watershed("d8_pointer.tif", "depression_id.tif", "depression_catchment.tif") == 0
catch, _ = read("depression_catchment.tif")
catch = np.where(valid & (catch > 0), catch, 0).astype(np.int32)
write("depression_catchment.tif", catch, prof, "int32", 0)
catch_area = np.bincount(catch.ravel(), minlength=m + 1)[1:].astype(np.float64)

# ---------------------------------------------------------------- outlets / cascade topology
ptr, _ = read("d8_pointer.tif")
acc, _ = read("flow_acc_cells.tif")
rows, cols = np.nonzero(lab)
own = lab[rows, cols]
p = ptr[rows, cols].astype(np.int32)
dr = np.zeros_like(p); dc = np.zeros_like(p)
for code, (r, c) in D8.items():
    sel = p == code
    dr[sel], dc[sel] = r, c
rr, cc = rows + dr, cols + dc
inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W) & (p > 0)
rr_c, cc_c = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
recv_lab = np.where(inside, lab[rr_c, cc_c], -1)
exits = recv_lab != own
# downstream depression = catchment id of the receiving cell (first depression downstream)
recv_catch = np.where(inside & valid[rr_c, cc_c], catch[rr_c, cc_c], 0)
recv_acc = np.where(inside, acc[rr_c, cc_c], 0)
e_own, e_down, e_acc = own[exits], recv_catch[exits], recv_acc[exits]
order = np.lexsort((-e_acc, e_own))            # per depression, outlet with largest flow first
e_own, e_down = e_own[order], e_down[order]
first = np.r_[True, e_own[1:] != e_own[:-1]]
downstream = np.zeros(m + 1, np.int32)
downstream[e_own[first]] = e_down[first]
downstream[downstream == np.arange(m + 1)] = 0

# spill elevation and centroid
filled, _ = read("dem_filled.tif")
spill = np.zeros(m + 1, np.float64)
np.maximum.at(spill, own, filled[rows, cols])
cy = np.bincount(own, weights=rows, minlength=m + 1)[1:] / area
cx = np.bincount(own, weights=cols, minlength=m + 1)[1:] / area
tr = prof["transform"]
ex, ny = tr * (cx + 0.5, cy + 0.5)
lon, lat = warp_transform(prof["crs"], "EPSG:4326", list(ex), list(ny))
del filled, ptr, acc

# ---------------------------------------------------------------- hypsometry for fill levels
z = dem[rows, cols].astype(np.float64)
o = np.lexsort((z, own))
own_s, z_s = own[o], z[o]
rows_s, cols_s = rows[o], cols[o]
start = np.searchsorted(own_s, np.arange(1, m + 1))
rank = np.arange(own_s.size) - start[own_s - 1] + 1           # 1-based rank within depression
csum = np.cumsum(z_s)
S = csum - np.r_[0, csum[start[1:] - 1]][own_s - 1] if m > 1 else csum
V = rank * z_s - S                                              # volume below z_k

proc = np.argsort(-spill[1:]) + 1                               # upstream (higher spill) first
results = {}
for P in SCENARIOS_MM:
    local_in = P / 1000.0 * RUNOFF_COEF * catch_area            # m3, direct runoff
    inflow = np.r_[0.0, local_in]
    stored = np.zeros(m + 1)
    overflow = np.zeros(m + 1)
    for d in proc:
        stored[d] = min(inflow[d], vol[d - 1])
        overflow[d] = inflow[d] - stored[d]
        if downstream[d] > 0 and overflow[d] > 0:
            inflow[downstream[d]] += overflow[d]
    Vin = stored[own_s]
    k = np.bincount(own_s[V <= Vin + 1e-9], minlength=m + 1)[1:]
    k = np.maximum(k, 1)
    level = (stored[1:] + S[start + k - 1]) / k
    level = np.minimum(level, spill[1:])
    pond = np.maximum(level[own_s - 1] - z_s, 0).astype(np.float32)
    out = np.zeros((H, W), np.float32)
    out[rows_s, cols_s] = pond
    write(f"ponding_depth_{P}mm.tif", out, prof, "float32", 0)
    flooded_area = np.bincount(own_s, weights=(pond > 0.05), minlength=m + 1)[1:]
    pmax = np.zeros(m + 1, np.float32); np.maximum.at(pmax, own_s, pond)
    results[P] = dict(fill=stored[1:] / vol, inflow=inflow[1:], area=flooded_area, maxd=pmax[1:],
                      spills=overflow[1:] > 0)
    print(f"P={P:>3} mm: depressions full={int((overflow[1:] > 0).sum())}/{m}, "
          f"ponded area>5cm={flooded_area.sum() / 1e4:.1f} ha, stored={stored.sum() / 1e3:.0f} k m3")

# ---------------------------------------------------------------- table
with open(os.path.join(OUT, "depressions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    hdr = ["id", "lat", "lon", "x_9377", "y_9377", "area_m2", "max_depth_m", "volume_m3", "spill_elev_m",
           "catchment_m2", "downstream_id"]
    for P in SCENARIOS_MM:
        hdr += [f"fill_pct_{P}mm", f"ponded_m2_{P}mm", f"pond_max_m_{P}mm"]
    w.writerow(hdr)
    for i in range(m):
        row = [i + 1, f"{lat[i]:.6f}", f"{lon[i]:.6f}", f"{ex[i]:.1f}", f"{ny[i]:.1f}", int(area[i]),
               f"{maxd[i]:.2f}", f"{vol[i]:.1f}", f"{spill[i + 1]:.2f}", int(catch_area[i]), int(downstream[i + 1])]
        for P in SCENARIOS_MM:
            r = results[P]
            row += [f"{100 * r['fill'][i]:.0f}", int(r["area"][i]), f"{r['maxd'][i]:.2f}"]
        w.writerow(row)
print("done")
