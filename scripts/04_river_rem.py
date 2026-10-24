"""Stage 4: Relative Elevation Model (REM) above the Sinú river water surface.

The river is in the DTM as a (hydro-flattened) water surface ~4 m below its banks.
We detect it as a large, deeply incised low band, keep its water-surface cells, and
express every cell's elevation relative to the nearest river water-surface cell.
REM < ~3-4 m means the ground is lower than the river banks: it floods if the Sinú
overtops or breaches its banks (or backs up through caños and storm drains).
"""
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
F = 2                      # work at 2 m
WIN_M = 600                # neighbourhood for "surrounding terrain" level
INCISION_M = 1.5          # channel must be this much below surroundings
MIN_RIVER_HA = 20

with rasterio.open(os.path.join(ROOT, "dem.tif")) as s:
    h, w = s.height // F, s.width // F
    d = s.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
    d[d <= s.nodata + 1] = np.nan
    prof = s.profile
    tr = s.transform * s.transform.scale(s.width / w, s.height / h)

valid = ~np.isnan(d)
size = WIN_M // F + 1
num = ndimage.uniform_filter(np.where(valid, d, 0), size)
den = ndimage.uniform_filter(valid.astype(np.float32), size)
surround = num / np.maximum(den, 1e-6)
channel = valid & (d - surround < -INCISION_M)
lab, n = ndimage.label(channel, structure=np.ones((3, 3)))
sizes = np.bincount(lab.ravel())
sizes[0] = 0
big = np.flatnonzero(sizes * F * F / 1e4 >= MIN_RIVER_HA)
print("incised components >= %d ha:" % MIN_RIVER_HA, [(int(i), round(sizes[i] * F * F / 1e4, 1)) for i in big])
# the river is the largest incised band; smaller ones are quarries/pits (e.g. by the cerro)
river_band = lab == big[np.argmax(sizes[big])]
# water surface = lowest part of the band (within 0.3 m of the local channel minimum)
dmin = ndimage.minimum_filter(np.where(river_band, d, np.inf), size=51)
water = river_band & (d - dmin < 0.3)
print("river water-surface cells: %.1f ha, elev range %.2f-%.2f m" %
      (water.sum() * F * F / 1e4, np.nanmin(d[water]), np.nanmax(d[water])))

_, (ir, ic) = ndimage.distance_transform_edt(~water, return_indices=True)
rem = (d - d[ir, ic]).astype(np.float32)
rem[~valid] = -9999
dist = (np.hypot(np.arange(h)[:, None] - ir, np.arange(w)[None, :] - ic) * F).astype(np.float32)
dist[~valid] = -9999

p = prof.copy()
p.update(width=w, height=h, transform=tr, dtype="float32", nodata=-9999, compress="deflate", tiled=True,
         blockxsize=512, blockysize=512, predictor=2)
for name, arr in (("rem_rio_sinu.tif", rem), ("dist_rio_sinu.tif", dist)):
    with rasterio.open(os.path.join(OUT, name), "w", **p) as dst:
        dst.write(arr, 1)
p.update(dtype="uint8", nodata=0, predictor=1)
with rasterio.open(os.path.join(OUT, "rio_sinu_mask.tif"), "w", **p) as dst:
    dst.write(water.astype(np.uint8), 1)

v = rem[valid & ~water]
area_ha = lambda m: m.sum() * F * F / 1e4  # noqa: E731
for t in (1, 2, 3, 4):
    print(f"area with REM < {t} m (excl. river): {area_ha(valid & ~river_band & (rem < t)):.0f} ha "
          f"of {area_ha(valid):.0f} ha")
