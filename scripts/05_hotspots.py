"""Stage 5: rank ponding hotspots and draw zoomed maps for the top ones."""
import csv
import json
import os

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from rasterio.windows import from_bounds
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "maps")
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
N_ZOOM = 8
P = 60  # reference scenario for ranking

rows = list(csv.DictReader(open(os.path.join(OUT, "depressions.csv"))))
# depressions whose ponded area touches the survey boundary may be bounded by unsurveyed terrain
with rasterio.open(os.path.join(OUT, "depression_id.tif")) as s_id, rasterio.open(os.path.join(ROOT, "dem.tif")) as s_dem:
    did = s_id.read(1)
    nod = s_dem.read(1) == s_dem.nodata
touch = set(np.unique(did[ndimage.binary_dilation(nod, iterations=2)]).tolist()) - {0}
del did, nod
for r in rows:
    md = float(r["max_depth_m"])
    r["confidence"] = "alta" if md >= 0.5 else ("media" if md >= 0.25 else "baja")
    r["touches_edge"] = int(r["id"]) in touch
    if r["touches_edge"] and r["confidence"] != "baja":
        r["confidence"] = "media (borde)"
    r["score"] = int(r[f"ponded_m2_{P}mm"]) * float(r[f"pond_max_m_{P}mm"])
hot = sorted([r for r in rows if int(r[f"ponded_m2_{P}mm"]) >= 2000 and float(r[f"pond_max_m_{P}mm"]) >= 0.25],
             key=lambda r: -r["score"])
print(f"{len(hot)} hotspots (>=0.2 ha and >=0.25 m ponding at {P} mm)")
json.dump(hot, open(os.path.join(OUT, "hotspots.json"), "w"), indent=1)

srcs = {k: rasterio.open(os.path.join(OUT, f)) for k, f in
        [("hs", "hillshade.tif"), ("pond", f"ponding_depth_{P}mm.tif"), ("acc", "flow_acc_cells.tif")]}
norm = BoundaryNorm([0.05, 0.25, 0.5, 1, 1.5, 2, 3, 10], SEQ.N)
fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=130)
for ax, (rank, r) in zip(axes.ravel(), enumerate(hot[:N_ZOOM], 1)):
    x, y = float(r["x_9377"]), float(r["y_9377"])
    half = max(250, (float(r["area_m2"]) ** 0.5) * 0.9)
    b = (x - half, y - half, x + half, y + half)
    win = from_bounds(*b, srcs["hs"].transform)
    hs = srcs["hs"].read(1, window=win, boundless=True, fill_value=0).astype(float)
    pond = srcs["pond"].read(1, window=win, boundless=True, fill_value=0)
    acc = srcs["acc"].read(1, window=win, boundless=True, fill_value=0)
    ext = (b[0], b[2], b[1], b[3])
    lo, hi = np.percentile(hs[hs > 0], [1, 99]) if (hs > 0).any() else (0, 1)
    ax.imshow(hs, cmap="gray", vmin=lo - (hi - lo) * 0.4, vmax=hi, extent=ext)
    ax.imshow(np.where(acc >= 2000, 1, np.nan), cmap=matplotlib.colors.ListedColormap(["#6b6a66"]), extent=ext,
              interpolation="nearest", alpha=0.8)
    ax.imshow(np.where(pond >= 0.05, pond, np.nan), cmap=SEQ, norm=norm, extent=ext, interpolation="nearest")
    ax.set_title(f"#{rank} · depresión {r['id']} · {r['lat']}, {r['lon']}", fontsize=9, loc="left", color="#1f1f1d")
    ax.text(0.02, 0.02, f"{int(r[f'ponded_m2_{P}mm']) / 1e4:.1f} ha · máx {r[f'pond_max_m_{P}mm']} m\n"
            f"confianza {r['confidence']}", transform=ax.transAxes, fontsize=8, color="#1f1f1d",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([b[0] + 0.05 * 2 * half, b[0] + 0.05 * 2 * half + 100], [b[3] - 0.06 * 2 * half] * 2, color="#1f1f1d", lw=2)
    ax.text(b[0] + 0.05 * 2 * half + 50, b[3] - 0.05 * 2 * half, "100 m", ha="center", fontsize=7)
fig.suptitle(f"Puntos críticos de encharcamiento – lluvia de {P} mm (gris: rutas de flujo ≥ 0.2 ha)",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color="#1f1f1d")
cax = fig.add_axes([0.92, 0.25, 0.008, 0.5])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
cb.set_label("Lámina de agua (m)"); cb.outline.set_visible(False)
fig.subplots_adjust(left=0.01, right=0.9, wspace=0.04, hspace=0.12)
fig.savefig(os.path.join(FIG, "08_puntos_criticos.png"), bbox_inches="tight", facecolor="white")
print("wrote 08_puntos_criticos.png")
