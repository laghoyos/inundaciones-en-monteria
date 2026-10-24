"""Stage 3: map figures (PNG) of the hydrological products."""
import os

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from rasterio.enums import Resampling
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "maps")
os.makedirs(FIG, exist_ok=True)
F = 3  # downsample factor for full-extent maps (1 m -> 3 m)

BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
INK, MUTED = "#1f1f1d", "#6b6a66"


def load(name, resampling=Resampling.average, factor=F):
    path = name if os.path.isabs(name) else os.path.join(OUT, name)
    with rasterio.open(path) as s:
        h, w = s.height // factor, s.width // factor
        if resampling == Resampling.max:  # not supported by plain reads: block max-pool instead
            a = s.read(1)[: h * factor, : w * factor].astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            a = np.nanmax(a.reshape(h, factor, w, factor), axis=(1, 3))
        else:
            a = s.read(1, out_shape=(h, w), resampling=resampling).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
        b = s.bounds
    return a, (b.left, b.right, b.bottom, b.top)


hs, ext = load("hillshade.tif")
dem, _ = load(os.path.join(ROOT, "dem.tif"))
mask = np.isnan(dem)
hs[mask] = np.nan
hs_n = (hs - np.nanpercentile(hs, 1)) / (np.nanpercentile(hs, 99) - np.nanpercentile(hs, 1))


def base(title, subtitle):
    fig, ax = plt.subplots(figsize=(11, 14), dpi=150)
    ax.set_facecolor("#ffffff")
    ax.imshow(np.clip(hs_n, 0, 1), cmap="gray", vmin=-0.4, vmax=1.3, extent=ext, interpolation="bilinear")
    ax.set_title(title, loc="left", fontsize=16, color=INK, pad=24, fontweight="bold")
    ax.text(0, 1.008, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#d6d5d0")
    ax.set_xlabel("Este (m, EPSG:9377)", color=MUTED, fontsize=8)
    ax.set_ylabel("Norte (m, EPSG:9377)", color=MUTED, fontsize=8)
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def colorbar(fig, ax, im, label, ticks=None):
    cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.015, ticks=ticks)
    cb.set_label(label, color=INK, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)


# 1. Elevation
fig, ax = base("Elevación del terreno", "DTM 1 m · hillshade multidireccional")
terrain = LinearSegmentedColormap.from_list("terr", ["#1c5cab", "#86b6ef", "#e9e4cf", "#c9a46b", "#8a5a2b", "#ffffff"])
im = ax.imshow(dem, cmap=terrain, vmin=14, vmax=40, alpha=0.6, extent=ext)
colorbar(fig, ax, im, "Elevación (m)")
save(fig, "01_elevacion.png")

# 2. Flow paths (max-pool the 1 m flow accumulation so thin paths survive downsampling)
acc, _ = load("flow_acc_cells.tif", resampling=Resampling.max)
fig, ax = base("Rutas de escorrentía superficial",
               "Acumulación de flujo D8 · área drenante ≥ 0.5 ha (líneas finas) a ≥ 25 ha (cauces principales)")
acc = np.nan_to_num(acc)
# draw larger flow classes as thicker lines (dilate each class by a width that grows with area)
classes = [(5e3, 1), (5e4, 2), (5e5, 3), (5e6, 4)]
paths = np.full(acc.shape, np.nan, np.float32)
for k, (thr, width) in enumerate(classes):
    m = ndimage.maximum_filter(acc >= thr, size=width) if width > 1 else acc >= thr
    paths[m] = k
im = ax.imshow(paths, cmap=ListedColormap(BLUES[2::1][:4]), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
               extent=ext, interpolation="antialiased")
colorbar(fig, ax, im, "Área drenante", ticks=[0, 1, 2, 3])
fig.axes[-1].set_yticklabels(["≥ 0.5 ha", "≥ 5 ha", "≥ 50 ha", "≥ 500 ha"])
save(fig, "02_rutas_escorrentia.png")
del acc, paths

# 3. Depressions (static storage depth)
dd, _ = load("depression_depth.tif", resampling=Resampling.max)
did, _ = load("depression_id.tif", resampling=Resampling.max)
dd[(did <= 0) | np.isnan(did)] = np.nan
fig, ax = base("Depresiones del terreno (zonas donde el agua queda atrapada)",
               "Profundidad máxima de almacenamiento si la depresión se llena hasta su punto de desborde")
norm = BoundaryNorm([0.1, 0.25, 0.5, 1, 1.5, 2, 3, 10], SEQ.N)
im = ax.imshow(dd, cmap=SEQ, norm=norm, extent=ext, interpolation="nearest")
colorbar(fig, ax, im, "Profundidad (m)")
save(fig, "03_depresiones.png")

# 4. Ponding per scenario
for P in (30, 60, 100):
    pd_, _ = load(f"ponding_depth_{P}mm.tif", resampling=Resampling.max)
    pd_[pd_ < 0.05] = np.nan
    fig, ax = base(f"Encharcamiento estimado – lluvia de {P} mm",
                   "Llenado y desborde en cascada de depresiones · coef. escorrentía 0.6 · sin drenaje pluvial")
    im = ax.imshow(pd_, cmap=SEQ, norm=norm, extent=ext, interpolation="nearest")
    colorbar(fig, ax, im, "Lámina de agua (m)")
    save(fig, f"04_encharcamiento_{P}mm.png")

# 5. HAND
hand, _ = load("hand.tif")
hand[mask] = np.nan
fig, ax = base("Altura sobre el drenaje más cercano (HAND)",
               "Valores bajos = terreno apenas por encima de caños/río → susceptible a inundación por desborde")
cls = [0, 0.5, 1, 2, 3, 5]
cmap = ListedColormap(BLUES[::-1][:5])
cmap.set_over((0, 0, 0, 0))
im = ax.imshow(np.where(hand <= 5, hand, np.nan), cmap=cmap, norm=BoundaryNorm(cls, 5), extent=ext, alpha=0.85,
               interpolation="nearest")
colorbar(fig, ax, im, "HAND (m)", ticks=cls)
save(fig, "05_hand.png")

# 6. TWI
twi, _ = load("twi.tif")
twi[mask] = np.nan
fig, ax = base("Índice topográfico de humedad (TWI)", "ln(a / tan β) · valores altos = acumulación y saturación")
lo, hi = np.nanpercentile(twi, [50, 99.5])
im = ax.imshow(np.where(twi > lo, twi, np.nan), cmap=SEQ, vmin=lo, vmax=hi, extent=ext, alpha=0.9)
colorbar(fig, ax, im, "TWI")
save(fig, "06_twi.png")

# 7. REM above the Sinú river water surface (2 m grid)
rem, ext2 = load("rem_rio_sinu.tif", factor=1)
dist, _ = load("dist_rio_sinu.tif", factor=1)
river, _ = load("rio_sinu_mask.tif", factor=1, resampling=Resampling.nearest)
rem[dist > 3000] = np.nan
fig, ax = base("Altura sobre la lámina de agua del río Sinú (REM)",
               "Terreno con REM menor a la altura de las orillas (~3–4 m) queda bajo el nivel del río si éste se desborda · hasta 3 km del cauce")
cls = [-1, 1, 2, 3, 4, 6]
cmap = ListedColormap(BLUES[::-1][:5])
im = ax.imshow(np.where(rem <= 6, rem, np.nan), cmap=cmap, norm=BoundaryNorm(cls, 5), extent=ext2, alpha=0.85,
               interpolation="nearest")
ax.imshow(np.where(river > 0, 1, np.nan), cmap=ListedColormap(["#0d366b"]), extent=ext2, interpolation="nearest")
colorbar(fig, ax, im, "Altura sobre el río (m)", ticks=cls)
save(fig, "07_rem_rio_sinu.png")
