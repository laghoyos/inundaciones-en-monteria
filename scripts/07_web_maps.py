"""Stage 7: render the QGIS project (inundaciones.qgz) map composition as PNGs for the web page.

Reproduces the project's layer stack (OSM drawn via OSM France tiles: OSM's own tile
servers refuse scripted downloads) and styles, bottom to top:
  hillshade (gray, stretch 1-196) · dem_filled (QGIS ramp, 70 %) · OpenStreetMap (40 %)
  · ponding_depth_100mm (QGIS ramp) · rutas_flujo (navy, width by class) · puntos_criticos (gray dots)
Outputs: outputs/web/mapa_general.png and zoom_<rank>.png for the top hotspots.
"""
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import contextily as cx
import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.windows import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
WEB = os.path.join(OUT, "web")
os.makedirs(WEB, exist_ok=True)
GPKG = os.path.join(OUT, "monteria_hidrologia.gpkg")
MM_TO_PT = 1 / 0.3528
N_ZOOM = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# ---- read styles straight from the QGIS project so the images match what people see there
qgs = ET.fromstring(zipfile.ZipFile(os.path.join(ROOT, "inundaciones.qgz")).read("inundaciones.qgs"))
layers = {ml.findtext("layername"): ml for ml in qgs.iter("maplayer")}


def ramp(layer_name):
    sh = layers[layer_name].find(".//colorrampshader")
    items = [(float(i.get("value")), i.get("color")) for i in sh.iter("item")]
    vmin, vmax = items[0][0], items[-1][0]
    stops = sorted({(round((v - vmin) / (vmax - vmin), 6), c) for v, c in items})
    return LinearSegmentedColormap.from_list(layer_name, stops), vmin, vmax


def rgba(s):
    r, g, b, a = (int(x) for x in s.split(",")[:4])
    return (r / 255, g / 255, b / 255, a / 255)


dem_cmap, dem_min, dem_max = ramp("dem_filled")
dem_opacity = float(layers["dem_filled"].find(".//rasterrenderer").get("opacity"))
pond_cmap, pond_min, pond_max = ramp("ponding_depth_100mm")
osm_opacity = float(layers["OpenStreetMap"].find(".//rasterrenderer").get("opacity"))
ce = layers["hillshade"].find(".//contrastEnhancement")
hs_min, hs_max = float(ce.findtext("minValue")), float(ce.findtext("maxValue"))

flow_ml = next(v for k, v in layers.items() if k.endswith("rutas_flujo"))
flow_syms = {}
for s in flow_ml.find("renderer-v2/symbols"):
    opts = {o.get("name"): o.get("value") for o in s.iter("Option")}
    flow_syms[s.get("name")] = (rgba(opts["line_color"]), float(opts["line_width"]) * MM_TO_PT)
flow_style = {c.get("value"): flow_syms[c.get("symbol")] for c in flow_ml.iter("category")}
pts_ml = next(v for k, v in layers.items() if k.endswith("puntos_criticos"))
popts = {o.get("name"): o.get("value") for o in pts_ml.find("renderer-v2/symbols/symbol").iter("Option")}
pt_fill, pt_edge, pt_size_mm = rgba(popts["color"]), rgba(popts["outline_color"]), float(popts["size"])
print("flow styles:", {k: round(v[1] / MM_TO_PT, 2) for k, v in flow_style.items()})

flow = gpd.read_file(GPKG, layer="rutas_flujo")
pts = gpd.read_file(GPKG, layer="puntos_criticos")


def read_window(name, bounds, factor):
    with rasterio.open(os.path.join(OUT, name)) as s:
        win = from_bounds(*bounds, s.transform)
        h, w = int(win.height // factor), int(win.width // factor)
        a = s.read(1, window=win, out_shape=(h, w), boundless=True, fill_value=s.nodata or 0).astype(np.float32)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
    return a


def render(bounds, factor, fname, figsize, zoom, label_points=False, scalebar_m=1000):
    xmin, ymin, xmax, ymax = bounds
    ext = (xmin, xmax, ymin, ymax)
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    ax.set_facecolor("white")
    hs = read_window("hillshade.tif", bounds, factor)
    ax.imshow(hs, cmap="gray", vmin=hs_min, vmax=hs_max, extent=ext, interpolation="bilinear", zorder=1)
    dem = read_window("dem_filled.tif", bounds, factor)
    ax.imshow(dem, cmap=dem_cmap, vmin=dem_min, vmax=dem_max, alpha=dem_opacity, extent=ext, zorder=2)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, crs="EPSG:9377", source=cx.providers.OpenStreetMap.France, zoom=zoom, alpha=osm_opacity,
                   attribution=False, zorder=3)
    pond = read_window("ponding_depth_100mm.tif", bounds, factor)
    pond[pond <= 0] = np.nan
    ax.imshow(pond, cmap=pond_cmap, vmin=pond_min, vmax=pond_max, extent=ext, interpolation="nearest", zorder=4)
    vis = flow.cx[xmin:xmax, ymin:ymax]
    for cls, (col, lw) in flow_style.items():
        sub = vis[vis["clase"].fillna("") == cls] if cls else vis[vis["clase"].isin(["", "nan"]) | vis["clase"].isna()]
        if len(sub):
            sub.plot(ax=ax, color=matplotlib.colors.to_hex(col), linewidth=lw * (figsize[0] / 11), zorder=5)
    p = pts.cx[xmin:xmax, ymin:ymax]
    ax.scatter(p.geometry.x, p.geometry.y, s=(pt_size_mm * MM_TO_PT * 1.6) ** 2, c=[pt_fill], edgecolors=[pt_edge],
               linewidths=0.6, zorder=6)
    if label_points:
        for _, r in p[p["rank"] <= label_points].iterrows():
            ax.annotate(str(int(r["rank"])), (r.geometry.x, r.geometry.y), xytext=(5, 5), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#1b1b2f", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    # scale bar + north arrow + attribution
    sx, sy = xmin + (xmax - xmin) * 0.04, ymin + (ymax - ymin) * 0.035
    ax.plot([sx, sx + scalebar_m], [sy, sy], color="#1b1b2f", lw=3, solid_capstyle="butt", zorder=8)
    ax.text(sx + scalebar_m / 2, sy + (ymax - ymin) * 0.012, f"{scalebar_m:g} m" if scalebar_m < 1000 else
            f"{scalebar_m / 1000:g} km", ha="center", fontsize=9, color="#1b1b2f", zorder=8)
    ax.annotate("N", xy=(0.96, 0.97), xytext=(0.96, 0.90), xycoords="axes fraction", ha="center", fontsize=11,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="#1b1b2f"), zorder=8)
    ax.text(0.995, 0.005, "Calles © colaboradores de OpenStreetMap, OSM France", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6, color="#444", zorder=8)
    ax.set_axis_off()
    fig.savefig(os.path.join(WEB, fname), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", fname)


with rasterio.open(os.path.join(ROOT, "dem.tif")) as s:
    b = s.bounds
render((b.left, b.bottom, b.right, b.top), 3, "mapa_general.png", (11, 15), zoom=15, label_points=10)

for _, r in pts.sort_values("rank").head(N_ZOOM).iterrows():
    x, y = r.geometry.x, r.geometry.y
    half = 450
    render((x - half, y - half * 0.75, x + half, y + half * 0.75), 1, f"zoom_{int(r['rank'])}.png", (11, 8.25),
           zoom=17, label_points=99, scalebar_m=200)
