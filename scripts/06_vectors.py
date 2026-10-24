"""Stage 6: export hydrological results as vector layers (GeoPackage, EPSG:9377).

Layers in outputs/monteria_hidrologia.gpkg:
  rutas_flujo            flow-path lines (Strahler order, drainage area)
  depresiones            depression polygons with storage and fill attributes
  cuencas_depresiones    direct catchment of each depression
  encharcamiento_{P}mm   ponding-depth class polygons per rain scenario
  rem_rio_sinu           height-above-river class polygons
  rio_sinu               river water-surface polygon
  puntos_criticos        hotspot points (ranked)
Also writes puntos_criticos.geojson (WGS84) for web maps / Google Earth.
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import whitebox
from rasterio import features
from shapely.geometry import shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
GPKG = os.path.join(OUT, "monteria_hidrologia.gpkg")
CRS = "EPSG:9377"
SCENARIOS_MM = [30, 60, 100]
DEPTH_CLASSES = [(0.05, 0.25, "0.05-0.25 m"), (0.25, 0.5, "0.25-0.5 m"), (0.5, 1.0, "0.5-1 m"), (1.0, 99, "> 1 m")]
REM_CLASSES = [(-99, 1, "< 1 m"), (1, 2, "1-2 m"), (2, 3, "2-3 m"), (3, 4, "3-4 m")]


def read(name):
    with rasterio.open(os.path.join(OUT, name)) as s:
        return s.read(1), s.transform


def polygonize(arr, transform, mask=None, value_name="value", simplify=1.0):
    geoms = [(shape(g), v) for g, v in features.shapes(arr, mask=mask, transform=transform, connectivity=8)]
    gdf = gpd.GeoDataFrame({value_name: [v for _, v in geoms]}, geometry=[g for g, _ in geoms], crs=CRS)
    gdf = gdf.dissolve(by=value_name, as_index=False)
    gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True).buffer(0)
    return gdf


def write(gdf, layer):
    gdf.to_file(GPKG, layer=layer, driver="GPKG", engine="pyogrio")
    print(f"{layer:<24} {len(gdf):>6} features")


if os.path.exists(GPKG):
    os.remove(GPKG)

# ------------------------------------------------------------ flow paths
wbt = whitebox.WhiteboxTools()
wbt.set_working_dir(OUT)
wbt.set_verbose_mode(False)
assert wbt.raster_streams_to_vector("flowpaths_strahler.tif", "d8_pointer.tif", "_flowpaths.shp") == 0
lines = gpd.read_file(os.path.join(OUT, "_flowpaths.shp")).set_crs(CRS, allow_override=True)
lines = lines.rename(columns={"STRM_VAL": "orden_strahler"})[["orden_strahler", "geometry"]]
with rasterio.open(os.path.join(OUT, "flow_acc_cells.tif")) as s:
    # drainage area of a segment = max flow accumulation at its vertices (usually its downstream end)
    acc = s.read(1)
    inv = ~s.transform
    def seg_area(g):
        xy = np.asarray(g.coords)
        c, r = inv * (xy[:, 0], xy[:, 1])
        r = np.clip(r.astype(int), 0, acc.shape[0] - 1); c = np.clip(c.astype(int), 0, acc.shape[1] - 1)
        return float(acc[r, c].max())
    lines["area_drenante_ha"] = [round(seg_area(g) / 1e4, 2) for g in lines.geometry]
    del acc
lines["clase"] = pd.cut(lines["area_drenante_ha"], [0, 5, 50, 500, 1e9],
                        labels=["0.5-5 ha", "5-50 ha", "50-500 ha", "> 500 ha"]).astype(str)
lines["longitud_m"] = lines.length.round(1)
lines["geometry"] = lines.geometry.simplify(0.5)
write(lines, "rutas_flujo")
for ext in ("shp", "shx", "dbf", "prj"):
    p = os.path.join(OUT, f"_flowpaths.{ext}")
    if os.path.exists(p):
        os.remove(p)

# ------------------------------------------------------------ depressions + catchments
dep = pd.read_csv(os.path.join(OUT, "depressions.csv"))
hot = json.load(open(os.path.join(OUT, "hotspots.json")))
rank = {int(h["id"]): i for i, h in enumerate(hot, 1)}
conf = {int(h["id"]): h["confidence"] for h in hot}

did, tr = read("depression_id.tif")
poly = polygonize(did, tr, mask=did > 0, value_name="id")
poly["id"] = poly["id"].astype(int)
poly = poly.merge(dep, on="id", how="left")
poly["rank_critico_60mm"] = poly["id"].map(rank)
poly["confianza"] = poly["id"].map(conf)
write(poly, "depresiones")
del did

cid, tr = read("depression_catchment.tif")
cpoly = polygonize(cid, tr, mask=cid > 0, value_name="id", simplify=2.0)
cpoly["id"] = cpoly["id"].astype(int)
cpoly["area_ha"] = (cpoly.area / 1e4).round(2)
write(cpoly, "cuencas_depresiones")
del cid

# ------------------------------------------------------------ ponding depth classes
for P in SCENARIOS_MM:
    pond, tr = read(f"ponding_depth_{P}mm.tif")
    cls = np.zeros(pond.shape, np.uint8)
    for k, (lo, hi, _) in enumerate(DEPTH_CLASSES, 1):
        cls[(pond >= lo) & (pond < hi)] = k
    g = polygonize(cls, tr, mask=cls > 0, value_name="clase_id")
    g["profundidad"] = g["clase_id"].map({k: lbl for k, (_, _, lbl) in enumerate(DEPTH_CLASSES, 1)})
    g = g.explode(index_parts=False).reset_index(drop=True)
    g = g[g.area >= 10]  # drop slivers under 10 m2
    g["lluvia_mm"] = P
    g["area_ha"] = (g.area / 1e4).round(2)
    write(g, f"encharcamiento_{P}mm")
    del pond, cls

# ------------------------------------------------------------ river + REM classes (2 m grid)
rem, tr = read("rem_rio_sinu.tif")
dist, _ = read("dist_rio_sinu.tif")
river, _ = read("rio_sinu_mask.tif")
write(polygonize(river, tr, mask=river > 0, value_name="rio", simplify=2.0), "rio_sinu")
cls = np.zeros(rem.shape, np.uint8)
ok = (rem != -9999) & (dist <= 3000) & (river == 0)
for k, (lo, hi, _) in enumerate(REM_CLASSES, 1):
    cls[ok & (rem >= lo) & (rem < hi)] = k
g = polygonize(cls, tr, mask=cls > 0, value_name="clase_id", simplify=2.0)
g["altura_sobre_rio"] = g["clase_id"].map({k: lbl for k, (_, _, lbl) in enumerate(REM_CLASSES, 1)})
g["area_ha"] = (g.area / 1e4).round(1)
write(g, "rem_rio_sinu")

# ------------------------------------------------------------ hotspot points
hp = pd.DataFrame(hot)
hp.insert(0, "rank", range(1, len(hp) + 1))
for c in hp.columns:
    if c not in ("confidence", "touches_edge"):
        hp[c] = pd.to_numeric(hp[c])
pts = gpd.GeoDataFrame(hp, geometry=gpd.points_from_xy(hp["x_9377"].astype(float), hp["y_9377"].astype(float)),
                       crs=CRS)
write(pts, "puntos_criticos")
pts.to_crs("EPSG:4326").to_file(os.path.join(OUT, "puntos_criticos.geojson"), driver="GeoJSON")
print("wrote", GPKG)
