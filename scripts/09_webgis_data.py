"""Stage 9: data for the dynamic web GIS in docs/ (served by GitHub Pages).

  docs/tiles/terrain/{z}/{x}/{y}.png   512 px Terrarium-encoded DEM (MapLibre raster-dem: live hillshade + 3D)
  docs/tiles/altura/{z}/{x}/{y}.webp   512 px elevation colour relief (QGIS ramp), transparent outside the survey
  docs/data/*.geojson                  flow paths, ponding 30/60/100 mm, river REM classes, critical points
Areas outside the survey get a smooth, coarse fill in the terrain tiles so the hillshade has no cliff at the edge.
"""
import json
import os
import xml.etree.ElementTree as ET
import zipfile

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as from_bounds_transform
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")
GPKG = os.path.join(OUT, "monteria_hidrologia.gpkg")
GRID_ZOOMS = range(11, 17)
TILE = 512
BUFFER_DEG = 0.01
FILL_RES = 20  # m, coarse fill for areas outside the survey

qgs = ET.fromstring(zipfile.ZipFile(os.path.join(ROOT, "inundaciones.qgz")).read("inundaciones.qgs"))
sh = next(ml for ml in qgs.iter("maplayer") if ml.findtext("layername") == "dem_filled").find(".//colorrampshader")
items = [(float(i.get("value")), i.get("color")) for i in sh.iter("item")]
DEM_MIN, DEM_MAX = items[0][0], items[-1][0]
DEM_CMAP = LinearSegmentedColormap.from_list(
    "dem", sorted({(round((v - DEM_MIN) / (DEM_MAX - DEM_MIN), 6), c) for v, c in items}))


def coarse_fill(src):
    """Coarse DEM with nodata filled by nearest value and smoothed, as an in-memory dataset."""
    f = int(FILL_RES / src.res[0])
    h, w = src.height // f, src.width // f
    a = src.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
    bad = (a == src.nodata) | (a < 0)
    _, (ir, ic) = ndimage.distance_transform_edt(bad, return_indices=True)
    a = ndimage.gaussian_filter(a[ir, ic], 6)
    # extend well past the DEM's bounding box so tiles beyond it do not drop to 0 m
    pad = 25000 // FILL_RES
    a = ndimage.gaussian_filter(np.pad(a, pad, mode="edge"), 4)
    tr = src.transform * src.transform.scale(src.width / w, src.height / h) * src.transform.translation(-pad, -pad)
    return a, tr


def terrarium(elev):
    v = np.clip(elev + 32768.0, 0, 65535.99)
    r = np.floor(v / 256)
    g = np.floor(v) - r * 256
    b = np.floor((v - np.floor(v)) * 256)
    return np.dstack([r, g, b]).astype(np.uint8)


def build_tiles():
    src = rasterio.open(os.path.join(ROOT, "dem.tif"))
    fill, fill_tr = coarse_fill(src)
    w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    count = 0
    for z in GRID_ZOOMS:
        for t in mercantile.tiles(w - BUFFER_DEG, s - BUFFER_DEG, e + BUFFER_DEG, n + BUFFER_DEG, zooms=z):
            b = mercantile.xy_bounds(t)
            tr = from_bounds_transform(b.left, b.bottom, b.right, b.top, TILE, TILE)
            dem = np.full((TILE, TILE), src.nodata, np.float32)
            reproject(rasterio.band(src, 1), dem, dst_transform=tr, dst_crs="EPSG:3857", dst_nodata=src.nodata,
                      resampling=Resampling.bilinear if z >= 15 else Resampling.average)
            bg = np.zeros((TILE, TILE), np.float32)
            reproject(fill, bg, src_transform=fill_tr, src_crs=src.crs, dst_transform=tr, dst_crs="EPSG:3857",
                      resampling=Resampling.bilinear)
            valid = (dem != src.nodata) & (dem > 0)
            # soften the seam between survey and coarse fill
            wgt = ndimage.gaussian_filter(valid.astype(np.float32), 3) * valid
            elev = np.where(valid, dem * wgt + bg * (1 - wgt), bg)
            for kind in ("terrain", "altura"):
                os.makedirs(os.path.join(DOCS, "tiles", kind, str(z), str(t.x)), exist_ok=True)
            Image.fromarray(terrarium(elev)).save(
                os.path.join(DOCS, "tiles", "terrain", str(z), str(t.x), f"{t.y}.png"), optimize=True)
            if valid.any():
                rgba = DEM_CMAP(np.clip((dem - DEM_MIN) / (DEM_MAX - DEM_MIN), 0, 1))
                rgba[..., 3] = valid
                Image.fromarray((rgba * 255).round().astype(np.uint8)).save(
                    os.path.join(DOCS, "tiles", "altura", str(z), str(t.x), f"{t.y}.webp"), quality=82)
            count += 1
        print(f"zoom {z}: {count} tiles so far", flush=True)


def geojson(gdf, name, simplify_m, keep, digits=6):
    g = gdf[keep + ["geometry"]].copy()
    if simplify_m:
        g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
    g = g[~g.geometry.is_empty].to_crs("EPSG:4326")
    path = os.path.join(DOCS, "data", f"{name}.geojson")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    g.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=digits)
    print(f"{name}: {len(g)} features, {os.path.getsize(path) / 1e6:.2f} MB")


def build_vectors():
    flow = gpd.read_file(GPKG, layer="rutas_flujo")
    flow["clase"] = flow["clase"].fillna("")
    geojson(flow, "rutas_flujo", 1.5, ["clase", "area_drenante_ha", "orden_strahler"])
    for P in (30, 60, 100):
        g = gpd.read_file(GPKG, layer=f"encharcamiento_{P}mm")
        geojson(g, f"encharcamiento_{P}mm", 1.0, ["profundidad", "area_ha"])
    rem = gpd.read_file(GPKG, layer="rem_rio_sinu")
    rem = rem.explode(index_parts=False)
    rem = rem[rem.area >= 400]
    geojson(rem, "altura_sobre_rio", 3.0, ["altura_sobre_rio"], digits=5)
    pts = gpd.read_file(GPKG, layer="puntos_criticos")
    barrios = {1: "Barrio El Dorado, cerca de la Calle 30 · Comuna 1", 2: "Cerca de la Calle 78 · Comuna 8",
               3: "Urbanización Villa Campestre, Calle 50A · Comuna 8", 4: "San Martín · Comuna 3",
               5: "Cerca de la Carrera 201 · Comuna 8", 6: "El Amparo, Carrera 7W · Comuna 2", 7: "Comuna 6",
               8: "Comuna 8", 9: "Los Corales · Comuna 6", 10: "Comuna 8"}
    pts["barrio"] = pts["rank"].map(barrios).fillna("")
    for P in (30, 60, 100):
        pts[f"pond_ha_{P}mm"] = (pts[f"ponded_m2_{P}mm"] / 1e4).round(1)
    pts["catchment_ha"] = (pts["catchment_m2"] / 1e4).round(0)
    keep = ["rank", "barrio", "lat", "lon", "catchment_ha", "confidence"]
    for P in (30, 60, 100):
        keep += [f"pond_ha_{P}mm", f"pond_max_m_{P}mm", f"fill_pct_{P}mm"]
    geojson(pts, "puntos_criticos", 0, keep)
    with rasterio.open(os.path.join(ROOT, "dem.tif")) as s:
        b = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    json.dump({"bounds": [round(v, 5) for v in b], "minzoom": GRID_ZOOMS.start, "maxzoom": GRID_ZOOMS.stop - 1},
              open(os.path.join(DOCS, "data", "meta.json"), "w"))


if __name__ == "__main__":
    build_vectors()
    build_tiles()
