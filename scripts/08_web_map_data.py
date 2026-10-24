"""Stage 8: data for the interactive web map (outputs/web/).

  tiles/{z}/{x}/{y}.jpg   512 px web-mercator tiles of the QGIS project's bottom three layers,
                          composited as QGIS draws them: hillshade -> dem_filled (70 %) -> OSM (40 %).
                          Grid zoom z holds map-zoom z+1 detail (Leaflet: tileSize 512, zoomOffset -1).
  capas/*.geojson         flow paths, 100 mm ponding and hotspot points in WGS84.
Street tiles come from OSM France (OSM's own servers refuse scripted downloads) and are cached
in outputs/web/_osm_cache so re-runs do not download them again.
"""
import io
import json
import os
import time
import xml.etree.ElementTree as ET
import zipfile

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
import requests
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as from_bounds_transform
from rasterio.warp import reproject, transform_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
WEB = os.path.join(OUT, "web")
CACHE = os.path.join(WEB, "_osm_cache")
GPKG = os.path.join(OUT, "monteria_hidrologia.gpkg")
GRID_ZOOMS = range(12, 16)          # map zooms 13-16
TILE = 512
OSM_URL = "https://a.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png"
HEADERS = {"User-Agent": "inundaciones-en-monteria flood study (one-off static map build)"}
BUFFER_DEG = 0.003

# ---- QGIS styles
qgs = ET.fromstring(zipfile.ZipFile(os.path.join(ROOT, "inundaciones.qgz")).read("inundaciones.qgs"))
layers = {ml.findtext("layername"): ml for ml in qgs.iter("maplayer")}
sh = layers["dem_filled"].find(".//colorrampshader")
items = [(float(i.get("value")), i.get("color")) for i in sh.iter("item")]
dem_min, dem_max = items[0][0], items[-1][0]
dem_cmap = LinearSegmentedColormap.from_list(
    "dem", sorted({(round((v - dem_min) / (dem_max - dem_min), 6), c) for v, c in items}))
dem_alpha = float(layers["dem_filled"].find(".//rasterrenderer").get("opacity"))
osm_alpha = float(layers["OpenStreetMap"].find(".//rasterrenderer").get("opacity"))
ce = layers["hillshade"].find(".//contrastEnhancement")
hs_min, hs_max = float(ce.findtext("minValue")), float(ce.findtext("maxValue"))


def ensure_overviews(path):
    with rasterio.open(path, "r+") as s:
        if not s.overviews(1):
            s.build_overviews([2, 4, 8, 16, 32], Resampling.average)


def osm_tile(z, x, y):
    p = os.path.join(CACHE, str(z), str(x), f"{y}.png")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for attempt in range(3):
            r = requests.get(OSM_URL.format(z=z, x=x, y=y), headers=HEADERS, timeout=30)
            if r.ok:
                break
            time.sleep(2)
        r.raise_for_status()
        open(p, "wb").write(r.content)
        time.sleep(0.15)
    return Image.open(p).convert("RGB")


def build_tiles():
    for f in ("hillshade.tif", "dem_filled.tif"):
        ensure_overviews(os.path.join(OUT, f))
    with rasterio.open(os.path.join(ROOT, "dem.tif")) as s:
        w, so, e, n = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    hs_src = rasterio.open(os.path.join(OUT, "hillshade.tif"))
    dem_src = rasterio.open(os.path.join(OUT, "dem_filled.tif"))
    count = 0
    for g in GRID_ZOOMS:
        for t in mercantile.tiles(w - BUFFER_DEG, so - BUFFER_DEG, e + BUFFER_DEG, n + BUFFER_DEG, zooms=g):
            b = mercantile.xy_bounds(t)
            dst_tr = from_bounds_transform(b.left, b.bottom, b.right, b.top, TILE, TILE)
            hs = np.zeros((TILE, TILE), np.float32)
            dem = np.full((TILE, TILE), dem_src.nodata, np.float32)
            for src, arr in ((hs_src, hs), (dem_src, dem)):
                reproject(rasterio.band(src, 1), arr, dst_transform=dst_tr, dst_crs="EPSG:3857",
                          dst_nodata=arr[0, 0], resampling=Resampling.average if g < 15 else Resampling.bilinear)
            valid = (dem > 0) & (dem != dem_src.nodata)
            img = np.ones((TILE, TILE, 3), np.float32)                     # white canvas
            g_ = np.clip((hs - hs_min) / (hs_max - hs_min), 0, 1)[..., None]
            img = np.where(valid[..., None], np.repeat(g_, 3, axis=2), img)
            dem_rgb = dem_cmap(np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1))[..., :3]
            img = np.where(valid[..., None], img * (1 - dem_alpha) + dem_rgb * dem_alpha, img)
            # OSM at map zoom g+1: the 2x2 children of this tile
            osm = Image.new("RGB", (TILE, TILE))
            for c in mercantile.children(t):
                osm.paste(osm_tile(c.z, c.x, c.y), ((c.x - 2 * t.x) * 256, (c.y - 2 * t.y) * 256))
            img = img * (1 - osm_alpha) + (np.asarray(osm, np.float32) / 255) * osm_alpha
            p = os.path.join(WEB, "tiles", str(g), str(t.x))
            os.makedirs(p, exist_ok=True)
            Image.fromarray((img * 255).round().astype(np.uint8)).save(
                os.path.join(p, f"{t.y}.jpg"), quality=80, optimize=True, progressive=True)
            count += 1
        print(f"grid zoom {g}: done ({count} tiles so far)", flush=True)
    return count


def to_geojson(gdf, name, simplify_m, keep, digits=6):
    g = gdf[keep + ["geometry"]].copy()
    if simplify_m:
        g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
    g = g.to_crs("EPSG:4326")
    path = os.path.join(WEB, "capas", f"{name}.geojson")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=digits)
    print(f"{name}: {len(g)} features, {os.path.getsize(path) / 1e6:.2f} MB")


def build_vectors():
    flow = gpd.read_file(GPKG, layer="rutas_flujo")
    flow["clase"] = flow["clase"].fillna("")
    to_geojson(flow, "rutas_flujo", 1.5, ["clase", "area_drenante_ha"])
    pond = gpd.read_file(GPKG, layer="encharcamiento_100mm")
    to_geojson(pond, "encharcamiento_100mm", 1.0, ["profundidad", "area_ha"])
    pts = gpd.read_file(GPKG, layer="puntos_criticos")
    barrios = {1: "Barrio El Dorado, cerca de la Calle 30 · Comuna 1", 2: "Cerca de la Calle 78 · Comuna 8",
               3: "Urbanización Villa Campestre, Calle 50A · Comuna 8", 4: "San Martín · Comuna 3",
               5: "Cerca de la Carrera 201 · Comuna 8", 6: "El Amparo, Carrera 7W · Comuna 2", 7: "Comuna 6",
               8: "Comuna 8", 9: "Los Corales · Comuna 6", 10: "Comuna 8"}
    pts["barrio"] = pts["rank"].map(barrios).fillna("")
    pts["pond_ha_100mm"] = (pts["ponded_m2_100mm"] / 1e4).round(1)
    pts["catchment_ha"] = (pts["catchment_m2"] / 1e4).round(0)
    to_geojson(pts, "puntos_criticos", 0,
               ["rank", "barrio", "lat", "lon", "pond_ha_100mm", "pond_max_m_100mm", "fill_pct_30mm",
                "fill_pct_60mm", "fill_pct_100mm", "catchment_ha", "confidence"])


if __name__ == "__main__":
    build_vectors()
    print("tiles:", build_tiles())
