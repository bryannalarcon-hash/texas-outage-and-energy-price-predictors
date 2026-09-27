# Texas terrain assets

The two transparent, grayscale hillshades are calculated from public elevation measurements. Their ridges, valleys and flat coastal plains follow the DEM; they contain no noise, repeated contour pattern or AI-generated texture. They are ready to use offline.

| Mode | Asset, relative to `frontend/` | SVG image rectangle / view box | Raster size |
| --- | --- | --- | --- |
| Energy | `data/texas-terrain-energy.webp` | `0 0 1654 1504` | 2400 × 2182 |
| Outages | `data/texas-terrain-outages.webp` | `0 0 900 720` | 2400 × 1920 |

`data/texas-terrain.json` contains these rectangles, image hashes, projection parameters, registration controls, source information, terrain checks, and 15 major cities with coordinates in both SVG coordinate systems. City coordinates represent Census internal points within city limits, not downtown addresses. No roads are included.

## SVG integration

Use a separate `userSpaceOnUse` pattern for each mode, covering its original geometry view box. Keep the terrain in the same transformed group as its region so its features move with the raised surface. Apply the existing map tilt and zoom to that group; do not bake them into the raster.

```html
<pattern id="energy-terrain" patternUnits="userSpaceOnUse"
         x="0" y="0" width="1654" height="1504">
  <image href="data/texas-terrain-energy.webp"
         x="0" y="0" width="1654" height="1504"
         preserveAspectRatio="none" />
</pattern>
```

The outage pattern uses width `900`, height `720`, and `texas-terrain-outages.webp`. Use the pattern on the existing region path above its data color, with `pointer-events="none"` and `mix-blend-mode: hard-light`. Flat terrain is neutral gray 128, brighter pixels are illuminated slopes, and darker pixels are shaded slopes. The asset already includes 2.3× contrast; do not apply another contrast filter. Use opacity around `.86` at 100% zoom, easing toward `.68` at 250%, to adjust strength without per-county SVG filtering. Transparent pixels outside the existing Texas shape keep the page gradient clear. The pattern must not be stretched independently to each region's bounding box.

## Public sources and attribution

- [AWS Terrain Tiles registry](https://registry.opendata.aws/terrain-tiles/) documents the public, unsigned `elevation-tiles-prod` bucket; an AWS account or API key is unnecessary. Tile URL: `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/9/{x}/{y}.png`.
- [Tilezen's format specification](https://github.com/tilezen/joerd/blob/master/docs/formats.md) defines 256-pixel Web Mercator Terrarium tiles. Elevation in meters is `red * 256 + green + blue / 256 - 32768`.
- [Tilezen's source documentation](https://github.com/tilezen/joerd/blob/master/docs/data-sources.md) describes SRTM at zoom 9, with GMTED2010 and NOAA ocean bathymetry among the contributing datasets. The build records each tile's actual `x-amz-meta-x-imagery-sources` header, rather than assuming this is a current, full-resolution USGS 3DEP survey.
- [Tilezen's attribution and source terms](https://github.com/tilezen/joerd/blob/master/docs/attribution.md) identify the applicable USGS and NOAA public-domain sources and requested credits. Suggested visible credit: **Terrain: Mapzen / AWS Terrain Tiles; SRTM and GMTED2010 courtesy of the U.S. Geological Survey; ocean bathymetry: NOAA ETOPO1.** These are derived and visually exaggerated hillshades, not endorsed survey products.
- [2025 U.S. Census Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2025.html), specifically the [Texas places file](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_gaz_place_48.txt), supplies the city positions.
- County registration comes from this project's `build_outage_chart_data.py:county_shapes` and its cached [Texas Water Development Board county geometry](https://services.twdb.texas.gov/arcgis/rest/services/PWS/Texas_Counties_FIPS/FeatureServer/0). Energy registration follows the existing hand-traced [ERCOT load-zone illustration](https://www.ercot.com/files/assets/2023/06/05/ERCOT-Maps_Load-Zone.jpg).

Sources accessed September 27, 2026. Complete input URLs, SHA-256 hashes, modification headers and contributing DEM filenames are retained in the build cache's `source-manifest.json`; its hash is included in the shipped metadata.

## Registration

**Counties:** The original generator uses spherical Mercator coordinates measured in degrees: `mx = longitude`, `my = -180/π × log(tan(π/4 + latitude × π/360))`. Its minimum is `[-106.64622519560922, -39.25362994980602]`, scale is `55.7093689692846`, and padding is `[84.0335161586695, 12]`. SVG coordinates are `(mercator - minimum) × scale + padding`. The terrain uses the exact inverse of that transform. It does not infer a projection from the rendered outline or change county geometry.

**Energy:** The 1654 × 1504 source is a schematic illustration, not georeferenced boundaries. A smooth thin-plate spline ties real Mercator positions to 16 explicit controls: panhandle corners, New Mexico's corner, western and southern border landmarks, the northeast border, Sabine mouth, and Austin, San Antonio, Houston, Corpus Christi and Laredo. Border coordinates come from the same TWDB geometry; city controls use Census positions placed within the corresponding illustrated zones. The raster and city markers use this registration. The result aligns major features approximately but must not be used for address lookup or precise load-zone membership. The JSON exposes every control so the approximation remains inspectable.

## Rebuild

The existing project environment already provides NumPy, SciPy and Pillow. They are build-time dependencies only; the frontend loads static images and JSON. The metadata records their exact versions.

```sh
cd /home/ubuntu/projects/bpc-hackathon
.venv/bin/python tooling/build-terrain.py --self-test
.venv/bin/python tooling/build-terrain.py --download
.venv/bin/python tooling/build-terrain.py
```

Only `--download` permits network access. The first build caches public inputs in `~/.cache/bpc-terrain`; subsequent builds verify cached input hashes and run entirely offline. Keep that cache, including its receipts and manifest, for exact reproduction, or copy it and use `--cache-dir /path/to/cache`. Missing or modified cached inputs cause an error rather than fabricated fallback terrain. Source data can change upstream, so a fresh download is reproducible against its recorded hashes, not guaranteed to match an older source snapshot.

For a separate build environment, install `numpy==2.5.3 scipy==1.16.3 Pillow==12.3.0`; no new browser dependency is needed. `--width` accepts 1600–2400; the shipped images use 2400. `--zoom 9` is the default and preserves more real detail than the optional zoom 8 build.

The generator decodes and mosaics elevation first, smooths it by 0.65 source pixels, then calculates normals using ground spacing corrected for latitude. Northwest light supplies 72% of illumination; northeast and west lights add 18% and 10%. Light altitude is 40°. A 7× slope exaggeration and a small, DEM-derived local-relief term make mountains and escarpments legible at statewide scale. Bilinear resampling registers the hillshade to each SVG. The registered grayscale pixels then receive `clip(128 + 2.3 × (shade - 128), 0, 255)`, rounded to 8-bit values, before lossless WebP encoding. This changes only contrast: elevation, topology, lighting, registration, image bounds and transparency are unchanged. Metadata records `generation.shade_contrast = 2.3`. No random process is involved.

## Verification and limits

The script checks Terrarium decoding, projection round trips, constant shading on flat elevation, neutral-gray preservation, 2.3× contrast with black/white clipping, and light direction. Every build requires sensible sampled elevations in the Guadalupe and Davis Mountains, Palo Duro, Amarillo, Austin and Houston; it also requires stronger mountain relief than the Houston coastal plain, bounded schematic round-trip error, and complete DEM coverage of visible pixels. Numerical results and landmark SVG coordinates are in `texas-terrain.json`; shade-variation measurements include the baked contrast.

The delivered build uses 440 cached tiles over `[-107.578125, 25.165173, -92.109375, 37.160317]` in west/south/east/north order. Independent comparison with the original TWDB file matched all 4,418 vertices across 254 county paths within 0.070 SVG units, consistent with the existing one-decimal rounding. Sample elevations are 2,546.54 m in the Guadalupe Mountains, 1,954.34 m in the Davis Mountains, 161.59 m in Austin and 14.87 m in Houston. Both registrations were visually inspected with landmark markers. All 15 city points fall inside both rendered silhouettes. Offline rebuilds are checked with both `urllib` and socket networking blocked.

The contrast revision was rebuilt with networking blocked and compared pixel-for-pixel with the original hillshades. Every RGB value matches the rounded 2.3× transform; alpha, image dimensions, registration, source metadata and city positions are unchanged.

Hillshade is a lighting image, not a 3D mesh or geology map. The apparent relief is exaggerated; raw elevation values in the verification metadata are not. Source zoom 9 provides roughly 250–280 m ground samples across Texas before final resampling, sufficient for statewide landforms but not street-level terrain. Satellite/source seams and older surveys can remain. The energy warp introduces additional cartographic distortion, and regions outside the illustrated ERCOT footprint still have real terrain but no implied energy coverage.
