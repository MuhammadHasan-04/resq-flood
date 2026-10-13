# DADU_DATA — every ResQFlood data layer for Dadu district, in one place

Study area: Dadu district, Sindh (GADM v4.1 polygon, 8,984 km², bbox 67.144–68.035 E, 26.213–27.473 N).
Pilot sector: Johi–Dadu (67.35–68.05 E, 26.45–27.05 N, 3,653 km² inside the district).
All rasters are GeoTIFF in EPSG:4326 unless stated. All vectors are WGS84. Open everything in QGIS (drag and drop).

| Folder | What the panel can ask for | Main files | Key numbers |
|---|---|---|---|
| 01_Boundary | "Show your study area" | dadu_district_boundary_gadm41 (.shp/.geojson/.kml), johi_dadu_pilot_sector | 8,984 km² district, 3,653 km² sector |
| 02_Sentinel1_SAR / a_Dadu_2022-08-30_RTC_VV_VH | "Show the SAR imagery of Dadu" | dadu_S1A_20220830_RTC_VV_dBx100.tif, ..._VH_dBx100.tif (divide by 100 = dB), quicklook PNG | Sentinel-1A, 30 Aug 2022 01:26 UTC, orbit 78 descending, ~30 m, covers 100% of Dadu. Same acquisition as the TU Wien 30 Aug mask |
| 02 / b_Sen1Floods11_Pakistan_training_chips_hand_labelled | "Show your training data" | 28 SAR chips (VV+VH, 512×512) with LabelHand, Otsu and JRC labels | 5.38 M labelled px, 7.5% water. These are 2017 Pakistan chips (Sen1Floods11), not Dadu |
| 02 / c_matching_scene_list_for_TUWien | "Which SAR scenes match the masks?" | 60 Sentinel-1 acquisitions for the 164 TU Wien masks + GEE export script | 14 dates, 18 Aug–23 Sep 2022 |
| 03_TU_Wien_Flood_Masks / a_daily_masks_clipped_to_Dadu | "Show the TU Wien flood maps" | one GeoTIFF per acquisition date (0 = not flooded, 1 = flood, 255 = masked) | 10 dates; peak 1,986 km² on 8 Sep 2022 (see dadu_flood_area_by_date.csv) |
| 03 / b_raw_equi7_tiles_covering_Dadu | "Show the original TU Wien files" | 57 original FLOOD-HM-MASKED tiles (Equi7 Asia, 20 m) for tiles E015N027T3 and E015N030T3 | untouched originals from FLOOD-HM-MASKED.zip |
| 03 / c_summary_rasters, d_flood_vs_permanent_water | flood frequency, first detection, flood vs JRC water | dadu_flood_frequency.tif, dadu_first_flood_detection.tif, dadu_new_flood_vs_permanent_20220830.tif | 2,432 km² flooded at least once |
| 04_OSM_Network | "Show the road network" | dadu_osm_all_layers.gpkg (all layers), shapefiles/, geojson/, layer summary CSV | 5,248 roads (4,665 km), 268 bridges, 782 places, 204 waterways (2,172 km), 119 railway lines, 75 water areas, 453 landuse, 1,831 OSM buildings |
| 05_DEM_Copernicus_GLO30 | "Show the elevation data" | raw_tiles_1x1deg (4 original tiles), clipped/ DEM (×0.1 = m) and slope (×0.01 = degrees) | 25–1,916 m, median 47 m; 69% of cells slope < 2° |
| 06_JRC_Surface_Water | "How do you separate rivers from floods?" | occurrence, seasonality, extent, transitions, permanent_water_mask | 179 km² permanent water |
| 07_Buildings | "Where are the settlements / exposure?" | Microsoft ML building footprints clipped to Dadu | see 07_Buildings/README |
| 08_Routing_and_CostGrid_Results | "Show the routing output" | road graph nodes/edges CSV, roads with 30 Aug flood exposure, Johi→Dadu routes, boat cost grid and path | 52,615 nodes / 54,638 edges; shortest 18.6 km (3.0 km flooded) vs flood-aware 21.2 km (1.5 km flooded); boat path 23.5 km |
| 09_Crowdsourced_Reports | "Where is your crowd data?" | report_schema.json, empty template, verification targets | 0 real reports (collected by the app in FYP-2); 49 flooded bridges + 550 blocked-candidate segments to verify first |
| 10_Figures | maps for slides | f1–f8 PNG | |
| 11_Scripts_and_Stats | "How did you compute this?" | Python scripts and JSON results | baselines, routing, cost grid, sensitivity |

## SAR check you can show live
Simple Otsu threshold on the 30 Aug 2022 Sentinel-1 VV image (threshold −14.15 dB) agrees with the TU Wien flood mask of the same acquisition at IoU 0.86, F1 0.92 (4.31 M observed pixels; 02_Sentinel1_SAR/a_.../sar_threshold_vs_tuwien_20220830.json).
Note: TU Wien is itself derived from Sentinel-1, so this is a consistency check, not independent ground truth.

## Honest gaps (say these before the panel finds them)
- No real crowdsourced reports yet (by design; the app collects them).
- Sen1Floods11 training chips are from a 2017 Pakistan event, not Dadu.
- GADM Dadu polygon may differ from the current official boundary (e.g. Manchar Lake area is administered under Jamshoro).
- WorldPop population raster in the workspace is an incomplete download (.part) and is not included.

## Sources and licences
Sentinel-1 RTC: Copernicus Sentinel data 2022, processed RTC from Microsoft Planetary Computer (sentinel-1-rtc).
TU Wien: Roth et al. (2022) doi:10.48436/zvvmh-nan78, CC BY 4.0. OSM: © OpenStreetMap contributors, ODbL (Geofabrik extract).
Copernicus DEM GLO-30: © DLR e.V. / Airbus, provided under COPERNICUS by the European Union and ESA.
JRC Global Surface Water v1.4 (Pekel et al. 2016). Microsoft Global ML Building Footprints (ODbL). GADM v4.1.
