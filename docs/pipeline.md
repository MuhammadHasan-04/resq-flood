# Pipeline

```mermaid
flowchart LR
  S1[Sentinel-1 VV/VH\nexport_sar.py] --> PRE[Preprocess: dB, no-data,\nalign to mask grid]
  TW[TU Wien dated masks] --> PRE
  PRE --> TILES[Tiles + date split]
  TILES --> MODEL[Otsu baseline → U-Net\nFinal FYP-1]
  MODEL -->|flood mask| ING
  TW -->|reference mask| ING[Flood-mask ingestion\nscore 5,248 roads]
  JRC[JRC permanent water] --> COST
  ING --> G[Road graph\n52k nodes]
  REP[Crowd reports\nvalidate + score] -->|verified open/blocked| G
  G --> ROUTE[A* + 3 alternatives]
  ING --> COST[Boat cost grid\nflood + JRC + roads + DEM]
  DEM[Copernicus DEM] --> COST
  COST --> BOAT[Least-cost boat path]
  ROUTE --> API[FastAPI]
  BOAT --> API
  REP --> API
  API --> UI[Frontend map]
```

Cost formulas

- Road edge: `cost = length × (1 + 20 × flooded share + 2 × unknown share)`; verified-blocked edges are removed, verified-open edges lose the flood penalty.
- Boat cell: `cost = 1 + obstacle + uncertainty + terrain + report` (weights in `config/config.yaml → boat.penalties`).
