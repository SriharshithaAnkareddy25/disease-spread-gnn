# Disease Spread Prediction using GNNs — Data + Graph Construction

Person 1's deliverables: district-level dataset, 4 graph variants, and the
preprocessing code that builds them. This is the input layer for Person 2's
spatio-temporal GNN and Person 3's SIR/evaluation stage.

## Quickstart

`data/processed/` (the actual deliverables below) is committed, so if you
just want to consume the dataset/graphs, pulling the repo is enough — no
need to run anything. To regenerate them yourself, or to get the raw source
data (not committed — see below):

```
pip install -r requirements.txt

# fetch raw data (not committed - ~870MB, mostly one huge upstream repo we only need one file from)
git clone --depth 1 https://github.com/datameet/maps.git data/raw/boundaries/datameet-maps
curl -o data/raw/census/india-districts-census-2011.csv https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv
curl -o data/raw/covid_cases/districts.csv https://data.incovid19.org/csv/latest/districts.csv

python src/graph_construction.py    # builds nodes + geographic/gravity graphs
python src/merge_case_data.py       # builds weekly case-data features
python src/visualize_graph.py       # renders the adjacency map
```

Run in that order — `merge_case_data.py` and `visualize_graph.py` both read
files `graph_construction.py` produces. `data/raw/` is gitignored (~870MB,
almost all of it irrelevant shapefiles bundled in the Datameet repo we only
need one file from) — re-run the fetch commands above after cloning.

## What's in `data/processed/` (the actual deliverables)

| File | What it is |
|---|---|
| `nodes.csv` | 640 districts: `censuscode`, `DISTRICT`, `ST_NM`, `Population`, `area_km2`, `density_per_km2` |
| `nodes.geojson` | Same, with polygon geometry (for mapping/plotting) |
| `node_order.csv` | Row/col index → `censuscode` for the two `.npz` matrices below — **use this to align node features with the adjacency matrices** |
| `geographic_adjacency.npz` | 640×640 sparse binary adjacency matrix (scipy `csr`), shared-border edges |
| `gravity_adjacency.npz` | 640×640 sparse weighted adjacency matrix (scipy `csr`), gravity-model edges |
| `edges_geographic_adjacency.csv` | Same graph as an edge list (`src`, `dst`, `weight=1`) |
| `edges_gravity.csv` | Same, sparsified gravity edges (`src`, `dst`, `weight`) |
| `edges_gravity_dense.csv` | All 204,480 pairwise gravity weights, unthresholded — use this if the top-6 sparsification isn't what you want |
| `weekly_case_features.csv` | `censuscode`, `week`, `Confirmed`, `Recovered`, `Deceased`, `Tested` — weekly, per district |
| `case_data_unmatched_districts.csv` | Case-data districts that couldn't be reconciled to a 2011 node (see Known Gaps) |
| `geographic_adjacency_graph.png` | Visualization |

`censuscode` is the join key across every file — use it to line up node
features, adjacency matrices, and case-data rows.

## The 4 graph variants — status

1. **Geographic adjacency** — done. `geographic_adjacency.npz` / `edges_geographic_adjacency.csv`.
2. **Gravity model** — done. `w_ij = (pop_i * pop_j) / dist_ij²`, sparsified to each node's top-6 heaviest edges (see Design Notes for why). `gravity_adjacency.npz` / `edges_gravity.csv`.
3. **Raw mobility** — **not built** — district-level OD mobility data isn't publicly available for India (Facebook/Google mobility only goes to state level). The gravity model is the agreed stand-in for this variant, as discussed up front — this is the same limitation multiple papers in the lit review flag.
4. **Learned graph** — needs no edge file. Hand the GNN `nodes.csv` (or whatever node-feature table you build from `nodes.csv` + `weekly_case_features.csv`) and let it learn adjacency as a parameter.

## Data sources

- **Boundaries + geometry**: Datameet `maps` repo, 2011 Census district shapefiles (`data/raw/boundaries/`)
- **Population**: `nishusharma1608/India-Census-2011-Analysis`, 2011 census district CSV (`data/raw/census/`)
- **Disease cases**: `data.incovid19.org`, daily district-wise Confirmed/Recovered/Deceased/Tested, 2020-04-26 to 2023-08-22 (`data/raw/covid_cases/`)
- Area/density are computed from geometry (Albers Equal-Area projection), not sourced directly — the raw census file doesn't include area.

## Known gaps (read before you rely on population/case coverage)

- **Case-data coverage: 567/640 districts, 90.9% of population.** The case
  data uses present-day districts; the node table uses 2011-census
  districts. Reconciled via `src/district_aliases.py` (handles Telangana's
  2014 split from Andhra Pradesh + 2016/2019 district carve-outs, a handful
  of Karnataka renames, and Ladakh's 2019 split from J&K). **Six of the nine
  Telangana splits are confirmed directly from Wikipedia**; three
  (Khammam, Mahbubnagar, Nalgonda splits) are filled in from general public
  record, not independently verified — flagged in the file if you want to
  double-check or extend it (e.g. for Arunachal Pradesh, present-day Andhra
  Pradesh's own 2022 split, and Delhi, which remain unreconciled).
- **Remaining unmatched districts** are listed in
  `case_data_unmatched_districts.csv` rather than silently dropped.
- **46 of the 567 matched districts have fewer than 100 of the 174 weeks**
  (median is a full 174) — mostly newer administrative units that didn't
  exist as separate reporting districts for the full 2020-2023 window.
  Worth masking/padding explicitly rather than assuming a dense panel.
- One negative value in the raw case data (an upstream correction/revision)
  is clipped to 0 during aggregation.
- 4 nodes are legitimately edgeless in the geographic-adjacency graph:
  the 3 Andaman & Nicobar districts and Lakshadweep have no land border with
  anywhere, by definition.

## Design notes (the non-obvious choices, in case you need to change them)

- **Join key is `censuscode`** (shapefile) / `District code` (census CSV),
  not district name strings — names mismatch across sources, codes don't
  (640/641 match on the first try).
- **Geographic adjacency uses `.intersects()`, not `.touches()`** — the
  shapefile's borders have microscopic floating-point overlap slivers, so
  `.touches()` (which requires exactly zero intersection) silently drops
  genuinely adjacent districts. Verified against known-adjacent pairs
  (e.g. Allahabad/Chitrakoot) before and after the fix.
- **Gravity edges are top-6-per-node, not a global weight threshold** — raw
  gravity weights span ~7 orders of magnitude, so any single global cutoff
  either keeps a dense metro-only clique or strands remote districts
  entirely. Top-k per node guarantees every node keeps its k strongest
  partners regardless of absolute weight scale (`edges_gravity_dense.csv`
  has the untouched pairwise values if you want to threshold differently).
