# Satellite Timelapse GIF Tool (No Signup)

Build a yearly satellite timelapse GIF for a location over 10-20 years using public STAC data (Earth Search on AWS Open Data).  
No Google Earth Engine account is required.

## Features

- AOI input as:
  - `--lat --lon --radius-km`
  - or `--bbox west,south,east,north`
- Time range by year (`--start-year`, `--end-year`)
- Sensor presets:
  - `landsat` (best for long history)
  - `sentinel2` (modern years, best detail)
- Visualization presets:
  - `true_color`
  - `false_color`
  - `ndvi`
- Output artifacts:
  - `*.gif`
  - optional `frames/*.png`
  - `metadata.json`
  - optional `*.mp4`

## Data Source

- Default STAC API: `https://earth-search.aws.element84.com/v1`
- Collection mapping:
  - `landsat` -> `landsat-c2-l2`
  - `sentinel2` -> `sentinel-2-l2a`

## Setup

1. Create and activate virtualenv.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

No authentication is needed for the default source.

## Usage

### Example 1: Landsat NDVI (long timeline)

```bash
python src/timelapse_tool.py \
  --lat 28.6139 --lon 77.2090 --radius-km 8 \
  --start-year 2006 --end-year 2025 \
  --sensor landsat \
  --viz ndvi \
  --cloud-threshold 30 \
  --frame-size 768 \
  --fps 2 \
  --name delhi_growth \
  --output-dir output \
  --export-frames
```

### Example 2: Sentinel-2 false color

```bash
python src/timelapse_tool.py \
  --bbox -74.10,40.60,-73.70,40.95 \
  --start-year 2018 --end-year 2025 \
  --sensor sentinel2 \
  --viz false_color \
  --name nyc_s2 \
  --output-dir output
```

### Optional controls

- `--max-items-per-year` (default `120`): limit fetched items.
- `--composite-items` (default `40`): number of least-cloudy scenes used in yearly composite.
- `--stac-api`: override STAC endpoint.
- `--mp4`: also write MP4 (if your local ffmpeg/imageio setup supports it).

## Output Structure

```text
output/
  timelapse.gif
  metadata.json
  frames/           # only with --export-frames
    2006.png
    ...
```

`metadata.json` includes per-year item counts, selected band assets, normalization ranges, and errors (if any).
