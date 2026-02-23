# Satellite Timelapse GIF Tool

Build a yearly satellite timelapse GIF for a location over 10-20 years using Google Earth Engine.

## Features

- AOI input as:
  - `--lat --lon --radius-km`
  - or `--bbox west,south,east,north`
- Time range by year (`--start-year`, `--end-year`)
- Sensor presets:
  - `landsat` (long history, good for 10-20 years)
  - `sentinel2` (2015+)
- Visualization presets:
  - `true_color`
  - `false_color`
  - `ndvi`
- Outputs:
  - `*.gif`
  - optional `frames/*.png`
  - `metadata.json`
  - optional `*.mp4`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Authenticate Earth Engine:

```bash
earthengine authenticate
```

If your account requires an explicit project, pass it with `--ee-project`.

## Usage

### Example 1: City center + radius

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

### Example 2: Explicit bounding box

```bash
python src/timelapse_tool.py \
  --bbox -74.10,40.60,-73.70,40.95 \
  --start-year 2015 --end-year 2025 \
  --sensor sentinel2 \
  --viz false_color \
  --name nyc_s2 \
  --output-dir output
```

### Optional MP4 output

Add `--mp4` to also attempt MP4 writing (depends on local ffmpeg/imageio setup).

## Output Structure

```text
output/
  timelapse.gif
  metadata.json
  frames/           # only with --export-frames
    2006.png
    ...
```

`metadata.json` stores the years, image counts, selected datasets, and frame statuses.
