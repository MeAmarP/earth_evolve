#!/usr/bin/env python3
"""Build a yearly satellite timelapse GIF from public STAC data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import stackstac
from PIL import Image, ImageDraw, ImageFont
from pystac import Item
from pystac_client import Client

# Configure GDAL/rasterio to access public S3 data without AWS credentials
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

CURRENT_YEAR = dt.datetime.utcnow().year
STAC_API_DEFAULT = "https://earth-search.aws.element84.com/v1"
SENSOR_TO_COLLECTION = {
    "landsat": "landsat-c2-l2",
    "sentinel2": "sentinel-2-l2a",
}


@dataclass
class Config:
    """Validated runtime configuration shared across the pipeline."""

    start_year: int
    end_year: int
    sensor: str
    viz: str
    cloud_threshold: int
    frame_size: int
    fps: float
    output_dir: Path
    name: str
    export_frames: bool
    make_mp4: bool
    bbox: tuple[float, float, float, float]
    stac_api: str
    max_items_per_year: int
    composite_items: int


def parse_args() -> argparse.Namespace:
    """Define the CLI and return parsed arguments.
    
    Args:
        None
    
    Returns:
        argparse.Namespace: Parsed command-line arguments including lat, lon, radius-km,
            bbox, start-year, end-year, sensor, viz, cloud-threshold, frame-size,
            fps, output-dir, name, export-frames, mp4, stac-api, max-items-per-year,
            and composite-items.
    """

    parser = argparse.ArgumentParser(
        description="Generate a yearly satellite timelapse GIF from public STAC collections."
    )
    parser.add_argument("--lat", type=float, help="Latitude of center point.")
    parser.add_argument("--lon", type=float, help="Longitude of center point.")
    parser.add_argument(
        "--radius-km",
        type=float,
        default=5.0,
        help="Radius in km around lat/lon (ignored if --bbox is set).",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as west,south,east,north.",
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--sensor",
        choices=["landsat", "sentinel2"],
        default="landsat",
        help="landsat supports longer history; sentinel2 has modern coverage.",
    )
    parser.add_argument(
        "--viz",
        choices=["true_color", "false_color", "ndvi"],
        default="true_color",
    )
    parser.add_argument(
        "--cloud-threshold",
        type=int,
        default=30,
        help="Cloud percent threshold filter (eo:cloud_cover <= value).",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=768,
        help="Maximum frame dimension in px (max 1024).",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--name", type=str, default="timelapse")
    parser.add_argument(
        "--export-frames",
        action="store_true",
        help="Write individual PNG frames in output/frames.",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Also attempt MP4 export using imageio ffmpeg plugin.",
    )
    parser.add_argument(
        "--stac-api",
        type=str,
        default=STAC_API_DEFAULT,
        help=f"STAC API endpoint (default: {STAC_API_DEFAULT}).",
    )
    parser.add_argument(
        "--max-items-per-year",
        type=int,
        default=120,
        help="Maximum STAC items fetched per year before filtering.",
    )
    parser.add_argument(
        "--composite-items",
        type=int,
        default=40,
        help="Top N least-cloudy items used for yearly composite.",
    )
    return parser.parse_args()


def latlon_radius_to_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Approximate a square lon/lat bounding box around a center point.
    
    Args:
        lat (float): Latitude of the center point in degrees.
        lon (float): Longitude of the center point in degrees.
        radius_km (float): Radius around the center point in kilometers.
    
    Returns:
        tuple[float, float, float, float]: Bounding box as (west, south, east, north) in degrees.
    """

    lat_deg = radius_km / 110.574
    lon_deg = radius_km / (111.320 * max(math.cos(math.radians(lat)), 1e-6))
    west, east = lon - lon_deg, lon + lon_deg
    south, north = lat - lat_deg, lat + lat_deg
    return west, south, east, north


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse a `west,south,east,north` string into numeric bounds.
    
    Args:
        bbox_str (str): Comma-separated bounding box string in format 'west,south,east,north'.
    
    Returns:
        tuple[float, float, float, float]: Bounding box as (west, south, east, north) in degrees.
    
    Raises:
        ValueError: If bbox format is invalid or bounds are logically incorrect.
    """

    raw = [x.strip() for x in bbox_str.split(",")]
    if len(raw) != 4:
        raise ValueError("bbox must be west,south,east,north")
    west, south, east, north = [float(x) for x in raw]
    if west >= east or south >= north:
        raise ValueError("invalid bbox ordering; expected west < east and south < north")
    return west, south, east, north


def build_config(args: argparse.Namespace) -> Config:
    """Validate CLI input and normalize it into a Config object.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        Config: Validated configuration dataclass with all required parameters.
    
    Raises:
        ValueError: If any input validation check fails.
    """

    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")
    if args.end_year > CURRENT_YEAR:
        raise ValueError(f"end-year must be <= {CURRENT_YEAR}")
    if args.start_year < 1984:
        raise ValueError("start-year must be >= 1984 for Landsat history")
    if args.sensor == "sentinel2" and args.start_year < 2015:
        raise ValueError("sentinel2 only supports years >= 2015")
    if args.frame_size < 128 or args.frame_size > 1024:
        raise ValueError("frame-size must be between 128 and 1024")
    if args.fps <= 0:
        raise ValueError("fps must be > 0")
    if args.cloud_threshold < 0 or args.cloud_threshold > 100:
        raise ValueError("cloud-threshold must be between 0 and 100")
    if args.max_items_per_year <= 0:
        raise ValueError("max-items-per-year must be > 0")
    if args.composite_items <= 0:
        raise ValueError("composite-items must be > 0")
    if args.composite_items > args.max_items_per_year:
        raise ValueError("composite-items must be <= max-items-per-year")

    if args.bbox:
        bbox = parse_bbox(args.bbox)
    else:
        if args.lat is None or args.lon is None:
            raise ValueError("provide --bbox or both --lat and --lon")
        if args.radius_km <= 0:
            raise ValueError("radius-km must be positive")
        bbox = latlon_radius_to_bbox(args.lat, args.lon, args.radius_km)

    return Config(
        start_year=args.start_year,
        end_year=args.end_year,
        sensor=args.sensor,
        viz=args.viz,
        cloud_threshold=args.cloud_threshold,
        frame_size=args.frame_size,
        fps=args.fps,
        output_dir=args.output_dir,
        name=args.name,
        export_frames=args.export_frames,
        make_mp4=args.mp4,
        bbox=bbox,
        stac_api=args.stac_api,
        max_items_per_year=args.max_items_per_year,
        composite_items=args.composite_items,
    )


def query_items(config: Config, client: Client, year: int) -> list[Item]:
    """Fetch and sort STAC items for one year by increasing cloud cover.
    
    Args:
        config (Config): Runtime configuration with sensor, cloud threshold, and bounding box.
        client (Client): Authenticated STAC API client.
        year (int): Year to query for imagery.
    
    Returns:
        list[Item]: List of STAC Items sorted by cloud cover, limited to composite_items count.
    """

    collection_id = SENSOR_TO_COLLECTION[config.sensor]
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"
    search = client.search(
        collections=[collection_id],
        bbox=list(config.bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lte": config.cloud_threshold}},
        max_items=config.max_items_per_year,
    )
    items = list(search.items())
    items.sort(
        key=lambda it: (
            float(it.properties.get("eo:cloud_cover", 1000.0)),
            str(it.datetime or ""),
        )
    )
    return items[: config.composite_items]


def pick_band_key(items: list[Item], candidates: list[str]) -> str | None:
    """Find the first asset key matching one of the requested band aliases.
    
    Args:
        items (list[Item]): List of STAC Items to search (checks up to 10 items).
        candidates (list[str]): List of band name aliases to match (case-insensitive).
    
    Returns:
        str | None: The asset key name if found, None otherwise.
    """

    lower_candidates = [c.lower() for c in candidates]
    for item in items[: min(len(items), 10)]:
        for key, asset in item.assets.items():
            key_l = key.lower()
            if key_l in lower_candidates:
                return key
            eo_bands = asset.extra_fields.get("eo:bands")
            if isinstance(eo_bands, list):
                for band in eo_bands:
                    if isinstance(band, dict):
                        common_name = str(band.get("common_name", "")).lower()
                        name = str(band.get("name", "")).lower()
                        if common_name in lower_candidates or name in lower_candidates:
                            return key
    return None


def resolve_asset_keys(items: list[Item], sensor: str) -> dict[str, str]:
    """Map logical band names (`blue/green/red/nir`) to collection asset keys.
    
    Args:
        items (list[Item]): List of STAC Items from the target collection.
        sensor (str): Sensor type ('sentinel2' or 'landsat') to determine band names.
    
    Returns:
        dict[str, str]: Mapping of {'blue', 'green', 'red', 'nir'} to actual asset keys.
    
    Raises:
        RuntimeError: If required RGB/NIR assets cannot be found in the collection.
    """

    if sensor == "sentinel2":
        blue = pick_band_key(items, ["blue", "b02"])
        green = pick_band_key(items, ["green", "b03"])
        red = pick_band_key(items, ["red", "b04"])
        nir = pick_band_key(items, ["nir", "nir08", "b08"])
    else:
        blue = pick_band_key(items, ["blue", "sr_b2", "sr_b1"])
        green = pick_band_key(items, ["green", "sr_b3", "sr_b2"])
        red = pick_band_key(items, ["red", "sr_b4", "sr_b3"])
        nir = pick_band_key(items, ["nir", "nir08", "sr_b5", "sr_b4"])

    if not all([blue, green, red, nir]):
        raise RuntimeError("Could not map required RGB/NIR assets for selected collection.")
    return {"blue": blue, "green": green, "red": red, "nir": nir}


def filter_items_with_assets(items: list[Item], asset_keys: dict[str, str]) -> list[Item]:
    """Drop STAC items that do not expose every required band asset.
    
    Args:
        items (list[Item]): List of STAC Items to filter.
        asset_keys (dict[str, str]): Mapping of band names to required asset keys.
    
    Returns:
        list[Item]: Filtered list containing only items with all required assets.
    """

    required = list(asset_keys.values())
    filtered = [item for item in items if all(k in item.assets for k in required)]
    return filtered


def target_resolution_meters(bbox: tuple[float, float, float, float], frame_size: int, sensor: str) -> float:
    """Estimate output pixel size in meters, respecting native sensor resolution.
    
    Args:
        bbox (tuple[float, float, float, float]): Bounding box as (west, south, east, north) in degrees.
        frame_size (int): Target frame dimension in pixels.
        sensor (str): Sensor type ('sentinel2' or 'landsat') to determine native resolution.
    
    Returns:
        float: Target resolution in meters per pixel, respecting minimum native resolution.
    """

    west, south, east, north = bbox
    lat_mid = (south + north) / 2
    width_m = abs(east - west) * 111_320 * max(math.cos(math.radians(lat_mid)), 1e-6)
    height_m = abs(north - south) * 110_574
    max_span = max(width_m, height_m)
    approx_resolution = max_span / frame_size
    native = 10.0 if sensor == "sentinel2" else 30.0
    return max(native, approx_resolution)


def fetch_year_composite(
    config: Config,
    items: list[Item],
    asset_keys: dict[str, str],
    resolution_m: float,
) -> np.ndarray:
    """Read selected scenes, stack RGB+NIR bands, and build a yearly median composite.
    
    Args:
        config (Config): Runtime configuration with bounding box and projection settings.
        items (list[Item]): List of STAC Items to composite.
        asset_keys (dict[str, str]): Mapping of band names to asset keys.
        resolution_m (float): Target resolution in meters per pixel.
    
    Returns:
        np.ndarray: 4D array of shape (4, height, width) containing [blue, green, red, nir] bands.
    
    Raises:
        RuntimeError: If no items have required assets or composite is empty/nodata.
    """

    usable_items = filter_items_with_assets(items, asset_keys)
    if not usable_items:
        raise RuntimeError("No items contain all required RGB/NIR assets.")

    ordered_assets = [
        asset_keys["blue"],
        asset_keys["green"],
        asset_keys["red"],
        asset_keys["nir"],
    ]
    data = stackstac.stack(
        usable_items,
        assets=ordered_assets,
        bounds_latlon=config.bbox,
        epsg=3857,
        resolution=resolution_m,
        fill_value=np.nan,
        rescale=True,
    )
    arr = data.transpose("time", "band", "y", "x").astype("float32").compute().values
    if arr.size == 0:
        raise RuntimeError("Composite array is empty after stacking.")
    with np.errstate(invalid="ignore"):
        composite = np.nanmedian(arr, axis=0)
    if not np.isfinite(composite).any():
        raise RuntimeError("Composite contains only nodata pixels.")
    return composite


def robust_min_max(values: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> tuple[float, float]:
    """Return percentile-based display bounds while ignoring nodata values.
    
    Args:
        values (np.ndarray): Input array with potentially non-finite values.
        low_pct (float): Lower percentile for minimum value (default: 2.0).
        high_pct (float): Upper percentile for maximum value (default: 98.0).
    
    Returns:
        tuple[float, float]: (min_value, max_value) for display normalization.
    """

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale values into the `[0, 1]` display range and clamp outliers.
    
    Args:
        values (np.ndarray): Input array to normalize.
        lo (float): Lower bound of the input range.
        hi (float): Upper bound of the input range.
    
    Returns:
        np.ndarray: Normalized array with values clipped to [0.0, 1.0].
    """

    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a hex color string into an RGB tuple.
    
    Args:
        color (str): Hex color string in format '#RRGGBB' or 'RRGGBB'.
    
    Returns:
        tuple[int, int, int]: RGB values as (red, green, blue) with range [0, 255].
    """

    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def ndvi_to_rgb(ndvi: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Colorize NDVI values using a simple brown-to-green gradient.
    
    Args:
        ndvi (np.ndarray): NDVI array with values typically in range [-1, 1].
        lo (float): Lower bound of the NDVI range for color mapping.
        hi (float): Upper bound of the NDVI range for color mapping.
    
    Returns:
        np.ndarray: uint8 RGB image array of shape (height, width, 3).
    """

    palette = [hex_to_rgb("#8c510a"), hex_to_rgb("#f6e8c3"), hex_to_rgb("#1b7837")]
    x = normalize(ndvi, lo, hi)
    xp = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    red = np.interp(x, xp, [palette[0][0], palette[1][0], palette[2][0]])
    green = np.interp(x, xp, [palette[0][1], palette[1][1], palette[2][1]])
    blue = np.interp(x, xp, [palette[0][2], palette[1][2], palette[2][2]])
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def letterbox_to_square(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a frame into a square canvas without distorting aspect ratio.
    
    Args:
        frame (np.ndarray): Input RGB image array.
        size (int): Target square dimension in pixels.
    
    Returns:
        np.ndarray: uint8 RGB image padded/letterboxed to (size, size, 3).
    """

    image = Image.fromarray(frame, mode="RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(20, 20, 20))
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(resized, (x, y))
    return np.asarray(canvas)


def annotate_frame(frame: np.ndarray, year: int) -> np.ndarray:
    """Overlay the frame year in the top-left corner.
    
    Args:
        frame (np.ndarray): Input RGB image array.
        year (int): Year to display as text annotation.
    
    Returns:
        np.ndarray: uint8 RGB image with year text overlaid.
    """

    image = Image.fromarray(frame, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    text = str(year)
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw = right - left
    th = bottom - top
    pad = 8
    x = 12
    y = 12
    draw.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image.convert("RGB"))


def make_placeholder_frame(year: int, frame_size: int, message: str) -> np.ndarray:
    """Create a fallback frame when imagery is missing or processing fails.
    
    Args:
        year (int): Year to display on the frame.
        frame_size (int): Dimension for the square output image in pixels.
        message (str): Error or status message to display.
    
    Returns:
        np.ndarray: uint8 RGB image array of shape (frame_size, frame_size, 3).
    """

    image = Image.new("RGB", (frame_size, frame_size), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{year}\n{message}"
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(
        ((frame_size - tw) / 2, (frame_size - th) / 2),
        text,
        fill=(240, 240, 240),
        font=font,
        align="center",
    )
    return np.asarray(image)


def compute_global_stats(composites: list[np.ndarray | None]) -> dict[str, tuple[float, float]]:
    """Derive shared normalization ranges so all frames use a consistent scale.
    
    Args:
        composites (list[np.ndarray | None]): List of yearly composite arrays or None if failed.
            Each array has shape (4, height, width) with [blue, green, red, nir] bands.
    
    Returns:
        dict[str, tuple[float, float]]: Normalization bounds for 'blue', 'green', 'red', 'nir', 'ndvi'.
    """

    valid = [c for c in composites if c is not None]
    if not valid:
        return {
            "blue": (0.0, 1.0),
            "green": (0.0, 1.0),
            "red": (0.0, 1.0),
            "nir": (0.0, 1.0),
            "ndvi": (-0.2, 0.8),
        }
    stack = np.stack(valid, axis=0)
    stats = {
        "blue": robust_min_max(stack[:, 0, :, :]),
        "green": robust_min_max(stack[:, 1, :, :]),
        "red": robust_min_max(stack[:, 2, :, :]),
        "nir": robust_min_max(stack[:, 3, :, :]),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (stack[:, 3, :, :] - stack[:, 2, :, :]) / (stack[:, 3, :, :] + stack[:, 2, :, :] + 1e-6)
    ndvi_lo, ndvi_hi = robust_min_max(ndvi)
    stats["ndvi"] = (max(-1.0, ndvi_lo), min(1.0, ndvi_hi))
    return stats


def render_composite(composite: np.ndarray, viz: str, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Render a composite into an RGB frame for the chosen visualization preset.
    
    Args:
        composite (np.ndarray): Input array of shape (4, height, width) with [blue, green, red, nir].
        viz (str): Visualization type ('true_color', 'false_color', or 'ndvi').
        stats (dict[str, tuple[float, float]]): Normalization bounds for each band and NDVI.
    
    Returns:
        np.ndarray: uint8 RGB image array of shape (height, width, 3).
    """

    blue, green, red, nir = composite
    if viz == "true_color":
        r = normalize(red, *stats["red"])
        g = normalize(green, *stats["green"])
        b = normalize(blue, *stats["blue"])
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)
    if viz == "false_color":
        r = normalize(nir, *stats["nir"])
        g = normalize(red, *stats["red"])
        b = normalize(green, *stats["green"])
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi_to_rgb(ndvi, *stats["ndvi"])


def render_timelapse(config: Config) -> None:
    """Run the full workflow: query imagery, build frames, and write outputs.
    
    Args:
        config (Config): Runtime configuration with all user settings.
    
    Returns:
        None. Writes GIF, optional PNG frames, optional MP4, and metadata.json to output directory.
    """

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if config.export_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    client = Client.open(config.stac_api)
    resolution_m = target_resolution_meters(config.bbox, config.frame_size, config.sensor)

    frame_records: list[dict[str, Any]] = []
    composites: list[np.ndarray | None] = []

    for year in range(config.start_year, config.end_year + 1):
        print(f"Fetching imagery for {year}...")
        frame_meta: dict[str, Any] = {"year": year, "collection": SENSOR_TO_COLLECTION[config.sensor]}
        try:
            items = query_items(config, client, year)
            frame_meta["image_count"] = len(items)
            if not items:
                composites.append(None)
                frame_meta["status"] = "no_data"
                frame_records.append(frame_meta)
                continue
            asset_keys = resolve_asset_keys(items, config.sensor)
            frame_meta["assets"] = asset_keys
            composite = fetch_year_composite(config, items, asset_keys, resolution_m)
            composites.append(composite)
            frame_meta["status"] = "ok"
        except Exception as exc:
            composites.append(None)
            frame_meta["status"] = "error"
            frame_meta["error"] = str(exc)
        frame_records.append(frame_meta)

    stats = compute_global_stats(composites)
    frames: list[np.ndarray] = []
    for idx, year in enumerate(range(config.start_year, config.end_year + 1)):
        composite = composites[idx]
        if composite is None:
            status = frame_records[idx].get("status", "error")
            msg = "No imagery" if status == "no_data" else "Frame error"
            frame = make_placeholder_frame(year, config.frame_size, msg)
        else:
            rendered = render_composite(composite, config.viz, stats)
            squared = letterbox_to_square(rendered, config.frame_size)
            frame = annotate_frame(squared, year)
        frames.append(frame)
        if config.export_frames:
            imageio.imwrite(frames_dir / f"{year}.png", frame)

    gif_path = output_dir / f"{config.name}.gif"
    imageio.mimsave(gif_path, frames, duration=1.0 / config.fps, loop=0)

    metadata: dict[str, Any] = {
        "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stac_api": config.stac_api,
        "collection": SENSOR_TO_COLLECTION[config.sensor],
        "bbox": {
            "west": config.bbox[0],
            "south": config.bbox[1],
            "east": config.bbox[2],
            "north": config.bbox[3],
        },
        "start_year": config.start_year,
        "end_year": config.end_year,
        "sensor": config.sensor,
        "visualization": config.viz,
        "cloud_threshold": config.cloud_threshold,
        "frame_size": config.frame_size,
        "fps": config.fps,
        "max_items_per_year": config.max_items_per_year,
        "composite_items": config.composite_items,
        "resolution_meters": resolution_m,
        "normalization_stats": {
            key: {"min": value[0], "max": value[1]} for key, value in stats.items()
        },
        "frames": frame_records,
    }

    if config.make_mp4:
        mp4_path = output_dir / f"{config.name}.mp4"
        try:
            writer = imageio.get_writer(mp4_path, fps=config.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            metadata["mp4"] = str(mp4_path)
        except Exception as exc:
            metadata["mp4_error"] = f"MP4 export failed: {exc}"

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"GIF written: {gif_path}")
    if config.export_frames:
        print(f"Frames written: {frames_dir}")
    print(f"Metadata written: {metadata_path}")


def main() -> None:
    """CLI entry point.
    
    Args:
        None (reads from sys.argv)
    
    Returns:
        None. Initiates the full timelapse rendering pipeline.
    """

    args = parse_args()
    try:
        config = build_config(args)
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc
    render_timelapse(config)


if __name__ == "__main__":
    main()
