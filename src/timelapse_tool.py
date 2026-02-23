#!/usr/bin/env python3
"""Build a yearly satellite timelapse GIF for a lat/lon AOI."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import ee
import imageio.v2 as imageio
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


CURRENT_YEAR = dt.datetime.utcnow().year

LANDSAT_SOURCES: list[dict[str, Any]] = [
    {
        "id": "LANDSAT/LT05/C02/T1_L2",
        "start_year": 1984,
        "end_year": 2012,
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4"],
    },
    {
        "id": "LANDSAT/LE07/C02/T1_L2",
        "start_year": 1999,
        "end_year": CURRENT_YEAR,
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4"],
    },
    {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "start_year": 2013,
        "end_year": CURRENT_YEAR,
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5"],
    },
    {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "start_year": 2021,
        "end_year": CURRENT_YEAR,
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5"],
    },
]


@dataclass
class Config:
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
    ee_project: str | None
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a yearly satellite timelapse GIF from Earth Engine."
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
        help="landsat supports long history; sentinel2 starts in 2015.",
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
        help="Cloud percent threshold metadata filter.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=768,
        help="Square frame size in px (max 1024).",
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
        "--ee-project",
        type=str,
        help="Optional Google Cloud project for Earth Engine initialization.",
    )
    return parser.parse_args()


def latlon_radius_to_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_deg = radius_km / 110.574
    lon_deg = radius_km / (111.320 * max(math.cos(math.radians(lat)), 1e-6))
    west, east = lon - lon_deg, lon + lon_deg
    south, north = lat - lat_deg, lat + lat_deg
    return west, south, east, north


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    raw = [x.strip() for x in bbox_str.split(",")]
    if len(raw) != 4:
        raise ValueError("bbox must be west,south,east,north")
    west, south, east, north = [float(x) for x in raw]
    if west >= east or south >= north:
        raise ValueError("invalid bbox ordering; expected west < east and south < north")
    return west, south, east, north


def build_config(args: argparse.Namespace) -> Config:
    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")
    if args.end_year > CURRENT_YEAR:
        raise ValueError(f"end-year must be <= {CURRENT_YEAR}")
    if args.start_year < 1984:
        raise ValueError("start-year must be >= 1984 for available datasets")
    if args.sensor == "sentinel2" and args.start_year < 2015:
        raise ValueError("sentinel2 only supports years >= 2015")
    if args.frame_size < 128 or args.frame_size > 1024:
        raise ValueError("frame-size must be between 128 and 1024")
    if args.fps <= 0:
        raise ValueError("fps must be > 0")
    if args.cloud_threshold < 0 or args.cloud_threshold > 100:
        raise ValueError("cloud-threshold must be between 0 and 100")
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
        ee_project=args.ee_project,
        bbox=bbox,
    )


def init_earth_engine(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to initialize Earth Engine. Run `earthengine authenticate` first "
            "and pass --ee-project if required."
        ) from exc


def landsat_mask_and_scale(input_bands: list[str]) -> Callable[[ee.Image], ee.Image]:
    def _fn(image: ee.Image) -> ee.Image:
        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(1 << 1).eq(0)  # Dilated cloud
            .And(qa.bitwiseAnd(1 << 3).eq(0))  # Cloud
            .And(qa.bitwiseAnd(1 << 4).eq(0))  # Cloud shadow
            .And(qa.bitwiseAnd(1 << 5).eq(0))  # Snow
        )
        scaled = (
            image.select(input_bands, ["blue", "green", "red", "nir"])
            .multiply(0.0000275)
            .add(-0.2)
        )
        return scaled.updateMask(mask).copyProperties(image, image.propertyNames())

    return _fn


def sentinel2_mask_and_scale(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = (
        scl.neq(1)  # Saturated/defective
        .And(scl.neq(3))  # Cloud shadow
        .And(scl.neq(8))  # Medium probability cloud
        .And(scl.neq(9))  # High probability cloud
        .And(scl.neq(10))  # Thin cirrus
        .And(scl.neq(11))  # Snow/Ice
    )
    scaled = image.select(["B2", "B3", "B4", "B8"], ["blue", "green", "red", "nir"]).divide(10000)
    return scaled.updateMask(mask).copyProperties(image, image.propertyNames())


def get_year_collection(config: Config, year: int, aoi: ee.Geometry) -> tuple[ee.ImageCollection, list[str]]:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    if config.sensor == "sentinel2":
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.cloud_threshold))
            .map(sentinel2_mask_and_scale)
        )
        return coll, ["COPERNICUS/S2_SR_HARMONIZED"]

    merged: ee.ImageCollection | None = None
    datasets: list[str] = []
    for source in LANDSAT_SOURCES:
        if not (source["start_year"] <= year <= source["end_year"]):
            continue
        datasets.append(source["id"])
        coll = (
            ee.ImageCollection(source["id"])
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUD_COVER", config.cloud_threshold))
            .map(landsat_mask_and_scale(source["bands"]))
        )
        merged = coll if merged is None else merged.merge(coll)

    if merged is None:
        merged = ee.ImageCollection([])
    return merged, datasets


def apply_viz(image: ee.Image, viz: str) -> ee.Image:
    if viz == "true_color":
        return image.select(["red", "green", "blue"]).visualize(min=0.02, max=0.35, gamma=1.2)
    if viz == "false_color":
        return image.select(["nir", "red", "green"]).visualize(min=0.02, max=0.45, gamma=1.2)
    ndvi = image.normalizedDifference(["nir", "red"]).rename("ndvi")
    return ndvi.visualize(
        min=-0.2,
        max=0.8,
        palette=["#8c510a", "#f6e8c3", "#1b7837"],
    )


def fetch_png_bytes(image: ee.Image, region: dict[str, Any], frame_size: int) -> bytes:
    url = image.getThumbURL(
        {
            "region": region,
            "dimensions": f"{frame_size}x{frame_size}",
            "format": "png",
        }
    )
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content


def annotate_frame(frame_bytes: bytes, year: int) -> np.ndarray:
    image = Image.open(io.BytesIO(frame_bytes)).convert("RGBA")
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


def build_region_geojson(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def render_timelapse(config: Config) -> None:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if config.export_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = config.bbox
    aoi = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
    region = build_region_geojson(config.bbox)

    frames: list[np.ndarray] = []
    metadata: dict[str, Any] = {
        "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "bbox": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "start_year": config.start_year,
        "end_year": config.end_year,
        "sensor": config.sensor,
        "visualization": config.viz,
        "cloud_threshold": config.cloud_threshold,
        "frame_size": config.frame_size,
        "fps": config.fps,
        "frames": [],
    }

    for year in range(config.start_year, config.end_year + 1):
        print(f"Building frame {year}...")
        frame_meta: dict[str, Any] = {"year": year}
        try:
            collection, datasets = get_year_collection(config, year, aoi)
            count = int(collection.size().getInfo())
            frame_meta["image_count"] = count
            frame_meta["datasets"] = datasets
            if count == 0:
                frame = make_placeholder_frame(year, config.frame_size, "No imagery")
                frame_meta["status"] = "no_data"
            else:
                composite = collection.median().clip(aoi)
                vis = apply_viz(composite, config.viz)
                frame_bytes = fetch_png_bytes(vis, region, config.frame_size)
                frame = annotate_frame(frame_bytes, year)
                frame_meta["status"] = "ok"
        except Exception as exc:
            frame = make_placeholder_frame(year, config.frame_size, "Frame error")
            frame_meta["status"] = "error"
            frame_meta["error"] = str(exc)

        frames.append(frame)
        metadata["frames"].append(frame_meta)
        if config.export_frames:
            imageio.imwrite(frames_dir / f"{year}.png", frame)

    gif_path = output_dir / f"{config.name}.gif"
    imageio.mimsave(gif_path, frames, duration=1.0 / config.fps, loop=0)

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
    args = parse_args()
    try:
        config = build_config(args)
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc
    init_earth_engine(config.ee_project)
    render_timelapse(config)


if __name__ == "__main__":
    main()
