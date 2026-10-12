"""Small, offline LAS/LAZ ML prototype. See README_ZH.txt for scope."""
from __future__ import annotations

import base64
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import laspy
import numpy as np
import sklearn
from pyproj import CRS
from scipy.ndimage import generic_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "0.1.0-prototype"
DISPLAY_LIMIT = 30_000
WEB_LIMIT = 3_000_000  # Points in the browser (WebGL) 3D view.
WEB_TEMPLATE = Path(__file__).resolve().parent / "viewer3d.html"
# Measured peak when a whole file is held in memory (training) is ~430 bytes per point; keep headroom.
BYTES_PER_POINT = 600
# Processing streams the file and works tile by tile; per-point results stay in memory (~20 bytes per point)
# and each point is stored once in a temporary tile file next to the results.
PROCESS_BYTES_PER_POINT = 40
TILE_POINTS = 1_000_000  # Average points per processing tile; bounds the feature / k-NN memory.
CHUNK_POINTS = 1_000_000  # Points decoded at a time when streaming a LAS/LAZ (a multiple of the LAZ chunk size).
TILE_RECORD = np.dtype([("index", "<i8"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<u2"),
                        ("return_number", "u1"), ("number_of_returns", "u1"), ("scan_angle", "<f8"),
                        ("classification", "u1")])
WORKERS = os.cpu_count() or 1
HEADERS = ["object ID", "x", "y", "z", "Nr_of_points", "Intensity", "Area", "confidence"]
FEATURES = ["local_height", "roughness", "linearity", "planarity", "sphericity",
            "radius", "relative_intensity", "raw_intensity_scaled", "return_position",
            "multi_return", "last_return", "scan_angle", "return_missing",
            "height_above_bed", "bed_known", "las_bathymetric_bottom", "las_water_surface",
            "las_water_column", "las_ground", "las_noise"]
# ASPRS LAS 1.4 (R15) classification codes, with the topo-bathy codes 40-45 that YellowScan uses.
# All class numbers in this program are these codes. Input codes are kept on every point the model does
# not mark as a target. Codes not listed here and below 64 are reserved by the standard.
LAS_CLASSES = {0: "created, never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
               4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "low point (noise)",
               9: "water", 10: "rail", 11: "road surface", 13: "wire - guard (shield)",
               14: "wire - conductor (phase)", 15: "transmission tower", 16: "wire-structure connector",
               17: "bridge deck", 18: "high noise", 19: "overhead structure", 20: "ignored ground",
               21: "snow", 22: "temporal exclusion", 40: "bathymetric point (bottom)", 41: "water surface",
               42: "derived water surface", 43: "submerged object", 44: "IHO S-57 object", 45: "water column"}
# LAS 1.0-1.3 define fewer classes (no user-definable range, no topo-bathy codes) and store only 0-31.
LAS12_CLASSES = {0: "created, never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
                 4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "low point (noise)",
                 8: "model key-point", 9: "water", 12: "overlap points"}
LAS_OUTPUTS = ("same as input", "1.4")
USER_CLASS_MIN = 64  # 64..255 are user-definable in the standard; own target classes must use these codes.
NOT_TARGET = 0  # Internal model label for points that are not a target; never written to LAS output.


def check_target_code(code):
    """A trained target class must be a defined ASPRS class or a user-definable code (64-255)."""
    if code in (0, 1):
        raise ValueError(f"Class {code} ({LAS_CLASSES[code]}) means a point has no class and cannot be a target")
    if not (code in LAS_CLASSES or USER_CLASS_MIN <= code <= 255):
        raise ValueError(f"Class {code} is reserved in the ASPRS LAS standard; use a defined class "
                         f"(e.g. 43 submerged object) or a user-definable code {USER_CLASS_MIN}-255")
BED_CELL = 1.0  # metres; grid used to estimate the local riverbed / seabed surface.
BED_WINDOW = 5  # cells; median over this window, so objects up to ~2 m do not lift the bed.


@dataclass
class Settings:
    neighbors: int = 20
    max_points: int = 0  # 0 = no limit; whole file is processed
    threshold: float = 0.7
    cluster_eps: float = 0.3
    min_points: int = 10
    area_method: str = "unresolved"
    crs_override: str = ""
    coordinates_in_meters: bool = False
    vertical_datum: str = "unspecified"
    las_output: str = "same as input"  # Version of classified.laz: "same as input" or "1.4".


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def validate_settings(cfg):
    if not 3 <= cfg.neighbors <= 128 or not (cfg.max_points == 0 or cfg.max_points >= 10):
        raise ValueError("neighbors must be 3..128; max_points must be 0 (no limit) or at least 10")
    if not 0 <= cfg.threshold <= 1 or cfg.cluster_eps <= 0 or cfg.min_points < 2:
        raise ValueError("Invalid probability threshold / cluster radius / minimum points")
    if cfg.area_method not in ("unresolved", "convex_hull_xy"):
        raise ValueError("Unsupported area method")
    if cfg.las_output not in LAS_OUTPUTS:
        raise ValueError(f"Classified LAS version must be one of {LAS_OUTPUTS}")


def physical_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # Windows: no sysconf; skip the check.
        return None


def check_memory(count, need=None):
    """Refuse early, before decoding, when a file clearly cannot fit in RAM."""
    total = physical_memory()
    need = count * BYTES_PER_POINT if need is None else need
    if total and need > total * 0.8:
        raise ValueError(f"{count:,} points needs about {need / 1024**3:.1f} GB RAM; this computer has "
                         f"{total / 1024**3:.1f} GB. Split the file (e.g. by area) before processing.")


def process_memory(count):
    """RAM needed to process a file: per-point results plus about two tiles of whole-file working memory."""
    return count * PROCESS_BYTES_PER_POINT + min(count, 2 * TILE_POINTS) * BYTES_PER_POINT


def check_disk(folder, count):
    free = shutil.disk_usage(folder).free
    need = count * TILE_RECORD.itemsize
    if need > free * 0.9:
        raise ValueError(f"{count:,} points needs about {need / 1024**3:.1f} GB free disk space for temporary "
                         f"tiles in the results directory; it has {free / 1024**3:.1f} GB")


def inspect_file(path):
    """Header-only summary: no points are decoded and no settings are needed."""
    path = Path(path)
    with laspy.open(path) as reader:
        header = reader.header
        dims = set(header.point_format.dimension_names)
        try:
            crs = header.parse_crs()
        except Exception as exc:  # Malformed CRS records should not hide the rest of the summary.
            crs, crs_note = None, f"CRS unreadable: {exc}"
        else:
            crs_note = None
    count = header.point_count
    issues = []
    missing = {"intensity", "return_number", "number_of_returns"} - dims
    if missing:
        issues.append(f"missing fields: {', '.join(sorted(missing))}")
    if not {"scan_angle", "scan_angle_rank"} & dims:
        issues.append("no scan angle")
    if crs is None:
        issues.append(crs_note or "no CRS in file: enter Fallback CRS")
        crs_text = ""
    else:
        crs_text = crs.to_string()
        if crs.is_geographic or not crs.is_projected:
            issues.append("CRS is not projected: reproject to metres first")
        elif any(abs(a.unit_conversion_factor - 1.0) > 1e-8 for a in crs.axis_info[:2]):
            issues.append("CRS horizontal units are not metres")
    total = physical_memory()
    need = process_memory(count)
    if total and need > total * 0.8:
        issues.append("too large for this computer's memory: split the file")
    if count < 3:
        issues.append("fewer than 3 points")
    return {"source": str(path.resolve()), "name": path.name, "file_bytes": path.stat().st_size,
            "point_count": count, "las_version": str(header.version), "point_format": header.point_format.id,
            "mins": header.mins.tolist(), "maxs": header.maxs.tolist(), "crs": crs_text,
            "horizontal_unit": crs.axis_info[0].unit_name if crs is not None and crs.axis_info else "",
            "fields": sorted(dims), "memory_bytes": need, "ram_bytes": total, "issues": issues}


def open_cloud(path, cfg):
    """Check a LAS/LAZ and the settings from its header alone; returns the input description for the results."""
    validate_settings(cfg)
    path = Path(path)
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError("Only LAS / LAZ files are supported")
    with laspy.open(path) as reader:
        header = reader.header
    count = header.point_count
    if count < 3:
        raise ValueError("Point cloud must contain at least 3 points")
    if cfg.max_points and count > cfg.max_points:
        raise ValueError(f"{count:,} points exceeds the Max points / file setting {cfg.max_points:,} "
                         "(set it to 0 for no limit); no silent downsampling")
    dims = set(header.point_format.dimension_names)
    required = {"intensity", "return_number", "number_of_returns"}
    if not required <= dims:
        raise ValueError(f"Required attributes missing: {required - dims}")
    notes = []
    native_crs = header.parse_crs()
    selected_crs = native_crs or (CRS.from_user_input(cfg.crs_override) if cfg.crs_override else None)
    if selected_crs is not None:
        if selected_crs.is_geographic or not selected_crs.is_projected:
            raise ValueError("Prototype requires projected metric coordinates; reproject before import")
        if any(abs(a.unit_conversion_factor - 1.0) > 1e-8 for a in selected_crs.axis_info[:2]):
            raise ValueError("CRS horizontal units are not metres; reproject before import")
    else:
        raise ValueError("CRS missing: specify the known projected CRS (e.g. EPSG:32632)")
    if not cfg.coordinates_in_meters:
        raise ValueError("Confirm that all X/Y/Z coordinates use metres, including Z")
    if not {"scan_angle", "scan_angle_rank"} & dims:
        raise ValueError("Scan angle is unavailable")
    if native_crs is None:
        notes.append("CRS supplied by operator; no reprojection performed")
    if cfg.vertical_datum == "unspecified" or not cfg.vertical_datum.strip():
        notes.append("Vertical datum unspecified; must be confirmed before acceptance")
    waveform_dims = sorted(d for d in dims if "wave" in d or d in ("byte_offset_to_waveform_data",))
    sidecar = path.with_suffix(".wdp")
    waveform_info = {
        "pointer_fields": waveform_dims,
        "external_wdp_exists": sidecar.exists(),
        "internal_waveform_flag": bool(header.global_encoding.waveform_data_packets_internal),
        "external_waveform_flag": bool(header.global_encoding.waveform_data_packets_external),
        "samples_decoded": False,
        "used_by_model": False,
    }
    notes.append("Full waveform is not decoded or used; pointer fields do not prove samples are available")
    return {"source": str(path.resolve()), "point_count": count,
            "las_version": str(header.version), "point_format": header.point_format.id,
            "crs": selected_crs.to_string(), "crs_wkt": selected_crs.to_wkt(),
            "units": "metres (operator confirmed)", "vertical_datum": cfg.vertical_datum,
            "scales": header.scales.tolist(), "offsets": header.offsets.tolist(),
            "fields": sorted(dims), "waveform": waveform_info, "warnings": notes}


def point_columns(points):
    """The per-point values the model uses, from a whole LAS or one streamed chunk of it."""
    dims = set(points.point_format.dimension_names)
    if "scan_angle" in dims:
        angle = np.asarray(points.scan_angle, dtype=np.float64) * 0.006
    else:
        angle = np.asarray(points.scan_angle_rank, dtype=np.float64)
    xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite coordinates in input")
    return {"xyz": xyz, "intensity": np.asarray(points.intensity, dtype=np.float64),
            "return_number": np.asarray(points.return_number, dtype=np.float64),
            "number_of_returns": np.asarray(points.number_of_returns, dtype=np.float64),
            "scan_angle": angle, "classification": np.asarray(points.classification, dtype=np.uint8)}


ZERO_INTENSITY = "All intensity values are zero; retained as recorded, not assumed calibrated or missing"


def read_cloud(path, cfg):
    """Whole file in memory (training and labelling); processing streams the file instead."""
    info = open_cloud(path, cfg)
    check_memory(info["point_count"])
    cloud = point_columns(laspy.read(path).points)
    if not cloud["intensity"].any():
        info["warnings"].append(ZERO_INTENSITY)
    info["point_count"] = len(cloud["xyz"])
    return {**cloud, "info": info}


def stream(path, chunk=CHUNK_POINTS):
    """Yield (index of first point, points) for successive chunks of a LAS/LAZ, in file order."""
    with laspy.open(path) as reader:
        start = 0
        for points in reader.chunk_iterator(chunk):
            yield start, points
            start += len(points)


def gather(path, rows, info=None):
    """Values of the given points (ascending indices), streaming the file so only those points are held.

    Also returns the mean of all coordinates. info, when given, receives the intensity warning.
    """
    rows = np.asarray(rows, dtype=np.int64)
    parts, total, intensity_seen = [], np.zeros(3), False
    for start, points in stream(path):
        cloud = point_columns(points)
        # Row by row, as numpy reduces axis 0, so the mean equals that of the whole array.
        total = np.add.reduce(np.vstack((total[None], cloud["xyz"])), axis=0)
        intensity_seen = intensity_seen or bool(cloud["intensity"].any())
        lo, hi = np.searchsorted(rows, [start, start + len(points)])
        parts.append({k: v[rows[lo:hi] - start] for k, v in cloud.items()})
        count = start + len(points)
    if info is not None and not intensity_seen:
        info["warnings"].append(ZERO_INTENSITY)
    values = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    return values, total / count


def bed_grid(xyz, classification):
    """Origin and shape of the riverbed grid from the bathymetric bottom points (LAS class 40); None if too few."""
    bed = classification == 40
    if bed.sum() < 10:
        return None
    low = xyz[bed, :2].min(axis=0)
    return low, np.floor((xyz[bed, :2].max(axis=0) - low) / BED_CELL).astype(np.int64) + 1


def bed_height(xyz, classification, grid=None, rows=None):
    """Height of each point above the local bathymetric bottom (LAS class 40) and whether it is known.

    Bottom points are binned into BED_CELL cells; each cell's median Z is smoothed by a BED_WINDOW median,
    so a target lying on the bed does not raise the bed estimate beneath itself.
    grid: from bed_grid (default: of these points; () = no bed). rows: points to return (default all).
    Only bottom points within BED_WINDOW // 2 cells of a row affect it, so a tile that holds those
    points gives the whole-file value.
    """
    rows = np.arange(len(xyz)) if rows is None else rows
    height, known = np.zeros(len(rows), dtype=np.float64), np.zeros(len(rows), dtype=bool)
    grid = bed_grid(xyz, classification) if grid is None else grid
    if not grid:
        return height, known
    low, shape = grid
    cell = np.floor((xyz[rows, :2] - low) / BED_CELL).astype(np.int64)
    inside = (cell >= 0).all(axis=1) & (cell < shape).all(axis=1)
    bed = classification == 40
    bed_cell = np.floor((xyz[bed, :2] - low) / BED_CELL).astype(np.int64)
    if not inside.any() or not len(bed_cell):
        return height, known
    # Grid over just these cells; cells outside it hold no bottom point, as do the (NaN) cells around it.
    first = np.minimum(bed_cell.min(axis=0), cell[inside].min(axis=0))
    size = np.maximum(bed_cell.max(axis=0), cell[inside].max(axis=0)) - first + 1
    bed_cell -= first
    order = np.lexsort((xyz[bed, 2], bed_cell[:, 1], bed_cell[:, 0]))
    flat = (bed_cell[:, 0] * size[1] + bed_cell[:, 1])[order]
    z = xyz[bed, 2][order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    counts = np.diff(np.r_[starts, len(flat)])
    surface_grid = np.full(size[0] * size[1], np.nan)
    surface_grid[flat[starts]] = z[starts + counts // 2]  # Z is sorted within each cell: middle value is the median.
    smooth = generic_filter(surface_grid.reshape(size), lambda v: np.nan if np.isnan(v).all() else np.nanmedian(v),
                            size=BED_WINDOW, mode="constant", cval=np.nan)
    surface = np.full(len(rows), np.nan)
    surface[inside] = smooth[cell[inside, 0] - first[0], cell[inside, 1] - first[1]]
    known = np.isfinite(surface)
    height[known] = xyz[rows[known], 2] - surface[known]
    return np.clip(height, -5, 5), known


def neighbours(tree, local, rows, k, number):
    """Distances and indices of each row's k nearest points, ordered by distance, then by point number.

    Equal distances are common (coordinates are stored in millimetres), and a k-d tree may return either of
    two equally near points. Choosing by point number makes the result independent of the tree, so a tile
    gives the same neighbours as the whole file. number: original point number of every point in the tree.
    """
    more = k < len(local)
    distances, idx = tree.query(local[rows], k=k + more, workers=1)
    if more:  # A tie at the k-th distance: take the lowest numbers among all points at that distance.
        for row in np.flatnonzero(distances[:, k] == distances[:, k - 1]):
            limit, wanted = distances[row, k - 1], 2 * k
            while True:
                near, ids = tree.query(local[rows[row]], k=min(wanted, len(local)))
                if near[-1] > limit or wanted >= len(local):
                    break
                wanted *= 2
            ids = ids[near <= limit]
            near = near[near <= limit]
            choose = np.lexsort((number[ids], near))[:k]
            distances[row, :k], idx[row, :k] = near[choose], ids[choose]
        distances, idx = distances[:, :k], idx[:, :k]
    tied = np.flatnonzero((distances[:, 1:] == distances[:, :-1]).any(axis=1))  # Order equal ones by number too.
    if len(tied):
        order = np.lexsort((number[idx[tied]], distances[tied]), axis=1)
        distances[tied] = np.take_along_axis(distances[tied], order, axis=1)
        idx[tied] = np.take_along_axis(idx[tied], order, axis=1)
    return distances, idx


def extract_features(cloud, neighbors=20, log=None, rows=None, origin=None, bed=None, k=None):
    """Full resolution; chunks bound the N*k temporaries and run on all CPU cores.

    rows: points to describe (default all); every point of cloud can be their neighbour. origin: of the
    local frame (default the first point); bed: see bed_height; k: neighbours (default neighbors, at most
    the point count). A tile given the same origin, grid and k gives the whole-file values for its rows.
    """
    xyz = cloud["xyz"]
    rows = np.arange(len(xyz)) if rows is None else np.asarray(rows)
    classification = cloud.get("classification", np.zeros(len(xyz), dtype=np.uint8))
    above_bed, bed_known = bed_height(xyz, classification, bed, rows)
    local = xyz - (xyz[0] if origin is None else origin)
    tree = cKDTree(local)
    k = min(neighbors, len(xyz)) if k is None else k
    number = cloud.get("index", np.arange(len(xyz)))
    features = np.empty((len(rows), len(FEATURES)), dtype=np.float32)
    rn, nr = cloud["return_number"], cloud["number_of_returns"]

    def chunk(start):
        stop = min(start + 8192, len(rows))
        own = rows[start:stop]
        distances, idx = neighbours(tree, local, own, k, number)
        points = local[idx]
        centered = points - points.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / k
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
        largest = np.maximum(eigen[:, 2], 1e-12)
        heights = local[own, 2] - np.quantile(points[:, :, 2], 0.1, axis=1)
        values = cloud["intensity"][idx]
        relative = (cloud["intensity"][own] - np.median(values, axis=1)) / (values.std(axis=1) + 1)
        r, n = rn[own], nr[own]
        invalid = (n < 1) | (r < 1) | (r > n)
        features[start:stop] = np.column_stack((
            heights, np.sqrt(eigen[:, 0]), (eigen[:, 2] - eigen[:, 1]) / largest,
            (eigen[:, 1] - eigen[:, 0]) / largest, eigen[:, 0] / largest,
            distances[:, -1], relative, cloud["intensity"][own] / 65535,
            np.where(invalid, 0, (r - 1) / np.maximum(n - 1, 1)),
            (~invalid & (n > 1)), (~invalid & (r == n)),
            cloud["scan_angle"][own] / 90, invalid,
            above_bed[start:stop], bed_known[start:stop], *class_flags(classification[own]),
        ))
    starts = range(0, len(rows), 8192)
    # Threads share the arrays; numpy/scipy release the GIL, and slices are disjoint.
    with ThreadPoolExecutor(WORKERS) as pool:
        for done, _ in enumerate(pool.map(chunk, starts), 1):
            if log and len(starts) > 100 and done % (len(starts) // 4) == 0:
                log(f"  features {100 * done // len(starts)}%")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    return features


def class_flags(codes):
    """Vendor LAS classes the model can use: bottom, water surface, water column, ground, noise."""
    return (codes == 40, np.isin(codes, (9, 41, 42)), codes == 45, codes == 2, np.isin(codes, (7, 18)))


TILE_MARGIN = 5.0  # metres read around a tile's points at first; more than the riverbed window (3 m).


def tile_layout(counts):
    """Split a grid of point counts into rectangles of at most TILE_POINTS points (or single cells).

    Each split is across the longer side, where the points divide in half, so tiles follow the data,
    not the (often mostly empty) bounding box. Returns the tile number of every cell, -1 for empty ones.
    """
    label = np.full(counts.shape, -1, dtype=np.int64)
    pending, number = [(0, counts.shape[0], 0, counts.shape[1])], 0
    while pending:
        x0, x1, y0, y1 = pending.pop()
        block = counts[x0:x1, y0:y1]
        total = block.sum()
        if not total:
            continue
        if total <= TILE_POINTS or block.size == 1:
            label[x0:x1, y0:y1] = number
            number += 1
            continue
        axis = 0 if x1 - x0 >= y1 - y0 else 1
        cut = int(np.searchsorted(np.cumsum(block.sum(axis=1 - axis)), total / 2)) + 1
        cut = min(max(cut, 1), block.shape[axis] - 1)
        if axis == 0:
            pending += [(x0, x0 + cut, y0, y1), (x0 + cut, x1, y0, y1)]
        else:
            pending += [(x0, x1, y0, y0 + cut), (x0, x1, y0 + cut, y1)]
    return label


def make_tiles(path, folder):
    """Stream a LAS/LAZ into spatial tiles of at most about TILE_POINTS points, stored in folder.

    A first pass counts points on a fine grid to lay out the tiles; a second writes each point to its tile.
    Returns the tiles (file, point count, XY bounds), the first point (origin of the local frame) and the
    riverbed grid of the whole file, so tiles are described exactly as the whole file would be.
    """
    folder = Path(folder)
    folder.mkdir()
    with laspy.open(path) as reader:
        count, low, high = reader.header.point_count, reader.header.mins[:2], reader.header.maxs[:2]
    extent = np.maximum(high - low, 1e-9)
    # About 16 grid cells per tile if points were spread evenly over the bounding box.
    side = max(np.sqrt(extent.prod() * TILE_POINTS / count) / 4, TILE_MARGIN)
    shape = np.ceil(extent / side).astype(np.int64) + 1

    def cells(x, y):  # Header bounds only place the grid; points outside them go to the edge cells.
        cell = np.floor((np.column_stack((x, y)) - low) / side).astype(np.int64)
        cell = np.clip(cell, 0, shape - 1)
        return cell[:, 0] * shape[1] + cell[:, 1]
    if count <= TILE_POINTS:
        label = np.zeros(shape.prod(), dtype=np.int64)
    else:
        counts = np.zeros(shape.prod(), dtype=np.int64)
        for _, points in stream(path):
            counts += np.bincount(cells(points.x, points.y), minlength=len(counts))
        label = tile_layout(counts.reshape(shape)).ravel()
    tiles = {}
    origin, bed_count, bed_low, bed_high = None, 0, np.full(2, np.inf), np.full(2, -np.inf)
    for start, points in stream(path):
        cloud = point_columns(points)
        xyz = cloud["xyz"]
        origin = xyz[0] if origin is None else origin
        bed = cloud["classification"] == 40
        if bed.any():
            bed_count += int(bed.sum())
            bed_low = np.minimum(bed_low, xyz[bed, :2].min(axis=0))
            bed_high = np.maximum(bed_high, xyz[bed, :2].max(axis=0))
        records = np.empty(len(xyz), dtype=TILE_RECORD)
        records["index"] = np.arange(start, start + len(xyz))
        records["x"], records["y"], records["z"] = xyz.T
        for name in ("intensity", "return_number", "number_of_returns", "scan_angle", "classification"):
            records[name] = cloud[name]
        tile = label[cells(xyz[:, 0], xyz[:, 1])]
        order = np.argsort(tile, kind="stable")
        ids, first = np.unique(tile[order], return_index=True)
        for number, part in zip(ids, np.split(records[order], first[1:])):
            entry = tiles.setdefault(int(number), {"path": folder / f"{number}.bin", "count": 0,
                                                   "low": np.full(2, np.inf), "high": np.full(2, -np.inf)})
            with entry["path"].open("ab") as stream_out:
                part.tofile(stream_out)
            entry["count"] += len(part)
            entry["low"] = np.minimum(entry["low"], (part["x"].min(), part["y"].min()))
            entry["high"] = np.maximum(entry["high"], (part["x"].max(), part["y"].max()))
    grid = (bed_low, np.floor((bed_high - bed_low) / BED_CELL).astype(np.int64) + 1) if bed_count >= 10 else ()
    return [tiles[number] for number in sorted(tiles)], origin, grid


def tile_cloud(records):
    return {"xyz": np.column_stack((records["x"], records["y"], records["z"])), "index": records["index"],
            "intensity": records["intensity"].astype(np.float64),
            "return_number": records["return_number"].astype(np.float64),
            "number_of_returns": records["number_of_returns"].astype(np.float64),
            "scan_angle": records["scan_angle"], "classification": records["classification"]}


def tile_features(tiles, number, neighbors, origin, grid, log=None):
    """Features of one tile's points, equal to those computed over the whole file.

    Points of all tiles within a margin around the tile are read as neighbours. A point whose k-th
    neighbour is farther than the margin might have a nearer one outside it, so it is done again with a
    four times wider margin, until every point is certain (or the whole file has been read).
    """
    total = sum(tile["count"] for tile in tiles)
    k = min(neighbors, total)
    own = np.fromfile(tiles[number]["path"], dtype=TILE_RECORD)
    features = np.empty((len(own), len(FEATURES)), dtype=np.float32)
    todo, reach = np.arange(len(own)), TILE_MARGIN
    while len(todo):
        low = np.array((own["x"][todo].min(), own["y"][todo].min())) - reach
        high = np.array((own["x"][todo].max(), own["y"][todo].max())) + reach
        parts = [own[todo]]
        for index, tile in enumerate(tiles):
            if (tile["high"] < low).any() or (tile["low"] > high).any():
                continue
            records = own if index == number else np.fromfile(tile["path"], dtype=TILE_RECORD)
            near = ((records["x"] >= low[0]) & (records["x"] <= high[0]) &
                    (records["y"] >= low[1]) & (records["y"] <= high[1]))
            if index == number:
                near[todo] = False
            parts.append(records[near])
        context = np.concatenate(parts)
        everything = len(context) == total
        if len(context) >= k:
            values = extract_features(tile_cloud(context), neighbors, rows=np.arange(len(todo)), origin=origin,
                                      bed=grid, k=k)
            # Every point outside the read area is more than reach away in XY alone.
            certain = np.full(len(todo), True) if everything else values[:, FEATURES.index("radius")] < reach * 0.999
            features[todo[certain]] = values[certain]
            todo = todo[~certain]
        if len(todo) and log:
            log(f"  {len(todo):,} sparse points: reading {reach * 4:g} m around them")
        reach *= 4
    return own["index"], features


def classify_tiles(path, count, estimator, neighbors, folder, log=None, stage=None):
    """Winning class and its probability for every point, computed tile by tile from a temporary tile store."""
    classes = np.zeros(count, dtype=np.uint8)
    confidence = np.zeros(count, dtype=np.float32)
    try:
        tiles, origin, grid = make_tiles(path, folder)
        if log:
            log(f"  {len(tiles)} tile(s)")

        def one(number):
            index, features = tile_features(tiles, number, neighbors, origin, grid, log)
            proba = predict(estimator, features)
            winning = proba.argmax(axis=1)
            return index, np.asarray(estimator.classes_)[winning], proba[np.arange(len(features)), winning]
        # Two tiles at a time, so one tile's serial steps (reading, k-d tree) overlap the other's parallel ones.
        with ThreadPoolExecutor(2) as pool:
            for number, (index, winners, scores) in enumerate(pool.map(one, range(len(tiles))), 1):
                classes[index], confidence[index] = winners, scores
                if stage:
                    stage("classifying", number / len(tiles))
                if log and len(tiles) > 1:
                    log(f"  tile {number} / {len(tiles)} done")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return classes, confidence


def label_template(path, output, cfg):
    cloud = read_cloud(path, cfg)
    with Path(output).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "class_id"])
        writer.writerows((i, "") for i in range(len(cloud["xyz"])))
    write_label_source(output, path, len(cloud["xyz"]))
    return cloud["info"]


def write_label_source(labels, source, count):
    """Bind index-based CSV labels to the exact source file, without storing a private path."""
    write_json(str(labels) + ".source.json", {"source_sha256": sha256(source), "point_count": count})


def load_labels(path, count, targets=None, *, source_xyz=None, label_field="classification",
                source_sha256=None, labels_source_sha256=None):
    """point_index,class_id CSV, or a LAS/LAZ whose classification field marks the targets.

    Labels are ASPRS LAS class codes. A point whose code is one of the manifest's target classes is that
    target; any other code (0, 2, 40, ...) means "not a target" (internal label NOT_TARGET).
    In a CSV, a blank class_id or -1 means unlabelled.
    """
    targets = list(targets or ())
    if Path(path).suffix.lower() in (".las", ".laz"):
        las = laspy.read(path)
        if len(las.points) != count:
            raise ValueError("Label LAS/LAZ has a different point count from its point cloud")
        if source_xyz is None:
            raise ValueError("LAS labels require original point coordinates to verify point order")
        if np.shape(source_xyz) != (count, 3):
            raise ValueError("Original point coordinates have an invalid shape")
        for axis, values in enumerate((las.x, las.y, las.z)):
            if not np.allclose(values, source_xyz[:, axis], rtol=0, atol=1e-7):
                raise ValueError("LAS label coordinates / point order differ from the source; restore original order")
        if label_field not in ("classification", "object_class", "user_data"):
            raise ValueError("Unsupported LAS label_field")
        if label_field not in set(las.point_format.dimension_names):
            raise ValueError(f"LAS label field is missing: {label_field}")
        raw_codes = np.asarray(las[label_field])
        if raw_codes.ndim != 1 or raw_codes.dtype.kind not in "ui" or np.any(raw_codes > 255) or np.any(raw_codes < 0):
            raise ValueError("LAS label field must contain integer class codes 0..255")
        codes = raw_codes.astype(np.int32)
        return np.where(np.isin(codes, targets), codes, NOT_TARGET)
    if label_field != "classification":
        raise ValueError("label_field applies only to LAS/LAZ labels")
    if source_sha256 is not None and labels_source_sha256 != source_sha256:
        raise ValueError("CSV labels need a matching labels_source_sha256 or .source.json binding; verify their source first")
    labels = np.full(count, -1, dtype=np.int32)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"point_index", "class_id"} <= set(reader.fieldnames or []):
            raise ValueError("Labels must have point_index,class_id columns")
        for row in reader:
            idx = int(row["point_index"])
            if not 0 <= idx < count or idx in seen:
                raise ValueError("Duplicate / out-of-range point_index in labels")
            seen.add(idx)
            value = row["class_id"].strip()
            if value:
                label = int(value)
                if not -1 <= label <= 255:
                    raise ValueError("class_id must be an ASPRS LAS class code 0..255, or -1 / blank = unlabelled")
                labels[idx] = label
    return np.where((labels >= 0) & ~np.isin(labels, targets), NOT_TARGET, labels)


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    classes = manifest.get("classes", {})
    if "0" in classes:
        raise ValueError('List only target classes: remove "0". Points that are not targets keep their own '
                         "LAS class; there is no separate background class")
    if not classes:
        raise ValueError("classes must list at least one target class (ASPRS LAS code)")
    for code in classes:
        check_target_code(int(code))
    classes = {str(NOT_TARGET): "not a target", **{str(int(k)): v for k, v in classes.items()}}
    records, groups, files, hashes = [], {}, set(), set()
    for entry in manifest["datasets"]:
        entry = dict(entry)
        if entry["split"] not in ("train", "validation", "test") or not entry.get("scene_id"):
            raise ValueError("Every dataset needs split and scene_id")
        previous = groups.setdefault(entry["scene_id"], entry["split"])
        if previous != entry["split"]:
            raise ValueError("A scene_id occurs in multiple splits; spatial leakage")
        for key in ("path", "labels"):
            entry[key] = str((path.parent / entry[key]).resolve())
        resolved = entry["path"]
        digest = sha256(resolved)
        if resolved in files or digest in hashes:
            raise ValueError("Repeated point cloud / identical file content in manifest")
        files.add(resolved)
        hashes.add(digest)
        entry["sha256"] = digest
        entry["labels_sha256"] = sha256(entry["labels"])
        entry.setdefault("label_field", "classification")
        if Path(entry["labels"]).suffix.lower() not in (".las", ".laz"):
            sidecar = Path(entry["labels"] + ".source.json")
            binding = entry.get("labels_source_sha256")
            if sidecar.exists():
                provenance = json.loads(sidecar.read_text(encoding="utf-8"))
                with laspy.open(resolved) as reader:
                    if provenance.get("point_count") != reader.header.point_count:
                        raise ValueError("CSV label source point count differs")
                if binding is not None and binding != provenance.get("source_sha256"):
                    raise ValueError("CSV label source bindings disagree")
                binding = provenance.get("source_sha256")
            if binding != digest:
                raise ValueError("CSV labels need a matching labels_source_sha256 or .source.json binding; verify their source first")
            entry["labels_source_sha256"] = binding
        records.append(entry)
    if set(e["split"] for e in records) != {"train", "validation", "test"}:
        raise ValueError("Provide separate train, validation and test datasets")
    balance = manifest.get("balance_classes", True)
    if not isinstance(balance, bool):
        raise ValueError("balance_classes must be true or false")
    return {"classes": classes, "balance_classes": balance, "datasets": records}


def metrics(y, pred, classes):
    ids = sorted(int(k) for k in classes)
    return {"point_metrics": classification_report(y, pred, labels=ids, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=ids).tolist(),
            "class_order": ids,
            "note": "Point metrics only; object detection AP and instance IoU not evaluated"}


def collect(records, split, cfg, classes, log):
    xs, ys = [], []
    count = 0
    for record in records:
        if record["split"] != split:
            continue
        log(f"{split}: {Path(record['path']).name}")
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Point cloud changed since split registration")
        if sha256(record["labels"]) != record.get("labels_sha256"):
            raise ValueError("Labels changed or lack a registered fingerprint; register verified labels and retrain")
        cloud = read_cloud(record["path"], cfg)
        labels = load_labels(record["labels"], len(cloud["xyz"]), {int(k) for k in classes if int(k)},
                             source_xyz=cloud["xyz"], label_field=record.get("label_field", "classification"),
                             source_sha256=record["sha256"], labels_source_sha256=record.get("labels_source_sha256"))
        selected = labels >= 0
        if not selected.any():
            raise ValueError(f"No labels: {record['labels']}")
        count += int(selected.sum())
        if cfg.max_points and count > cfg.max_points:
            raise ValueError("Total labelled points in split exceeds prototype max_points limit")
        xs.append(extract_features(cloud, cfg.neighbors)[selected])
        ys.append(labels[selected])
    return np.concatenate(xs), np.concatenate(ys)


def train_model(manifest_path, output_path, cfg, backend="random_forest", iterations=100, log=print):
    validate_settings(cfg)
    if Path(output_path).exists():
        raise FileExistsError("Model output already exists; choose a new filename")
    if not 1 <= iterations <= 10000:
        raise ValueError("Training iterations / trees must be 1..10000")
    manifest = load_manifest(manifest_path)
    x, y = collect(manifest["datasets"], "train", cfg, manifest["classes"], log)
    if set(np.unique(y)) != {int(k) for k in manifest["classes"]}:
        raise ValueError("Training labels must contain every target class and some non-target points")
    if backend == "random_forest":
        estimator = RandomForestClassifier(n_estimators=iterations,
                                           class_weight="balanced" if manifest["balance_classes"] else None,
                                           min_samples_leaf=2, max_depth=24, random_state=42, n_jobs=WORKERS)
    elif backend == "mlp":
        estimator = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64, 32), max_iter=iterations, random_state=42,
            early_stopping=False, batch_size=min(256, len(y))))
    else:
        raise ValueError("Unknown model backend")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y)
    training_warnings = [str(w.message) for w in caught]
    for message in training_warnings:
        log(message)
    xv, yv = collect(manifest["datasets"], "validation", cfg, manifest["classes"], log)
    validation = metrics(yv, predict_classes(estimator, xv), manifest["classes"])
    # Preserve provenance and add relocatable references for a model/data folder.
    for record in manifest["datasets"]:
        for key in ("path", "labels"):
            try:
                record[key + "_relative_to_model"] = os.path.relpath(record[key], Path(output_path).resolve().parent)
            except ValueError:  # Different Windows drive: preserve absolute reference.
                pass
    payload = {"format": "MOST_ML_V1", "software_version": VERSION,
               "model_id": str(uuid.uuid4()), "created_at": datetime.now().astimezone().isoformat(),
               "backend": backend, "estimator": estimator, "classes": manifest["classes"],
               "balance_classes": manifest["balance_classes"],
               "features": FEATURES, "neighbors": cfg.neighbors, "training_settings": asdict(cfg),
               "datasets": manifest["datasets"], "validation": validation,
               "training_warnings": training_warnings, "sklearn_version": sklearn.__version__}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    write_json(str(output_path) + ".json", {k: v for k, v in payload.items() if k != "estimator"})
    return str(output_path)


def load_model(path):
    # joblib models are executable pickle objects. Load only locally trusted models.
    model = joblib.load(path)
    if model.get("format") != "MOST_ML_V1" or model.get("features") != FEATURES:
        raise ValueError("Model format or feature schema is incompatible")
    if model.get("sklearn_version") != sklearn.__version__:
        raise ValueError("scikit-learn version differs from model; use its recorded environment")
    return model


def evaluate_model(model_path, output_path, cfg, log=print):
    if Path(output_path).exists():
        raise FileExistsError("Evaluation output already exists")
    model = load_model(model_path)
    cfg = Settings(**{**asdict(cfg), "neighbors": model["neighbors"]})
    for record in model["datasets"]:
        for key in ("path", "labels"):
            if key + "_relative_to_model" in record:
                record[key] = str((Path(model_path).resolve().parent / record[key + "_relative_to_model"]).resolve())
        if record["split"] == "test" and sha256(record["path"]) != record["sha256"]:
            raise ValueError("Test file changed since split registration")
    x, y = collect(model["datasets"], "test", cfg, model["classes"], log)
    result = metrics(y, predict_classes(model["estimator"], x), model["classes"])
    result["model_id"] = model["model_id"]
    write_json(output_path, result)
    return str(output_path)


def make_run_directory(first_path, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(first_path).stem}_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    for suffix in range(10000):
        candidate = root / (stem if suffix == 0 else f"{stem}_{suffix:03d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("Too many duplicate run directories")


def predict(estimator, features, batch=65536):
    """Class probabilities, batches of rows in parallel on all cores.

    Each batch runs on one thread, so a forest adds its trees' votes in a fixed order. With threads inside a
    batch the order varies, and a point whose two best classes are almost equal could change class between runs.
    """
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    starts = range(0, len(features), batch)
    with ThreadPoolExecutor(WORKERS) as pool:
        parts = list(pool.map(lambda a: estimator.predict_proba(features[a:a + batch]), starts))
    return np.concatenate(parts) if parts else np.empty((0, len(estimator.classes_)))


def predict_classes(estimator, features):
    return np.asarray(estimator.classes_)[predict(estimator, features).argmax(axis=1)]


def dbscan(points, eps, min_points, chunk=32768):
    """DBSCAN with memory bounded by chunk size instead of the whole neighbour graph.

    Core points (>= min_points within eps, self included) and their clusters match
    scikit-learn exactly. A border point joins the cluster of its nearest core point.
    """
    n = len(points)
    tree = cKDTree(points)
    counts = np.concatenate([tree.query_ball_point(points[a:a + chunk], eps, return_length=True, workers=WORKERS)
                             for a in range(0, n, chunk)])
    core = np.flatnonzero(counts >= min_points)
    labels = np.full(n, -1, dtype=np.int64)
    if not len(core):
        return labels
    core_xyz = points[core]
    core_tree = cKDTree(core_xyz)
    root = np.arange(len(core))  # Representative (smallest member) of each core point's cluster.
    for a in range(0, len(core), chunk):
        pairs = cKDTree(core_xyz[a:a + chunk]).sparse_distance_matrix(core_tree, eps, output_type="ndarray")
        i, j = root[pairs["i"] + a], root[pairs["j"]]
        keep = i != j
        if not keep.any():
            continue
        nodes, inverse = np.unique(np.concatenate((i[keep], j[keep])), return_inverse=True)
        m = int(keep.sum())
        graph = coo_matrix((np.ones(m, dtype=np.int8), (inverse[:m], inverse[m:])), shape=(len(nodes), len(nodes)))
        _, component = connected_components(graph, directed=False)
        smallest = np.full(component.max() + 1, len(core))
        np.minimum.at(smallest, component, nodes)
        mapping = np.arange(len(core))
        mapping[nodes] = smallest[component]
        root = mapping[root]
    labels[core] = np.unique(root, return_inverse=True)[1]
    border = np.flatnonzero(counts < min_points)
    for a in range(0, len(border), chunk):
        ids = border[a:a + chunk]
        dist, nearest = core_tree.query(points[ids], k=1, distance_upper_bound=np.nextafter(eps, np.inf),
                                        workers=WORKERS)
        found = np.isfinite(dist)
        labels[ids[found]] = labels[core[nearest[found]]]
    return labels


def instance_groups(xyz, classes, probabilities, cfg):
    """Global per-file clustering: no processing tiles, hence no tile duplicates."""
    groups = []
    for cls in sorted(np.unique(classes)):
        if cls == 0:
            continue
        idx = np.flatnonzero((classes == cls) & (probabilities >= cfg.threshold))
        if len(idx) < cfg.min_points:
            continue
        labels = dbscan(xyz[idx] - xyz[idx[0]], cfg.cluster_eps, cfg.min_points)
        # One sort instead of a full scan per cluster; stable, so point order is kept.
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
        for label in range(labels.max() + 1):
            object_idx = idx[order[bounds[label]:bounds[label + 1]]]
            if len(object_idx) >= cfg.min_points:
                groups.append((int(cls), object_idx, float(probabilities[object_idx].mean())))
    return groups


def projected_area(xyz, method):
    if method == "unresolved":
        return None
    xy = np.unique(xyz[:, :2] - xyz[0, :2], axis=0)
    if len(xy) < 3 or np.linalg.matrix_rank(xy - xy[0]) < 2:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)  # scipy's 2D volume is polygon area
    except QhullError:
        return None


def display_indices(instance, limit=DISPLAY_LIMIT, maximum=DISPLAY_LIMIT):
    """Bound display size, retaining a representative of each instance if possible."""
    if not 1 <= limit <= maximum:
        raise ValueError(f"Display limit must be 1..{maximum}")
    if len(instance) <= limit:
        return np.arange(len(instance))
    labels, first = np.unique(instance, return_index=True)
    representatives = first[labels > 0]
    rng = np.random.default_rng(42)
    if len(representatives) > limit:
        return np.sort(rng.choice(representatives, limit, replace=False))
    available = np.ones(len(instance), dtype=bool)
    available[representatives] = False
    extra = rng.choice(np.flatnonzero(available), limit - len(representatives), replace=False)
    return np.sort(np.concatenate((representatives, extra)))


def web_indices(instance, limit=WEB_LIMIT, keep=None):
    """Browser sample: object points get at least half the budget, so objects stay dense.

    keep: optional mask of further points to treat like object points (e.g. ground-truth targets).
    """
    if len(instance) <= limit:
        return np.arange(len(instance))
    important = instance > 0 if keep is None else (instance > 0) | keep
    if keep is not None:  # Sample over the combined set; display_indices only needs group labels.
        instance = np.where(instance > 0, instance, important.astype(instance.dtype) * (instance.max() + 1))
    objects, background = np.flatnonzero(important), np.flatnonzero(~important)
    take_objects = min(len(objects), max(limit // 2, limit - len(background)))
    take_background = min(len(background), limit - take_objects)
    chosen = objects[display_indices(instance[objects], take_objects, len(objects))] if take_objects else objects[:0]
    rng = np.random.default_rng(42)
    return np.sort(np.concatenate((chosen, rng.choice(background, take_background, replace=False))))


def write_web_view(dest, name, xyz, instance, objects, origin, total, predicted=None, class_names=None,
                   intensity=None, las_classes=None, las_names=LAS_CLASSES, truth=None, truth_objects=None):
    """Self-contained offline HTML page drawing the points with WebGL (GPU).

    predicted: class ID per point, class_names: {class ID: name}; intensity: raw LAS intensity per point;
    las_classes: input LAS classification code per point; truth: per-point ground-truth status
    (TRUTH_STATUS) and truth_objects: ground-truth targets, when the input has a ground-truth file.
    """
    def encode(array):
        return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")
    data = {"name": name, "total": int(total), "origin_z": float(origin[2]),
            "xyz": encode((xyz - origin).astype("<f4")), "instance": encode(instance.astype("<i4")),
            "objects": [[o["instance_index"], o["object_id"], o["class_name"], o["point_count"], o["confidence"],
                         *(np.asarray(o["centroid"]) - origin).tolist()] for o in objects]}
    if predicted is not None and class_names:
        ids = sorted(int(k) for k in class_names)
        data["classes"] = [[i, class_names[str(i)]] for i in ids]
        data["cls"] = encode(np.searchsorted(ids, predicted).astype(np.uint8))
    if intensity is not None and len(intensity):
        low, high = np.percentile(intensity, [2, 98])
        scaled = np.clip((intensity - low) / max(high - low, 1e-9) * 255, 0, 255)
        data["intensity"] = encode(scaled.astype(np.uint8))
    if las_classes is not None:
        codes = sorted(int(c) for c in np.unique(las_classes))
        data["las"] = encode(np.asarray(las_classes, dtype=np.uint8))
        data["las_names"] = [[c, las_names.get(c, "user-defined" if c >= USER_CLASS_MIN else "reserved")]
                             for c in codes]
    if truth is not None:
        data["truth"] = encode(np.asarray(truth, dtype=np.uint8))
        data["truth_objects"] = [[t["object_id"], t["class_name"], t["status"], *(np.asarray(t["center"]) - origin).tolist()]
                                 for t in truth_objects]
    # "</" is escaped so no value can close the <script> element that holds the data.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = WEB_TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", payload, 1)
    Path(dest).write_text(page, encoding="utf-8")


def refresh_web_view(page):
    """Move a saved 3D page onto the current viewer template, keeping its embedded data."""
    page = Path(page)
    text = page.read_text(encoding="utf-8")
    marker = '<script id="data" type="application/json">'
    start = text.find(marker) + len(marker)
    end = text.find("</script>", start)
    if start < len(marker) or end < 0:
        return
    head, tail = WEB_TEMPLATE.read_text(encoding="utf-8").split("/*DATA*/", 1)
    if text[:start] == head and text[end:] == tail:
        return
    page.write_text(head + text[start:end] + tail, encoding="utf-8")


TRUTH_STATUS = {0: "not a target", 1: "detected target", 2: "false alarm", 3: "missed target"}


def truth_path(source):
    """Ground truth for an input lives next to it: scene.laz -> scene.truth.json."""
    return Path(source).with_suffix(".truth.json")


def load_truth(path, count, *, source_sha256=None):
    truth = json.loads(Path(path).read_text(encoding="utf-8"))
    if truth.get("point_count") != count:
        raise ValueError("Ground truth was made for a different point cloud (point count differs)")
    if source_sha256 is not None and truth.get("source_sha256") != source_sha256:
        raise ValueError("Ground truth source_sha256 is missing or differs; verify the source and point order")
    if not isinstance(truth.get("objects"), list):
        raise ValueError("Ground truth objects must be a list")
    instance = np.zeros(count, dtype=np.int32)
    seen = set()
    for number, target in enumerate(truth["objects"], 1):
        if not isinstance(target, dict):
            raise ValueError("Ground truth object must be a record")
        oid = target.get("object_id")
        if not isinstance(oid, str) or not oid.strip() or oid in seen:
            raise ValueError("Ground truth object IDs must be nonempty and unique")
        seen.add(oid)
        if type(target.get("class_id")) is not int:
            raise ValueError("Ground truth class_id must be an integer")
        check_target_code(target["class_id"])
        if not isinstance(target.get("class_name"), str) or not target["class_name"].strip():
            raise ValueError("Ground truth class_name must be nonempty")
        center = target.get("center")
        if not isinstance(center, list) or len(center) != 3 or any(
                type(v) not in (int, float) or not np.isfinite(v) for v in center):
            raise ValueError("Ground truth center must contain three finite coordinates")
        if ("point_range" in target) == ("point_indices" in target):
            raise ValueError("Ground truth needs exactly one of point_range or point_indices")
        if "point_range" in target:
            bounds = target["point_range"]
            if (not isinstance(bounds, list) or len(bounds) != 2 or
                    any(type(v) is not int for v in bounds) or not 0 <= bounds[0] <= bounds[1] <= count):
                raise ValueError("Ground truth point_range is invalid or out of range")
            selected = slice(*bounds)
        else:
            indices = target["point_indices"]
            if (not isinstance(indices, list) or any(type(v) is not int or not 0 <= v < count for v in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("Ground truth point_indices contain duplicate, invalid or out-of-range indices")
            selected = np.asarray(indices, dtype=np.int64)
        if np.any(instance[selected]):
            raise ValueError("Ground truth objects have overlapping point indices")
        instance[selected] = number
    return truth["objects"], instance


def evaluate_objects(point_instance, objects, targets, truth_instance, min_iou=0.25):
    """Match detected objects to ground-truth targets one-to-one by point IoU (greedy, highest first).

    A match needs IoU >= min_iou; it is class-correct when the detected class equals the target class.
    Objects and targets are annotated in place; returns the summary and per-point TRUTH_STATUS.
    """
    detected = np.bincount(point_instance, minlength=len(objects) + 1)
    true_size = np.bincount(truth_instance, minlength=len(targets) + 1)
    both = (point_instance > 0) & (truth_instance > 0)
    pairs, overlap = np.unique(np.stack((point_instance[both], truth_instance[both])), axis=1, return_counts=True)
    iou = overlap / (detected[pairs[0]] + true_size[pairs[1]] - overlap)
    match_target, match_object = {}, {}
    for k in np.argsort(-iou, kind="stable"):
        d, t = int(pairs[0, k]), int(pairs[1, k])
        if iou[k] >= min_iou and d not in match_object and t not in match_target:
            match_object[d], match_target[t] = (t, float(iou[k])), (d, float(iou[k]))
    for obj in objects:
        found = match_object.get(obj["instance_index"])
        if found:
            target = targets[found[0] - 1]
            obj["truth"] = {"target_id": target["object_id"], "iou": found[1],
                            "class_correct": obj["class_id"] == target["class_id"],
                            "centre_error_m": float(np.linalg.norm(np.subtract(obj["centroid"], target["center"])))}
        else:
            obj["truth"] = {"target_id": None, "iou": 0.0, "class_correct": False, "centre_error_m": None}
    for number, target in enumerate(targets, 1):
        found = match_target.get(number)
        target["visible_points"] = int(true_size[number])
        target["status"] = "detected" if found else "missed"
        target["detected_as"] = objects[found[0] - 1]["object_id"] if found else None
        target["iou"] = found[1] if found else 0.0
    status = np.zeros(len(point_instance), dtype=np.uint8)
    status[truth_instance > 0] = 3
    matched = np.zeros(len(objects) + 1, dtype=bool)
    matched[list(match_object)] = True
    status[point_instance > 0] = np.where(matched[point_instance[point_instance > 0]], 1, 2)
    tp, fp, fn = len(match_object), len(objects) - len(match_object), len(targets) - len(match_object)
    matches = [o["truth"] for o in objects if o["truth"]["target_id"]]
    classes = sorted({t["class_name"] for t in targets})
    summary = {
        "targets": len(targets), "detections": len(objects), "true_positives": tp,
        "false_positives": fp, "missed": fn,
        "precision": tp / len(objects) if objects else None, "recall": tp / len(targets) if targets else None,
        "f1": 2 * tp / (len(objects) + len(targets)) if objects or targets else None,
        "class_correct": sum(m["class_correct"] for m in matches),
        "mean_iou": float(np.mean([m["iou"] for m in matches])) if matches else None,
        "mean_centre_error_m": float(np.mean([m["centre_error_m"] for m in matches])) if matches else None,
        "recall_by_class": {c: sum(t["status"] == "detected" for t in targets if t["class_name"] == c)
                            / sum(t["class_name"] == c for t in targets) for c in classes},
        "recall_by_visible_points": {
            label: f"{sum(t['status'] == 'detected' for t in group)}/{len(group)}"
            for label, group in (("<8", [t for t in targets if t["visible_points"] < 8]),
                                 ("8-19", [t for t in targets if 8 <= t["visible_points"] < 20]),
                                 (">=20", [t for t in targets if t["visible_points"] >= 20])) if group},
        "missed_targets": [t["object_id"] for t in targets if t["status"] == "missed"],
        "matching": f"one-to-one, point IoU >= {min_iou}",
    }
    return summary, status


POINT_FORMAT_WITH_FULL_CLASS = {0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 10}


def write_classified(source, dest, groups, point_instance, las_output="same as input"):
    """Preserve source attributes except target classifications and reserved prediction fields.

    Points not in a detected object keep the input (vendor) class. An extra dimension object_instance
    holds the object number (0 = none). LAS 1.2/1.3 point formats (0-5) store only classes 0-31:
    object_class holds current predicted target codes in every version (0 = no detected target).
    User Data is never changed. Legacy class fields retain their input value for codes above 31.
    Returns where the target codes were written.
    """
    if Path(source).resolve() == Path(dest).resolve() or Path(dest).exists():
        raise FileExistsError("Classified output must be a new file, separate from its source")
    if las_output not in LAS_OUTPUTS:
        raise ValueError("Unsupported classified LAS version")
    with laspy.open(source) as reader:
        header = copy.deepcopy(reader.header)
        count = header.point_count
        point_instance = np.asarray(point_instance)
        if (point_instance.shape != (count,) or point_instance.dtype.kind not in "iu" or
                np.any(point_instance < 0) or np.any(point_instance > np.iinfo(np.uint32).max)):
            raise ValueError("object_instance must contain one uint32-compatible integer per source point")
        predicted = np.zeros(count, dtype=np.uint8)
        expected = np.zeros(count, dtype=np.uint32)
        for number, (cls, selected, _) in enumerate(groups, 1):
            if not isinstance(cls, (int, np.integer)):
                raise ValueError("Target class must be an integer")
            check_target_code(cls)
            selected = np.asarray(selected)
            if (selected.ndim != 1 or selected.dtype.kind not in "iu" or
                    np.any(selected < 0) or np.any(selected >= count) or len(np.unique(selected)) != len(selected)):
                raise ValueError("Object point indices are invalid, repeated or out of range")
            if np.any(expected[selected]):
                raise ValueError("Exported objects have overlapping points")
            expected[selected] = number
            predicted[selected] = cls
        if not np.array_equal(expected, point_instance):
            raise ValueError("object_instance does not match exported object groups")
        if las_output == "1.4" and header.point_format.id < 6:  # As laspy.convert, without loading the points.
            crs = header.parse_crs()
            point_format = laspy.PointFormat(POINT_FORMAT_WITH_FULL_CLASS[header.point_format.id])
            point_format.dimensions.extend(header.point_format.extra_dimensions)
            header.set_version_and_point_format(laspy.header.Version.from_str("1.4"), point_format)
            if crs is not None:  # Formats 6-10 must store the CRS as WKT.
                header.add_crs(crs)
        full = header.point_format.id >= 6
        in_extra_field = any(not full and cls > 31 for cls, _, _ in groups)
        in_class_field = (predicted > 0) & (full | (predicted <= 31))
        for name, dtype, description in (
                ("object_instance", np.uint32, "Detected object number, 0 = none"),
                ("object_class", np.uint8, "Predicted target class, 0 = none")):
            if name in set(header.point_format.dimension_names):
                dimension = header.point_format.dimension_by_name(name)
                if dimension.dtype != np.dtype(dtype) or dimension.scales is not None or dimension.offsets is not None:
                    raise ValueError(f"Existing {name} field has an incompatible type or scaling")
            else:
                header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype, description=description))
        if header.version.minor >= 4:  # As LasData.update_header does before writing.
            header.start_of_waveform_data_packet_record = 0
            header.number_of_evlrs = len(header.evlrs) if header.evlrs is not None else 0
        else:
            header.number_of_evlrs = 0
        # Chunk by chunk, so memory does not grow with the file.
        with laspy.open(dest, mode="w", header=header) as writer:
            start = 0
            for points in reader.chunk_iterator(CHUNK_POINTS):
                stop = start + len(points)
                out = laspy.PackedPointRecord.from_point_record(points, header.point_format)
                codes = np.asarray(points.classification, dtype=np.uint8).copy()
                codes[in_class_field[start:stop]] = predicted[start:stop][in_class_field[start:stop]]
                out["classification"] = codes
                out["object_instance"] = point_instance[start:stop].astype(np.uint32)
                out["object_class"] = predicted[start:stop]
                writer.write_points(out)
                start = stop
            if header.version.minor >= 4 and header.evlrs:
                writer.write_evlrs(header.evlrs)
    return {"file": Path(dest).name, "las_version": str(header.version),
            "point_format": header.point_format.id,
            "target_codes_in": "object_class" if in_extra_field else "classification"}


# Rough share of a file's processing time per stage, for the progress bar.
STAGES = {"reading": 0.0, "classifying": 0.1, "grouping": 0.85, "writing": 0.9, "done": 1.0}


PREDICTIONS = "predictions.npz"  # Per-point class and confidence of a run, so it can be regrouped.
REGROUP_SETTINGS = ("threshold", "cluster_eps", "min_points", "area_method", "las_output")


def process_batch(paths, model_path, output_root, cfg, log=print, progress=None):
    """progress(file_number, file_count, name, stage, fraction_of_batch) is called as work advances."""
    validate_settings(cfg)
    if not paths:
        raise ValueError("Select at least one LAS / LAZ")
    model = load_model(model_path)
    cfg = Settings(**{**asdict(cfg), "neighbors": model["neighbors"]})
    info = {"model_id": model["model_id"], "model_path": str(Path(model_path).resolve()),
            "model_sha256": sha256(model_path), "backend": model["backend"], "classes": model["classes"]}

    def classify(index, path, info, stage, folder):
        check_disk(folder, info["point_count"])
        log(f"{Path(path).name}: classifying points")
        return classify_tiles(path, info["point_count"], model["estimator"], cfg.neighbors, folder / "tiles.tmp",
                              log, stage)
    return run_batch(paths, output_root, cfg, info, classify, log, progress)


def regroup_run(run_path, output_root, cfg, log=print, progress=None):
    """Group a previous run's saved point predictions again, with new grouping settings, into a new run.

    Only threshold, cluster radius, minimum points, area method and classified LAS version are taken from
    cfg; reading settings come from the previous run, and points are not classified again.
    """
    validate_settings(cfg)
    run_path = Path(run_path)
    previous = json.loads((run_path / "run_metadata.json").read_text(encoding="utf-8"))
    entries = [e for e in previous["datasets"] if e["status"] != "failed"]
    if not entries or not all((run_path / e["result_path"] / PREDICTIONS).exists() for e in entries):
        raise ValueError("These results have no saved point predictions (made by an older version); process again")
    cfg = Settings(**{**previous["settings"], **{k: getattr(cfg, k) for k in REGROUP_SETTINGS}})
    info = {k: previous[k] for k in ("model_id", "model_path", "model_sha256", "backend", "classes")}
    info["regrouped_from"] = str(run_path.resolve())

    def classify(index, path, info, stage, folder):
        stage("classifying")
        folder, digest = run_path / entries[index]["result_path"], entries[index]["source_sha256"]
        if sha256(path) != digest:
            raise ValueError("Input file changed since the previous run; process it again")
        with np.load(folder / PREDICTIONS, allow_pickle=False) as data:
            classes, confidence = data["classes"], data["confidence"]
        if len(classes) != info["point_count"]:
            raise ValueError("Saved predictions do not match the input point count")
        return classes, confidence
    return run_batch([e["source"] for e in entries], output_root, cfg, info, classify, log, progress)


def run_batch(paths, output_root, cfg, info, classify, log, progress):
    """Shared by processing and regrouping. classify(index, path, info, stage, folder) gives per-point class and
    confidence. Files are streamed, never held whole: memory grows by about PROCESS_BYTES_PER_POINT per point."""
    run = make_run_directory(paths[0], output_root)
    (run / "objects").mkdir()
    (run / "datasets").mkdir()
    summary = {"software_version": VERSION, "started_at": datetime.now().astimezone().isoformat(),
               "timezone": str(datetime.now().astimezone().tzinfo), "settings": asdict(cfg),
               **info,
               "definitions": {"centroid": "mean of exported original coordinates",
                   "Intensity": "arithmetic mean of recorded raw LAS intensity; no radiometric correction",
                   "Area": cfg.area_method,
                   "confidence": "mean winning-class point probability over final exported instance; uncalibrated",
                   "missing_value": "empty CSV cell / null JSON", "downsampling": "none for processing/export",
                   "instance_method": "per-class global DBSCAN within each dataset; separate files are not merged",
                   "processing": "file streamed in chunks; features per spatial tile, reading neighbours across tile "
                                 "edges until each point's are certain, so values equal whole-file processing",
                   "classified_laz": "input copy; detected target points get their target LAS class, others keep the input class"},
               "platform": platform.platform(), "datasets": []}
    write_json(run / "run_metadata.json", summary)
    for number, path in enumerate(paths, 1):
        dataset_id = f"D{number:04d}"
        folder = run / "datasets" / dataset_id
        folder.mkdir()
        entry = {"dataset_id": dataset_id, "source": str(Path(path).resolve()),
                 "result_path": str(folder.relative_to(run)), "status": "processing"}
        summary["datasets"].append(entry)
        log(f"{dataset_id}: {Path(path).name}")

        def stage(name, part=0.0):
            """part: share of this stage already done, moving the bar toward the next stage."""
            if progress:
                names = list(STAGES)
                share = STAGES[name] + part * (STAGES[names[names.index(name) + 1]] - STAGES[name])
                progress(number, len(paths), Path(path).name, name, (number - 1 + share) / len(paths))
        exported = []
        try:
            stage("reading")
            entry["input"] = source = open_cloud(path, cfg)
            check_memory(source["point_count"], process_memory(source["point_count"]))
            entry["source_sha256"] = sha256(path)
            log(f"{dataset_id}: {source['point_count']:,} points")
            classes, confidence = classify(number - 1, path, source, stage, folder)
            # Stored types are used here too, so regrouping with the same settings gives the same objects.
            classes, confidence = classes.astype(np.uint8), confidence.astype(np.float32)
            np.savez_compressed(folder / PREDICTIONS, classes=classes, confidence=confidence)
            stage("grouping")
            log(f"{dataset_id}: grouping points into objects")
            # Only candidate target points are read back and clustered, over the whole file at once, so an
            # object is never cut at a tile edge.
            candidates = np.flatnonzero((classes != NOT_TARGET) & (confidence >= cfg.threshold))
            found, origin = gather(path, candidates, source)
            local_groups = instance_groups(found["xyz"], classes[candidates], confidence[candidates], cfg)
            groups = [(cls, candidates[selected], score) for cls, selected, score in local_groups]
            stage("writing")
            log(f"{dataset_id}: writing {len(groups)} objects")
            point_instance = np.zeros(len(classes), dtype=np.int32)
            objects, rows = [], []
            for index, ((cls, selected, score), (_, own, _)) in enumerate(zip(groups, local_groups), 1):
                object_id = f"{dataset_id}_O{index:05d}"
                points = found["xyz"][own]
                point_instance[selected] = index
                center = points.mean(axis=0)
                area = projected_area(points, cfg.area_method)
                intensity = float(found["intensity"][own].mean())
                dest = run / "objects" / f"{object_id}.csv"
                exported.append(dest)
                with dest.open("x", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["x", "y", "z"])
                    writer.writerows(points)
                rows.append([object_id, *center, len(points), intensity, area, score])
                objects.append({"object_id": object_id, "instance_index": index, "class_id": cls,
                                "class_name": info["classes"][str(cls)], "point_count": len(points),
                                "centroid": center.tolist(), "confidence": score, "area": area,
                                "csv": str(dest.relative_to(run))})
            with (folder / "report.csv").open("x", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADERS)
                writer.writerows(rows)
            entry["classified"] = write_classified(path, folder / "classified.laz", groups, point_instance,
                                                   cfg.las_output)
            truth_status, targets = None, None
            if truth_path(path).exists():
                targets, truth_instance = load_truth(truth_path(path), len(classes),
                                                     source_sha256=entry["source_sha256"])
                summary_truth, truth_status = evaluate_objects(point_instance, objects, targets, truth_instance)
                write_json(folder / "evaluation.json", {"source": str(truth_path(path)), "summary": summary_truth,
                                                        "targets": targets})
                entry["evaluation"] = summary_truth
                log(f"{dataset_id}: ground truth: {summary_truth['true_positives']}/{len(targets)} targets found, "
                    f"{summary_truth['false_positives']} false alarms")
            web = web_indices(point_instance, keep=None if truth_status is None else truth_status > 0)
            shown, _ = gather(path, web)
            write_web_view(folder / "view3d.html", Path(path).name, shown["xyz"], point_instance[web],
                           objects, origin, len(classes), classes[web], info["classes"],
                           shown["intensity"], shown["classification"],
                           LAS_CLASSES if source["point_format"] >= 6 else LAS12_CLASSES,
                           None if truth_status is None else truth_status[web], targets)
            entry["display"] = {"source_points": len(classes), "shown_points": len(web), "limit": WEB_LIMIT,
                                "page": "view3d.html", "method": "object points first, then random background"}
            write_json(folder / "objects.json", objects)
            entry.update(status="success" if objects else "success_no_objects", object_count=len(objects))
            log(f"{dataset_id}: {len(objects)} objects")
        except Exception as exc:
            # Never expose partial outputs as successful objects.
            for dest in exported:
                dest.unlink(missing_ok=True)
            for filename in ("report.csv", "view3d.html", "objects.json", "classified.laz", "evaluation.json", PREDICTIONS):
                (folder / filename).unlink(missing_ok=True)
            shutil.rmtree(folder / "tiles.tmp", ignore_errors=True)
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log(f"{dataset_id}: FAILED: {exc}")
        write_json(folder / "status.json", entry)
        write_json(run / "run_metadata.json", summary)
    if progress:
        progress(len(paths), len(paths), "", "done", 1.0)
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(run / "run_metadata.json", summary)
    return str(run)
