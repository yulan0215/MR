"""Put synthetic metal plates and spheres on the riverbed of real YellowScan scenes, with ground truth.

The scenes (terrain, vegetation, water surface, water column, riverbed) are real survey data; only the
targets are simulated. Results made with these scenes show that the software works end to end. They are
no evidence of how well real underwater targets would be detected.

Each target is sampled from the side the scanner sees (from above) with the local riverbed point density,
range noise, a brighter but noisy intensity, attributes copied from nearby bottom returns, imperfect
vendor classes, the riverbed hidden underneath it, and a few water-column returns above it.
"""
import csv
import json
import os
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from core import sha256, write_label_source

TARGETS = {64: "metal_plate", 65: "metal_sphere"}  # LAS user-definable class codes (64..255).
# Public YellowScan Navigator samples; set MR_PUBLIC_DATA to use another folder.
DATA = Path(os.environ.get("MR_PUBLIC_DATA", Path(__file__).resolve().parent / "data" / "public" / "yellowscan_navigator"))
SAINT_BAUZILLE = "BC40-NA-NOA-80mAGL-7mps-High-SACL-Saint_Bauzille_de_Putois-1.las"
DEVILS_BRIDGE = "BC40-NA-M600-80mAGL-5mps-Medium-SACLCO-Devils Bridge.laz"
# (name, split, source file, box xmin, ymin, xmax, ymax in EPSG:2154 metres, number of targets)
SCENES = [
    ("SAINT_BAUZILLE_A", "train", SAINT_BAUZILLE, (758844, 6313200, 759044, 6313350), 70),
    ("SAINT_BAUZILLE_B", "validation", SAINT_BAUZILLE, (759144, 6313000, 759260, 6313200), 30),
    ("DEVILS_BRIDGE", "test", DEVILS_BRIDGE, (744877, 6289525, 745027, 6289925), 40),
]
SPACING = 6.0  # Minimum distance between targets, metres.
MIN_BED_DENSITY = 8.0  # Bottom points per m² needed around a target position.
RANGE_NOISE = (0.02, 0.03)  # Standard deviation horizontal, vertical (m).


def crop(path, box):
    las = laspy.read(path)
    x, y = np.asarray(las.x), np.asarray(las.y)
    inside = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
    return las, inside


def surface_points(kind, rng, size, count):
    """Points on the visible (upper) surface in a local frame: centre of the footprint at (0, 0), base at z=0.

    Returns points, the surface's z component of the normal (for intensity) and footprint samples.
    """
    if kind == "metal_sphere":
        radius = size / 2
        burial = rng.uniform(0, 0.3) * size
        rho = radius * np.sqrt(rng.uniform(0, 1, count))
        angle = rng.uniform(0, 2 * np.pi, count)
        top = np.sqrt(np.maximum(radius ** 2 - rho ** 2, 0))
        points = np.column_stack((rho * np.cos(angle), rho * np.sin(angle), radius - burial + top))
        visible = points[:, 2] > 0.01  # The buried part is not seen.
        ring = np.linspace(0, 2 * np.pi, 48, endpoint=False)  # Burial is under half: footprint is the full disc.
        footprint = np.column_stack((radius * np.cos(ring), radius * np.sin(ring)))
        return points[visible], top[visible] / radius, footprint, {"diameter_m": size, "burial_m": burial}
    thickness = rng.uniform(0.005, 0.03)
    tilt = np.radians(rng.uniform(0, 25))
    yaw, axis = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
    raised = rng.uniform(0, 0.1)  # Resting on debris or stones.
    uv = rng.uniform(-size / 2, size / 2, (count, 2))
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2

    def place(local):
        cos, sin = np.cos(yaw), np.sin(yaw)
        xy = local @ np.array([[cos, sin], [-sin, cos]])
        z = np.tan(tilt) * (xy[:, 0] * np.cos(axis) + xy[:, 1] * np.sin(axis))
        return xy, z
    xy, z = place(uv)
    corner_xy, corner_z = place(corners)
    base = raised + thickness - corner_z.min()
    points = np.column_stack((xy, z + base))
    normal_z = np.full(len(points), np.cos(tilt))
    return points, normal_z, corner_xy, {"side_m": size, "thickness_m": thickness, "tilt_deg": np.degrees(tilt),
                                          "raised_m": raised}


def inject(las, inside, rng, count, split):
    """Returns the new LasData, per-point labels and the ground-truth targets."""
    xyz = np.column_stack((las.x, las.y, las.z))[inside]
    record = las.points.array[inside]
    codes = np.asarray(las.classification)[inside]
    intensity = np.asarray(las.intensity, dtype=np.float64)[inside]
    bed = np.flatnonzero(codes == 40)
    bed_tree = cKDTree(xyz[bed, :2])
    low, high = xyz[:, :2].min(axis=0) + 4, xyz[:, :2].max(axis=0) - 4
    remove = np.zeros(len(xyz), dtype=bool)
    new_xyz, new_intensity, new_source, new_codes, labels, targets = [], [], [], [], [], []
    placed = []
    for anchor in rng.permutation(bed):
        if len(targets) == count:
            break
        centre = xyz[anchor]
        if (centre[:2] < low).any() or (centre[:2] > high).any():
            continue
        if placed and np.min(np.hypot(*(np.array(placed) - centre[:2]).T)) < SPACING:
            continue
        near = bed[bed_tree.query_ball_point(centre[:2], 2.0)]
        density = len(near) / (np.pi * 4)
        if density < MIN_BED_DENSITY:
            continue
        bed_z = float(np.median(xyz[bed[bed_tree.query_ball_point(centre[:2], 1.0)], 2]))
        kind = TARGETS[64] if rng.random() < 0.5 else TARGETS[65]
        size = rng.uniform(0.3, 1.0)
        gain = rng.uniform(0.8, 1.5)  # Metal returns more often than sediment, but varies with angle.
        footprint_area = size ** 2 if kind == "metal_plate" else np.pi * size ** 2 / 4
        n = rng.poisson(density * footprint_area * gain)
        points, normal_z, footprint, shape = surface_points(kind, rng, size, n)
        points += [centre[0], centre[1], bed_z]
        points[:, :2] += rng.normal(0, RANGE_NOISE[0], (len(points), 2))
        points[:, 2] += rng.normal(0, RANGE_NOISE[1], len(points))
        # Hide everything the target covers below its top: the bed underneath and returns inside it.
        top = points[:, 2].max() if len(points) else bed_z + 0.05
        hull = Delaunay(footprint + centre[:2])
        candidates = np.flatnonzero(np.abs(xyz[:, :2] - centre[:2]).max(axis=1) < size)
        covered = candidates[(hull.find_simplex(xyz[candidates, :2]) >= 0) & (xyz[candidates, 2] < top)]
        remove[covered] = True
        bright = np.median(intensity[near]) * rng.uniform(1.5, 4.0)
        value = bright * (0.4 + 0.6 * normal_z) * rng.lognormal(0, 0.25, len(points))
        source = near[cKDTree(xyz[near, :2]).query(points[:, :2])[1]] if len(points) else near[:0]
        vendor = rng.choice([40, 45, 1], len(points), p=[0.85, 0.1, 0.05])  # Imperfect vendor classes.
        code = next(k for k, v in TARGETS.items() if v == kind)
        # A few water-column returns above the target (turbidity, fish, bubbles): background.
        m = rng.poisson(3)
        column = np.column_stack((centre[0] + rng.normal(0, 0.5, m), centre[1] + rng.normal(0, 0.5, m),
                                  top + rng.uniform(0.1, 1.0, m)))
        column_source = near[rng.integers(0, len(near), m)]
        start = sum(len(p) for p in new_xyz)
        new_xyz += [points, column]
        new_intensity += [value, intensity[column_source] * rng.uniform(0.3, 1.0, m)]
        new_source += [source, column_source]
        new_codes += [vendor, np.full(m, 45)]
        labels += [np.full(len(points), code), np.zeros(m, dtype=int)]
        placed.append(centre[:2])
        targets.append({"object_id": f"T{len(targets) + 1:03d}", "class_id": code, "class_name": kind,
                        "center": (points.mean(axis=0) if len(points) else [*centre[:2], bed_z]).tolist(),
                        "bed_z": bed_z, "bed_density_per_m2": density, "simulated_points": len(points),
                        "offset": start, **{k: float(v) for k, v in shape.items()}})
    if len(targets) < count:
        raise ValueError(f"Only room for {len(targets)} of {count} targets on this riverbed")
    keep = np.flatnonzero(~remove)
    added_xyz = np.vstack(new_xyz)
    added = record[np.concatenate(new_source)].copy()
    out = laspy.LasData(laspy.LasHeader(point_format=las.header.point_format, version=las.header.version))
    out.header.offsets, out.header.scales = las.header.offsets, las.header.scales
    for vlr in las.header.vlrs:
        out.header.vlrs.append(vlr)
    out.points = laspy.ScaleAwarePointRecord(np.concatenate((record[keep], added)), las.header.point_format,
                                             las.header.scales, las.header.offsets)
    out.x = np.r_[xyz[keep, 0], added_xyz[:, 0]]
    out.y = np.r_[xyz[keep, 1], added_xyz[:, 1]]
    out.z = np.r_[xyz[keep, 2], added_xyz[:, 2]]
    out.intensity = np.clip(np.r_[intensity[keep], np.concatenate(new_intensity)], 0, 65535).astype(np.uint16)
    out.classification = np.r_[codes[keep], np.concatenate(new_codes)].astype(np.uint8)
    point_labels = np.r_[np.zeros(len(keep), dtype=int), np.concatenate(labels)]
    for target in targets:
        start = len(keep) + target.pop("offset")
        target["point_range"] = [start, start + target["simulated_points"]]
    return out, point_labels, targets


def make_scene(name, split, source, box, count, destination, seed):
    rng = np.random.default_rng(seed)
    las, inside = crop(DATA / source, box)
    out, labels, targets = inject(las, inside, rng, count, split)
    path = destination / f"{name}.laz"
    out.write(path)
    if split == "train":  # For training speed, only 5% of points off the water (-1 = unlabelled) are used.
        water = np.isin(np.asarray(out.classification), (40, 45))
        labels = np.where((labels == 0) & ~water & (rng.random(len(labels)) > 0.05), -1, labels)
    with path.with_suffix(".labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "class_id"])
        writer.writerows((i, "" if v < 0 else v) for i, v in enumerate(labels))
    write_label_source(path.with_suffix(".labels.csv"), path, len(labels))
    path.with_suffix(".truth.json").write_text(json.dumps(
        {"point_count": len(out.points), "source_sha256": sha256(path), "synthetic_targets": True, "source": source, "box": box,
         "classes": {str(k): v for k, v in TARGETS.items()}, "objects": targets}, indent=1), encoding="utf-8")
    return {"path": path.name, "labels": path.with_suffix(".labels.csv").name, "split": split, "scene_id": name}


def create_metal_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError("Destination already contains a manifest")
    records = []
    for seed, (name, split, source, box, count) in enumerate(SCENES):
        print(f"{name}: placing {count} targets")
        records.append(make_scene(name, split, source, box, count, destination, seed + 7))
    classes = {str(k): v for k, v in TARGETS.items()}
    manifest = destination / "manifest.json"
    # Unweighted classes: with class balancing the riverbed gave several times more false alarms.
    manifest.write_text(json.dumps({"classes": classes, "balance_classes": False, "datasets": records}, indent=2),
                        encoding="utf-8")
    (destination / "SYNTHETIC_TARGETS.txt").write_text(
        "Real YellowScan scenes with simulated metal plates and spheres on the riverbed.\n"
        "Detection results on these files show that the workflow runs; they are not evidence of real-world\n"
        "underwater target detection performance.\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    from core import Settings, process_batch, train_model
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("destination", nargs="?", default="metal_demo")
    parser.add_argument("--train-and-run", action="store_true")
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()
    manifest = create_metal_demo(args.destination)
    print(f"Manifest: {manifest.resolve()}")
    if args.train_and_run:
        cfg = Settings(coordinates_in_meters=True, vertical_datum="as delivered (unconfirmed)",
                       threshold=0.6, cluster_eps=0.3, min_points=8, area_method="convex_hull_xy")
        model = manifest.parent / "METAL_MODEL.joblib"
        train_model(manifest, model, cfg, iterations=args.trees)
        print(process_batch([manifest.parent / "DEVILS_BRIDGE.laz"], model, manifest.parent / "results", cfg))
