"""Generate clearly synthetic LAS/LAZ/labels; never a real-world benchmark."""
import csv
import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

from core import Settings, process_batch, train_model, write_label_source


def create_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError("Demo destination already contains a manifest")
    records = []
    for seed, split in enumerate(("train", "validation", "test")):
        rng = np.random.default_rng(seed + 42)
        background = np.column_stack((rng.uniform(0, 8, 1800), rng.uniform(0, 8, 1800),
                                      rng.normal(-2.6, 0.015, 1800)))
        one = rng.uniform([-0.35, -0.3, -0.1], [0.35, 0.3, 0.1], size=(140, 3)) + [2, 2, -2]
        two = rng.uniform([-0.35, -0.3, -0.1], [0.35, 0.3, 0.1], size=(140, 3)) + [6, 5, -2]
        xyz = np.vstack((background, one, two))
        xyz += [500000 + seed * 100, 6000000, 0]
        labels = np.r_[np.zeros(len(background), dtype=int), np.full(len(one) + len(two), 64)]
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.001, 0.001, 0.001]
        header.offsets = [500000, 6000000, 0]
        header.add_crs(CRS.from_epsg(32632))
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = xyz.T
        cloud.intensity = np.where(labels, rng.integers(42000, 47000, len(labels)),
                                   rng.integers(5000, 8000, len(labels))).astype(np.uint16)
        cloud.return_number = np.where(labels, 2, 3).astype(np.uint8)
        cloud.number_of_returns = np.full(len(labels), 3, dtype=np.uint8)
        cloud.scan_angle_rank = rng.integers(-12, 12, len(labels), dtype=np.int8)
        cloud.point_source_id = np.full(len(labels), seed + 1, dtype=np.uint16)
        filename = destination / f"SYNTHETIC_{split}.laz"
        cloud.write(filename)
        label_path = filename.with_suffix(".labels.csv")
        with label_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_index", "class_id"])
            writer.writerows(enumerate(labels))
        write_label_source(label_path, filename, len(labels))
        records.append({"path": filename.name, "labels": label_path.name,
                        "split": split, "scene_id": f"SYNTHETIC_scene_{seed}"})
    (destination / "manifest.json").write_text(json.dumps(
        {"classes": {"64": "synthetic_object"}, "datasets": records}, indent=2), encoding="utf-8")
    (destination / "SYNTHETIC_ONLY.txt").write_text(
        "Synthetic geometry and intensity for software workflow checks only. Not real Ratman project data.\n"
        "A model trained on these data has no demonstrated real-world underwater detection performance.\n", encoding="utf-8")
    return destination / "manifest.json"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="synthetic_demo")
    parser.add_argument("--train-and-run", action="store_true")
    args = parser.parse_args()
    manifest = create_demo(args.destination)
    print(f"Synthetic manifest: {manifest.resolve()}")
    if args.train_and_run:
        cfg = Settings(coordinates_in_meters=True, vertical_datum="synthetic_local_z",
                       area_method="convex_hull_xy")
        model = manifest.parent / "SYNTHETIC_MODEL.joblib"
        train_model(manifest, model, cfg, iterations=40)
        print(process_batch([manifest.parent / "SYNTHETIC_test.laz"], model,
                            manifest.parent / "results", cfg))
