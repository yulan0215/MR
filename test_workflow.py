"""Meaningful workflow checks on synthetic LAZ. Run: python -m unittest -v."""
import base64
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

import core
from core import (HEADERS, Settings, dbscan, display_indices, evaluate_model, extract_features, inspect_file,
                  instance_groups, load_manifest, load_model, make_run_directory, process_batch,
                  projected_area, read_cloud, train_model)
from core import refresh_web_view
from create_demo import create_demo
from synthetic_targets import DATA
from sklearn.cluster import DBSCAN



REAL_LAS_12 = Path(DATA, "SW_NAV_60mAGL_5ms_SACLCOmod_EPSG_32633_Barsebackshamn_harbour-F001.las")


def web_data(folder):
    page = (Path(folder) / "view3d.html").read_text(encoding="utf-8")
    return json.loads(page.split('type="application/json">', 1)[1].split("</script>", 1)[0])


class WorkflowTest(unittest.TestCase):
    def test_display_sampling_bounds_and_preserves_small_objects(self):
        instance = np.zeros(100_000, dtype=np.int32)
        instance[123] = 1
        instance[-7:] = 2
        for budget in (1000, 5000, 30000):
            selected = display_indices(instance, budget)
            self.assertEqual(len(selected), budget)
            self.assertEqual(len(np.unique(selected)), budget)
            self.assertEqual(set(instance[selected]), {0, 1, 2})
            np.testing.assert_array_equal(selected, display_indices(instance, budget))
        np.testing.assert_array_equal(display_indices(instance[:10]), np.arange(10))
        with self.assertRaises(ValueError):
            display_indices(instance, 30001)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.manifest = create_demo(cls.root / "synthetic")
        cls.cfg = Settings(coordinates_in_meters=True, vertical_datum="synthetic_local_z",
                           area_method="convex_hull_xy")
        cls.model = cls.root / "model.joblib"
        train_model(cls.manifest, cls.model, cls.cfg, iterations=30, log=lambda x: None)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_batch_export_and_failure_isolation(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample, self.root / "missing.laz", sample], self.model,
                                 self.root / "runs", self.cfg, log=lambda x: None))
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual([e["status"] for e in metadata["datasets"]], ["success", "failed", "success"])
        all_ids = set()
        original = read_cloud(sample, self.cfg)["xyz"]
        for key in ("D0001", "D0003"):
            folder = run / "datasets" / key
            with (folder / "report.csv").open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, HEADERS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            view = web_data(folder)
            self.assertEqual(view["total"], len(original))
            shown = np.frombuffer(base64.b64decode(view["xyz"]), dtype="<f4").reshape(-1, 3)
            np.testing.assert_allclose(shown, original - original.mean(axis=0), atol=1e-3)
            instance = np.frombuffer(base64.b64decode(view["instance"]), dtype="<i4")
            # Small cloud: every point is in the view, in source order.
            for row, obj in zip(rows, objects):
                object_id = row["object ID"]
                self.assertNotIn(object_id, all_ids)
                all_ids.add(object_id)
                self.assertEqual(object_id, obj["object_id"])
                point_file = run / "objects" / f"{object_id}.csv"
                points = np.loadtxt(point_file, delimiter=",", skiprows=1, encoding="utf-8-sig")
                self.assertEqual(len(points), int(row["Nr_of_points"]))
                self.assertTrue(np.allclose(points.mean(axis=0), [float(row[k]) for k in ("x", "y", "z")]))
                self.assertTrue(np.array_equal(points, original[instance == obj["instance_index"]]))
                self.assertGreaterEqual(float(row["confidence"]), self.cfg.threshold)
        other = Path(process_batch([sample], self.model, self.root / "runs", self.cfg, log=lambda x: None))
        self.assertNotEqual(run, other)

    def test_no_objects_is_not_failure(self):
        cfg = Settings(**{**self.cfg.__dict__, "min_points": 1000, "area_method": "unresolved"})
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "empty", cfg, log=lambda x: None))
        entry = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))["datasets"][0]
        self.assertEqual(entry["status"], "success_no_objects")
        with (run / "datasets/D0001/report.csv").open(encoding="utf-8-sig") as f:
            self.assertEqual(len(list(csv.reader(f))), 1)

    def test_regroup_reuses_predictions_without_reclassifying(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "regroup", self.cfg, log=lambda x: None))
        objects = (run / "datasets/D0001/objects.json").read_text(encoding="utf-8")
        same = Path(core.regroup_run(run, self.root / "regroup", Settings(**{**self.cfg.__dict__, "neighbors": 7}),
                                     log=lambda x: None))
        # Same grouping settings give the same objects; reading settings stay those of the first run.
        self.assertEqual((same / "datasets/D0001/objects.json").read_text(encoding="utf-8"), objects)
        metadata = json.loads((same / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["regrouped_from"], str(run.resolve()))
        self.assertEqual(metadata["settings"]["neighbors"], self.cfg.neighbors)
        self.assertEqual(metadata["model_id"], json.loads((run / "run_metadata.json").read_text())["model_id"])
        strict = Settings(**{**self.cfg.__dict__, "min_points": 1000})
        empty = Path(core.regroup_run(same, self.root / "regroup", strict, log=lambda x: None))
        entry = json.loads((empty / "run_metadata.json").read_text(encoding="utf-8"))["datasets"][0]
        self.assertEqual((entry["status"], entry["object_count"]), ("success_no_objects", 0))
        self.assertEqual(json.loads((empty / "run_metadata.json").read_text())["settings"]["min_points"], 1000)

    def test_regroup_refuses_changed_input_and_old_runs(self):
        sample = self.root / "regroup_input.laz"
        shutil.copy(self.manifest.parent / "SYNTHETIC_test.laz", sample)
        run = Path(process_batch([sample], self.model, self.root / "regroup_checks", self.cfg, log=lambda x: None))
        las = laspy.read(sample)
        las.intensity = las.intensity // 2
        las.write(sample)
        changed = Path(core.regroup_run(run, self.root / "regroup_checks", self.cfg, log=lambda x: None))
        entry = json.loads((changed / "run_metadata.json").read_text(encoding="utf-8"))["datasets"][0]
        self.assertEqual(entry["status"], "failed")
        self.assertIn("changed", entry["error"])
        (run / "datasets/D0001" / core.PREDICTIONS).unlink()
        with self.assertRaisesRegex(ValueError, "no saved point predictions"):
            core.regroup_run(run, self.root / "regroup_checks", self.cfg, log=lambda x: None)

    def test_inspect_reads_header_without_settings(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        info = inspect_file(sample)
        cloud = read_cloud(sample, self.cfg)
        self.assertEqual(info["point_count"], len(cloud["xyz"]))
        np.testing.assert_allclose(info["mins"], cloud["xyz"].min(axis=0), atol=1e-3)
        np.testing.assert_allclose(info["maxs"], cloud["xyz"].max(axis=0), atol=1e-3)
        self.assertIn("32632", info["crs"])
        self.assertEqual(info["issues"], [])
        self.assertIn("intensity", info["fields"])

    def test_run_metadata_records_model_path(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "meta", self.cfg, log=lambda x: None))
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_path"], str(self.model.resolve()))
        self.assertEqual(metadata["settings"]["threshold"], self.cfg.threshold)

    def test_progress_reports_each_file_and_finishes(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        calls = []
        process_batch([sample, self.root / "missing.laz"], self.model, self.root / "progress", self.cfg,
                      log=lambda x: None, progress=lambda *a: calls.append(a))
        fractions = [c[4] for c in calls]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(calls[-1][3:], ("done", 1.0))
        self.assertEqual({c[0] for c in calls if c[3] == "reading"}, {1, 2})

    def test_saved_page_moves_to_current_viewer(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "refresh", self.cfg, log=lambda x: None))
        page = run / "datasets" / "D0001" / "view3d.html"
        before = web_data(page.parent)
        page.write_text(page.read_text(encoding="utf-8").replace("setVisibleObjects", "oldViewer"), encoding="utf-8")
        refresh_web_view(page)
        self.assertIn("setVisibleObjects", page.read_text(encoding="utf-8"))
        self.assertEqual(web_data(page.parent), before)

    def test_area_definition(self):
        square = np.array([[500000, 6000000, -2], [500002, 6000000, -2],
                           [500002, 6000003, -2], [500000, 6000003, -2]])
        self.assertAlmostEqual(projected_area(square, "convex_hull_xy"), 6.0)
        self.assertIsNone(projected_area(square, "unresolved"))
        self.assertEqual(projected_area(square[:2], "convex_hull_xy"), 0)

    def test_units_limit_and_no_training_overwrite(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        with self.assertRaisesRegex(ValueError, "Confirm"):
            read_cloud(sample, Settings())
        with self.assertRaisesRegex(ValueError, "exceeds"):
            read_cloud(sample, Settings(max_points=100, coordinates_in_meters=True))
        with self.assertRaises(FileExistsError):
            train_model(self.manifest, self.model, self.cfg)

    def test_no_point_limit_by_default_but_memory_guard(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        self.assertEqual(Settings().max_points, 0)
        self.assertGreater(len(read_cloud(sample, self.cfg)["xyz"]), 0)
        total = core.physical_memory()
        if total:
            with self.assertRaisesRegex(ValueError, "Split the file"):
                core.check_memory(total // core.BYTES_PER_POINT)

    def test_browser_view_embeds_points_and_objects(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "web", self.cfg, log=lambda x: None))
        folder = run / "datasets" / "D0001"
        data = web_data(folder)
        count = len(read_cloud(sample, self.cfg)["xyz"])
        self.assertEqual(data["total"], count)
        self.assertEqual(len(base64.b64decode(data["xyz"])), count * 12)
        instance = np.frombuffer(base64.b64decode(data["instance"]), dtype="<i4")
        objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
        self.assertEqual({o[1] for o in data["objects"]}, {o["object_id"] for o in objects})
        self.assertEqual(set(instance) - {0}, {o["instance_index"] for o in objects})
        self.assertEqual(data["classes"], [[0, "not a target"], [64, "synthetic_object"]])
        cls = np.frombuffer(base64.b64decode(data["cls"]), dtype=np.uint8)
        self.assertEqual(len(cls), count)
        self.assertTrue(set(cls[instance > 0]) - {0})  # Object points carry a target class.
        self.assertEqual(len(base64.b64decode(data["intensity"])), count)

    def test_chunked_dbscan_matches_sklearn_core_clusters(self):
        rng = np.random.default_rng(0)
        centres = rng.uniform(0, 20, (40, 3))
        points = np.concatenate([c + rng.normal(0, 0.25, (300, 3)) for c in centres] +
                                [rng.uniform(0, 20, (2000, 3))])
        reference = DBSCAN(eps=0.3, min_samples=10).fit(points)
        core_idx = reference.core_sample_indices_
        # Small chunks force clusters to be joined across many chunk boundaries.
        labels = dbscan(points, 0.3, 10, chunk=500)
        np.testing.assert_array_equal(labels[core_idx], reference.labels_[core_idx])
        np.testing.assert_array_equal(labels == -1, reference.labels_ == -1)

    def test_scene_and_content_leakage_checks(self):
        source = json.loads(self.manifest.read_text())
        source["datasets"][1]["scene_id"] = source["datasets"][0]["scene_id"]
        changed = self.manifest.parent / "bad_scene.json"
        changed.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_manifest(changed)
        source["datasets"][1]["scene_id"] = "different"
        source["datasets"][1]["path"] = source["datasets"][0]["path"]
        changed.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            load_manifest(changed)

    def test_model_reload_test_evaluation_and_mlp_backend(self):
        loaded = load_model(self.model)
        self.assertIn("validation", loaded)
        self.assertNotIn("test_metrics", loaded)
        dest = self.root / "test_metrics.json"
        evaluate_model(self.model, dest, self.cfg, log=lambda x: None)
        result = json.loads(dest.read_text())
        self.assertEqual(result["class_order"], [0, 64])
        self.assertIn("confusion_matrix", result)
        mlp_path = self.root / "mlp.joblib"
        train_model(self.manifest, mlp_path, self.cfg, backend="mlp", iterations=2, log=lambda x: None)
        self.assertEqual(load_model(mlp_path)["backend"], "mlp")

    def test_relocated_model_and_test_data(self):
        relocated = self.root / "relocated"
        relocated.mkdir()
        shutil.copytree(self.manifest.parent, relocated / "synthetic")
        shutil.copy2(self.model, relocated / "model.joblib")
        evaluate_model(relocated / "model.joblib", relocated / "metrics.json", self.cfg, log=lambda x: None)
        self.assertTrue((relocated / "metrics.json").exists())

    def test_global_instances_cross_arbitrary_tile_boundary(self):
        xyz = np.array([[x, y, 0] for x in np.arange(-0.5, 0.51, 0.1) for y in (0, 0.1)])
        cfg = Settings(cluster_eps=0.16, min_points=3)
        groups = instance_groups(xyz, np.ones(len(xyz), dtype=int), np.ones(len(xyz)), cfg)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0][1]), len(xyz))

    def tiling_scene(self):
        """60 x 40 m riverbed with water surface, grid points (many equal distances), duplicate points and
        isolated points far from all others, so tiles need wider margins."""
        rng = np.random.default_rng(7)
        bed = np.column_stack((rng.uniform(0, 60, 9000), rng.uniform(0, 40, 9000), rng.normal(-3, 0.05, 9000)))
        grid = np.array([[x, y, -2.5] for x in np.arange(20, 26, 0.1) for y in np.arange(10, 13, 0.1)])
        surface = np.column_stack((rng.uniform(0, 60, 3000), rng.uniform(0, 40, 3000), np.zeros(3000)))
        lonely = np.array([[30, 20, 15], [30.5, 20, 15], [59, 1, 9], [1, 39, 12]])
        plates = np.vstack([rng.uniform([-0.35, -0.3, -0.1], [0.35, 0.3, 0.1], (140, 3)) + [x, y, -2.4]
                            for x, y in ((8, 8), (30, 20), (45, 30), (15, 33), (52, 12))])
        xyz = np.vstack((bed, grid, grid[:200], surface, lonely, plates))
        classes = np.r_[np.full(len(bed), 40), np.full(len(grid) + 200, 1), np.full(len(surface), 41),
                        np.full(len(lonely) + len(plates), 1)]
        order = rng.permutation(len(xyz))  # Acquisition order is not spatial.
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales, header.offsets = [0.001] * 3, [500000, 6000000, 0]
        header.add_crs(CRS.from_epsg(32632))
        las = laspy.LasData(header)
        las.x, las.y, las.z = (xyz[order] + [500000, 6000000, 0]).T
        las.intensity = rng.integers(100, 3000, len(xyz)).astype(np.uint16)
        las.classification = classes[order].astype(np.uint8)
        las.return_number = las.number_of_returns = np.ones(len(xyz), dtype=np.uint8)
        las.scan_angle = rng.integers(-2000, 2000, len(xyz)).astype(np.int16)
        path = self.root / "tiling_scene.laz"
        if not path.exists():
            las.write(path)
        return path

    def test_tiled_features_equal_whole_file(self):
        path = self.tiling_scene()
        cloud = read_cloud(path, self.cfg)
        whole = extract_features(cloud)
        self.assertTrue(cloud["classification"].tolist().count(40) >= 10)
        saved = core.TILE_POINTS
        try:
            core.TILE_POINTS = 1500
            tiles, origin, grid = core.make_tiles(path, self.root / "tiles_check")
            self.assertGreater(len(tiles), 5)
            self.assertEqual(sum(t["count"] for t in tiles), len(whole))
            self.assertLessEqual(max(t["count"] for t in tiles), 1500 * 2)
            tiled = np.full_like(whole, np.nan)
            for number in range(len(tiles)):
                index, values = core.tile_features(tiles, number, 20, origin, grid)
                tiled[index] = values
        finally:
            core.TILE_POINTS = saved
            shutil.rmtree(self.root / "tiles_check", ignore_errors=True)
        np.testing.assert_array_equal(tiled, whole)  # Bit for bit, including bed height and isolated points.

    def test_tiled_run_equals_one_tile_run(self):
        path = self.tiling_scene()
        cfg = Settings(**{**self.cfg.__dict__, "threshold": 0.0, "min_points": 3})
        runs = []
        saved = core.TILE_POINTS
        try:
            for tile_points in (10 ** 9, 1500):
                core.TILE_POINTS = tile_points
                runs.append(Path(process_batch([path], self.model, self.root / f"tiled_{tile_points}", cfg,
                                               log=lambda x: None)) / "datasets" / "D0001")
        finally:
            core.TILE_POINTS = saved
        one, tiled = runs
        self.assertFalse((tiled / "tiles.tmp").exists())
        self.assertGreater(len(json.loads((tiled / "objects.json").read_text(encoding="utf-8"))), 0)
        for name in ("objects.json", "report.csv", "view3d.html"):
            self.assertEqual((one / name).read_bytes(), (tiled / name).read_bytes(), name)
        with np.load(one / core.PREDICTIONS) as a, np.load(tiled / core.PREDICTIONS) as b:
            np.testing.assert_array_equal(a["classes"], b["classes"])
            np.testing.assert_array_equal(a["confidence"], b["confidence"])
        self.assertEqual((one / "classified.laz").read_bytes(), (tiled / "classified.laz").read_bytes())

    def test_equally_near_neighbours_do_not_depend_on_point_order(self):
        xyz = np.array([[x, y, 0.0] for x in np.arange(0, 3, 0.5) for y in np.arange(0, 3, 0.5)])
        rng = np.random.default_rng(3)
        cloud = {"xyz": xyz, "intensity": rng.integers(0, 1000, len(xyz)).astype(float),
                 "return_number": np.ones(len(xyz)), "number_of_returns": np.ones(len(xyz)),
                 "scan_angle": np.zeros(len(xyz))}
        order = rng.permutation(len(xyz))
        shuffled = {k: v[order] for k, v in cloud.items()}
        shuffled["index"] = order
        reference = extract_features(cloud, 6)
        features = extract_features(shuffled, 6, origin=xyz[0])
        np.testing.assert_array_equal(features, reference[order])

    def test_forest_probabilities_are_reproducible(self):
        model = load_model(self.model)["estimator"]
        x = np.random.default_rng(0).normal(size=(200_000, len(core.FEATURES))).astype(np.float32)
        first = core.predict(model, x)
        np.testing.assert_array_equal(first, core.predict(model, x))
        np.testing.assert_array_equal(first[:5000], model.predict_proba(x[:5000]))

    def test_features_do_not_depend_on_map_translation(self):
        cloud = read_cloud(self.manifest.parent / "SYNTHETIC_test.laz", self.cfg)
        shifted = {**cloud, "xyz": cloud["xyz"] + [10000, -10000, 5]}
        self.assertTrue(np.allclose(extract_features(cloud), extract_features(shifted), atol=1e-5))

    def test_bed_height_ignores_a_small_target_on_the_bed(self):
        grid = np.array([[x, y, 10.0] for x in np.arange(0, 10, 0.2) for y in np.arange(0, 10, 0.2)])
        target = np.array([[5 + dx, 5 + dy, 10.5] for dx in (-0.2, 0, 0.2) for dy in (-0.2, 0, 0.2)])
        xyz = np.vstack((grid, target))
        height, known = core.bed_height(xyz, np.full(len(xyz), 40, dtype=np.uint8))
        self.assertTrue(known.all())
        np.testing.assert_allclose(height[len(grid):], 0.5, atol=1e-9)
        height, known = core.bed_height(xyz, np.full(len(xyz), 2, dtype=np.uint8))
        self.assertFalse(known.any())

    def test_object_evaluation_matches_one_to_one(self):
        point_instance = np.array([0, 1, 1, 1, 2, 2, 0, 0, 3, 3])
        truth_instance = np.array([0, 1, 1, 1, 1, 0, 2, 2, 0, 0])
        objects = [{"instance_index": i, "object_id": f"O{i}", "class_id": c, "centroid": [0, 0, 0]}
                   for i, c in ((1, 64), (2, 64), (3, 65))]
        targets = [{"object_id": "T1", "class_id": 64, "class_name": "metal_plate", "center": [0, 0, 1]},
                   {"object_id": "T2", "class_id": 65, "class_name": "metal_sphere", "center": [0, 0, 0]}]
        summary, status = core.evaluate_objects(point_instance, objects, targets, truth_instance)
        self.assertEqual((summary["true_positives"], summary["false_positives"], summary["missed"]), (1, 2, 1))
        self.assertEqual(objects[0]["truth"]["target_id"], "T1")
        self.assertAlmostEqual(objects[0]["truth"]["iou"], 0.75)
        self.assertTrue(objects[0]["truth"]["class_correct"])
        self.assertIsNone(objects[1]["truth"]["target_id"])  # T1 is already taken by the better match.
        self.assertEqual(summary["missed_targets"], ["T2"])
        np.testing.assert_array_equal(status, [0, 1, 1, 1, 2, 2, 3, 3, 2, 2])

    def test_classified_laz_for_las_1_2_and_1_4(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        original = laspy.read(sample)
        self.assertEqual((str(original.header.version), original.header.point_format.id), ("1.2", 3))
        original.classification = np.full(len(original.points), 2, dtype=np.uint8)
        original.user_data = np.full(len(original.points), 7, dtype=np.uint8)
        source = self.root / "classes.laz"
        original.write(source)
        selected, other = np.arange(5, 15), np.arange(20, 25)
        instance = np.zeros(len(original.points), dtype=np.int32)
        instance[selected], instance[other] = 1, 2
        groups = [(64, selected, 0.9), (6, other, 0.8)]
        # Same version: code 64 goes to an extra field; User Data is preserved.
        dest = self.root / "classified_12.laz"
        info = core.write_classified(source, dest, groups, instance)
        self.assertEqual(info, {"file": "classified_12.laz", "las_version": "1.2", "point_format": 3,
                                "target_codes_in": "object_class"})
        result = laspy.read(dest)
        codes, user = np.asarray(result.classification), np.asarray(result.user_data)
        self.assertTrue((codes[selected] == 2).all() and (result.object_class[selected] == 64).all())
        np.testing.assert_array_equal(user, original.user_data)
        self.assertTrue((codes[other] == 6).all() and (user[other] == 7).all())
        np.testing.assert_array_equal(result.object_instance, instance)
        np.testing.assert_allclose(result.x, original.x)
        # Labels read back from that LAS 1.2 file recover the target codes.
        labels = core.load_labels(dest, len(codes), {64, 6}, source_xyz=np.asarray(original.xyz),
                                  label_field="object_class")
        self.assertTrue((labels[selected] == 64).all() and (labels[other] == 6).all())
        self.assertEqual(int((labels > 0).sum()), len(selected) + len(other))
        # LAS 1.4: matching point format, every code in the class field, CRS stored as WKT.
        dest = self.root / "classified_14.laz"
        info = core.write_classified(source, dest, groups, instance, "1.4")
        self.assertEqual((info["las_version"], info["point_format"], info["target_codes_in"]),
                         ("1.4", 7, "classification"))
        result = laspy.read(dest)
        codes = np.asarray(result.classification)
        self.assertTrue((codes[selected] == 64).all() and (codes[other] == 6).all())
        self.assertTrue((np.delete(codes, np.r_[selected, other]) == 2).all())
        self.assertTrue(result.header.global_encoding.wkt)
        self.assertEqual(result.header.parse_crs().to_epsg(), 32632)
        np.testing.assert_array_equal(result.intensity, original.intensity)
        with self.assertRaisesRegex(ValueError, "Classified LAS"):
            core.validate_settings(Settings(las_output="1.3"))

    @unittest.skipUnless(REAL_LAS_12.exists(), "public YellowScan LAS 1.2 sample not available")
    def test_real_yellowscan_las_1_2_file(self):
        las = laspy.read(REAL_LAS_12)
        self.assertEqual((str(las.header.version), las.header.point_format.id), ("1.2", 3))
        x, y = np.asarray(las.x), np.asarray(las.y)
        keep = (np.abs(x - np.median(x)) < 15) & (np.abs(y - np.median(y)) < 15)
        crop = laspy.LasData(las.header)
        crop.points = las.points[keep].copy()
        path = self.root / "real_12.laz"
        crop.write(path)
        cloud = read_cloud(path, self.cfg)
        self.assertEqual(set(np.unique(cloud["classification"])) - {1, 2, 7, 9}, set())
        self.assertEqual(extract_features(cloud).shape, (keep.sum(), len(core.FEATURES)))
        run = Path(process_batch([path], self.model, self.root / "real_12_runs", self.cfg, log=lambda x: None))
        entry = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))["datasets"][0]
        self.assertTrue(entry["status"].startswith("success"), entry.get("error"))
        result = laspy.read(run / "datasets" / "D0001" / "classified.laz")
        self.assertEqual((str(result.header.version), result.header.point_format.id), ("1.2", 3))
        np.testing.assert_array_equal(np.asarray(result.classification)[np.asarray(result.object_instance) == 0],
                                      np.asarray(crop.classification)[np.asarray(result.object_instance) == 0])

    def test_labels_from_las_classification(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        las = laspy.convert(laspy.read(sample), point_format_id=7, file_version="1.4")
        codes = np.full(len(las.points), 40, dtype=np.uint8)
        codes[:7] = 64
        codes[7:9] = 2
        las.classification = codes
        path = self.root / "labels_as_las.laz"
        las.write(path)
        labels = core.load_labels(path, len(codes), {64}, source_xyz=np.asarray(las.xyz))
        np.testing.assert_array_equal(labels, np.where(codes == 64, 64, 0))
        with self.assertRaisesRegex(ValueError, "point count"):
            core.load_labels(path, len(codes) + 1, {64})

    def test_csv_labels_are_las_codes(self):
        path = self.root / "codes.csv"
        path.write_text("point_index,class_id\n0,64\n1,40\n2,\n3,-1\n4,0\n5,65\n", encoding="utf-8")
        np.testing.assert_array_equal(core.load_labels(path, 6, {64}), [64, 0, -1, -1, 0, 0])
        path.write_text("point_index,class_id\n0,256\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "0..255"):
            core.load_labels(path, 1, {64})

    def test_las_labels_require_matching_order_and_explicit_user_data(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        las = laspy.read(sample)
        xyz = np.asarray(las.xyz)
        las.classification = np.full(len(xyz), 2, dtype=np.uint8)
        las.user_data = np.full(len(xyz), 64, dtype=np.uint8)
        path = self.root / "legacy_labels.laz"
        las.write(path)
        self.assertFalse(core.load_labels(path, len(xyz), {64}, source_xyz=xyz).any())
        self.assertTrue((core.load_labels(path, len(xyz), {64}, source_xyz=xyz,
                                         label_field="user_data") == 64).all())
        with self.assertRaisesRegex(ValueError, "original point coordinates"):
            core.load_labels(path, len(xyz), {64})
        las.points = las.points[::-1].copy()
        las.write(path)
        with self.assertRaisesRegex(ValueError, "point order"):
            core.load_labels(path, len(xyz), {64}, source_xyz=xyz)

    def test_classified_output_can_be_reprocessed_without_losing_attributes(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        las = laspy.read(sample)
        n = len(las.points)
        las.user_data = np.arange(n, dtype=np.uint32).astype(np.uint8)
        las.add_extra_dim(laspy.ExtraBytesParams(name="survey_quality", type=np.float32))
        las.survey_quality = np.linspace(0, 1, n, dtype=np.float32)
        source = self.root / "preserve_source.laz"
        las.write(source)
        ids = np.zeros(n, dtype=np.uint32)
        ids[:10] = 1
        first = self.root / "preserve_first.laz"
        core.write_classified(source, first, [(64, np.arange(10), .9)], ids)
        result = laspy.read(first)
        for name in las.point_format.dimension_names:
            np.testing.assert_array_equal(result[name], las[name], err_msg=name)
        # Reprocessing replaces reserved prediction fields, including stale predictions.
        second = self.root / "preserve_second.laz"
        core.write_classified(first, second, [], np.zeros(n, dtype=np.uint32))
        result = laspy.read(second)
        self.assertFalse(np.asarray(result.object_class).any())
        self.assertFalse(np.asarray(result.object_instance).any())
        np.testing.assert_array_equal(result.user_data, las.user_data)
        np.testing.assert_array_equal(result.survey_quality, las.survey_quality)
        with self.assertRaises(FileExistsError):
            core.write_classified(source, source, [], np.zeros(n, dtype=np.uint32))
        # Exercise the full batch path, not only the writer.
        run = Path(process_batch([first], self.model, self.root / "rerun", self.cfg, log=lambda _: None))
        entry = json.loads((run / "run_metadata.json").read_text())["datasets"][0]
        self.assertTrue(entry["status"].startswith("success"), entry.get("error"))

    def test_export_rejects_field_collisions_and_inconsistent_instances(self):
        las = laspy.read(self.manifest.parent / "SYNTHETIC_test.laz")
        n = len(las.points)
        las.add_extra_dim(laspy.ExtraBytesParams(name="object_instance", type=np.float32))
        source = self.root / "bad_instance_type.laz"
        las.write(source)
        dest = self.root / "bad_export.laz"
        with self.assertRaisesRegex(ValueError, "incompatible type"):
            core.write_classified(source, dest, [], np.zeros(n, dtype=np.uint32))
        self.assertFalse(dest.exists())
        with self.assertRaisesRegex(ValueError, "does not match"):
            core.write_classified(source, dest, [(64, np.arange(5), .9)], np.zeros(n, dtype=np.uint32))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            core.write_classified(source, dest, [(64, np.arange(5), .9), (65, np.arange(3), .9)],
                                  np.zeros(n, dtype=np.uint32))

    def test_csv_binding_and_label_fingerprint_are_enforced(self):
        manifest = create_demo(self.root / "bound_labels")
        registered = load_manifest(manifest)
        record = next(r for r in registered["datasets"] if r["split"] == "test")
        sidecar = Path(record["labels"] + ".source.json")
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "binding"):
            load_manifest(manifest)
        data = json.loads(manifest.read_text())
        data["datasets"][2]["labels_source_sha256"] = record["sha256"]
        manifest.write_text(json.dumps(data))
        load_manifest(manifest)  # An explicitly verified manifest binding is supported.
        data["datasets"][2]["labels_source_sha256"] = "0" * 64
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "binding"):
            load_manifest(manifest)
        # Keep cloud bytes unchanged but alter labels after registration.
        with Path(record["labels"]).open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Labels changed"):
            core.collect(registered["datasets"], "test", self.cfg, registered["classes"], lambda _: None)

    def test_truth_rejects_bad_indices_and_source_binding(self):
        path = self.root / "truth_validation.json"
        target = {"object_id": "T1", "class_id": 64, "class_name": "target", "center": [0, 0, 0]}
        def write(objects, digest="a" * 64):
            path.write_text(json.dumps({"point_count": 10, "source_sha256": digest, "objects": objects}))
        bad = [{"point_range": [-1, 2]}, {"point_range": [0, 11]}, {"point_range": [4, 3]},
               {"point_range": [0.5, 2]}, {"point_indices": [-1]}, {"point_indices": [10]},
               {"point_indices": [1, 1]}, {"point_indices": [True]}, {"point_indices": [1.5]},
               {"point_indices": [[1]]}, {"point_range": [0, 2], "point_indices": [0]}]
        for indices in bad:
            with self.subTest(indices=indices):
                write([{**target, **indices}])
                with self.assertRaises(ValueError):
                    core.load_truth(path, 10, source_sha256="a" * 64)
        write([{**target, "point_range": [1, 4]}, {**target, "object_id": "T2", "point_indices": [3]}])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            core.load_truth(path, 10)
        write([{**target, "point_range": [0, 1]}, {**target, "point_range": [1, 2]}])
        with self.assertRaisesRegex(ValueError, "unique"):
            core.load_truth(path, 10)
        write([{**target, "point_range": [0, 0]}])  # A physically present but unsampled target is valid.
        _, ids = core.load_truth(path, 10, source_sha256="a" * 64)
        self.assertFalse(ids.any())
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            core.load_truth(path, 10, source_sha256="b" * 64)

    def test_manifest_classes_must_follow_asprs(self):
        source = json.loads(self.manifest.read_text())
        changed = self.manifest.parent / "classes.json"
        for classes, message in (({"0": "background", "64": "x"}, "remove"), ({"1": "x"}, "no class"),
                                 ({"0": "x"}, "remove"), ({"30": "x"}, "reserved"), ({}, "at least one")):
            changed.write_text(json.dumps({**source, "classes": classes}))
            with self.assertRaisesRegex(ValueError, message):
                load_manifest(changed)
        changed.write_text(json.dumps({**source, "classes": {"43": "submerged object", "200": "own class"}}))
        self.assertEqual(load_manifest(changed)["classes"],
                         {"0": "not a target", "43": "submerged object", "200": "own class"})

    def test_ground_truth_evaluation_in_batch(self):
        folder = self.root / "with_truth"
        folder.mkdir()
        sample = folder / "scene.laz"
        shutil.copy2(self.manifest.parent / "SYNTHETIC_test.laz", sample)
        count = len(read_cloud(sample, self.cfg)["xyz"])
        # create_demo writes 1800 background points, then two objects of 140 points.
        targets = [{"object_id": f"T{i}", "class_id": 64, "class_name": "synthetic_object",
                    "center": [0, 0, 0], "point_range": [1800 + 140 * (i - 1), 1800 + 140 * i]} for i in (1, 2)]
        targets.append({"object_id": "T3", "class_id": 64, "class_name": "synthetic_object",
                        "center": [0, 0, 0], "point_indices": [0, 1, 2]})
        core.truth_path(sample).write_text(json.dumps({"point_count": count, "source_sha256": core.sha256(sample),
                                                      "objects": targets}))
        run = Path(process_batch([sample], self.model, self.root / "truth_runs", self.cfg, log=lambda x: None))
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        result = metadata["datasets"][0]["evaluation"]
        self.assertEqual((result["true_positives"], result["missed"], result["false_positives"]), (2, 1, 0))
        objects = json.loads((run / "datasets" / "D0001" / "objects.json").read_text(encoding="utf-8"))
        self.assertEqual({o["truth"]["target_id"] for o in objects}, {"T1", "T2"})
        data = web_data(run / "datasets" / "D0001")
        self.assertEqual([t[0] for t in data["truth_objects"]], ["T1", "T2", "T3"])
        self.assertEqual(len(base64.b64decode(data["las"])), count)
        self.assertTrue((run / "datasets" / "D0001" / "classified.laz").exists())
        core.truth_path(sample).write_text(json.dumps({"point_count": count + 1, "objects": targets}))
        run = Path(process_batch([sample], self.model, self.root / "truth_runs", self.cfg, log=lambda x: None))
        status = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))["datasets"][0]
        self.assertEqual(status["status"], "failed")
        self.assertIn("different point cloud", status["error"])

    def test_synthetic_targets_on_a_flat_riverbed(self):
        import synthetic_targets
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.scales, header.offsets = [0.001] * 3, [700000, 6300000, 0]
        las = laspy.LasData(header)
        rng = np.random.default_rng(1)
        n = 40 * 40 * 25
        las.x, las.y = 700000 + rng.uniform(0, 40, n), 6300000 + rng.uniform(0, 40, n)
        las.z = rng.normal(80, 0.01, n)
        las.intensity = np.full(n, 1000, dtype=np.uint16)
        las.classification = np.full(n, 40, dtype=np.uint8)
        las.return_number = las.number_of_returns = np.ones(n, dtype=np.uint8)
        out, labels, targets = synthetic_targets.inject(las, np.ones(n, dtype=bool), rng, 6, "test")
        self.assertEqual(len(targets), 6)
        self.assertEqual(len(labels), len(out.points))
        for target in targets:
            start, stop = target["point_range"]
            self.assertTrue((labels[start:stop] == target["class_id"]).all())
            self.assertGreater(np.min(np.asarray(out.z)[start:stop]), 79.9)
            self.assertIn(target["class_name"], ("metal_plate", "metal_sphere"))
        self.assertEqual(int((labels > 0).sum()), sum(t["simulated_points"] for t in targets))
        self.assertLess(int((labels == 0).sum()), n + 6 * 30)  # The covered riverbed was removed.
        self.assertGreater(np.median(np.asarray(out.intensity)[labels > 0]), 1000)


if __name__ == "__main__":
    unittest.main()
