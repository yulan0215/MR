# Ratman project

Object Detection

A small offline desktop prototype for detecting and separating underwater objects in georeferenced Ratman project LAS/LAZ point clouds. The current implementation uses a Random Forest baseline, an optional MLP feature network, and per-class DBSCAN instance grouping.

This is a research prototype. Real Ratman project detection performance, waveform processing and Linux deployment have not yet been validated. Pointcept and SoftGroup are assessed as future options; neither is integrated.

## Software architecture

![English software architecture](docs/Ratman_Project_Software_Architecture_EN.png)

The architecture is fully editable in [diagrams.net](docs/Ratman_Project_Software_Architecture_EN.drawio). An [SVG version](docs/Ratman_Project_Software_Architecture_EN.svg) and [architecture notes](docs/Software_Architecture.md) are included.

## Installation

Python 3.11 is required. The interface uses PySide6 (Qt) with an embedded Qt WebEngine 3D view, installed from `requirements.txt` (about 300 MB download). Processing and the 3D view run fully offline afterwards.

On Windows, run `install_windows.cmd`, then `start_windows.cmd`.

On Linux, install Python 3.11 and venv (Qt WebEngine also needs the usual desktop libraries such as `libnss3` and `libxkbcommon`), then run:

```sh
sh install_linux.sh
sh start_linux.sh
```

A graphical desktop is required. The source supports both platforms, but only the Windows workflow has been checked in this delivery. This is a source package rather than a standalone EXE.

## Synthetic demonstration

From the project directory after installation, generate and train an explicitly synthetic demonstration:

```powershell
.venv\Scripts\python.exe create_demo.py synthetic_demo --train-and-run
```

On Linux, use `.venv/bin/python` instead. Start the application and click **Synthetic demo**. The button opens the saved demonstration result; it does not run training automatically. Use a new destination name when regenerating the data.

The demonstration contains no real Ratman project measurements. Its trained model and metrics provide no evidence of real-world underwater recognition performance.

## Metal targets demonstration (real scenes, simulated targets)

The intended targets are square metal plates and metal spheres under about 1 m, lying on or partly buried in a riverbed. No real target data exist yet, so `synthetic_targets.py` places simulated targets on the riverbed (LAS class 40) of real YellowScan Navigator scenes and records where they are:

```sh
.venv/bin/python synthetic_targets.py metal_demo --train-and-run
```

It needs the public sample files in `data/public/yellowscan_navigator/` (or the folder named by the `MR_PUBLIC_DATA` environment variable) and takes a few minutes. Three crops are written: Saint-Bauzille A (train, 70 targets), Saint-Bauzille B (validation, 30) and Devil's Bridge (test, 40), each as `<scene>.laz`, `<scene>.labels.csv` and `<scene>.truth.json`. Each target is sampled on the side the scanner sees, with the local riverbed point density, 2–3 cm range noise, a brighter but noisy intensity, return attributes copied from nearby bottom points, imperfect vendor classes (mostly 40, some 45 and 1), the riverbed underneath removed, and a few water-column returns above it. Click **Metal targets demo** to open the result.

On Devil's Bridge, the saved demonstration finds 24 of 40 targets with 3 false alarms, using threshold 0.6, cluster radius 0.3 m and at least 8 points per object. Detection depends mainly on how many laser points land on a target: 22 of 24 targets with at least 20 points were found, but only 2 of 16 with fewer. These numbers show that the workflow runs. Simulated targets are not evidence of real-world performance.

To try other grouping settings on a result, change Score threshold, Cluster radius or Min object points and press **Regroup**. It reuses the run's saved point predictions instead of classifying again (about 3 s instead of 12 s for Devil's Bridge) and saves a new run. On Devil's Bridge, threshold 0.5 finds 26 targets with 17 false alarms. Forest settings (depth, leaf size, tree count, class weights) were compared on the validation and test scenes and none did better than the defaults beyond the spread between random seeds.

## LAS classes

Input classes follow ASPRS LAS 1.4, including the topo-bathymetric codes YellowScan uses: 1 unassigned, 2 ground, 7 low noise, 40 bathymetric bottom, 41 water surface, 45 water column, and others. The model uses them as inputs (bottom, water surface, water column, ground, noise), together with each point's height above the local riverbed, estimated from class 40 points.

Every class number in the program is an ASPRS code; there is no separate numbering. A manifest lists only its target classes, for example `{"64": "metal_plate", "65": "metal_sphere"}`. A target class must be a defined ASPRS class other than 0 and 1 (for example 43 submerged object) or a user-definable code 64–255. Reserved codes (such as 8, 12, 23–39 and 46–63) are rejected. There is no background class: points that are not targets keep their own LAS class. The model uses 0 internally for "not a target", but never writes it to a file.

The metal demonstration uses **64 = metal plate** and **65 = metal sphere**. The standard code 43 means a submerged object without further detail, so it cannot separate plates from spheres.

Each processed file gets `classified.laz`. Source attributes, including User Data and unrelated extra dimensions, are preserved except for detected target classifications and the reserved prediction fields. `object_instance` (uint32) holds the current object number and `object_class` (uint8) holds the current predicted target class; both are 0 for points outside detected objects. Reprocessing an exported file replaces these two fields after checking their types. Incompatible existing fields are rejected, and the writer refuses to overwrite an existing output or the source file.

### LAS 1.2 and LAS 1.4

Both versions are read and processed; YellowScan delivers both (the Barsebäck harbour sample is LAS 1.2, point format 3; the river samples are LAS 1.4, point format 7). The differences that matter:

- LAS 1.2 (and 1.3) point formats 0–5 store only classes 0–31 and define no user-definable range or topo-bathy codes. The Barsebäck file marks all water as 9; the river files split water into 40, 41 and 45. Without class 40, the height-above-riverbed feature is unavailable and the model relies on the other features.
- The **Classified LAS** setting chooses the version of `classified.laz`. With **same as input** (default), the version and point format stay unchanged. A target code above 31 (such as 64) cannot fit legacy point formats 0–5, so it is stored in `object_class`; the classification field keeps its input value and User Data is untouched. With **1.4**, legacy formats convert to the matching LAS 1.4 format (0, 1 → 6; 2, 3 → 7; 4 → 9; 5 → 10), with the CRS stored as WKT, and target codes also go in the classification field. `run_metadata.json` records the encoding. `object_class` is present in both output modes and contains all current target predictions, including classes that fit the native classification field.
- LAS/LAZ labels use `classification` by default. Set a dataset entry's `"label_field": "object_class"` to explicitly use this program's new predictions as labels. For a verified legacy label file that deliberately stores target codes in User Data, set `"label_field": "user_data"`. User Data is never interpreted as labels automatically. Predictions used as labels are not independent ground truth.

Training labels are LAS codes too. In a `point_index,class_id` CSV, a target class marks a target point. Any other code (0, 2, 40, …) marks a point that is not a target, and a blank value or -1 means unlabelled. Labels can also be a LAS/LAZ file: mark target points with their target class (for example in CloudCompare) and set the manifest's `labels` to that file.

LAS/LAZ label files must have the same point count and the same XYZ coordinates in the same order as the source (absolute coordinate tolerance 1e-7 m, no relative tolerance). Files that reorder or change coordinates are rejected; restore the original order before training. Coordinate checks cannot distinguish records with identical XYZ, so annotation tools must preserve their original record order too.

CSV labels require a source binding: `<labels.csv>.source.json` containing `source_sha256` (SHA-256 of the original LAS/LAZ file bytes) and `point_count`, or `labels_source_sha256` in the manifest's dataset entry. Label templates and both demo generators create the sidecar automatically. For existing CSV labels, first verify their original source and point order, then record the binding; a checksum alone does not prove annotation correctness. Re-encoding a source changes its checksum and requires a verified new binding. Models register label-file hashes as well as source hashes; subsequent evaluation refuses changed labels. Older models without registered label hashes remain usable for inference when compatible, but must be retrained from verified labels before held-out evaluation.

## Ground-truth evaluation

When an input `scene.laz` has `scene.truth.json` next to it, processing compares the detected objects with the known targets and saves `evaluation.json`. A detection and a target are matched one-to-one, highest point overlap first, when they share at least a quarter of their combined points (point IoU ≥ 0.25). The window then shows how many targets were found, false alarms, wrong classes, mean IoU, centre error and the missed target IDs; the table gets a **Ground truth** column. In the 3D view, colour by **Truth** to see detected, false-alarm and missed points, or type a target ID such as `T004` in the find box. Colour by **LAS class** to see the input classes.

A truth file lists `source_sha256` (the input LAS/LAZ file's SHA-256), `point_count` and `objects`; each object has a unique nonempty `object_id`, a valid target `class_id`, `class_name`, a finite three-coordinate `center` and exactly one of `point_range` (`[start, stop)`) or `point_indices`. Indices must be integers within the source point range, with no repetitions or overlap between objects. Empty point sets are allowed for targets with no measured returns. Processing rejects missing or mismatched source bindings. Existing truth files need their source and point order verified before adding a binding; the target demo generator writes it automatically.

## Real data workflow

1. Export appropriately georeferenced and corrected LAS/LAZ data from CloudStation. The prototype does not process Ratman project RAW, reconstruct trajectories or perform underwater refraction correction.
2. Confirm that X, Y and Z are in metres. Supply the known projected CRS if it is absent from the LAS header. This setting records the CRS and does not reproject coordinates. Geographic or non-metric horizontal coordinates are rejected.
3. Prepare labels in `point_index,class_id` CSV files, or as a LAS/LAZ file (see [LAS classes](#las-classes)). Indices refer to the original zero-based point-record order. Class IDs are ASPRS LAS codes (see [LAS classes](#las-classes)): a manifest target class marks a target, any other code marks a non-target point, and blank or -1 means unlabelled. External annotation tools must preserve or recover this mapping.
4. Create a dataset manifest using `manifest_example.json`. Separate independent scenes or non-overlapping regions into train, validation and test sets. Use the same `scene_id` for overlapping data; the software checks duplicate paths, identical file content and cross-split scene IDs, but cannot prove that different files do not overlap spatially. Random-forest classes are weighted by inverse frequency unless the manifest sets `"balance_classes": false`.
5. Select the manifest in **Train and Evaluate**, choose a backend and training iterations, and save a new model. Every target class, and some non-target points, must appear in the training labels. Validation metrics and model settings are saved alongside the model. Test labels are used only when **Evaluate held-out test set** is clicked.
6. In **Process and View**, select a trusted trained model, one or more input files, and a results directory. Files run sequentially; a failed file does not stop the remaining batch.
7. The object table (left) lists every detected object; click a heading to sort. The 3D view (right) is part of the window. Click an object in the table to fly to it in 3D, or click a point in 3D to select its object in the table. **Open in browser** shows the same view in a web browser. Saved results can be reopened without rerunning inference or reading the original input file.

## Outputs and definitions

Each run creates `<first_input_stem>_<YYYYMMDD_HHMMSS>`, using the computer's local run-start time. Metadata records the timezone. Existing directories receive a unique suffix.

```text
run_directory/
  run_metadata.json
  objects/
    D0001_O00001.csv
  datasets/
    D0001/
      report.csv
      objects.json
      view3d.html
      classified.laz
      evaluation.json   (only with ground truth)
      status.json
```

Each dataset report preserves the proposed requirements header:

```text
object ID,x,y,z,Nr_of_points,Intensity,Area,confidence
```

- **Coordinates** are the arithmetic mean of all original points assigned to the exported instance. They are not necessarily the physical object's centre.
- **Nr_of_points** is the number of rows in that object's XYZ CSV.
- **Intensity** is the mean recorded raw LAS intensity. Scaling is not radiometric or water-attenuation correction. An all-zero input is flagged and retained as recorded.
- **Area** is blank by default (`unresolved`). An explicitly selected `convex_hull_xy` computes the horizontal convex-hull footprint in square metres. It fills concavities and holes and is not 3D surface area. The method needs project approval before acceptance.
- **Confidence** is the mean winning-class probability of the points assigned to the final instance. It is uncalibrated and is not measured detection accuracy.

Processing and CSV export use original points without downsampling. Each processed file also gets `view3d.html`, an offline page that draws up to 3,000,000 points with the GPU (WebGL); object points get at least half of that budget. It is shown inside the application (and can be opened in a browser). Drag to rotate, right-drag to pan, scroll to zoom, click a point to identify its object, or type an object ID to jump to it. Measured on Devil's Bridge: 3 M points render in about 3–5 ms per frame. The page is about 22 bytes per shown point, so up to about 66 MB. Display sampling does not change inference, object coordinates, reports or CSV exports. Results saved by earlier versions (with `view.npz`) still open; their 3D view shows the 30,000 cached points. Object IDs match across the report, object files and 3D view. Source mapping is retained through per-dataset reports and metadata. CSV is UTF-8 with a BOM; missing values use blank cells and JSON nulls.

## Scope and limits

By default there is no point limit (Max points / file = 0): every point of every file is processed. A positive value restores an explicit per-file and per-labelled-split cap. Processing streams each file instead of loading it whole. Points are first sorted into spatial tiles of at most about 1 M points, laid out where the points are (not over the bounding box), in a temporary folder inside the results directory (about 45 bytes per point on disk, deleted afterwards). Features and classes are then computed tile by tile. Each tile also reads the points around it; a point whose 20th neighbour is farther than that margin is done again with a wider one, so every point's neighbours and riverbed window are those of the whole file and the results are bit-for-bit identical to whole-file processing. Equally near neighbours are chosen by point number, and forest votes are added in a fixed order, so repeated runs also give identical results. Only candidate target points are then read back and clustered, over the whole file at once, so a tile edge never cuts an object. Memory grows by about 40 bytes per point plus the tile working set, so a 64 GB computer can process about 1 billion points in one file; a file is refused before decoding if that estimate exceeds 80% of RAM or the results disk lacks room for the temporary tiles. Training still loads each labelled file whole (about 600 bytes per point). Feature extraction, classification and clustering use all CPU cores. Clustering is a chunked DBSCAN whose memory is bounded by chunk size; core points and clusters match scikit-learn's DBSCAN, and each border point joins the cluster of its nearest core point. Measured on a 10-core, 64 GB Mac with a 100-tree random forest: 14.7 M points (Saint-Bauzille LAS) 27 s with 2.0 GB peak memory (whole-file processing: 27 s, 6.5 GB); 58.9 M points (four Saint-Bauzille copies side by side) 113 s with 3.1 GB (whole-file: 107 s, 19.2 GB).

The MLP is a three-hidden-layer network on engineered features (64/64/32), not an end-to-end PointNet or Pointcept model. Local height uses the neighbourhood Z 10th percentile, not a reconstructed seabed. DBSCAN can merge nearby same-class objects or split sparse objects. Each file is clustered globally, avoiding within-file tile seams; overlapping input files are not reconciled with each other.

Full-waveform samples are not decoded or used. The importer records some waveform fields, flags and same-stem `.wdp` existence; this does not establish that linked waveform samples are complete or decodable. No annotation editor, cancellation, checkpoint resume or production packaging is included. Load only trusted locally generated `.joblib` models.

## Requirements and method assessment

- [English requirements v0.2](preparation/Ratman_Project_Requirements.txt)
- [GitHub method assessment](docs/Model_Assessment.md)
- [English data notes](docs/Ratman_Project_Data_Notes.md)
- [Functional validation record](docs/Validation.txt)

The four supplied `preparation` documents are included in `preparation/` with project names updated to Ratman project. This includes the original Chinese source report and Chinese requirements translation. Client-facing software, current architecture and explanatory documentation are in English. The original supplied architecture is a legacy source document; use the current architecture in `docs/` for this implementation.

## Verification

```powershell
.venv\Scripts\python.exe -m unittest -v
```

Workflow tests cover LAZ reading, training and model reload, relocated model/data references, MLP availability, batch failure isolation, no-object results, IDs and exported original points, point counts, centroids, known-area geometry, split leakage checks and coordinate translation invariance. Label/export regressions also cover explicit legacy User Data opt-in, LAS point-order checks, CSV source bindings, changed label fingerprints, invalid/overlapping truth indices, attribute preservation and reprocessing classified outputs. These are functional checks, not a real-world detection benchmark.

The application window and heading are named **Object Detection**. The interface keeps labels short (format details are in hover tooltips); model limitations are documented here. The model uses scikit-learn RandomForestClassifier by default, or StandardScaler plus MLPClassifier (64/64/32), followed by per-class DBSCAN grouping. There is no pretrained real-survey model bundled.
