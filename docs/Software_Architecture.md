# Software architecture

The prototype runs as one local Python desktop process. A GUI thread handles interaction and a worker thread executes long-running tasks. There is no application server, database or cloud inference service.

## Components

| Component | Responsibility | Implementation |
|---|---|---|
| Desktop interface and queue | Training, model selection, single/batch jobs, saved-result viewing, settings and error log | PySide6 (Qt), worker thread and queued Qt signal in `app.py` |
| Import and validation | Read LAS/LAZ, check attributes, CRS, metric units, point limits and waveform indicators | laspy, lazrs and pyproj in `core.py` |
| Feature preparation | Preserve original point mapping; calculate local geometry, intensity, return and scan-angle features | NumPy, SciPy and cKDTree |
| Model management | Train RF or MLP, validate, save/reload schema and settings, predict point classes and scores | scikit-learn and joblib |
| Instance grouping and properties | Per-class global clustering, batch-unique IDs, centroids, counts, intensity, agreed area and confidence | DBSCAN, NumPy and SciPy |
| Export and visualisation | Per-object XYZ CSV, per-dataset reports, run metadata and reopenable 3D view | CSV, JSON and offline WebGL HTML |

## Data flow

LAS/LAZ import produces original double-precision coordinates and attributes. Feature calculations use translated local coordinates to avoid unnecessary sensitivity to large map offsets. Predictions retain the original record order. Per-class clustering produces final instance membership; all exported object properties are calculated from the original assigned coordinates.

The 3D view (`view3d.html`) subtracts the scene centre and samples up to 3,000,000 points, object points first, and is drawn by the GPU in an embedded Qt WebEngine view (or any browser). The display transformation and sampling do not change reports or CSV exports.

Training consumes labelled independent scenes. The manifest explicitly separates train, validation and test datasets. A model stores its class definitions, feature schema, neighbour count, preprocessing settings, provenance and validation results. Independent test evaluation is a separate user action.

## Model extension boundary

Pointcept and SoftGroup are future backend candidates; no adapter is implemented in this version. A future adapter should accept local coordinates, selected attribute channels and original point indices, then return per-original-point classes and scores or instance membership and instance scores.

Voxelisation must retain an explicit mapping back to original points. Tiled processing must reconcile overlapping predictions and instances before property calculation and export. Waveform decoding needs a validated source-specific decoder and point/pulse linkage. These additions can preserve the existing GUI and output contract.

## Requirements still requiring agreement

Target classes, annotation effort, final train/validation/test scenes, area definition, intensity and confidence treatment, output precision, supported platform versions, hardware, object-level quality metrics and acceptance thresholds remain project decisions. See the original v0.2 requirements.

Processing is CPU execution over spatial tiles streamed from each file, with results identical to whole-file processing. It does not provide learning-based instance separation, full-waveform integration, level-of-detail display of a whole survey or cross-file duplicate merging.

## Display memory update

The application is titled Object Detection. The desktop window shows a sortable object table beside the embedded 3D view; selecting in either one selects in the other. Reports and exports retain full-resolution detected points. Processing has no default point limit; files are streamed and processed in spatial tiles (memory about 40 bytes per point plus one tile), with a RAM and disk estimate checked before decoding and multi-core feature extraction, classification and clustering.
