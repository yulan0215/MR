"""PySide6 desktop UI with an embedded WebGL 3D view; long-running operations stay off the UI thread."""
from __future__ import annotations

import html
import json
import sys
import threading
import webbrowser
from pathlib import Path

import laspy
import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel, Qt, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QDoubleSpinBox, QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from core import (Settings, evaluate_model, inspect_file, label_template, process_batch, refresh_web_view, regroup_run,
                  train_model, write_web_view)


def size_text(count):
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024


class InspectDialog(QDialog):
    """Header summary of every input file; read-only, so it needs no settings."""
    HEADERS = ["File", "LAS version", "Points", "File size", "Width X m", "Width Y m", "Z min", "Z max", "CRS", "Memory needed",
               "Check"]

    def __init__(self, summaries, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input files")
        layout = QVBoxLayout(self)
        table = QTableWidget(len(summaries), len(self.HEADERS))
        table.setHorizontalHeaderLabels(self.HEADERS)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, info in enumerate(summaries):
            if "error" in info:
                cells = [info["name"]] + [""] * 9 + [info["error"]]
            else:
                (x0, y0, z0), (x1, y1, z1) = info["mins"], info["maxs"]
                memory = size_text(info["memory_bytes"])
                if info["ram_bytes"]:
                    memory += f" of {size_text(info['ram_bytes'])}"
                cells = [info["name"], f"{info['las_version']} (format {info['point_format']})",
                         f"{info['point_count']:,}", size_text(info["file_bytes"]),
                         f"{x1 - x0:,.2f}", f"{y1 - y0:,.2f}", f"{z0:,.2f}", f"{z1:,.2f}",
                         info["crs"] or "none", memory, "; ".join(info["issues"]) or "OK"]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 2 <= column <= 7:
                    item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                if "error" not in info:
                    item.setToolTip(f"{info['source']}\nLAS {info['las_version']}, point format "
                                    f"{info['point_format']}\nCRS unit: {info['horizontal_unit'] or 'unknown'}\n"
                                    f"Fields: {', '.join(info['fields'])}")
                table.setItem(row, column, item)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        frame = table.frameWidth() * 2
        width = table.verticalHeader().width() + table.horizontalHeader().length() + frame
        height = (table.horizontalHeader().height() + table.verticalHeader().length() + frame
                  + table.horizontalScrollBar().sizeHint().height())
        screen = self.screen().availableGeometry()
        table.setMinimumSize(min(width, screen.width() - 80), min(height, screen.height() // 2))
        layout.addWidget(table)
        note = QLabel("Read from the file headers only. Z units are not stored in LAS; "
                      "confirm metres in Settings before processing.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ObjectTable(QAbstractTableModel):
    """Detected objects; numbers stay numeric so column sorting is numeric."""
    HEADERS = ["Object ID", "Class", "Points", "Confidence", "X", "Y", "Z", "Ground truth"]
    TRUTH_TIP = ("Ground-truth target this object matches (point IoU = shared points / all points of both), "
                 "or false alarm when it matches none. Shown only for inputs with a .truth.json file.")

    def __init__(self):
        super().__init__()
        self.rows = []

    @staticmethod
    def truth_text(o):
        truth = o.get("truth")
        if truth is None:
            return ""
        if not truth["target_id"]:
            return "false alarm"
        return f"{truth['target_id']} · IoU {truth['iou']:.2f}" + ("" if truth["class_correct"] else " · wrong class")

    def load(self, objects):
        self.beginResetModel()
        self.rows = [(o["object_id"], f"{o['class_name']} ({o['class_id']})", o["point_count"], o["confidence"],
                      *o["centroid"], self.truth_text(o)) for o in objects]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        value = self.rows[index.row()][index.column()]
        column = index.column()
        if role == Qt.DisplayRole:
            if column == 2:
                return f"{value:,}"
            return f"{value:.3f}" if 3 <= column <= 6 else value
        if role == Qt.UserRole:  # Sort key.
            return value
        if role == Qt.ForegroundRole and column == 7 and value:
            return QColor("#c0392b") if value == "false alarm" or "wrong" in value else QColor("#1e8449")
        if role == Qt.TextAlignmentRole and 2 <= column <= 6:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal and section == 7:
            return self.TRUTH_TIP
        return None


class ObjectFilter(QSortFilterProxyModel):
    """Sorts by the numeric value and hides objects outside the class / size / confidence filter."""

    def __init__(self):
        super().__init__()
        self.class_name, self.min_points, self.min_confidence = "", 0, 0.0

    def set_filter(self, class_name, min_points, min_confidence):
        self.class_name, self.min_points, self.min_confidence = class_name, min_points, min_confidence
        self.invalidateFilter()

    def active(self):
        return bool(self.class_name or self.min_points or self.min_confidence)

    def filterAcceptsRow(self, row, parent):
        _, class_name, points, confidence, *_ = self.sourceModel().rows[row]
        return ((not self.class_name or class_name == self.class_name) and points >= self.min_points
                and confidence >= self.min_confidence)


class ViewerPage(QWebEnginePage):
    """Forwards 'PICK <object id>' console messages from the 3D page to the window."""

    def __init__(self, on_pick, parent=None):
        super().__init__(parent)
        self.on_pick = on_pick

    def javaScriptConsoleMessage(self, level, message, line, source):
        if message.startswith("PICK "):
            self.on_pick(message[5:])


class Bridge(QObject):
    """Carries worker-thread events to the UI thread (queued signal)."""
    event = Signal(str, object)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.resize(1400, 950)
        self.bridge = Bridge()
        self.bridge.event.connect(self.handle_event)
        self.busy = False
        self.paths = []
        self.run_path = None
        self.datasets = {}
        self.loaded_page = None
        self.pending_focus = None
        self.run_metadata = {}
        self.build_ui()

    # ---- layout -------------------------------------------------------------
    def build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        title = QLabel("Object Detection")
        title.setStyleSheet("font-size: 17pt; font-weight: bold;")
        layout.addWidget(title)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        tabs.addTab(self.build_processing(), "Process and View")
        tabs.addTab(self.build_training(), "Train and Evaluate")
        layout.addWidget(self.build_settings())
        progress_row = QHBoxLayout()
        self.status = QLabel("")
        progress_row.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setToolTip("Progress of the current task, by file and processing stage.")
        self.status.setToolTip("Current file and processing stage.")
        progress_row.addWidget(self.progress, 1)
        layout.addLayout(progress_row)
        self.log_box = QPlainTextEdit(readOnly=True)
        self.log_box.setFixedHeight(90)
        layout.addWidget(self.log_box)
        self.setCentralWidget(central)

    def path_row(self, grid, row, title, edit, command, tip):
        label = QLabel(title)
        grid.addWidget(label, row, 0)
        grid.addWidget(edit, row, 1)
        button = QPushButton("Browse")
        button.clicked.connect(command)
        grid.addWidget(button, row, 2)
        for widget in (label, edit):
            widget.setToolTip(tip)
        button.setToolTip("Browse for: " + tip)

    def button(self, text, command, tip=None):
        button = QPushButton(text)
        button.clicked.connect(command)
        if tip:
            button.setToolTip(tip)
        return button

    def build_processing(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.model_path = QLineEdit()
        self.output_path = QLineEdit(str((Path.cwd() / "results").resolve()))
        self.path_row(grid, 0, "Model", self.model_path, self.pick_model,
                      "Trained model file (.joblib) used to classify every point when processing.")
        self.path_row(grid, 1, "Results directory", self.output_path, self.pick_output,
                      "Folder for results; each run creates a new subfolder inside it.")
        layout.addLayout(grid)
        controls = QHBoxLayout()
        for text, command, tip in (
                ("Add LAS / LAZ", self.add_files, "Add LAS / LAZ point cloud files to process; you can select several at once."),
                ("Clear list", self.clear_files, "Empty the file list below. The files themselves are not deleted."),
                ("Inspect input", self.inspect, "Read only the file headers to quickly show point count, extent and CRS, and check whether each file can be processed."),
                ("Process", self.process, "Run the selected model on every listed file to detect and segment objects, saving results to the results directory."),
                ("Open saved results", self.open_run, "Open a results folder from an earlier run and show it in the table and 3D view."),
                ("Synthetic demo", self.open_demo, "Open the bundled synthetic example and its results to get familiar with the window."),
                ("Metal targets demo", self.open_metal_demo,
                 "Open real YellowScan river scenes with simulated metal plates and spheres on the riverbed, "
                 "the model trained on them and its results compared with the known targets.")):
            controls.addWidget(self.button(text, command, tip))
        controls.addStretch()
        layout.addLayout(controls)
        self.inputs = QListWidget()
        self.inputs.setFixedHeight(60)
        self.inputs.setToolTip("Point cloud files that Process will run on.")
        layout.addWidget(self.inputs)
        viewer_controls = QHBoxLayout()
        self.dataset = QComboBox()
        self.dataset.setMinimumWidth(380)
        self.dataset.setToolTip("Choose which file's results are shown in the table and 3D view.\n"
                                "Files marked \u2716 failed show their error instead.")
        self.dataset.currentIndexChanged.connect(lambda _: self.view())
        viewer_controls.addWidget(self.dataset)
        self.view_note = QLabel("")
        viewer_controls.addWidget(self.view_note)
        viewer_controls.addStretch()
        viewer_controls.addWidget(self.button("Open in browser", self.open_browser_view,
                                              "Open the same 3D view in your web browser, e.g. for a second screen."))
        layout.addLayout(viewer_controls)
        run_row = QHBoxLayout()
        self.run_info = QLabel("")
        self.run_info.setWordWrap(True)
        self.run_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.run_info.setToolTip("Settings used when these results were produced (not the current Settings).")
        run_row.addWidget(self.run_info, 1)
        self.reuse_button = self.button("Use these settings", self.reuse_run_settings,
                                        "Copy this run's settings and model into the current Settings, to process again the same way.")
        self.reuse_button.setVisible(False)
        run_row.addWidget(self.reuse_button)
        self.regroup_button = self.button(
            "Regroup", self.regroup,
            "Group this run's points into objects again with the current Score threshold, Cluster radius and "
            "Min object points.\nPoints are not classified again, so this takes a fraction of Process. "
            "The result is saved as a new run.")
        self.regroup_button.setVisible(False)
        run_row.addWidget(self.regroup_button)
        layout.addLayout(run_row)
        self.evaluation_note = QLabel("")
        self.evaluation_note.setWordWrap(True)
        self.evaluation_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.evaluation_note.setToolTip("Detected objects compared with the known (ground-truth) targets of this file.\n"
                                        "A detection counts when it shares at least a quarter of its points with a target "
                                        "(point IoU ≥ 0.25).\nType a missed target ID (e.g. T004) in the 3D view's "
                                        "Find box to see it; colour by Truth to see all of them.")
        self.evaluation_note.setVisible(False)
        layout.addWidget(self.evaluation_note)
        splitter = QSplitter(Qt.Horizontal)
        table_side = QWidget()
        table_layout = QVBoxLayout(table_side)
        table_layout.setContentsMargins(0, 0, 0, 0)
        filters = QHBoxLayout()
        self.class_filter = QComboBox()
        self.class_filter.addItem("All classes")
        self.min_points_filter = QSpinBox()
        self.min_points_filter.setRange(0, 1_000_000_000)
        self.min_points_filter.setSingleStep(10)
        self.min_points_filter.setPrefix("≥ ")
        self.min_points_filter.setSuffix(" pts")
        self.min_confidence_filter = QDoubleSpinBox()
        self.min_confidence_filter.setRange(0, 1)
        self.min_confidence_filter.setSingleStep(0.05)
        self.min_confidence_filter.setDecimals(2)
        self.min_confidence_filter.setPrefix("conf ≥ ")
        self.class_filter.setToolTip("Show only objects of this class.")
        self.min_points_filter.setToolTip("Show only objects with at least this many points.")
        self.min_confidence_filter.setToolTip("Show only objects with at least this mean confidence.")
        for widget in (self.class_filter, self.min_points_filter, self.min_confidence_filter):
            filters.addWidget(widget)
        self.class_filter.currentIndexChanged.connect(lambda _: self.apply_filter())
        self.min_points_filter.valueChanged.connect(lambda _: self.apply_filter())
        self.min_confidence_filter.valueChanged.connect(lambda _: self.apply_filter())
        filters.addWidget(self.button("Clear", self.clear_filter, "Remove all filters and show every object."))
        filters.addStretch()
        table_layout.addLayout(filters)
        self.objects = ObjectTable()
        self.sorted_objects = ObjectFilter()
        self.sorted_objects.setSourceModel(self.objects)
        self.sorted_objects.setSortRole(Qt.UserRole)
        self.table = QTableView()
        self.table.setModel(self.sorted_objects)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(-1, Qt.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setToolTip("Click an object to show it in the 3D view. Click a heading to sort.")
        self.table.selectionModel().currentRowChanged.connect(self.table_selected)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_side)
        self.web = QWebEngineView()
        # Keep a Python reference: Qt calls the overridden method, so the wrapper must outlive the view.
        self.viewer_page = ViewerPage(self.picked_in_view, self.web)
        self.web.setPage(self.viewer_page)
        self.web.loadFinished.connect(self.page_loaded)
        splitter.addWidget(self.web)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([380, 1000])
        layout.addWidget(splitter, 1)
        return page

    def build_training(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.manifest_path = QLineEdit()
        self.path_row(grid, 0, "Dataset manifest JSON", self.manifest_path, self.pick_manifest,
                      "Dataset manifest JSON listing the point clouds and label files for training, validation and testing.")
        layout.addLayout(grid)
        hint = QLabel("Labels: point_index,class_id as ASPRS LAS class codes (blank or -1 = unlabelled), "
                      "or a LAS/LAZ file")
        hint.setToolTip("Target classes use the manifest's codes: user-definable 64-255 or a defined ASPRS class "
                        "such as 43 (submerged object).\nAny other code (0, 2, 40, ...) means the point is not a target.\n"
                        "Train, validation and test must come from different scenes.")
        layout.addWidget(hint)
        row = QHBoxLayout()
        method_label = QLabel("Method")
        row.addWidget(method_label)
        self.backend = QComboBox()
        self.backend.addItems(["random_forest", "mlp"])
        for widget in (method_label, self.backend):
            widget.setToolTip("Training method: random_forest is a random forest (robust, fast to train); mlp is a small neural network.")
        row.addWidget(self.backend)
        iterations_label = QLabel("Trees")
        self.backend.currentTextChanged.connect(
            lambda text: iterations_label.setText("Trees" if text == "random_forest" else "Max epochs"))
        row.addWidget(iterations_label)
        self.iterations = QLineEdit("100")
        self.iterations.setFixedWidth(70)
        tip = ("Number of trees for random_forest, or maximum training epochs for mlp.\n"
               "Higher is usually more accurate but slower to train; lower is faster but may underfit.")
        iterations_label.setToolTip(tip)
        self.iterations.setToolTip(tip)
        row.addWidget(self.iterations)
        row.addStretch()
        layout.addLayout(row)
        buttons = QHBoxLayout()
        for text, command, tip in (
                ("Create label template", self.make_labels,
                 "Create a blank label CSV for a point cloud; fill in each point's class to use it for training."),
                ("Train and save model", self.train, "Train a model on the dataset manifest and save it as a .joblib file."),
                ("Evaluate on test set", self.evaluate, "Measure the accuracy of the Model above on the test set and save the results as JSON.")):
            buttons.addWidget(self.button(text, command, tip))
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        return page

    def build_settings(self):
        box = QGroupBox("Settings")
        grid = QGridLayout(box)
        self.crs, self.datum = QLineEdit(), QLineEdit("unspecified")
        self.meters = QCheckBox("X/Y/Z in metres")
        self.threshold, self.radius, self.minimum = QLineEdit("0.7"), QLineEdit("0.3"), QLineEdit("10")
        self.neighbors, self.limit = QLineEdit("20"), QLineEdit("0")

        self.area = QComboBox()
        self.area.addItems(["unresolved", "convex_hull_xy"])
        self.las_output = QComboBox()
        self.las_output.addItems(["same as input", "1.4"])
        fields = [[("Fallback CRS", self.crs), ("Vertical datum", self.datum), (None, self.meters)],
                  [("Score threshold", self.threshold), ("Cluster radius m", self.radius),
                   ("Min object points", self.minimum), ("Area method", self.area)],
                  [("Neighbours", self.neighbors), ("Max points / file", self.limit),
                   ("Classified LAS", self.las_output)]]
        tips = {
            self.crs: "Projected CRS to use when a file has none, e.g. EPSG:32632. Ignored when the file has its own CRS.",
            self.datum: "Name of the vertical datum. It is recorded in the results and does not affect calculations.",
            self.meters: "Confirm that X, Y and Z (including height) are all in metres. Processing is refused until this is ticked.",
            self.threshold: "Minimum confidence (0 to 1) for a point to count as part of an object.\n"
                            "Higher gives more reliable results but misses some objects; lower finds more objects but more false detections.",
            self.radius: "Search radius in metres for grouping object points into one object.\n"
                         "Higher merges neighbouring objects into one; lower splits one object into pieces.",
            self.minimum: "Fewest points an object may have; smaller groups are discarded.\n"
                          "Higher removes scattered noise but may drop small objects; lower keeps more small objects.",
            self.area: "How object area is computed: unresolved leaves Area blank; convex_hull_xy uses the area of the XY convex hull.",
            self.neighbors: "Neighbouring points (3 to 128) used to compute each point's local shape features during training.\n"
                            "Processing uses the model's training value. Higher is smoother but slower; lower is more detailed but more sensitive to noise.",
            self.las_output: "LAS version of classified.laz, the copy of each input with detected targets classified.\n"
                             "same as input: LAS 1.4 stays 1.4; LAS 1.2/1.3 stays 1.2/1.3. Those versions store only "
                             "classes 0-31, so a target code such as 64 is stored in the object_class extra field. User Data is preserved.\n"
                             "1.4: always LAS 1.4, where the class field holds every code (64, 65, ...).",
            self.limit: "Maximum points allowed per file; 0 means no limit and the whole file is processed.\n"
                        "A limit guards against processing huge files by mistake; files above it are refused.",
        }
        for row, items in enumerate(fields):
            for col, (title, widget) in enumerate(items):
                widget.setToolTip(tips[widget])
                if title:
                    label = QLabel(title)
                    label.setToolTip(tips[widget])
                    grid.addWidget(label, row, col * 2)
                grid.addWidget(widget, row, col * 2 + 1)
        grid.setColumnStretch(8, 1)
        return box

    # ---- tasks ----------------------------------------------------------------
    def settings(self):
        return Settings(neighbors=int(self.neighbors.text()), max_points=int(self.limit.text()),
                        threshold=float(self.threshold.text()), cluster_eps=float(self.radius.text()),
                        min_points=int(self.minimum.text()), area_method=self.area.currentText(),
                        crs_override=self.crs.text().strip(), coordinates_in_meters=self.meters.isChecked(),
                        vertical_datum=self.datum.text().strip(), las_output=self.las_output.currentText())

    def log(self, message):
        """Thread-safe: may be called from the worker thread."""
        self.bridge.event.emit("log", str(message))

    def job(self, fn, callback=None):
        if self.busy:
            QMessageBox.information(self, "Processing", "A task is already running.")
            return
        self.busy = True
        self.progress.setRange(0, 0)  # Busy indicator until the task reports progress.
        self.status.setText("Working…")
        def work():
            try:
                self.bridge.event.emit("done", (fn(), callback))
            except Exception as exc:
                self.bridge.event.emit("error", f"{type(exc).__name__}: {exc}")
        threading.Thread(target=work, daemon=True).start()

    def handle_event(self, kind, value):
        if kind == "log":
            self.log_box.appendPlainText(value)
            return
        if kind == "progress":
            number, count, name, stage, fraction = value
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(fraction * 1000))
            self.status.setText(f"File {number} / {count} · {stage} · {name}" if stage != "done" else "Finishing…")
            return
        self.busy = False
        self.progress.setRange(0, 1)
        self.status.setText("")
        if kind == "error":
            self.log_box.appendPlainText(value)
            QMessageBox.critical(self, "Task failed", value)
        else:
            result, callback = value
            self.log_box.appendPlainText(f"Completed: {result}")
            if callback:
                self.safe_action(lambda: callback(result))

    def safe_action(self, action):
        try:
            action()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Model", "", "Model (*.joblib)")
        if path:
            self.model_path.setText(path)

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "Results directory")
        if path:
            self.output_path.setText(path)

    def pick_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, "Dataset manifest", "", "Dataset manifest (*.json)")
        if path:
            self.manifest_path.setText(path)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Point clouds", "", "Point clouds (*.las *.laz)")
        for path in paths:
            if path not in self.paths:
                self.add_input(path)

    def add_input(self, path):
        """List a file with its LAS version and point format, read from the header only."""
        self.paths.append(path)
        try:
            with laspy.open(path) as reader:
                header = reader.header
                version, point_format, count = str(header.version), header.point_format.id, header.point_count
        except Exception as exc:  # Listed anyway; processing reports the problem for this file.
            self.inputs.addItem(f"{path}    ·    header unreadable: {exc}")
            return
        self.inputs.addItem(f"{path}    ·    LAS {version} · point format {point_format} · {count:,} points")
        self.inputs.item(self.inputs.count() - 1).setToolTip(
            f"{path}\nLAS {version}, point format {point_format}: "
            + ("class field holds codes 0-255" if point_format >= 6 else
               "class field holds codes 0-31 only (target codes above 31 go to object_class unless "
               "Classified LAS is set to 1.4)"))

    def clear_files(self):
        self.paths = []
        self.inputs.clear()

    def inspect(self):
        def action():
            if not self.paths:
                raise ValueError("Add a LAS / LAZ file first")
            summaries = []
            for path in self.paths:
                try:
                    summaries.append(inspect_file(path))
                except Exception as exc:
                    summaries.append({"name": Path(path).name, "error": f"{type(exc).__name__}: {exc}"})
            InspectDialog(summaries, self).exec()
        self.safe_action(action)

    def process(self):
        def action():
            cfg, paths = self.settings(), self.paths.copy()
            model, output = self.model_path.text(), self.output_path.text()
            if not model or not paths or not output:
                raise ValueError("Select input files, a model and a results directory")
            self.job(lambda: process_batch(paths, model, output, cfg, self.log, self.report_progress), self.load_run)
        self.safe_action(action)

    def regroup(self):
        def action():
            cfg, run = self.settings(), self.run_path
            output = self.output_path.text() or str(run.parent)
            self.job(lambda: regroup_run(run, output, cfg, self.log, self.report_progress), self.load_run)
        self.safe_action(action)

    def report_progress(self, *value):
        """Thread-safe: called from the worker thread."""
        self.bridge.event.emit("progress", value)

    def train(self):
        def action():
            cfg, manifest = self.settings(), self.manifest_path.text()
            backend, iterations = self.backend.currentText(), int(self.iterations.text())
            if not manifest:
                raise ValueError("Select a dataset manifest JSON")
            dest, _ = QFileDialog.getSaveFileName(self, "Trained model", "", "Trained model (*.joblib)")
            if dest:
                dest = dest if dest.endswith(".joblib") else dest + ".joblib"
                self.job(lambda: train_model(manifest, dest, cfg, backend, iterations, self.log),
                         self.model_path.setText)
        self.safe_action(action)

    def evaluate(self):
        def action():
            cfg, model = self.settings(), self.model_path.text()
            if not model:
                raise ValueError("Select a model")
            dest, _ = QFileDialog.getSaveFileName(self, "Evaluation", "", "Evaluation (*.json)")
            if dest:
                dest = dest if dest.endswith(".json") else dest + ".json"
                self.job(lambda: evaluate_model(model, dest, cfg, self.log))
        self.safe_action(action)

    def make_labels(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(self, "Point cloud", "", "Point cloud (*.las *.laz)")
            if not path:
                return
            cfg = self.settings()
            dest, _ = QFileDialog.getSaveFileName(self, "Label template", Path(path).stem + ".labels.csv",
                                                  "CSV (*.csv)")
            if dest:
                self.job(lambda: (label_template(path, dest, cfg), dest)[1])
        self.safe_action(action)

    # ---- results and 3D view -----------------------------------------------------
    def open_run(self):
        path = QFileDialog.getExistingDirectory(self, "Select a results directory containing run_metadata.json")
        if path:
            self.safe_action(lambda: self.load_run(path))

    def open_metal_demo(self):
        def action():
            demo = Path(__file__).resolve().parent / "metal_demo"
            candidates = sorted((demo / "results").glob("*/run_metadata.json"))
            if not candidates:
                raise ValueError("Metal targets demo missing; follow README to generate it.")
            self.model_path.setText(str(demo / "METAL_MODEL.joblib"))
            self.manifest_path.setText(str(demo / "manifest.json"))
            self.clear_files()
            self.add_input(str(demo / "DEVILS_BRIDGE.laz"))
            self.load_run(candidates[-1].parent)
            self.reuse_run_settings()
            self.log("Loaded metal targets demo (real scenes, simulated targets).")
        self.safe_action(action)

    def open_demo(self):
        def action():
            demo = Path(__file__).resolve().parent / "synthetic_demo"
            candidates = sorted((demo / "results").glob("*/run_metadata.json"))
            if not candidates:
                raise ValueError("Synthetic demo missing; follow README to generate it.")
            self.model_path.setText(str(demo / "SYNTHETIC_MODEL.joblib"))
            self.meters.setChecked(True)
            self.datum.setText("synthetic_local_z")
            self.area.setCurrentText("convex_hull_xy")
            self.clear_files()
            self.add_input(str(demo / "SYNTHETIC_test.laz"))
            self.load_run(candidates[-1].parent)
            self.log("Loaded synthetic demo.")
        self.safe_action(action)

    def load_run(self, path):
        self.run_path = Path(path)
        metadata = json.loads((self.run_path / "run_metadata.json").read_text(encoding="utf-8"))
        self.show_run_info(metadata)
        self.datasets = {e["dataset_id"]: e for e in metadata["datasets"]}
        failed = [e for e in metadata["datasets"] if e["status"] == "failed"]
        self.log(f"Loaded {len(self.datasets) - len(failed)} datasets, {len(failed)} failed")
        for e in failed:
            self.log(f"  {e['dataset_id']} {Path(e['source']).name} FAILED: {e.get('error', 'unknown error')}")
        self.loaded_page = None
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for key, e in self.datasets.items():
            failed_text = "  ✖ failed" if e["status"] == "failed" else ""
            version = e.get("input", {}).get("las_version")
            version_text = f" | LAS {version}" if version else ""
            self.dataset.addItem(f"{key} | {Path(e['source']).name}{version_text}{failed_text}")
            if failed_text:
                self.dataset.setItemData(self.dataset.count() - 1, e.get("error", ""), Qt.ToolTipRole)
        self.dataset.blockSignals(False)
        if self.datasets:
            self.view()
        else:
            self.objects.load([])
            self.show_evaluation(None)
            self.web.setHtml("")
            self.view_note.setText("No datasets to view.")

    def show_run_info(self, metadata):
        self.run_metadata = metadata
        cfg = metadata.get("settings", {})
        model = metadata.get("model_path")
        model_text = Path(model).name if model else f"model {metadata.get('model_id', '?')[:8]}"
        started = metadata.get("started_at", "")[:16].replace("T", " ")
        parts = [started, f"{model_text} ({metadata.get('backend', '?')})"]
        if cfg:
            parts += [f"threshold {cfg['threshold']}", f"cluster radius {cfg['cluster_eps']} m",
                      f"min points {cfg['min_points']}", f"neighbours {cfg['neighbors']}",
                      f"area {cfg['area_method']}"]
            if cfg.get("crs_override"):
                parts.append(f"fallback CRS {cfg['crs_override']}")
        if metadata.get("regrouped_from"):
            parts.append(f"regrouped from {Path(metadata['regrouped_from']).name}")
        self.run_info.setText("This run: " + " · ".join(parts))
        self.reuse_button.setVisible(bool(cfg))
        self.regroup_button.setVisible(bool(cfg))

    def reuse_run_settings(self):
        def action():
            cfg = self.run_metadata["settings"]
            self.threshold.setText(str(cfg["threshold"]))
            self.radius.setText(str(cfg["cluster_eps"]))
            self.minimum.setText(str(cfg["min_points"]))
            self.neighbors.setText(str(cfg["neighbors"]))
            self.limit.setText(str(cfg["max_points"]))
            self.area.setCurrentText(cfg["area_method"])
            self.las_output.setCurrentText(cfg.get("las_output", "same as input"))
            self.crs.setText(cfg["crs_override"])
            self.datum.setText(cfg["vertical_datum"])
            self.meters.setChecked(cfg["coordinates_in_meters"])
            model = self.run_metadata.get("model_path")
            if model and Path(model).exists():
                self.model_path.setText(model)
                self.log("Copied this run's settings and model.")
            else:
                self.log("Copied this run's settings; select its model manually "
                         f"(model ID {self.run_metadata.get('model_id', '?')}).")
        self.safe_action(action)

    def selected_folder(self):
        if not self.run_path or not self.dataset.currentText():
            raise ValueError("Process a dataset or open saved results first")
        key = self.dataset.currentText().split(" | ", 1)[0]
        if self.datasets[key]["status"] == "failed":
            raise ValueError(f"{key} failed: {self.datasets[key].get('error', 'unknown error')}")
        return key, self.run_path / self.datasets[key]["result_path"]

    def view_page(self, key, folder):
        page = folder / "view3d.html"
        if page.exists():
            refresh_web_view(page)  # Older results get the current viewer.
        else:  # Results saved before the 3D page existed: build it from the display cache.
            objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            with np.load(folder / "view.npz", allow_pickle=False) as data:
                write_web_view(page, Path(self.datasets[key]["source"]).name, data["xyz"], data["instance"],
                               objects, data["origin"], int(data["total_points"]))
        return page

    def view(self):
        def action():
            if not self.run_path or not self.dataset.currentText():
                return
            key = self.dataset.currentText().split(" | ", 1)[0]
            entry = self.datasets[key]
            if entry["status"] == "failed":
                self.objects.load([])
                self.reset_class_filter()
                self.show_evaluation(None)
                self.loaded_page = None
                self.web.setHtml("")
                error = entry.get("error", "unknown error")
                short = error if len(error) <= 90 else error[:87] + "…"
                self.view_note.setText(f"<span style='color:#c0392b'>Failed: {html.escape(short)}</span>")
                self.view_note.setToolTip(error)
                return
            folder = self.run_path / entry["result_path"]
            self.show_evaluation(entry.get("evaluation"))
            self.view_note.setToolTip("")
            self.objects.load(json.loads((folder / "objects.json").read_text(encoding="utf-8")))
            self.reset_class_filter()  # Also writes the point / object count note.
            page = self.view_page(key, folder)
            if page != self.loaded_page:
                self.loaded_page = page
                self.pending_focus = None
                # Loaded by URL: setHtml() is limited to 2 MB and the page can be tens of MB.
                self.web.load(QUrl.fromLocalFile(str(page.resolve())))
        self.safe_action(action)

    def show_evaluation(self, result):
        if not result:
            self.evaluation_note.setVisible(False)
            return
        def ratio(value):
            return "–" if value is None else f"{value:.2f}"
        missed = result["missed_targets"]
        by_class = ", ".join(f"{name} {ratio(value)}" for name, value in result["recall_by_class"].items())
        text = (f"<b>Ground truth:</b> {result['true_positives']} of {result['targets']} targets found "
                f"(recall {ratio(result['recall'])}; {by_class}) · {result['false_positives']} false alarms "
                f"(precision {ratio(result['precision'])}) · right class {result['class_correct']} of "
                f"{result['true_positives']} · mean IoU {ratio(result['mean_iou'])}")
        if result.get("recall_by_visible_points"):
            text += " · found by points on target: " + ", ".join(
                f"{html.escape(k)} pts {v}" for k, v in result["recall_by_visible_points"].items())
        if result["mean_centre_error_m"] is not None:
            text += f" · mean centre error {result['mean_centre_error_m']:.2f} m"
        if missed:
            text += f"<br>Missed: {', '.join(missed)}"
        self.evaluation_note.setText(text)
        self.evaluation_note.setVisible(True)

    def page_loaded(self, ok):
        if ok and self.sorted_objects.active():
            self.apply_filter()
        if ok and self.pending_focus:
            self.focus_in_view(self.pending_focus)

    def focus_in_view(self, object_id):
        self.pending_focus = object_id
        self.web.page().runJavaScript(f"window.focusObject && focusObject({json.dumps(object_id)})")

    def table_selected(self, current, previous):
        if current.isValid():
            source = self.sorted_objects.mapToSource(current)
            self.focus_in_view(self.objects.rows[source.row()][0])

    def picked_in_view(self, object_id):
        """A point clicked in the 3D view selects its object in the table (or clears the selection)."""
        self.table.selectionModel().blockSignals(True)
        self.table.clearSelection()
        for row, values in enumerate(self.objects.rows):
            if values[0] == object_id:
                index = self.sorted_objects.mapFromSource(self.objects.index(row, 0))
                self.table.selectRow(index.row())
                self.table.scrollTo(index)
                break
        self.table.selectionModel().blockSignals(False)

    def reset_class_filter(self):
        """Offer the classes of the loaded objects; keep the chosen class if it is still there."""
        current = self.class_filter.currentText()
        names = sorted({row[1] for row in self.objects.rows})
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItems(["All classes", *names])
        self.class_filter.setCurrentText(current if current in names else "All classes")
        self.class_filter.blockSignals(False)
        self.apply_filter()

    def clear_filter(self):
        for widget in (self.class_filter, self.min_points_filter, self.min_confidence_filter):
            widget.blockSignals(True)
        self.class_filter.setCurrentIndex(0)
        self.min_points_filter.setValue(0)
        self.min_confidence_filter.setValue(0)
        for widget in (self.class_filter, self.min_points_filter, self.min_confidence_filter):
            widget.blockSignals(False)
        self.apply_filter()

    def apply_filter(self):
        class_name = "" if self.class_filter.currentIndex() <= 0 else self.class_filter.currentText()
        self.sorted_objects.set_filter(class_name, self.min_points_filter.value(), self.min_confidence_filter.value())
        total, shown = self.objects.rowCount(), self.sorted_objects.rowCount()
        if self.sorted_objects.active():
            ids = [self.objects.rows[self.sorted_objects.mapToSource(self.sorted_objects.index(r, 0)).row()][0]
                   for r in range(shown)]
            script = f"window.setVisibleObjects && setVisibleObjects({json.dumps(ids)})"
        else:
            script = "window.setVisibleObjects && setVisibleObjects(null)"
        self.web.page().runJavaScript(script)
        if self.run_path and self.dataset.currentText():
            key = self.dataset.currentText().split(" | ", 1)[0]
            if self.datasets.get(key, {}).get("status") != "failed":
                points = self.datasets[key].get("input", {}).get("point_count")
                prefix = f"{points:,} points · " if points else ""
                shown_text = f"{shown:,} of {total:,}" if self.sorted_objects.active() else f"{total:,}"
                self.view_note.setText(f"{prefix}{shown_text} objects")

    def open_browser_view(self):
        def action():
            key, folder = self.selected_folder()
            webbrowser.open(self.view_page(key, folder).resolve().as_uri())
        self.safe_action(action)

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.information(self, "Task running", "Wait for the current task to finish.")
            event.ignore()
            return
        event.accept()


def main():
    application = QApplication(sys.argv)
    window = App()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
