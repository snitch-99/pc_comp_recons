
"""
gui.py — Rock Reconstruction Pipeline GUI
==========================================
PyQt5 + PyVista (pyvistaqt) graphical interface for Mode B:
  Mode B: pre-separated rock files (.ply/.obj) → staged as Cluster_N.ply
  - Input pre-separated rock point clouds / meshes
  - Run Steps 1-5 individually with real-time log output
  - Overlay any combination of pipeline geometries in an embedded 3D viewport

Requirements:
    pip install PyQt5 pyvista pyvistaqt
"""

import ast
import glob
import os
import re
import sys

import numpy as np
import open3d as o3d
import pyvista as pv

from PyQt5.QtCore import (QProcess, QProcessEnvironment, QThread, Qt, pyqtSignal)
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSplitter, QSpinBox,
    QDoubleSpinBox, QStatusBar, QTextEdit,
    QVBoxLayout, QWidget, QProgressBar,
)
from pyvistaqt import QtInteractor

# ---------------------------------------------------------------------------
# Make sure our src/ directory is on the path
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Geometry registry: one entry per displayable object
# ---------------------------------------------------------------------------
GEOM_NONE   = "— none —"

GEOM_DEFS = [
    # (key,              label,                     color,          opacity)
    ("rock_mesh",       "Original rock mesh",       "#c8a882",      0.9),
    ("cluster_raw",     "Rock Cluster (raw)",       "#4da6ff",      1.0),
    ("sq_mesh",         "Superquadric mesh",        "#ff8c42",      0.55),
    ("poisson_mesh",    "Poisson mesh",             "#e84545",      0.9),
    ("dem_pts",         "DEM surface points",       "#4dcc66",      1.0),
    ("dem_wire",        "SQ DEM grid wireframe",    "#aaaaaa",      0.7),
    ("dem_recon",       "DEM reconstructed mesh",   "#a64dff",      0.9),
    ("scene",           "Full scene",               "#ffd966",      0.85),
]

GEOM_KEYS   = [g[0] for g in GEOM_DEFS]
GEOM_LABELS = {g[0]: g[1] for g in GEOM_DEFS}
GEOM_COLORS = {g[0]: g[2] for g in GEOM_DEFS}
GEOM_OPAC   = {g[0]: g[3] for g in GEOM_DEFS}


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Ubuntu, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 6px 14px;
    min-width: 80px;
}
QPushButton:hover  { background-color: #45475a; }
QPushButton:pressed{ background-color: #585b70; }
QPushButton:disabled { color: #6c7086; border-color: #313244; }
QPushButton#step_btn {
    background-color: #1e3a5f;
    border-color: #89b4fa;
    color: #89b4fa;
    font-weight: bold;
}
QPushButton#step_btn:hover   { background-color: #2a4f7f; }
QPushButton#step_btn:disabled{ background-color: #1a2035; color: #45475a; border-color: #45475a; }
QPushButton#run_btn {
    background-color: #1e4d2b;
    border-color: #a6e3a1;
    color: #a6e3a1;
    font-weight: bold;
    padding: 8px 20px;
}
QPushButton#run_btn:hover { background-color: #2a6b3b; }
QPushButton#run_all_btn {
    background-color: #2d1a4f;
    border-color: #cba6f7;
    color: #cba6f7;
    font-weight: bold;
    padding: 8px 14px;
}
QPushButton#run_all_btn:hover    { background-color: #3e2a6f; }
QPushButton#run_all_btn:disabled { background-color: #1a1a2e; color: #45475a; border-color: #45475a; }
QPushButton#stop_btn {
    background-color: #4d1a1a;
    border-color: #f38ba8;
    color: #f38ba8;
    font-weight: bold;
    padding: 8px 10px;
}
QPushButton#stop_btn:hover    { background-color: #6d2a2a; }
QPushButton#stop_btn:disabled { background-color: #1a1a2e; color: #45475a; border-color: #45475a; }
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 6px;
    color: #cdd6f4;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #89b4fa; }
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    border: 1px solid #313244;
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QListWidget {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
}
QListWidget::item { padding: 4px 8px; }
QListWidget::item:selected { background-color: #45475a; }
QScrollBar:vertical {
    background: #1e1e2e; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #45475a; background: #313244; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QProgressBar {
    border: 1px solid #45475a; border-radius: 4px;
    background: #313244; height: 10px; text-align: center;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 3px; }
QSplitter::handle { background: #45475a; }
QLabel#section_title { color: #89b4fa; font-weight: bold; font-size: 14px; }
"""


# ---------------------------------------------------------------------------
# Config editor helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(SRC_DIR, "config.py")

PARAM_DEFS = [
    # (config_key, label, widget_type, min, max, step/decimals)
    ("EMS_MAX_ITERS",         "EMS max iters",       "int",    1,   2000, 1),
    ("EMS_DOWNSAMPLE_N",      "EMS downsample N",    "int",    0, 500000, 1000),
    ("EMS_CIRCUM_TARGET",     "Circum. target",      "float",  0.5,  2.0, 2),
    ("POISSON_DEPTH",         "Poisson depth",       "int",    4,     14, 1),
    ("POISSON_NORMAL_KNN",    "Poisson kNN",         "int",    5,    200, 1),
    ("POISSON_DENSITY_QTLE",  "Density quantile",    "float",  0.0,  0.5, 3),
    ("DEM_W",                 "DEM width",           "int",    8,   2048, 1),
    ("DEM_H",                 "DEM height",          "int",    8,   1024, 1),
    ("DEM_RECON_POISSON_DEPTH","Recon Poisson depth","int",    4,     14, 1),
    ("DEM_RECON_NORMAL_KNN",  "Recon kNN",           "int",    5,    200, 1),
]


def read_config_value(key: str):
    """Read a single value from config.py by regex."""
    with open(CONFIG_PATH) as f:
        src = f.read()
    m = re.search(rf"^{key}\s*=\s*(.+)$", src, re.MULTILINE)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1).split("#")[0].strip())
    except Exception:
        return m.group(1).strip()


def write_config_value(key: str, value):
    """Overwrite a single assignment line in config.py."""
    with open(CONFIG_PATH) as f:
        src = f.read()
    new_src = re.sub(
        rf"^({key}\s*=\s*)(.+)$",
        lambda m: m.group(1) + repr(value),
        src, flags=re.MULTILINE,
    )
    with open(CONFIG_PATH, "w") as f:
        f.write(new_src)


# ---------------------------------------------------------------------------
# Open3D → PyVista conversion helpers
# ---------------------------------------------------------------------------

def o3d_pc_to_pv(pcd: o3d.geometry.PointCloud) -> pv.PolyData:
    pts = np.asarray(pcd.points)
    cloud = pv.PolyData(pts)
    if pcd.has_colors():
        cloud["colors"] = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return cloud


def o3d_mesh_to_pv(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    return pv.PolyData(verts, faces)


def o3d_lines_to_pv(ls: o3d.geometry.LineSet) -> pv.PolyData:
    pts   = np.asarray(ls.points)
    lines = np.asarray(ls.lines)
    cells = np.hstack([np.full((len(lines), 1), 2), lines]).ravel()
    return pv.PolyData(pts, lines=cells)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class RockReconGUI(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rock Reconstruction Pipeline")
        self.resize(1600, 920)

        # State
        self._input_files: list[str] = []
        self._map_dir: str = ""
        self._process: QProcess | None = None
        self._run_all_queue: list = []
        self._run_all_active: bool = False
        self._geom_actors: dict[str, list] = {}   # key → [actor, ...]
        self._geom_available: set[str] = set()
        self._geom_color_overrides: dict[str, str] = {}  # key → hex color

        # Log signal → slot (cross-thread safe)
        self.log_signal.connect(self._append_log)

        self._build_ui()
        self._apply_style()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ① Left: config panel
        splitter.addWidget(self._build_config_panel())

        # ② Centre: 3D viewport
        splitter.addWidget(self._build_viewport_panel())

        # ③ Right: log panel
        splitter.addWidget(self._build_log_panel())

        self._splitter = splitter
        self._splitter_sizes = [320, 860, 380]
        self._viewport_expanded = False
        splitter.setSizes(self._splitter_sizes)

        # Status bar + progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setFixedWidth(200)
        self.progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress)
        self.status_bar.showMessage("Ready")

    # ─── Config panel ────────────────────────────────────────────────────────

    def _build_config_panel(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Configuration")
        title.setObjectName("section_title")
        vbox.addWidget(title)

        # Input files
        grp_in = QGroupBox("Input Rock Files")
        g_in   = QVBoxLayout(grp_in)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(120)
        g_in.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("＋ Add files")
        btn_add.clicked.connect(self._add_input_files)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_input_files)
        btn_row.addWidget(btn_add); btn_row.addWidget(btn_clear)
        g_in.addLayout(btn_row)

        # MAP_DIR
        lbl_dir = QLabel("Output folder (MAP_DIR)")
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("io_op/Apollo_Rock1")
        self._dir_edit.setText(read_config_value("MAP_FOLDER") or "")
        btn_dir = QPushButton("Browse")
        btn_dir.clicked.connect(self._browse_map_dir)
        row_dir = QHBoxLayout()
        row_dir.addWidget(self._dir_edit, 1); row_dir.addWidget(btn_dir)
        g_in.addWidget(lbl_dir)
        g_in.addLayout(row_dir)

        vbox.addWidget(grp_in)

        # Params
        grp_p  = QGroupBox("Parameters")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        p_inner = QWidget()
        g_p = QVBoxLayout(p_inner)
        g_p.setSpacing(4)
        scroll.setWidget(p_inner)
        grp_p_layout = QVBoxLayout(grp_p)
        grp_p_layout.addWidget(scroll)

        self._param_widgets: dict[str, QWidget] = {}
        for key, label, wtype, lo, hi, step in PARAM_DEFS:
            row = QHBoxLayout()
            row.addWidget(QLabel(label), 1)
            cur = read_config_value(key)
            if wtype == "int":
                sb = QSpinBox()
                sb.setRange(lo, hi); sb.setSingleStep(step)
                sb.setValue(int(cur) if cur is not None else lo)
                row.addWidget(sb)
                self._param_widgets[key] = sb
            else:
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi); sb.setDecimals(step)
                sb.setSingleStep(10 ** (-step))
                sb.setValue(float(cur) if cur is not None else lo)
                row.addWidget(sb)
                self._param_widgets[key] = sb
            g_p.addLayout(row)

        # EMS circumscribe checkbox
        row_circ = QHBoxLayout()
        row_circ.addWidget(QLabel("Circumscribe SQ"), 1)
        self._circ_cb = QCheckBox()
        self._circ_cb.setChecked(bool(read_config_value("EMS_CIRCUMSCRIBE")))
        row_circ.addWidget(self._circ_cb)
        g_p.addLayout(row_circ)

        vbox.addWidget(grp_p, 1)

        btn_save = QPushButton("💾  Save Config")
        btn_save.clicked.connect(self._save_config)
        vbox.addWidget(btn_save)

        # Step buttons
        grp_steps = QGroupBox("Pipeline Steps")
        g_steps   = QVBoxLayout(grp_steps)

        step_defs = [
            ("1", "Step 1: SQ Fit",         "Step_1_sq_ems_fit.py"),
            ("2", "Step 2: Poisson",         "Step_2_poisson_recons.py"),
            ("3", "Step 3: DEM",             "Step_3_dem_generation.py"),
            ("4", "Step 4: Reconstruct",     "Step_4_reconstruction.py"),
            ("5", "Step 5: Scene",           "Step_5_scene_comparison.py"),
        ]
        self._step_btns: dict[str, QPushButton] = {}
        for sid, slabel, sscript in step_defs:
            btn = QPushButton(slabel)
            btn.setObjectName("step_btn")
            btn.clicked.connect(lambda checked, s=sscript, n=sid: self._run_step(s, n))
            g_steps.addWidget(btn)
            self._step_btns[sid] = btn

        # ── Run All / Stop ───────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #45475a;")
        g_steps.addWidget(sep)

        run_all_row = QHBoxLayout()
        self._btn_run_all = QPushButton("▶▶  Run All Steps")
        self._btn_run_all.setObjectName("run_all_btn")
        self._btn_run_all.setToolTip(
            "Run Steps 1 → 5 sequentially.\n"
            "The viewport auto-updates after each step so you can watch the pipeline build.")
        self._btn_run_all.clicked.connect(self._run_all_steps)
        run_all_row.addWidget(self._btn_run_all, 2)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("stop_btn")
        self._btn_stop.setToolTip("Cancel the run-all sequence")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_run_all)
        run_all_row.addWidget(self._btn_stop, 1)
        g_steps.addLayout(run_all_row)

        vbox.addWidget(grp_steps)

        # Compare button
        btn_compare = QPushButton("🔍  Compare Meshes")
        btn_compare.setToolTip("Compare two mesh files and visualise the error heatmap")
        btn_compare.clicked.connect(self._open_compare_dialog)
        vbox.addWidget(btn_compare)

        # Compression analysis button
        btn_compress = QPushButton("📊  Compression Analysis")
        btn_compress.setToolTip(
            "Show per-cluster file size: Poisson mesh vs DEM + SQ parameters")
        btn_compress.clicked.connect(self._open_compression_dialog)
        vbox.addWidget(btn_compress)

        return w

    # ─── Viewport panel ──────────────────────────────────────────────────────

    def _build_viewport_panel(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(4, 4, 4, 4)

        title = QLabel("3D Viewport")
        title.setObjectName("section_title")
        vbox.addWidget(title)

        # Search + checklist
        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("🔍  Search geometries...")
        self._search_edit.textChanged.connect(self._filter_geom_list)
        search_row.addWidget(self._search_edit)
        vbox.addLayout(search_row)

        # Geometry checklist
        self._geom_list = QListWidget()
        self._geom_list.setMaximumHeight(190)
        self._geom_list.setSelectionMode(QListWidget.NoSelection)
        self._geom_list.itemChanged.connect(self._on_geom_toggle)
        vbox.addWidget(self._geom_list)
        self._populate_geom_list()

        # Opacity + color row
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(5, 100)
        self._opacity_slider.setValue(90)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        op_row.addWidget(self._opacity_slider, 1)
        self._opacity_label = QLabel("90%")
        op_row.addWidget(self._opacity_label)
        btn_color = QPushButton("🎨")
        btn_color.setToolTip("Change color of selected geometry")
        btn_color.setFixedWidth(34)
        btn_color.clicked.connect(self._pick_color)
        op_row.addWidget(btn_color)
        vbox.addLayout(op_row)

        # Toolbar row
        tb_row = QHBoxLayout()
        btn_reset_cam = QPushButton("⟳ Reset camera")
        btn_reset_cam.clicked.connect(self._reset_camera)
        btn_clear_vp  = QPushButton("🗑 Clear viewport")
        btn_clear_vp.clicked.connect(self._clear_viewport)
        btn_grid = QPushButton("⊞ Toggle grid")
        btn_grid.clicked.connect(self._toggle_grid)
        tb_row.addWidget(btn_reset_cam); tb_row.addWidget(btn_grid)
        self._btn_expand = QPushButton("⛶ Expand")
        self._btn_expand.setToolTip("Expand viewport / restore layout")
        self._btn_expand.clicked.connect(self._toggle_viewport_expand)
        tb_row.addWidget(self._btn_expand)
        tb_row.addWidget(btn_clear_vp)
        vbox.addLayout(tb_row)

        # PyVista interactor
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self._grid_actor = None
        self._add_floor_grid()
        vbox.addWidget(self.plotter.interactor, 1)

        return w

    # ─── Log panel ───────────────────────────────────────────────────────────

    def _build_log_panel(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Output Log")
        title.setObjectName("section_title")
        vbox.addWidget(title)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        vbox.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        btn_clear_log = QPushButton("Clear log")
        btn_clear_log.clicked.connect(self._log.clear)
        btn_row.addWidget(btn_clear_log)
        vbox.addLayout(btn_row)
        return w

    # ─── Style ───────────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet(STYLE)

    # ── Geometry list ────────────────────────────────────────────────────────

    def _populate_geom_list(self):
        self._geom_list.blockSignals(True)
        self._geom_list.clear()
        query = self._search_edit.text().lower() if hasattr(self, "_search_edit") else ""
        for key, label, color, _ in GEOM_DEFS:
            if query and query not in label.lower():
                continue
            item = QListWidgetItem(f"  {label}")
            item.setData(Qt.UserRole, key)
            item.setCheckState(Qt.Checked if key in self._geom_actors and
                               self._geom_actors[key] else Qt.Unchecked)
            if key not in self._geom_available:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            r, g, b = _hex_to_rgb01(color)
            dot = "●"
            item.setText(f"{dot}  {label}")
            fg = QColor(int(r*255), int(g*255), int(b*255))
            item.setForeground(fg)
            self._geom_list.addItem(item)
        self._geom_list.blockSignals(False)

    def _filter_geom_list(self):
        self._populate_geom_list()

    def _on_geom_toggle(self, item: QListWidgetItem):
        key = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._show_geom(key)
        else:
            self._hide_geom(key)

    def _on_opacity_changed(self, val: int):
        self._opacity_label.setText(f"{val}%")

    # ── Viewport helpers ──────────────────────────────────────────────────────

    def _show_geom(self, key: str):
        if key in self._geom_actors and self._geom_actors[key]:
            return  # already shown
        self._load_and_add_geom(key)

    def _hide_geom(self, key: str):
        actors = self._geom_actors.pop(key, [])
        for actor in actors:
            self.plotter.remove_actor(actor)
        self.plotter.render()

    def _clear_viewport(self):
        for key in list(self._geom_actors.keys()):
            self._hide_geom(key)
        self._populate_geom_list()

    def _reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def _toggle_viewport_expand(self):
        """Expand the 3D viewport to fill the window, or restore the panel layout."""
        if not self._viewport_expanded:
            # Save current sizes then collapse left + right
            self._splitter_sizes = self._splitter.sizes()
            total = sum(self._splitter_sizes)
            self._splitter.setSizes([0, total, 0])
            self._btn_expand.setText("⋹ Restore")
            self._viewport_expanded = True
        else:
            self._splitter.setSizes(self._splitter_sizes)
            self._btn_expand.setText("⛶ Expand")
            self._viewport_expanded = False

    def _add_floor_grid(self, size: float = 5.0, divisions: int = 20):
        """Add an XZ-plane reference grid to the viewport."""
        grid = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 1, 0),   # normal points up (Y-up)
            i_size=size, j_size=size,
            i_resolution=divisions, j_resolution=divisions,
        )
        self._grid_actor = self.plotter.add_mesh(
            grid, style="wireframe",
            color="#cccccc", opacity=0.8,
            name="__floor_grid__",
        )

    def _toggle_grid(self):
        """Show or hide the floor grid."""
        if self._grid_actor is None:
            self._add_floor_grid()
        else:
            self.plotter.remove_actor(self._grid_actor)
            self._grid_actor = None
        self.plotter.render()

    def _pick_color(self):
        """Open a color picker for the currently selected (highlighted) geometry."""
        # Find a checked+enabled item in the geom list
        key = None
        for i in range(self._geom_list.count()):
            item = self._geom_list.item(i)
            if item.checkState() == Qt.Checked and (item.flags() & Qt.ItemIsEnabled):
                key = item.data(Qt.UserRole)
                break   # pick first checked geometry

        if key is None:
            QMessageBox.information(self, "Color picker",
                "Check (enable) at least one geometry first.")
            return

        # Default to the current color
        cur_hex = self._geom_color_overrides.get(key, GEOM_COLORS.get(key, "#ffffff"))
        initial = QColor(cur_hex)
        chosen  = QColorDialog.getColor(initial, self, f"Pick color for: {GEOM_LABELS[key]}")
        if not chosen.isValid():
            return

        new_hex = chosen.name()   # e.g. "#ff8c42"
        self._geom_color_overrides[key] = new_hex

        # Re-render: hide then show with new color
        self._hide_geom(key)
        self._show_geom(key)
        # Update the dot color in the list item
        self._populate_geom_list()

    def _load_and_add_geom(self, key: str):
        """Load the appropriate file from MAP_DIR and add to PyVista plotter."""
        map_dir  = self._map_dir or os.path.join(SRC_DIR, "..", "io_op",
                    read_config_value("MAP_FOLDER") or "")
        # Use override color if the user has changed it, else default
        color    = self._geom_color_overrides.get(key, GEOM_COLORS.get(key, "#ffffff"))
        opacity  = self._opacity_slider.value() / 100.0
        actors   = []

        def _add_mesh(pv_mesh, col, op, style="surface", **kw):
            a = self.plotter.add_mesh(pv_mesh, color=col, opacity=op,
                                      style=style, **kw)
            actors.append(a)

        def _add_pc(pv_pc, col, ps=3):
            a = self.plotter.add_points(pv_pc, color=col, point_size=ps,
                                        render_points_as_spheres=False)
            actors.append(a)

        try:
            if key == "cluster_raw":
                for f in sorted(glob.glob(os.path.join(map_dir, "Cluster_*.ply"))):
                    pcd = o3d.io.read_point_cloud(f)
                    if not pcd.is_empty():
                        _add_pc(o3d_pc_to_pv(pcd), color, ps=2)

            elif key == "sq_mesh":
                # Rebuild SQ meshes from txt files
                import importlib, Step_1_sq_ems_fit as s1
                from ems_core import Superquadric
                from utils import parse_sq_fit_txt
                for f in sorted(glob.glob(os.path.join(map_dir, "sq_fit_*.txt"))):
                    sq = parse_sq_fit_txt(f)
                    model  = Superquadric(sq["params"])
                    V, F   = model.sample_surface(nu=120, nv=60)
                    center = sq["center"]
                    R      = sq["R"]
                    V_w    = (V @ R.T) + center
                    pvmesh = pv.PolyData(V_w, np.hstack([np.full((len(F),1),3), F]).ravel())
                    _add_mesh(pvmesh, color, opacity, smooth_shading=True)

            elif key == "rock_mesh":
                # Prefer an .obj from the user's input file list; fall back to map_dir
                obj_path = next(
                    (fp for fp in self._input_files if fp.lower().endswith(".obj")),
                    os.path.join(map_dir, "rock.obj"),
                )
                if os.path.exists(obj_path):
                    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
                    if not mesh.is_empty():
                        _add_mesh(o3d_mesh_to_pv(mesh), color, opacity,
                                  smooth_shading=True)

            elif key == "poisson_mesh":
                for f in sorted(glob.glob(os.path.join(map_dir, "poisson_Cluster_*.ply"))):
                    mesh = o3d.io.read_triangle_mesh(f)
                    if not mesh.is_empty():
                        _add_mesh(o3d_mesh_to_pv(mesh), color, opacity,
                                  smooth_shading=True, show_edges=False)

            elif key == "dem_pts":
                for f in sorted(glob.glob(os.path.join(map_dir, "recon_pts_Cluster_*.ply"))):
                    pcd = o3d.io.read_point_cloud(f)
                    if not pcd.is_empty():
                        _add_pc(o3d_pc_to_pv(pcd), color, ps=3)

            elif key == "dem_wire":
                # Just show recon pts as proxy if no dedicated wireframe file
                for f in sorted(glob.glob(os.path.join(map_dir, "recon_pts_Cluster_*.ply"))):
                    pcd = o3d.io.read_point_cloud(f)
                    if not pcd.is_empty():
                        _add_pc(o3d_pc_to_pv(pcd), color, ps=1)

            elif key == "dem_recon":
                for f in sorted(glob.glob(os.path.join(map_dir, "dem_recon_Cluster_*.ply"))):
                    mesh = o3d.io.read_triangle_mesh(f)
                    if not mesh.is_empty():
                        _add_mesh(o3d_mesh_to_pv(mesh), color, opacity,
                                  smooth_shading=True)

            elif key == "scene":
                f = os.path.join(map_dir, "scene_reconstruction.ply")
                if os.path.exists(f):
                    mesh = o3d.io.read_triangle_mesh(f)
                    if not mesh.is_empty():
                        _add_mesh(o3d_mesh_to_pv(mesh), color, opacity,
                                  smooth_shading=True)

        except Exception as e:
            self._append_log(f"[VIEWPORT ERROR] {key}: {e}")
            return

        self._geom_actors[key] = actors
        if actors:
            self.plotter.reset_camera()
            self.plotter.render()

    # ── Config I/O ───────────────────────────────────────────────────────────

    def _add_input_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select rock files", "",
            "Point cloud / Mesh (*.ply *.obj *.stl *.off)"
        )
        for f in files:
            if f not in self._input_files:
                self._input_files.append(f)
                self._file_list.addItem(os.path.basename(f))
        self._geom_available.add("cluster_raw")
        if any(fp.lower().endswith(".obj") for fp in self._input_files):
            self._geom_available.add("rock_mesh")
        self._populate_geom_list()

    def _clear_input_files(self):
        self._input_files.clear()
        self._file_list.clear()

    def _browse_map_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._map_dir = d
            folder_name   = os.path.basename(d)
            self._dir_edit.setText(folder_name)

    def _save_config(self):
        # MAP_FOLDER — save full absolute path if available so config.py
        # resolves MAP_DIR correctly regardless of where the folder lives.
        folder = self._dir_edit.text().strip()
        if folder:
            if self._map_dir:
                write_config_value("MAP_FOLDER", self._map_dir)
            else:
                write_config_value("MAP_FOLDER", folder)
                self._map_dir = os.path.join(SRC_DIR, "..", "io_op", folder)

        # All spinbox params
        for key, widget in self._param_widgets.items():
            write_config_value(key, widget.value())

        # Circumscribe
        write_config_value("EMS_CIRCUMSCRIBE", self._circ_cb.isChecked())

        self.status_bar.showMessage("Config saved ✓", 3000)
        self._append_log("[GUI] config.py updated.")

    # ── Step runner ───────────────────────────────────────────────────────────

    def _run_step(self, script: str, step_id: str):
        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Busy",
                "Another step is already running. Please wait.")
            return

        self._save_config()

        # Copy input files to MAP_DIR as Cluster_N.ply (Mode B)
        if self._input_files and step_id == "1":
            self._stage_input_files()

        script_path = os.path.join(SRC_DIR, script)
        self._append_log(f"\n{'─'*50}\n▶  Running {script}\n{'─'*50}")

        for btn in self._step_btns.values():
            btn.setEnabled(False)
        self._btn_run_all.setEnabled(False)
        self.progress.setVisible(True)
        self.status_bar.showMessage(f"Running {script}…")

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        # Inherit the full system environment, then suppress blocking Open3D windows
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PC_HEADLESS", "1")
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(lambda code, _: self._on_step_done(code, step_id))
        self._process.start(sys.executable, [script_path])

    def _stage_input_files(self):
        """Copy/rename input files into MAP_DIR as Cluster_N.ply.
        
        Also clears SINGLE_MESH_INPUT in config.py so that Step 1 uses
        the staged Cluster_N.ply files rather than the original paths.
        """
        map_dir = self._map_dir
        if not map_dir:
            self._append_log("[WARN] MAP_DIR not set — files not staged.")
            return

        # Clear SINGLE_MESH_INPUT so Step 1 uses Cluster_*.ply (Mode B)
        write_config_value("SINGLE_MESH_INPUT", None)
        self._append_log("[GUI] SINGLE_MESH_INPUT cleared for Mode B.")

        os.makedirs(map_dir, exist_ok=True)
        for i, fp in enumerate(self._input_files, start=1):
            dest = os.path.join(map_dir, f"Cluster_{i}.ply")
            pcd  = None

            # Try mesh load first (handles .obj, .stl etc.)
            try:
                mesh = o3d.io.read_triangle_mesh(fp)
                if not mesh.is_empty() and len(mesh.triangles) > 0:
                    pcd = mesh.sample_points_uniformly(number_of_points=100_000)
                    self._append_log(f"[STAGE] Loaded as mesh: {os.path.basename(fp)}")
            except Exception:
                pass

            # Fallback: try trimesh (handles .obj without ASSIMP)
            if pcd is None:
                try:
                    import trimesh
                    tm = trimesh.load(fp, force='mesh')
                    pts = np.asarray(tm.sample(100_000))
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    self._append_log(f"[STAGE] Loaded via trimesh: {os.path.basename(fp)}")
                except Exception as e:
                    self._append_log(f"[STAGE] trimesh failed: {e}")

            # Fallback: direct point cloud read
            if pcd is None:
                pcd = o3d.io.read_point_cloud(fp)

            if pcd is None or pcd.is_empty():
                self._append_log(f"[STAGE ERROR] Could not load {os.path.basename(fp)} — skipping.")
                continue

            o3d.io.write_point_cloud(dest, pcd)
            self._append_log(f"[STAGE] {os.path.basename(fp)} → {dest}  ({len(pcd.points):,} pts)")

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.log_signal.emit(data)

    def _on_step_done(self, exit_code: int, step_id: str):
        self.progress.setVisible(False)
        for btn in self._step_btns.values():
            btn.setEnabled(True)

        if exit_code == 0:
            self._append_log(f"\n✓ Step {step_id} completed successfully.\n")
            self.status_bar.showMessage(f"Step {step_id} done ✓")
            self._mark_geoms_available(step_id)
            if self._run_all_active:
                self._auto_show_step_geoms(step_id)
                self._run_next_in_queue()
            else:
                self._btn_run_all.setEnabled(True)
        else:
            self._append_log(f"\n✗ Step {step_id} exited with code {exit_code}.\n")
            self.status_bar.showMessage(f"Step {step_id} failed  (exit {exit_code})")
            if self._run_all_active:
                self._run_all_active = False
                self._run_all_queue.clear()
                self._btn_stop.setEnabled(False)
                self._append_log("■  Run-all aborted due to step failure.\n")
            self._btn_run_all.setEnabled(True)

    def _mark_geoms_available(self, step_id: str):
        mapping = {
            "0": ["cluster_raw"],
            "1": ["cluster_raw", "sq_mesh"],
            "2": ["poisson_mesh"],
            "3": ["dem_pts", "dem_wire"],
            "4": ["dem_recon"],
            "5": ["scene"],
        }
        for key in mapping.get(step_id, []):
            self._geom_available.add(key)
        self._populate_geom_list()

    # ── Run-All orchestration ─────────────────────────────────────────────────

    def _run_all_steps(self):
        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Busy",
                "A step is already running. Please wait.")
            return

        self._save_config()
        if self._input_files:
            self._stage_input_files()

        self._run_all_queue = [
            ("Step_1_sq_ems_fit.py",      "1"),
            ("Step_2_poisson_recons.py",   "2"),
            ("Step_3_dem_generation.py",   "3"),
            ("Step_4_reconstruction.py",   "4"),
            ("Step_5_scene_comparison.py", "5"),
        ]
        self._run_all_active = True
        self._btn_run_all.setEnabled(False)
        self._btn_stop.setEnabled(True)

        # Start fresh: clear viewport so the build-up is visible step by step
        self._clear_viewport()
        # Show the raw cluster immediately (already staged)
        if "cluster_raw" in self._geom_available:
            self._show_geom("cluster_raw")

        self._append_log(
            f"\n{'═'*50}\n"
            f"▶▶  RUN ALL  (Steps 1 → 5)\n"
            f"{'═'*50}\n"
        )
        self._run_next_in_queue()

    def _run_next_in_queue(self):
        if not self._run_all_queue:
            self._run_all_active = False
            self._btn_run_all.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._append_log(
                f"\n{'═'*50}\n"
                f"✓✓  ALL STEPS COMPLETE\n"
                f"{'═'*50}\n"
            )
            self.status_bar.showMessage("All steps complete ✓")
            return
        script, sid = self._run_all_queue.pop(0)
        self._run_step(script, sid)

    def _stop_run_all(self):
        self._run_all_queue.clear()
        self._run_all_active = False
        self._btn_run_all.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._append_log("\n■  Run-all stopped by user.\n")
        self.status_bar.showMessage("Stopped.")

    def _auto_show_step_geoms(self, step_id: str):
        """After a step completes in run-all mode, auto-load its output geometry."""
        step_to_geoms = {
            "1": ["sq_mesh"],
            "2": ["poisson_mesh"],
            "3": ["dem_pts"],
            "4": ["dem_recon"],
            "5": ["scene"],
        }
        for key in step_to_geoms.get(step_id, []):
            if key in self._geom_available:
                self._show_geom(key)
        self._populate_geom_list()

    # ── Log ──────────────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        self._log.moveCursor(self._log.textCursor().End)
        self._log.insertPlainText(text)
        self._log.moveCursor(self._log.textCursor().End)

    # ── Mesh comparison ───────────────────────────────────────────────────────

    def _open_compare_dialog(self):
        # Reuse existing dialog so previous results are preserved
        if not hasattr(self, "_compare_dlg") or self._compare_dlg is None:
            self._compare_dlg = CompareMeshesDialog(
                self, self.plotter, self._append_log, self._geom_actors)
        self._compare_dlg.show()

    def _open_compression_dialog(self):
        # Reuse existing dialog
        if not hasattr(self, "_compression_dlg") or self._compression_dlg is None:
            self._compression_dlg = CompressionDialog(self, self._append_log)
        map_dir = self._map_dir or os.path.join(
            SRC_DIR, "..", "io_op",
            read_config_value("MAP_FOLDER") or "")
        self._compression_dlg.refresh(map_dir)
        self._compression_dlg.show()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
        self.plotter.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Mesh Comparison Dialog
# ---------------------------------------------------------------------------

class CompareMeshesDialog:
    """
    Modal dialog: pick two mesh files → compute geometric error → show heatmap.
    """

    SAMPLE_N = 50_000   # points sampled on the source mesh

    def __init__(self, parent, plotter, log_fn, geom_actors):
        from PyQt5.QtWidgets import QDialog
        self._dialog  = QDialog(parent)
        self._plotter = plotter
        self._log     = log_fn
        self._actors  = geom_actors
        self._distances: np.ndarray | None  = None
        self._sample_pts: np.ndarray | None = None
        self._normal_angles: np.ndarray | None = None
        self._heatmap_actor = None
        self._src_path = ""
        self._tgt_path = ""
        self._build_ui(parent)

    def exec_(self):
        """Show as a non-modal, minimizable window."""
        self._dialog.show()

    def show(self):
        """Bring the window to front (reuse after minimise)."""
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self, parent):
        from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem
        dlg = self._dialog
        dlg.setWindowTitle("Compare Mesh Models")
        dlg.resize(560, 580)
        # Qt.Window gives it its own taskbar entry and a minimise button
        dlg.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint
                           | Qt.WindowCloseButtonHint)
        dlg.setStyleSheet(parent.styleSheet())
        vbox = QVBoxLayout(dlg)

        # File pickers
        def _file_row(label_text, attr_name):
            grp = QGroupBox(label_text)
            row = QHBoxLayout(grp)
            edit = QLineEdit()
            edit.setPlaceholderText("Browse for a .ply / .obj mesh…")
            btn  = QPushButton("Browse")
            def _browse(checked=False, a=attr_name, e=edit):
                f, _ = QFileDialog.getOpenFileName(
                    dlg, f"Select mesh", "",
                    "Mesh files (*.ply *.obj *.stl *.off)")
                if f:
                    e.setText(f)
                    setattr(self, a, f)
            btn.clicked.connect(_browse)
            row.addWidget(edit, 1); row.addWidget(btn)
            vbox.addWidget(grp)
            return edit

        self._src_edit = _file_row("Source mesh  (reference)",     "_src_path")
        self._tgt_edit = _file_row("Target mesh  (reconstructed)", "_tgt_path")

        btn_run = QPushButton("⚡  Run Comparison")
        btn_run.setObjectName("run_btn")
        btn_run.clicked.connect(self._run_comparison)
        vbox.addWidget(btn_run)

        # Metrics table
        grp_m    = QGroupBox("Similarity Metrics")
        m_layout = QVBoxLayout(grp_m)

        method_note = QLabel(
            "<b>Methodology</b><br>"
            "<b>Bidirectional distance</b>: Sample N points on each surface. "
            "For every source point find the closest point on the target surface "
            "(RaycastingScene for meshes, cKDTree for point clouds), and vice versa. "
            "All 2N distances are pooled for the aggregate statistics.<br>"
            "<b>Hausdorff</b>: max over the full pooled set → worst-case error.<br>"
            "<b>Normal error</b>: angle between src & tgt normals at each point "
            "(0°=perfectly aligned, 90°=perpendicular). "
            "Mesh normals = Open3D area-weighted vertex normals → propagated to "
            "sample pts via triangle primitive_id. "
            "Point cloud normals = PCA with KNN=20."
        )
        method_note.setWordWrap(True)
        method_note.setStyleSheet("font-size: 10px; color: #6c7086;")
        m_layout.addWidget(method_note)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setMinimumHeight(220)
        m_layout.addWidget(self._table)
        vbox.addWidget(grp_m)

        # Text histogram
        self._hist_label = QLabel("Error distribution will appear here after comparison.")
        self._hist_label.setWordWrap(True)
        self._hist_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #a6e3a1;")
        vbox.addWidget(self._hist_label)

        # Heatmap controls
        self._btn_show = QPushButton("🌈  Show distance heatmap in viewport")
        self._btn_show.clicked.connect(self._show_heatmap)
        self._btn_show.setEnabled(False)
        vbox.addWidget(self._btn_show)

        self._btn_show_n = QPushButton("🟡  Show normal angle heatmap in viewport")
        self._btn_show_n.clicked.connect(self._show_heatmap_normals)
        self._btn_show_n.setEnabled(False)
        vbox.addWidget(self._btn_show_n)

        btn_clr = QPushButton("🗑  Remove heatmap from viewport")
        btn_clr.clicked.connect(self._clear_heatmap)
        vbox.addWidget(btn_clr)

        # ── Overlay both meshes with offset ──────────────────────────────────
        grp_ov = QGroupBox("Overlay both meshes in viewport")
        ov_layout = QVBoxLayout(grp_ov)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Target offset  X:"))
        self._ox = QDoubleSpinBox(); self._ox.setRange(-20, 20)
        self._ox.setSingleStep(0.01); self._ox.setDecimals(4); self._ox.setValue(0.0)
        offset_row.addWidget(self._ox)
        offset_row.addWidget(QLabel("Y:"))
        self._oy = QDoubleSpinBox(); self._oy.setRange(-20, 20)
        self._oy.setSingleStep(0.01); self._oy.setDecimals(4); self._oy.setValue(0.0)
        offset_row.addWidget(self._oy)
        offset_row.addWidget(QLabel("Z:"))
        self._oz = QDoubleSpinBox(); self._oz.setRange(-20, 20)
        self._oz.setSingleStep(0.01); self._oz.setDecimals(4); self._oz.setValue(0.0)
        offset_row.addWidget(self._oz)
        ov_layout.addLayout(offset_row)

        btn_overlay_row = QHBoxLayout()
        self._btn_overlay = QPushButton("🖼  Show both meshes")
        self._btn_overlay.clicked.connect(self._show_overlay)
        self._btn_overlay.setEnabled(False)
        btn_clr_ov = QPushButton("🗑  Remove meshes")
        btn_clr_ov.clicked.connect(self._clear_overlay)
        btn_overlay_row.addWidget(self._btn_overlay)
        btn_overlay_row.addWidget(btn_clr_ov)
        ov_layout.addLayout(btn_overlay_row)

        tip = QLabel("Source = grey  ·  Target = blue  ·  Set offset then click Show")
        tip.setStyleSheet("color: #6c7086; font-size: 11px;")
        ov_layout.addWidget(tip)

        vbox.addWidget(grp_ov)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        vbox.addWidget(btn_close)

    # ── Comparison ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_geometry(path):
        """Load file as TriangleMesh or PointCloud.
        Returns (points: ndarray N×3, normals: ndarray N×3 or None,
                 tri_norms: ndarray T×3 or None, rayscene or None, mode_str)
        """
        from scipy.spatial import cKDTree
        mesh = o3d.io.read_triangle_mesh(path)
        has_tris = not mesh.is_empty() and len(mesh.triangles) > 0

        if has_tris:
            # ── Mesh path ──────────────────────────────────────────────
            mesh.compute_vertex_normals()      # area-weighted per-vertex normals
            mesh.compute_triangle_normals()
            pcd = mesh.sample_points_uniformly(CompareMeshesDialog.SAMPLE_N)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
            pcd.normalize_normals()
            pts = np.asarray(pcd.points,   dtype=np.float32)
            nrm = np.asarray(pcd.normals,  dtype=np.float32)

            verts = np.asarray(mesh.vertices,  dtype=np.float32)
            tris  = np.asarray(mesh.triangles, dtype=np.int32)
            tri_n = np.asarray(mesh.triangle_normals, dtype=np.float32)
            tm = o3d.t.geometry.TriangleMesh()
            tm.vertex["positions"] = o3d.core.Tensor(
                verts, dtype=o3d.core.Dtype.Float32)
            tm.triangle["indices"] = o3d.core.Tensor(
                tris, dtype=o3d.core.Dtype.Int32)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tm)
            return pts, nrm, tri_n, scene, "mesh"
        else:
            # ── Point cloud fallback ───────────────────────────────────
            pcd = o3d.io.read_point_cloud(path)
            if pcd.is_empty():
                return None, None, None, None, "empty"
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
            pcd.normalize_normals()
            # Downsample to SAMPLE_N if larger
            pts_all = np.asarray(pcd.points, dtype=np.float32)
            nrm_all = np.asarray(pcd.normals, dtype=np.float32)
            if len(pts_all) > CompareMeshesDialog.SAMPLE_N:
                idx = np.random.choice(len(pts_all),
                                       CompareMeshesDialog.SAMPLE_N, replace=False)
                pts_all = pts_all[idx]
                nrm_all = nrm_all[idx]
            return pts_all, nrm_all, None, None, "pcd"

    @staticmethod
    def _query_distances(src_pts, tgt_scene_or_tree, is_scene: bool):
        """Return (dists, closest_pts) for src_pts against target."""
        from scipy.spatial import cKDTree
        if is_scene:
            res = tgt_scene_or_tree.compute_closest_points(
                o3d.core.Tensor(src_pts, dtype=o3d.core.Dtype.Float32))
            closest = res["points"].numpy()
            prim_ids = res["primitive_ids"].numpy()
            dists = np.linalg.norm(src_pts - closest, axis=1)
            return dists, closest, prim_ids
        else:
            dists, idx = tgt_scene_or_tree.query(src_pts)
            return dists.astype(np.float32), None, idx

    def _run_comparison(self):
        from PyQt5.QtWidgets import QTableWidgetItem
        from scipy.spatial import cKDTree

        src_p = self._src_edit.text().strip()
        tgt_p = self._tgt_edit.text().strip()
        if not src_p or not tgt_p:
            QMessageBox.warning(self._dialog, "Missing files",
                                "Please select both source and target meshes.")
            return

        self._log(f"\n[COMPARE] {os.path.basename(src_p)}  ←→  "
                  f"{os.path.basename(tgt_p)}\n")

        # ── Load both geometries ──────────────────────────────────────────────
        src_pts, src_nrm, _,       src_scene, src_mode = self._load_geometry(src_p)
        tgt_pts, tgt_nrm, tgt_tri_n, tgt_scene, tgt_mode = self._load_geometry(tgt_p)

        for label, pts, mode in [("Source", src_pts, src_mode),
                                  ("Target", tgt_pts, tgt_mode)]:
            if pts is None:
                QMessageBox.critical(self._dialog, "Load error",
                                     f"{label}: could not load file.")
                return
            self._log(f"[COMPARE] {label}: {len(pts):,} pts  ({mode})\n")

        tgt_is_scene = tgt_scene is not None
        src_is_scene = src_scene is not None
        tgt_ref = tgt_scene if tgt_is_scene else cKDTree(tgt_pts)
        src_ref = src_scene if src_is_scene else cKDTree(src_pts)

        # ── Bidirectional distances ───────────────────────────────────────────
        # Forward:  for each src point, closest point on tgt surface/cloud
        fwd_dists, _, tgt_prim_ids = self._query_distances(
            src_pts, tgt_ref, tgt_is_scene)
        # Backward: for each tgt point, closest point on src surface/cloud
        bwd_dists, _, _ = self._query_distances(
            tgt_pts, src_ref, src_is_scene)

        bidir = np.concatenate([fwd_dists, bwd_dists])  # combined distribution

        mean_bidir = float(bidir.mean())
        p50_bidir  = float(np.percentile(bidir, 50))
        p95_bidir  = float(np.percentile(bidir, 95))
        hausdorff  = float(bidir.max())          # max over both directions
        rmse       = float(np.sqrt((bidir**2).mean()))
        mean_fwd   = float(fwd_dists.mean())
        mean_bwd   = float(bwd_dists.mean())

        # ── Normal angle deviation ────────────────────────────────────────────
        # Normals computed as:
        #   Mesh:        Open3D area-weighted vertex normals, propagated to
        #                uniformly sampled surface points via nearest triangle
        #                (primitive_id from RaycastingScene)
        #   Point cloud: PCA-based normals with KNN=20 neighbourhood
        normal_angles_deg = None
        na_mean = na_med = na_p95 = na_max = float("nan")

        if src_nrm is not None and tgt_nrm is not None:
            # Source normals at sample pts (already estimated)
            s_n = src_nrm / (np.linalg.norm(src_nrm, axis=1, keepdims=True) + 1e-12)

            if tgt_is_scene and tgt_tri_n is not None and tgt_prim_ids is not None:
                # True surface normals from triangle index
                t_n = tgt_tri_n[tgt_prim_ids]
            else:
                # Nearest neighbour normals from tgt_nrm
                tree_n = cKDTree(tgt_pts)
                _, idx_n = tree_n.query(src_pts)
                t_n = tgt_nrm[idx_n]

            t_n = t_n / (np.linalg.norm(t_n, axis=1, keepdims=True) + 1e-12)
            dot = np.clip(np.abs(np.sum(s_n * t_n, axis=1)), 0.0, 1.0)
            normal_angles_deg = np.degrees(np.arccos(dot))
            na_mean = float(normal_angles_deg.mean())
            na_med  = float(np.percentile(normal_angles_deg, 50))
            na_p95  = float(np.percentile(normal_angles_deg, 95))
            na_max  = float(normal_angles_deg.max())

        self._distances     = fwd_dists   # used for fwd heatmap
        self._sample_pts    = src_pts
        self._normal_angles = normal_angles_deg

        # ── Metrics table ─────────────────────────────────────────────────────
        def _f(v, unit=""):
            return f"{v:.6f}{unit}" if not np.isnan(v) else "N/A"

        rows = [
            # Section header
            ("── Bidirectional mesh-to-mesh distance ──", ""),
            ("  Mean (bidirectional)",
             f"{mean_bidir:.6f}  [avg of all fwd+bwd]"),
            ("  Median (bidirectional)",
             f"{p50_bidir:.6f}"),
            ("  95th percentile (bidir)",
             f"{p95_bidir:.6f}  [95% of pts within]"),
            ("  Max = Hausdorff distance",
             f"{hausdorff:.6f}  [max over both dirs]"),
            ("  RMSE (bidirectional)",
             f"{rmse:.6f}"),
            ("  Mean src→tgt (forward)",
             f"{mean_fwd:.6f}"),
            ("  Mean tgt→src (backward)",
             f"{mean_bwd:.6f}"),
            ("── Normal angle error ──", ""),
            ("  Mean normal deviation",
             _f(na_mean, "°")),
            ("  Median normal deviation",
             _f(na_med, "°")),
            ("  95th pct normal deviation",
             _f(na_p95, "°")),
            ("  Max normal deviation",
             _f(na_max, "°")),
            ("── Info ──", ""),
            ("  Samples per side",    f"{len(src_pts):,} / {len(tgt_pts):,}"),
            ("  Source type",    src_mode),
            ("  Target type",    tgt_mode),
        ]
        self._table.setRowCount(len(rows))
        from PyQt5.QtGui import QFont
        bold = QFont(); bold.setBold(True)
        for r, (name, val) in enumerate(rows):
            n_item = QTableWidgetItem(name)
            v_item = QTableWidgetItem(val)
            if name.startswith("──"):
                n_item.setFont(bold)
            self._table.setItem(r, 0, n_item)
            self._table.setItem(r, 1, v_item)
        self._table.resizeColumnsToContents()

        # ── Histogram ────────────────────────────────────────────────────────
        icons = ["🔵", "🟦", "🟩", "🟨", "🟧", "🟥"]
        lines = [f"Bidirectional distance distribution  "
                 f"({len(bidir):,} values, ≤ p95 shown)"]
        bins = np.linspace(0, p95_bidir, 7)
        cnts, _ = np.histogram(bidir[bidir <= p95_bidir], bins=bins)
        bmax = max(int(cnts.max()), 1)
        for i in range(len(cnts)):
            bar = "█" * max(1, int(25 * cnts[i] / bmax))
            lines.append(f"{icons[i]} [{bins[i]:.4f}–{bins[i+1]:.4f}]  "
                         f"{bar}  {cnts[i]:,}")
        if normal_angles_deg is not None:
            lines.append("")
            lines.append("Normal angle distribution  (🔵 aligned → 🔴 divergent)")
            bins_n = np.linspace(0, na_p95, 7)
            cnts_n, _ = np.histogram(
                normal_angles_deg[normal_angles_deg <= na_p95], bins=bins_n)
            bmax_n = max(int(cnts_n.max()), 1)
            for i in range(len(cnts_n)):
                bar = "█" * max(1, int(25 * cnts_n[i] / bmax_n))
                lines.append(f"{icons[i]} [{bins_n[i]:.2f}°–{bins_n[i+1]:.2f}°]  "
                             f"{bar}  {cnts_n[i]:,}")
        self._hist_label.setText("\n".join(lines))

        self._log(
            f"[COMPARE] bidir mean={mean_bidir:.5f}  p95={p95_bidir:.5f}  "
            f"Hausdorff={hausdorff:.5f}\n"
            f"[COMPARE] normal mean={na_mean:.2f}°  p95={na_p95:.2f}°\n"
        )
        self._btn_show.setEnabled(True)
        self._btn_show_n.setEnabled(normal_angles_deg is not None)
        self._btn_overlay.setEnabled(True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _mesh_to_rayscene_tensor(mesh: o3d.geometry.TriangleMesh):
        """Build an o3d tensor TriangleMesh suitable for RaycastingScene.

        from_legacy() can raise IndexError on some Open3D builds because
        vertex positions are stored under an unexpected key. Building the
        tensor mesh directly from numpy arrays always works.
        """
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        tris  = np.asarray(mesh.triangles, dtype=np.int32)
        tm = o3d.t.geometry.TriangleMesh()
        tm.vertex["positions"]  = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
        tm.triangle["indices"] = o3d.core.Tensor(tris,  dtype=o3d.core.Dtype.Int32)
        return tm

    # ── Error heatmap ─────────────────────────────────────────────────────────

    @staticmethod
    def _colormap(t: np.ndarray) -> np.ndarray:
        """Blue→Cyan→Yellow→Red per-point colormap, t in [0,1]."""
        colors = np.zeros((len(t), 3), dtype=np.float32)
        m1 = t < 0.33;  s1 = t[m1] / 0.33
        colors[m1] = np.c_[np.zeros_like(s1), s1, np.ones_like(s1)]
        m2 = (t >= 0.33) & (t < 0.67); s2 = (t[m2] - 0.33) / 0.34
        colors[m2] = np.c_[s2, np.ones_like(s2), 1 - s2]
        m3 = t >= 0.67; s3 = (t[m3] - 0.67) / 0.33
        colors[m3] = np.c_[np.ones_like(s3), 1 - s3, np.zeros_like(s3)]
        return colors

    def _show_heatmap(self):
        if self._distances is None:
            return
        self._clear_heatmap()
        pts   = self._sample_pts
        dists = self._distances
        p95   = float(np.percentile(dists, 95))
        t     = np.clip(dists / (p95 + 1e-12), 0.0, 1.0)
        cloud = pv.PolyData(pts.astype(np.float64))
        cloud["error_rgb"] = (self._colormap(t) * 255).astype(np.uint8)
        self._heatmap_actor = self._plotter.add_points(
            cloud, scalars="error_rgb", rgb=True,
            point_size=4, render_points_as_spheres=False,
        )
        self._plotter.reset_camera()
        self._plotter.render()
        self._log("[COMPARE] Heatmap: 🔵 blue=close  🔴 red=far  "
                  "(colour range = 0 … 95th percentile)\n")

    def _show_heatmap_normals(self):
        """Show a heatmap of normal angle deviation (blue=aligned, red=divergent)."""
        if self._normal_angles is None:
            return
        self._clear_heatmap()
        pts    = self._sample_pts
        angles = self._normal_angles
        p95    = float(np.percentile(angles, 95))
        t      = np.clip(angles / (p95 + 1e-12), 0.0, 1.0)
        cloud  = pv.PolyData(pts.astype(np.float64))
        cloud["normal_rgb"] = (self._colormap(t) * 255).astype(np.uint8)
        self._heatmap_actor = self._plotter.add_points(
            cloud, scalars="normal_rgb", rgb=True,
            point_size=4, render_points_as_spheres=False,
        )
        self._plotter.reset_camera()
        self._plotter.render()
        self._log("[COMPARE] Normal heatmap: 🔵 blue=aligned  🔴 red=divergent  "
                  f"(range 0° … {p95:.1f}°)\n")

    def _clear_heatmap(self):
        if self._heatmap_actor is not None:
            self._plotter.remove_actor(self._heatmap_actor)
            self._heatmap_actor = None
            self._plotter.render()

    # ── Mesh overlay ────────────────────────────────────────────────────────

    def _show_overlay(self):
        """Load source and target meshes into viewport; translate target by XYZ offset."""
        src_p = self._src_edit.text().strip()
        tgt_p = self._tgt_edit.text().strip()
        if not src_p or not tgt_p:
            QMessageBox.warning(self._dialog, "Missing files",
                                "Browse for both meshes first.")
            return
        self._clear_overlay()

        offset = np.array([self._ox.value(), self._oy.value(), self._oz.value()])

        def _load_pv(path, off=np.zeros(3)):
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh.is_empty():
                return None
            verts = np.asarray(mesh.vertices) + off
            tris  = np.asarray(mesh.triangles)
            faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
            return pv.PolyData(verts, faces)

        src_pv = _load_pv(src_p)
        tgt_pv = _load_pv(tgt_p, off=offset)
        self._overlay_actors = []

        if src_pv is not None:
            a = self._plotter.add_mesh(src_pv, color="#aaaaaa", opacity=0.55,
                                       smooth_shading=True, label="Source")
            self._overlay_actors.append(a)
        if tgt_pv is not None:
            a = self._plotter.add_mesh(tgt_pv, color="#4da6ff", opacity=0.55,
                                       smooth_shading=True, label="Target")
            self._overlay_actors.append(a)

        if offset.any():
            self._log(f"[OVERLAY] Target offset: X={offset[0]:.4f}  "
                      f"Y={offset[1]:.4f}  Z={offset[2]:.4f}\n")
        else:
            self._log("[OVERLAY] Meshes superimposed (zero offset)\n")

        self._plotter.reset_camera()
        self._plotter.render()

    def _clear_overlay(self):
        actors = getattr(self, "_overlay_actors", [])
        for a in actors:
            self._plotter.remove_actor(a)
        self._overlay_actors = []
        self._plotter.render()




# ---------------------------------------------------------------------------
# Compression Analysis Dialog
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    """Human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class CompressionDialog:
    """Per-cluster file-size breakdown: Poisson vs SQ params + DEM + mask."""

    def __init__(self, parent, log_fn):
        self._log    = log_fn
        self._map_dir = ""
        self._dialog  = self._build(parent)

    def _build(self, parent):
        from PyQt5.QtWidgets import (QDialog, QTableWidget, QHeaderView)
        dlg = QDialog(parent)
        dlg.setWindowTitle("Compression Analysis")
        dlg.resize(700, 500)
        dlg.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint
                           | Qt.WindowCloseButtonHint)
        dlg.setStyleSheet(parent.styleSheet())
        vbox = QVBoxLayout(dlg)

        hdr_lbl = QLabel("Per-cluster file size:  Poisson mesh  vs  SQ params + DEM + mask")
        hdr_lbl.setObjectName("section_title")
        vbox.addWidget(hdr_lbl)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "Cluster", "Poisson .ply",
            "SQ .txt", "DEM .npy", "Mask .npy",
            "Total compressed", "Ratio ×",
        ])
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setStretchLastSection(True)
        vbox.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #a6e3a1;")
        vbox.addWidget(self._summary)

        self._chart = QLabel("")
        self._chart.setWordWrap(True)
        self._chart.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #cdd6f4;")
        vbox.addWidget(self._chart)

        btn_row = QHBoxLayout()
        btn_ref = QPushButton("🔄  Refresh")
        btn_ref.clicked.connect(lambda: self.refresh(self._map_dir))
        btn_cls = QPushButton("Close")
        btn_cls.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_ref); btn_row.addWidget(btn_cls)
        vbox.addLayout(btn_row)
        return dlg

    def show(self):
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def refresh(self, map_dir: str):
        from PyQt5.QtWidgets import QTableWidgetItem
        import re as _re
        self._map_dir = map_dir

        poisson_files = sorted(glob.glob(
            os.path.join(map_dir, "poisson_Cluster_*.ply")))

        if not poisson_files:
            self._summary.setText(
                f"No poisson_Cluster_*.ply files found in:\n{map_dir}\n"
                "Run Steps 1–4 first.")
            self._table.setRowCount(0)
            self._chart.setText("")
            return

        def _sz(path):
            return os.path.getsize(path) if os.path.exists(path) else 0

        rows_data = []
        for pf in poisson_files:
            m = _re.search(r"Cluster_(\d+)", pf)
            if not m:
                continue
            n = m.group(1)
            sz_poisson = _sz(pf)
            sz_sq  = _sz(os.path.join(map_dir, f"sq_fit_Cluster_{n}.txt"))
            sz_dem = _sz(os.path.join(map_dir, f"dem_Cluster_{n}.npy"))
            sz_msk = _sz(os.path.join(map_dir, f"mask_Cluster_{n}.npy"))
            sz_c   = sz_sq + sz_dem + sz_msk
            ratio  = (sz_poisson / sz_c) if sz_c > 0 else float("inf")
            rows_data.append((n, sz_poisson, sz_sq, sz_dem, sz_msk, sz_c, ratio))

        self._table.setRowCount(len(rows_data))
        for r, (n, sp, sq, sd, sm, sc, ratio) in enumerate(rows_data):
            self._table.setItem(r, 0, QTableWidgetItem(f"Cluster {n}"))
            self._table.setItem(r, 1, QTableWidgetItem(_fmt_bytes(sp)))
            self._table.setItem(r, 2, QTableWidgetItem(_fmt_bytes(sq)))
            self._table.setItem(r, 3, QTableWidgetItem(_fmt_bytes(sd)))
            self._table.setItem(r, 4, QTableWidgetItem(_fmt_bytes(sm)))
            self._table.setItem(r, 5, QTableWidgetItem(_fmt_bytes(sc)))
            self._table.setItem(r, 6, QTableWidgetItem(
                f"{ratio:.1f}×" if ratio != float("inf") else "N/A"))

        total_orig = sum(r[1] for r in rows_data)
        total_comp = sum(r[5] for r in rows_data)
        total_ratio = (total_orig / total_comp) if total_comp > 0 else 0
        savings = max(0.0, (1 - total_comp / total_orig) * 100) if total_orig > 0 else 0

        self._summary.setText(
            f"Total original  (Poisson):       {_fmt_bytes(total_orig)}\n"
            f"Total compressed (SQ+DEM+mask):  {_fmt_bytes(total_comp)}\n"
            f"Overall ratio: {total_ratio:.1f}×   ({savings:.1f}% size reduction)"
        )

        max_r = max((r[6] for r in rows_data if r[6] != float("inf")), default=1.0)
        lines = ["Compression ratio per cluster (longer bar = better compression):"]
        for n, _, _, _, _, _, ratio in rows_data:
            if ratio == float("inf"):
                lines.append(f"  Cluster {n}:  (no compressed data yet)")
            else:
                bar = "█" * max(1, int(30 * ratio / max_r))
                lines.append(f"  Cluster {n}:  {bar}  {ratio:.1f}×")
        self._chart.setText("\n".join(lines))

        self._log(f"[COMPRESS] {len(rows_data)} cluster(s): "
                  f"{_fmt_bytes(total_orig)} → {_fmt_bytes(total_comp)} "
                  f"({total_ratio:.1f}×)\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # High-DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    win = RockReconGUI()
    win.show()
    sys.exit(app.exec_())
