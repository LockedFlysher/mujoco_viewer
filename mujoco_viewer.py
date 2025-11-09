import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QSlider, QLineEdit, QFileDialog, QScrollArea,
                             QGroupBox, QSplitter, QMessageBox, QSpinBox,
                             QDoubleSpinBox, QFrame, QCheckBox, QComboBox,
                             QTabWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPainter
import mujoco
import mujoco.viewer
import threading
import time


def euler_to_quaternion(rpy):
    """Convert Euler angle sequence to quaternion."""
    # Assuming rpy is in radians, order is 'xyz' (roll, pitch, yaw)
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # MuJoCo quaternion order is (w, x, y, z)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def quaternion_to_euler(quat):
    """Convert quaternion to Euler angle sequence."""
    # Assuming quat order is (w, x, y, z)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # If out of range, use 90 degrees
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class MujocoWidget(QWidget):
    """A widget to display a MuJoCo scene and handle mouse interaction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.data = None
        self.renderer = None
        self.camera = None

        self._last_mouse_pos = None
        self._mouse_button = None

        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

    def set_model(self, model, data):
        """Set the MuJoCo model and data to render."""
        self.model = model
        self.data = data
        if self.model:
            self.renderer = mujoco.Renderer(self.model)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            # Custom camera placement
            self.camera.azimuth = 90
            self.camera.elevation = -20
            self.camera.distance = 5.0
        else:
            self.renderer = None
            self.camera = None
        self.update()

    def paintEvent(self, event):
        """Render the scene."""
        if self.renderer and self.data:
            self.renderer.update_scene(self.data, self.camera)
            pixels = self.renderer.render()
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], QImage.Format_RGB888)
            painter = QPainter(self)
            painter.drawImage(self.rect(), image)
        else:
            # Draw a placeholder text if no model is loaded
            painter = QPainter(self)
            painter.setPen(Qt.white)
            painter.fillRect(self.rect(), Qt.black)
            painter.drawText(self.rect(), Qt.AlignCenter, "Load a model here to begin")


    def wheelEvent(self, event):
        """Handle mouse wheel zoom."""
        if not self.camera:
            return
        scroll = event.angleDelta().y() / 120.0
        scale = 1.1 ** (-scroll)
        self.camera.distance *= scale
        self.camera.distance = max(0.1, self.camera.distance)  # Prevent zooming too close
        self.update()

    def mousePressEvent(self, event):
        self._last_mouse_pos = event.pos()
        self._mouse_button = event.button()

    def mouseReleaseEvent(self, event):
        self._last_mouse_pos = None
        self._mouse_button = None

    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera control."""
        if not self.camera or not self._last_mouse_pos:
            return

        dx = event.x() - self._last_mouse_pos.x()
        dy = event.y() - self._last_mouse_pos.y()

        if self._mouse_button == Qt.LeftButton:  # Rotate
            self.camera.azimuth -= dx * 0.25
            self.camera.elevation -= dy * 0.25
            self.camera.elevation = max(-89.0, min(89.0, self.camera.elevation))
        elif self._mouse_button == Qt.RightButton:  # Pan
            az_rad = np.deg2rad(self.camera.azimuth)
            el_rad = np.deg2rad(self.camera.elevation)
            
            right = np.array([-np.sin(az_rad), np.cos(az_rad), 0])
            up = np.array([-np.cos(az_rad) * np.sin(el_rad),
                           -np.sin(az_rad) * np.sin(el_rad),
                           np.cos(el_rad)])
            
            move_speed = 0.002 * self.camera.distance
            self.camera.lookat -= move_speed * dx * right
            self.camera.lookat += move_speed * dy * up

        self._last_mouse_pos = event.pos()
        self.update()


class BaseControlWidget(QWidget):
    """Base control widget (XYZ + RPY)"""
    valueChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        # Labels
        layout.addWidget(QLabel("Pos X:"), 0, 0)
        layout.addWidget(QLabel("Pos Y:"), 0, 2)
        layout.addWidget(QLabel("Pos Z:"), 0, 4)
        layout.addWidget(QLabel("Roll (rad):"), 1, 0)
        layout.addWidget(QLabel("Pitch (rad):"), 1, 2)
        layout.addWidget(QLabel("Yaw (rad):"), 1, 4)

        # SpinBoxes
        self.pos_x = self._create_spinbox(layout, 0, 1)
        self.pos_y = self._create_spinbox(layout, 0, 3)
        self.pos_z = self._create_spinbox(layout, 0, 5)

        self.rot_r = self._create_spinbox(layout, 1, 1, is_angle=True)
        self.rot_p = self._create_spinbox(layout, 1, 3, is_angle=True)
        self.rot_y = self._create_spinbox(layout, 1, 5, is_angle=True)

    def _create_spinbox(self, layout, row, col, is_angle=False):
        spinbox = QDoubleSpinBox()
        if is_angle:
            spinbox.setRange(-np.pi, np.pi)
            spinbox.setSingleStep(0.05)
        else:
            spinbox.setRange(-5.0, 5.0)
            spinbox.setSingleStep(0.01)
        spinbox.setDecimals(3)
        spinbox.setMinimumWidth(80)
        spinbox.valueChanged.connect(self.valueChanged.emit)
        layout.addWidget(spinbox, row, col)
        return spinbox

    def set_values(self, pos, rpy):
        self.block_all_signals(True)
        self.pos_x.setValue(pos[0])
        self.pos_y.setValue(pos[1])
        self.pos_z.setValue(pos[2])
        self.rot_r.setValue(rpy[0])
        self.rot_p.setValue(rpy[1])
        self.rot_y.setValue(rpy[2])
        self.block_all_signals(False)

    def get_values(self):
        pos = np.array([
            self.pos_x.value(),
            self.pos_y.value(),
            self.pos_z.value()
        ])
        rpy = np.array([
            self.rot_r.value(),
            self.rot_p.value(),
            self.rot_y.value()
        ])
        return pos, rpy

    def block_all_signals(self, block):
        self.pos_x.blockSignals(block)
        self.pos_y.blockSignals(block)
        self.pos_z.blockSignals(block)
        self.rot_r.blockSignals(block)
        self.rot_p.blockSignals(block)
        self.rot_y.blockSignals(block)


class JointControlWidget(QWidget):
    """Single joint control widget"""
    valueChanged = pyqtSignal(int, float)  # joint_id, value

    def __init__(self, joint_id, joint_name, joint_range, joint_type):
        super().__init__()
        self.joint_id = joint_id
        self.joint_name = joint_name
        self.joint_range = joint_range
        self.joint_type = joint_type

        self.setupUI()

    def setupUI(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)

        # Joint name label
        name_label = QLabel(f"{self.joint_name}")
        name_label.setMinimumWidth(120)
        name_label.setMaximumWidth(120)
        layout.addWidget(name_label)

        # Value input box
        self.value_spinbox = QDoubleSpinBox()
        self.value_spinbox.setRange(self.joint_range[0], self.joint_range[1])
        self.value_spinbox.setSingleStep(0.01)
        self.value_spinbox.setDecimals(3)
        self.value_spinbox.setValue(0.0)
        self.value_spinbox.setMinimumWidth(80)
        self.value_spinbox.setMaximumWidth(80)
        self.value_spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.value_spinbox)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(self.joint_range[0] * 1000), int(self.joint_range[1] * 1000))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        # Range label
        range_label = QLabel(f"[{self.joint_range[0]:.2f}, {self.joint_range[1]:.2f}]")
        range_label.setMinimumWidth(100)
        range_label.setMaximumWidth(100)
        range_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(range_label)

        self.setLayout(layout)

    def on_slider_changed(self, value):
        real_value = value / 1000.0
        self.value_spinbox.blockSignals(True)
        self.value_spinbox.setValue(real_value)
        self.value_spinbox.blockSignals(False)
        self.valueChanged.emit(self.joint_id, real_value)

    def on_spinbox_changed(self, value):
        slider_value = int(value * 1000)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.joint_id, value)

    def set_value(self, value):
        self.value_spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        self.value_spinbox.setValue(value)
        self.slider.setValue(int(value * 1000))
        self.value_spinbox.blockSignals(False)
        self.slider.blockSignals(False)


class MujocoViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self.viewer = None
        self.viewer_thread = None
        self.joint_controls = []
        self.is_viewer_running = False
        self.is_simulation_running = False
        self.free_joint_id = -1
        self.free_joint_qpos_addr = -1
        self.config_file_path = os.path.join(os.path.expanduser('~'), '.mujoco_viewer_last_path.txt')
        
        # Trajectory playback state
        self.trajectory_data = None   # numpy array shape (N, D)
        self.current_frame = 0
        self.controlled_joint_count = 0
        self.trajectory_includes_base = False
        
        # Store names in the model
        self.body_names = []
        self.joint_names = []
        self.geom_names = []
        
        # Store original colors for highlighting
        self.original_geom_colors = {}
        self.highlighted_objects = {'source': None, 'target': None}

        # 🔧 线程安全的joint更新队列
        self._pending_joint_updates = {}
        self._pending_base_update = None
        self._update_lock = threading.Lock()

        # Integrated Qt renderer (fallback for macOS)
        self.render_widget = None
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._render_tick)

        self.setupUI()
        self._load_last_path()
        self.setWindowTitle("MuJoCo Model Viewer")
        self.resize(1200, 800)

    def setupUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left control panel
        self.setupControlPanel(splitter)

        # Right info panel
        self.setupInfoPanel(splitter)

        # Set splitter proportions (prioritize right panel viewing area)
        splitter.setSizes([450, 850])

    def setupControlPanel(self, parent):
        """Setup left control panel"""
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)

        # File loading area
        file_group = QGroupBox("Model File")
        file_layout = QVBoxLayout()

        file_input_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select XML file path...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        load_btn.setProperty('variant', 'primary')

        file_input_layout.addWidget(self.file_path_edit)
        file_input_layout.addWidget(browse_btn)
        file_input_layout.addWidget(load_btn)

        file_layout.addLayout(file_input_layout)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # Base control
        self.base_control_group = QGroupBox("Base Control (XYZ + RPY)")
        self.base_control_widget = BaseControlWidget()
        self.base_control_widget.valueChanged.connect(self.on_base_control_changed)
        base_group_layout = QVBoxLayout()
        base_group_layout.addWidget(self.base_control_widget)
        self.base_control_group.setLayout(base_group_layout)
        control_layout.addWidget(self.base_control_group)
        self.base_control_group.setVisible(False) # Initially hidden

        # Viewer control
        viewer_group = QGroupBox("Viewer Control")
        viewer_layout = QVBoxLayout()
        
        # First row: viewer buttons
        viewer_row1 = QHBoxLayout()
        self.start_viewer_btn = QPushButton("Start Viewer")
        self.start_viewer_btn.clicked.connect(self.start_viewer)
        self.start_viewer_btn.setEnabled(False)

        self.stop_viewer_btn = QPushButton("Stop Viewer")
        self.stop_viewer_btn.clicked.connect(self.stop_viewer)
        self.stop_viewer_btn.setEnabled(False)
        self.start_viewer_btn.setProperty('variant', 'info')
        self.stop_viewer_btn.setProperty('variant', 'danger')

        self.reset_btn = QPushButton("Reset Pose")
        self.reset_btn.clicked.connect(self.reset_pose)
        self.reset_btn.setEnabled(False)

        viewer_row1.addWidget(self.start_viewer_btn)
        viewer_row1.addWidget(self.stop_viewer_btn)
        viewer_row1.addWidget(self.reset_btn)
        
        # Second row: simulation control buttons
        sim_row = QHBoxLayout()
        self.start_sim_btn = QPushButton("Start Simulation (Fixed Joints)")
        self.start_sim_btn.clicked.connect(self.start_simulation)
        self.start_sim_btn.setEnabled(False)
        self.start_sim_btn.setProperty('variant', 'info')

        self.stop_sim_btn = QPushButton("Stop Simulation")
        self.stop_sim_btn.clicked.connect(self.stop_simulation)
        self.stop_sim_btn.setEnabled(False)
        self.stop_sim_btn.setProperty('variant', 'danger')

        sim_row.addWidget(self.start_sim_btn)
        sim_row.addWidget(self.stop_sim_btn)
        
        viewer_layout.addLayout(viewer_row1)
        viewer_layout.addLayout(sim_row)

        viewer_group.setLayout(viewer_layout)
        control_layout.addWidget(viewer_group)

        # Joint control area with tabs (manual control + trajectory playback)
        self.joint_group = QGroupBox("Joint Control")
        joint_group_layout = QVBoxLayout()

        # Tabs container
        self.joint_tabs = QTabWidget()

        # --- Manual Control Tab ---
        manual_tab = QWidget()
        manual_layout = QVBoxLayout()
        self.joint_scroll = QScrollArea()
        self.joint_scroll.setWidgetResizable(True)
        self.joint_scroll.setMinimumHeight(400)
        self.joint_widget = QWidget()
        self.joint_layout = QVBoxLayout()
        self.joint_widget.setLayout(self.joint_layout)
        self.joint_scroll.setWidget(self.joint_widget)
        manual_layout.addWidget(self.joint_scroll)
        manual_tab.setLayout(manual_layout)

        # --- Trajectory Playback Tab ---
        playback_tab = QWidget()
        playback_layout = QVBoxLayout()

        # CSV selector
        csv_row = QHBoxLayout()
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("选择CSV轨迹文件...")
        csv_browse_btn = QPushButton("浏览")
        csv_browse_btn.clicked.connect(self.browse_csv)
        csv_load_btn = QPushButton("加载CSV")
        csv_load_btn.clicked.connect(self.load_csv_trajectory)
        csv_row.addWidget(self.csv_path_edit)
        csv_row.addWidget(csv_browse_btn)
        csv_row.addWidget(csv_load_btn)
        playback_layout.addLayout(csv_row)

        # Info/status
        self.traj_info_label = QLabel("未加载轨迹")
        playback_layout.addWidget(self.traj_info_label)

        # Timeline controls
        timeline_row = QHBoxLayout()
        self.prev_frame_btn = QPushButton("上一帧")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn = QPushButton("下一帧")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.traj_slider = QSlider(Qt.Horizontal)
        self.traj_slider.setRange(0, 0)
        self.traj_slider.setEnabled(False)
        self.prev_frame_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)
        self.traj_slider.valueChanged.connect(self.on_traj_slider_changed)
        timeline_row.addWidget(self.prev_frame_btn)
        timeline_row.addWidget(self.traj_slider)
        timeline_row.addWidget(self.next_frame_btn)
        playback_layout.addLayout(timeline_row)

        playback_tab.setLayout(playback_layout)

        # Add tabs
        self.joint_tabs.addTab(manual_tab, "手动控制")
        self.joint_tabs.addTab(playback_tab, "轨迹播放")

        joint_group_layout.addWidget(self.joint_tabs)
        self.joint_group.setLayout(joint_group_layout)
        control_layout.addWidget(self.joint_group)

        parent.addWidget(control_widget)

    def setupInfoPanel(self, parent):
        """Setup right info panel with tabs + status"""
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_widget.setLayout(info_layout)

        # Tabs container
        self.info_tabs = QTabWidget()

        # --- Viewer Tab ---
        viewer_tab = QWidget()
        viewer_layout = QVBoxLayout()
        self.render_widget = MujocoWidget()
        viewer_layout.addWidget(self.render_widget)
        viewer_tab.setLayout(viewer_layout)
        self.info_tabs.addTab(viewer_tab, "Viewer")

        # --- Model Info Tab ---
        model_tab = QWidget()
        model_layout = QVBoxLayout()
        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        model_layout.addWidget(self.model_info_label)
        model_tab.setLayout(model_layout)
        self.info_tabs.addTab(model_tab, "Model Info")

        # --- Relative Pose Tab ---
        pose_tab = QWidget()
        pose_layout = QGridLayout()
        pose_layout.setColumnStretch(1, 1)

        pose_layout.addWidget(QLabel("Source Type:"), 0, 0)
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Body", "Geom", "Joint"])
        pose_layout.addWidget(self.source_type_combo, 0, 1)

        pose_layout.addWidget(QLabel("Source Name:"), 1, 0)
        self.source_name_combo = QComboBox()
        self.source_name_combo.setMinimumWidth(150)
        self.source_name_combo.setMaxVisibleItems(10)
        pose_layout.addWidget(self.source_name_combo, 1, 1)

        pose_layout.addWidget(QLabel("Target Type:"), 2, 0)
        self.target_type_combo = QComboBox()
        self.target_type_combo.addItems(["Body", "Geom", "Joint"])
        pose_layout.addWidget(self.target_type_combo, 2, 1)

        pose_layout.addWidget(QLabel("Target Name:"), 3, 0)
        self.target_name_combo = QComboBox()
        self.target_name_combo.setMaxVisibleItems(10)
        pose_layout.addWidget(self.target_name_combo, 3, 1)

        self.source_type_combo.currentIndexChanged.connect(
            lambda: self._on_selection_changed(self.source_type_combo, self.source_name_combo))
        self.target_type_combo.currentIndexChanged.connect(
            lambda: self._on_selection_changed(self.target_type_combo, self.target_name_combo))
        self.source_name_combo.currentTextChanged.connect(self._update_highlights)
        self.target_name_combo.currentTextChanged.connect(self._update_highlights)

        buttons = QHBoxLayout()
        self.calculate_pose_btn = QPushButton("Calculate Relative Pose")
        self.calculate_pose_btn.clicked.connect(self.calculate_relative_pose)
        buttons.addWidget(self.calculate_pose_btn)
        self.clear_highlight_btn = QPushButton("Clear Highlights")
        self.clear_highlight_btn.clicked.connect(self._clear_all_highlights)
        self.clear_highlight_btn.setProperty('variant', 'warning')
        buttons.addWidget(self.clear_highlight_btn)
        pose_layout.addLayout(buttons, 4, 0, 1, 2)

        self.highlight_status_label = QLabel("Highlight Status: No objects selected")
        self.highlight_status_label.setStyleSheet(
            "QLabel { background-color: #e3f2fd; padding: 5px; border: 1px solid #90caf9; border-radius: 3px; font-size: 10px; }")
        pose_layout.addWidget(self.highlight_status_label, 5, 0, 1, 2)

        self.relative_pose_label = QLabel("Please load a model and select two objects first")
        self.relative_pose_label.setWordWrap(True)
        self.relative_pose_label.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 5px; min-height: 80px;")
        pose_layout.addWidget(self.relative_pose_label, 6, 0, 1, 2)

        pose_tab.setLayout(pose_layout)
        self.info_tabs.addTab(pose_tab, "Relative Pose")

        # --- Inertia Tab ---
        inertia_tab = QWidget()
        inertia_layout = QVBoxLayout()

        btns = QHBoxLayout()
        self.calc_current_inertia_btn = QPushButton("Calculate Current Inertia")
        self.calc_current_inertia_btn.clicked.connect(self.calculate_current_inertia)
        self.calc_current_inertia_btn.setProperty('variant', 'info')
        btns.addWidget(self.calc_current_inertia_btn)
        self.calc_q0_inertia_btn = QPushButton("Calculate q0 Inertia")
        self.calc_q0_inertia_btn.clicked.connect(self.calculate_q0_inertia)
        self.calc_q0_inertia_btn.setProperty('variant', 'primary')
        btns.addWidget(self.calc_q0_inertia_btn)
        inertia_layout.addLayout(btns)

        self.inertia_results_scroll = QScrollArea()
        self.inertia_results_scroll.setWidgetResizable(True)
        self.inertia_results_scroll.setMinimumHeight(200)
        self.inertia_results_scroll.setMaximumHeight(400)
        self.inertia_results_label = QLabel(
            "Load a model and click a button to calculate inertia properties")
        self.inertia_results_label.setWordWrap(True)
        self.inertia_results_label.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 5px; font-family: monospace; font-size: 10px;")
        self.inertia_results_label.setAlignment(Qt.AlignTop)
        self.inertia_results_scroll.setWidget(self.inertia_results_label)
        inertia_layout.addWidget(self.inertia_results_scroll)

        inertia_tab.setLayout(inertia_layout)
        self.info_tabs.addTab(inertia_tab, "Mass/Inertia")

        info_layout.addWidget(self.info_tabs)

        # Status information (always visible beneath tabs)
        status_group = QGroupBox("Status Information")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setProperty('status', 'success')
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        info_layout.addWidget(status_group)

        parent.addWidget(info_widget)

    def _use_integrated_viewer(self):
        # On macOS, prefer integrated Qt rendering to avoid mjpython requirement
        return sys.platform == 'darwin'

    def _set_status(self, text, role='info'):
        """Unified status label styling via dynamic property."""
        self.status_label.setText(text)
        self.status_label.setProperty('status', role)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.status_label.update()

    def _save_last_path(self, path):
        """Save the last successfully loaded path"""
        try:
            with open(self.config_file_path, 'w') as f:
                f.write(path)
        except Exception as e:
            print(f"Warning: Cannot save last path: {e}")

    def _load_last_path(self):
        """Load and set the last path on startup"""
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r') as f:
                    path = f.read().strip()
                    if path and os.path.exists(path):
                        self.file_path_edit.setText(path)
        except Exception as e:
            print(f"Warning: Cannot load last path: {e}")

    def browse_file(self):
        """Browse file"""
        # Get the directory of the last opened file as default path
        start_dir = ""
        current_path = self.file_path_edit.text().strip()
        if current_path and os.path.exists(current_path):
            # If current path is valid, use its directory
            start_dir = os.path.dirname(current_path)
        elif os.path.exists(self.config_file_path):
            # Otherwise try to load last path from config file
            try:
                with open(self.config_file_path, 'r') as f:
                    last_path = f.read().strip()
                    if last_path and os.path.exists(last_path):
                        start_dir = os.path.dirname(last_path)
            except Exception:
                pass
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MuJoCo XML File", start_dir, "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def load_model(self):
        """Load model"""
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select an XML file first!")
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "File does not exist!")
            return

        try:
            # Stop current viewer
            if self.is_viewer_running:
                self.stop_viewer()

            # Load model
            self.model = mujoco.MjModel.from_xml_path(file_path)
            self.data = mujoco.MjData(self.model)
            
            print("MuJoCo model loaded successfully")
            
            self._save_last_path(file_path)

            # Update interface
            # Read all available names
            self.populate_name_lists()
            # Save original colors for highlighting
            self._save_original_colors()

            # First create joint controls
            self.create_joint_controls()
            # Reset trajectory UI for new model context
            self._reset_trajectory_ui()
            # Update model information
            self.update_model_info() 
            # Populate pose calculation selectors
            self.populate_pose_selectors()
            # Reset all states
            self.reset_pose()

            self.start_viewer_btn.setEnabled(True)
            self.start_sim_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)

            self._set_status("Model loaded successfully (MuJoCo only)", 'success')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self._set_status(f"Loading failed: {str(e)}", 'danger')

    def update_model_info(self):
        """Update model information"""
        if self.model is None:
            return

        joint_count = self.model.njnt
        # If a free joint (base) exists, subtract 1 from total count
        if self.free_joint_id != -1:
            joint_count -= 1

        # Calculate total mass and COM-referenced inertia matrix
        total_mass, com_inertia = self.calculate_mass_and_inertia()
        
        info_text = f"""
Model Name: {getattr(self.model, 'name', 'Unnamed')}
Joint Count: {joint_count}
Degrees of Freedom: {self.model.nv}
Body Count: {self.model.nbody}
Geometry Count: {self.model.ngeom}
Total Mass: {total_mass:.3f} kg
COM Inertia Matrix (kg⋅m²):
  [{com_inertia[0,0]:.4f}, {com_inertia[0,1]:.4f}, {com_inertia[0,2]:.4f}]
  [{com_inertia[1,0]:.4f}, {com_inertia[1,1]:.4f}, {com_inertia[1,2]:.4f}]
  [{com_inertia[2,0]:.4f}, {com_inertia[2,1]:.4f}, {com_inertia[2,2]:.4f}]
      """.strip()

        self.model_info_label.setText(info_text)

    def create_joint_controls(self):
        """Create joint control widgets"""
        # Clear existing controls
        for control in self.joint_controls:
            control.deleteLater()
        self.joint_controls.clear()

        # Clear all items in layout
        while self.joint_layout.count():
            child = self.joint_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.free_joint_id = -1
        self.free_joint_qpos_addr = -1
        self.base_control_group.setVisible(False)


        if self.model is None:
            return

        # Create control widgets for each joint
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                joint_name = f"joint_{i}"

            # Get joint type and range
            joint_type = self.model.jnt_type[i]
            
            # --- Base detection and handling ---
            if joint_type == mujoco.mjtJoint.mjJNT_FREE and self.free_joint_id == -1:
                self.free_joint_id = i
                self.free_joint_qpos_addr = self.model.jnt_qposadr[i]
                self.base_control_group.setVisible(True)
                continue # Don't create regular joint controller for base

            joint_range = self.model.jnt_range[i]

            # If joint range is 0, set default range
            if joint_range[0] == joint_range[1]:
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:  # Revolute joint
                    joint_range = [-np.pi, np.pi]
                else:  # Prismatic joint
                    joint_range = [-1.0, 1.0]

            # Create control widget
            control = JointControlWidget(i, joint_name, joint_range, joint_type)
            control.valueChanged.connect(self.on_joint_value_changed)

            self.joint_controls.append(control)
            self.joint_layout.addWidget(control)

        # Add stretch
        self.joint_layout.addStretch()
        self.controlled_joint_count = len(self.joint_controls)

    # ---------------- Trajectory Playback (CSV) ----------------
    def _reset_trajectory_ui(self):
        self.trajectory_data = None
        self.current_frame = 0
        self.trajectory_includes_base = False
        if hasattr(self, 'traj_slider'):
            self.traj_slider.blockSignals(True)
            self.traj_slider.setRange(0, 0)
            self.traj_slider.setValue(0)
            self.traj_slider.setEnabled(False)
            self.traj_slider.blockSignals(False)
        if hasattr(self, 'prev_frame_btn'):
            self.prev_frame_btn.setEnabled(False)
        if hasattr(self, 'next_frame_btn'):
            self.next_frame_btn.setEnabled(False)
        if hasattr(self, 'traj_info_label'):
            expected_nonbase = self.controlled_joint_count if self.model is not None else 0
            expected_text = str(expected_nonbase)
            if self.free_joint_id != -1:
                expected_text += f" 或 {expected_nonbase}+6(含基座XYZRPY)"
            self.traj_info_label.setText(f"未加载轨迹（期望维度: {expected_text}）")

    def browse_csv(self):
        start_dir = os.path.dirname(self.csv_path_edit.text().strip()) if self.csv_path_edit.text().strip() else ""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSV轨迹文件", start_dir, "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.csv_path_edit.setText(file_path)

    def load_csv_trajectory(self):
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载MuJoCo模型！")
            return
        path = self.csv_path_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "提示", "请选择有效的CSV文件！")
            return
        try:
            import numpy as _np
            try:
                data = _np.loadtxt(path, delimiter=",", dtype=float)
            except Exception:
                # Fallback allowing headers or mixed whitespace
                data = _np.genfromtxt(path, delimiter=",", dtype=float)
            if data.ndim == 1:
                data = _np.atleast_2d(data)

            N, D = data.shape
            expected_nonbase = self.controlled_joint_count
            base_ok = (self.free_joint_id != -1)
            ok_without_base = (D == expected_nonbase)
            ok_with_base = (base_ok and D == expected_nonbase + 6)
            if not (ok_without_base or ok_with_base):
                expect_text = f"{expected_nonbase}"
                if base_ok:
                    expect_text += f" 或 {expected_nonbase}+6(含基座XYZRPY)"
                QMessageBox.critical(self, "维度不匹配",
                                     f"CSV列数为 {D}。期望列数：{expect_text}。")
                self._reset_trajectory_ui()
                return

            self.trajectory_data = data
            self.current_frame = 0
            self.trajectory_includes_base = ok_with_base

            # Update UI
            self.traj_slider.blockSignals(True)
            self.traj_slider.setRange(0, max(0, N-1))
            self.traj_slider.setValue(0)
            self.traj_slider.setEnabled(True)
            self.traj_slider.blockSignals(False)
            self.prev_frame_btn.setEnabled(True)
            self.next_frame_btn.setEnabled(True)
            base_text = "，含基座" if self.trajectory_includes_base else ""
            self.traj_info_label.setText(f"已加载轨迹：帧数 {N}，维度 {D}{base_text}")

            # Apply first frame
            self.apply_trajectory_frame(0)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"读取CSV失败：{str(e)}")
            self._reset_trajectory_ui()

    def on_traj_slider_changed(self, value):
        if self.trajectory_data is None:
            return
        self.current_frame = int(value)
        self.apply_trajectory_frame(self.current_frame)

    def prev_frame(self):
        if self.trajectory_data is None:
            return
        new_idx = max(0, self.current_frame - 1)
        if new_idx != self.current_frame:
            self.current_frame = new_idx
            self.traj_slider.blockSignals(True)
            self.traj_slider.setValue(self.current_frame)
            self.traj_slider.blockSignals(False)
            self.apply_trajectory_frame(self.current_frame)

    def next_frame(self):
        if self.trajectory_data is None:
            return
        max_idx = self.trajectory_data.shape[0] - 1
        new_idx = min(max_idx, self.current_frame + 1)
        if new_idx != self.current_frame:
            self.current_frame = new_idx
            self.traj_slider.blockSignals(True)
            self.traj_slider.setValue(self.current_frame)
            self.traj_slider.blockSignals(False)
            self.apply_trajectory_frame(self.current_frame)

    def apply_trajectory_frame(self, index):
        """Apply the CSV row at `index` to all controllable joints.
        Keeps base pose unchanged if present."""
        if self.model is None or self.data is None or self.trajectory_data is None:
            return
        if index < 0 or index >= self.trajectory_data.shape[0]:
            return

        row = self.trajectory_data[index]

        # Split row into base and joints if needed
        joint_offset = 0
        if self.trajectory_includes_base and self.free_joint_id != -1:
            # Base: [x, y, z, roll, pitch, yaw]
            pos = np.array(row[0:3], dtype=float)
            rpy = np.array(row[3:6], dtype=float)

            # Update base qpos directly
            self.data.qpos[self.free_joint_qpos_addr:self.free_joint_qpos_addr + 3] = pos
            quat = euler_to_quaternion(rpy)
            self.data.qpos[self.free_joint_qpos_addr + 3:self.free_joint_qpos_addr + 7] = quat

            # Update base UI controls to reflect
            if hasattr(self, 'base_control_widget') and self.base_control_group.isVisible():
                self.base_control_widget.set_values(pos, rpy)

            joint_offset = 6

        # Update UI controls for non-base joints without emitting signals
        for ui_idx, control in enumerate(self.joint_controls):
            csv_idx = joint_offset + ui_idx
            if csv_idx < row.shape[0]:
                control.set_value(float(row[csv_idx]))

        # Schedule all joint updates atomically
        with self._update_lock:
            for ui_idx, control in enumerate(self.joint_controls):
                csv_idx = joint_offset + ui_idx
                if csv_idx < row.shape[0]:
                    joint_id = control.joint_id
                    value = float(row[csv_idx])
                    self._pending_joint_updates[joint_id] = value

        # Apply updates immediately if viewer not running or base changed
        self._apply_pending_updates()
        mujoco.mj_forward(self.model, self.data)
        # Status
        if hasattr(self, 'traj_info_label'):
            total = self.trajectory_data.shape[0]
            self.traj_info_label.setText(f"帧 {index+1}/{total} 已应用；维度 {row.shape[0]}")

    def on_joint_value_changed(self, joint_id, value):
        """Callback when joint value changes"""
        if self.data is not None and self.model is not None:
            # 🔧 线程安全：始终通过调度机制更新
            self._schedule_joint_update(joint_id, value)

            # 如果viewer没运行，立即应用更新
            if not self.is_viewer_running:
                self._apply_pending_updates()

    def _schedule_joint_update(self, joint_id, value):
        """🔧 线程安全地调度joint更新"""
        with self._update_lock:
            self._pending_joint_updates[joint_id] = value

    def _apply_pending_updates(self):
        """🔧 应用所有待处理的关节更新（线程安全）"""
        if self.model is None or self.data is None:
            return

        # 获取并清空待处理更新
        with self._update_lock:
            updates = self._pending_joint_updates.copy()
            self._pending_joint_updates.clear()

        # 应用所有更新
        for joint_id, value in updates.items():
            qpos_addr = self.model.jnt_qposadr[joint_id]
            if qpos_addr < self.model.nq:
                self.data.qpos[qpos_addr] = value

        # 只在有更新时才执行forward
        if updates:
            mujoco.mj_forward(self.model, self.data)

    def _schedule_base_update(self, pos, quat):
        """🔧 线程安全地调度base更新"""
        with self._update_lock:
            self._pending_base_update = {'pos': pos.copy(), 'quat': quat.copy()}

    def on_base_control_changed(self):
        """Callback when base controller value changes"""
        if self.data is None or self.free_joint_id == -1:
            return

        pos, rpy = self.base_control_widget.get_values()

        # Update position
        self.data.qpos[self.free_joint_qpos_addr:self.free_joint_qpos_addr + 3] = pos

        # Update orientation (RPY -> quaternion)
        quat = euler_to_quaternion(rpy)
        self.data.qpos[self.free_joint_qpos_addr + 3:self.free_joint_qpos_addr + 7] = quat

        mujoco.mj_forward(self.model, self.data)

    def start_viewer(self):
        """Start viewer"""
        if self.model is None:
            return

        if self.is_viewer_running:
            return

        if self._use_integrated_viewer():
            # Use embedded Qt renderer
            self.render_widget.set_model(self.model, self.data)
            self.render_timer.start(16)  # ~60 FPS
            self.is_viewer_running = True
            self.start_viewer_btn.setEnabled(False)
            self.stop_viewer_btn.setEnabled(True)
            self.start_sim_btn.setEnabled(True)
            self._set_status("Integrated viewer running (Qt)", 'info')
        else:
            self.is_viewer_running = True
            self.viewer_thread = threading.Thread(target=self.viewer_loop, daemon=True)
            self.viewer_thread.start()

            self.start_viewer_btn.setEnabled(False)
            self.stop_viewer_btn.setEnabled(True)
            self.start_sim_btn.setEnabled(True)
            self._set_status("Viewer running...", 'info')

    def stop_viewer(self):
        """Stop viewer"""
        # First stop simulation
        if self.is_simulation_running:
            self.stop_simulation()
            
        if self._use_integrated_viewer():
            self.render_timer.stop()
            self.is_viewer_running = False

            self.start_viewer_btn.setEnabled(True)
            self.stop_viewer_btn.setEnabled(False)
            self.start_sim_btn.setEnabled(False)
            self.stop_sim_btn.setEnabled(False)
            self._set_status("Integrated viewer stopped", 'warning')
        else:
            self.is_viewer_running = False

            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

            if self.viewer_thread is not None:
                self.viewer_thread.join(timeout=2.0)
                self.viewer_thread = None

            self.start_viewer_btn.setEnabled(True)
            self.stop_viewer_btn.setEnabled(False)
            self.start_sim_btn.setEnabled(False)
            self.stop_sim_btn.setEnabled(False)
            self._set_status("Viewer stopped", 'warning')

    def _render_tick(self):
        """Tick function for integrated Qt viewer."""
        if self.model is None or self.data is None:
            return
        # Apply pending joint updates
        self._apply_pending_updates()
        # Step simulation if enabled
        if self.is_simulation_running:
            self._apply_joint_holding_torques()
            mujoco.mj_step(self.model, self.data)
        # Request repaint
        if self.render_widget is not None:
            self.render_widget.update()

    def viewer_loop(self):
        """Viewer main loop"""
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer

                # Set camera
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.distance = 3.0

                while self.is_viewer_running and viewer.is_running():
                    # 🔧 应用待处理的joint更新（线程安全）
                    self._apply_pending_updates()

                    # If simulation is running, step simulation
                    if self.is_simulation_running:
                        # Add control torques for joint fixing
                        self._apply_joint_holding_torques()
                        mujoco.mj_step(self.model, self.data)

                    # Sync viewer
                    viewer.sync()

                    # Control frame rate
                    time.sleep(0.01)

        except Exception as e:
            print(f"Viewer error: {e}")
            self.is_viewer_running = False

        # Update UI in main thread
        QTimer.singleShot(0, self._on_viewer_closed)

    def _on_viewer_closed(self):
        """Handle viewer closed event"""
        self.viewer = None
        self.is_viewer_running = False
        self.is_simulation_running = False
        self.start_viewer_btn.setEnabled(True)
        self.stop_viewer_btn.setEnabled(False)
        self.start_sim_btn.setEnabled(False)
        self.stop_sim_btn.setEnabled(False)
        self._set_status("Viewer closed", 'muted')

    def reset_pose(self):
        """Reset to initial pose"""
        if self.data is None:
            return

        # Stop simulation
        if self.is_simulation_running:
            self.stop_simulation()

        # Reset all joint positions
        mujoco.mj_resetData(self.model, self.data)

        # Load keyframe 'home' if it exists
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_key_id >= 0:
            # Copy keyframe qpos to current data
            self.data.qpos[:] = self.model.key_qpos[home_key_id]
            print(f"Loaded keyframe 'home': qpos = {self.data.qpos}")
        else:
            print("No 'home' keyframe found, using default reset")

        # Update base control display
        if self.free_joint_id != -1:
            self.update_base_controls_from_data()

        # Update joint control display
        for control in self.joint_controls:
            joint_id = control.joint_id
            # Find the qpos address corresponding to the joint
            qpos_addr = self.model.jnt_qposadr[joint_id]
            if qpos_addr < len(self.data.qpos):
                control.set_value(self.data.qpos[qpos_addr])

        # Forward kinematics calculation
        mujoco.mj_forward(self.model, self.data)

        self._set_status("Pose reset to 'home' keyframe", 'success')

    def start_simulation(self):
        """Start simulation (fixed joints)"""
        if self.model is None or self.data is None:
            return

        if not self.is_viewer_running:
            QMessageBox.warning(self, "Warning", "Please start the viewer first!")
            return

        self.is_simulation_running = True
        
        # Save current joint target positions for fixing
        self._save_joint_target_positions()
        
        self.start_sim_btn.setEnabled(False)
        self.stop_sim_btn.setEnabled(True)
        
        self._set_status("Simulation running (fixed joints), observing stability...", 'info')

    def stop_simulation(self):
        """Stop simulation"""
        self.is_simulation_running = False
        
        self.start_sim_btn.setEnabled(True)
        self.stop_sim_btn.setEnabled(False)
        
        self._set_status("Simulation stopped", 'warning')

    def _save_joint_target_positions(self):
        """Save current joint positions as fixed targets"""
        if self.model is None or self.data is None:
            return
            
        self.joint_target_positions = {}
        
        for i in range(self.model.njnt):
            # Skip free joint (base)
            if i == self.free_joint_id:
                continue
                
            qpos_addr = self.model.jnt_qposadr[i]
            if qpos_addr < len(self.data.qpos):
                self.joint_target_positions[i] = self.data.qpos[qpos_addr]

    def _apply_joint_holding_torques(self):
        """Apply joint holding torques to fix joint positions"""
        if self.model is None or self.data is None:
            return
            
        if not hasattr(self, 'joint_target_positions'):
            return
            
        # PD control parameters
        kp = 500.0  # Position gain
        kd = 50.0   # Velocity gain
        
        for joint_id, target_pos in self.joint_target_positions.items():
            # Get joint positions in arrays
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            
            if qpos_addr < len(self.data.qpos) and qvel_addr < len(self.data.qvel):
                # Calculate position error and velocity
                pos_error = target_pos - self.data.qpos[qpos_addr]
                velocity = self.data.qvel[qvel_addr]
                
                # PD control torque
                torque = kp * pos_error - kd * velocity
                
                # Apply torque (limit maximum torque)
                max_torque = 100.0
                torque = np.clip(torque, -max_torque, max_torque)
                
                # Set control input
                if qvel_addr < len(self.data.ctrl):
                    self.data.ctrl[qvel_addr] = torque

    def update_base_controls_from_data(self):
        """Update base UI controls from MuJoCo data"""
        if self.data is None or self.free_joint_id == -1:
            return

        # Get position
        pos = self.data.qpos[self.free_joint_qpos_addr:self.free_joint_qpos_addr + 3]

        # Get orientation (quaternion -> RPY)
        quat = self.data.qpos[self.free_joint_qpos_addr + 3:self.free_joint_qpos_addr + 7]
        rpy = quaternion_to_euler(quat)

        self.base_control_widget.set_values(pos, rpy)

    def populate_name_lists(self):
        """When loading model, read and store all body, geom, joint names"""
        if self.model is None:
            self.body_names, self.joint_names, self.geom_names = [], [], []
            return

        # First get all names (may contain None)
        raw_body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, self.model.nbody)]
        raw_joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        raw_geom_names = []
        
        # Handle geom names, including unnamed geoms
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                # Generate default name for unnamed geoms
                name = f"geom_{i}"
            raw_geom_names.append(name)
        
        # First filter out None values, then sort
        self.body_names = sorted([name for name in raw_body_names if name])
        self.joint_names = sorted([name for name in raw_joint_names if name])
        self.geom_names = sorted(raw_geom_names)  # geom_names already doesn't contain None

    def _on_type_changed(self, type_combo, name_combo):
        """When type dropdown changes, update name dropdown content"""
        selected_type = type_combo.currentText()
        name_combo.clear()

        if selected_type == "Body":
            name_combo.addItems(self.body_names)
        elif selected_type == "Geom":
            name_combo.addItems(self.geom_names)
        elif selected_type == "Joint":
            name_combo.addItems(self.joint_names)

    def _on_selection_changed(self, type_combo, name_combo):
        """Handler for when type selection changes"""
        self._on_type_changed(type_combo, name_combo)
        # Delay highlight update to let name dropdown update first
        if hasattr(self, 'model') and self.model is not None:
            QTimer.singleShot(50, self._update_highlights)

    def populate_pose_selectors(self):
        """Populate selectors for relative pose calculation"""
        # Trigger type switch signal once to populate initial name lists
        self._on_type_changed(self.source_type_combo, self.source_name_combo)
        self._on_type_changed(self.target_type_combo, self.target_name_combo)
        self.relative_pose_label.setText("Please select two objects for calculation")
        
        # Initialize highlight status
        self._update_highlight_status()

    def _save_original_colors(self):
        """Save original colors of all geoms"""
        if self.model is None:
            return
        self.original_geom_colors = {}
        for i in range(self.model.ngeom):
            # Save original rgba values
            self.original_geom_colors[i] = self.model.geom_rgba[i].copy()

    def _highlight_object(self, obj_type, obj_name, color, highlight_type):
        """Highlight specified object
        Args:
            obj_type: Object type ("Body", "Geom", "Joint")
            obj_name: Object name
            color: Highlight color [r, g, b, a]
            highlight_type: 'source' or 'target'
        """
        if self.model is None or not obj_name:
            return

        # First restore previously highlighted object
        old_obj = self.highlighted_objects[highlight_type]
        if old_obj:
            self._restore_object_color(old_obj['type'], old_obj['name'])

        try:
            if obj_type == "Geom":
                # Get geom ID
                if obj_name.startswith("geom_"):
                    geom_id = int(obj_name.split("_")[1])
                else:
                    geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
                    if geom_id == -1:
                        return

                # Set highlight color
                self.model.geom_rgba[geom_id] = color
                
            elif obj_type == "Body":
                # For body, highlight all associated geoms
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if body_id == -1:
                    return
                
                # Find all geoms belonging to this body
                for geom_id in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_id] == body_id:
                        self.model.geom_rgba[geom_id] = color

            elif obj_type == "Joint":
                # For joint, highlight geoms of its body
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_name)
                if joint_id == -1:
                    return
                
                body_id = self.model.jnt_bodyid[joint_id]
                # Find all geoms belonging to this body
                for geom_id in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_id] == body_id:
                        self.model.geom_rgba[geom_id] = color

            # Record currently highlighted object
            self.highlighted_objects[highlight_type] = {
                'type': obj_type,
                'name': obj_name
            }

        except (ValueError, IndexError):
            pass  # Ignore errors

    def _restore_object_color(self, obj_type, obj_name):
        """Restore object's original color"""
        if self.model is None or not obj_name:
            return

        try:
            if obj_type == "Geom":
                # Get geom ID
                if obj_name.startswith("geom_"):
                    geom_id = int(obj_name.split("_")[1])
                else:
                    geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
                    if geom_id == -1:
                        return

                # Restore original color
                if geom_id in self.original_geom_colors:
                    self.model.geom_rgba[geom_id] = self.original_geom_colors[geom_id]
                
            elif obj_type == "Body":
                # For body, restore colors of all associated geoms
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if body_id == -1:
                    return
                
                # Restore all geoms belonging to this body
                for geom_id in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_id] == body_id:
                        if geom_id in self.original_geom_colors:
                            self.model.geom_rgba[geom_id] = self.original_geom_colors[geom_id]

            elif obj_type == "Joint":
                # For joint, restore geom colors of its body
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_name)
                if joint_id == -1:
                    return
                
                body_id = self.model.jnt_bodyid[joint_id]
                # Restore all geoms belonging to this body
                for geom_id in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_id] == body_id:
                        if geom_id in self.original_geom_colors:
                            self.model.geom_rgba[geom_id] = self.original_geom_colors[geom_id]

        except (ValueError, IndexError):
            pass  # Ignore errors

    def _update_highlights(self):
        """Update highlight display based on current selection"""
        if self.model is None:
            return

        # Get current selection
        source_type = self.source_type_combo.currentText()
        source_name = self.source_name_combo.currentText()
        target_type = self.target_type_combo.currentText()
        target_name = self.target_name_combo.currentText()

        # Red highlight for source object
        self._highlight_object(source_type, source_name, [1.0, 0.0, 0.0, 1.0], 'source')
        
        # Blue highlight for target object
        self._highlight_object(target_type, target_name, [0.0, 0.0, 1.0, 1.0], 'target')
        

        
        # Update status indicator
        self._update_highlight_status()

    def _clear_all_highlights(self):
        """Clear all highlights and restore original colors"""
        if self.model is None:
            return

        # Restore original colors for all geoms
        for geom_id, original_color in self.original_geom_colors.items():
            if geom_id < self.model.ngeom:
                self.model.geom_rgba[geom_id] = original_color

        # Clear highlight records
        self.highlighted_objects = {'source': None, 'target': None}
        

        
        # Update status indicator
        self._update_highlight_status()

    def calculate_mass_and_inertia(self):
        """Calculate total mass and inertia matrix using proper MuJoCo API."""
        if self.model is None or self.data is None:
            return 0.0, np.eye(3)
        
        # Forward kinematics to ensure data is current
        mujoco.mj_forward(self.model, self.data)
        
        # Use MuJoCo's built-in function to get total mass
        total_mass = mujoco.mj_getTotalmass(self.model)
        
        # Get joint-space inertia matrix using MuJoCo's built-in function
        inertia_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, inertia_matrix, self.data.qM)
        
        # Extract rotational inertia for floating base robot
        if self.model.nv >= 6:
            # For floating base, rotational inertia is in bottom-right 3x3 of 6x6 base block
            rotational_inertia = inertia_matrix[3:6, 3:6]
        else:
            # Fallback for non-floating base robots
            rotational_inertia = np.eye(3)
        
        return total_mass, rotational_inertia
    
    def calculate_mass_and_inertia_at_config(self, joint_config=None):
        """Calculate mass and inertia at a specific joint configuration.
        
        Args:
            joint_config: Joint configuration array. If None, uses current configuration.
        """
        if self.model is None or self.data is None:
            return 0.0, np.eye(3), np.zeros(3)
        
        # Save current state
        original_qpos = self.data.qpos.copy()
        
        try:
            # Set joint configuration if provided
            if joint_config is not None:
                # Ensure configuration matches model DOFs
                if len(joint_config) <= len(self.data.qpos):
                    # For floating base robots, preserve base pose and set joint angles
                    if self.free_joint_id != -1 and self.free_joint_qpos_addr != -1:
                        # Keep base pose unchanged, set joint angles
                        joint_start = self.free_joint_qpos_addr + 7  # After base pose (3 pos + 4 quat)
                        joint_end = min(joint_start + len(joint_config), len(self.data.qpos))
                        self.data.qpos[joint_start:joint_end] = joint_config[:joint_end-joint_start]
                    else:
                        # Direct joint configuration
                        self.data.qpos[:len(joint_config)] = joint_config
            
            # Update kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Calculate mass, inertia, and COM
            total_mass, inertia_matrix = self.calculate_mass_and_inertia()
            
            # Calculate center of mass
            com_position = self._calculate_center_of_mass()
            
            return total_mass, inertia_matrix, com_position
            
        finally:
            # Restore original state
            self.data.qpos[:] = original_qpos
            mujoco.mj_forward(self.model, self.data)
    
    def _calculate_center_of_mass(self):
        """Calculate center of mass of the system."""
        if self.model is None or self.data is None:
            return np.zeros(3)
        
        total_mass = 0.0
        com_numerator = np.zeros(3)
        
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            if body_mass > 0:  # Skip massless bodies
                total_mass += body_mass
                body_pos = self.data.xpos[i]
                com_numerator += body_mass * body_pos
        
        if total_mass > 0:
            return com_numerator / total_mass
        else:
            return np.zeros(3)
    
    def _quat_to_rotation_matrix(self, quat):
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def _skew_symmetric_matrix(self, vec):
        """Create skew-symmetric matrix from 3D vector."""
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

    def _update_highlight_status(self):
        """Update highlight status indicator"""
        source_obj = self.highlighted_objects.get('source')
        target_obj = self.highlighted_objects.get('target')
        
        status_parts = []
        
        if source_obj and source_obj['name']:
            status_parts.append(f"🔴 Source: {source_obj['name']}")
        
        if target_obj and target_obj['name']:
            status_parts.append(f"🔵 Target: {target_obj['name']}")
        
        if status_parts:
            status_text = "Highlight Status: " + " | ".join(status_parts)
        else:
            status_text = "Highlight Status: No objects selected"
        
        if hasattr(self, 'highlight_status_label'):
            self.highlight_status_label.setText(status_text)
    
    def calculate_current_inertia(self):
        """Calculate and display mass/inertia properties at current joint configuration."""
        if self.model is None or self.data is None:
            self.inertia_results_label.setText("No model loaded!")
            return
        
        try:
            # Calculate at current configuration
            total_mass, inertia_matrix, com_pos = self.calculate_mass_and_inertia_at_config()
            
            # Format results for display
            results_text = f"""CURRENT CONFIGURATION ANALYSIS:

Mass: {total_mass:.3f} kg
COM: [{com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f}] m

Inertia Matrix (kg⋅m²):
[{inertia_matrix[0,0]:.3f}, {inertia_matrix[0,1]:.3f}, {inertia_matrix[0,2]:.3f}]
[{inertia_matrix[1,0]:.3f}, {inertia_matrix[1,1]:.3f}, {inertia_matrix[1,2]:.3f}]
[{inertia_matrix[2,0]:.3f}, {inertia_matrix[2,1]:.3f}, {inertia_matrix[2,2]:.3f}]

Diagonal Values:
Ixx: {inertia_matrix[0,0]:.3f} kg⋅m²
Iyy: {inertia_matrix[1,1]:.3f} kg⋅m²
Izz: {inertia_matrix[2,2]:.3f} kg⋅m²"""
            
            self.inertia_results_label.setText(results_text)
            
        except Exception as e:
            error_text = f"Error calculating current inertia:\n{str(e)}"
            self.inertia_results_label.setText(error_text)
    
    def calculate_q0_inertia(self):
        """Calculate and display mass/inertia properties at q0 configuration from H1 SRBD config."""
        if self.model is None or self.data is None:
            self.inertia_results_label.setText("No model loaded!")
            return
        
        try:
            # Import q0 from H1 SRBD config
            try:
                # Import the H1 SRBD config to get q0 values
                import sys
                import os
                config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
                if config_dir not in sys.path:
                    sys.path.append(config_dir)
                
                from config_h1_srbd import q0 as h1_q0
                
                # Convert JAX array to numpy if needed
                if hasattr(h1_q0, 'tolist'):
                    q0_config = np.array(h1_q0.tolist())
                else:
                    q0_config = np.array(h1_q0)
                    
            except ImportError as e:
                error_text = f"Could not import H1 SRBD config:\n{str(e)}\n\nUsing default q0 configuration..."
                # Default H1 q0 configuration (legs only, 10 DOF)
                q0_config = np.array([0, 0, -0.4, 0.8, -0.4,      # Left leg (5 DOF)
                                     0, 0, -0.4, 0.8, -0.4])      # Right leg (5 DOF)
                self.inertia_results_label.setText(error_text)
                return
            
            # Calculate at q0 configuration
            total_mass, inertia_matrix, com_pos = self.calculate_mass_and_inertia_at_config(q0_config)
            
            # Compare with current config values from H1 SRBD config
            try:
                from config_h1_srbd import mass as config_mass, inertia as config_inertia
                if hasattr(config_inertia, 'tolist'):
                    config_inertia_np = np.array(config_inertia.tolist())
                else:
                    config_inertia_np = np.array(config_inertia)
                    
                # Calculate ratios
                mass_ratio = total_mass / config_mass if config_mass > 0 else 0
                inertia_ratios = np.diag(inertia_matrix) / np.diag(config_inertia_np)
                
            except ImportError:
                config_mass = 87.89  # Default from previous analysis
                config_inertia_np = np.array([[17.993, 0.0, 0.0],
                                             [0.0, 17.168, 0.0], 
                                             [0.0, 0.0, 1.377]])
                mass_ratio = total_mass / config_mass
                inertia_ratios = np.diag(inertia_matrix) / np.diag(config_inertia_np)
            
            # Format results for display
            results_text = f"""H1 q0 CONFIGURATION ANALYSIS:

q0: {q0_config}

Mass: {total_mass:.3f} kg
COM: [{com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f}] m

Inertia Matrix (kg⋅m²):
[{inertia_matrix[0,0]:.3f}, {inertia_matrix[0,1]:.3f}, {inertia_matrix[0,2]:.3f}]
[{inertia_matrix[1,0]:.3f}, {inertia_matrix[1,1]:.3f}, {inertia_matrix[1,2]:.3f}]
[{inertia_matrix[2,0]:.3f}, {inertia_matrix[2,1]:.3f}, {inertia_matrix[2,2]:.3f}]

CONFIG COMPARISON:
Config Mass: {config_mass:.1f} kg (Ratio: {mass_ratio:.2f})
Config Inertia: [{config_inertia_np[0,0]:.1f}, {config_inertia_np[1,1]:.1f}, {config_inertia_np[2,2]:.1f}]
Ratios: [{inertia_ratios[0]:.2f}, {inertia_ratios[1]:.2f}, {inertia_ratios[2]:.2f}]

SUGGESTED CONFIG UPDATE:
mass = {total_mass:.3f}
inertia = jnp.array([
  [{inertia_matrix[0,0]:.3f}, 0.0, 0.0],
  [0.0, {inertia_matrix[1,1]:.3f}, 0.0],
  [0.0, 0.0, {inertia_matrix[2,2]:.3f}]])"""
            
            self.inertia_results_label.setText(results_text)
            
        except Exception as e:
            error_text = f"Error calculating q0 inertia:\n{str(e)}"
            self.inertia_results_label.setText(error_text)



    def get_object_pose_mujoco(self, obj_type, obj_name):
        """Get object's global pose based on type and name (using MuJoCo)"""
        mujoco.mj_forward(self.model, self.data)

        if obj_type == "Body":
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if obj_id == -1: raise ValueError(f"Body not found: {obj_name}")
            pos = self.data.xpos[obj_id]
            mat = self.data.xmat[obj_id].reshape(3, 3)
            return pos, mat
        elif obj_type == "Geom":
            # Check if it's a generated default name
            if obj_name.startswith("geom_"):
                try:
                    obj_id = int(obj_name.split("_")[1])
                    if obj_id >= self.model.ngeom:
                        raise ValueError(f"Geom ID out of range: {obj_id}")
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid Geom name: {obj_name}")
            else:
                obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
                if obj_id == -1: 
                    raise ValueError(f"Geom not found: {obj_name}")
            
            pos = self.data.geom_xpos[obj_id]
            mat = self.data.geom_xmat[obj_id].reshape(3, 3)
            return pos, mat
        elif obj_type == "Joint":
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_name)
            if obj_id == -1: raise ValueError(f"Joint not found: {obj_name}")
            # Position: use joint anchor's global position
            pos = self.data.xanchor[obj_id]
            # Orientation: use its parent body's orientation as proxy
            body_id = self.model.jnt_bodyid[obj_id]
            mat = self.data.xmat[body_id].reshape(3, 3)
            return pos, mat
        else:
            raise TypeError(f"Unknown object type: {obj_type}")

    def calculate_relative_pose(self):
        """Calculate and display relative pose between two objects"""
        if self.model:
            try:
                self.calculate_relative_pose_mujoco()
            except Exception as e:
                self.relative_pose_label.setText(f"MuJoCo calculation error: {e}")
        else:
            self.relative_pose_label.setText("Please load a model first")



    def calculate_relative_pose_mujoco(self):
        """Calculate relative pose using MuJoCo"""
        source_type = self.source_type_combo.currentText()
        source_name = self.source_name_combo.currentText()
        target_type = self.target_type_combo.currentText()
        target_name = self.target_name_combo.currentText()

        if not source_name or not target_name:
            self.relative_pose_label.setText("Please select source and target objects")
            return

        source_pos, source_mat = self.get_object_pose_mujoco(source_type, source_name)
        target_pos, target_mat = self.get_object_pose_mujoco(target_type, target_name)

        # Calculate relative position
        source_mat_inv = source_mat.T
        relative_pos = source_mat_inv @ (target_pos - source_pos)

        # Calculate relative rotation
        relative_mat = source_mat_inv @ target_mat
        
        # Convert rotation matrix to Euler angles (ZYX order, commonly used in MuJoCo)
        sy = np.sqrt(relative_mat[0,0] * relative_mat[0,0] +  relative_mat[1,0] * relative_mat[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(relative_mat[2,1] , relative_mat[2,2])
            y = np.arctan2(-relative_mat[2,0], sy)
            z = np.arctan2(relative_mat[1,0], relative_mat[0,0])
        else:
            x = np.arctan2(-relative_mat[1,2], relative_mat[1,1])
            y = np.arctan2(-relative_mat[2,0], sy)
            z = 0

        relative_rpy_rad = np.array([x, y, z])
        relative_rpy_deg = np.rad2deg(relative_rpy_rad)

        result_text = f"""
<b>Relative pose from {source_type} '{source_name}' to {target_type} '{target_name}':</b>
<br>
<b>Relative Position (m):</b><br>
&nbsp;&nbsp;X: {relative_pos[0]:.4f}<br>
&nbsp;&nbsp;Y: {relative_pos[1]:.4f}<br>
&nbsp;&nbsp;Z: {relative_pos[2]:.4f}<br>
<br>
<b>Relative Rotation (Euler ZYX, degrees):</b><br>
&nbsp;&nbsp;Roll (X): {relative_rpy_deg[0]:.2f}°<br>
&nbsp;&nbsp;Pitch (Y): {relative_rpy_deg[1]:.2f}°<br>
&nbsp;&nbsp;Yaw (Z): {relative_rpy_deg[2]:.2f}°
        """.strip()

        self.relative_pose_label.setText(result_text)

    def closeEvent(self, event):
        """Window close event"""
        if self.is_viewer_running:
            self.stop_viewer()
        event.accept()

class ModernStyle:
    """Modern interface style"""

    @staticmethod
    def apply(app):
        app.setStyle("Fusion")

        # Create palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)

        app.setPalette(palette)

        # Set stylesheet
        app.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }

            QPushButton {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px 15px;
                background-color: #ffffff;
                min-height: 25px;
            }

            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #999999;
            }

            QPushButton:pressed {
                background-color: #d0d0d0;
            }

            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #999999;
            }

            /* Button variants */
            QPushButton[variant="primary"] {
                background-color: #4CAF50;
                color: white;
                border-color: #45a049;
                font-weight: 600;
            }
            QPushButton[variant="primary"]:hover { background-color: #43a047; }
            QPushButton[variant="primary"]:pressed { background-color: #388e3c; }

            QPushButton[variant="info"] {
                background-color: #2196F3;
                color: white;
                border-color: #1e88e5;
                font-weight: 600;
            }
            QPushButton[variant="info"]:hover { background-color: #1e88e5; }
            QPushButton[variant="info"]:pressed { background-color: #1976d2; }

            QPushButton[variant="danger"] {
                background-color: #f44336;
                color: white;
                border-color: #e53935;
                font-weight: 600;
            }
            QPushButton[variant="danger"]:hover { background-color: #e53935; }
            QPushButton[variant="danger"]:pressed { background-color: #d32f2f; }

            QPushButton[variant="warning"] {
                background-color: #ffeb3b;
                color: #333333;
                border-color: #fdd835;
                font-weight: 600;
            }
            QPushButton[variant="warning"]:hover { background-color: #ffe035; }
            QPushButton[variant="warning"]:pressed { background-color: #ffd600; }

            QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }

            QLineEdit:focus {
                border-color: #4CAF50;
            }

            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #45a049;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }

            QSlider::handle:horizontal:hover {
                background: #45a049;
            }

            QScrollArea {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }

            QLabel {
                color: #333333;
            }

            QSpinBox, QDoubleSpinBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 3px;
                background-color: white;
            }

            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #4CAF50;
            }

            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                color: black;
                min-height: 20px;
            }

            QComboBox:focus {
                border-color: #4CAF50;
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #666;
                margin-right: 5px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #cccccc;
                background-color: white;
                selection-background-color: #4CAF50;
                selection-color: white;
                outline: none;
                margin: 0px;
                padding: 0px;
            }

            QComboBox QAbstractItemView::item {
                padding: 3px 8px;
                border: none;
                margin: 0px;
                height: 20px;
            }

            QComboBox QAbstractItemView::item:selected {
                background-color: #4CAF50;
                color: white;
            }

            QComboBox QAbstractItemView::item:hover {
                background-color: #e8f5e8;
            }

            /* Tabs styling */
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #fafafa;
                border: 1px solid #cccccc;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover { background: #f0f0f0; }

            /* Splitter */
            QSplitter::handle {
                background: #e0e0e0;
            }

            /* Status labels */
            QLabel[status] {
                padding: 10px;
                border-radius: 5px;
                font-weight: 500;
            }
            QLabel[status="success"] { background-color: #e8f5e8; color: #2e7d32; }
            QLabel[status="info"] { background-color: #e3f2fd; color: #1565c0; }
            QLabel[status="warning"] { background-color: #fff3e0; color: #e65100; }
            QLabel[status="danger"] { background-color: #ffebee; color: #c62828; }
            QLabel[status="muted"] { background-color: #f5f5f5; color: #616161; }
        """)

def main():
    """Main function"""
    app = QApplication(sys.argv)

    # Apply modern style
    ModernStyle.apply(app)

    # Set application information
    app.setApplicationName("MuJoCo Model Viewer")
    app.setOrganizationName("MujocoViewer")

    # Create main window
    viewer = MujocoViewer()
    viewer.show()

    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
