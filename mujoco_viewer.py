import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QSlider, QLineEdit, QFileDialog, QScrollArea,
                             QGroupBox, QSplitter, QMessageBox, QSpinBox,
                             QDoubleSpinBox, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPainter
import mujoco
import mujoco.viewer
import threading
import time


def euler_to_quaternion(rpy):
    """将欧拉角序列转换为四元数。"""
    # 假设 rpy 是弧度单位，顺序为 'xyz' (roll, pitch, yaw)
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # MuJoCo 四元数顺序为 (w, x, y, z)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def quaternion_to_euler(quat):
    """将四元数转换为欧拉角序列。"""
    # 假设 quat 顺序为 (w, x, y, z)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # 横滚 (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # 俯仰 (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 如果超出范围，则使用90度
    else:
        pitch = np.arcsin(sinp)

    # 偏航 (z-axis rotation)
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
            painter.drawText(self.rect(), Qt.AlignCenter, "在此处加载模型以开始")


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
    """基座控制组件 (XYZ + RPY)"""
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
    """单个关节控制组件"""
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

        # 关节名称标签
        name_label = QLabel(f"{self.joint_name}")
        name_label.setMinimumWidth(120)
        name_label.setMaximumWidth(120)
        layout.addWidget(name_label)

        # 数值输入框
        self.value_spinbox = QDoubleSpinBox()
        self.value_spinbox.setRange(self.joint_range[0], self.joint_range[1])
        self.value_spinbox.setSingleStep(0.01)
        self.value_spinbox.setDecimals(3)
        self.value_spinbox.setValue(0.0)
        self.value_spinbox.setMinimumWidth(80)
        self.value_spinbox.setMaximumWidth(80)
        self.value_spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.value_spinbox)

        # 滑动条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(self.joint_range[0] * 1000), int(self.joint_range[1] * 1000))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        # 范围标签
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
        self.free_joint_id = -1
        self.free_joint_qpos_addr = -1
        self.config_file_path = os.path.join(os.path.expanduser('~'), '.mujoco_viewer_last_path.txt')

        self.setupUI()
        self._load_last_path()
        self.setWindowTitle("Mujoco模型查看器")
        self.resize(1200, 800)

    def setupUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧控制面板
        self.setupControlPanel(splitter)

        # 右侧信息面板
        self.setupInfoPanel(splitter)

        # 设置分割器比例
        splitter.setSizes([800, 400])

    def setupControlPanel(self, parent):
        """设置左侧控制面板"""
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)

        # 文件加载区域
        file_group = QGroupBox("模型文件")
        file_layout = QVBoxLayout()

        file_input_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("选择XML文件路径...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_file)
        load_btn = QPushButton("加载模型")
        load_btn.clicked.connect(self.load_model)
        load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        file_input_layout.addWidget(self.file_path_edit)
        file_input_layout.addWidget(browse_btn)
        file_input_layout.addWidget(load_btn)

        file_layout.addLayout(file_input_layout)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # 基座控制
        self.base_control_group = QGroupBox("基座控制 (XYZ + RPY)")
        self.base_control_widget = BaseControlWidget()
        self.base_control_widget.valueChanged.connect(self.on_base_control_changed)
        base_group_layout = QVBoxLayout()
        base_group_layout.addWidget(self.base_control_widget)
        self.base_control_group.setLayout(base_group_layout)
        control_layout.addWidget(self.base_control_group)
        self.base_control_group.setVisible(False) # 初始隐藏


        # 查看器控制
        viewer_group = QGroupBox("查看器控制")
        viewer_layout = QHBoxLayout()

        self.start_viewer_btn = QPushButton("启动查看器")
        self.start_viewer_btn.clicked.connect(self.start_viewer)
        self.start_viewer_btn.setEnabled(False)

        self.stop_viewer_btn = QPushButton("停止查看器")
        self.stop_viewer_btn.clicked.connect(self.stop_viewer)
        self.stop_viewer_btn.setEnabled(False)

        self.reset_btn = QPushButton("重置姿态")
        self.reset_btn.clicked.connect(self.reset_pose)
        self.reset_btn.setEnabled(False)

        viewer_layout.addWidget(self.start_viewer_btn)
        viewer_layout.addWidget(self.stop_viewer_btn)
        viewer_layout.addWidget(self.reset_btn)

        viewer_group.setLayout(viewer_layout)
        control_layout.addWidget(viewer_group)

        # 关节控制区域
        self.joint_group = QGroupBox("关节控制")
        self.joint_scroll = QScrollArea()
        self.joint_scroll.setWidgetResizable(True)
        self.joint_scroll.setMinimumHeight(400)

        self.joint_widget = QWidget()
        self.joint_layout = QVBoxLayout()
        self.joint_widget.setLayout(self.joint_layout)
        self.joint_scroll.setWidget(self.joint_widget)

        joint_group_layout = QVBoxLayout()
        joint_group_layout.addWidget(self.joint_scroll)
        self.joint_group.setLayout(joint_group_layout)

        control_layout.addWidget(self.joint_group)

        parent.addWidget(control_widget)

    def setupInfoPanel(self, parent):
        """设置右侧信息面板"""
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_widget.setLayout(info_layout)

        # 模型信息
        model_info_group = QGroupBox("模型信息")
        model_info_layout = QVBoxLayout()

        self.model_info_label = QLabel("未加载模型")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")

        model_info_layout.addWidget(self.model_info_label)
        model_info_group.setLayout(model_info_layout)
        info_layout.addWidget(model_info_group)

        # 关节信息
        joint_info_group = QGroupBox("关节信息")
        joint_info_layout = QVBoxLayout()

        self.joint_info_label = QLabel("未加载模型")
        self.joint_info_label.setWordWrap(True)
        self.joint_info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")

        joint_info_layout.addWidget(self.joint_info_label)
        joint_info_group.setLayout(joint_info_layout)
        info_layout.addWidget(joint_info_group)

        # 状态信息
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("padding: 10px; background-color: #e8f5e8; border-radius: 5px; color: #2e7d32;")

        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        info_layout.addWidget(status_group)

        parent.addWidget(info_widget)

    def _save_last_path(self, path):
        """保存最后成功加载的路径"""
        try:
            with open(self.config_file_path, 'w') as f:
                f.write(path)
        except Exception as e:
            print(f"警告: 无法保存上次路径: {e}")

    def _load_last_path(self):
        """在启动时加载并设置上次的路径"""
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r') as f:
                    path = f.read().strip()
                    if path and os.path.exists(path):
                        self.file_path_edit.setText(path)
        except Exception as e:
            print(f"警告: 无法加载上次路径: {e}")

    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Mujoco XML文件", "", "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def load_model(self):
        """加载模型"""
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择XML文件！")
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", "文件不存在！")
            return

        try:
            # 停止当前查看器
            if self.is_viewer_running:
                self.stop_viewer()

            # 加载模型
            self.model = mujoco.MjModel.from_xml_path(file_path)
            self.data = mujoco.MjData(self.model)

            self._save_last_path(file_path)

            # 更新界面
            # 先创建控件，这样可以识别出自由关节
            self.create_joint_controls()
            # 然后更新信息，此时可以正确计算关节数量
            self.update_model_info() 
            self.reset_pose() # 重置并更新所有控件

            self.start_viewer_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)

            self.status_label.setText("模型加载成功")
            self.status_label.setStyleSheet(
                "padding: 10px; background-color: #e8f5e8; border-radius: 5px; color: #2e7d32;")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")
            self.status_label.setText(f"加载失败：{str(e)}")
            self.status_label.setStyleSheet(
                "padding: 10px; background-color: #ffebee; border-radius: 5px; color: #c62828;")

    def update_model_info(self):
        """更新模型信息"""
        if self.model is None:
            return

        joint_count = self.model.njnt
        # 如果存在自由关节（基座），则在总数中减去1
        if self.free_joint_id != -1:
            joint_count -= 1

        info_text = f"""
模型名称: {getattr(self.model, 'name', '未命名')}
关节数量: {joint_count}
自由度: {self.model.nv}
刚体数量: {self.model.nbody}
几何体数量: {self.model.ngeom}
      """.strip()

        self.model_info_label.setText(info_text)

    def create_joint_controls(self):
        """创建关节控制组件"""
        # 清除现有控件
        for control in self.joint_controls:
            control.deleteLater()
        self.joint_controls.clear()

        # 清除布局中的所有项目
        while self.joint_layout.count():
            child = self.joint_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.free_joint_id = -1
        self.free_joint_qpos_addr = -1
        self.base_control_group.setVisible(False)


        if self.model is None:
            return

        joint_info_text = "关节列表:\n"

        # 为每个关节创建控制组件
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                joint_name = f"joint_{i}"

            # 获取关节类型和范围
            joint_type = self.model.jnt_type[i]
            
            # --- 基座检测和处理 ---
            if joint_type == mujoco.mjtJoint.mjJNT_FREE and self.free_joint_id == -1:
                self.free_joint_id = i
                self.free_joint_qpos_addr = self.model.jnt_qposadr[i]
                self.base_control_group.setVisible(True)
                joint_info_text += f"  - 基座: {joint_name} (XYZ, RPY)\n"
                continue # 不为基座创建普通关节控制器

            joint_range = self.model.jnt_range[i]

            # 如果关节范围为0，设置默认范围
            if joint_range[0] == joint_range[1]:
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:  # 旋转关节
                    joint_range = [-np.pi, np.pi]
                else:  # 滑动关节
                    joint_range = [-1.0, 1.0]

            joint_info_text += f"  {i}: {joint_name} (类型: {joint_type})\n"

            # 创建控制组件
            control = JointControlWidget(i, joint_name, joint_range, joint_type)
            control.valueChanged.connect(self.on_joint_value_changed)

            self.joint_controls.append(control)
            self.joint_layout.addWidget(control)

        # 添加弹簧
        self.joint_layout.addStretch()

        self.joint_info_label.setText(joint_info_text)

    def on_joint_value_changed(self, joint_id, value):
        """关节值改变时的回调"""
        if self.data is not None and self.model is not None:
            # 获取该关节在qpos数组中的起始地址
            qpos_addr = self.model.jnt_qposadr[joint_id]

            # 更新qpos中的值. 对于单自由度关节，这会直接设置正确的值。
            if qpos_addr < self.model.nq:
                self.data.qpos[qpos_addr] = value
                # 前向运动学计算，这对于更新模型中依赖此关节的其他部分至关重要
                mujoco.mj_forward(self.model, self.data)

    def on_base_control_changed(self):
        """基座控制器值改变时的回调"""
        if self.data is None or self.free_joint_id == -1:
            return

        pos, rpy = self.base_control_widget.get_values()

        # 更新位置
        self.data.qpos[self.free_joint_qpos_addr:self.free_joint_qpos_addr + 3] = pos

        # 更新姿态 (RPY -> 四元数)
        quat = euler_to_quaternion(rpy)
        self.data.qpos[self.free_joint_qpos_addr + 3:self.free_joint_qpos_addr + 7] = quat

        mujoco.mj_forward(self.model, self.data)

    def start_viewer(self):
        """启动查看器"""
        if self.model is None:
            return

        if self.is_viewer_running:
            return

        self.is_viewer_running = True
        self.viewer_thread = threading.Thread(target=self.viewer_loop, daemon=True)
        self.viewer_thread.start()

        self.start_viewer_btn.setEnabled(False)
        self.stop_viewer_btn.setEnabled(True)
        self.status_label.setText("查看器运行中...")
        self.status_label.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 5px; color: #1565c0;")

    def stop_viewer(self):
        """停止查看器"""
        self.is_viewer_running = False

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.viewer_thread is not None:
            self.viewer_thread.join(timeout=2.0)
            self.viewer_thread = None

        self.start_viewer_btn.setEnabled(True)
        self.stop_viewer_btn.setEnabled(False)
        self.status_label.setText("查看器已停止")
        self.status_label.setStyleSheet("padding: 10px; background-color: #fff3e0; border-radius: 5px; color: #e65100;")

    def viewer_loop(self):
        """查看器主循环"""
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer

                # 设置相机
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.distance = 3.0

                while self.is_viewer_running and viewer.is_running():
                    # 步进仿真
                    # mujoco.mj_step(self.model, self.data)

                    # 同步查看器
                    viewer.sync()

                    # 控制帧率
                    time.sleep(0.01)

        except Exception as e:
            print(f"查看器错误: {e}")
            self.is_viewer_running = False

        # 在主线程中更新UI
        QTimer.singleShot(0, self._on_viewer_closed)

    def _on_viewer_closed(self):
        """查看器关闭后的处理"""
        self.viewer = None
        self.is_viewer_running = False
        self.start_viewer_btn.setEnabled(True)
        self.stop_viewer_btn.setEnabled(False)
        self.status_label.setText("查看器已关闭")
        self.status_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 5px; color: #616161;")

    def reset_pose(self):
        """重置到初始姿态"""
        if self.data is None:
            return

        # 重置所有关节位置
        mujoco.mj_resetData(self.model, self.data)

        # 更新基座控件显示
        if self.free_joint_id != -1:
            self.update_base_controls_from_data()

        # 更新关节控件显示
        for control in self.joint_controls:
            joint_id = control.joint_id
            # 找到关节对应的qpos地址
            qpos_addr = self.model.jnt_qposadr[joint_id]
            if qpos_addr < len(self.data.qpos):
                control.set_value(self.data.qpos[qpos_addr])

        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)

        self.status_label.setText("姿态已重置")
        self.status_label.setStyleSheet("padding: 10px; background-color: #e8f5e8; border-radius: 5px; color: #2e7d32;")

    def update_base_controls_from_data(self):
        """从Mujoco数据更新基座UI控件"""
        if self.data is None or self.free_joint_id == -1:
            return

        # 获取位置
        pos = self.data.qpos[self.free_joint_qpos_addr:self.free_joint_qpos_addr + 3]

        # 获取姿态 (四元数 -> RPY)
        quat = self.data.qpos[self.free_joint_qpos_addr + 3:self.free_joint_qpos_addr + 7]
        rpy = quaternion_to_euler(quat)

        self.base_control_widget.set_values(pos, rpy)


    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.is_viewer_running:
            self.stop_viewer()
        event.accept()

class ModernStyle:
    """现代化的界面样式"""

    @staticmethod
    def apply(app):
        app.setStyle("Fusion")

        # 创建调色板
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

        # 设置样式表
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
        """)

def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 应用现代化样式
    ModernStyle.apply(app)

    # 设置应用程序信息
    app.setApplicationName("Mujoco模型查看器")
    app.setOrganizationName("MujocoViewer")

    # 创建主窗口
    viewer = MujocoViewer()
    viewer.show()

    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
