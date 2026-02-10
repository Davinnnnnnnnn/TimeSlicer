import sys
import os
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QFileDialog, QLabel, QSlider, QGroupBox, QComboBox, 
                             QColorDialog, QSplitter, QCheckBox, QGraphicsView, 
                             QGraphicsScene, QGraphicsLineItem, QGraphicsEllipseItem, 
                             QMenu, QInputDialog, QAbstractItemView, 
                             QProgressBar, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter

# ---- [Core Logic] 마스크 계산 ----
def calculate_mask_weights(h, w, n_files, params):
    # 메모리 절약을 위해 float32 사용
    y_indices, x_indices = np.mgrid[0:h, 0:w].astype(np.float32)
    
    cx = w * params['center_x']
    cy = h * params['center_y']
    
    pattern = params['pattern']
    reverse = params['reverse'] 
    val_map = None 
    
    if pattern.startswith("Linear"):
        angle = params['angle']
        margin_s = params['margin_s']
        margin_e = params['margin_e']

        rad = np.radians(angle)
        vec_x = np.cos(rad)
        vec_y = np.sin(rad)
        
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
        projs = corners[:, 0] * vec_x + corners[:, 1] * vec_y
        min_p = np.min(projs)
        max_p = np.max(projs)
        total_len = max_p - min_p
        if total_len == 0: total_len = 1.0
        
        curr_proj = x_indices * vec_x + y_indices * vec_y
        
        effective_len = total_len * (1.0 - margin_s - margin_e)
        if effective_len <= 0: effective_len = 1.0
        
        start_val = min_p + (total_len * margin_s)
        
        norm_val = (curr_proj - start_val) / effective_len
        if reverse:
            norm_val = 1.0 - norm_val
            
        val_map = norm_val * n_files
        
    elif pattern == "Radial":
        angle_offset = params['angle'] 
        
        theta = np.arctan2(y_indices - cy, x_indices - cx)
        
        if reverse:
            theta = -theta
            
        rad_offset = np.radians(angle_offset)
        theta = np.mod(theta - rad_offset, 2 * np.pi)
        
        val_map = (theta / (2 * np.pi)) * n_files
        
    elif pattern == "Concentric":
        dist = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
        dists_corners = np.sqrt((corners[:,0]-cx)**2 + (corners[:,1]-cy)**2)
        max_dist = np.max(dists_corners)
        if max_dist == 0: max_dist = 1
        
        norm_dist = dist / max_dist
        
        # Reverse가 True면: 밖에서 안으로
        if reverse:
            norm_dist = 1.0 - norm_dist
            
        val_map = norm_dist * n_files

    return val_map.astype(np.float32)


# ---- [Worker] 싱글 스레드 + 알고리즘 최적화 ----
class TimeSliceWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, file_paths, output_path, params):
        super().__init__()
        self.file_paths = file_paths
        self.output_path = output_path
        self.params = params
        self.is_running = True

    def run(self):
        n_files = len(self.file_paths)
        if n_files == 0: return

        try:
            stream = open(self.file_paths[0], "rb")
            bytes_data = bytearray(stream.read())
            stream.close()
            ref_img = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            if ref_img is None:
                self.error_signal.emit("Failed to load first image.")
                return
            
            h, w = ref_img.shape[:2]
            
            val_map = calculate_mask_weights(h, w, n_files, self.params)
            
            accum_img = np.zeros((h, w, 3), dtype=np.float32)
            accum_weight = np.zeros((h, w, 1), dtype=np.float32)
            
            bw = max(0.001, self.params['blend_val'] / 50.0)
            half_width = 0.5
            dist_limit_outer = half_width + bw / 2.0
            blend_alpha = 1.0 - self.params['blend_func']

            for i, path in enumerate(self.file_paths):
                if not self.is_running: break
                
                center_val = i + 0.5
                is_first = (i == 0)
                is_last = (i == n_files - 1)
                
                min_v = center_val - dist_limit_outer
                max_v = center_val + dist_limit_outer
                
                if is_first and is_last:
                    mask_roi = np.ones((h, w), dtype=bool)
                elif is_first:
                    mask_roi = val_map <= max_v
                elif is_last:
                    mask_roi = val_map >= min_v
                else:
                    mask_roi = (val_map >= min_v) & (val_map <= max_v)
                
                if not np.any(mask_roi):
                    self.progress_signal.emit(int((i+1)/n_files*100))
                    continue

                mask_u8 = mask_roi.astype(np.uint8)
                x, y, rw, rh = cv2.boundingRect(mask_u8)
                
                if rw == 0 or rh == 0:
                    self.progress_signal.emit(int((i+1)/n_files*100))
                    continue
                    
                y0, y1 = y, y + rh
                x0, x1 = x, x + rw
                
                sub_val_map = val_map[y0:y1, x0:x1]
                sub_mask = mask_roi[y0:y1, x0:x1]
                
                sub_dist = np.abs(sub_val_map - center_val)
                
                if is_first:
                    sub_dist[sub_val_map < center_val] = 0
                if is_last:
                    sub_dist[sub_val_map > center_val] = 0
                    
                sub_weight = (dist_limit_outer - sub_dist) / bw
                sub_weight = np.clip(sub_weight, 0, 1)
                sub_weight[~sub_mask] = 0
                
                if blend_alpha > 0:
                    trig_weight = np.sin(sub_weight * np.pi / 2) ** 2
                    sub_weight = sub_weight * (1 - blend_alpha) + trig_weight * blend_alpha
                    
                sub_w_3d = sub_weight[:, :, np.newaxis]

                stream = open(path, "rb")
                bytes_data = bytearray(stream.read())
                stream.close()
                np_arr = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                
                if img is None: continue
                
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if img.shape[:2] != (h, w): img = cv2.resize(img, (w, h))
                
                sub_img = img[y0:y1, x0:x1].astype(np.float32)
                
                accum_img[y0:y1, x0:x1] += sub_img * sub_w_3d
                accum_weight[y0:y1, x0:x1] += sub_w_3d
                
                self.progress_signal.emit(int((i+1)/n_files*100))

            if not self.is_running: return

            accum_weight[accum_weight == 0] = 1.0
            final_img = accum_img / accum_weight
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)

            is_success, im_buf = cv2.imencode(".tif", final_img)
            if is_success:
                im_buf.tofile(self.output_path)
                self.finished_signal.emit(f"Saved: {self.output_path}")
            else:
                self.error_signal.emit("Failed to encode image.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Error: {str(e)}")


# ---- [Preview Worker] ----
class PreviewWorker(QThread):
    result_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int) 

    def __init__(self, file_paths, params):
        super().__init__()
        self.file_paths = file_paths
        self.params = params

    def run(self):
        n_files = len(self.file_paths)
        if n_files == 0: return

        preview_imgs = []
        target_w = 640
        h, w = 0, 0
        
        try:
            stream = open(self.file_paths[0], "rb")
            bytes_data = bytearray(stream.read())
            first = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            stream.close()
        except: return

        if first is None: return
        h_orig, w_orig = first.shape[:2]
        scale = target_w / w_orig
        h, w = int(h_orig * scale), int(w_orig * scale)
        if h <= 0 or w <= 0: return
        
        for idx, path in enumerate(self.file_paths):
            try:
                stream = open(path, "rb")
                bytes_data = bytearray(stream.read())
                img = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                stream.close()
                if img is None: 
                    preview_imgs.append(np.zeros((h, w, 3), dtype=np.float32))
                else:
                    img = cv2.resize(img, (w, h))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    preview_imgs.append(img.astype(np.float32))
            except:
                preview_imgs.append(np.zeros((h, w, 3), dtype=np.float32))
            
            if idx % 10 == 0:
                self.progress_signal.emit(int((idx / n_files) * 50))

        accum_img = np.zeros((h, w, 3), dtype=np.float32)
        accum_weight = np.zeros((h, w, 1), dtype=np.float32)
        
        val_map = calculate_mask_weights(h, w, n_files, self.params)
        
        bw = max(0.001, self.params['blend_val'] / 50.0)
        half_width = 0.5
        dist_limit_outer = half_width + bw / 2.0
        blend_alpha = 1.0 - self.params['blend_func']

        for i in range(n_files):
            center_val = i + 0.5
            is_first = (i == 0)
            is_last = (i == n_files - 1)
            
            if is_first:
                dist = val_map - center_val
                dist_abs = np.abs(dist)
                dist_abs[val_map < center_val] = 0
            elif is_last:
                dist = val_map - center_val
                dist_abs = np.abs(dist)
                dist_abs[val_map > center_val] = 0
            else:
                dist_abs = np.abs(val_map - center_val)
            
            if np.min(dist_abs) > dist_limit_outer: 
                if i % 5 == 0:
                    self.progress_signal.emit(50 + int((i / n_files) * 50))
                continue

            weight = (dist_limit_outer - dist_abs) / bw
            weight = np.clip(weight, 0, 1)
            
            if blend_alpha > 0:
                trig_weight = np.sin(weight * np.pi / 2) ** 2
                weight = weight * (1 - blend_alpha) + trig_weight * blend_alpha
            
            weight = weight[:, :, np.newaxis]
            accum_img += preview_imgs[i] * weight
            accum_weight += weight
            
            if i % 5 == 0:
                self.progress_signal.emit(50 + int((i / n_files) * 50))

        accum_weight[accum_weight == 0] = 1.0
        final_img = accum_img / accum_weight
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        q_img = QImage(final_img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.progress_signal.emit(100)
        self.result_signal.emit(q_img.copy()) 


# ---- [Custom View] 클릭 가능한 뷰 ----
class InteractiveGraphicsView(QGraphicsView):
    clicked_pos = pyqtSignal(float, float)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            scene_rect = self.sceneRect()
            if scene_rect.width() > 0 and scene_rect.height() > 0:
                x_ratio = (scene_pos.x() - scene_rect.left()) / scene_rect.width()
                y_ratio = (scene_pos.y() - scene_rect.top()) / scene_rect.height()
                x_ratio = max(0.0, min(1.0, x_ratio))
                y_ratio = max(0.0, min(1.0, y_ratio))
                self.clicked_pos.emit(x_ratio, y_ratio)
        super().mousePressEvent(event)


class TimeSliceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TimeSlicer")
        self.resize(1280, 950)
        self.file_list = [] 
        self.line_color = QColor(0, 255, 255) 
        self.is_bulk_updating = False
        self.current_preview_pixmap = None 
        self.result_preview_pixmap = None  
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ---- LEFT PANEL ----
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.open_folder)
        left_layout.addWidget(self.btn_open_folder)

        # Selection Tools
        btn_select_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("All")
        self.btn_unselect_all = QPushButton("None")
        self.btn_select_opts = QPushButton("Auto...") 
        
        self.btn_select_all.clicked.connect(self.select_all_files)
        self.btn_unselect_all.clicked.connect(self.unselect_all_files)
        
        self.select_menu = QMenu()
        self.select_menu.addAction("Every N-th File", self.select_every_n)
        self.select_menu.addAction("Total N Files (Evenly)", self.select_total_n)
        self.btn_select_opts.setMenu(self.select_menu)

        btn_select_layout.addWidget(self.btn_select_all)
        btn_select_layout.addWidget(self.btn_unselect_all)
        btn_select_layout.addWidget(self.btn_select_opts)
        left_layout.addLayout(btn_select_layout)

        self.btn_invert_chk = QPushButton("Invert Checks (Selected Range)")
        self.btn_invert_chk.setStyleSheet("background-color: #555555; color: white;")
        self.btn_invert_chk.clicked.connect(self.invert_selection_checks)
        left_layout.addWidget(self.btn_invert_chk)

        self.lbl_count = QLabel("Selected: 0 files")
        self.lbl_count.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_count)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemClicked.connect(self.on_file_clicked)
        self.list_widget.itemChanged.connect(self.on_item_changed)
        left_layout.addWidget(self.list_widget)

        # ---- RIGHT PANEL ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.scene = QGraphicsScene()
        
        # Interactive View
        self.view = InteractiveGraphicsView(self.scene)
        self.view.clicked_pos.connect(self.on_view_clicked)
        
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_layout.addWidget(self.view, stretch=4)

        # CONTROLS
        control_group = QGroupBox("Configuration")
        c_layout = QVBoxLayout()
        
        # Preview Controls
        row_prev = QHBoxLayout()
        self.chk_simple = QCheckBox("View Less Lines (Max 50)")
        self.chk_simple.setChecked(True)
        self.chk_simple.stateChanged.connect(self.update_preview_overlay)
        
        self.btn_gen_preview = QPushButton("Show Output Preview")
        self.btn_gen_preview.setCheckable(True)
        self.btn_gen_preview.clicked.connect(self.toggle_result_preview)
        
        self.btn_refresh_preview = QPushButton("Refresh") 
        self.btn_refresh_preview.clicked.connect(self.run_preview_generation)
        
        row_prev.addWidget(self.chk_simple)
        row_prev.addWidget(self.btn_gen_preview)
        row_prev.addWidget(self.btn_refresh_preview)
        c_layout.addLayout(row_prev)

        # Pattern & Color & Direction (체크박스 위치 이동!)
        row_pat = QHBoxLayout()
        row_pat.addWidget(QLabel("Pattern:"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(["Linear", "Radial", "Concentric"])
        self.combo_pattern.currentIndexChanged.connect(self.on_pattern_changed)
        row_pat.addWidget(self.combo_pattern)
        
        self.btn_color = QPushButton("Guide Color")
        self.btn_color.clicked.connect(self.pick_color)
        row_pat.addWidget(self.btn_color)
        
        # [수정] 체크박스를 항상 보이는 이 곳(Pattern 줄)으로 이동
        self.chk_reverse = QCheckBox("Reverse Direction")
        self.chk_reverse.stateChanged.connect(self.update_preview_overlay)
        row_pat.addWidget(self.chk_reverse)
        
        c_layout.addLayout(row_pat)

        # Center Position
        self.grp_center = QGroupBox("Center Position (Click Image to Set)")
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("X:"))
        self.sl_cx = QSlider(Qt.Horizontal)
        self.sl_cx.setRange(0, 100)
        self.sl_cx.setValue(50)
        self.sl_cx.valueChanged.connect(self.update_preview_overlay)
        center_layout.addWidget(self.sl_cx)
        
        center_layout.addWidget(QLabel("Y:"))
        self.sl_cy = QSlider(Qt.Horizontal)
        self.sl_cy.setRange(0, 100)
        self.sl_cy.setValue(50)
        self.sl_cy.valueChanged.connect(self.update_preview_overlay)
        center_layout.addWidget(self.sl_cy)
        self.grp_center.setLayout(center_layout)
        c_layout.addWidget(self.grp_center)

        # Rotation
        self.grp_angle = QGroupBox("Rotation")
        row_rot = QHBoxLayout()
        self.lbl_rot = QLabel("Angle:")
        row_rot.addWidget(self.lbl_rot)
        self.slider_angle = QSlider(Qt.Horizontal)
        self.slider_angle.setRange(-180, 180)
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-180, 180)
        self.slider_angle.valueChanged.connect(lambda v: (self.spin_angle.setValue(v), self.update_preview_overlay()))
        self.spin_angle.valueChanged.connect(lambda v: (self.slider_angle.setValue(v), self.update_preview_overlay()))
        row_rot.addWidget(self.slider_angle)
        row_rot.addWidget(self.spin_angle)
        self.grp_angle.setLayout(row_rot)
        c_layout.addWidget(self.grp_angle)

        # Linear Margins
        self.grp_margin = QGroupBox("Linear Margins")
        m_layout = QVBoxLayout()
        row_m1 = QHBoxLayout()
        row_m1.addWidget(QLabel("Start:"))
        self.sl_margin_s = QSlider(Qt.Horizontal)
        self.sl_margin_s.setRange(0, 45) 
        self.sl_margin_s.valueChanged.connect(self.update_preview_overlay)
        row_m1.addWidget(self.sl_margin_s)
        
        row_m1.addWidget(QLabel("End:"))
        self.sl_margin_e = QSlider(Qt.Horizontal)
        self.sl_margin_e.setRange(0, 45)
        self.sl_margin_e.valueChanged.connect(self.update_preview_overlay)
        row_m1.addWidget(self.sl_margin_e)
        m_layout.addLayout(row_m1)
        self.grp_margin.setLayout(m_layout)
        c_layout.addWidget(self.grp_margin)

        # Blending
        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Blend Amount:"))
        self.sl_blend = QSlider(Qt.Horizontal)
        self.sl_blend.setRange(0, 100)
        self.sl_blend.valueChanged.connect(self.update_preview_overlay)
        row_b.addWidget(self.sl_blend)
        
        row_b.addWidget(QLabel("Blend Linearity:"))
        self.sl_blend_func = QSlider(Qt.Horizontal)
        self.sl_blend_func.setRange(0, 100) 
        self.sl_blend_func.setValue(100)
        row_b.addWidget(self.sl_blend_func)
        c_layout.addLayout(row_b)

        control_group.setLayout(c_layout)
        right_layout.addWidget(control_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        self.btn_generate = QPushButton("Generate High-Res TIFF")
        self.btn_generate.setFixedHeight(50)
        self.btn_generate.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #0078D7; color: white;")
        self.btn_generate.clicked.connect(self.start_generation)
        right_layout.addWidget(self.btn_generate)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 980])
        main_layout.addWidget(splitter)
        
        self.on_pattern_changed(0) 

    def on_view_clicked(self, rx, ry):
        mode = self.combo_pattern.currentText()
        if mode == "Linear": return
        val_x = int(rx * 100)
        val_y = int(ry * 100)
        self.sl_cx.setValue(val_x)
        self.sl_cy.setValue(val_y)

    def resizeEvent(self, event):
        if self.current_preview_pixmap or self.result_preview_pixmap:
             self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def on_pattern_changed(self, index):
        mode = self.combo_pattern.currentText()
        is_linear = mode == "Linear"
        is_radial = mode == "Radial"
        is_concentric = mode == "Concentric"
        
        self.grp_margin.setVisible(is_linear)
        self.grp_angle.setVisible(is_linear or is_radial)
        self.grp_center.setVisible(is_radial or is_concentric)
        
        if is_radial:
            self.chk_reverse.setText("Clockwise")
            self.chk_reverse.setVisible(True)
        elif is_concentric:
            self.chk_reverse.setText("Start from Outside") 
            self.chk_reverse.setVisible(True)
        elif is_linear:
            self.chk_reverse.setText("Reverse Order")
            self.chk_reverse.setVisible(True)
        
        self.update_preview_overlay()

    def invert_selection_checks(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items: return
        
        self.list_widget.blockSignals(True) 
        for item in selected_items:
            current = item.checkState()
            item.setCheckState(Qt.Unchecked if current == Qt.Checked else Qt.Checked)
        self.list_widget.blockSignals(False)
        
        self.update_count_label()
        self.update_preview_overlay()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder: self.load_files(folder)

    def load_files(self, folder):
        self.list_widget.clear()
        self.file_list = []
        exts = ('.jpg', '.jpeg', '.tif', '.tiff', '.png')
        try:
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
        except: return

        for f in files:
            path = os.path.join(folder, f)
            self.file_list.append(path)
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)
            
        if self.file_list:
            self.list_widget.setCurrentRow(0)
            self.on_file_clicked(self.list_widget.item(0))
        self.update_count_label()

    def on_item_changed(self, item):
        if not self.is_bulk_updating:
            self.update_count_label()
            self.update_preview_overlay()

    def update_count_label(self):
        count = self.get_checked_count()
        self.lbl_count.setText(f"Selected: {count} files")

    def get_checked_items(self):
        return [self.list_widget.item(i).data(Qt.UserRole) 
                for i in range(self.list_widget.count()) 
                if self.list_widget.item(i).checkState() == Qt.Checked]

    def get_checked_count(self):
        return len(self.get_checked_items())
    
    def set_all(self, state):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)
        self.list_widget.blockSignals(False)
        self.update_count_label()
        self.update_preview_overlay()

    def select_all_files(self): self.set_all(Qt.Checked)
    def unselect_all_files(self): self.set_all(Qt.Unchecked)
    
    def select_every_n(self):
        n, ok = QInputDialog.getInt(self, "Interval", "N:", 2, 1, 1000)
        if ok:
            self.list_widget.blockSignals(True)
            for i in range(self.list_widget.count()):
                self.list_widget.item(i).setCheckState(Qt.Checked if i % n == 0 else Qt.Unchecked)
            self.list_widget.blockSignals(False)
            self.update_count_label()
            self.update_preview_overlay()

    def select_total_n(self):
        total = self.list_widget.count()
        n, ok = QInputDialog.getInt(self, "Total Count", "N:", min(10, total), 2, total)
        if ok:
            indices = set(np.linspace(0, total - 1, n).round().astype(int))
            self.list_widget.blockSignals(True)
            for i in range(total):
                self.list_widget.item(i).setCheckState(Qt.Checked if i in indices else Qt.Unchecked)
            self.list_widget.blockSignals(False)
            self.update_count_label()
            self.update_preview_overlay()

    def on_file_clicked(self, item):
        path = item.data(Qt.UserRole)
        self.btn_gen_preview.setChecked(False)
        
        try:
            stream = open(path, "rb")
            bytes_data = bytearray(stream.read())
            img = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            stream.close()

            if img is not None:
                if img.dtype == np.uint16: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else: img = img.astype(np.uint8)
                if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                h, w, ch = img.shape
                q_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
                self.current_preview_pixmap = QPixmap.fromImage(q_img)
                
                self.display_current_mode()
        except: pass

    def toggle_result_preview(self):
        if self.btn_gen_preview.isChecked():
            if self.result_preview_pixmap is None:
                self.run_preview_generation()
            else:
                self.display_current_mode()
        else:
            self.display_current_mode()

    def run_preview_generation(self):
        if not self.btn_gen_preview.isChecked():
            self.btn_gen_preview.setChecked(True)
            
        checked = self.get_checked_items()
        if not checked: return
        
        params = self.get_current_params()
        
        self.btn_refresh_preview.setText("Generating...")
        self.btn_refresh_preview.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.prev_worker = PreviewWorker(checked, params)
        self.prev_worker.result_signal.connect(self.on_preview_generated)
        self.prev_worker.progress_signal.connect(self.progress_bar.setValue) 
        self.prev_worker.start()

    def on_preview_generated(self, q_img):
        self.result_preview_pixmap = QPixmap.fromImage(q_img)
        self.btn_refresh_preview.setText("Refresh")
        self.btn_refresh_preview.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.display_current_mode()

    def display_current_mode(self):
        self.scene.clear()
        target_pixmap = None
        
        if self.btn_gen_preview.isChecked() and self.result_preview_pixmap:
            target_pixmap = self.result_preview_pixmap
        elif self.current_preview_pixmap:
            target_pixmap = self.current_preview_pixmap

        if target_pixmap:
            self.scene.addPixmap(target_pixmap)
            self.scene.setSceneRect(QRectF(target_pixmap.rect()))
            
            if not self.btn_gen_preview.isChecked():
                self.update_preview_overlay()
        
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.line_color = c
            self.update_preview_overlay()

    def get_current_params(self):
        return {
            'pattern': self.combo_pattern.currentText(),
            'angle': self.slider_angle.value(),
            'reverse': self.chk_reverse.isChecked(),
            'margin_s': self.sl_margin_s.value() / 100.0,
            'margin_e': self.sl_margin_e.value() / 100.0,
            'blend_val': self.sl_blend.value(),
            'blend_func': self.sl_blend_func.value() / 100.0,
            'center_x': self.sl_cx.value() / 100.0,
            'center_y': self.sl_cy.value() / 100.0,
        }

    def update_preview_overlay(self):
        if self.btn_gen_preview.isChecked(): return

        for item in list(self.scene.items()):
            if isinstance(item, (QGraphicsLineItem, QGraphicsEllipseItem)):
                self.scene.removeItem(item)

        n_slices = self.get_checked_count()
        if n_slices < 2: return

        rect = self.scene.sceneRect()
        w, h = rect.width(), rect.height()
        if w == 0: return

        params = self.get_current_params()
        
        cx = w * params['center_x']
        cy = h * params['center_y']
        reverse = params['reverse']
        
        pen = QPen(self.line_color, 2)
        
        inv_r = 255 - self.line_color.red()
        inv_g = 255 - self.line_color.green()
        inv_b = 255 - self.line_color.blue()
        start_pen = QPen(QColor(inv_r, inv_g, inv_b), 4) 

        pattern = params['pattern']
        
        start_line_item = None

        if self.chk_simple.isChecked() and n_slices > 50:
            indices = np.linspace(0, n_slices, 50).round().astype(int)
        else:
            indices = range(n_slices + 1)

        if pattern.startswith("Linear"):
            rad = math.radians(params['angle'])
            vec_x = math.cos(rad)
            vec_y = math.sin(rad)
            
            corners = [(0,0), (w,0), (0,h), (w,h)]
            projs = [x*vec_x + y*vec_y for x, y in corners]
            min_p = min(projs)
            max_p = max(projs)
            total_len = max_p - min_p
            
            margin_s = params['margin_s']
            margin_e = params['margin_e']
            
            effective_len = total_len * (1.0 - margin_s - margin_e)
            start_val = min_p + total_len * margin_s
            
            ortho_x = -vec_y
            ortho_y = vec_x
            line_len = max(w, h) * 2
            
            img_cx, img_cy = w/2, h/2
            c_proj = img_cx * vec_x + img_cy * vec_y

            for i in indices:
                idx_norm = i / n_slices
                if reverse: idx_norm = 1.0 - idx_norm
                
                curr_p = start_val + idx_norm * effective_len
                
                diff = curr_p - c_proj
                p_base_x = img_cx + diff * vec_x
                p_base_y = img_cy + diff * vec_y
                
                x1 = p_base_x - line_len * ortho_x
                y1 = p_base_y - line_len * ortho_y
                x2 = p_base_x + line_len * ortho_x
                y2 = p_base_y + line_len * ortho_y
                
                line_item = QGraphicsLineItem(x1, y1, x2, y2)
                
                if i == 0: 
                    start_line_item = line_item
                    start_line_item.setPen(start_pen)
                else:
                    line_item.setPen(pen)
                    self.scene.addItem(line_item)

        elif pattern == "Radial":
            r_len = max(w, h) * 1.5
            angle_offset = params['angle']
            
            for i in indices:
                ratio = i / n_slices
                deg = ratio * 360
                if reverse: deg = -deg
                deg += angle_offset
                theta = math.radians(deg)
                
                x2 = cx + r_len * math.cos(theta)
                y2 = cy + r_len * math.sin(theta)
                
                line_item = QGraphicsLineItem(cx, cy, x2, y2)
                
                if i == 0:
                    start_line_item = line_item
                    start_line_item.setPen(start_pen)
                else:
                    line_item.setPen(pen)
                    self.scene.addItem(line_item)
                
        elif pattern == "Concentric":
            corners = [(0,0), (w,0), (0,h), (w,h)]
            dists = [math.sqrt((x-cx)**2 + (y-cy)**2) for x, y in corners]
            max_r = max(dists)
            if max_r == 0: max_r = 1
            
            for i in indices:
                ratio = i / n_slices
                if reverse: ratio = 1.0 - ratio
                
                r = ratio * max_r
                ellipse_item = QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
                
                if i == 0:
                    ellipse_item.setPen(start_pen)
                    start_line_item = ellipse_item
                else:
                    ellipse_item.setPen(pen)
                    self.scene.addItem(ellipse_item)

        if start_line_item:
            self.scene.addItem(start_line_item)

    def start_generation(self):
        checked = self.get_checked_items()
        if not checked: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save", "", "TIFF (*.tif)")
        if not path: return
        if not path.lower().endswith('.tif'): path += '.tif'

        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        params = self.get_current_params()
        
        self.worker = TimeSliceWorker(checked, path, params)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(lambda m: (self.btn_generate.setEnabled(True), self.progress_bar.setVisible(False), QMessageBox.information(self, "Done", m)))
        self.worker.error_signal.connect(lambda m: (self.btn_generate.setEnabled(True), QMessageBox.critical(self, "Error", m)))
        self.worker.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TimeSliceApp()
    window.show()
    sys.exit(app.exec_())