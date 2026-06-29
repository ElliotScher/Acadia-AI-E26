import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QCoreApplication, Slot, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QListWidget, QFileDialog, QSplitter,
    QScrollArea, QGroupBox, QMessageBox, QProgressDialog, QFrame, QFormLayout,
    QTabWidget, QComboBox
)

# Add project root to python path to allow imports from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.processing.entityprofiling import (
    EntityFeatureExtractor,
    ProfileDatabase,
    detect_entities
)

DARK_STYLESHEET = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QFrame#sidebar {
    background-color: #1a1a1a;
    border-right: 1px solid #333333;
}
QListWidget {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 5px;
    color: #ffffff;
}
QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #2a2a2a;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #007acc;
    color: white;
}
QListWidget::item:hover {
    background-color: #2a2a2a;
}
QGroupBox {
    border: 1px solid #333333;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
    color: #007acc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
QPushButton {
    background-color: #333333;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 8px 15px;
    color: #ffffff;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #444444;
    border-color: #007acc;
}
QPushButton:pressed {
    background-color: #007acc;
}
QPushButton:disabled {
    background-color: #222222;
    border-color: #2c2c2c;
    color: #666666;
}
QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
}
QLineEdit:focus {
    border-color: #007acc;
}
QLabel {
    color: #e0e0e0;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #444444;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #555555;
}
"""

class ZoomTooltip(QWidget):
    """
    A floating preview window that displays a zoomed-in
    image of the hovered entity ID, staying visible until closed.
    """
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: #121212; border: 2px solid #555; border-radius: 4px;")
        self.resize(250, 250)
        
        # Inner layout and label to hold the image resizably
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
        self.original_pixmap = None
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def set_original_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            w = max(50, self.label.width())
            h = max(50, self.label.height())
            scaled = self.original_pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.label.setPixmap(scaled)
        else:
            self.label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        self.update_scaled_pixmap()
        super().resizeEvent(event)


class HoverThumbnailLabel(QLabel):
    """
    A thumbnail label that shows a small crop of a registered entity,
    and displays/updates a larger zoomed-in view using the ZoomTooltip.
    """
    def __init__(self, entity_id, thumbs_dir, zoom_tooltip, parent=None):
        super().__init__(parent)
        self.entity_id = entity_id
        self.thumbs_dir = thumbs_dir
        self.zoom_tooltip = zoom_tooltip
        self.setFixedSize(45, 45)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #222; border: 1px solid #444; border-radius: 2px;")
        
        self.thumb_path = self.thumbs_dir / f"id_{entity_id}.jpg"
        import sys
        print(f"[DEBUG] HoverThumbnailLabel: id={entity_id}, path={self.thumb_path}, exists={self.thumb_path.exists()}", file=sys.stderr)
        if self.thumb_path.exists():
            pix = QPixmap(str(self.thumb_path))
            print(f"[DEBUG] QPixmap load of {self.thumb_path.name}: isNull={pix.isNull()}, size={pix.width()}x{pix.height()}", file=sys.stderr)
            self.pixmap_small = pix.scaled(45, 45, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(self.pixmap_small)
        else:
            self.setPixmap(QPixmap())
            self.setText(f"#{entity_id}")
            
    def enterEvent(self, event):
        if self.thumb_path.exists():
            pix = QPixmap(str(self.thumb_path))
            if not pix.isNull():
                self.zoom_tooltip.set_original_pixmap(pix)
                # Position near cursor
                pos = self.mapToGlobal(self.rect().topRight())
                self.zoom_tooltip.move(pos.x() + 15, pos.y() - 100)
                self.zoom_tooltip.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.zoom_tooltip.hide()
        super().leaveEvent(event)


class ImageWidget(QWidget):
    """
    Custom widget to display the camera frame and draw green bounding boxes.
    Allows highlighting of the active box, selecting boxes via mouse click.
    All coordinates are normalized (0.0 to 1.0) so they remain robust
    across images with different resolutions.
    """
    entityClicked = Signal(int)  # emits the box index
    zonesChanged = Signal(object)  # emits the list of zones [{'type': 'exclude'/'include', 'rect': (nx, ny, nw, nh)}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap = None
        self.boxes = []
        self.labels = {}
        self.active_idx = -1
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # List of zones: list of dict {'type': 'exclude'|'include', 'rect': (nx, ny, nw, nh)}
        self.zones = []
        self.active_zone_idx = -1
        
        self.drag_mode = None  # None, 'move', 'resize_tl', 'resize_tr', 'resize_bl', 'resize_br', 'create'
        self.drag_start_pos = (0, 0)
        self.drag_start_rect = (0, 0, 0, 0)
        
        self.is_draw_mode = None  # None, 'exclude', or 'include'
        self.setMouseTracking(True)  # track mouse for hover cursor changes

    def set_image(self, image_path, boxes, labels, active_idx):
        self.image_path = image_path
        self.pixmap = QPixmap(str(image_path))
        self.boxes = boxes
        self.labels = labels
        self.active_idx = active_idx
        self.update()

    def set_zones(self, zones):
        self.zones = zones
        self.active_zone_idx = -1
        self.update()

    def set_draw_mode(self, mode):
        self.is_draw_mode = mode
        if mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def map_to_orig(self, pos):
        if not self.pixmap:
            return 0, 0
        mx, my = pos.x(), pos.y()
        orig_x = (mx - self.offset_x) / self.scale
        orig_y = (my - self.offset_y) / self.scale
        # Clip to image boundaries
        orig_x = max(0, min(orig_x, self.pixmap.width()))
        orig_y = max(0, min(orig_y, self.pixmap.height()))
        return int(orig_x), int(orig_y)

    def get_drag_mode_for_zone(self, pos, zone):
        nx, ny, nw, nh = zone['rect']
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        
        ex = nx * img_w
        ey = ny * img_h
        ew = nw * img_w
        eh = nh * img_h
        
        sx = self.offset_x + ex * self.scale
        sy = self.offset_y + ey * self.scale
        sw = ew * self.scale
        sh = eh * self.scale
        
        mx, my = pos.x(), pos.y()
        threshold = 12  # Interaction handle area in screen pixels
        
        # TL corner check
        if abs(mx - sx) <= threshold and abs(my - sy) <= threshold:
            return 'resize_tl'
        # TR corner check
        elif abs(mx - (sx + sw)) <= threshold and abs(my - sy) <= threshold:
            return 'resize_tr'
        # BL corner check
        elif abs(mx - sx) <= threshold and abs(my - (sy + sh)) <= threshold:
            return 'resize_bl'
        # BR corner check
        elif abs(mx - (sx + sw)) <= threshold and abs(my - (sy + sh)) <= threshold:
            return 'resize_br'
            
        # Inside rect check
        if sx <= mx <= sx + sw and sy <= my <= sy + sh:
            return 'move'
            
        return None

    def find_active_zone(self, pos):
        if not self.pixmap:
            return -1, None
            
        # Check resize handles first (higher priority than overlapping moves)
        for i, zone in enumerate(self.zones):
            mode = self.get_drag_mode_for_zone(pos, zone)
            if mode and mode.startswith('resize'):
                return i, mode
                
        # Check move areas
        for i, zone in enumerate(self.zones):
            mode = self.get_drag_mode_for_zone(pos, zone)
            if mode == 'move':
                return i, mode
                
        return -1, None

    def paintEvent(self, event):
        if not self.pixmap or self.pixmap.isNull():
            painter = QPainter(self)
            painter.fillRect(event.rect(), QColor("#1e1e1e"))
            painter.setPen(QColor("#888888"))
            painter.drawText(event.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw black background
        painter.fillRect(event.rect(), QColor("#121212"))

        # Scale image to fit widget maintaining aspect ratio
        widget_w, widget_h = self.width(), self.height()
        pix_w, pix_h = self.pixmap.width(), self.pixmap.height()

        scale_w = widget_w / pix_w
        scale_h = widget_h / pix_h
        self.scale = min(scale_w, scale_h)

        new_w = int(pix_w * self.scale)
        new_h = int(pix_h * self.scale)

        self.offset_x = (widget_w - new_w) // 2
        self.offset_y = (widget_h - new_h) // 2

        # Draw scaled image
        painter.drawPixmap(self.offset_x, self.offset_y, new_w, new_h, self.pixmap)

        # Draw detected bounding boxes
        for i, (bx, by, bw, bh) in enumerate(self.boxes):
            # Scale coordinates
            x = int(self.offset_x + bx * self.scale)
            y = int(self.offset_y + by * self.scale)
            w = int(bw * self.scale)
            h = int(bh * self.scale)

            # Determine box styling
            if i == self.active_idx:
                pen = QPen(QColor(255, 152, 0), 4, Qt.PenStyle.DashLine)
            elif i in self.labels:
                pen = QPen(QColor(0, 122, 204), 3, Qt.PenStyle.SolidLine)
            else:
                pen = QPen(QColor(76, 175, 80), 2, Qt.PenStyle.SolidLine)

            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x, y, w, h)

            # Draw labels above boxes
            label_text = f"ID: {self.labels[i]}" if i in self.labels else f"Box {i+1}"
            if i == self.active_idx:
                label_text += " [ACTIVE]"
                
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label_text)
            th = fm.height()
            
            painter.setPen(Qt.PenStyle.NoPen)
            if i == self.active_idx:
                painter.setBrush(QColor(255, 152, 0, 220))
            elif i in self.labels:
                painter.setBrush(QColor(0, 122, 204, 220))
            else:
                painter.setBrush(QColor(76, 175, 80, 220))
                
            painter.drawRect(x, max(0, y - th - 6), tw + 10, th + 6)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 5, max(th, y - 5), label_text)

        # Draw exclusion and inclusion zones
        if self.pixmap:
            img_w = self.pixmap.width()
            img_h = self.pixmap.height()
            
            for i, zone in enumerate(self.zones):
                nx, ny, nw, nh = zone['rect']
                ex = nx * img_w
                ey = ny * img_h
                ew = nw * img_w
                eh = nh * img_h

                x = int(self.offset_x + ex * self.scale)
                y = int(self.offset_y + ey * self.scale)
                w = int(ew * self.scale)
                h = int(eh * self.scale)

                is_active = (i == self.active_zone_idx)
                z_type = zone['type']
                
                # Apply visual colors: Red for exclude, Teal for include
                if z_type == 'exclude':
                    border_color = QColor(244, 67, 54)
                    fill_color = QColor(244, 67, 54, 40)
                    label_text = f"EXCLUDE {i+1}"
                else:
                    border_color = QColor(0, 188, 212)
                    fill_color = QColor(0, 188, 212, 40)
                    label_text = f"INCLUDE {i+1}"

                if is_active:
                    border_pen = QPen(border_color, 3, Qt.PenStyle.DashLine)
                else:
                    border_pen = QPen(border_color, 2, Qt.PenStyle.SolidLine)

                painter.setPen(border_pen)
                painter.setBrush(fill_color)
                painter.drawRect(x, y, w, h)

                # Drawing Label
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.setPen(border_color)
                painter.drawText(x + 5, y + 18, label_text)

                # Draw corner anchor squares for active zone
                if is_active:
                    anchor_pen = QPen(border_color, 2, Qt.PenStyle.SolidLine)
                    painter.setPen(anchor_pen)
                    painter.setBrush(QColor(255, 255, 255))
                    
                    anchor_size = 8
                    half_size = anchor_size // 2
                    
                    corners = [
                        (x, y),
                        (x + w, y),
                        (x, y + h),
                        (x + w, y + h)
                    ]
                    for cx, cy in corners:
                        painter.drawRect(cx - half_size, cy - half_size, anchor_size, anchor_size)

    def mousePressEvent(self, event):
        if not self.pixmap:
            return

        pos = event.position()
        idx, mode = self.find_active_zone(pos)
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()

        # 1. If in draw mode: Left-click starts drawing a new zone
        if self.is_draw_mode and event.button() == Qt.MouseButton.LeftButton:
            self.drag_mode = 'create'
            self.drag_start_pos = self.map_to_orig(pos)
            new_zone = {
                'type': self.is_draw_mode,
                'rect': (self.drag_start_pos[0] / img_w, self.drag_start_pos[1] / img_h, 0.0, 0.0)
            }
            self.zones.append(new_zone)
            self.active_zone_idx = len(self.zones) - 1
            self.drag_start_rect = (self.drag_start_pos[0], self.drag_start_pos[1], 0, 0)
            self.update()
            return

        # 2. Ordinary left-click: select zone, move, resize, or select box
        if event.button() == Qt.MouseButton.LeftButton:
            if idx != -1:
                # Drag or resize existing zone
                self.active_zone_idx = idx
                self.drag_mode = mode
                self.drag_start_pos = self.map_to_orig(pos)
                
                # Convert normalized start rect to absolute
                nx, ny, nw, nh = self.zones[idx]['rect']
                self.drag_start_rect = (nx * img_w, ny * img_h, nw * img_w, nh * img_h)
                self.update()
            else:
                self.active_zone_idx = -1
                self.update()
                
                # Standard left-click selection of bounding boxes
                orig_x, orig_y = self.map_to_orig(pos)
                import sys
                print(f"[DEBUG] Mouse click at widget position: {event.position().x():.1f}, {event.position().y():.1f} -> mapped to original image coordinates: {orig_x}, {orig_y}", file=sys.stderr)
                clicked_idx = -1
                min_area = float('inf')
                for i, (bx, by, bw, bh) in enumerate(self.boxes):
                    if bx <= orig_x <= bx + bw and by <= orig_y <= by + bh:
                        area = bw * bh
                        if area < min_area:
                            min_area = area
                            clicked_idx = i
                print(f"[DEBUG] Click mapped to box index: {clicked_idx}", file=sys.stderr)
                if clicked_idx != -1:
                    self.entityClicked.emit(clicked_idx)
            return

        elif event.button() == Qt.MouseButton.RightButton:
            if idx != -1:
                # Right-click inside a zone deletes it
                self.zones.pop(idx)
                self.active_zone_idx = -1
                self.zonesChanged.emit(self.zones)
                self.update()

    def mouseMoveEvent(self, event):
        if not self.pixmap:
            return

        pos = event.position()

        if self.drag_mode:
            curr_ox, curr_oy = self.map_to_orig(pos)
            start_ox, start_oy = self.drag_start_pos
            sx, sy, sw, sh = self.drag_start_rect
            
            img_w = self.pixmap.width()
            img_h = self.pixmap.height()
            dx = curr_ox - start_ox
            dy = curr_oy - start_oy

            if self.drag_mode == 'create':
                x1, y1 = start_ox, start_oy
                x2, y2 = curr_ox, curr_oy
                self.zones[self.active_zone_idx]['rect'] = (
                    min(x1, x2) / img_w,
                    min(y1, y2) / img_h,
                    abs(x2 - x1) / img_w,
                    abs(y2 - y1) / img_h
                )

            elif self.drag_mode == 'move':
                new_x = max(0, min(sx + dx, img_w - sw))
                new_y = max(0, min(sy + dy, img_h - sh))
                self.zones[self.active_zone_idx]['rect'] = (new_x / img_w, new_y / img_h, sw / img_w, sh / img_h)

            elif self.drag_mode == 'resize_br':
                new_w = max(10, min(sw + dx, img_w - sx))
                new_h = max(10, min(sh + dy, img_h - sy))
                self.zones[self.active_zone_idx]['rect'] = (sx / img_w, sy / img_h, new_w / img_w, new_h / img_h)

            elif self.drag_mode == 'resize_tl':
                dx = min(dx, sw - 10)
                dy = min(dy, sh - 10)
                new_x = max(0, sx + dx)
                new_y = max(0, sy + dy)
                new_w = sw - (new_x - sx)
                new_h = sh - (new_y - sy)
                self.zones[self.active_zone_idx]['rect'] = (new_x / img_w, new_y / img_h, new_w / img_w, new_h / img_h)

            elif self.drag_mode == 'resize_tr':
                dy = min(dy, sh - 10)
                new_y = max(0, sy + dy)
                new_w = max(10, min(sw + dx, img_w - sx))
                new_h = sh - (new_y - sy)
                self.zones[self.active_zone_idx]['rect'] = (sx / img_w, new_y / img_h, new_w / img_w, new_h / img_h)

            elif self.drag_mode == 'resize_bl':
                dx = min(dx, sw - 10)
                new_x = max(0, sx + dx)
                new_w = sw - (new_x - sx)
                new_h = max(10, min(sh + dy, img_h - sy))
                self.zones[self.active_zone_idx]['rect'] = (new_x / img_w, sy / img_h, new_w / img_w, new_h / img_h)

            self.update()

        else:
            # Hover cursor update based on mouse position
            idx, mode = self.find_active_zone(pos)
            self.active_zone_idx = idx
            if idx != -1:
                if mode == 'move':
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                elif mode in ['resize_tl', 'resize_br']:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif mode in ['resize_tr', 'resize_bl']:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                if self.is_draw_mode:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.unsetCursor()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.pixmap:
            return

        if self.drag_mode:
            if self.drag_mode == 'create':
                nx, ny, nw, nh = self.zones[self.active_zone_idx]['rect']
                img_w = self.pixmap.width()
                img_h = self.pixmap.height()
                if (nw * img_w) < 10 or (nh * img_h) < 10:
                    self.zones.pop(self.active_zone_idx)
                    self.active_zone_idx = -1

            self.drag_mode = None
            if self.is_draw_mode:
                self.is_draw_mode = None
                
            self.zonesChanged.emit(self.zones)
            self.update()
            self.unsetCursor()


class LabelerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acadia AI - Entity Reidentification Labeler")
        self.resize(1200, 800)
        # self.setStyleSheet(DARK_STYLESHEET)

        # State Variables
        self.dir_path = None
        self.labels_file = None
        self.cache_file = None
        self.thumbs_dir = None
        
        self.all_images = []
        self.labels_db = {}       # filename -> list of {'box': [x,y,w,h], 'id': ID}
        self.features_cache = {}  # filename -> {str(box): list_of_floats}
        
        self.current_img_path = None
        self.current_img = None
        self.detected_boxes = []
        self.current_labels = {}  # box_idx -> ID
        self.active_box_idx = -1
        self.active_crop_feat = None
        self.predicted_id = None
        
        self.extractor = None
        self.profile_db = ProfileDatabase()
        self.shortcuts = []
        self.zoom_tooltip = ZoomTooltip(self)

        # Setup UI Components
        self.init_ui()
        self.setup_global_shortcuts()

    def init_ui(self):
        # Central splitter separating side panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # 1. Left Sidebar: Frame list & controls
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        self.select_dir_btn = QPushButton("Open Images Directory")
        self.select_dir_btn.clicked.connect(self.on_select_directory)
        sidebar_layout.addWidget(self.select_dir_btn)
        self.dir_label = QLabel("No directory loaded")
        self.dir_label.setWordWrap(True)
        sidebar_layout.addWidget(self.dir_label)

        # Compact Zones controls layout
        zones_h_layout = QHBoxLayout()
        zones_h_layout.setSpacing(4)
        
        self.draw_exclude_btn = QPushButton("Exclude (D)")
        self.draw_exclude_btn.setCheckable(True)
        self.draw_exclude_btn.clicked.connect(self.on_draw_exclude_clicked)
        self.draw_exclude_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 3px 6px; font-weight: bold; }"
            "QPushButton:checked { background-color: #d32f2f; color: white; }"
        )
        self.draw_exclude_btn.setToolTip("Draw Exclusion Zone (D): candidate boxes overlapping >10% of box or zone area are excluded.")
        zones_h_layout.addWidget(self.draw_exclude_btn)

        self.draw_include_btn = QPushButton("Include (I)")
        self.draw_include_btn.setCheckable(True)
        self.draw_include_btn.clicked.connect(self.on_draw_include_clicked)
        self.draw_include_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 3px 6px; font-weight: bold; }"
            "QPushButton:checked { background-color: #00bcd4; color: #121212; }"
        )
        self.draw_include_btn.setToolTip("Draw Inclusion Zone (I): candidate boxes must overlap at least one inclusion zone.")
        zones_h_layout.addWidget(self.draw_include_btn)

        self.clear_exclusion_btn = QPushButton("Clear")
        self.clear_exclusion_btn.setEnabled(False)
        self.clear_exclusion_btn.clicked.connect(self.on_clear_exclusion_clicked)
        self.clear_exclusion_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 3px 6px; background-color: #555; color: white; }"
            "QPushButton:disabled { background-color: #333; color: #777; }"
        )
        self.clear_exclusion_btn.setToolTip("Clear all drawn zones on this folder.")
        zones_h_layout.addWidget(self.clear_exclusion_btn)

        sidebar_layout.addLayout(zones_h_layout)

        # Sidebar Tabs Widget
        self.sidebar_tabs = QTabWidget()
        sidebar_layout.addWidget(self.sidebar_tabs)

        # Tab 1: Frames List
        frames_tab = QWidget()
        frames_tab_layout = QVBoxLayout(frames_tab)
        frames_tab_layout.setContentsMargins(0, 5, 0, 0)
        self.frame_list_widget = QListWidget()
        self.frame_list_widget.currentRowChanged.connect(self.on_frame_selected)
        frames_tab_layout.addWidget(self.frame_list_widget)
        self.sidebar_tabs.addTab(frames_tab, "Frames List")

        # Tab 2: Unique IDs Gallery
        gallery_tab = QWidget()
        gallery_tab_layout = QVBoxLayout(gallery_tab)
        gallery_tab_layout.setContentsMargins(0, 5, 0, 0)
        
        gallery_scroll = QScrollArea()
        gallery_scroll.setWidgetResizable(True)
        gallery_scroll_widget = QWidget()
        self.gallery_layout = QVBoxLayout(gallery_scroll_widget)
        self.gallery_layout.setContentsMargins(5, 5, 5, 5)
        self.gallery_layout.addStretch()
        gallery_scroll.setWidget(gallery_scroll_widget)
        gallery_tab_layout.addWidget(gallery_scroll)
        self.sidebar_tabs.addTab(gallery_tab, "Unique IDs")

        self.stats_label = QLabel("Unique Entities: 0 | Frames: 0")
        sidebar_layout.addWidget(self.stats_label)

        splitter.addWidget(sidebar)

        # 2. Center Panel: Image Viewer with Top-Bar controls
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        # Image Viewer
        self.image_widget = ImageWidget()
        self.image_widget.entityClicked.connect(self.on_entity_clicked)
        self.image_widget.zonesChanged.connect(self.on_zones_changed)
        center_layout.addWidget(self.image_widget)
        
        splitter.addWidget(center_widget)

        # 3. Right Panel: Labeling panel
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Group Box: Active Entity & Best Match Profile
        active_group = QGroupBox("Active Entity & Best Match Profile")
        active_group_layout = QHBoxLayout(active_group)
        active_group_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left: Active Entity crop
        active_pane = QWidget()
        active_pane_layout = QVBoxLayout(active_pane)
        active_pane_layout.setContentsMargins(0, 0, 0, 0)
        active_pane_layout.setSpacing(4)
        self.active_crop_label = QLabel()
        self.active_crop_label.setFixedSize(80, 80)
        self.active_crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.active_crop_label.setStyleSheet("border: 1px solid #444; background-color: #222; border-radius: 4px;")
        active_pane_layout.addWidget(self.active_crop_label, alignment=Qt.AlignmentFlag.AlignCenter)
        active_lbl = QLabel("Selected Active Crop")
        active_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        active_lbl.setStyleSheet("font-size: 11px; color: #888; font-weight: normal;")
        active_pane_layout.addWidget(active_lbl)
        active_group_layout.addWidget(active_pane)

        # Right: Best Match crop
        match_pane = QWidget()
        match_pane_layout = QVBoxLayout(match_pane)
        match_pane_layout.setContentsMargins(0, 0, 0, 0)
        match_pane_layout.setSpacing(4)
        self.match_crop_label = QLabel()
        self.match_crop_label.setFixedSize(80, 80)
        self.match_crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.match_crop_label.setStyleSheet("border: 1px solid #444; background-color: #222; border-radius: 4px;")
        match_pane_layout.addWidget(self.match_crop_label, alignment=Qt.AlignmentFlag.AlignCenter)
        match_lbl = QLabel("Best Match Profile")
        match_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        match_lbl.setStyleSheet("font-size: 11px; color: #888; font-weight: normal;")
        match_pane_layout.addWidget(match_lbl)
        active_group_layout.addWidget(match_pane)

        right_layout.addWidget(active_group)

        # Group Box: ResNet-50 Auto-Prediction
        predict_group = QGroupBox("ResNet-50 Auto-ReID Prediction")
        predict_layout = QVBoxLayout(predict_group)
        self.prediction_label = QLabel("Prediction: N/A")
        self.prediction_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff9800;")
        predict_layout.addWidget(self.prediction_label)

        self.accept_predict_btn = QPushButton("Accept Prediction (Space)")
        self.accept_predict_btn.clicked.connect(self.on_accept_prediction)
        self.accept_predict_btn.setEnabled(False)
        self.accept_predict_btn.setStyleSheet("background-color: #ff9800; color: #121212;")
        predict_layout.addWidget(self.accept_predict_btn)
        right_layout.addWidget(predict_group)

        manual_group = QGroupBox("Add or Select Identifier")
        manual_form = QFormLayout(manual_group)
        
        self.custom_id_input = QLineEdit()
        self.custom_id_input.setPlaceholderText("Enter numeric ID...")
        self.custom_id_input.returnPressed.connect(self.on_assign_custom_id)
        manual_form.addRow("Identifier ID:", self.custom_id_input)

        btn_layout = QHBoxLayout()
        self.assign_id_btn = QPushButton("Assign Custom ID (Enter)")
        self.assign_id_btn.clicked.connect(self.on_assign_custom_id)
        btn_layout.addWidget(self.assign_id_btn)

        self.new_id_btn = QPushButton("New ID (N)")
        self.new_id_btn.clicked.connect(self.on_assign_new_id)
        btn_layout.addWidget(self.new_id_btn)
        manual_form.addRow(btn_layout)
        right_layout.addWidget(manual_group)

        # Group Box: Quick matches list
        self.matches_group = QGroupBox("Fast ID Reidentification Search (Cosine Sim)")
        self.matches_scroll = QScrollArea()
        self.matches_scroll.setWidgetResizable(True)
        self.matches_scroll_widget = QWidget()
        self.matches_layout = QVBoxLayout(self.matches_scroll_widget)
        self.matches_layout.setContentsMargins(5, 5, 5, 5)
        self.matches_layout.addStretch()
        self.matches_scroll.setWidget(self.matches_scroll_widget)
        
        m_group_layout = QVBoxLayout(self.matches_group)
        m_group_layout.addWidget(self.matches_scroll)
        right_layout.addWidget(self.matches_group)

        # Group Box: Navigation Controls
        nav_group = QGroupBox("Controls / Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        self.prev_btn = QPushButton("Prev Box")
        self.prev_btn.clicked.connect(self.on_prev_box)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Skip Box (S)")
        self.next_btn.clicked.connect(self.on_skip_box)
        nav_layout.addWidget(self.next_btn)

        self.prev_frame_btn = QPushButton("Prev Frame")
        self.prev_frame_btn.clicked.connect(self.on_prev_frame)
        nav_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next Frame")
        self.next_frame_btn.clicked.connect(self.on_next_frame)
        nav_layout.addWidget(self.next_frame_btn)
        right_layout.addWidget(nav_group)

        splitter.addWidget(right_panel)

        # Set sizes of splitter columns
        splitter.setSizes([200, 750, 250])

    def setup_global_shortcuts(self):
        # Accept Prediction Space shortcut
        self.predict_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.predict_shortcut.activated.connect(self.on_accept_prediction)

        # New ID shortcut (N)
        self.new_id_shortcut = QShortcut(QKeySequence(Qt.Key.Key_N), self)
        self.new_id_shortcut.activated.connect(self.on_assign_new_id)

        # Skip box shortcut (S)
        self.skip_shortcut = QShortcut(QKeySequence(Qt.Key.Key_S), self)
        self.skip_shortcut.activated.connect(self.on_skip_box)

        # Frame navigation shortcuts (PageUp, PageDown)
        self.page_up_shortcut = QShortcut(QKeySequence(Qt.Key.Key_PageUp), self)
        self.page_up_shortcut.activated.connect(self.on_prev_frame)

        # Frame navigation shortcuts (PageUp, PageDown)
        self.page_down_shortcut = QShortcut(QKeySequence(Qt.Key.Key_PageDown), self)
        self.page_down_shortcut.activated.connect(self.on_next_frame)

        # Draw exclusion zone shortcut (D)
        self.exclude_shortcut = QShortcut(QKeySequence(Qt.Key.Key_D), self)
        self.exclude_shortcut.activated.connect(self.on_draw_exclude_shortcut)

        # Draw inclusion zone shortcut (I)
        self.include_shortcut = QShortcut(QKeySequence(Qt.Key.Key_I), self)
        self.include_shortcut.activated.connect(self.on_draw_include_shortcut)

    def clear_matches_layout(self):
        # Clear matches UI
        for s in self.shortcuts:
            s.setEnabled(False)
            s.deleteLater()
        self.shortcuts = []

        # Remove widgets from layout
        while self.matches_layout.count() > 1:
            child = self.matches_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    @Slot()
    def on_select_directory(self):
        dir_selected = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if dir_selected:
            self.load_directory(dir_selected)

    def load_directory(self, dir_path):
        self.dir_path = Path(dir_path)
        self.dir_label.setText(str(self.dir_path))
        self.labels_file = self.dir_path / "labels.json"
        self.cache_file = self.dir_path / "features_cache.json"
        
        self.thumbs_dir = self.dir_path / ".thumbnails"
        self.thumbs_dir.mkdir(exist_ok=True)

        # Load labels database
        if self.labels_file.exists():
            try:
                with open(self.labels_file, "r") as f:
                    self.labels_db = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load labels.json: {e}")
                self.labels_db = {}
        else:
            self.labels_db = {}

        # Load zones if present
        self.zones = self.labels_db.get("__zones__", [])
        self.image_widget.set_zones(self.zones)
        self.clear_exclusion_btn.setEnabled(len(self.zones) > 0)



        # Load feature cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.features_cache = json.load(f)
            except Exception as e:
                self.features_cache = {}
        else:
            self.features_cache = {}

        # Scan for images
        self.all_images = sorted([
            p for p in self.dir_path.glob("*")
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        if not self.all_images:
            QMessageBox.information(self, "No Images", "No images (.jpg, .jpeg, .png, .bmp) found in the selected folder.")
            return

        # Initialize ProfileDatabase
        self.profile_db = ProfileDatabase()

        # Find which items need feature extraction
        to_extract = []
        for img_path in self.all_images:
            rel_name = img_path.name
            if rel_name in self.labels_db:
                for item in self.labels_db[rel_name]:
                    box = item["box"]
                    box_str = str(box)
                    # If not in cache, we need to extract feature
                    if rel_name not in self.features_cache or box_str not in self.features_cache[rel_name]:
                        to_extract.append((img_path, box, item["id"]))
                    else:
                        feat = np.array(self.features_cache[rel_name][box_str])
                        self.profile_db.add_feature(item["id"], feat)

        # If any features need extraction, show a progress dialog while we run ResNet-50
        if to_extract:
            progress = QProgressDialog("Loading existing labels and extracting features...", "Cancel", 0, len(to_extract), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(0)
            progress.show()

            if self.extractor is None:
                self.extractor = EntityFeatureExtractor(use_gpu=False)

            cache_updated = False
            for i, (img_path, box, entity_id) in enumerate(to_extract):
                if progress.wasCanceled():
                    break
                
                img = cv2.imread(str(img_path))
                if img is not None:
                    x, y, w, h = box
                    crop = img[y:y+h, x:x+w]
                    if crop.size > 0:
                        feat = self.extractor.extract_features(crop)
                        self.profile_db.add_feature(entity_id, feat)
                        
                        # Cache feature
                        rel_name = img_path.name
                        if rel_name not in self.features_cache:
                            self.features_cache[rel_name] = {}
                        self.features_cache[rel_name][str(box)] = feat.tolist()
                        cache_updated = True

                        # Make sure thumbnail exists
                        thumb_path = self.thumbs_dir / f"id_{entity_id}.jpg"
                        if not thumb_path.exists():
                            cv2.imwrite(str(thumb_path), crop)

                progress.setValue(i + 1)
                QCoreApplication.processEvents()

            progress.close()

            if cache_updated:
                with open(self.cache_file, "w") as f:
                    json.dump(self.features_cache, f)

        # Update stats
        self.update_stats()

        # Update frame list in UI
        self.frame_list_widget.clear()
        for img_path in self.all_images:
            self.frame_list_widget.addItem(img_path.name)

        # Load first image
        self.frame_list_widget.setCurrentRow(0)
        self.update_ids_gallery()

    def update_stats(self):
        total_frames = len(self.all_images)
        unique_entities = len(self.profile_db.profiles)
        self.stats_label.setText(f"Unique Entities: {unique_entities} | Frames: {total_frames}")

    @Slot(int)
    def on_frame_selected(self, row_idx):
        if row_idx < 0 or row_idx >= len(self.all_images):
            return
        
        self.current_img_path = self.all_images[row_idx]
        self.current_img = cv2.imread(str(self.current_img_path))
        
        # Load existing labels for this image
        rel_name = self.current_img_path.name
        existing_labels = self.labels_db.get(rel_name, [])
        img_h, img_w = self.current_img.shape[:2]

        # Detect boxes inside the image and filter by zones
        raw_boxes = detect_entities(self.current_img)
        self.detected_boxes = self.filter_boxes_by_zones(raw_boxes, img_w, img_h)
        
        # Prune stale labels and update active labels coordinates to prevent duplicates
        cleaned_labels = []
        db_changed = False
        
        for el in existing_labels:
            ex, ey, ew, eh = el["box"]
            matched_raw_box = None
            
            # Find matching raw detected box in this frame
            for rbox in self.detected_boxes:
                rx, ry, rw, rh = rbox
                ix1 = max(rx, ex)
                iy1 = max(ry, ey)
                ix2 = min(rx + rw, ex + ew)
                iy2 = min(ry + rh, ey + eh)
                if ix2 > ix1 and iy2 > iy1:
                    int_area = (ix2 - ix1) * (iy2 - iy1)
                    min_area = min(rw * rh, ew * eh)
                    if int_area / min_area > 0.8:
                        matched_raw_box = rbox
                        break
            
            if matched_raw_box is None:
                # No longer matches any detected entity (stale/ghost box) -> prune
                db_changed = True
                continue
                
            # Align coordinates exactly to prevent duplicate entries on subsequent writes
            if el["box"] != list(matched_raw_box):
                el["box"] = list(matched_raw_box)
                db_changed = True
                
            cleaned_labels.append(el)

        if db_changed:
            self.labels_db[rel_name] = cleaned_labels
            existing_labels = cleaned_labels
            if self.labels_file:
                with open(self.labels_file, "w") as f:
                    json.dump(self.labels_db, f, indent=2)

        self.current_labels = {}

        # Match detected boxes with the cleaned/aligned database labels
        for box_idx, dbox in enumerate(self.detected_boxes):
            dbox_list = list(dbox)
            for el in existing_labels:
                if el["box"] == dbox_list:
                    self.current_labels[box_idx] = el["id"]
                    break

        # Set active box to first unlabeled box. If all are labeled, make first active.
        self.active_box_idx = -1
        for i in range(len(self.detected_boxes)):
            if i not in self.current_labels:
                self.active_box_idx = i
                break
        if self.active_box_idx == -1 and self.detected_boxes:
            self.active_box_idx = 0

        self.update_ui_for_active_box()

    def update_ui_for_active_box(self):
        if not self.current_img_path:
            return

        self.image_widget.set_image(
            self.current_img_path,
            self.detected_boxes,
            self.current_labels,
            self.active_box_idx
        )

        # Reset active display fields
        self.active_crop_label.clear()
        self.match_crop_label.clear()
        self.prediction_label.setText("Prediction: N/A")
        self.accept_predict_btn.setEnabled(False)
        self.accept_predict_btn.setText("Accept Prediction (Space)")
        self.active_crop_feat = None
        self.custom_id_input.clear()
        self.clear_matches_layout()

        if self.active_box_idx == -1 or not self.detected_boxes:
            self.active_crop_label.setText("No active entity")
            self.match_crop_label.setText("No active match")
            return

        x, y, w, h = self.detected_boxes[self.active_box_idx]
        crop = self.current_img[y:y+h, x:x+w]
        if crop.size == 0:
            return

        # Show crop thumbnail
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        qh, qw, qc = rgb_crop.shape
        bytes_per_line = qc * qw
        qimg = QImage(rgb_crop.data, qw, qh, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.active_crop_label.setPixmap(pix)

        # Get features
        rel_name = self.current_img_path.name
        box_str = str([x, y, w, h])
        
        if rel_name in self.features_cache and box_str in self.features_cache[rel_name]:
            self.active_crop_feat = np.array(self.features_cache[rel_name][box_str])
        else:
            if self.extractor is None:
                self.extractor = EntityFeatureExtractor(use_gpu=False)
            self.active_crop_feat = self.extractor.extract_features(crop)
            
            # Cache it
            if rel_name not in self.features_cache:
                self.features_cache[rel_name] = {}
            self.features_cache[rel_name][box_str] = self.active_crop_feat.tolist()
            with open(self.cache_file, "w") as f:
                json.dump(self.features_cache, f)

        # Run Auto prediction
        best_id, best_sim = self.profile_db.predict_id(self.active_crop_feat, threshold=0.0)
        if best_id is not None:
            self.predicted_id = best_id
            self.prediction_label.setText(f"Prediction: ID {best_id} (Sim: {best_sim*100:.1f}%)")
            self.accept_predict_btn.setEnabled(True)
            self.accept_predict_btn.setText(f"Accept Prediction (Space) [ID: {best_id}]")
            
            # Show matching profile representative thumbnail
            thumb_path = self.thumbs_dir / f"id_{best_id}.jpg"
            if thumb_path.exists():
                match_pix = QPixmap(str(thumb_path)).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.match_crop_label.setPixmap(match_pix)
            else:
                self.match_crop_label.setText("[No Profile Image]")
        else:
            self.predicted_id = None
            self.prediction_label.setText("Prediction: No profiles in DB")
            self.accept_predict_btn.setEnabled(False)
            self.accept_predict_btn.setText("Accept Prediction (Space)")
            self.match_crop_label.setText("[No Match Profile]")

        self.update_top_matches()

    def update_top_matches(self):
        self.clear_matches_layout()
        if self.active_crop_feat is None:
            return

        # Find similarity against all profiles
        matches = []
        for pid, feats in self.profile_db.profiles.items():
            max_sim = -1.0
            for p_feat in feats:
                sim = float(np.dot(self.active_crop_feat, p_feat))
                if sim > max_sim:
                    max_sim = sim
            matches.append((pid, max_sim))

        # Sort and take top 5
        matches = sorted(matches, key=lambda m: m[1], reverse=True)[:5]

        for i, (pid, sim) in enumerate(matches):
            match_frame = QFrame()
            match_frame.setFrameShape(QFrame.Shape.StyledPanel)
            match_frame.setStyleSheet("QFrame { background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; }")
            
            m_layout = QHBoxLayout(match_frame)
            m_layout.setContentsMargins(4, 4, 4, 4)

            # Load profile representative thumbnail
            thumb_label = QLabel()
            thumb_path = self.thumbs_dir / f"id_{pid}.jpg"
            if thumb_path.exists():
                thumb_pix = QPixmap(str(thumb_path)).scaled(45, 45, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                thumb_label.setPixmap(thumb_pix)
            else:
                thumb_label.setText("No Img")
                thumb_label.setFixedSize(45, 45)
                thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
            m_layout.addWidget(thumb_label)

            info_label = QLabel(f"<b>ID {pid}</b><br/>Sim: {sim*100:.1f}%")
            info_label.setStyleSheet("font-size: 11px; color: #fff;")
            m_layout.addWidget(info_label)

            m_layout.addStretch()

            assign_btn = QPushButton(f"Assign (Ctrl+{i+1})")
            assign_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 4px 8px; background-color: #2a2a2a; }")
            # Connect click slot
            assign_btn.clicked.connect(lambda checked=False, p_id=pid: self.assign_id(p_id))
            m_layout.addWidget(assign_btn)

            self.matches_layout.insertWidget(self.matches_layout.count() - 1, match_frame)

            # Bind global hotkey
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i+1}"), self)
            shortcut.activated.connect(lambda p_id=pid: self.assign_id(p_id))
            self.shortcuts.append(shortcut)

    def assign_id(self, entity_id):
        import sys
        print(f"[DEBUG] assign_id: active_box_idx={self.active_box_idx}, assigned entity_id={entity_id}", file=sys.stderr)
        if self.active_box_idx == -1 or not self.detected_boxes:
            return

        try:
            entity_id = int(entity_id)
        except ValueError:
            QMessageBox.warning(self, "Invalid ID", "Please enter a valid numeric ID.")
            return

        x, y, w, h = self.detected_boxes[self.active_box_idx]
        print(f"[DEBUG] Box coords assigned to ID {entity_id}: {[x, y, w, h]}", file=sys.stderr)
        rel_name = self.current_img_path.name

        # Save to database
        if rel_name not in self.labels_db:
            self.labels_db[rel_name] = []

        # Remove previous label for the same box if it existed
        self.labels_db[rel_name] = [el for el in self.labels_db[rel_name] if el["box"] != [x, y, w, h]]
        self.labels_db[rel_name].append({"box": [x, y, w, h], "id": entity_id})

        with open(self.labels_file, "w") as f:
            json.dump(self.labels_db, f, indent=2)

        # Update local states
        self.current_labels[self.active_box_idx] = entity_id

        # Update ProfileDatabase
        if self.active_crop_feat is not None:
            self.profile_db.add_feature(entity_id, self.active_crop_feat)

        # Save/overwrite thumbnail crop
        thumb_path = self.thumbs_dir / f"id_{entity_id}.jpg"
        crop = self.current_img[y:y+h, x:x+w]
        if crop.size > 0:
            cv2.imwrite(str(thumb_path), crop)

        self.update_stats()
        self.update_ids_gallery()

        # Advance to next unlabeled box or next frame
        self.advance_next()

    def advance_next(self):
        # Look for next unlabeled box
        next_idx = -1
        for i in range(len(self.detected_boxes)):
            if i not in self.current_labels:
                next_idx = i
                break
                
        import sys
        print(f"[DEBUG] advance_next: next unlabeled box is {next_idx}", file=sys.stderr)
        if next_idx != -1:
            self.active_box_idx = next_idx
            self.update_ui_for_active_box()
        else:
            # Advance to next frame
            current_row = self.frame_list_widget.currentRow()
            print(f"[DEBUG] advance_next: advancing to next frame, current row={current_row}", file=sys.stderr)
            if current_row < len(self.all_images) - 1:
                self.frame_list_widget.setCurrentRow(current_row + 1)
            else:
                QMessageBox.information(self, "Finished", "All frames in this directory have been processed!")
                # Reset active index so image widget redraws without active selection box
                self.active_box_idx = -1
                self.update_ui_for_active_box()

    @Slot()
    def on_accept_prediction(self):
        if self.predicted_id is not None:
            self.assign_id(self.predicted_id)

    @Slot()
    def on_assign_custom_id(self):
        text = self.custom_id_input.text().strip()
        if text:
            self.assign_id(text)
        else:
            # If input is empty, assign a brand new ID
            self.on_assign_new_id()

    @Slot()
    def on_assign_new_id(self):
        new_id = self.profile_db.get_next_id()
        self.assign_id(new_id)

    @Slot(int)
    def on_entity_clicked(self, box_idx):
        import sys
        print(f"[DEBUG] on_entity_clicked: updating active_box_idx from {self.active_box_idx} to {box_idx}", file=sys.stderr)
        self.active_box_idx = box_idx
        self.update_ui_for_active_box()

    @Slot()
    def on_prev_box(self):
        if len(self.detected_boxes) > 0:
            self.active_box_idx = (self.active_box_idx - 1) % len(self.detected_boxes)
            self.update_ui_for_active_box()

    @Slot()
    def on_skip_box(self):
        if len(self.detected_boxes) > 0:
            self.active_box_idx = (self.active_box_idx + 1) % len(self.detected_boxes)
            self.update_ui_for_active_box()

    @Slot()
    def on_prev_frame(self):
        row = self.frame_list_widget.currentRow()
        if row > 0:
            self.frame_list_widget.setCurrentRow(row - 1)

    @Slot()
    def on_next_frame(self):
        row = self.frame_list_widget.currentRow()
        if row < len(self.all_images) - 1:
            self.frame_list_widget.setCurrentRow(row + 1)


    def update_ids_gallery(self):
        # Clear previous gallery widgets
        while self.gallery_layout.count() > 1:
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Populate gallery with all registered entity profiles in profile_db
        # Sort keys numerically
        for pid in sorted(self.profile_db.profiles.keys(), key=int):
            item_frame = QFrame()
            item_frame.setFrameShape(QFrame.Shape.StyledPanel)
            item_frame.setStyleSheet("QFrame { background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; }")
            
            layout = QHBoxLayout(item_frame)
            layout.setContentsMargins(4, 4, 4, 4)
            
            # Hover zoom thumbnail
            thumb_lbl = HoverThumbnailLabel(pid, self.thumbs_dir, self.zoom_tooltip)
            layout.addWidget(thumb_lbl)

            # Text info
            info_lbl = QLabel(f"<b>ID: {pid}</b>")
            info_lbl.setStyleSheet("color: white; font-size: 12px;")
            layout.addWidget(info_lbl)
            
            layout.addStretch()
            
            self.gallery_layout.insertWidget(self.gallery_layout.count() - 1, item_frame)

    def is_box_excluded(self, box, img_w, img_h):
        # Coordinates of the box
        bx, by, bw, bh = box
        
        # Check exclusion zones
        has_inclusion_zones = False
        inside_at_least_one_inclusion = False
        
        for zone in self.zones:
            nx, ny, nw, nh = zone['rect']
            zx = nx * img_w
            zy = ny * img_h
            zw = nw * img_w
            zh = nh * img_h
            
            if zone['type'] == 'exclude':
                # Calculate intersection
                ix1 = max(bx, zx)
                iy1 = max(by, zy)
                ix2 = min(bx + bw, zx + zw)
                iy2 = min(by + bh, zy + zh)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersect_area = (ix2 - ix1) * (iy2 - iy1)
                    # Exclude if overlap is > 10% of either box area or zone area
                    box_area = bw * bh
                    zone_area = zw * zh
                    if box_area > 0 and (intersect_area / box_area) > 0.10:
                        return True
                    if zone_area > 0 and (intersect_area / zone_area) > 0.10:
                        return True
            
            elif zone['type'] == 'include':
                has_inclusion_zones = True
                # Check overlap with inclusion zone
                ix1 = max(bx, zx)
                iy1 = max(by, zy)
                ix2 = min(bx + bw, zx + zw)
                iy2 = min(by + bh, zy + zh)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersect_area = (ix2 - ix1) * (iy2 - iy1)
                    box_area = bw * bh
                    zone_area = zw * zh
                    if box_area > 0 and (intersect_area / box_area) > 0.10:
                        inside_at_least_one_inclusion = True
                    elif zone_area > 0 and (intersect_area / zone_area) > 0.10:
                        inside_at_least_one_inclusion = True

        if has_inclusion_zones and not inside_at_least_one_inclusion:
            return True
            
        return False

    def filter_boxes_by_zones(self, boxes, img_w, img_h):
        if not self.zones:
            return boxes
        return [b for b in boxes if not self.is_box_excluded(b, img_w, img_h)]

    @Slot(object)
    def on_zones_changed(self, zones):
        self.zones = zones
        
        # Enable/disable clear button
        self.clear_exclusion_btn.setEnabled(len(self.zones) > 0)
        
        # Save zones to database
        if self.dir_path:
            self.labels_db["__zones__"] = self.zones
            if self.labels_file:
                with open(self.labels_file, "w") as f:
                    json.dump(self.labels_db, f, indent=2)
                    
        # Reset check states of draw buttons
        self.draw_exclude_btn.setChecked(False)
        self.draw_include_btn.setChecked(False)
        
        # Reload the current frame to re-evaluate filtering
        self.reload_current_frame()

    @Slot()
    def on_clear_exclusion_clicked(self):
        self.zones = []
        self.image_widget.set_zones([])
        self.clear_exclusion_btn.setEnabled(False)
        
        # Save to database
        if self.dir_path:
            if "__zones__" in self.labels_db:
                del self.labels_db["__zones__"]
            if self.labels_file:
                with open(self.labels_file, "w") as f:
                    json.dump(self.labels_db, f, indent=2)
                    
        # Reset check states of draw buttons
        self.draw_exclude_btn.setChecked(False)
        self.draw_include_btn.setChecked(False)
        
        # Reload the current frame to re-evaluate filtering
        self.reload_current_frame()

    @Slot()
    def on_draw_exclude_clicked(self):
        if self.draw_exclude_btn.isChecked():
            self.draw_include_btn.setChecked(False)
            self.image_widget.set_draw_mode('exclude')
        else:
            self.image_widget.set_draw_mode(None)

    @Slot()
    def on_draw_include_clicked(self):
        if self.draw_include_btn.isChecked():
            self.draw_exclude_btn.setChecked(False)
            self.image_widget.set_draw_mode('include')
        else:
            self.image_widget.set_draw_mode(None)

    @Slot()
    def on_draw_exclude_shortcut(self):
        # Toggle Exclude mode
        self.draw_exclude_btn.setChecked(not self.draw_exclude_btn.isChecked())
        self.on_draw_exclude_clicked()

    @Slot()
    def on_draw_include_shortcut(self):
        # Toggle Include mode
        self.draw_include_btn.setChecked(not self.draw_include_btn.isChecked())
        self.on_draw_include_clicked()

    def reload_current_frame(self):
        # Reload current frame (re-evaluates bounding boxes filtration)
        row = self.frame_list_widget.currentRow()
        if row != -1:
            self.on_frame_selected(row)


def main():
    app = QApplication(sys.argv)
    window = LabelerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
