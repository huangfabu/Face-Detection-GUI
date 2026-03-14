# encoding: utf-8
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QApplication
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QBrush
from PyQt5.QtCore import Qt
import cv2 as cv
import numpy as np
import os
from main import FaceMeshDetector

def cvimg_to_qtimg(cv_img):
    """BGR OpenCV图像转QImage"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    return QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

class RoundedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(80)
        self.setMinimumWidth(120)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        self.update_style()
    
    def update_style(self):
        size = min(self.width(), self.height())
        if size < 10:
            size = 100
        font_size = max(22, min(36, int(size / 12)))
        border_radius = int(size / 8)
        margin = max(5, int(size / 30))
        
        self.setStyleSheet(f'''
            QPushButton {{
                border-radius: {border_radius}px;
                background-color: #f0f4fa;
                color: #333;
                font-size: {font_size}px;
                font-weight: bold;
                border: 2px solid #a0b4d4;
                margin: {margin}px;
                padding: {margin}px;
            }}
            QPushButton:hover {{
                background-color: #e0e8f8;
                border: 2px solid #5a8dee;
            }}
        ''')
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_style()


class FaceMeshApp(QWidget):
    current_pixmap = None  # 缓存当前显示内容

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_label_font()
        if hasattr(self, 'current_pixmap') and self.current_pixmap is not None:
            pix = self.current_pixmap.scaled(self.display_label.width(), self.display_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.display_label.setPixmap(pix)
    
    def update_display_label_font(self):
        if not hasattr(self, 'display_label'):
            return
        size = min(self.display_label.width(), self.display_label.height())
        if size < 10:
            size = 100
        font_size = max(14, min(32, int(size / 25)))
        self.display_label.setStyleSheet(f'font-size: {font_size}px; color: #888;')

    def __init__(self):
        super().__init__()
        self.setWindowTitle('人脸网格检测系统')
        self.setMinimumSize(900, 600)
        self.detector = FaceMeshDetector(model_path=os.path.join(os.path.dirname(__file__), 'face_landmarker.task'))
        self.init_ui()
        self.update_display_label_font()

    def init_ui(self):
        # 左右布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 左侧图片检测按钮
        self.img_btn = RoundedButton('+\n图片检测')
        self.img_btn.clicked.connect(self.select_image)
        
        # 中间展示区
        self.display_label = QLabel('请点击左侧或右侧按钮选择图片或视频')
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # 右侧视频检测按钮
        self.vid_btn = RoundedButton('+\n视频检测')
        self.vid_btn.clicked.connect(self.select_video)
        
        # 布局
        main_layout.addWidget(self.img_btn, 1)
        main_layout.addWidget(self.display_label, 3)
        main_layout.addWidget(self.vid_btn, 1)
        self.setLayout(main_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if file_path:
            qt_img = QImage(file_path)
            if qt_img.isNull():
                self.display_label.setText('无法加载图片')
                self.current_pixmap = None
                self.update_display_label_font()
                return
            qt_img = qt_img.convertToFormat(QImage.Format_RGB888)
            w, h = qt_img.width(), qt_img.height()
            bytes_per_line = qt_img.bytesPerLine()
            ptr = qt_img.bits()
            ptr.setsize(qt_img.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape((h, bytes_per_line))
            arr = arr[:, :w*3]
            img = arr.reshape((h, w, 3))
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            processed_frame, skeleton_img, _ = self.detector.find_face_mesh(img, draw=True)
            dst = self.detector.frame_combine(processed_frame, skeleton_img)
            qt_img2 = cvimg_to_qtimg(dst)
            pix = QPixmap.fromImage(qt_img2)
            self.current_pixmap = pix
            pix = pix.scaled(self.display_label.width(), self.display_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.display_label.setPixmap(pix)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi *.mov *.mkv)')
        if file_path:
            self.display_label.setText('正在处理视频...\n按Q退出播放')
            self.update_display_label_font()
            self.process_video(file_path)

    def process_video(self, video_path):
        cap = cv.VideoCapture(video_path)
        pTime = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, skeleton_img, _ = self.detector.find_face_mesh(frame, draw=True)
            cTime = cv.getTickCount() / cv.getTickFrequency()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime
            text = "FPS : " + str(int(fps))
            cv.putText(processed_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            dst = self.detector.frame_combine(processed_frame, skeleton_img)
            qt_img = cvimg_to_qtimg(dst)
            pix = QPixmap.fromImage(qt_img)
            self.current_pixmap = pix
            pix = pix.scaled(self.display_label.width(), self.display_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.display_label.setPixmap(pix)
            QApplication.processEvents()
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        self.display_label.setText('视频播放结束')
        self.current_pixmap = None
    current_pixmap = None  # 缓存当前显示内容

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceMeshApp()
    window.show()
    sys.exit(app.exec_())
