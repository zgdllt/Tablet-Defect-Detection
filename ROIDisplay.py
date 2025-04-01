from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import QRect, Qt

class ROIDisplay(QWidget):
    def __init__(self, parent=None):
        super(ROIDisplay, self).__init__(parent)
        self.roi = None  # ROI: (x, y, w, h)

    def updateROI(self, roi):
        self.roi = roi
        self.update()  # 触发重绘

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.roi:
            painter = QPainter(self)
            # 设置红色画笔
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            x, y, w, h = self.roi
            painter.drawRect(QRect(x, y, w, h))
