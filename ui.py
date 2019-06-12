import time
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *

class VideoBox(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    def __init__(self, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        QWidget.__init__(self)
        self.video_url = ''
        self.cwd = os.getcwd()
        self.video_type = video_type  # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause
        self.resize(512, 512)

        # 组件展示
        self.pictureLabel = QLabel()
        self.pictureLabel.setPixmap(QPixmap("./player.jpeg"))
        # self.pictureLabel.setText('Select your video file')
        self.pictureLabel.setScaledContents(True)
        
        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.openButton = QPushButton()
        self.openButton.setEnabled(True)
        self.openButton.setText('File')
        self.openButton.clicked.connect(self.choosefile)

        control_box = QHBoxLayout()
        control_box.setContentsMargins(0, 0, 0, 0)
        control_box.addWidget(self.playButton)
        control_box.addWidget(self.openButton)

        layout = QVBoxLayout()
        layout.addWidget(self.pictureLabel)
        layout.addLayout(control_box)

        self.setLayout(layout)

    def choosefile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self, "Select file", self.cwd, "All Files (*);;MP4 Files (*.mp4)") 
        self.video_url = fileName_choose
        self.close()

if __name__ == "__main__":
   app = QApplication(sys.argv)
   box = VideoBox()
   box.show()
   sys.exit(app.exec_())
