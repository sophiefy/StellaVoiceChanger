import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtMultimedia
import threading
import time
import shutil
from StellaVC import voice_conversion, load_audio, convert_audio

class EmitStr(QObject):
    textWrite = pyqtSignal(str)

    def write(self, text):
        self.textWrite.emit(str(text))


class BaseGuiWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        sys.stdout = EmitStr(textWrite=self.outputWrite)  # redirect stdout
        sys.stderr = EmitStr(textWrite=self.outputWrite)  # redirect stderr

        self.initUI()

    def initUI(self):
        self.resize(1600, 800)
        self.setFixedSize(1600, 800)
        self.center()
        self.setWindowTitle('Stella Voice Changer')
        self.setObjectName('MainWindow')
        self.setWindowIcon(QIcon('assets/icon-nat.ico'))
        self.setStyleSheet('#MainWindow{border-image:url(assets/bg-nat.jpg)}')

        col = QColor(245, 245, 245)

        # ------------------- #
        # configuration panel #
        # ------------------- #
        self.configFrame = QFrame(self)
        self.configFrame.setFrameShape(QFrame.StyledPanel)
        self.configFrame.setStyleSheet("QWidget {background-color: rgba(245, 245, 245, 220)}")
        self.configFrame.setWindowOpacity(0.6)
        self.configFrame.setGeometry(20, 20, 400, 280)

        self.hubertButton = QPushButton('hubert path', self.configFrame)
        self.hubertButton.setFixedSize(100, 50)
        self.hubertButton.move(10, 10)
        self.hubertButton.clicked.connect(lambda: self.chooseFile('pt'))
        self.vitsButton = QPushButton('vits path', self.configFrame)
        self.vitsButton.setFixedSize(100, 50)
        self.vitsButton.move(10, 80)
        self.vitsButton.clicked.connect(lambda: self.chooseFile('pth'))
        self.configButton = QPushButton('config path', self.configFrame)
        self.configButton.setFixedSize(100, 50)
        self.configButton.move(10, 150)
        self.configButton.clicked.connect(lambda: self.chooseFile('json'))
        self.loadButton = QPushButton('load model', self.configFrame)
        self.loadButton.setFixedSize(380, 50)
        self.loadButton.move(10, 220)
        self.loadButton.clicked.connect(self.loadModel)

        self.hubertPath = QTextEdit(self.configFrame)
        self.hubertPath.setFocusPolicy(Qt.NoFocus)
        self.hubertPath.setFixedSize(270, 50)
        self.hubertPath.move(120, 10)
        self.vitsPath = QTextEdit(self.configFrame)
        self.vitsPath.setFocusPolicy(Qt.NoFocus)
        self.vitsPath.setFixedSize(270, 50)
        self.vitsPath.move(120, 80)
        self.configPath = QTextEdit(self.configFrame)
        self.configPath.setFocusPolicy(Qt.NoFocus)
        self.configPath.setFixedSize(270, 50)
        self.configPath.move(120, 150)

        self.hubert_path = ''
        self.vits_path = ''
        self.config_path = ''

        # ----------------- #
        # information panel #
        # ----------------- #
        self.infoFrame = QFrame(self)
        self.infoFrame.setFrameShape(QFrame.StyledPanel)
        self.infoFrame.setStyleSheet("QWidget {background-color: rgba(245, 245, 245, 220)}")
        self.infoFrame.setGeometry(20, 320, 400, 460)

        self.infoBox = QTextEdit(self.infoFrame)
        self.infoBox.setFocusPolicy(Qt.NoFocus)
        self.infoBox.setFixedSize(390, 450)
        self.infoBox.move(5, 5)

        # ---------- #
        # main panel #
        # ---------- #
        self.mainFrame = QFrame(self)
        self.mainFrame.setFrameShape(QFrame.StyledPanel)
        self.mainFrame.setStyleSheet("QWidget {background-color: rgba(245, 245, 245, 220)}")
        self.mainFrame.setGeometry(440, 20, 1140, 760)

        # character portrait
        self.charFrame = QLabel(self.mainFrame)
        self.charFrame.setFrameShape(QFrame.StyledPanel)
        self.charFrame.resize(256, 256)
        self.charFrame.move(442, 30)

        self.charPortrait = QPixmap('assets/natsume.jpg')

        self.charFrame.setPixmap(self.charPortrait)

        # TODO: 支持多人模型
        # self.charCombo = QComboBox(self.mainFrame)
        # self.charCombo.addItem('Shiki Natsume')
        # self.charCombo.addItem('Misaka Mikoto')
        # self.charCombo.addItem('Shirai Kuroko')
        # self.charCombo.addItem('Renge')
        #
        # self.charCombo.setGeometry(442, 300, 256, 30)
        # self.charCombo.currentIndexChanged.connect(lambda: self.chooseChar(self.charCombo.currentIndex()))

        # record
        self.recordButton = QPushButton('', self.mainFrame)
        self.recordButton.setStyleSheet("QPushButton {border-image: url(assets/microphone.png); background-color: transparent}")
        self.recordButton.setFixedSize(100, 100)
        self.recordButton.move(171, 100) # cp.x: 221
        self.recordButton.clicked.connect(self.record)
        self.recordLabel = QLabel('', self.mainFrame)
        self.recordLabel.setStyleSheet("QLabel {border-image: url(assets/from-mic-1.png); background-color: transparent}")
        self.recordLabel.setFixedSize(100, 100)
        self.recordLabel.move(311, 100)

        # upload
        self.uploadButton = QPushButton('', self.mainFrame)
        self.uploadButton.setStyleSheet("QPushButton {border-image: url(assets/upload.png); background-color: transparent}")
        self.uploadButton.setFixedSize(100, 100)
        self.uploadButton.move(869, 100) # cp.x: 919
        self.uploadButton.clicked.connect(self.upload)
        self.uploadLabel = QLabel('', self.mainFrame)
        self.uploadLabel.setStyleSheet("QLabel {border-image: url(assets/from-upload-1.png); background-color: transparent}")
        self.uploadLabel.setFixedSize(100, 100)
        self.uploadLabel.move(729, 100)


        # convert
        self.convertButton = QPushButton('', self.mainFrame)
        self.convertButton.setStyleSheet("QPushButton {border-image: url(assets/start.png); background-color: transparent}")
        self.convertButton.setFixedSize(50, 50)
        self.convertButton.move(440, 605)
        self.convertButton.clicked.connect(self.convert)

        # play
        self.playFrame = QFrame(self.mainFrame)
        self.playFrame.setFixedSize(1100, 200)
        self.playFrame.move(20, 540)
        self.playFrame.setFrameShape(QFrame.Box)
        self.playFrame.setFrameShadow(QFrame.Raised)
        self.playFrame.setStyleSheet("QFrame {border-width: 3px; border-style: solid; border-color: rgb(18, 150, 219); background-color: transparent}")

        self.playButton = QPushButton('', self.mainFrame)
        self.playButton.setFixedSize(100, 100)
        self.playButton.move(520, 580)
        self.playButton.setStyleSheet("QPushButton {border-image: url(assets/play.png); background-color: transparent}")
        self.playButton.clicked.connect(self.play)
        self.playFlag = False

        self.audio_path = '../sovits_cache/temp.wav'
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(50)

        self.startTimeLabel = QLabel('00:00', self.mainFrame)
        self.endTimeLabel = QLabel('00:00', self.mainFrame)
        self.startTimeLabel.setFixedSize(40,40)
        self.endTimeLabel.setFixedSize(40, 40)
        self.startTimeLabel.move(70, 687)
        self.endTimeLabel.move(1030, 687)
        self.slider = QSlider(Qt.Horizontal, self.mainFrame)
        self.slider.setFixedSize(800, 20)
        self.slider.move(170, 700)

        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.updateSlider)
        self.slider.sliderMoved[int].connect(lambda: self.player.setPosition(self.slider.value()))

        # download
        self.downloadButton = QPushButton('', self.mainFrame)
        self.downloadButton.setStyleSheet("QPushButton {border-image: url(assets/download.png); background-color: transparent}")
        self.downloadButton.setFixedSize(50, 50)
        self.downloadButton.move(650, 605)
        self.downloadButton.clicked.connect(self.download)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def chooseFile(self, file_type):
        if file_type == 'pt':
            file_path = QFileDialog.getOpenFileName(self, f'Choose hubert model', '/home', f'(*.pt)')
            if file_path[0]:
                self.hubertPath.setText(file_path[0])
                self.hubert_path = file_path[0]
        elif file_type == 'pth':
            file_path = QFileDialog.getOpenFileName(self, f'Choose vits model', '/home', f'(*.pth)')
            if file_path[0]:
                self.vitsPath.setText(file_path[0])
                self.vits_path = file_path[0]
        elif file_type == 'json':
            file_path = QFileDialog.getOpenFileName(self, f'Choose configuration file', '/home', f'(*.json)')
            if file_path[0]:
                self.configPath.setText(file_path[0])
                self.config_path = file_path[0]
        else:
            raise ValueError('Unsupported file type!')

    # TODO: 根据配置文件改变图片
    # def chooseChar(self, index=0):
    #     if index == 0:
    #         self.charPortrait.load('assets/natsume.jpg')
    #         self.charFrame.setPixmap(self.charPortrait)

    def loadModel(self):
        if self.hubert_path and self.hubert_path and self.config_path:
            thread = threading.Thread(target=lambda: voice_conversion(self.hubert_path,
                                                                      self.vits_path,
                                                                      self.config_path))
            thread.setDaemon(True) # 防止主线程结束后子线程仍然运行
            thread.start()

    def outputWrite(self, text):
        self.infoBox.append(text)

    # TODO: 连接到麦克风
    def record(self):
        QMessageBox.about(self,
                          'Warning',
                          'Recording is not supported yet!')

    # TODO: 上传音频
    def upload(self):
        file_path = QFileDialog.getOpenFileName(self, 'Choose source audio file', '/home', f'(*.wav)')
        if file_path[0]:
            self.uploadLabel.setStyleSheet("QLabel {border-image: url(assets/from-upload-2.png); background-color: transparent}")
            load_audio(file_path[0])

    def convert(self):
        convert_audio()

    # TODO: 播放音频
    def setCurrentPlaying(self):
        content = QtMultimedia.QMediaContent(QUrl.fromLocalFile(self.audio_path))
        self.player.setMedia(content)

    def play(self):
        if not self.player.isAudioAvailable():
            self.setCurrentPlaying()
        if not self.playFlag:
            self.playFlag = True
            #self.playButton.setText('pause')
            self.playButton.setStyleSheet("QPushButton {border-image: url(assets/pause.png)}")
            self.player.play()
        else:
            self.playFlag = False
            #self.playButton.setText('play')
            self.playButton.setStyleSheet("QPushButton {border-image: url(assets/play.png)}")
            self.player.pause()

    def updateSlider(self):
        # update slider when playing
        if self.playFlag:
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.player.duration())
            self.slider.setValue(self.slider.value() + 1000) # update

        self.startTimeLabel.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.endTimeLabel.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        # reset the slider when playing over
        if self.player.position() == self.player.duration():
            self.slider.setValue(0)
            self.startTimeLabel.setText('00:00')
            self.playFlag = False
            # self.playButton.setText('play')
            self.playButton.setStyleSheet("QPushButton {border-image: url(assets/play.png)}")


    # TODO: 调节音量
    def changeVolume(self):
        pass

    # TODO: 下载音频
    def download(self):
        file_path = QFileDialog.getSaveFileName(self, 'Chosse save file path', 'converted.wav', f'(*.wav)')
        if file_path[0]:
            shutil.copy(self.audio_path, file_path[0])

    def closeEvent(self, e):
        reply = QMessageBox.question(self,
                                     'Warning',
                                     "Are you sure to close StellaVC?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            e.accept()
            sys.exit(0)  # 退出程序
        else:
            e.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BaseGuiWidget()
    print('Welcome to Stella Voice Changer!\n\n'
          'Please specify the model and configuration file paths to load models!')
    sys.exit(app.exec_())
