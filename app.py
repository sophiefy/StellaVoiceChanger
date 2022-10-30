import threading
import wave
import os
import shutil
import time
import pyaudio
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtMultimedia
import sys
sys.path.append('frontend')
sys.path.append('backend/sovits')
sys.path.append('backend/starganv2')
from frontend.basic import Ui_MainWindow
from frontend.subwin import Ui_SubWindow
from frontend.utilities import get_model_info
import backend.sovits.inference as SovitsVC
import backend.starganv2.inference as StarGANv2VC


class EmitStr(QObject):
    textWrite = pyqtSignal(str)

    def write(self, text):
        self.textWrite.emit(str(text))

class SubWindow(QWidget, Ui_SubWindow):
    def __init__(self, file_path):
        super().__init__()

        self.setupUi(self)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()

        self.textBrowser.setMarkdown(data)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        sys.stdout = EmitStr(textWrite=self.outputWriteInfo)  # redirect stdout
        sys.stderr = EmitStr(textWrite=self.outputWriteError)  # redirect stderr

        self.setupUi(self)
        self.setMinimumSize(1600, 900)
        self.initBasicAttributes()

        # set up sub-windows
        file_path_usage = 'README.md'
        file_path_faq = 'infomation/faq.md'
        file_path_author = 'infomation/author.md'
        file_path_project = 'infomation/project.md'
        self.usageWin = SubWindow(file_path_usage)
        self.usageWin.setWindowTitle('Usage')
        self.faqWin = SubWindow(file_path_faq)
        self.faqWin.setWindowTitle('FAQ')
        self.authorWin = SubWindow(file_path_author)
        self.authorWin.setWindowTitle('Author')
        self.projectWin = SubWindow(file_path_project)
        self.projectWin.setWindowTitle('Project')

        self.connectSignalSlots()
        self.shutMainFrame()



    def initBasicAttributes(self):
        # --------------- #
        # SECTION: common
        # --------------- #
        self.resource_dir = 'resources'

        # ---------------------- #
        # SECTION: configuration
        # ---------------------- #
        self.curent_model = 'sovits'
        self.model_dir = ''
        self.thread_model = threading.Thread()
        self.thread_model.setDaemon(True)

        # ------------- #
        # SECTION: main
        # ------------- #

        # SUBSECTION: device
        self.gpu_mode = False

        # SUBSECTION: character portrait
        self.charPortrait = QPixmap()

        # SUBSECTION: record
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.rec_save_path = 'cache/record.wav'
        self.rec_sec = 0
        self.is_recording = False
        self.event_start_rec = threading.Event()
        self.thread_rec = threading.Thread(target=self.recordAudio)
        self.thread_rec.setDaemon(True)
        self.thread_rec.start()

        # SUBSECTION: play
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(50)
        self.play_path = 'cache/play.wav'
        self.play_timer = QTimer()
        self.is_playing = False

        # SUBSECTION: download
        self.download_path = 'cache/download.wav'

        # SUBSECTION: mode
        self.vc_mode = 'svt_hubert'

    def connectSignalSlots(self):

        # ----------------- #
        # SECTION: menu bar
        # ----------------- #
        self.actionSovits.triggered.connect(self.page2sovits)
        self.actionStarganv2.triggered.connect(self.page2stargan)
        self.actionUsage.triggered.connect(self.usageWin.show)
        self.actionAuthor.triggered.connect(self.authorWin.show)
        self.actionFAQ.triggered.connect(self.faqWin.show)
        self.actionProject.triggered.connect(self.projectWin.show)
        # ---------------------- #
        # SECTION: configuration
        # ---------------------- #

        # SUBSECTION: Sovits
        self.pageSovits.dirBtn.clicked.connect(self.chooseModelDir)
        self.pageSovits.loadBtn.clicked.connect(self.loadModel)
        self.pageSovits.deviceBtn.clicked.connect(self.changeDevice)

        # SUBSECTION: StarGANv2
        self.pageStarganv2.dirBtn.clicked.connect(self.chooseModelDir)
        self.pageStarganv2.loadBtn.clicked.connect(self.loadModel)
        self.pageStarganv2.deviceBtn.clicked.connect(self.changeDevice)

        # ------------- #
        # SECTION: main
        # ------------- #

        # SUBSECTION: record
        self.recordBtn.clicked.connect(self.startOrStopRecord)

        # SUBSECTION: upload
        self.uploadBtn.clicked.connect(self.uploadAudio)

        # SUBSECTION：choose character
        self.charCombo.currentIndexChanged.connect(lambda: self.chooseChar(self.charCombo.currentIndex()))

        # SUBSECTION: convert
        self.convertBtn.clicked.connect(self.convertAudio)

        # SUBSECTION: play
        self.playBtn.clicked.connect(self.playAudio)
        self.play_timer.timeout.connect(self.updateSlider)

        # SUBSECTION: download
        self.downloadBtn.clicked.connect(self.downloadAudio)

    # --------------- #
    # NOTE: COMMON
    # --------------- #
    def page2sovits(self):
        if self.curent_model != 'sovits':
            reply = QMessageBox.question(self,
                                         'Question',
                                         'Change to Sovits?\nThis will terminate current thread.',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.shutMainFrame()
                self.curent_model = 'sovits'
                self.stackedWidget.setCurrentIndex(0)

            else:
                pass
        else:
            pass

    def page2stargan(self):
        if self.curent_model != 'starganv2':
            reply = QMessageBox.question(self,
                                         'Question',
                                         'Change to StarGANv2?\nThis will terminate current thread.',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.shutMainFrame()
                self.curent_model = 'starganv2'
                self.stackedWidget.setCurrentIndex(1)
            else:
                pass
        else:
            pass

    def initMainFrame(self):
        self.recordBtn.setDisabled(False)
        self.uploadBtn.setDisabled(False)
        self.convertBtn.setDisabled(False)
        self.playBtn.setDisabled(False)
        self.downloadBtn.setDisabled(False)
        self.micLbl.setStyleSheet("QLabel {border-image: url(resources/common/mic.png); background-color: transparent}")
        self.frmMicLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/from-mic.png); background-color: transparent}")
        self.recordBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/start-record.png); background-color: transparent}")
        self.uploadBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/upload.png); background-color: transparent}")
        self.frmUploadLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/from-upload.png); background-color: transparent}")
        self.convertBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/start.png); background-color: transparent}")
        self.playBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/play.png); background-color: transparent}")
        self.downloadBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/download.png); background-color: transparent}")
        self.play_timer = QTimer()
        self.play_timer.start(1000)
        self.play_timer.timeout.connect(self.updateSlider)

    def shutMainFrame(self):
        while self.thread_model.is_alive():
            self.endInference()

        self.recordBtn.setDisabled(True)
        self.uploadBtn.setDisabled(True)
        self.convertBtn.setDisabled(True)
        self.playBtn.setDisabled(True)
        self.downloadBtn.setDisabled(True)
        self.micLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/mic-unable.png); background-color: transparent}")
        self.frmMicLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/from-mic-unable.png); background-color: transparent}")
        self.recordBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/start-record-unable.png); background-color: transparent}")
        self.uploadBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/upload-unable.png); background-color: transparent}")
        self.frmUploadLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/from-upload-unable.png); background-color: transparent}")
        self.convertBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/start-unable.png); background-color: transparent}")
        self.playBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/play-off.png); background-color: transparent}")
        self.downloadBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/download-off.png); background-color: transparent}")

        self.is_recording = False
        self.is_playing = False
        self.play_timer = QTimer()
        self.charPortrait.load('resources/common/portrait-init.png')
        self.portraitLbl.setPixmap(self.charPortrait)
        self.charCombo.clear()
        self.setStyleSheet('#MainWindow {border-image:url(resources/common/bg-init.png)}')

    def loadModel(self):
        if self.model_dir:
            self.initMainFrame()

            while self.thread_model.is_alive():
                self.endInference()

            self.thread_model = threading.Thread(target=self.startInference)
            self.thread_model.setDaemon(True)
            self.thread_model.start()

            # 选择speaker面板加载
            hps = get_model_info(self.model_dir)
            model_name = hps.info.model_name
            speakers = hps.info.speakers

            self.model_resource_dir = f'{self.resource_dir}/{model_name}'
            self.setStyleSheet('#MainWindow {border-image:url(%s)}'
                               % f'{self.model_resource_dir}/bg.png')

            self.charCombo.clear()
            for speaker in speakers:
                self.charCombo.addItem(speaker)

            self.charPortrait.load(os.path.join(self.model_resource_dir, '0.png'))
            self.portraitLbl.setPixmap(self.charPortrait)

            if len(speakers) == 1:
                self.charCombo.setDisabled(True)
            else:
                self.charCombo.setDisabled(False)

    def recordAudio(self):
        while True:
            self.event_start_rec.wait()
            self.event_start_rec.clear()

            print('Start recording...')
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            frames_per_buffer=self.CHANNELS,
                            input=True)
            frames = []
            while self.is_recording:
                data = stream.read(self.CHUNK)
                frames.append(data)

            print('Stop recording...')
            stream.stop_stream()  # 停止錄音
            stream.close()  # 關閉串流
            p.terminate()

            # 保存录音文件到cache
            wave_file = wave.open(self.rec_save_path, 'wb')
            wave_file.setnchannels(self.CHANNELS)  # 設定聲道
            wave_file.setsampwidth(p.get_sample_size(self.FORMAT))  # 設定格式
            wave_file.setframerate(self.RATE)  # 設定取樣頻率
            wave_file.writeframes(b''.join(frames))  # 存檔
            wave_file.close()

            # 将录音文件加载到模型中
            self.loadExistAudio(self.rec_save_path)

    def startOrStopRecord(self):
        if not self.is_recording:
            self.rec_sec = 0
            self.recordBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/stop-record.png); background-color: transparent}}")
            self.uploadBtn.setDisabled(True)
            self.uploadBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/upload-unable.png); background-color: transparent}")
            self.frmUploadLbl.setStyleSheet(
                "QLabel {border-image: url(resources/common/from-upload-unable.png); background-color: transparent}")
            self.is_recording = True
            self.event_start_rec.set()
        else:
            self.recordBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/start-record.png); background-color: transparent}}")
            self.uploadBtn.setDisabled(False)
            self.uploadBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/upload.png); background-color: transparent}")
            self.frmUploadLbl.setStyleSheet(
                "QLabel {border-image: url(resources/common/from-upload.png); background-color: transparent}")
            self.is_recording = False

    def setCurrentPlaying(self):
        try:
            content = QtMultimedia.QMediaContent(QUrl.fromLocalFile(self.play_path))
            self.player.setMedia(content)
        except:
            print('Nothing to play!')

    def playAudio(self):
        if not self.player.isAudioAvailable():
            self.setCurrentPlaying()
        if not self.is_playing:
            self.convertBtn.setDisabled(True)
            self.convertBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/start-unable.png); background-color: transparent}}")
            self.is_playing = True
            self.playBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/pause.png); background-color: transparent}}")
            self.player.play()
        else:
            self.is_playing = False
            self.playButton.setStyleSheet(
                "QPushButton {border-image: url(resources/common/play.png); background-color: transparent}}")
            self.player.pause()

    def updateSlider(self):
        # update slider when playing
        if self.is_playing:
            self.playSlider.setMinimum(0)
            self.playSlider.setMaximum(self.player.duration())
            self.playSlider.setValue(self.playSlider.value() + 1000)  # update

        self.startTimeLbl.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.endTimeLbl.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        # reset the slider when playing over
        if self.player.position() == self.player.duration():
            self.playSlider.setValue(0)
            self.startTimeLbl.setText('00:00')
            self.is_playing = False
            self.player.stop()
            self.playBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/play.png); background-color: transparent}}")
            self.convertBtn.setDisabled(False)
            self.convertBtn.setStyleSheet(
                "QPushButton {border-image: url(resources/common/start.png); background-color: transparent}}")

    def downloadAudio(self):
        file_path = QFileDialog.getSaveFileName(self, 'Chosse save file path', 'converted.wav', f'(*.wav)')
        if file_path[0]:
            try:
                shutil.copy(self.download_path, file_path[0])
            except:
                print('Nothing to download!')

    def closeEvent(self, e):
        reply = QMessageBox.question(self,
                                     'Question',
                                     "Are you sure to close Stella Voice Changer?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            e.accept()
            sys.exit(0)
        else:
            e.ignore()

    def outputWriteInfo(self, text):
        self.textInfo.append(text)

    def outputWriteError(self, text):
        self.textInfo.append(f'<font color=\'#FF0000\'>{text}</font>')

    # -------------------- #
    # NOTE: MODEL SPECIFIC
    # -------------------- #

    def chooseModelDir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Choose model directory', '/home')
        if dir_path:
            self.model_dir = dir_path
            if self.curent_model == 'sovits':
                self.pageSovits.dirPth.setText(dir_path)
            elif self.curent_model == 'starganv2':
                self.pageStarganv2.dirPth.setText(dir_path)
            else:
                pass

    def startInference(self):
        if self.curent_model == 'sovits':
            SovitsVC.voice_conversion(self.model_dir)
        elif self.curent_model == 'starganv2':
            StarGANv2VC.voice_conversion(self.model_dir)

    def endInference(self):
        if self.curent_model == 'sovits':
            SovitsVC.terminate_vc()
        elif self.curent_model == 'starganv2':
            StarGANv2VC.terminate_vc()

    # TODO: GPU mode
    def changeDevice(self):
        QMessageBox.warning(self, 'Warning', 'GPU mode is not supported yet!')

    def chooseChar(self, index=0):
        self.charPortrait.load(os.path.join(self.model_resource_dir, f'{index}.png'))
        self.portraitLbl.setPixmap(self.charPortrait)
        if self.curent_model == 'sovits':
            SovitsVC.select_speaker(index)
        elif self.curent_model == 'starganv2':
            StarGANv2VC.select_speaker(index)

    def loadExistAudio(self, audio_path):
        if self.curent_model == 'sovits':
            SovitsVC.load_audio(audio_path)
        elif self.curent_model == 'starganv2':
            StarGANv2VC.load_audio(audio_path)

    def uploadAudio(self):
        self.recordBtn.setDisabled(True)
        self.recordBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/start-record-unable.png); background-color: transparent}")
        self.frmMicLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/from-mic-unable.png); background-color: transparent}")
        self.micLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/mic-unable.png); background-color: transparent}")
        file_path = QFileDialog.getOpenFileName(self, 'Chosse source audio file', '/home', '(*.wav *mp3 *ogg *flv)')
        if file_path[0]:
            if self.curent_model == 'sovits':
                SovitsVC.load_audio(file_path[0])
            elif self.curent_model == 'starganv2':
                StarGANv2VC.load_audio(file_path[0])
            else:
                pass

        self.recordBtn.setDisabled(False)
        self.recordBtn.setStyleSheet(
            "QPushButton {border-image: url(resources/common/start-record.png); background-color: transparent}")
        self.frmMicLbl.setStyleSheet(
            "QLabel {border-image: url(resources/common/from-mic.png); background-color: transparent}")
        self.micLbl.setStyleSheet("QLabel {border-image: url(resources/common/mic.png); background-color: transparent}")

    def convertAudio(self):
        self.player.setMedia(QtMultimedia.QMediaContent())
        if self.curent_model == 'sovits':
            SovitsVC.convert_audio()
        elif self.curent_model == 'starganv2':
            StarGANv2VC.convert_audio()
        else:
            pass





if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()

    mainWin.show()
    print('Welcome to Stella Voice Changer!\n'
          'Please load models and corresponding configuration files!')
    sys.exit(app.exec_())
