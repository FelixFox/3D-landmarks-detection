import sys
import PyQt5 as qt
import ui.py.main as MainWindow
from utils.key_points_recognizer import  KeyPointsRecognizer
from utils.key_points_renderer import KeyPointsRenderer
from services.bg_function_worker import BgFunctionWorker
from utils.qt import opencv2qtpixmap

class App(qt.QtWidgets.QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__init_ui()
        self.__bind_controls()
        self.recognizer = KeyPointsRecognizer(checkpoint_fp='./models/phase1_wpdc_vdc_v2.pth.tar', frames_step=1)
        # self.bg_function_worker = BgFunctionWorker(fn=self.__get_keypoints)
        # self.bg_function_worker.fn_finished.connect(self.__animate_keypoints)
        
    def __init_ui(self):
        self.pbarAnimation.setEnabled(False)
        self.image_height = 480
        self.image_width = 640
        self.lblImage.move(0, 0)
        self.lblImage.resize(self.image_width, self.image_height)

    def __bind_controls(self):
        self.btnBrowse.clicked.connect(self.__browse_file)
        self.btnAnimate.setEnabled(False)
        self.btnAnimate.clicked.connect(self.__recongize_keypoints_bg)

    def __recongize_keypoints_bg(self):
        pts = self.__get_keypoints()
        self.__animate_keypoints(pts)

    def __file_opened(self):
        self.btnAnimate.setEnabled(True)

    def __browse_file(self):
        # TODO: change to 1 file
        # filepath = qt.QtWidgets.QFileDialog.getExistingDirectory(self, "Choose directory...")
        filepath, _ = qt.QtWidgets.QFileDialog.getOpenFileName(self, "Choose file...")
        if (filepath):
            print(filepath)
            self.__keypoints_path = filepath
            self.__file_opened()

    def __get_keypoints(self):
        self.pbarAnimation.setEnabled(True)
        def on_frame_processed(frame, index, total_number, _):
            self.pbarAnimation.setValue(index / total_number * 100)
            self.lblImage.setPixmap(opencv2qtpixmap(frame, self.image_width, self.image_height))
        
        return self.recognizer.get_key_points_from_video(video_fp=self.__keypoints_path, on_frame_processed= on_frame_processed)

    def __animate_keypoints(self, key_points):
        self.renderer = KeyPointsRenderer()
        self.renderer.frames = key_points
        self.renderer.animate_key_points_from_frames(key_points)

        self.pbarAnimation.setEnabled(False)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    app = qt.QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()