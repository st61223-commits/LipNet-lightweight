import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pathlib, sys, queue
from collections import deque
pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import tensorflow as tf
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import math
import socket
import struct
import sounddevice as sd
import psutil
import whisper

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from googletrans import Translator

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QGraphicsDropShadowEffect, QComboBox,
    QApplication, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ===== 翻譯設定 =====
translator = Translator()

LANGUAGES = {
    '不翻譯': None,
    '繁體中文': 'zh-tw',
    '簡體中文': 'zh-cn',
    '日文': 'ja',
    '韓文': 'ko',
    '法文': 'fr',
    '德文': 'de',
    '西班牙文': 'es',
    '越南文': 'vi',
}

# ===== GRID 51 詞彙表（用於驗證 Whisper 輸出）=====
GRID_VOCAB = {
    'bin', 'lay', 'place', 'set',
    'blue', 'green', 'red', 'white',
    'at', 'by', 'in', 'with',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'again', 'now', 'please', 'soon',
}

# Whisper 有時輸出數字符號，轉換成英文
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

# ===== GPU 設定 =====
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# ===== ZeroTier IP =====
def get_zerotier_ip_pair():
    zerotier_ips = []
    local_ip = None
    target_ip = '172.29.132.124'
    for interface_name, interface_addresses in psutil.net_if_addrs().items():
        if "zerotier" in interface_name.lower():
            for address in interface_addresses:
                if address.family.name == "AF_INET":
                    zerotier_ips.append(address.address)
    if len(zerotier_ips) > 0:
        local_ip = zerotier_ips[0]
        if len(zerotier_ips) > 1:
            target_ip = zerotier_ips[1]
    return local_ip, target_ip

my_zerotier_ip, target_zerotier_ip = get_zerotier_ip_pair()
local_ip = my_zerotier_ip
video_port = 8082
local_text_port = 9082
target_ip = target_zerotier_ip
target_port = 8081
target_text_port = 9081

# ===== 詞彙與模型 =====
optimizer = Adam()

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

try:
    lipnet_model_s99 = load_model(
        r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_width05_yolov8.h5',
        custom_objects={'CTCLoss': CTCLoss}, compile=False)
    lipnet_model_s1 = lipnet_model_s99
except Exception as e:
    import traceback
    with open(r'C:\Users\Tno\error_log.txt', 'w') as f:
        f.write("LipNet載入失敗:\n" + traceback.format_exc())
    raise

try:
    yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
except Exception as e:
    import traceback
    with open(r'C:\Users\Tno\error_log.txt', 'w') as f:
        f.write("YOLOv8載入失敗:\n" + traceback.format_exc())
    raise

# Whisper 跑 CPU，避免與 TensorFlow 搶 GPU 記憶體
print("載入 Whisper tiny.en 模型（第一次會自動下載約 39MB）...")
whisper_model = whisper.load_model("tiny.en", device="cpu")
print("Whisper 載入完成！")

face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Tno\miniconda3\envs\py39\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

# ===== 前處理函式 =====
input_shape = (75, 46, 140, 1)
predicted_text = ""

def preprocess_lip_shape(lip_region):
    return tf.image.resize(lip_region, (input_shape[1], input_shape[2]))

def preprocess_lip_std(lip_regionf):
    mean = tf.math.reduce_mean(lip_regionf)
    std = tf.math.reduce_std(tf.cast(lip_regionf, tf.float32))
    return tf.cast((lip_regionf - mean), tf.float32) / std


# ===== 錄音器 =====
class AudioRecorder:
    """與影像幀同步的錄音器"""
    SAMPLE_RATE = 16000  # Whisper 標準取樣率

    def __init__(self):
        self._q = queue.Queue()
        self._stream = None
        self._recording = False

    def start(self):
        self._q = queue.Queue()
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE, channels=1, dtype='float32',
            callback=self._callback)
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._q.put(indata.copy())

    def stop(self):
        """停止並回傳 float32 mono 陣列"""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get())
        if chunks:
            return np.concatenate(chunks, axis=0)[:, 0]
        return np.zeros(self.SAMPLE_RATE * 3, dtype=np.float32)


def fuse_results(lipnet_text: str, audio_array: np.ndarray, is_grid_mode: bool):
    """
    Whisper 辨識 + 融合邏輯。
    - GRID 模式：Whisper 輸出若有 50%+ 詞在 GRID 詞彙 → 信任 Whisper
    - s99 模式：Whisper 有輸出 → 直接用 Whisper
    回傳 (最終文字, 來源)
    """
    if audio_array is None or len(audio_array) < AudioRecorder.SAMPLE_RATE:
        return lipnet_text, 'LipNet'
    try:
        result = whisper_model.transcribe(audio_array, language='en', fp16=False)
        whisper_raw = result.get('text', '').strip()
    except Exception:
        return lipnet_text, 'LipNet'

    if not whisper_raw:
        return lipnet_text, 'LipNet'

    if is_grid_mode:
        words = [DIGIT_MAP.get(w, w) for w in whisper_raw.lower().split()]
        valid = [w for w in words if w in GRID_VOCAB]
        if not valid:
            return lipnet_text, 'LipNet'
        if len(valid) / max(len(words), 1) >= 0.5:
            return ' '.join(valid), 'Whisper'
        return lipnet_text, 'LipNet'
    else:
        if len(whisper_raw) > 2:
            return whisper_raw, 'Whisper'
        return lipnet_text, 'LipNet'


# ===== 網路類別 =====
class VideoDisplay(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, capture):
        super().__init__()
        self.capture = capture
        self.status = False
        self.enable = True
        self.MAX_DGRAM = 2**16
        self.MAX_IMAGE_DGRAM = self.MAX_DGRAM - 64

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(self.frame)
            if self.status and self.enable:
                self.enable = False
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.status and not self.enable:
                self.transfer_img()

    def transfer_img(self):
        compress_img = cv2.imencode('.jpg', self.frame)[1]
        data = compress_img.tobytes()
        size = len(data)
        num_of_segments = math.ceil(size / self.MAX_IMAGE_DGRAM)
        array_pos_start = 0
        while num_of_segments:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            self.client_socket.sendto(
                struct.pack('B', num_of_segments) + data[array_pos_start:array_pos_end],
                (self.ip, self.port))
            array_pos_start = array_pos_end
            num_of_segments -= 1

    def enable_send(self, ip, port):
        self.ip = ip; self.port = port; self.status = True

    def disable_send(self):
        self.status = False; self.enable = True


class VideoReceiver(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, port):
        super().__init__()
        self.port = port
        self.server_socket = None
        self.MAX_DGRAM = 2**16

    def run(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(("0.0.0.0", self.port))
        data = b''
        self.dump_buffer(self.server_socket)
        while True:
            seg, _ = self.server_socket.recvfrom(self.MAX_DGRAM)
            if struct.unpack('B', seg[0:1])[0] > 1:
                data += seg[1:]
            else:
                data += seg[1:]
                try:
                    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
                    self.frame_ready.emit(frame)
                except:
                    pass
                data = b''

    def dump_buffer(self, s):
        while True:
            seg, _ = s.recvfrom(self.MAX_DGRAM)
            if struct.unpack('B', seg[0:1])[0] == 1:
                break

    def __del__(self):
        if self.server_socket:
            self.server_socket.close()
        self.exit()


class SubtitleSender(threading.Thread):
    def __init__(self, target_ip, subtitle_port, predicted_text):
        super().__init__()
        self.target_ip = target_ip
        self.subtitle_port = subtitle_port
        self.predicted_text = predicted_text
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.success = False

    def run(self):
        try:
            self.socket.sendto(self.predicted_text.encode('utf-8'), (self.target_ip, self.subtitle_port))
            if self.predicted_text == "*** Connection from Target IP is allowed ***":
                self.socket.settimeout(2)
                try:
                    data, addr = self.socket.recvfrom(1024)
                    if data.decode('utf-8') == "ACK":
                        self.success = True
                except socket.timeout:
                    self.success = False
        except Exception:
            self.success = False
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.socket.close()


class SubtitleReceiver(threading.Thread):
    def __init__(self, subtitle_port, update_ui_callback):
        super().__init__()
        self.subtitle_port = subtitle_port
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', self.subtitle_port))
        self.update_ui_callback = update_ui_callback

    def run(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                subtitle = data.decode('utf-8')
                self.update_ui_callback(subtitle)
                self.socket.sendto("ACK".encode('utf-8'), addr)
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.socket.close()


# ===== 主介面 =====
class Endpoint(QWidget):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.lip_regions = []
        self.start_time = None
        self.prediction_done = True
        self.box_history = deque(maxlen=10)
        self.no_detect_count = 0
        self.audio_recorder = AudioRecorder()
        global predicted_text
        predicted_text = ""
        self.init_ui()

    def _apply_button_style(self, btn, kind="primary"):
        if kind == "primary":
            style = """QPushButton { border: none; border-radius: 10px; padding: 10px 18px;
                background-color: #2563eb; color: #f9fafb; font-size: 16px; font-weight: 600; }
                QPushButton:hover { background-color: #1d4ed8; }
                QPushButton:pressed { background-color: #1e3a8a; }"""
        else:
            style = """QPushButton { border: none; border-radius: 10px; padding: 10px 18px;
                background-color: #4b5563; color: #f9fafb; font-size: 16px; font-weight: 600; }
                QPushButton:hover { background-color: #374151; }
                QPushButton:pressed { background-color: #1f2937; }"""
        btn.setStyleSheet(style)
        btn.setCursor(Qt.PointingHandCursor)

    def _add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24); shadow.setXOffset(0); shadow.setYOffset(8)
        shadow.setColor(Qt.black)
        widget.setGraphicsEffect(shadow)

    def init_ui(self):
        self.setWindowTitle('LipNet Connect 2（音視覺融合 + Whisper）')
        self.setGeometry(80, 60, 1400, 900)
        self.setStyleSheet("""
            QWidget { background: #0f172a; color: #e5e7eb; }
            QLineEdit { background: #111827; border: 1px solid #374151; color: #e5e7eb;
                border-radius: 10px; padding: 8px 10px; font-size: 16px; }
            QComboBox { background: #111827; border: 1px solid #374151; color: #e5e7eb;
                border-radius: 10px; padding: 8px 10px; font-size: 16px; }
            QLabel[head="true"] { font-size: 26px; font-weight: 800; color: #f8fafc; }
            QLabel[subtle="true"] { color: #9ca3af; font-size: 14px; }
            QGroupBox { border: 1px solid #293042; border-radius: 14px; margin-top: 18px; }
            QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 2px 6px;
                color: #a5b4fc; font-weight: 700; }
        """)

        title = QLabel("LipNet Connect 2　音視覺融合版（LipNet + Whisper）")
        title.setProperty("head", True)
        title.setAlignment(Qt.AlignCenter)

        # ── 連線區 ──
        self.local_ip_label = QLabel("Local IP:")
        self.local_ip_label.setProperty("subtle", True)
        self.local_ip = QLineEdit(str(local_ip))
        self.local_port_label = QLabel("Local Camera / Text Port:")
        self.local_port_label.setProperty("subtle", True)
        self.local_port = QLineEdit(f"{video_port} / {local_text_port}")
        self.server_button = QPushButton("Start Server")
        self._apply_button_style(self.server_button)
        self.server_button.clicked.connect(self.start_threads)

        row1 = QHBoxLayout()
        row1.setSpacing(12)
        row1.addWidget(self.local_ip_label)
        row1.addWidget(self.local_ip, 1)
        row1.addWidget(self.local_port_label)
        row1.addWidget(self.local_port, 1)
        row1.addWidget(self.server_button)

        self.target_ip_label = QLabel("Target IP:")
        self.target_ip_label.setProperty("subtle", True)
        self.target_ip = QLineEdit(str(target_ip))
        self.target_port_label = QLabel("Target Camera / Text Port:")
        self.target_port_label.setProperty("subtle", True)
        self.target_port = QLineEdit(f"{target_port} / {target_text_port}")
        self.connect_button = QPushButton("Connect")
        self._apply_button_style(self.connect_button)
        self.connect_button.clicked.connect(self.connect_threads)
        self.disconnect_button = QPushButton("Disconnect")
        self._apply_button_style(self.disconnect_button, "secondary")
        self.disconnect_button.clicked.connect(self.disconnect_threads)
        self.predict_button = QPushButton("Start Prediction")
        self._apply_button_style(self.predict_button)
        self.predict_button.clicked.connect(self.start_prediction)
        self.video_file_button = QPushButton("選擇影片")
        self._apply_button_style(self.video_file_button, "secondary")
        self.video_file_button.clicked.connect(self.select_video_file)

        self.lang_label = QLabel("翻譯語言:")
        self.lang_label.setProperty("subtle", True)
        self.lang_combo = QComboBox()
        for lang_name in LANGUAGES.keys():
            self.lang_combo.addItem(lang_name)
        self.lang_combo.setCurrentText('繁體中文')

        self.mode_label = QLabel("辨識模式:")
        self.mode_label.setProperty("subtle", True)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("一般模式（s99）")
        self.mode_combo.addItem("GRID 模式（s1）")
        self.mode_combo.setCurrentText("一般模式（s99）")

        row2_inputs = QHBoxLayout()
        row2_inputs.setSpacing(12)
        row2_inputs.addWidget(self.target_ip_label)
        row2_inputs.addWidget(self.target_ip, 1)
        row2_inputs.addWidget(self.target_port_label)
        row2_inputs.addWidget(self.target_port, 1)

        row2_buttons = QHBoxLayout()
        row2_buttons.setSpacing(12)
        row2_buttons.addWidget(self.mode_label)
        row2_buttons.addWidget(self.mode_combo)
        row2_buttons.addWidget(self.lang_label)
        row2_buttons.addWidget(self.lang_combo)
        row2_buttons.addStretch(1)
        row2_buttons.addWidget(self.connect_button)
        row2_buttons.addWidget(self.disconnect_button)
        row2_buttons.addWidget(self.predict_button)
        row2_buttons.addWidget(self.video_file_button)

        # ── 影像區 ──
        self.video_label1 = QLabel("Local Camera\n(Not started)")
        self.video_label1.setAlignment(Qt.AlignCenter)
        self.video_label1.setScaledContents(True)
        self.video_label1.setMinimumSize(640, 480)
        self.video_label1.setStyleSheet(
            "QLabel { background-color: #111827; border: 2px solid #374151; border-radius: 16px; font-size: 18px; color: #9ca3af; }")
        self._add_shadow(self.video_label1)

        self.video_label2 = QLabel("Remote Connection\n(Not started)")
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.video_label2.setScaledContents(True)
        self.video_label2.setMinimumSize(640, 480)
        self.video_label2.setStyleSheet(
            "QLabel { background-color: #111827; border: 2px solid #374151; border-radius: 16px; font-size: 18px; color: #9ca3af; }")
        self._add_shadow(self.video_label2)

        # ── 結果區 ──
        self.predicted_text_label = QLabel("辨識結果:")
        self.predicted_text_label.setProperty("subtle", True)
        self.predicted_text = QLineEdit()
        self.predicted_text.setReadOnly(True)

        # 來源標籤：顯示 Whisper 或 LipNet
        self.source_label = QLabel("來源: --")
        self.source_label.setMinimumWidth(130)
        self.source_label.setStyleSheet("color: #9ca3af; font-size: 14px; font-weight: 600;")

        self.translated_text_label = QLabel("翻譯結果:")
        self.translated_text_label.setProperty("subtle", True)
        self.translated_text = QLineEdit()
        self.translated_text.setReadOnly(True)
        self.subtitle_label = QLabel("")

        # ── 版面 ──
        group_conn = QGroupBox("Connection")
        conn_box = QVBoxLayout(group_conn)
        conn_box.setSpacing(10)
        conn_box.addLayout(row1)
        conn_box.addLayout(row2_inputs)
        conn_box.addLayout(row2_buttons)

        group_video = QGroupBox("Video")
        video_grid = QGridLayout(group_video)
        video_grid.setHorizontalSpacing(16)
        video_grid.addWidget(self.video_label1, 0, 0)
        video_grid.addWidget(self.video_label2, 0, 1)

        status_bar = QHBoxLayout()
        status_bar.setSpacing(12)
        status_bar.addWidget(self.predicted_text_label)
        status_bar.addWidget(self.predicted_text, 1)
        status_bar.addWidget(self.source_label)
        status_bar.addWidget(self.translated_text_label)
        status_bar.addWidget(self.translated_text, 1)

        root = QVBoxLayout(self)
        root.setSpacing(18)
        root.addWidget(title)
        root.addWidget(group_conn)
        root.addWidget(group_video, 1)
        root.addLayout(status_bar)
        self.setLayout(root)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 15)

        self.video_display = VideoDisplay(self.capture)
        self.subtitle_receiver = SubtitleReceiver(target_text_port, self.update_subtitle)
        self.video_display.frame_ready.connect(self.update_local_video)
        self.video_display.start()
        self.subtitle_receiver.start()

    def update_subtitle(self, subtitle):
        if subtitle == "*** Connection from Target IP is allowed ***":
            self.predicted_text.setText(subtitle)
        else:
            self.subtitle_label.setText(subtitle)

    def start_threads(self):
        sender = SubtitleSender(target_ip, local_text_port, "*** Connection from Target IP is allowed ***")
        sender.start(); sender.join()
        if sender.success:
            self.video_label2.setText("Waiting for remote connection...")
            self.video_receiver = VideoReceiver(video_port)
            self.video_receiver.frame_ready.connect(self.update_remote_video)
            self.video_receiver.start()
        else:
            self.video_label2.setText("Target device is not yet started.")

    def connect_threads(self):
        tport = self.target_port.text().split('/')[0].strip()
        if self.predicted_text.text() and self.predicted_text.text() != "----- CONNECTION FAILED -----":
            self.video_display.enable_send(self.target_ip.text(), int(tport))
            s = SubtitleSender(self.target_ip.text(), local_text_port, "----- CONNECTION SUCCESSFUL -----")
            s.start(); s.join()
            self.predicted_text.setText("----- CONNECTION SUCCESSFUL -----")
        else:
            s = SubtitleSender(self.target_ip.text(), local_text_port, "----- CONNECTION FAILED -----")
            s.start(); s.join()
            self.predicted_text.setText("----- CONNECTION FAILED -----")

    def disconnect_threads(self):
        self.predicted_text.setText("----- DISCONNECTED -----")
        s = SubtitleSender(target_ip, local_text_port, "----- DISCONNECTED -----")
        s.start(); s.join()
        threading.Event().wait(1.0)
        self.video_display.disable_send()

    def select_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", "", "影片檔案 (*.mp4 *.mpg *.avi *.mov)")
        if path:
            new_cap = cv2.VideoCapture(path)
            self.capture.release()
            self.capture = new_cap
            self.video_display.capture = new_cap
            self.video_label1.setText(f"影片：{os.path.basename(path)}")

    def start_prediction(self):
        self.predicted_text.setText("(Prediction in progress...)")
        self.prediction_done = False

    def update_local_video(self, frame):
        frame = self.process_frame(frame)
        height, width, _ = frame.shape
        text = self.predicted_text.text()
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        def draw_centered(t, color):
            (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, t, ((width - tw) // 2, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if text == "----- CONNECTION SUCCESSFUL -----":
            draw_centered(text, (0, 255, 0))
        elif text in ["----- CONNECTION FAILED -----", "----- DISCONNECTED -----"]:
            draw_centered(text, (255, 0, 0))
        elif text in ["(Prediction in progress...)", "(Waiting for prediction to start...)"]:
            draw_centered(text, (128, 128, 128))
        else:
            draw_centered(text, (255, 255, 255))
        h, w, ch = frame.shape
        self.video_label1.setPixmap(
            QPixmap.fromImage(QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)))

    def update_remote_video(self, frame):
        height, width, _ = frame.shape
        subtitle = self.subtitle_label.text()
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        if subtitle:
            (tw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, subtitle, ((width - tw) // 2, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        h, w, ch = frame.shape
        self.video_label2.setPixmap(
            QPixmap.fromImage(QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)))

    def process_frame(self, frame):
        global predicted_text
        if self.prediction_done:
            return frame

        # ── 第一幀：開始計時 + 開始錄音 ──
        if self.frame_counter == 0:
            self.start_time = time.time()
            self.audio_recorder.start()
            s = SubtitleSender(target_ip, local_text_port, "(Waiting for remote prediction...)")
            s.start(); s.join()

        self.frame_counter = (self.frame_counter + 1) % 75
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        detected = False
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            pad = int(fh * 0.1)
            fx1, fy1 = max(0, fx - pad), max(0, fy - pad)
            fx2, fy2 = min(frame.shape[1], fx + fw + pad), min(frame.shape[0], fy + fh + pad)
            face_crop = frame_bgr[fy1:fy2, fx1:fx2]
            results = yolo_model(face_crop, verbose=False)
            for r in results:
                for box in r.boxes:
                    if float(box.conf[0]) >= 0.1:
                        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
                        x1, y1, x2, y2 = fx1+lx1, fy1+ly1, fx1+lx2, fy1+ly2
                        self.box_history.append((x1, y1, x2, y2))
                        detected = True; break
                if detected: break

        if not detected:
            results = yolo_model(frame_bgr, verbose=False)
            for r in results:
                for box in r.boxes:
                    if float(box.conf[0]) >= 0.1:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        self.box_history.append((x1, y1, x2, y2))
                        detected = True; break
                if detected: break

        self.no_detect_count = 0 if detected else self.no_detect_count + 1

        if len(self.box_history) > 0 and self.no_detect_count <= 20:
            x1 = int(sum(b[0] for b in self.box_history) / len(self.box_history))
            y1 = int(sum(b[1] for b in self.box_history) / len(self.box_history))
            x2 = int(sum(b[2] for b in self.box_history) / len(self.box_history))
            y2 = int(sum(b[3] for b in self.box_history) / len(self.box_history))
            lip_region = frame[y1:y2, x1:x2]
            if lip_region.size > 0:
                lip_region = preprocess_lip_shape(lip_region)
                lip_region = tf.image.rgb_to_grayscale(lip_region)
                self.lip_regions.append(lip_region)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(self.lip_regions) >= 75:
            self.lip_regions = self.lip_regions[:75]

            # ── 停止錄音 ──
            audio_array = self.audio_recorder.stop()

            # ── LipNet 預測 ──
            lip_region_processed = preprocess_lip_std(self.lip_regions)
            is_grid = self.mode_combo.currentText() == "GRID 模式（s1）"
            active_model = lipnet_model_s1 if is_grid else lipnet_model_s99
            yhat = active_model.predict(tf.expand_dims(lip_region_processed, axis=0))
            decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
            predicted_texts = [
                tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8')
                for sentence in decoded]
            raw_result = " ".join(predicted_texts) if predicted_texts else "No prediction"
            lipnet_result = correct_sentence(raw_result)

            # ── Whisper 融合 ──
            final_result, source = fuse_results(lipnet_result, audio_array, is_grid)

            elapsed = time.time() - self.start_time
            print(f"LipNet: {lipnet_result} | 融合後: {final_result} | 來源: {source} | {elapsed:.1f}s")

            # ── 更新 UI ──
            self.predicted_text.setText(final_result)
            color = "#34d399" if source == 'Whisper' else "#60a5fa"
            self.source_label.setText(f"來源: {source}")
            self.source_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 600;")

            selected_lang = LANGUAGES.get(self.lang_combo.currentText())
            translated_result = ""
            if selected_lang and final_result not in ["No prediction", ""]:
                try:
                    translated_result = translator.translate(final_result, dest=selected_lang).text
                except Exception as e:
                    translated_result = f"翻譯失敗: {e}"
            self.translated_text.setText(translated_result)

            send_text = f"{final_result} | {translated_result}" if translated_result else final_result
            s = SubtitleSender(target_ip, local_text_port, send_text)
            s.start(); s.join()

            self.prediction_done = True
            self.lip_regions = []
            self.start_time = None

        cv2.putText(frame, f"Frame: {self.frame_counter}/75", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = Endpoint()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        with open(r'C:\Users\Tno\error_log.txt', 'w') as f:
            f.write(traceback.format_exc())
        raise
