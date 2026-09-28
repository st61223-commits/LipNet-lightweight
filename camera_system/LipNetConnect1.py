import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pathlib, sys
from collections import deque
pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

SLIDE_STEP = 15  # 每 15 幀預測一次（滑動視窗）
# 2026-09-03調整：原本是5。實驗(scripts/experiment_slide_step.py)用8支測試影片
# 比較SLIDE_STEP=5/10/15/25：15能讓LipNet的CPU推論總耗時降到約38%(46.28s→17.69s)，
# 字幕收斂到最終正確答案的速度幾乎沒變(55.6幀→56.2幀)，正確率不變(8/8)。
# 25雖然耗時更低但收斂變慢(68.8幀)，15是目前測到的甜蜜點。
# ⚠️只在8支「已知答對」的測試影片上驗證過，樣本數小，之後若在舊筆電上發現不夠準，
# 可以考慮調回5~10之間再試。
MIN_FRAMES = 15  # 緩衝區至少集滿這麼多「真實幀」就開始提早預測（前幀複製補滿到75幀）

import cv2
import tensorflow as tf
# 2026-09-18修正：cv2要在tensorflow之前import。打包成PyInstaller單一資料夾exe後
# 執行會出現"ImportError: DLL load failed while importing cv2: DLL初始化例行程序失敗"，
# 直接跑.py反而正常——這是TensorFlow跟conda版OpenCV在同一個行程共用DLL目錄時常見的
# 衝突模式：TensorFlow初始化時載入的執行期DLL(如MKL/OpenMP相關)搶先佔用了cv2需要的
# 版本，等cv2才要初始化自己的DLL時就初始化失敗。cv2先import可以避開這個載入順序問題。
import numpy as np
from ultralytics import YOLO
import time
import pickle
import threading
import math
import socket
import struct
import pyaudio
import sounddevice as sd
import psutil

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from typing import List
from googletrans import Translator
import httpx

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QSizePolicy, QGraphicsDropShadowEffect, QComboBox, QApplication, QFileDialog
)
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ===== 翻譯設定 =====
# 2026-09-18新增timeout：Translator()原本沒設定逾時，googletrans是非官方套件，
# 連不上Google翻譯伺服器時（防火牆/網路問題）translate()會卡住不返回，而這通call
# 是在LipNetInferenceThread背景執行緒裡跑、又搭配inference_busy旗標，一旦卡住就會
# 讓辨識功能表面上「沒回應」。加5秒逾時，逾時就丟例外（run()裡已經有try/except接住）。
translator = Translator(timeout=httpx.Timeout(5.0))

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

# ===== GPU 設定 =====
physical_devices = tf.config.list_physical_devices('GPU')
# 沒有GPU（例如老筆電、低階硬體）時，改用ONNX在CPU上推論。
# 實測CPU推論：.h5 300.3ms、ONNX 158.1ms（快1.9倍），檔案也從47.15MB縮小到15.70MB，
# 正確率幾乎無差異（全量7925筆測試：76.6%→76.7%）。詳見專題「量化/ONNX」實驗記錄。
USE_ONNX = len(physical_devices) == 0
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

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
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

H5_MODEL_PATH = 'C:\\Users\\Tno\\OneDrive\\Lipnet_nchu\\LipNet\\models\\trained_model_grid_multi_width05_yolov8.h5'
ONNX_MODEL_PATH = 'C:\\Users\\Tno\\OneDrive\\Lipnet_nchu\\LipNet\\models\\width05_yolov8.onnx'
# 2026-09-18再更新：從YOLOv10退回YOLOv8。9/18先換過一次YOLOv10(LipNet WER
# 校正後98.0%→98.6%，準確率小勝)，但補測「偵測器本身」的CPU推論速度後發現
# YOLOv10比YOLOv8慢(平均41.3ms vs 36.9ms，慢約12%)，不是官方宣稱的「去NMS後
# 處理更快」，推測是lip偵測這種小尺度單類別任務還顯現不出YOLOv10的架構優勢。
# 偵測器是每一幀畫面都要跑一次的即時瓶頸，即時體感速度的重要性大於WER這0.6個
# 百分點的微幅進步，因此改為維持YOLOv8。LipNet模型(下方H5/ONNX)同步退回
# width05_yolov8，維持訓練資料跟即時偵測用同一版偵測器，避免Training-Inference
# Mismatch重演(同2026-09-03那次YOLOv5→YOLOv8的教訓)。
# 詳見 project_lipnet_thesis_metrics.md。width05_yolov10.h5/onnx/yolov10_lip
# 權重保留未刪除，可隨時再換回來。

lipnet_model = None
onnx_session = None
onnx_input_name = None
onnx_output_name = None

try:
    if USE_ONNX:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        onnx_input_name = onnx_session.get_inputs()[0].name
        onnx_output_name = onnx_session.get_outputs()[0].name
        print(f'[LipNet] 未偵測到GPU，改用ONNX（CPU）推論：{ONNX_MODEL_PATH}')
    else:
        lipnet_model = load_model(H5_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
        print(f'[LipNet] 偵測到GPU，使用原本.h5模型：{H5_MODEL_PATH}')
except Exception as e:
    import traceback
    with open('C:\\Users\\Tno\\error_log.txt', 'w') as f:
        f.write("LipNet載入失敗:\n" + traceback.format_exc())
    raise


def run_lipnet_inference(lip_region_processed):
    """跑一次LipNet推論，回傳 yhat（shape (1,75,41) 的 numpy array）。
    有GPU時走原本的Keras .h5路徑（不變）；沒有GPU（老筆電/低階硬體）時走ONNX路徑，
    是專題「低階硬體可用」目標的實際應用。
    """
    x = tf.expand_dims(lip_region_processed, axis=0)
    if USE_ONNX:
        x_np = x.numpy().astype(np.float32)
        return onnx_session.run([onnx_output_name], {onnx_input_name: x_np})[0]
    else:
        # 2026-09-18修正：改用直接呼叫模型(model(x))取代model.predict(x)。
        # 實機測試連續OOM當機，faulthandler抓到真正錯誤堆疊是cv2在capture.read()
        # 配不到300KB記憶體，代表系統記憶體被榨乾，不是單純的執行緒卡死。
        # 追查發現是Keras/TF官方文件記載的已知問題：在迴圈裡重複呼叫model.predict()
        # （這裡滑動視窗每15幀呼叫一次）會持續增加記憶體且不確實釋放，符合觀察到的
        # 「模型讀取成功→跑完第一次推論→用一陣子後才OOM」模式。官方建議小量輸入、
        # 重複呼叫的情境改用model(x)直接呼叫，可避開這個洩漏。
        return lipnet_model(x, training=False).numpy()

try:
    yolo_model = YOLO('C:\\Users\\Tno\\OneDrive\\Lipnet_nchu\\yolov8_lip\\lip_detect\\weights\\best.pt')
except Exception as e:
    import traceback
    with open('C:\\Users\\Tno\\error_log.txt', 'w') as f:
        f.write("YOLOv8載入失敗:\n" + traceback.format_exc())
    raise

face_cascade = cv2.CascadeClassifier(r'C:\Users\Tno\miniconda3\envs\py39\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

# ===== 前處理函式 =====
input_shape = (75, 46, 140, 1)
predicted_text = ""

def preprocess_lip_shape(lip_region):
    return tf.image.resize(lip_region, (input_shape[1], input_shape[2]))

def preprocess_lip_std(lip_regionf):
    mean = tf.math.reduce_mean(lip_regionf)
    std = tf.math.reduce_std(tf.cast(lip_regionf, tf.float32))
    return tf.cast((lip_regionf - mean), tf.float32) / std

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
        # 2026-09-18修正：加time.sleep節流到25fps。使用者實機測試發現「用攝影機不會當掉，
        # 用data資料夾的影片檔測就會」——差異在於攝影機capture.read()本身受鏡頭硬體速度
        # 限制(約25fps天然節流)，但讀影片檔時capture.read()只是解碼硬碟檔案，電腦能跑多快
        # 就讀多快(可能是攝影機的數十倍)，導致process_frame()裡的YOLO偵測被以遠超正常的
        # 頻率呼叫，記憶體/CPU被榨乾的速度也跟著暴增。這裡固定節流到25fps(跟下方
        # self.capture.set(cv2.CAP_PROP_FPS, 25)給攝影機用的目標速度一致)，讓影片檔跟
        # 攝影機用同一個真實時間節奏播放，不要用電腦全速狂讀。
        frame_interval = 1.0 / 25
        while True:
            t_start = time.time()
            ret, frame = self.capture.read()
            if not ret:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(frame_interval)
                continue
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                self.frame_ready.emit(self.frame)
                if self.status and self.enable:
                    self.enable = False
                    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                if self.status and not self.enable:
                    self.transfer_img()
            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

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
                (self.ip, self.port)
            )
            array_pos_start = array_pos_end
            num_of_segments -= 1

    def enable_send(self, ip, port):
        self.ip = ip
        self.port = port
        self.status = True

    def disable_send(self):
        self.status = False
        self.enable = True


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
        last_good_frame = None
        self.dump_buffer(self.server_socket)
        while True:
            seg, _ = self.server_socket.recvfrom(self.MAX_DGRAM)
            if struct.unpack('B', seg[0:1])[0] > 1:
                data += seg[1:]
            else:
                data += seg[1:]
                try:
                    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
                    if frame is not None:
                        last_good_frame = frame
                        self.frame_ready.emit(frame)
                    elif last_good_frame is not None:
                        self.frame_ready.emit(last_good_frame)
                except:
                    if last_good_frame is not None:
                        self.frame_ready.emit(last_good_frame)
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
        except Exception as e:
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


class LipNetInferenceThread(QThread):
    """在背景執行緒跑一次LipNet推論（含解碼、翻譯、送字幕），不佔用GUI主執行緒。
    背景：CPU（無GPU的低階硬體）單次推論可能比SLIDE_STEP的間隔還久，如果像原本一樣
    寫在process_frame裡直接同步呼叫，會卡住整個介面的事件迴圈（畫面、按鈕都會凍住），
    等於整個程式在推論期間「當機」。改成背景執行緒後，推論跑多久都不會擋住畫面更新，
    再搭配Endpoint裡的inference_busy旗標「上一次沒跑完就跳過這次」，避免慢速硬體上
    推論工作越堆越多。
    """
    result_ready = pyqtSignal(str, str, float, str)

    def __init__(self, frames_list, mode_tag, selected_lang):
        super().__init__()
        self.frames_list = frames_list
        self.mode_tag = mode_tag
        self.selected_lang = selected_lang

    def run(self):
        # 整個run()包一層try/except是必要的：QThread.run()裡沒接住的例外只會印到
        # console、不會讓result_ready訊號發射。on_inference_result是唯一會把
        # inference_busy重設回False的地方，一旦這裡中途出錯又沒有emit，
        # inference_busy會卡在True，等於這次意外之後整個滑動視窗辨識永久停擺
        # （之後每次都被「上一次還沒跑完」擋掉），使用者還很難注意到（沒有跳錯誤視窗）。
        # 所以任何例外都要在這裡攔下來，仍然emit一次讓忙碌旗標解除。
        t0 = time.time()
        try:
            lip_region_processed = preprocess_lip_std(self.frames_list)
            yhat = run_lipnet_inference(lip_region_processed)
            decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
            predicted_texts = [tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded]
            raw_result = " ".join(predicted_texts) if predicted_texts else "No prediction"
            english_result = correct_sentence(raw_result)

            if self.selected_lang and english_result not in ["No prediction", ""]:
                try:
                    translated_result = translator.translate(english_result, dest=self.selected_lang).text
                except Exception as e:
                    translated_result = f"翻譯失敗: {e}"
            else:
                translated_result = ""

            send_text = f"{english_result} | {translated_result}" if translated_result else english_result
            try:
                s = SubtitleSender(target_ip, local_text_port, send_text)
                s.start(); s.join()
            except Exception as e:
                print(f"[LipNet] 字幕傳送失敗（不影響本地顯示）: {e}")
        except Exception as e:
            import traceback
            print(f"[LipNet] 推論執行緒發生例外: {e}")
            traceback.print_exc()
            english_result, translated_result = f"[推論錯誤: {e}]", ""

        self.result_ready.emit(english_result, translated_result, time.time() - t0, self.mode_tag)


# ===== 主介面 =====
class Endpoint(QWidget):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.lip_regions = deque(maxlen=75)
        self.start_time = None
        self.prediction_done = True
        self.box_history = deque(maxlen=1)
        # 2026-09-03修正：maxlen從10降到1，等於拿掉10幀box平滑，只用最新一次偵測
        # 到的框，不跟過去的框做加權平均。原因見process_frame內的說明。
        self.last_box = None
        self.no_detect_count = 0
        self.local_ready = True
        self.remote_ready = True
        self.inference_busy = False
        self._infer_thread = None
        global predicted_text
        predicted_text = ""
        self.init_ui()

    def _apply_button_style(self, btn, kind="primary"):
        if kind == "primary":
            style = """
            QPushButton {
                border: none; border-radius: 10px; padding: 10px 18px;
                background-color: #2563eb; color: #f9fafb;
                font-size: 16px; font-weight: 600;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e3a8a; }
            """
        else:
            style = """
            QPushButton {
                border: none; border-radius: 10px; padding: 10px 18px;
                background-color: #4b5563; color: #f9fafb;
                font-size: 16px; font-weight: 600;
            }
            QPushButton:hover { background-color: #374151; }
            QPushButton:pressed { background-color: #1f2937; }
            """
        btn.setStyleSheet(style)
        btn.setCursor(Qt.PointingHandCursor)

    def _add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(Qt.black)
        widget.setGraphicsEffect(shadow)

    def init_ui(self):
        self.setWindowTitle('LipNet Connect 1')
        self.setGeometry(80, 60, 1400, 900)
        self.setStyleSheet("""
            QWidget { background: #0f172a; color: #e5e7eb; }
            QLineEdit { background: #111827; border: 1px solid #374151; color: #e5e7eb; border-radius: 10px; padding: 8px 10px; font-size: 16px; }
            QComboBox { background: #111827; border: 1px solid #374151; color: #e5e7eb; border-radius: 10px; padding: 8px 10px; font-size: 16px; }
            QLabel[head="true"] { font-size: 28px; font-weight: 800; color: #f8fafc; }
            QLabel[subtle="true"] { color: #9ca3af; font-size: 14px; }
            QGroupBox { border: 1px solid #293042; border-radius: 14px; margin-top: 18px; }
            QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 2px 6px; color: #a5b4fc; font-weight: 700; }
        """)

        title = QLabel("LipNet Connect System")
        title.setProperty("head", True)
        title.setAlignment(Qt.AlignCenter)

        self.local_ip_label = QLabel("Local IP:")
        self.local_ip_label.setProperty("subtle", True)
        self.local_ip = QLineEdit(str(local_ip))
        self.local_port_label = QLabel("Local Camera / Text Port:")
        self.local_port_label.setProperty("subtle", True)
        self.local_port = QLineEdit(str(video_port) + " / " + str(local_text_port))
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
        self.target_port = QLineEdit(str(target_port) + " / " + str(target_text_port))
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
        backend_label = "CPU / ONNX" if USE_ONNX else "GPU / .h5"
        mode_text = f"F_Width0.5 模型（輕量化，多說話者，{backend_label}）"
        self.mode_combo.addItem(mode_text)
        self.mode_combo.setCurrentText(mode_text)

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

        self.video_label1 = QLabel("Local Camera\n(Not started)")
        self.video_label1.setAlignment(Qt.AlignCenter)
        self.video_label1.setScaledContents(True)
        self.video_label1.setMinimumSize(640, 480)
        self.video_label1.setStyleSheet("QLabel { background-color: #111827; border: 2px solid #374151; border-radius: 16px; font-size: 18px; color: #9ca3af; }")
        self._add_shadow(self.video_label1)

        self.video_label2 = QLabel("Remote Connection\n(Not started)")
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.video_label2.setScaledContents(True)
        self.video_label2.setMinimumSize(640, 480)
        self.video_label2.setStyleSheet("QLabel { background-color: #111827; border: 2px solid #374151; border-radius: 16px; font-size: 18px; color: #9ca3af; }")
        self._add_shadow(self.video_label2)

        self.predicted_text_label = QLabel("辨識結果:")
        self.predicted_text_label.setProperty("subtle", True)
        self.predicted_text = QLineEdit()
        self.predicted_text.setReadOnly(True)
        self.translated_text_label = QLabel("翻譯結果:")
        self.translated_text_label.setProperty("subtle", True)
        self.translated_text = QLineEdit()
        self.translated_text.setReadOnly(True)
        self.subtitle_label = QLabel("")

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
        self.capture.set(cv2.CAP_PROP_FPS, 25)

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
        if self.prediction_done:
            self.prediction_done = False
            self.frame_counter = 0
            self.lip_regions.clear()
            self.predicted_text.setText("(Prediction in progress...)")
            self.predict_button.setText("Stop Prediction")
        else:
            self.prediction_done = True
            self.predict_button.setText("Start Prediction")

    def update_local_video(self, frame):
        frame = self.process_frame(frame)
        height, width, _ = frame.shape
        text = self.predicted_text.text()
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        def draw_centered(t, color):
            (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, t, ((width - tw) // 2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if text == "----- CONNECTION SUCCESSFUL -----":
            draw_centered(text, (0, 255, 0))
        elif text in ["----- CONNECTION FAILED -----", "----- DISCONNECTED -----"]:
            draw_centered(text, (255, 0, 0))
        elif text in ["(Prediction in progress...)", "(Waiting for prediction to start...)"]:
            draw_centered(text, (128, 128, 128))
        else:
            draw_centered(text, (255, 255, 255))
        h, w, ch = frame.shape
        self.video_label1.setPixmap(QPixmap.fromImage(QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)))

    def update_remote_video(self, frame):
        height, width, _ = frame.shape
        subtitle = self.subtitle_label.text()
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        if subtitle:
            (tw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, subtitle, ((width - tw) // 2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        h, w, ch = frame.shape
        self.video_label2.setPixmap(QPixmap.fromImage(QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)))

    def process_frame(self, frame):
        global predicted_text
        if self.prediction_done:
            return frame
        self.frame_counter += 1
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2026-09-03修正：拿掉人臉偵測預篩選(Haar cascade先框臉、YOLO只在臉部
        # 範圍內找)，改成直接對全畫面做YOLO偵測——比照訓練資料產生方式
        # (train_grid_multi.py/regenerate_cache_yolov8.py都是這樣做)。
        #
        # 背景：診斷實驗(diagnose_pipeline_factors.py)用8支「模型應該要答對」
        # 的影片逐一拆解四種管線設定，結果：
        #   純YOLO(比照訓練)                    8/8 全對
        #   只調低信心度門檻(0.1)                 8/8 全對（門檻不是問題）
        #   只加10幀box平滑                      3/8（大幅退步）
        #   只加人臉偵測預篩選                    2/8（大幅退步）
        #   人臉偵測+平滑+低門檻(=原本即時管線)     0/8（全部答錯）
        # 證實人臉偵測預篩選跟box平滑「都各自」是問題的一部分，拿掉兩者後
        # 同樣測試恢復到8/8。詳見 project_lipnet.md、實驗記錄.md「實驗七」。
        #
        # ⚠️ 已知取捨：人臉偵測+box平滑原本可能是為了處理真實攝影機畫面的
        # 雜訊/晃動/誤判而加的（離線測試影片是乾淨的GRID語料庫錄影，沒有
        # 這類真實世界雜訊），拿掉後在真實攝影機環境下的穩定性沒有實測過，
        # 實機測試時請特別留意：如果真的遇到偵測誤判到背景物體之類的問題，
        # 可以考慮把人臉偵測加回來，但box平滑(box_history maxlen)建議維持
        # 縮小，兩者造成的傷害以人臉偵測稍微更大(2/8 vs 3/8)。
        detected = False
        # 2026-09-18修正：加stream=True。實機測試時python.exe記憶體一路長到10GB以上，
        # 導致越用越卡、最後任何操作(含按「結束預測」)看起來都像當掉，其實是系統記憶體
        # 被榨乾造成的假當機。YOLO偵測是process_frame()裡每一幀畫面都會呼叫一次(不像
        # LipNet推論有SLIDE_STEP間隔)，Ultralytics官方文件明確警告：在迴圈裡重複呼叫
        # model()處理連續畫面(例如逐幀讀webcam)時，不加stream=True會把每次呼叫的結果
        # 都留在記憶體裡累積，是官方文件裡指名的記憶體洩漏成因。加stream=True後改成用
        # generator逐幀處理、用完即釋放，下面既有的for迴圈寫法不用改。
        results = yolo_model(frame_bgr, verbose=False, stream=True)
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= 0.1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    self.box_history.append((x1, y1, x2, y2))
                    detected = True
                    break
            if detected:
                break

        if detected:
            self.no_detect_count = 0
        else:
            self.no_detect_count += 1

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
        if len(self.lip_regions) >= MIN_FRAMES and self.frame_counter % SLIDE_STEP == 0:
            if self.inference_busy:
                # 上一次推論還沒跑完就先跳過這次滑動視窗。
                # 低階CPU上單次推論可能比SLIDE_STEP的間隔還久，若硬要每次都排一個新的推論，
                # 工作會越堆越多、越辨識越落後畫面，所以寧可跳過幾次滑動視窗，等目前這次跑完再繼續。
                print(f"[LipNet] 推論還沒跑完，跳過這次（buf={len(self.lip_regions)}）")
            else:
                frames_list = list(self.lip_regions)
                is_early = len(frames_list) < 75
                if is_early:
                    # 前幀複製：真實幀還沒集滿75幀，複製最後一幀補滿，讓模型可以提早開始預測
                    # 原理見「封包遺失補救」實驗：CTC對重複幀容忍度高，會解讀成嘴唇停頓，不會讓辨識完全失敗
                    pad_count = 75 - len(frames_list)
                    frames_list = frames_list + [frames_list[-1]] * pad_count
                mode_tag = f"提早預測 {len(self.lip_regions)}/75" if is_early else "完整75幀"
                selected_lang = LANGUAGES.get(self.lang_combo.currentText())
                self.inference_busy = True
                self._infer_thread = LipNetInferenceThread(frames_list, mode_tag, selected_lang)
                self._infer_thread.result_ready.connect(self.on_inference_result)
                # 2026-09-18新增：每次滑動視窗都會覆蓋self._infer_thread指向新物件，
                # 舊的QThread物件失去Python參考後由GC回收，若GC剛好在錯誤的執行緒
                # 上觸發C++端解構，PyQt5會噴"Timers cannot be stopped from another
                # thread"警告，嚴重時整個程式被Qt安靜終止（無Python traceback、
                # 無Windows當機記錄，跟實機測試觀察到的當機現象吻合）。改用
                # finished訊號接deleteLater，讓Qt在正確的時機點、正確的執行緒
                # 上做銷毀，而不是交給Python GC隨機決定。
                self._infer_thread.finished.connect(self._infer_thread.deleteLater)
                self._infer_thread.start()
        buf = len(self.lip_regions)
        cv2.putText(frame, f"Buf: {buf}/75", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def on_inference_result(self, english_result, translated_result, elapsed, mode_tag):
        """LipNetInferenceThread推論完成後的回呼，跑在GUI主執行緒（Qt跨執行緒signal自動排程），
        可以安全更新畫面文字。收到結果才把inference_busy解除，讓下一次滑動視窗可以再觸發推論。"""
        self.predicted_text.setText(english_result)
        self.translated_text.setText(translated_result)
        self.inference_busy = False
        print(f"辨識[{mode_tag}]: {english_result} | 翻譯: {translated_result} | 用時: {elapsed:.2f}s")


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = Endpoint()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        with open('C:\\Users\\Tno\\error_log.txt', 'w') as f:
            f.write(traceback.format_exc())
        raise
