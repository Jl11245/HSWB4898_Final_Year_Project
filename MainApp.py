import sys
import os
import re
import math
import json
import csv
import subprocess
import platform
from collections import deque
import threading
import queue
from pathlib import Path
import time
import sqlite3
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QInputDialog
from PyQt6.QtCore import QUrl

import torch
import torchvision
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
elif hasattr(torch.serialization, "safe_globals"):
    torch.serialization.safe_globals().add(np.core.multiarray.scalar)

from PyQt6.QtCore import (
    Qt, QTimer, QDateTime, QSize, QThread, pyqtSignal
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QListWidget, QGridLayout, QGroupBox,
    QFrame, QScrollArea, QMessageBox, QSizePolicy, QListWidgetItem
)
from PyQt6.QtGui import QImage, QFontMetrics
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from PyQt6.QtMultimediaWidgets import QVideoWidget

from ops.models import TSN
from ops.transforms import (
    Stack, ToTorchFormatTensor
)

#Global setting
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(BASE_DIR / "config.json")

LABELS = ["Normal", "SelfExtubated", "RailLower", "Agitated"]
EVENT_THRESHOLD = 0.5            
MIN_EVENT_HOLD_SEC = 1.0

EVENT_LOG_CSV = BASE_DIR / "events_log.csv"
DB_PATH = BASE_DIR / "Login" / "login_data.db"
sys.path.append(str(BASE_DIR / "temporal-shift-module"))


if not EVENT_LOG_CSV.exists():
    try:
        with open(EVENT_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "start_ts", "end_ts", "bed_id", "bed_name",
                "duration_sec", "trigger_reason", "severity"
            ])
    except Exception as e:
        print(f"[LOG] Failed to create events CSV: {e}")


#load config
def default_config():
    return {
        "total_videos": 4,
        "per_page": 4,
        "auto_interval_sec": 0,
        "bed_names": [f"Bed {i}" for i in range(1, 5)],
        "video_paths": [""] * 4,
    }

def load_config():
    print("[CONFIG] load_config(): using default_config()")
    return default_config()

def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[CONFIG] Saved to {CONFIG_PATH}")
    except Exception as e:
        print(f"[CONFIG] Save failed: {e}")


#video ratio
class AspectRatioVideoWidget(QVideoWidget):
    def sizeHint(self):
        parent = self.parentWidget()
        if not parent:
            return super().sizeHint()
        w = parent.width() if parent.width() > 0 else 480
        h = int(w * 9 / 16)
        return QSize(w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)


#login page
def verify_user(username, password):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            'SELECT "Identity" FROM "Login_Data" WHERE "User Name"=? AND "Password"=?',
            (username, password)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception as e:
        print("[LOGIN DB ERROR]", e)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    
class LoginPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(100, 80, 100, 80)

        title = QLabel("Login Page")
        title.setStyleSheet("font-size:24px; font-weight:600;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignLeft)

        form_box = QGroupBox()
        form_layout = QVBoxLayout()
        form_box.setLayout(form_layout)
        form_box.setStyleSheet("QGroupBox { background: #f5f5f5; padding: 18px; }")

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Login Name")
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pwd_input.setPlaceholderText("Password")

        form_layout.addWidget(QLabel("Login Name :"))
        form_layout.addWidget(self.user_input)
        form_layout.addSpacing(8)
        form_layout.addWidget(QLabel("Password :"))
        form_layout.addWidget(self.pwd_input)

        layout.addSpacing(30)
        layout.addWidget(form_box)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        login_btn = QPushButton("Login")
        login_btn.setFixedWidth(140)
        login_btn.clicked.connect(self.on_login)
        btn_layout.addWidget(login_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def on_login(self):
        username = self.user_input.text().strip()
        password = self.pwd_input.text().strip()

        err_msg = "User name or Password is wrong."

        if not username or not password:
            QMessageBox.warning(self, "Login Failed", err_msg)
            print("[LOGIN] Failed - empty username or password")
            return

        identity = verify_user(username, password)
        if identity:
            print(f"[LOGIN] Success, user = {username}, identity = {identity}")
            self.stacked_widget.current_user = username
            self.stacked_widget.main_page.update_user_info(username, identity)
            self.stacked_widget.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "Login Failed", err_msg)
            print(f"[LOGIN] Failed - invalid credentials for {username}")


#inference
class InferenceWorker(QThread):
    result_ready = pyqtSignal(int, object) 

    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.running = True
        self.pending_tasks = {}  
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def run(self):
        while self.running:
            with self.cond:
                while not self.pending_tasks and self.running:
                    self.cond.wait()
                if not self.running:
                    break
                
                bed_idx, frames = self.pending_tasks.popitem()
                
            try:
                probs = self.main_page.infer_on_buffer(frames)
                if probs is not None:
                    self.result_ready.emit(bed_idx, probs)
            except Exception as e:
                print(f"[Worker] Inference error: {e}")

    def add_task(self, bed_idx, frames):
        if not self.running:
            return
        with self.cond:
            self.pending_tasks[bed_idx] = frames
            self.cond.notify()

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()


#main page
class MainPage(QWidget):
    def __init__(self, stacked_widget, config):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.applied_settings = config.copy()
        self.current_page_index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_segments = 16
        self.model_path = (r"Model\ckpt.best.pth.tar")
        self.model = None
        self.transform = None

        self.streams = {}           
        self.buffers = {}           
        self.frame_counters = {}    
        
        self.alarm_states = {}      
        self.alarm_start_times = {} 
        self.alarm_active = {}      
        
        self.last_probs = {}        
        self.event_states = {}   
        
        self.handling_states = {}  
        
        self._preview_cache = {}    
        self._preview_items = {}

        self.infer_interval = 2
        self.alarm_trigger_count = 1 

        self.alarm_timer = QTimer()
        self.alarm_timer.timeout.connect(self._blink_alarm)
        self.alarm_timer.start(500)

        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.go_next_page_auto)

        self.bed_widgets = []

        self.init_ui()
        self.load_tsm_model()
        self.apply_settings(self.applied_settings)

    def update_user_info(self, username, identity):
        self.user_name_label.setText(f"User Name : {username}")
        self.identity_label.setText(f"Identity : {identity}")
        
    @staticmethod
    def detect_segment_from_checkpoint(path):
        m = re.search(r"segment(\d+)", path.lower())
        if m:
            return int(m.group(1))
        return 16

    def build_transform(self):
        self.transform = torchvision.transforms.Compose([
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
        ])

    def load_checkpoint(self, path, model):
        from collections import OrderedDict
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        
        new_state = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '').replace('encoder.', '')
            new_state[name] = v
            
        try:
            model.load_state_dict(new_state, strict=False)
        except Exception as e:
            print(f"[MODEL] Load failed: {e}")
            
        return model

    def load_tsm_model(self):
        try:
            seg = self.detect_segment_from_checkpoint(self.model_path)
            self.num_segments = seg

            self.model = TSN(
                num_class=4,
                num_segments=seg,
                modality="RGB",
                base_model="resnet50",
                consensus_type="avg",
                img_feature_dim=256,
                pretrain="imagenet",
                is_shift=True,
                shift_div=8,
                shift_place="blockres",
            )
            self.model = self.load_checkpoint(self.model_path, self.model)
            self.model.to(self.device)
            self.model.eval()
            
            if self.device.type == "cuda":
                try:
                    self.model.half()
                except Exception as e:
                    pass

            if not hasattr(self.model, "input_size"):
                self.model.input_size = 224
            if not hasattr(self.model, "input_mean"):
                self.model.input_mean = [0.485, 0.456, 0.406]
            if not hasattr(self.model, "input_std"):
                self.model.input_std = [0.229, 0.224, 0.225]

            self.build_transform()

            try:
                fname = Path(self.model_path).name
            except Exception:
                fname = os.path.basename(self.model_path)
            self.model_status_label.setText(f"Model: {fname}")

            if not hasattr(self, "inference_worker"):
                self.inference_worker = InferenceWorker(self)
                self.inference_worker.result_ready.connect(self.on_inference_result)
                self.inference_worker.start()

        except Exception as e:
            print(f"[MODEL] Error loading model: {e}")
            self.model = None
            self.transform = None
            self.model_status_label.setText(f"Model: Load Failed ({os.path.basename(self.model_path)})")

    def infer_on_buffer(self, frames_list):
        if self.model is None or self.transform is None:
            return None

        imgs = self.transform(frames_list)
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if not isinstance(imgs, torch.Tensor):
            try:
                imgs = torch.tensor(imgs)
            except Exception as e:
                print(f"[MODEL] Tensor conversion failed: {e}")
                return None

        if imgs.dim() == 3:
            inp = imgs.unsqueeze(0)
        else:
            inp = imgs

        inp = inp.to(self.device, non_blocking=True)

        c = int(inp.shape[1])
        seg = max(1, c // 3)
        mean = torch.tensor(self.model.input_mean * seg, device=self.device).view(1, c, 1, 1)
        std = torch.tensor(self.model.input_std * seg, device=self.device).view(1, c, 1, 1)
        inp = (inp - mean) / std

        if self.device.type == "cuda" and next(self.model.parameters()).dtype == torch.half:
            inp = inp.half()

        with torch.no_grad():
            logits = self.model(inp)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            logits = logits.squeeze(0).float()

            probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)

        return probs.tolist()

    # start UI
    def init_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        top_bar = QHBoxLayout()
        title = QLabel("Patient Monitor")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        top_bar.addWidget(title)
        top_bar.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Go to (bed id)")
        self.search_input.setFixedWidth(160)

        goto_btn = QPushButton("Enter")
        goto_btn.setFixedWidth(80)
        goto_btn.clicked.connect(self.on_goto)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setFixedSize(120, 40)
        settings_btn.clicked.connect(self.open_settings)

        top_bar.addWidget(self.search_input)
        top_bar.addWidget(goto_btn)
        top_bar.addWidget(settings_btn)

        outer.addLayout(top_bar)

        main_h = QHBoxLayout()
        main_h.setSpacing(10)

        events_box = QGroupBox("Events History")
        events_box.setFixedWidth(300)
        events_box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        events_layout = QVBoxLayout()
        events_layout.setContentsMargins(8, 8, 8, 8)
        events_layout.setSpacing(8)
        self.event_history = QListWidget()
        self.event_history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.event_history.setWordWrap(True)
        self.event_history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        events_layout.addWidget(self.event_history, stretch=1)
        view_log_btn = QPushButton("View Log")
        view_log_btn.setFixedWidth(120)
        view_log_btn.clicked.connect(self.open_event_log)
        view_bar = QHBoxLayout()
        view_bar.addStretch()
        view_bar.addWidget(view_log_btn)
        view_bar.addStretch()
        events_layout.addLayout(view_bar)
        events_box.setLayout(events_layout)
        main_h.addWidget(events_box) 


        right_container = QWidget()
        right_v = QVBoxLayout()
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(12)


        self.center_area = QGroupBox()
        self.center_area_layout = QHBoxLayout()
        self.center_area_layout.setContentsMargins(12, 12, 12, 12)
        self.center_area.setLayout(self.center_area_layout)
        self.page_up_btn = QPushButton("◀")
        self.page_up_btn.setFixedWidth(40)
        self.page_up_btn.clicked.connect(self.page_up)
        self.center_area_layout.addWidget(self.page_up_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.grid_container = QGroupBox()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(8)
        self.grid_container.setLayout(self.grid_layout)
        self.grid_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.center_area_layout.addWidget(self.grid_container)
        self.page_down_btn = QPushButton("▶")
        self.page_down_btn.setFixedWidth(40)
        self.page_down_btn.clicked.connect(self.page_down)
        self.center_area_layout.addWidget(self.page_down_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_v.addWidget(self.center_area, stretch=1)


        footer_widget = QWidget()
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(12, 6, 12, 6)
        footer_layout.setSpacing(20)
        alert_box = QGroupBox("Active Alerts")
        a_layout = QVBoxLayout()
        self.alert_label = QLabel("No Active Alerts")
        self.alert_label.setStyleSheet("font-size:18px; font-weight:700;")
        a_layout.addWidget(self.alert_label)
        alert_box.setLayout(a_layout)
        alert_box.setMinimumWidth(250)
        self.user_info_box = QGroupBox("User Info")
        user_layout = QVBoxLayout()
        self.user_name_label = QLabel("User Name : -")
        self.identity_label = QLabel("Identity : -")
        self.user_name_label.setStyleSheet("font-weight:600;")
        self.identity_label.setStyleSheet("font-weight:600;")
        user_layout.addWidget(self.user_name_label)
        user_layout.addWidget(self.identity_label)
        self.user_info_box.setLayout(user_layout)
        self.user_info_box.setMinimumWidth(250)

        status_box = QGroupBox("System / Model Status")
        s_layout = QVBoxLayout()
        self.sys_status_label = QLabel("System: Running")
        self.model_status_label = QLabel("Model: Loading...")
        s_layout.addWidget(self.sys_status_label)
        s_layout.addWidget(self.model_status_label)
        status_box.setLayout(s_layout)
        status_box.setMinimumWidth(320)

        footer_layout.addWidget(alert_box)
        footer_layout.addWidget(self.user_info_box)
        footer_layout.addWidget(status_box)
        footer_layout.addStretch()

        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-size:14px;")
        footer_layout.addWidget(self.time_label, 0, Qt.AlignmentFlag.AlignRight)

        footer_widget.setLayout(footer_layout)
        footer_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        right_v.addWidget(footer_widget, stretch=0)
        right_container.setLayout(right_v)

        main_h.addWidget(right_container, stretch=3)
        outer.addLayout(main_h, stretch=1)

        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)
        
        self.setLayout(outer)


    def build_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            w = self.grid_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        self.bed_widgets = []
        rows, cols = 2, 2

        for r in range(rows):
            for c in range(cols):
                grid_index = len(self.bed_widgets)

                cell_frame = QFrame()
                cell_frame.setStyleSheet("background:#dcdcdc; border:4px solid #bdbdbd; padding:6px;")
                cell_layout = QVBoxLayout(cell_frame)
                cell_layout.setContentsMargins(6, 6, 6, 6)
                cell_layout.setSpacing(6)

                top_bar = QHBoxLayout()
                monitor_result_tag = QLabel("Detected: Normal")
                monitor_result_tag.setMinimumHeight(40)
                monitor_result_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
                monitor_result_tag.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                monitor_result_tag.setStyleSheet(
                    "background:#90ee90; color:black; font-weight:bold; font-size:16px;"
                    "border-radius:4px; padding:6px 10px;"
                )
                top_bar.addWidget(monitor_result_tag)

                ack_btn = QPushButton("✔️ Check")
                ack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                ack_btn.setStyleSheet(
                    "background-color: #f39c12; color: white; font-weight: bold; "
                    "font-size: 14px; border-radius: 4px; padding: 6px 12px;"
                )
                ack_btn.hide()
                ack_btn.clicked.connect(lambda checked, gi=grid_index: self.toggle_handling_state(gi))
                top_bar.addWidget(ack_btn)

                overlay_text = QLabel()
                overlay_text.setMinimumHeight(36)
                overlay_text.setStyleSheet(
                    "background-color: rgba(0,0,0,0.65); color:white; padding:6px;"
                    "border-radius:4px; font-weight:600; font-size: 14px;"
                )
                overlay_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                overlay_text.setWordWrap(False)

                video_container = QFrame()
                video_container.setStyleSheet("background:black;")
                video_layout = QVBoxLayout(video_container)
                video_layout.setContentsMargins(0, 0, 0, 0)

                video_widget = AspectRatioVideoWidget(video_container)
                video_layout.addWidget(video_widget)

                ui_sink = video_widget.videoSink()
                if ui_sink:
                    ui_sink.videoFrameChanged.connect(
                        lambda frame, gi=grid_index: self.on_grid_video_frame(gi, frame)
                    )

                cell_layout.addLayout(top_bar)
                cell_layout.addWidget(overlay_text)
                cell_layout.addWidget(video_container)
                self.grid_layout.addWidget(cell_frame, r, c)

                self.bed_widgets.append({
                    "overlay": overlay_text,
                    "video_widget": video_widget,
                    "frame": cell_frame,
                    "monitor_result_tag": monitor_result_tag,
                    "ack_btn": ack_btn,
                    "bed_index": None,
                })

    def total_pages(self):
        total = self.applied_settings.get("total_videos", 4)
        per = self.applied_settings.get("per_page", 4)
        return max(1, math.ceil(total / per))


    def apply_settings(self, settings):
        print(f"[APPLY] Settings Updated. Initializing Background Streams.")
        self.applied_settings = settings.copy()
        total = self.applied_settings.get("total_videos", 4)
        
        for bed_idx, stream in self.streams.items():
            try:
                stream["player"].stop()
                stream["alarm_player"].stop()
            except Exception as e:
                print(f"[CLEANUP] Failed to stop stream {bed_idx}: {e}")
                
        self.streams.clear()
        self.buffers.clear()
        self.frame_counters.clear()
        self.alarm_states.clear()
        self.alarm_start_times.clear()
        self.alarm_active.clear()
        self._preview_cache.clear()
        self.last_probs.clear()
        self.event_states.clear()
        
        self.handling_states.clear()

        video_paths = self.applied_settings.get("video_paths", [""] * total)
        for bed_idx in range(total):
            player = QMediaPlayer()
            audio = QAudioOutput()
            audio.setMuted(True)
            player.setAudioOutput(audio)

            headless_sink = QVideoSink()
            headless_sink.videoFrameChanged.connect(
                lambda frame, b=bed_idx: self.on_video_frame(b, frame)
            )
            
            alarm_player = QMediaPlayer()
            alarm_audio = QAudioOutput()
            alarm_audio.setVolume(1.0)
            alarm_player.setAudioOutput(alarm_audio)
            alarm_player.setLoops(-1) 
            
            alarm_mp3 = BASE_DIR / "element" / "alarm.mp3"
            if alarm_mp3.exists():
                alarm_player.setSource(QUrl.fromLocalFile(str(alarm_mp3)))
            else:
                print(f"[WARNING] Alarm MP3 not found at {alarm_mp3}. Audio will not play.")

   
            self.streams[bed_idx] = {
                "player": player,
                "audio_output": audio, 
                "headless_sink": headless_sink,
                "alarm_player": alarm_player,
                "alarm_audio": alarm_audio 
            }
            
            self.buffers[bed_idx] = deque(maxlen=self.num_segments)
            self.frame_counters[bed_idx] = 0
            self.alarm_states[bed_idx] = {"count": 0, "active": False}
            self.handling_states[bed_idx] = False
            vp = video_paths[bed_idx] if bed_idx < len(video_paths) else ""

            if vp:
                try:
                    if vp.lower().startswith("rtsp://"):
                        player.setSource(QUrl(vp))
                    else:
                        player.setSource(QUrl.fromLocalFile(os.path.normpath(vp)))
                        
                    player.setVideoOutput(headless_sink)
                    player.play()
                except Exception as e:
                    print(f"[STREAM ERROR] Failed to start video {bed_idx}: {e}")

        self.build_grid()
        self.update_page(0)

        interval = int(self.applied_settings.get("auto_interval_sec", 0))
        if interval > 0:
            self.auto_timer.start(interval * 1000)
        else:
            self.auto_timer.stop()

    def update_page(self, page_index):
        new_page_index = page_index % self.total_pages()
        self.current_page_index = new_page_index
        
        per = self.applied_settings.get("per_page", 4)
        total = self.applied_settings.get("total_videos", 4)
        base = new_page_index * per

        for widget in self.bed_widgets:
            old_b = widget.get("bed_index")
            if old_b is not None and old_b in self.streams:
                try:
                    player = self.streams[old_b]["player"]
                    player.setVideoOutput(self.streams[old_b]["headless_sink"])
                except Exception as e:
                    pass
            widget["bed_index"] = None

        for grid_i, widget in enumerate(self.bed_widgets):
            bed_idx = base + grid_i
            
            if bed_idx < total:
                widget["bed_index"] = bed_idx
                self.sync_grid_ui(grid_i, bed_idx)
                
                try:
                    player = self.streams[bed_idx]["player"]
                    player.setVideoOutput(widget["video_widget"])
                except Exception as e:
                    pass
            else:
                self.clear_grid_ui(grid_i)

    def sync_grid_ui(self, grid_i, bed_idx):
        bw = self.bed_widgets[grid_i]
        
        bed_name = self.applied_settings["bed_names"][bed_idx] if bed_idx < len(self.applied_settings["bed_names"]) else f"Bed {bed_idx+1}"
        self.set_bed_overlay_text(bw["overlay"], f"{bed_idx+1:03d} | {bed_name}")
        
        is_handling = self.handling_states.get(bed_idx, False)
        is_alarming = self.alarm_active.get(bed_idx, False)
        
        if is_handling:
            bw["monitor_result_tag"].setText("Handling by Staff")
            bw["monitor_result_tag"].setStyleSheet("background:#3498db; color:white; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;")
            bw["ack_btn"].setText("✔️ Done")
            bw["ack_btn"].setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 12px;")
            bw["ack_btn"].show()
            bw["frame"].setStyleSheet("background:#dcdcdc; border:4px solid #bdbdbd; padding:6px;")
            bw["monitor_result_tag"].setProperty("_alarm_on", False)
            
        elif is_alarming:
            bw["ack_btn"].setText("✔️ Check")
            bw["ack_btn"].setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 12px;")
            bw["ack_btn"].show()
            
        else:
            bw["ack_btn"].hide()
            bw["frame"].setStyleSheet("background:#dcdcdc; border:4px solid #bdbdbd; padding:6px;")
            bw["monitor_result_tag"].setProperty("_alarm_on", False)
            
            probs = self.last_probs.get(bed_idx)
            if probs:
                event = self._get_event_result(probs)
                bw["monitor_result_tag"].setText(f"Detected: {event}")
                if event == "Normal":
                    bw["monitor_result_tag"].setStyleSheet("background:#90ee90; color:black; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;")
                else:
                    bw["monitor_result_tag"].setStyleSheet("background:#c0392b; color:white; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;")
            else:
                bw["monitor_result_tag"].setText("Detected: Normal")
                bw["monitor_result_tag"].setStyleSheet("background:#90ee90; color:black; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;")

    def clear_grid_ui(self, grid_i):
        bw = self.bed_widgets[grid_i]
        bw["overlay"].setText("")
        bw["monitor_result_tag"].setText("Detected: ...")
        bw["monitor_result_tag"].setStyleSheet("background:#dcdcdc; color:gray; font-size:16px; padding:6px 10px; border-radius:4px;")
        bw["frame"].setStyleSheet("background:#dcdcdc; border:4px solid #bdbdbd; padding:6px;")
        bw["ack_btn"].hide()

    def set_bed_overlay_text(self, label: QLabel, text: str):
        label.setToolTip(text)
        fm = QFontMetrics(label.font())
        avail = label.width()
        if avail <= 20: 
            avail = 260
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, max(30, avail - 20))
        label.setText(elided)

    def reset_event_debounce_state(self, bed_idx, stable_event="Normal"):
        self.event_states[bed_idx] = {
            "stable_event": stable_event,
            "pending_event": None,
            "pending_start": None,
            "pending_count": 0,
        }

    def _debounce_event_result(self, bed_idx, raw_event):

        now = QDateTime.currentDateTime()
        state = self.event_states.get(bed_idx)
        if state is None:
            self.reset_event_debounce_state(bed_idx, "Normal")
            state = self.event_states[bed_idx]

        stable_before = state["stable_event"]

        if raw_event == stable_before:
            state["pending_event"] = None
            state["pending_start"] = None
            state["pending_count"] = 0
            return stable_before, False, stable_before

        if raw_event == state["pending_event"]:
            state["pending_count"] += 1
            start_dt = state["pending_start"]
            elapsed_sec = 0.0
            if start_dt is not None:
                elapsed_sec = start_dt.msecsTo(now) / 1000.0

            if elapsed_sec >= MIN_EVENT_HOLD_SEC and state["pending_count"] >= 2:
                old_stable = stable_before
                state["stable_event"] = raw_event
                state["pending_event"] = None
                state["pending_start"] = None
                state["pending_count"] = 0
                return raw_event, True, old_stable

            return stable_before, False, stable_before

        state["pending_event"] = raw_event
        state["pending_start"] = now
        state["pending_count"] = 1
        return stable_before, False, stable_before

    def play_alarm_sound(self, bed_idx):
        played = False
        stream = self.streams.get(bed_idx)
        if stream:
            try:
                player = stream.get("alarm_player")
                audio = stream.get("alarm_audio")
                if audio is not None and hasattr(audio, "setMuted"):
                    audio.setMuted(False)
                    audio.setVolume(1.0)
                if player is not None:
                    try:
                        player.setLoops(QMediaPlayer.Loops.Infinite)
                    except Exception:
                        pass
                    if player.source().isEmpty():
                        played = False
                    else:
                        player.play()
                        played = True
                        QTimer.singleShot(1000, lambda b=bed_idx: self._fallback_alarm_beep_if_needed(b))
            except Exception as e:
                print(f"[ALARM ERROR] Failed to start alarm player: {e}")

        if not played:
            try:
                QApplication.beep()
            except Exception:
                pass
        return played

    def _fallback_alarm_beep_if_needed(self, bed_idx):
        stream = self.streams.get(bed_idx)
        if not stream:
            return
        try:
            player = stream.get("alarm_player")
            if player is None or player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                QApplication.beep()
        except Exception:
            try:
                QApplication.beep()
            except Exception:
                pass

    #managing state
    def close_alarm_event(self, bed_idx):
        state = self.alarm_states.get(bed_idx)
        if state and state.get("active"):
            end_ts_dt = QDateTime.currentDateTime()
            start_dt = self.alarm_start_times.get(bed_idx, end_ts_dt)
            duration_sec = start_dt.secsTo(end_ts_dt) if start_dt else 0
            
            start_ts = start_dt.toString("yyyy-MM-dd hh:mm:ss")
            end_ts = end_ts_dt.toString("yyyy-MM-dd hh:mm:ss")

            cached = self._preview_cache.get(bed_idx, {})
            trigger_reason = cached.get("reason_label", "Unknown")
            dangerous_pct = cached.get("dangerous_pct", 0.0)
            severity = cached.get("severity", "Info")

            bed_names_list = self.applied_settings.get("bed_names", [])
            bed_name = bed_names_list[bed_idx] if bed_idx < len(bed_names_list) else f"Bed {bed_idx+1}"
            
            try:
                with open(EVENT_LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        start_ts, end_ts, f"{bed_idx+1:03d}", bed_name,
                        duration_sec, trigger_reason, severity
                    ])
            except Exception:
                pass

            self.add_event_card_final(bed_idx, start_ts, end_ts, duration_sec, trigger_reason, dangerous_pct, severity)

            if bed_idx in self.streams:
                try:
                    self.streams[bed_idx]["alarm_player"].stop()
                except Exception:
                    pass

            state["active"] = False
            state["count"] = 0
            self.alarm_active[bed_idx] = False
            if bed_idx in self.alarm_start_times:
                del self.alarm_start_times[bed_idx]
            if bed_idx in self._preview_cache:
                del self._preview_cache[bed_idx]
                
            self.refresh_alert_label()

    def toggle_handling_state(self, grid_i):
        bw = self.bed_widgets[grid_i]
        bed_idx = bw.get("bed_index")
        if bed_idx is None:
            return

        is_handling = self.handling_states.get(bed_idx, False)

        if not is_handling:
            self.handling_states[bed_idx] = True

            if self.alarm_active.get(bed_idx, False):
                self.close_alarm_event(bed_idx)

            if bed_idx in self.buffers:
                self.buffers[bed_idx].clear()
            self.reset_event_debounce_state(bed_idx, "Normal")

            bw["monitor_result_tag"].setText("Handling by Staff")
            bw["monitor_result_tag"].setStyleSheet("background:#3498db; color:white; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;")
            bw["ack_btn"].setText("✔️ Done")
            bw["ack_btn"].setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 12px;")
            bw["ack_btn"].show()
            bw["frame"].setStyleSheet("background:#dcdcdc; border:4px solid #bdbdbd; padding:6px;")
            bw["monitor_result_tag"].setProperty("_alarm_on", False)

        else:
            self.handling_states[bed_idx] = False

            if bed_idx in self.buffers:
                self.buffers[bed_idx].clear()
            self.reset_event_debounce_state(bed_idx, "Normal")

            bw["monitor_result_tag"].setText("Detected: Normal")
            bw["monitor_result_tag"].setStyleSheet("background:#90ee90; color:black; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;")
            bw["ack_btn"].setText("✔️ Check")
            bw["ack_btn"].setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 12px;")
            bw["ack_btn"].hide()

    def on_grid_video_frame(self, grid_i, frame):
        bed_idx = self.bed_widgets[grid_i].get("bed_index")
        if bed_idx is not None:
            self.on_video_frame(bed_idx, frame)

    def on_video_frame(self, bed_idx, video_frame):
        if self.model is None or self.transform is None:
            return
            
        if self.handling_states.get(bed_idx, False):
            return 
        
        self.frame_counters[bed_idx] += 1
        if self.frame_counters[bed_idx] % self.infer_interval != 0:
            return
            
        if not video_frame or not video_frame.isValid():
            return

        try:
            img = video_frame.toImage()
            if img is None or img.isNull():
                return
                
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            w, h = img.width(), img.height()
            if w <= 0 or h <= 0:
                return

            ptr = img.constBits()
            ptr.setsize(w * h * 3)
            arr = np.asarray(ptr, dtype=np.uint8).reshape((h, w, 3))
            pil = Image.fromarray(arr, "RGB")

            if not hasattr(self, "_fixed_resize"):
                self._resize_worker = torchvision.transforms.Resize(256)
                self._crop_worker = torchvision.transforms.CenterCrop(224)
                self._fixed_resize = True
                
            pil = self._resize_worker(pil)
            pil = self._crop_worker(pil)
            
        except Exception:
            return

        buf = self.buffers.get(bed_idx)
        if buf is None:
            return
            
        buf.append(pil)
        
        if len(buf) == self.num_segments:
            if hasattr(self, "inference_worker"):
                self.inference_worker.add_task(bed_idx, list(buf))

    #detection and alarm
    def _get_event_result(self, probs):
        if probs is None:
            return "Normal"

        try:
            if len(probs) != len(LABELS):
                return "Normal"
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
        except Exception:
            return "Normal"
        
        if top_idx == 0:
            return "Normal"

        if top_prob < 0.25:
            return "Normal"

        return LABELS[top_idx]

    def on_inference_result(self, bed_idx, probs):
        if bed_idx not in self.alarm_states:
            return

        self.last_probs[bed_idx] = probs

        if not self.handling_states.get(bed_idx, False):
            self.update_detect_result(bed_idx, probs)

    def update_detect_result(self, bed_idx, probs):
        raw_event = self._get_event_result(probs)
        final_event, changed, previous_event = self._debounce_event_result(bed_idx, raw_event)

        dangerous_pct = 0.0
        try:
            if probs is not None and len(probs) > 1:
                dangerous_pct = float(max(probs[1:]) * 100.0)
        except Exception:
            dangerous_pct = 0.0

        severity = "Info"
        if final_event != "Normal":
            severity = "Alarm" if dangerous_pct >= 25.0 else "Warning"

        reason_label = final_event if final_event != "Normal" else "Unknown"

        if changed and self.alarm_active.get(bed_idx, False):
            self.close_alarm_event(bed_idx)

        grid_i = next((i for i, bw in enumerate(self.bed_widgets) if bw.get("bed_index") == bed_idx), None)

        if grid_i is not None:
            bw = self.bed_widgets[grid_i]
            bw["monitor_result_tag"].setText(f"Detected: {final_event}")

            if not self.alarm_active.get(bed_idx, False):
                if final_event == "Normal":
                    bw["monitor_result_tag"].setStyleSheet(
                        "background:#90ee90; color:black; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;"
                    )
                else:
                    bw["monitor_result_tag"].setStyleSheet(
                        "background:#c0392b; color:white; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;"
                    )

        self.update_alarm_state(bed_idx, final_event, dangerous_pct, severity, reason_label)

    def update_alarm_state(self, bed_idx, final_event, dangerous_pct=0.0, severity="Info", reason_label="Unknown"):
        state = self.alarm_states.get(bed_idx)
        if not state:
            return

        if final_event not in ("Normal", "Unknown"):
            state["count"] += 1
            if state["count"] >= self.alarm_trigger_count and not state["active"]:
                state["active"] = True
                self.alarm_active[bed_idx] = True
                self.alarm_start_times[bed_idx] = QDateTime.currentDateTime()

                ts = self.alarm_start_times[bed_idx].toString("yyyy-MM-dd hh:mm:ss")
                self._preview_cache[bed_idx] = {
                    "start_ts": ts,
                    "severity": severity,
                    "dangerous_pct": dangerous_pct,
                    "reason_label": reason_label,
                }

                self.add_event_card_preview(bed_idx, severity, dangerous_pct, reason_label)

                if bed_idx in self.streams:
                    try:
                        self.play_alarm_sound(bed_idx)
                    except Exception as e:
                        print(f"[ALARM ERROR] Failed to play sound: {e}")

                self.check_and_jump_to_alarm(bed_idx)

            self.refresh_alert_label()

        else:
            if state["active"]:
                self.close_alarm_event(bed_idx)
            state["count"] = 0
            self.refresh_alert_label()

    def check_and_jump_to_alarm(self, alarming_bed_idx):
        current_page_has_alarm = False
        
        for bw in self.bed_widgets:
            b = bw.get("bed_index")
            if b is not None and self.alarm_active.get(b, False):
                current_page_has_alarm = True
                break
        
        if not current_page_has_alarm:
            per = self.applied_settings.get("per_page", 4)
            target_page = alarming_bed_idx // per
            
            if target_page != self.current_page_index:
                print(f"[AUTO-JUMP] Alarm triggered on unseen Bed {alarming_bed_idx+1}. Jumping to page {target_page}.")
                self.update_page(target_page)

    def _blink_alarm(self):
        for bw in self.bed_widgets:
            mtag = bw.get("monitor_result_tag")
            bed_idx = bw.get("bed_index")
            
            if not mtag or bed_idx is None:
                continue

            if self.alarm_active.get(bed_idx, False):
                bw["ack_btn"].setText("✔️ Check")
                bw["ack_btn"].setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 12px;")
                bw["ack_btn"].show()

                if bed_idx in self.streams:
                    player = self.streams[bed_idx]["alarm_player"]
                    if player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                        self.play_alarm_sound(bed_idx)

                if mtag.property("_alarm_on"):
                    mtag.setStyleSheet(
                        "background:#e74c3c; color:white; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;"
                    )
                    mtag.setProperty("_alarm_on", False)
                else:
                    mtag.setStyleSheet(
                        "background:#8b0000; color:white; font-size:16px; font-weight:bold; padding:6px 10px; border-radius:4px;"
                    )
                    mtag.setProperty("_alarm_on", True)
            else:
                mtag.setProperty("_alarm_on", False)

    def go_next_page_auto(self):
        if any(self.alarm_active.values()):
            return 
        self.page_down()

    def page_up(self):
        new_idx = (self.current_page_index - 1) % self.total_pages()
        self.update_page(new_idx)

    def page_down(self):
        new_idx = (self.current_page_index + 1) % self.total_pages()
        self.update_page(new_idx)

    def on_goto(self):
        txt = self.search_input.text().strip()
        if txt.isdigit():
            num = int(txt)
            per = self.applied_settings.get("per_page", 4)
            if 1 <= num <= self.applied_settings.get("total_videos", 4):
                self.update_page((num - 1) // per)
                self.event_history.addItem(f"GoTo: Bed {num}")
            else:
                self.event_history.addItem("GoTo invalid bed id")
        else:
            self.event_history.addItem("GoTo invalid input")

    def open_settings(self):
        self.stacked_widget.setCurrentIndex(2)

    def update_time(self):
        now = QDateTime.currentDateTime().toString("yyyy / MM / dd   hh : mm : ss")
        self.time_label.setText(now)

    #event log
    def add_event_card_preview(self, bed_idx, severity, dangerous_pct, reason_label):
        try:
            bg = "#f8d7da" if severity == "Alarm" else "#fff3cd"
            border = "#f5c6cb" if severity == "Alarm" else "#ffeeba"

            widget = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(8, 6, 8, 6)
            widget.setLayout(layout)
            widget.setStyleSheet(f"background-color:{bg}; border:1px solid {border}; border-radius:6px;")

            header = QLabel(f"{QDateTime.currentDateTime().toString('yyyy / M / d')}")
            header.setStyleSheet("font-weight:700;")
            layout.addWidget(header)
            
            time_label = QLabel(f"{QDateTime.currentDateTime().toString('hh:mm:ss')}  -  ...")
            time_label.setStyleSheet("color:#333;")
            layout.addWidget(time_label)
            
            bed_names_list = self.applied_settings.get("bed_names", [])
            bed_name = bed_names_list[bed_idx] if bed_idx < len(bed_names_list) else "Unknown"
            
            layout.addWidget(QLabel(f"Bed ID : {bed_idx+1:03d}    Bed Name : {bed_name}"))
            
            duration_label = QLabel(f"Duration : 0 seconds")
            layout.addWidget(duration_label)
            
            reason_label_w = QLabel(f"Trigger reason : {reason_label}")
            layout.addWidget(reason_label_w)
            

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(sep)

            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.event_history.addItem(item)
            self.event_history.setItemWidget(item, widget)
            self.event_history.scrollToBottom()

            self._preview_items[bed_idx] = (item, widget)
        except Exception as e:
            pass

    def add_event_card_final(self, bed_idx, start_ts, end_ts, duration_sec, trigger_reason, dangerous_pct, severity):
        try:
            if bed_idx in self._preview_items:
                item, widget = self._preview_items[bed_idx]
                row = self.event_history.row(item)
                if row >= 0:
                    self.event_history.takeItem(row)
                del self._preview_items[bed_idx]

            bg = "#f8d7da" if severity == "Alarm" else "#fff3cd"
            border = "#f5c6cb" if severity == "Alarm" else "#ffeeba"

            widget = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(8, 6, 8, 6)
            widget.setLayout(layout)
            widget.setStyleSheet(f"background-color:{bg}; border:1px solid {border}; border-radius:6px;")

            date_header = QLabel(start_ts.split(" ")[0])
            date_header.setStyleSheet("font-weight:700;")
            layout.addWidget(date_header)
            
            time_row = QLabel(f"{start_ts.split(' ')[1]}  -  {end_ts.split(' ')[1]}")
            layout.addWidget(time_row)
            
            bed_names_list = self.applied_settings.get("bed_names", [])
            bed_name = bed_names_list[bed_idx] if bed_idx < len(bed_names_list) else "Unknown"
            
            bed_row = QLabel(f"Bed ID : {bed_idx+1:03d}    Bed Name : {bed_name}")
            layout.addWidget(bed_row)
            
            duration_row = QLabel(f"Duration : {duration_sec} seconds")
            layout.addWidget(duration_row)
            
            trigger_row = QLabel(f"Trigger reason : {trigger_reason}")
            layout.addWidget(trigger_row)
            

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(sep)

            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.event_history.addItem(item)
            self.event_history.setItemWidget(item, widget)
            self.event_history.scrollToBottom()
        except Exception as e:
            pass

    def open_event_log(self):
        try:
            p = str(EVENT_LOG_CSV)
            if platform.system() == "Windows":
                os.startfile(p)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            QMessageBox.warning(self, "Open Log Failed", f"Cannot open file: {e}")

    def refresh_alert_label(self):
        active_ids = []
        for bed_idx, active in self.alarm_active.items():
            if active:
                active_ids.append(f"{bed_idx+1:03d}")
                
        if len(active_ids) == 0:
            self.alert_label.setText("No Active Alerts")
            self.alert_label.setStyleSheet("color: black; font-size:18px; font-weight:700;")
        else:
            text = "Beds: " + ", ".join(active_ids)
            self.alert_label.setText(text)
            self.alert_label.setStyleSheet("color: #c0392b; font-size:18px; font-weight:700;")


#setting page
class SettingPage(QWidget):
    def __init__(self, stacked_widget, mainpage, initial_config):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.mainpage = mainpage
        self.working = initial_config.copy()
        
        self.init_ui()
        self.refresh_beds()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(60, 40, 60, 40)
        
        title = QLabel("Settings Page")
        title.setStyleSheet("font-size:22px; font-weight:600;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignLeft)

        interval_box = QGroupBox("General Settings")
        interval_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        self.interval_input = QLineEdit(str(self.working.get("auto_interval_sec", 0)))
        self.interval_input.setFixedWidth(80)
        row1.addWidget(QLabel("Auto Page Interval (sec):"))
        row1.addWidget(self.interval_input)
        row1.addStretch()
        interval_layout.addLayout(row1)

        interval_box.setLayout(interval_layout)
        layout.addWidget(interval_box)

        beds_box = QGroupBox("Bed/Video Management")
        beds_layout = QVBoxLayout()
        
        self.beds_scroll = QScrollArea()
        self.beds_scroll.setWidgetResizable(True)
        self.beds_widget = QWidget()
        self.beds_vlayout = QVBoxLayout()
        self.beds_widget.setLayout(self.beds_vlayout)
        self.beds_scroll.setWidget(self.beds_widget)
        beds_layout.addWidget(self.beds_scroll)

        add_bed_btn = QPushButton("Add Bed")
        add_bed_btn.clicked.connect(self.add_bed)
        beds_layout.addWidget(add_bed_btn, alignment=Qt.AlignmentFlag.AlignRight)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_to_default)
        beds_layout.addWidget(reset_btn, alignment=Qt.AlignmentFlag.AlignRight)

        beds_box.setLayout(beds_layout)
        layout.addWidget(beds_box)

        save_btn = QPushButton("Save Settings")
        save_btn.setFixedWidth(160)
        save_btn.clicked.connect(self.on_save)
        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        self.setLayout(layout)

    def reset_to_default(self):
        self.working = default_config()
        self.refresh_beds()
        self.interval_input.setText(str(self.working.get("auto_interval_sec", 0)))
        
        save_config(self.working)
        self.mainpage.apply_settings(self.working)

    def refresh_beds(self):
        for i in reversed(range(self.beds_vlayout.count())):
            item = self.beds_vlayout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)

        bed_names = self.working.get("bed_names", [f"Bed {i+1}" for i in range(4)])
        total_beds = max(4, len(bed_names))
        
        video_paths = self.working.get("video_paths", [])
        if len(video_paths) < total_beds:
            video_paths += [""] * (total_beds - len(video_paths))
        video_paths = video_paths[:total_beds]

        self.working["bed_names"] = bed_names[:total_beds]
        self.working["video_paths"] = video_paths
        self.working["total_videos"] = total_beds

        for idx in range(total_beds):
            row = QHBoxLayout()
            
            id_label = QLabel(f"ID: {idx+1:03d}")
            id_label.setFixedWidth(50)
            row.addWidget(id_label)
            
            name_edit = QLineEdit(self.working["bed_names"][idx])
            name_edit.setFixedWidth(140)
            name_edit.textChanged.connect(lambda text, i=idx: self.update_bed_name(i, text))
            row.addWidget(name_edit)

            path_label = QLabel(self.working["video_paths"][idx])
            path_label.setFixedWidth(220)
            row.addWidget(path_label)

            btn_browse = QPushButton("Browse")
            btn_browse.setFixedWidth(80)
            btn_browse.clicked.connect(lambda _, i=idx: self.browse_video(i))
            row.addWidget(btn_browse)

            btn_rmv_vid = QPushButton("Remove Video")
            btn_rmv_vid.setFixedWidth(100)
            btn_rmv_vid.clicked.connect(lambda _, i=idx: self.remove_video(i))
            row.addWidget(btn_rmv_vid)

            btn_rmv_bed = QPushButton("Remove Bed")
            btn_rmv_bed.setFixedWidth(100)
            btn_rmv_bed.clicked.connect(lambda _, i=idx: self.remove_bed(i))
            row.addWidget(btn_rmv_bed)

            row.addStretch()
            container = QWidget()
            container.setLayout(row)
            self.beds_vlayout.addWidget(container)

    def remove_video(self, idx):
        if 0 <= idx < len(self.working["video_paths"]):
            self.working["video_paths"][idx] = ""
        self.refresh_beds()

    def remove_bed(self, idx):
        if len(self.working["bed_names"]) <= 4:
            QMessageBox.information(self, "Minimum Beds", "At least 4 beds must remain.")
            return
            
        self.working["bed_names"].pop(idx)
        self.working["video_paths"].pop(idx)
        self.working["total_videos"] = len(self.working["bed_names"])
        self.refresh_beds()

    def update_bed_name(self, idx, text):
        if idx < len(self.working["bed_names"]):
            self.working["bed_names"][idx] = text

    def browse_video(self, idx):
        text, ok = QInputDialog.getText(
            self, 
            "Video Source", 
            "Enter video file path or RTSP URL:\n\nExamples:\n• /Users/xxx/video.mp4\n• rtsp://127.0.0.1:8554/cam1"
        )
        if not ok or not text.strip():
            return
        
        src = text.strip()
        if idx < len(self.working["video_paths"]): 
            self.working["video_paths"][idx] = src
        else: 
            self.working["video_paths"].append(src)
            
        self.refresh_beds()

    def add_bed(self):
        self.working["bed_names"].append(f"Bed {len(self.working['bed_names'])+1}")
        self.working["video_paths"].append("")
        self.working["total_videos"] = len(self.working["bed_names"])
        self.refresh_beds()

    def on_save(self):
        try:
            self.working["auto_interval_sec"] = int(self.interval_input.text())
        except Exception:
            QMessageBox.warning(self, "Invalid", "Values must be integers.")
            return
            
        self.working["per_page"] = 4
        
        if "bed_names" not in self.working:
            self.working["bed_names"] = [f"Bed {i+1}" for i in range(self.working.get("total_videos", 4))]
            
        self.mainpage.apply_settings(self.working)
        save_config(self.working)
        
        saved_msg = QMessageBox(self)
        saved_msg.setWindowTitle("Saved")
        saved_msg.setText("All settings saved. Returning to main page...")
        saved_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        saved_msg.show()
        
        def close_message():
            if saved_msg.isVisible():
                saved_msg.close()
                
        QTimer.singleShot(2000, close_message)
        self.stacked_widget.setCurrentIndex(1)


#main window
class MainWindow(QStackedWidget):
    def __init__(self, cfg):
        super().__init__()
        self.current_user = None
        self.login_page = LoginPage(self)
        self.main_page = MainPage(self, cfg)
        self.setting_page = SettingPage(self, self.main_page, cfg)
        
        self.addWidget(self.login_page)
        self.addWidget(self.main_page)
        self.addWidget(self.setting_page) 

        self.setCurrentIndex(0)
        self.setWindowTitle("CareVision Prototype")
        self.resize(1400, 900)

    def closeEvent(self, event):
        try:
            if hasattr(self.main_page, "inference_worker"):
                self.main_page.inference_worker.stop()
                self.main_page.inference_worker.wait(1000)
        except Exception as e:
            print(f"[CLOSE] Error stopping worker: {e}")
            
        try:
            self.main_page.auto_timer.stop()
            self.main_page.alarm_timer.stop()
            self.main_page.time_timer.stop()
        except Exception:
            pass
            
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(load_config())
    window.showMaximized()
    sys.exit(app.exec())