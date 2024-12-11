import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import time
import os
import json
import glob
import random
import matplotlib.pyplot as plt
from tkinter import ttk, filedialog, Label, Entry, Button, Listbox, Frame, StringVar,messagebox
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from training_gui import TrainingGUI
from anh_gui import CutImageGUI
tf.keras.utils.disable_interactive_logging()



# Xác định lớp ứng dụng chính
class FaceMaskDetectionApp:
    def __init__(self, root):
        self.root = root
        self.master = root
        root.geometry('950x750')
        root.resizable(True, True)
        self.root.title("Nhận diện cảm xúc")

        # Tạo sổ ghi chép với các tab khác nhau
        self.main_notebook = ttk.Notebook(root)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        # Xác định khung tab
        # Tạo tab tổng "Nhận diện"
        self.tab_detection = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.tab_detection, text="Nhận diện")

        self.tab_classroom = ttk.Frame(self.main_notebook)
        self.tab_train = ttk.Frame(self.main_notebook)
        self.tab_preprocessing = ttk.Frame(self.main_notebook)
        self.tab_evaluate_results = ttk.Frame(self.main_notebook)
        self.tab_setting = ttk.Frame(self.main_notebook)

        self.main_notebook.add(self.tab_classroom, text="Danh sách phòng học")
        training_frame = TrainingGUI(self.main_notebook)  # Sử dụng giao diện từ training_gui.py
        self.main_notebook.add(training_frame, text="Huấn luyện mô hình")
        preprocessing_frame = CutImageGUI(self.main_notebook)  # Sử dụng giao diện từ training_gui.py
        self.main_notebook.add(preprocessing_frame, text="Tiền xử lý dữ liệu")
        self.main_notebook.add(self.tab_setting, text="Cài đặt")

        # Tạo PanedWindow cho tab con
        self.paned_window = ttk.PanedWindow(self.tab_detection, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Khung để chứa các nút tab dọc
        self.tab_frame = ttk.Frame(self.paned_window, width=100)  # Thiết lập chiều rộng cố định
        self.tab_frame.pack_propagate(False)  # Không cho phép tự động điều chỉnh kích thước
        self.paned_window.add(self.tab_frame)

        # Khung chính để hiển thị nội dung
        self.content_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.content_frame)

        # Tạo các tab con
        self.tab_webcam = ttk.Frame(self.content_frame)
        self.tab_image = ttk.Frame(self.content_frame)
        self.tab_video = ttk.Frame(self.content_frame)
        self.tab_camera = ttk.Frame(self.content_frame)
        self.tab_evaluate_results = ttk.Frame(self.content_frame)

        # Danh sách các tab
        self.tab_list = {
            "Webcam": self.tab_webcam,
            "Hình ảnh": self.tab_image,
            "Video": self.tab_video,
            "Camera trực tiếp": self.tab_camera,
            "Kết quả đánh giá": self.tab_evaluate_results
        }

        # Tạo các nút để chọn tab con
        for tab_name, tab_frame in self.tab_list.items():
            button = ttk.Button(self.tab_frame, text=tab_name, command=lambda f=tab_frame: self.show_frame(f))
            button.pack(fill=tk.X)

        # Hiển thị tab mặc định
        self.show_frame(self.tab_webcam)

        self.EMOTION = ["Binh thuong", "Buon", "Cuoi", "Ngac nhien", "So hai", "Tuc gian"]
        self.EMOTION_COLORS = {
            "Binh thuong": (0, 255, 0), # Màu xanh lá
            "Buon": (0, 0, 255), # Màu đỏ
            "Cuoi": (0, 255, 255), # Màu vàng
            "Ngac nhien": (0,140, 255),  # Màu cam
            "So hai": (0, 0, 0),  # Màu đen
            "Tuc gian": (255, 0, 255)  # Màu hồng tím
        }


        self.webcam_source = 0
        self.wc = cv2.VideoCapture(self.webcam_source)
        self.is_webcam_on = False
        self.save_directory = "./captured_images/"
        self.save_face_directory = "./captured_faces/"

        self.is_auto_capture_on = False
        self.capture_timer = None

        # Khởi tạo biến để lưu thời điểm lưu ảnh cuối cùng
        self.last_capture_time = time.time()
        self.last_capture_time_emotions = time.time()
        self.last_update_time = time.time()
        self.capture_interval = 10  # khoảng thời gian giữa các lần lưu ảnh (tính bằng giây)
        # Cài đặt khoảng thời gian (phút) muốn nhận diện
        self.capture_interval_minutes = 10  # Ví dụ: mỗi 10 phút
        # Chuyển đổi từ phút sang giây
        self.capture_interval_seconds = 10
        self.capture_total_minutes_seconds = (self.capture_interval_minutes*60) + self.capture_interval_seconds

        self.config_file = os.path.abspath("./settings/settings.json") # Đường dẫn đến tệp cấu hình
        self.default_settings = {
            "capture_interval_minutes": 10,
            "capture_interval_seconds": 10,
            "checkbox_var_1": True,
            "checkbox_var_2": True,
            "checkbox_var_3": True
        }
        self.checkbox_var_1 = tk.BooleanVar()
        self.checkbox_var_2 = tk.BooleanVar()
        self.checkbox_var_3 = tk.BooleanVar()
        self.entry_var_minutes = tk.StringVar()
        self.entry_var_seconds = tk.StringVar()

        # Khởi tạo các BooleanVar với giá trị mặc định
        self.checkbox_var_1 = tk.BooleanVar(value=self.default_settings["checkbox_var_1"])
        self.checkbox_var_2 = tk.BooleanVar(value=self.default_settings["checkbox_var_2"])
        self.checkbox_var_3 = tk.BooleanVar(value=self.default_settings["checkbox_var_3"])


        self.saved_settings = {}
        # Khởi tạo cài đặt
        self.load_settings()


        # Tạo Canvas để hiển thị webcam
        self.live_webcam = tk.Canvas(self.tab_webcam, width=540, height=480)
        self.live_webcam.grid(row=0, column=1, padx=10, pady=10)
        # Nếu không có kết nối nào được thiết lập ban đầu, hiển thị màn hình trắng
        self.live_webcam.create_rectangle(0, 0, 545, 485, fill="white")

        self.update_webcam()
        
        self.model_path = "train/18-08-2024.h5"
        self.model = load_model(self.model_path)
        # self.face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        # Khởi tạo detector
        self.detector = MTCNN()
        # self.detector = MTCNN(scale_factor=0.6)

        self.last_frame = None

        # Các biến để phát trực tuyến webcam
        self.url = None
        self.wc = None

        # Ảnh cho nút cộng và trừ
        plus_icon = tk.PhotoImage(file='./anh/plus_icon.png')
        minus_icon = tk.PhotoImage(file='./anh/minus_icon.png')
        # Ảnh biểu tượng cho các nút chỉ
        left_icon = tk.PhotoImage(file='./anh/left_icon.png')
        right_icon = tk.PhotoImage(file='./anh/right_icon.png')
        up_icon = tk.PhotoImage(file='./anh/up_icon.png')
        down_icon = tk.PhotoImage(file='./anh/down_icon.png')
    
        # Thêm Sizegrip vào cửa sổ root
        sizegrip = ttk.Sizegrip(root)
        sizegrip.pack(side="bottom", anchor=tk.SE)
        self.main_notebook.pack()

        # Tab webcam
        # Canvas để hiển thị webcam 
        self.live_webcam = tk.Canvas(self.tab_webcam, width=540, height=480)
        self.live_webcam.grid(row=0, column=1, padx=10, pady=10)
        
        # Nếu không có kết nối nào được thiết lập ban đầu, hiển thị màn hình trắng
        self.live_webcam.create_rectangle(0, 0, 550, 490, fill="white")
        
        # Tạo Frame để chứa nút cộng và trừ tab_camera
        buttons_tab_webcam_frame_1 = ttk.Frame(self.tab_webcam)
        buttons_tab_webcam_frame_1.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        # Nút bật wecam chứa trong buttons_tab_webcam_frame_1
        self.toggle_button_webcam = ttk.Button(buttons_tab_webcam_frame_1, text="Bật Webcam", command=self.start_toggle_webcam_thread)
        self.toggle_button_webcam.pack(side=tk.LEFT)

        self.capture_toggle_button_webcam = ttk.Button(buttons_tab_webcam_frame_1, text="Bật chụp webcam", command=self.toggle_auto_capture_webcam)
        self.capture_toggle_button_webcam.pack(side=tk.LEFT)

        # Tab ảnh
        # Tạo khung 1 tab 2
        buttons_tab_images_frame_1 = ttk.Frame(self.tab_image)
        buttons_tab_images_frame_1.pack()
        # Tạo khung 2 tab 2
        buttons_tab_images_frame_2 = ttk.Frame(self.tab_image)
        buttons_tab_images_frame_2.pack()

        # Tab nhận diện hình ảnh
        self.num_images_label = Label(buttons_tab_images_frame_1, text="Nhập số lượng ảnh cần kiểm tra:")
        self.num_images_label.pack()

        self.num_images_entry = Entry(buttons_tab_images_frame_1)
        self.num_images_entry.pack()
        # self.images_file_path = None
        self.manual_selection_mode = False
        
        # Nút kiểm tra ảnh
        self.check_button = Button(buttons_tab_images_frame_1, text="Kiểm tra ảnh", command=self.check_images)
        self.check_button.pack()

        # Nút kiểmt chọn ảnh
        self.toggle_button_image = Button(buttons_tab_images_frame_1, text="Chọn ảnh thủ công", command=self.toggle_image_selection_mode)
        self.toggle_button_image.pack()

        # Tab video
        # Tạo Frame để chứa nút cộng và trừ tab_video
        self.video_file_path = None
        self.video_capture = None
        self.current_frame = 0
        self.is_running = False
        self.last_frame = None
        self.detecting_indirect_video = False
        self.capture_in_progress_video = False
        self.is_auto_capture_on_indirect = False
        self.is_video_closed = False
        self.frame_count = 0
        self.detect_frames = 1  # Số khung hình sẽ nhận diện sau mỗi 10 khung hình
        self.detecting = False  # Trạng thái nhận diện
        self.frames_per_second = 1  # Số frame được lưu mỗi giây
        self.clear_saved_data = True
        self.should_display_frames = True
        self.cached_frame_folder = "captured_cached_frame"
        # self.saved_frames_folder = "captured_images"  # Thư mục lưu các frame
        self.saved_faces_folder = "captured_faces"
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Tạo khung 1 tab video
        buttons_tab_video_frame_1 = ttk.Frame(self.tab_video)
        buttons_tab_video_frame_1.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Nút bật wecam chứa trong buttons_tab_webcam_frame_1
        self.browse_button_video = ttk.Button(buttons_tab_video_frame_1, text="Chọn video", command=self.browse_video)
        self.browse_button_video.pack(side=tk.LEFT)
        self.video_path_label = ttk.Label(self.tab_video, text="Chưa chọn video")
        self.video_path_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.detect_video_button = ttk.Button(buttons_tab_video_frame_1, text="Bật kiểm tra video", command=self.start_toggle_video_thread)
        self.detect_video_button.pack(side=tk.LEFT)

        self.detect_video_photos_button = ttk.Button(buttons_tab_video_frame_1, text="Bật kiểm tra video gián tiếp", command=self.start_toggle_indirect_video_thread)
        self.detect_video_photos_button.pack(side=tk.LEFT)

        self.capture_toggle_button_video = ttk.Button(buttons_tab_video_frame_1, text="Bật chụp video", command=self.toggle_capture_video)
        self.capture_toggle_button_video.pack(side=tk.LEFT)

        # Thêm Canvas cho tab video
        self.video_result_canvas = tk.Canvas(self.tab_video, width=640, height=450)
        self.video_result_canvas.create_rectangle(0, 0, 745, 685, fill="white")
        self.video_result_canvas.grid(column=1)

        # Menu phòng
        self.selected_room_var = tk.StringVar()
        self.selected_room_var.set("Chọn phòng")

        # Tạo menu phòng
        self.room_menu = ttk.Combobox(self.tab_camera, textvariable=self.selected_room_var, font=("cambria", 10), width=20, state="readonly")
        self.room_menu.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        
        # Khởi tạo biến camera_connected và đặt giá trị ban đầu là False
        self.camera_connected = False
        self.load_room_menu("./rooms/rooms.txt")

        # Canvas để hiển thị camera trực tiếp
        self.live_camera_canvas = tk.Canvas(self.tab_camera, width=640, height=580)
        self.live_camera_canvas.grid(row=0,column=1, padx=10, pady=10)
        # Nếu không có kết nối nào được thiết lập ban đầu, hiển thị màn hình trắng
        self.live_camera_canvas.create_rectangle(0, 0, 745, 585, fill="white")

        # Tạo Frame để chứa nút cộng và trừ tab_camera
        buttons_tab_camera_frame_1 = ttk.Frame(self.tab_camera)
        buttons_tab_camera_frame_1.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Tạo Frame để chứa cập nhật và kết nối tab_camera
        buttons_tab_camera_frame_2 = ttk.Frame(self.tab_camera)
        buttons_tab_camera_frame_2.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Nút kết nối ip Camera
        self.toggle_button_camera = ttk.Button(buttons_tab_camera_frame_2, text="Kết nối camera", command=self.start_toggle_camera_connection_thread)
        self.toggle_button_camera.pack(side=tk.LEFT)

        # Nút chụp ảnh tab camera ip 
        self.capture_toggle_button_camera = ttk.Button(buttons_tab_camera_frame_2, text="Bật chụp camera", command=self.toggle_capture_camera)
        self.capture_toggle_button_camera.pack(side=tk.LEFT)
        # Nút cập nhật danh sách phòng tab camera
        self.updated_camera_room = ttk.Button(buttons_tab_camera_frame_2, text="Cập nhật danh sách phòng", command=self.update_camera_list)
        self.updated_camera_room.pack(side=tk.LEFT)

        self.zoom_scale = 1  # Không có zoom khi bắt đầu
        self.zoom_scale = max(self.zoom_scale, 0.01)  # Ngăn không cho zoom_scale giảm quá nhỏ

        # Nút "Cộng" với ảnh, trong Frame tab_camera
        self.camera_zoom_in = ttk.Button(buttons_tab_camera_frame_1, image=plus_icon)
        self.camera_zoom_in.image = plus_icon  # Giữ tham chiếu đến ảnh
        self.camera_zoom_in.pack(side=tk.LEFT)

        # Nút "Trừ" với ảnh, trong cùng Frame tab_camera
        self.camera_zoom_out = ttk.Button(buttons_tab_camera_frame_1, image=minus_icon)
        self.camera_zoom_out.image = minus_icon  # Giữ tham chiếu đến ảnh
        self.camera_zoom_out.pack(side=tk.LEFT)

        # Nút chỉ tab_camera
        self.button_left = ttk.Button(buttons_tab_camera_frame_1, image=left_icon)
        self.button_left.image = left_icon  
        self.button_left.pack(side=tk.LEFT)

        self.button_right = ttk.Button(buttons_tab_camera_frame_1, image=right_icon)
        self.button_right.image = right_icon  
        self.button_right.pack(side=tk.LEFT)

        self.button_up = ttk.Button(buttons_tab_camera_frame_1, image=up_icon)
        self.button_up.image = up_icon
        self.button_up.pack(side=tk.LEFT)

        self.button_down = ttk.Button(buttons_tab_camera_frame_1, image=down_icon)
        self.button_down.image = down_icon  # Giữ tham chiếu đến ảnh
        self.button_down.pack(side=tk.LEFT)

        self.is_camera_on = False
        self.capture_in_progress_camera = False
        self.is_auto_capture_on_camera = False
        self.selected_room_ip = None
        
        self.selected_images = []
        self.so_anh_dung = 0
        self.so_anh_co_khuon_mat = 0
        self.so_anh_sai = 0

        self.default_evaluation_results = "./evaluation_results"
        # Kiểm tra và tạo thư mục mặc định nếu không tồn tại
        if not os.path.exists(self.default_evaluation_results):
            os.makedirs(self.default_evaluation_results)

        # Tab 2
        # Khởi tạo StringVar và gán vào self.id_room và self.name_room
        self.id_room = StringVar()
        self.name_room = StringVar()
        self.ip_room = StringVar()
        self.original_ip = ""

        # Thiết lập tab lớp học
        self.tab_classroom_label = Label(self.tab_classroom, text="Danh sách phòng học", fg="red", font=("cambria", 16), width=66)
        self.tab_classroom_label.grid(row=0, pady=10, columnspan=2)
    
        # Tạo một Treeview để hiển thị danh sách phòng học dưới dạng bảng
        self.classroom_treeview = ttk.Treeview(self.tab_classroom, selectmode="browse", columns=("id", "room_name", "ip_room"), show="headings", height=20)
        self.classroom_treeview.grid(row=1, padx=10, pady=10, columnspan=2)

        # Thiết lập các tiêu đề cột
        self.classroom_treeview.heading("id", text="ID")
        self.classroom_treeview.heading("room_name", text="Tên phòng")
        self.classroom_treeview.heading("ip_room", text="Mã IP Camera")

        # Thiết lập độ rộng cho các cột
        self.classroom_treeview.column("id", width=30)
        self.classroom_treeview.column("room_name", width=270)
        self.classroom_treeview.column("ip_room", width=270)


        # Tạo một ô nhập "Tên phòng", "Mã IP camera"
        # Tạo một Frame chứ khung 2 tab lớp học
        buttons_tab_classroom_frame_1 = ttk.Frame(self.tab_classroom)
        buttons_tab_classroom_frame_1.grid(row=2, column=1, pady=5, sticky="n")
        self.tab_classroom_label_2 = Label(buttons_tab_classroom_frame_1, text="Nhập tên phòng:")
        self.tab_classroom_label_2.pack(side=tk.LEFT)
        entry_name = Entry(buttons_tab_classroom_frame_1, width=40)
        entry_name.pack(side=tk.LEFT)

        # Tạo một Frame chứ khung 3 tab lớp học
        buttons_tab_classroom_frame_2 = ttk.Frame(self.tab_classroom)
        buttons_tab_classroom_frame_2.grid(row=3, column=1, pady=5, sticky="n")
        self.tab_classroom_label_2 = Label(buttons_tab_classroom_frame_2, text="Nhập IP camera: ")
        self.tab_classroom_label_2.pack(side=tk.LEFT)
        entry_id = Entry(buttons_tab_classroom_frame_2, width=40)
        entry_id.pack(side=tk.LEFT)

        # Tạo một Frame để đặt nút thêm , sửa, xóa
        buttons_tab_classroom_frame_3 = Frame(self.tab_classroom)
        buttons_tab_classroom_frame_3.grid(row=4, column=1, columnspan=3)

        # Nút Thêm
        self.add_button = Button(buttons_tab_classroom_frame_3, text="Thêm", command=lambda: self.add_room(entry_name.get(), entry_id.get()))
        self.add_button.pack(side=tk.LEFT)

        # Nút Sửa
        self.edit_window = None
        self.edit_button = Button(buttons_tab_classroom_frame_3, text="Sửa", command=self.edit_room)
        self.edit_button.pack(side=tk.LEFT)

        # Nút Xóa
        self.delete_button = Button(buttons_tab_classroom_frame_3, text="Xóa", command=self.delete_selected_room)
        self.delete_button.pack(side=tk.LEFT)

        # Nút Xóa
        self.delete_button = Button(buttons_tab_classroom_frame_3, text="Cập nhật danh sách", command=self.update_list)
        self.delete_button.pack(side=tk.LEFT)

        self.show_room()

        # Tab 4
        # Thiết lập tab kết quả
        self.emotion_counter = {emotion: 0 for emotion in self.EMOTION}
        self.tab_evaluate_results_label = tk.Label(self.tab_evaluate_results, text="Kết quả nhận diện", fg="red", font=("Cambria", 16), width=66)
        self.tab_evaluate_results_label.grid(row=0, pady=10, column=0, columnspan=3)

        self.tab_evaluate_results_treeview = ttk.Treeview(self.tab_evaluate_results, selectmode="browse", columns=("emotions","number_emotions","percentage",), show="headings", height=20)
        self.tab_evaluate_results_treeview.grid(row=1, padx=10, pady=10, column=0, columnspan=3)

        # Thiết lập các tiêu đề cột kết quả
        self.tab_evaluate_results_treeview.heading("emotions", text="Trạng thái cảm xúc")
        self.tab_evaluate_results_treeview.heading("number_emotions", text="Số cảm xúc")
        self.tab_evaluate_results_treeview.heading("percentage", text="Đánh giá tỉ lệ")
    
        # Chèn dữ liệu vào Treeview
        self.tab_evaluate_results_treeview.delete(*self.tab_evaluate_results_treeview.get_children())

        # Tạo khung chứa các nút bên dưới bảng
        buttons_tab_evaluate_results_1 = tk.Frame(self.tab_evaluate_results)
        buttons_tab_evaluate_results_1.grid(row=2, column=0, columnspan=3, pady=10)

        # Nút bắt đầu huấn luyện
        self.Save_evaluation_results_button = tk.Button(buttons_tab_evaluate_results_1, text="Lưu kết quả đánh giá",command=self.save_evaluation_results_data)
        self.Save_evaluation_results_button.pack(side=tk.LEFT, padx=5)

        # Nút mở file khác
        self.open_file_evaluation_results_button = tk.Button(buttons_tab_evaluate_results_1, text="Mở file lịch sử đánh giá",command=self.open_evaluation_results_data)
        self.open_file_evaluation_results_button.pack(side=tk.LEFT, padx=5)

        self.delete_file_evaluation_results_button = tk.Button(buttons_tab_evaluate_results_1, text="Xóa dữ liệu lịch sử đánh giá",command=self.delete_evaluation_results_data)
        self.delete_file_evaluation_results_button.pack(side=tk.LEFT, padx=5)

        # Tab 5
        # Thêm checkbox vào tab "Cài đặt"
        buttons_tab_setting_frame_1 = ttk.Frame(self.tab_setting)
        buttons_tab_setting_frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Tạo các Checkbutton và gán chúng với BooleanVar
        self.checkbox_1 = tk.Checkbutton(self.tab_setting, text="Lưu ảnh phân loại cảm xúc", variable=self.checkbox_var_1, command=self.checkbox_var_1)
        self.checkbox_1.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.checkbox_2 = tk.Checkbutton(self.tab_setting, text="Lưu ảnh toàn bộ khung hình", variable=self.checkbox_var_2, command=self.checkbox_var_2)
        self.checkbox_2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.checkbox_3 = tk.Checkbutton(self.tab_setting, text="Lưu khuôn mặt chưa phân loại", variable=self.checkbox_var_3, command=self.checkbox_var_1)
        self.checkbox_3.grid(row=2, column=0, padx=10, pady=10, sticky="w")
    
        buttons_tab_setting_frame_2 = ttk.Frame(self.tab_setting)
        buttons_tab_setting_frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Thêm nhãn và Entry cho phút
        self.entry_label_minutes = ttk.Label(buttons_tab_setting_frame_2, text="Phút:")
        self.entry_label_minutes.pack(side=tk.LEFT)
        self.entry_var_minutes = tk.StringVar(value=str(self.capture_interval_minutes))
        self.num_time_entry_minutes = tk.Entry(buttons_tab_setting_frame_2, textvariable=self.entry_var_minutes, width=5)
        self.num_time_entry_minutes.pack(side=tk.LEFT)

        # Thêm nhãn và Entry cho giây
        self.entry_label_seconds = ttk.Label(buttons_tab_setting_frame_2, text="Giây:")
        self.entry_label_seconds.pack(side=tk.LEFT)
        self.entry_var_seconds = tk.StringVar(value=str(self.capture_interval_seconds))
        self.num_time_entry_seconds = tk.Entry(buttons_tab_setting_frame_2, textvariable=self.entry_var_seconds, width=5)
        self.num_time_entry_seconds.pack(side=tk.LEFT)

        # Nút Áp dụng
        self.apply_button = ttk.Button(buttons_tab_setting_frame_2, text="Áp dụng", command=self.apply_settings)
        self.apply_button.pack(side=tk.LEFT)

        # Nút Cài lại mặc định
        self.reset_button = ttk.Button(buttons_tab_setting_frame_2, text="Cài lại mặc định", command=self.reset_to_default)
        self.reset_button.pack(side=tk.LEFT)
        


    def start_toggle_webcam_thread(self):
        threading.Thread(target=self.toggle_webcam, daemon=True).start()
        
    def toggle_webcam(self):
        self.is_webcam_on = not self.is_webcam_on
        if self.is_webcam_on:
            self.toggle_button_webcam.config(text="Tắt Webcam")
            self.capture_toggle_button_webcam.config(state=tk.NORMAL)
            self.wc = cv2.VideoCapture(self.webcam_source)
            if not self.wc.isOpened():
                print("Không mở được webcam.")
            if self.is_auto_capture_on:
                self.capture_images_auto()
        else:
            self.toggle_button_webcam.config(text="Bật Webcam")
            self.capture_toggle_button_webcam.config(state=tk.DISABLED)
            if self.capture_timer:
                self.tab_webcam.after_cancel(self.capture_timer)
            self.wc.release()
            self.wc = None  # Đảm bảo self.wc được đặt thành Không
            if self.last_frame is not None:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)))
                self.live_webcam.create_image(0, 0, image=self.photo, anchor=tk.NW)          
                
    def update_webcam(self):
        if self.is_webcam_on and self.wc is not None:
            ret, frame = self.wc.read()
            if ret:
                frame, detected_emotions_1 = self.detect_face_emotions(frame)  # Nhận hai giá trị
                frame = cv2.resize(frame, (540, 480))
                self.last_frame = frame
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.live_webcam.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # Cập nhật bộ đếm cảm xúc nếu đủ thời gian
                if time.time() - self.last_update_time >= self.capture_total_minutes_seconds:
                    for predicted_label, count in detected_emotions_1.items():
                        if count > 0:
                            self.update_emotion_counter(predicted_label, count)
                    self.last_update_time = time.time()  # Cập nhật thời gian sau khi cập nhật bảng

        self.tab_webcam.after(10, self.update_webcam)  # Đặt lại hàm để gọi sau 10ms

    # Logic chụp ảnh webcam
    def toggle_auto_capture_webcam(self):
        if self.is_webcam_on:
            self.is_auto_capture_on = not self.is_auto_capture_on
            if self.is_auto_capture_on:
                self.capture_toggle_button_webcam.config(text="Tắt chụp webcam")
                self.capture_images_auto()
            else:
                self.capture_toggle_button_webcam.config(text="Bật chụp webcam")
                if self.capture_timer:
                    self.tab_webcam.after_cancel(self.capture_timer)
        else:
            messagebox.showwarning("Lỗi", "Vui lòng bật webcam trước khi bật chụp.")

    def detect_faces(self, frame):
        # Đảm bảo bạn đã khởi tạo trình dò ​​tìm khuôn mặt
        faces = self.detector.detect_faces(frame)
        bounding_boxes = [face['box'] for face in faces]
        return bounding_boxes

    def detect_face_emotions(self, frame):
        image = frame
        if image is None:
            print("Lỗi đọc hình ảnh.")
            return None, {}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bounding_boxes = self.detect_faces(rgb_image)
        current_time = time.time()

        detected_emotions_1 = {emotion: 0 for emotion in self.EMOTION}
        save_images = False

        if self.is_auto_capture_on and self.checkbox_var_1.get():
            if current_time - self.last_capture_time_emotions >= self.capture_total_minutes_seconds:
                save_images = True

        if self.is_auto_capture_on_indirect and self.checkbox_var_1.get():
            save_images = True

        if self.is_auto_capture_on_camera and self.checkbox_var_1.get():
            if current_time - self.last_capture_time_emotions >= self.capture_total_minutes_seconds:
                save_images = True

        for idx, (x, y, w, h) in enumerate(bounding_boxes):
            face_roi = rgb_image[y:y+h, x:x+w]
            if not face_roi.size:
                continue

            resized_face = cv2.resize(face_roi, (48, 48))
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2GRAY)
            input_image = np.expand_dims(gray_face, axis=0)
            input_image = input_image.reshape((1, 48, 48, 1))
            # predictions = self.model.predict(input_image)
            predictions = self.model.predict(input_image, verbose=0)
            predicted_label = self.EMOTION[np.argmax(predictions)]
            color = self.EMOTION_COLORS.get(predicted_label, (255, 255, 255))

            # Cập nhật số liệu cảm xúc
            detected_emotions_1[predicted_label] += 1

            # Vẽ nhãn cảm xúc và khung mặt
            font_scale = 1.0
            font_thickness = 2
            label_size = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_TRIPLEX, font_scale, font_thickness)[0]
            while label_size[0] > w:
                font_scale -= 0.1
                if font_scale < 0.1:
                    font_scale = 0.1
                    break
                label_size = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_TRIPLEX, font_scale, font_thickness)[0]

            label_x = x + (w - label_size[0]) // 2
            label_y = y - 10 if y - 10 > 10 else y + label_size[1] + 10
            cv2.rectangle(image, (label_x, y - label_size[1] - 10), (label_x + label_size[0], y), color, cv2.FILLED)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{predicted_label}", (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (255, 255, 255), font_thickness)

            # Logic lưu ảnh
            if save_images:
                emotion_folder = os.path.join("captured_faces_emotions", predicted_label)
                if not os.path.exists(emotion_folder):
                    os.makedirs(emotion_folder)
                resized_face_image = cv2.resize(face_roi, (250, 250))
                unique_time = int(current_time * 1000) + idx
                file_path = os.path.join(emotion_folder, f"{predicted_label}_{unique_time}_{idx}.jpg")
                cv2.imwrite(file_path, cv2.cvtColor(resized_face_image, cv2.COLOR_RGB2BGR))
                # print(f"Ảnh đã được lưu vào: {file_path}")  

        if save_images:
            self.last_capture_time_emotions = current_time

        # Trả về hình ảnh đã xử lý và dữ liệu cảm xúc cho hàm gọi
        return image, detected_emotions_1

    def update_emotion_counter(self, predicted_label, count):
        # Tăng bộ đếm cho nhãn cảm xúc nhận diện được
        if predicted_label in self.emotion_counter:
            self.emotion_counter[predicted_label] += count
        else:
            print(f"Cảm xúc '{predicted_label}' không có trong danh sách cảm xúc.")

        # Cập nhật bảng kết quả
        self.insert_evaluation_results_data()

    def capture_images_auto(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_capture_time
        
        if elapsed_time >= self.capture_total_minutes_seconds and self.is_auto_capture_on:
            ret, frame = self.wc.read()
            if ret:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}.png"
                save_path = os.path.join(self.save_directory, filename)
                faces = self.detect_faces(frame)
                for i, (x, y, w, h) in enumerate(faces):
                    face = frame[y:y+h, x:x+w]  # Cắt ảnh khuôn mặt

                    # Thay đổi kích thước hình ảnh khuôn mặt thành 250x250
                    resized_face = cv2.resize(face, (250, 250))

                    # Tạo đường dẫn và lưu ảnh khuôn mặt
                    face_path = os.path.join(self.save_face_directory, f"{timestamp}_face_{i}.png")
                    if self.checkbox_var_3.get():
                        cv2.imwrite(face_path, resized_face)

                # Lưu ảnh gốc
                # Kiểm tra trạng thái của checkbox trước khi lưu ảnh gốc
                if self.checkbox_var_2.get():
                    cv2.imwrite(save_path, frame)

                # Cập nhật thời gian chụp cuối cùng
                self.last_capture_time = current_time
        self.capture_timer = self.tab_webcam.after(100, self.capture_images_auto) # Kiểm tra mỗi 100 ms

    # logic tab nhận diện ảnh
    def toggle_image_selection_mode(self):
        if hasattr(self, 'manual_selection_mode') and self.manual_selection_mode:
            # Nếu đang ở chế độ tự chọn ảnh, chuyển sang chế độ kiểm tra ảnh ngẫu nhiên
            self.browse_images()
        else:
            # Nếu không, chuyển sang chế độ tự chọn ảnh
            self.manual_selection_mode = True
            self.check_button.config(text="Kiểm tra ảnh")

    # Phương thức để chọn ảnh từ hộp thoại
    def browse_images(self):
        # Mở hộp thoại chọn tệp để chọn ảnh
        file_paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png; *.jpg; *.jpeg")])
        if file_paths:
            self.selected_images = list(file_paths)
            self.check_images()

    def check_images(self):
        if hasattr(self, 'manual_selection_mode') and self.manual_selection_mode:
            # Nếu đang ở chế độ tự chọn ảnh, hiển thị các ảnh đã chọn
            self.show_selected_images()
            self.manual_selection_mode = False  # Chuyển về chế độ kiểm tra ảnh ngẫu nhiên
            self.check_button.config(text="Kiểm tra ảnh")
        else:
            # Kiểm tra số lượng ảnh người dùng muốn kiểm tra
            input_text = self.num_images_entry.get()

            # Kiểm tra nếu người dùng chưa nhập số lượng ảnh hoặc nhập số lượng dưới 1
            if not input_text or int(input_text) < 1:
                messagebox.showwarning("Lỗi", "Vui lòng nhập số lượng ảnh hợp lệ (lớn hơn hoặc bằng 1).")
                return

            # Tiếp tục thực hiện kiểm tra ảnh
            self.select_random_images()

    # Phương thức để chọn ngẫu nhiên ảnh từ thư mục
    def select_random_images(self):
        # Kiểm tra số lượng ảnh ngẫu nhiên cần chọn
        input_text = self.num_images_entry.get()
        if input_text:
            num_images_to_select = int(input_text)
        else:
            num_images_to_select = 1

        # Chọn ngẫu nhiên ảnh từ thư mục
        IMAGES = []
        for label in LABELS:
            dir_path = os.path.join(DATA_DIR, label)
            for image_path in glob.glob(os.path.join(dir_path, '*.png')) + glob.glob(
                    os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.jpeg")):
                IMAGES.append((image_path, label))
        random.shuffle(IMAGES)
        IMAGES = IMAGES[:num_images_to_select]

        # Lưu danh sách các ảnh đã chọn
        self.selected_images = [image_path for image_path, _ in IMAGES]

        # Hiển thị các ảnh đã chọn
        self.show_selected_images()

    # Hiển thị các ảnh đã chọn  
    def show_selected_images(self):
        num_images = len(self.selected_images)

        if num_images > 100:
            num_rows = 10
        elif num_images > 20:
            num_rows = 5
        else:
            num_rows = 2

        num_cols = int(np.ceil(num_images / num_rows))
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 18))

        if num_rows == 1:
            axs = np.expand_dims(axs, axis=0)
        elif num_cols == 1:
            axs = np.expand_dims(axs, axis=1)

        for i in range(num_images):
            image_path = self.selected_images[i]
            row = i // num_cols
            col = i % num_cols

            detected_image = self.detect_and_display_image(image_path)
            nhan_goc = os.path.basename(os.path.dirname(image_path))

            # Kiểm tra xem hình ảnh đã được đọc và xử lý đúng không
            if detected_image is not None and isinstance(detected_image, np.ndarray):
                ax = axs[row, col]
                ax.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='auto')
                ax.axis('off')
                # Nhãn gốc
                # ax.set_title(nhan_goc, fontsize=18, color='red')
            else:
                ax = axs[row, col]
                ax.text(0.5, 0.5, 'Không thể tải ảnh', horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')

        for i in range(num_images, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            fig.delaxes(axs[row, col])

        plt.tight_layout()
        plt.show()
     

    def detect_and_display_image(self, image_path):
        image = cv2.imread(image_path)
        processed_image, detected_emotions = self.detect_face_emotions(image.copy())  # Lấy cả hình ảnh và dữ liệu cảm xúc

        # Cập nhật bộ đếm cảm xúc
        for emotion, count in detected_emotions.items():
            if count > 0:
                self.update_emotion_counter(emotion, count)

        return processed_image

    # Logic tab video 
    def update_canvas_size(self, event):
        # Kiểm tra xem tab hiện tại có phải là tab video không
        if self.notebook.index(self.notebook.select()) == 2:  # Tab video có chỉ số là 2 trong notebook
            # Cập nhật kích thước của canvas chỉ khi đang ở tab video
            self.video_result_canvas.config(width=event.width, height=event.height)
            
    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_file_path = file_path  # Cập nhật đường dẫn tệp video
            self.video_path_label.config(text=file_path)

    def start_toggle_video_thread(self):
        threading.Thread(target=self.toggle_video, daemon=True).start()
            
    def toggle_video(self):
        if self.is_running:
            self.stop_video()
        else:
            if self.video_file_path is not None:
                self.video_capture = cv2.VideoCapture(self.video_file_path)
                # Đặt lại vị trí khung hình tại vị trí đã lưu
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.is_running = True
                self.detect_video_button.config(text="Tắt kiểm tra video")
                self.display_video()
            else:
                messagebox.showwarning("Lỗi", "Không thể kiểm tra khi không chọn video.")
                
    def display_video(self):
        if self.is_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)  # Save current frame index
                self.last_frame = frame  # Save last frame
                detected_frame, detected_emotions_1 = self.detect_face_emotions(frame)  # Unpack the tuple
                detected_frame = cv2.resize(detected_frame, (640, 450))
                detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(detected_frame))
                self.video_result_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.video_result_canvas.image = photo
                self.video_result_canvas.after(10, self.display_video)
            else:
                self.stop_video()  # Stop video if no more frames
        else:
            self.stop_video()  # Stop video if capture not opened

    def stop_video(self):
        if self.video_capture:
            self.video_capture.release()
        self.is_running = False
        self.detect_video_button.config(text="Bật kiểm tra video")
        if self.last_frame is not None:
            # Hiển thị khung hình cuối cùng trên canvas nếu có
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)))
            self.video_result_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.video_result_canvas.image = photo
        else:
            self.video_result_canvas.delete("all")  # Xóa canvas nếu không có khung hình cuối cùng
            self.video_path_label.config(text="Chưa chọn video")  # Tùy chọn đặt lại nhãn đường dẫn

    def toggle_capture_video(self):
        if not self.capture_in_progress_video:
            if self.video_capture is not None and self.video_capture.isOpened():
                self.start_capture_video()
                self.capture_toggle_button_video.config(text="Tắt chụp video")
                self.check_video_end()  # Kiểm tra video kết thúc
            else:
                messagebox.showwarning("Lỗi", "Không thể bắt đầu chụp vì video không được mở.")
        else:
            self.stop_capture_video()
            self.capture_toggle_button_video.config(text="Bật chụp video")

    def check_video_end(self):
        if self.capture_in_progress_video and self.video_capture is not None and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                # Video đã chạy hết
                self.stop_capture_video()
                self.capture_toggle_button_video.config(text="Bật chụp video")
                self.is_video_closed = True
                # messagebox.showinfo("Thông báo", "Video đã kết thúc.")  # Thông báo khi video kết thúc
                messagebox.showwarning("Thông báo", "Video đã kết thúc.")  # Thông báo khi video kết thúc

    def start_capture_video(self):
        if not self.capture_in_progress_video:  # Kiểm tra xem có đang trong quá trình chụp không
            self.is_auto_capture_on = True
            self.capture_in_progress_video = True
            self.is_video_closed = False  # Đặt lại trạng thái của video khi bắt đầu chụp
            self.capture_frame()
        else:
            messagebox.showwarning("Lỗi", "Không thể bắt đầu chụp vì video đang được chụp.")

    def stop_capture_video(self):
        self.capture_in_progress_video = False
        self.is_auto_capture_on = False
        
    def capture_frame(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_capture_time
        
        if elapsed_time >= self.capture_total_minutes_seconds and self.capture_in_progress_video and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                # Lưu frame thành file ảnh
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}.png"
                save_path = os.path.join(self.save_directory, filename)

                # Chuyển đổi khung hình từ BGR sang RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Nhận diện khuôn mặt trong frame
                faces = self.detect_faces(rgb_frame)

                for i, (x, y, w, h) in enumerate(faces):
                    face = frame[y:y+h, x:x+w]  # Cắt ảnh khuôn mặt

                    # Thay đổi kích thước hình ảnh khuôn mặt thành 250x250
                    resized_face = cv2.resize(face, (250, 250))
                    
                    # Tạo đường dẫn và lưu ảnh khuôn mặt
                    face_path = os.path.join(self.save_face_directory, f"{timestamp}_face_{i}.png")
                    if self.checkbox_var_3.get():
                        cv2.imwrite(face_path, resized_face)
                    
                # Kiểm tra trạng thái của checkbox trước khi lưu ảnh gốc
                if self.checkbox_var_2.get():
                    cv2.imwrite(save_path, frame)
                
                # Cập nhật thời gian chụp cuối cùng
                self.last_capture_time = current_time
        self.video_result_canvas.after(100, self.capture_frame)  # Kiểm tra mỗi 100 ms

    def start_toggle_indirect_video_thread(self):
        threading.Thread(target=self.toggle_indirect_video, daemon=True).start()
            
    def toggle_indirect_video(self):
        if self.detecting_indirect_video:
            self.stop_indirect_video()
        else:
            if self.video_file_path is not None:
                self.video_capture = cv2.VideoCapture(self.video_file_path)
                if not self.video_capture.isOpened():
                    messagebox.showwarning("Lỗi", "Không thể mở video.")
                    return
                # print("Bắt đầu kiểm tra video gián tiếp")
                self.detecting_indirect_video = True
                self.detect_video_photos_button.config(text="Tắt kiểm tra video gián tiếp")
                self.is_auto_capture_on_indirect = True  # Bật chế độ chụp tự động
                self.process_indirect_video()
            else:
                messagebox.showwarning("Lỗi", "Không thể kiểm tra khi không chọn video.")

    def stop_indirect_video(self):
        if self.video_capture:
            self.video_capture.release()
        self.detecting_indirect_video = False
        self.detect_video_photos_button.config(text="Bật kiểm tra video gián tiếp")
        self.is_auto_capture_on_indirect = False  # Tắt chế độ chụp tự động
        # self.video_result_canvas.delete("all")

    def process_indirect_video(self):
        if not os.path.exists(self.cached_frame_folder):
            os.makedirs(self.cached_frame_folder)
        else:
            if self.clear_saved_data:
                for file in os.listdir(self.cached_frame_folder):
                    os.remove(os.path.join(self.cached_frame_folder, file))

        if not os.path.exists(self.saved_faces_folder):
            os.makedirs(self.saved_faces_folder)

        self.frame_rate = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        # Tính tổng số khung hình trong video
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tính toán khoảng thời gian giữa các khung hình cần lưu bằng giây
        interval_frames = int(self.frame_rate * self.capture_total_minutes_seconds)
        
        # Tính toán các khung hình cần nhận diện
        self.frame_positions = [i for i in range(0, total_frames, interval_frames)]
        
        # Bắt đầu xử lý các khung hình này
        self.process_next_frame_at_positions()

    def process_next_frame_at_positions(self):
        if self.detecting_indirect_video and self.frame_positions:
            # Lấy vị trí khung hình tiếp theo cần xử lý
            frame_position = self.frame_positions.pop(0)
            
            # Nhảy đến khung hình cần thiết
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            
            ret, frame = self.video_capture.read()
            if not ret:
                self.stop_indirect_video()
                print("Không đọc được khung hình")
                return
            
            # Chuyển đổi khung hình từ BGR sang RGB để nhận diện khuôn mặt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Lưu ảnh gốc (frame) trước khi vẽ nhãn hoặc nhận diện cảm xúc
            if self.checkbox_var_2.get():
                save_path = os.path.join(self.save_directory, f"frame_{frame_position}.png")
                cv2.imwrite(save_path, frame)
                # print(f"Lưu ảnh gốc khung hình: {save_path}")

            # Nhận diện khuôn mặt và lưu ảnh khuôn mặt nếu cần thiết
            faces = self.detect_faces(rgb_frame)
            # print(f"Số khuôn mặt nhận diện được tại khung hình {frame_position}: {len(faces)}")

            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]  # Cắt khuôn mặt từ ảnh gốc
                resized_face = cv2.resize(face_img, (250, 250))
                face_filename = os.path.join(self.saved_faces_folder, f"face_{frame_position}_{i}.png")
                if self.checkbox_var_3.get():
                    cv2.imwrite(face_filename, resized_face)
                    # print(f"Lưu ảnh khuôn mặt: {face_filename}")

            # Nhận diện cảm xúc và hiển thị frame đã nhận diện khuôn mặt (vẽ nhãn)
            detected_cached_frame, detected_emotions_1 = self.detect_face_emotions(frame)

            # Cập nhật bộ đếm cảm xúc
            for predicted_label, count in detected_emotions_1.items():
                if count > 0:
                    self.update_emotion_counter(predicted_label, count)

            # Lưu frame đã nhận diện cảm xúc (sau khi đã vẽ nhãn)
            frame_filename = os.path.join(self.cached_frame_folder, f"frame_{frame_position}.png")
            cv2.imwrite(frame_filename, detected_cached_frame)
            # print(f"Lưu frame đã vẽ nhãn cảm xúc: {frame_filename}")

            # Tiếp tục xử lý khung hình tiếp theo
            self.root.after(100, self.process_next_frame_at_positions)
        else:
            # Khi không còn khung hình nào để xử lý, dừng việc kiểm tra video
            # print("Đã xử lý xong toàn bộ video.")
            self.display_saved_frames()
            self.stop_indirect_video()





    def display_saved_frames(self):
        self.video_result_canvas.delete("all")
        frame_files = sorted(os.listdir(self.cached_frame_folder))
        if not frame_files:
            print("Không có khung hình để hiển thị")
        for frame_file in frame_files:
            frame_path = os.path.join(self.cached_frame_folder, frame_file)
            image = Image.open(frame_path)
            image = image.resize((640, 450), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.video_result_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.video_result_canvas.image = photo
            self.video_result_canvas.update()
            self.root.after(100)

    def on_closing(self):
        self.stop_indirect_video()
        self.root.destroy()    
    

    def load_room_menu(self, filename):
        # Đọc danh sách phòng từ tập tin
        room_ip_mapping = self.read_room_ip_mapping(filename)
        # Cập nhật menu phòng
        if room_ip_mapping:
            room_options = list(room_ip_mapping.keys())
            self.room_menu['values'] = room_options
            self.selected_room_ip = room_ip_mapping[room_options[0]]
        else:
            # Nếu không có phòng nào, đặt giá trị mặc định thành None hoặc xử lý theo ý muốn của bạn
            self.room_menu.set("Không có phòng")
            self.selected_room_ip = None
            
    # Tạo khung trắng tab camera trực tiếp
    def fill_canvas_with_white(self):
        self.live_camera_canvas.delete("all")
        self.live_camera_canvas.create_rectangle(0, 0, self.live_camera_canvas.winfo_width(), self.live_camera_canvas.winfo_height(), fill="white")
        
    # Logic kiểm danh sách phòng tab camera trực tiếp
    def populate_room_menu(self):
        # Đọc lại danh sách phòng từ tệp và cập nhật room_menu
        room_ip_mapping = self.read_room_ip_mapping("./rooms/rooms.txt")

        if room_ip_mapping:
            room_options = list(room_ip_mapping.keys())
            self.room_menu['values'] = room_options
            self.selected_room_var.set(room_options[0])  # Đặt giá trị chọn là phòng đầu tiên
            self.selected_room_ip = room_ip_mapping[room_options[0]]
        else:
            # Nếu không có phòng nào, đặt giá trị mặc định thành None hoặc xử lý theo ý muốn của bạn
            self.room_menu.set("Không có phòng")
            self.selected_room_ip = None
        
    def update_camera_list(self):
        # Đọc lại danh sách phòng từ tệp và cập nhật room_menu
        self.populate_room_menu()
        
        # Hiển thị thông báo cập nhật thành công
        messagebox.showinfo("Thông báo", "Danh sách phòng đã được cập nhật thành công.")
        
    def load_room_menu(self, filename):
        # Đọc danh sách phòng từ tập tin
        room_ip_mapping = self.read_room_ip_mapping(filename)

        # Cập nhật menu phòng
        if room_ip_mapping:
            room_options = list(room_ip_mapping.keys())
            self.room_menu['values'] = room_options
            self.selected_room_ip = room_ip_mapping[room_options[0]]
        else:
            # Nếu không có phòng nào, đặt giá trị mặc định thành None hoặc xử lý theo ý muốn của bạn
            self.room_menu.set("Không có phòng")
            self.selected_room_ip = None

    # Đọc ip phòng tab camera   
    def read_room_ip_mapping(self, filename):
        room_ip_mapping = {}
        with open(filename, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                room_number = parts[0].strip()
                room_name = parts[1].strip()
                ip_address = parts[2].strip()
                room_ip_mapping[f"{room_number}: {room_name}"] = ip_address
        return room_ip_mapping
    
    def start_toggle_camera_connection_thread(self):
        threading.Thread(target=self.toggle_camera_connection, daemon=True).start()

    # Logic kết nối camera trực tiếp
    def toggle_camera_connection(self):
        if self.is_camera_on:
            # Nếu webcam đang bật, thì ngắt kết nối
            self.is_camera_on = False
            if self.wc:
                self.wc.release()
            self.live_camera_canvas.create_rectangle(0, 0, 745, 595, fill="white")
            self.toggle_button_camera.config(text="Kết nối camera")
            self.display_live_camera()  # Hiển thị màn hình trắng khi ngắt kết nối
        else:
            # Nếu webcam chưa được bật, thì mở camera
            # Nhận phòng đã chọn từ Combobox
            selected_room = self.room_menu.get()

            # Kiểm tra xem phòng đã chọn
            if selected_room:
                # Cập nhật IP phòng đã chọn bằng phương thức
                room_ip_mapping = self.read_room_ip_mapping("./rooms/rooms.txt")
                self.selected_room_ip = room_ip_mapping.get(selected_room)

                # Kiểm tra xem IP có hợp lệ không
                if self.selected_room_ip:
                    # Xây dựng URL và kết nối với camera
                    self.url = f"http://{self.selected_room_ip}/video"
                    self.wc = cv2.VideoCapture(self.url)

                    # Kiểm tra kết nối camera có thành công không
                    if self.wc.isOpened():
                        self.is_camera_on = True
                        self.update_webcam()  
                        self.toggle_button_camera.config(text="Ngắt kết nối")
                        self.display_live_camera()  # Hiển thị trực tiếp hình ảnh từ camera
                    else:
                        # Xử lý trường hợp kết nối camera không thành công
                        messagebox.showwarning("Lỗi", "Không thể kết nối với máy ảnh.")
                else:
                    # Xử lý trường hợp camera đã bật sẵn
                    messagebox.showwarning("Cảnh báo", "Webcam đã bật.")
            else:
                # Xử lý trường hợp không chọn được phòng
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn phòng trước khi kết nối.")
                
    def display_live_camera(self):
        if self.is_camera_on:
            ret, frame = self.wc.read()
            if ret:
                frame, detected_emotions = self.detect_face_emotions(frame)  # Unpack the tuple returned by detect_face_emotions
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 580))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

                self.live_camera_canvas.delete("all")
                self.live_camera_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                self.live_camera_canvas.after(1, self.display_live_camera)

    def toggle_capture_camera(self):
        if not self.capture_in_progress_camera:
            if self.is_camera_on:
                self.start_capture_camera()
                self.capture_toggle_button_camera.config(text="Tắt chụp camera")
            else:
                messagebox.showwarning("Lỗi", "Không thể bắt đầu chụp vì camera chưa kết nối.")
        else:
            self.stop_capture_camera()
            self.capture_toggle_button_camera.config(text="Bắt đầu chụp camera")
            
    def start_capture_camera(self):
        if not self.capture_in_progress_camera:
            self.capture_in_progress_camera = True
            self.is_auto_capture_on_camera = True  # Bật chế độ chụp tự động
            if self.is_auto_capture_on_camera:
                self.capture_images_auto_camera()
        else:
            messagebox.showwarning("Lỗi", "Không thể bắt đầu chụp vì quá trình chụp đang diễn ra.")

    def stop_capture_camera(self):
        self.capture_in_progress_camera = False
        self.is_auto_capture_on_camera = False  # Tắt chế độ chụp tự động
        
    def capture_images_auto_camera(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_capture_time
        
        if elapsed_time >= self.capture_total_minutes_seconds and self.is_auto_capture_on_camera and self.capture_in_progress_camera:
            ret, frame = self.wc.read()
            if ret:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}.png"
                save_path = os.path.join(self.save_directory, filename)

                # Lưu khung gốc
                if self.checkbox_var_2.get():
                    cv2.imwrite(save_path, frame)

                # Phát hiện khuôn mặt trong khung
                faces = self.detect_faces(frame)

                for i, (x, y, w, h) in enumerate(faces):
                    face = frame[y:y+h, x:x+w]  # Cắt ảnh khuôn mặt
                    # Thay đổi kích thước hình ảnh khuôn mặt thành 250x250
                    resized_face = cv2.resize(face, (250, 250))

                    # Lưu hình ảnh khuôn mặt
                    face_path = os.path.join(self.save_face_directory, f"{timestamp}_face_{i}.png")
                    if self.checkbox_var_3.get():
                        cv2.imwrite(face_path, resized_face)

                # Cập nhật thời gian chụp cuối cùng
                self.last_capture_time = current_time

        # Đặt bộ đếm thời gian cho lần chụp tiếp theo
        self.capture_timer = self.live_camera_canvas.after(100, self.capture_images_auto_camera)  # Kiểm tra mỗi 100 ms


    def open_evaluation_results_data(self):
        # Mở hộp thoại để chọn file JSON trong thư mục mặc định
        file_path = filedialog.askopenfilename(
            initialdir=self.default_evaluation_results,
            filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            try:
                # Đọc nội dung từ file JSON
                with open(file_path, 'r') as json_file:
                    evaluation_results = json.load(json_file)

                # Xóa các hàng hiện có trong bảng trước khi thêm dữ liệu mới
                self.tab_evaluate_results_treeview.delete(*self.tab_evaluate_results_treeview.get_children())

                # Hiển thị kết quả trong Treeview
                for emotion, data in evaluation_results.items():
                    if emotion == "Total Images":
                        # Hiển thị tổng số ảnh
                        self.tab_evaluate_results_treeview.insert("", "end", values=("", "", "", ""))
                        self.tab_evaluate_results_treeview.insert("", "end", values=("Tổng số ảnh", data, "100%", ""))
                    else:
                        # Hiển thị từng nhãn cảm xúc với số lượng và phần trăm
                        self.tab_evaluate_results_treeview.insert("", "end", values=(emotion, data['count'], data['percentage']))

                # tk.messagebox.showinfo("Thành công", "Đã mở file lịch sử đánh giá thành công!")
            except Exception as e:
                tk.messagebox.showerror("Lỗi", f"Không thể mở file: {str(e)}")
             
    def insert_evaluation_results_data(self):
        # Xóa các hàng hiện có trong bảng
        self.tab_evaluate_results_treeview.delete(*self.tab_evaluate_results_treeview.get_children())

        # Tính tổng số lượng ảnh
        total_images = sum(self.emotion_counter.values())

        # Tạo một dictionary để lưu kết quả
        self.evaluation_results = {}

        # Thêm các dòng cho từng cảm xúc từ bộ đếm cảm xúc và tính phần trăm
        for emotion, count in self.emotion_counter.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            self.tab_evaluate_results_treeview.insert('', 'end', values=(emotion, count, f"{percentage:.2f}%"))
            # Lưu kết quả vào dictionary
            self.evaluation_results[emotion] = {
                "count": count,
                "percentage": f"{percentage:.2f}%"
            }

        # Thêm một dòng ô trống để cách các thông tin tổng quan
        self.tab_evaluate_results_treeview.insert("", "end", values=("", "", "", ""))

        # Thêm dòng đánh giá kết quả (Tổng số ảnh)
        self.tab_evaluate_results_treeview.insert("", "end", values=("Tổng số ảnh", total_images, "100%", ""))
        self.evaluation_results["Total Images"] = total_images
    
    def save_evaluation_results_data(self):
        # Kiểm tra nếu bảng trống (không có hàng nào)
        if not self.tab_evaluate_results_treeview.get_children():
            return  # Dừng việc lưu nếu bảng trống

        # Mở hộp thoại để chọn nơi lưu file JSON trong thư mục mặc định
        file_path = filedialog.asksaveasfilename(
            initialdir=self.default_evaluation_results,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                # Lưu kết quả vào file JSON
                with open(file_path, 'w') as json_file:
                    json.dump(self.evaluation_results, json_file, indent=4, ensure_ascii=False)
                tk.messagebox.showinfo("Thành công", "Kết quả đánh giá đã được lưu thành công!")
            except Exception as e:
                tk.messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")

    def delete_evaluation_results_data(self):
        # Xóa các hàng hiện có trong bảng Treeview
        self.tab_evaluate_results_treeview.delete(*self.tab_evaluate_results_treeview.get_children())
        # Đặt lại bộ đếm cảm xúc về 0
        self.emotion_counter = {emotion: 0 for emotion in self.EMOTION}
        # Đặt lại kết quả đánh giá
        self.evaluation_results = {}

    # Logic tab danh sách phòng
    def add_room(self, name_room, ip_room):
        # Kiểm tra xem name_room hoặc ip_room có ​​trống không
        if not name_room or not ip_room:
            messagebox.showwarning("Thông báo", "Dòng nhập không hợp lệ không được bỏ trống.")
            return

        # Kiểm tra khoảng trắng ở đầu name_room
        if name_room[0].isspace():
            messagebox.showwarning("Thông báo", "Kí tự đầu tiên của Tên phòng không hợp lệ.")
            return

        # Kiểm tra khoảng trắng ở đầu ip_room
        if ip_room[0].isspace():
            messagebox.showwarning("Thông báo", "Kí tự đầu tiên của Mã IP Camera không hợp lệ.")
            return

        # Đọc các phòng hiện có từ file
        existing_rooms = self.read_rooms()

        # Xác định id_room tiếp theo
        if existing_rooms:
            id_room = max(existing_rooms.keys()) + 1
        else:
            id_room = 1

        line = f"{id_room},{name_room},{ip_room}"
        with open("./rooms/rooms.txt", "a", encoding="utf-8") as file:
            file.write(line + "\n")

        self.show_room()

    def on_listbox_click(self, event):
        # Cập nhật tiện ích nhập thông tin của phòng đã chọn
        selected_index = self.classroom_treeview.curselection()
        if selected_index:
            selected_room = self.classroom_treeview.get(selected_index[0])
            room_info = selected_room.split()
            if len(room_info) >= 3:
                self.id_room.set(room_info[0])
                self.name_room.set(room_info[1])
                self.ip_room.set(room_info[2])
            else:
                messagebox.showwarning("Thông báo", "Dòng không hợp lệ trong danh sách phòng học.")

    def delete_selected_room(self):
        selected_item = self.classroom_treeview.selection()
        
        if selected_item:
            # Lấy thông tin từ mục được chọn
            room_info = self.classroom_treeview.item(selected_item)

            # Lấy ID phòng
            id_room = room_info['values'][0]

            # Gọi hàm xóa phòng
            self.delete_room(id_room)

            # Hiển thị lại danh sách phòng
            self.show_room()
        else:
            messagebox.showwarning("Thông báo", "Vui lòng chọn một dòng để xóa.")

    def delete_room(self, id_room):
        existing_rooms = self.read_rooms()

        try:
            id_room = int(id_room)
            del existing_rooms[id_room]
            
            # Lưu danh sách đã cập nhật vào tệp
            with open("./rooms/rooms.txt", "w", encoding="utf-8") as file:
                for room_id, (name_room, ip_room) in existing_rooms.items():
                    line = f"{room_id},{name_room},{ip_room}\n"
                    file.write(line)
        except ValueError:
            print(f"ID phòng không hợp lệ: {id_room}")

    def read_rooms(self):
        try:
            with open("./rooms/rooms.txt", "r", encoding="utf-8") as file:
                lines = file.readlines()
                rooms = {}
                for line in lines:
                    values = line.strip().split(',')
                    try:
                        if len(values) == 3:
                            room_id, name_room, ip_room = values
                            # Kiểm tra xem room_id đã tồn tại trong từ điển chưa
                            room_id = int(room_id)  # Chuyển room_id thành số nguyên
                            if room_id not in rooms:
                                rooms[room_id] = (name_room, ip_room)
                            else:
                                print(f"Bỏ qua room_id trùng lặp {room_id} trong 'rooms.txt': {line}")
                        else:
                            print(f"Bỏ qua dòng không hợp lệ trong 'rooms.txt': {line}")
                    except ValueError:
                        print(f"Bỏ qua dòng với giá trị không phải số nguyên: {line}")
                return rooms
        except FileNotFoundError:
            print("Không tìm thấy tệp 'rooms.txt'.")
            return {}

    def mask_ip_address(self, ip_address):
        # Lưu địa chỉ IP gốc
        self.original_ip = ip_address
        # Sử dụng chuỗi ký tự ẩn cố thay vì '*'
        hidden_characters = '*' * 20  # Thay đổi số lượng ký tự ẩn cố nếu cần
        return hidden_characters

    def show_room(self):
        # Xóa các mục hiện có trong Treeview
        self.classroom_treeview.delete(*self.classroom_treeview.get_children())

        # Đọc lớp học từ file
        existing_rooms = self.read_rooms()

        # Hiển thị các lớp học trong Treeview với địa chỉ IP bị che
        for room_id, (name_room, ip_room) in existing_rooms.items():
            # Gọi hàm che giấu IP và lưu lại địa chỉ IP gốc
            masked_ip_display = self.mask_ip_address(ip_room)

            self.classroom_treeview.insert("", tk.END, values=(room_id, name_room, masked_ip_display, self.original_ip))

    # Thêm hàm edit_selected_room
    def edit_selected_room(self):
        selected_item = self.classroom_treeview.selection()
        if selected_item:
            selected_room = self.classroom_treeview.item(selected_item[0])
            room_info = selected_room['values']
            if len(room_info) >= 4:
                room_id = room_info[0]
                name = room_info[1]
                masked_ip = room_info[2]
                original_ip = room_info[3]

                # Hiển thị mật khẩu đã ẩn trong ô chỉnh sửa
                self.edit_window = tk.Toplevel(self.root)
                self.edit_window.title("Sửa thông tin phòng")
                self.edit_window.geometry("300x150")

                Label(self.edit_window, text="Tên phòng:").grid(row=0, column=0, pady=5)
                name_entry = Entry(self.edit_window)
                name_entry.insert(0, name)
                name_entry.grid(row=0, column=1, pady=5)

                Label(self.edit_window, text="Mã IP camera:").grid(row=2, column=0, pady=5)
                original_ip_entry = Entry(self.edit_window)
                original_ip_entry.insert(0, original_ip)
                original_ip_entry.grid(row=2, column=1, pady=5)

                save_button = Button(self.edit_window, text="Lưu", command=lambda: self.save_edited_room(room_id, name_entry.get(), original_ip_entry.get()))
                save_button.grid(row=3, column=0, columnspan=2, pady=10)
        else:
            messagebox.showwarning("Thông báo", "Vui lòng chọn một dòng để sửa.")

    # Cập nhật hàm edit_room
    def edit_room(self):
        if not self.edit_window or not self.edit_window.winfo_exists():
            self.edit_selected_room()

    # Thêm hàm save_edited_room
    def save_edited_room(self, room_id, edited_name, edited_ip):
        # Kiểm tra tính hợp lệ của thông tin sửa
        if not edited_name or not edited_ip:
            messagebox.showwarning("Thông báo", "Dòng nhập không hợp lệ không được bỏ trống.")
            return

        # Lưu thông tin đã sửa vào danh sách phòng
        existing_rooms = self.read_rooms()
        existing_rooms[room_id] = (edited_name, edited_ip)

        # Ghi lại danh sách phòng vào tệp
        with open("./rooms/rooms.txt", "w", encoding="utf-8") as file:
            for room_id, (name, ip) in existing_rooms.items():
                line = f"{room_id},{name},{ip}\n"
                file.write(line)

        # Đóng cửa sổ sửa thông tin
        if self.edit_window and self.edit_window.winfo_exists():
            self.edit_window.destroy()

        # Hiển thị lại danh sách phòng
        self.show_room()
        
    def update_list(self):
        # Đọc danh sách phòng hiện tại từ tệp
        existing_rooms = self.read_rooms()
        messagebox.showinfo("Thông báo", "Cập nhật danh sách thành công.")

        # Gán lại các ID theo thứ tự
        updated_rooms = {}
        new_id = 1
        for old_id, (name, ip) in existing_rooms.items():
            updated_rooms[new_id] = (name, ip)
            new_id += 1

        # Ghi danh sách phòng đã cập nhật vào tệp
        with open("./rooms/rooms.txt", "w", encoding="utf-8") as file:
            for room_id, (name, ip) in updated_rooms.items():
                line = f"{room_id},{name},{ip}\n"
                file.write(line)

        # Làm mới Treeview
        self.show_room()

        # Xóa nội dung các trường nhập
        self.clear_entry_fields()

    def clear_entry_fields(self):
        self.id_room.set("")
        self.name_room.set("")
        self.ip_room.set("")

    # Logic tab settings
        
    def load_settings(self):
        if os.path.exists(self.config_file):
            # print(f"Đọc cấu hình từ tệp: {self.config_file}")
            try:
                with open(self.config_file, 'r') as file:
                    settings = json.load(file)
                    # print(f"Cấu hình đọc được: {settings}")
                    
                    # Cập nhật các giá trị từ cấu hình
                    self.capture_interval_minutes = settings.get("capture_interval_minutes", self.default_settings["capture_interval_minutes"])
                    self.capture_interval_seconds = settings.get("capture_interval_seconds", self.default_settings["capture_interval_seconds"])
                    
                    # Cập nhật giá trị cho các BooleanVar
                    self.checkbox_var_1.set(settings.get("checkbox_var_1", self.default_settings["checkbox_var_1"]))
                    self.checkbox_var_2.set(settings.get("checkbox_var_2", self.default_settings["checkbox_var_2"]))
                    self.checkbox_var_3.set(settings.get("checkbox_var_3", self.default_settings["checkbox_var_3"]))
                    
                    # Cập nhật các giá trị trong các entry widgets
                    self.entry_var_minutes.set(str(self.capture_interval_minutes))
                    self.entry_var_seconds.set(str(self.capture_interval_seconds))
                    
                    self.update_total_interval()
                    # Cập nhật giao diện
                    self.update_ui()
            except json.JSONDecodeError as e:
                # print(f"Lỗi khi đọc tệp cấu hình: {e}")
                self.reset_to_default()
        else:
            # print("Tệp cấu hình không tồn tại. Cài lại mặc định.")
            self.reset_to_default()

    def reset_to_default(self):
        # Tạo thư mục chứa tệp cấu hình nếu không tồn tại
        config_folder = os.path.dirname(self.config_file)
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
            print(f"Đã tạo thư mục cấu hình: {config_folder}")

        # Cài đặt lại mặc định
        self.capture_interval_minutes = self.default_settings["capture_interval_minutes"]
        self.capture_interval_seconds = self.default_settings["capture_interval_seconds"]
        self.entry_var_minutes.set(str(self.capture_interval_minutes))
        self.entry_var_seconds.set(str(self.capture_interval_seconds))
        self.checkbox_var_1.set(self.default_settings["checkbox_var_1"])
        self.checkbox_var_2.set(self.default_settings["checkbox_var_2"])
        self.checkbox_var_3.set(self.default_settings["checkbox_var_3"])
        self.update_total_interval()

        # Lưu cài đặt mặc định vào tệp cấu hình
        self.save_settings()
        print("Đã cài lại mặc định.")

    def update_total_interval(self):
        self.capture_total_minutes_seconds = (self.capture_interval_minutes * 60) + self.capture_interval_seconds

    def update_ui(self):
        # Cập nhật các giá trị của widget dựa trên cấu hình đã đọc
        self.entry_var_minutes.set(str(self.capture_interval_minutes))
        self.entry_var_seconds.set(str(self.capture_interval_seconds))
        self.checkbox_var_1.set(self.checkbox_var_1.get())
        self.checkbox_var_2.set(self.checkbox_var_2.get())
        self.checkbox_var_3.set(self.checkbox_var_3.get())
        # Cập nhật các widget khác nếu cần
        
        
    def save_settings(self):
        # Tạo thư mục chứa tệp cấu hình nếu không tồn tại
        config_folder = os.path.dirname(self.config_file)
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
            print(f"Đã tạo thư mục cấu hình: {config_folder}")

        settings = {
            "capture_interval_minutes": self.capture_interval_minutes,
            "capture_interval_seconds": self.capture_interval_seconds,
            "checkbox_var_1": self.checkbox_var_1.get(),
            "checkbox_var_2": self.checkbox_var_2.get(),
            "checkbox_var_3": self.checkbox_var_3.get()
        }

        # So sánh thiết lập hiện tại với thiết lập đã lưu trước đó
        if settings != self.saved_settings:
            with open(self.config_file, 'w') as file:
                json.dump(settings, file, indent=4)
            messagebox.showinfo("Thông báo", f"Đã lưu thiết lập")
            # Cập nhật thiết lập đã lưu
            self.saved_settings = settings
        # else:
        #     print("Không có thay đổi nào để lưu.")
            
    def apply_settings(self):
        try:
            minutes_str = self.entry_var_minutes.get().strip()
            seconds_str = self.entry_var_seconds.get().strip()

            # Kiểm tra nếu chuỗi nhập liệu trống
            if not minutes_str:
                raise ValueError("Trường phút không được bỏ trống.")
            if not seconds_str:
                raise ValueError("Trường giây không được bỏ trống.")
            
            # Chuyển đổi giá trị phút và giây
            self.capture_interval_minutes = int(minutes_str)
            self.capture_interval_seconds = int(seconds_str)
            
            # Kiểm tra giá trị phút và giây
            if self.capture_interval_minutes < 0:
                raise ValueError("Giá trị phút phải lớn hơn hoặc bằng 0.")
            
            if self.capture_interval_minutes >= 1:
                if self.capture_interval_seconds < 0:
                    raise ValueError("Giá trị giây phải lớn hơn hoặc bằng 0 khi phút lớn hơn hoặc bằng 1.")
            else:
                if self.capture_interval_seconds <= 0:
                    raise ValueError("Giá trị giây phải lớn hơn 0 khi phút nhỏ hơn 1.")
            
            self.update_total_interval()
            self.save_settings()
        except ValueError as e:
            messagebox.showerror("Lỗi Nhập Liệu", f"Vui lòng nhập một số nguyên hợp lệ. {e}")

    def update_capture_interval(self, event):
        try:
            minutes_str = self.entry_var_minutes.get().strip()
            seconds_str = self.entry_var_seconds.get().strip()

            if not minutes_str:
                raise ValueError("Trường phút không được bỏ trống.")
            if not seconds_str:
                raise ValueError("Trường giây không được bỏ trống.")
            
            self.capture_interval_minutes = int(minutes_str)
            self.capture_interval_seconds = int(seconds_str)
            
            if self.capture_interval_minutes < 0:
                raise ValueError("Giá trị phút phải lớn hơn hoặc bằng 0.")
            
            if self.capture_interval_minutes >= 1:
                if self.capture_interval_seconds < 0:
                    raise ValueError("Giá trị giây phải lớn hơn hoặc bằng 0 khi phút lớn hơn hoặc bằng 1.")
            else:
                if self.capture_interval_seconds <= 0:
                    raise ValueError("Giá trị giây phải lớn hơn 0 khi phút nhỏ hơn 1.")
            
            self.update_total_interval()
        except ValueError as e:
            messagebox.showerror("Lỗi Nhập Liệu", f"Vui lòng nhập một số nguyên hợp lệ. {e}")


    def show_frame(self, frame):
        # Ẩn tất cả các nội dung
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()

        # Hiển thị nội dung của tab được chọn
        frame.pack(fill=tk.BOTH, expand=True)
          
if __name__ == "__main__":
    DATA_DIR = r"data_cam_xuc"
    # LABELS = ["binh_thuong", "buon", "cuoi"]
    LABELS = ["binh_thuong", "buon", "cuoi", "ngac_nhien", "so_hai", "tuc_gian"]

    root = tk.Tk()
    app = FaceMaskDetectionApp(root)
    root.mainloop()