import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import numpy as np
import time
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class CutImageGUI(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # Tải mô hình nhận diện khuôn mặt
        self.pb_path = "models/opencv_face_detector_uint8.pb"
        self.config_path = "models/opencv_face_detector.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(self.pb_path, self.config_path)

        # Khởi tạo các biến cần thiết
        self.webcam_active = False
        self.active_capture = None
        self.last_time = 0
        # Thư mục lưu hình ảnh
        self.save_directory = "khuon_mat"
        self.original_image_directory = "khuon_mat_goc"
        os.makedirs(self.original_image_directory, exist_ok=True)
        os.makedirs(self.save_directory, exist_ok=True)

        
        self.start_button = tk.Button(self, text="Mở Webcam", command=self.toggle_webcam)
        self.start_button.pack(pady=10)

       
        # Nút xử lý tập tin video
        self.process_media_button = tk.Button(self, text="Chọn Video", command=self.process_media)
        self.process_media_button.pack(pady=10)

        # Nút chọn ảnh
        self.select_images_button = tk.Button(self, text="Chọn Hình Ảnh", command=self.select_and_process_images)
        self.select_images_button.pack(pady=10)

        # Bảng hiển thị video
        self.panel = tk.Label(self)
        self.panel.pack()

    # Chuyển đổi trạng thái webcam
    def toggle_webcam(self):
        self.webcam_active = not self.webcam_active

        if self.webcam_active:
            self.start_button.config(text="Dừng Webcam")
            self.start_webcam()
        else:
            self.start_button.config(text="Mở Webcam")
            self.stop_active_capture()

    # Phát hiện khuôn mặt và vẽ hình chữ nhật
    def detect_and_draw_faces(self, frame):
        frame_bgr = frame.copy()  # Copy original BGR frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame_rgb.copy()  # Copy for display

        blob = cv2.dnn.blobFromImage(frame_rgb, 1.0, (300, 300), [104, 117, 123], True, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []

        # Vẽ hình chữ nhật xung quanh các khuôn mặt được phát hiện
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x_start, y_start, x_end, y_end) = box.astype("int")
                cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                face = frame_rgb[y_start:y_end, x_start:x_end]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    faces.append(face)
        return display_frame, faces, frame_bgr  # Trả về khung hiển thị và khuôn mặt được phát hiện

    # Lưu khuôn mặt được phát hiện và khung gốc
    def save_faces(self, faces, original_frame):
        current_time = time.time()

        # Lưu khung hình gốc nếu phát hiện bất kỳ khuôn mặt nào
        if faces:  # Kiểm tra xem có phát hiện được khuôn mặt nào không
            original_filename = f"{self.original_image_directory}/khuon_mat_goc_{int(current_time)}.png"
            cv2.imwrite(original_filename, original_frame)

        if current_time - self.last_time >= 1:  # Lưu lại mỗi 1 giây
            for idx, face in enumerate(faces):
                resized_face = cv2.resize(face, (250, 250))  # Thay đổi kích thước khuôn mặt thành 250x250 pixel
                # Create unique filename
                face_filename = f"{self.save_directory}/khuon_mat_{int(current_time)}_{idx}.png"
                cv2.imwrite(face_filename, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
            self.last_time = current_time

    def save_faces_images(self, faces, original_frame, current_index):
        current_time = time.time()  # Lấy thời gian hiện tại
        # Lưu khuôn mặt được phát hiện
        for idx, face in enumerate(faces):
            resized_face = cv2.resize(face, (250, 250))  # Thay đổi kích thước khuôn mặt thành 250x250 pixel
            # Create unique filename
            face_filename = f"{self.save_directory}/khuon_mat_{current_index}_{int(current_time)}_{idx}.png"
            cv2.imwrite(face_filename, resized_face)

    # Xử lý từng khung hình để nhận diện và lưu khuôn mặt
    def process_frame(self, frame):
        processed_frame, faces, original_frame = self.detect_and_draw_faces(frame)
        self.save_faces(faces, original_frame)
        return processed_frame

    # Bắt đầu chụp từ webcam hoặc thiết bị video
    def capture_from_device(self, device):
        self.stop_active_capture()
        cap = cv2.VideoCapture(device)
        self.active_capture = cap

        while self.webcam_active or device != 0:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            img = Image.fromarray(processed_frame)
            img_tk = ImageTk.PhotoImage(image=img)

            self.panel.img = img_tk
            self.panel.config(image=img_tk)
            self.panel.update()

            if cv2.waitKey(1) == ord('q') or self.active_capture != cap:
                break

        cap.release()
        self.active_capture = None
        cv2.destroyAllWindows()

    # Bắt đầu webcam
    def start_webcam(self):
        self.webcam_active = True
        self.capture_from_device(0)

    # Dừng chụp từ webcam
    def stop_active_capture(self):
        if self.active_capture:
            self.active_capture.release()
            self.active_capture = None
            self.panel.config(image='')

    # Xử lý tập tin video
    def process_media(self):
        # Chỉ cho phép các định dạng video phổ biến
        filetypes = [("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv")]
        file_path = filedialog.askopenfilename(title="Chọn file video", filetypes=filetypes)

        if file_path:
            self.capture_from_device(file_path)
        else:
            messagebox.showerror("Lỗi", "Không mở được tập tin.")

    # Chọn và xử lý hình ảnh
    def select_and_process_images(self):
        file_paths = filedialog.askopenfilenames(title="Chọn hình ảnh", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_paths:
            # Xử lý từng hình ảnh
            images = []
            for idx, file_path in enumerate(file_paths):
                img = cv2.imread(file_path)  # Đọc hình ảnh ở định dạng BGR
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang RGB để xử lý
                processed_frame, faces, _ = self.detect_and_draw_faces(img_rgb)  # Sử dụng chức năng phát hiện hiện có
                self.save_faces_images(faces, img_rgb, idx)  # Lưu khuôn mặt được phát hiện vào thư mục
                
                # Chuyển đổi khung đã xử lý sang RGB trước khi thêm vào danh sách
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                images.append(processed_frame_rgb)  # Thêm vào danh sách ảnh đã xử lý
            
            # Hiển thị hình ảnh được phát hiện trong tab mới
            self.display_images_in_new_tab(images)
        else:
            messagebox.showerror("Lỗi", "Không có hình ảnh nào được chọn.")

    # Hiển thị hình ảnh được phát hiện trong một cửa sổ riêng
    def display_images_in_new_tab(self, images):
        fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
        
        # Điều chỉnh cho một trường hợp hình ảnh
        if len(images) == 1:
            axs = [axs]
        
        for i, img in enumerate(images):
            axs[i].imshow(img)
            axs[i].axis('off')  
        
        plt.tight_layout()
        plt.show()  
