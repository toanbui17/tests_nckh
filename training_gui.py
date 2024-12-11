import os
import cv2
import numpy as np
import glob
import time
import math
import json
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel
import matplotlib.pyplot as plt
import threading

class TrainingGUI(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.THUMUC = r"./data_cam_xuc"
        self.LOAI = ["binh_thuong", "buon", "cuoi", "ngac_nhien", "so_hai", "tuc_gian"]
        self.du_lieu = []
        self.nhan = []
        self.model = None
        self.history = None
        

        # Thiết lập tab kết quả
        self.result_label = tk.Label(self, text="Kết quả huấn luyện", fg="red", font=("Cambria", 16), width=66)
        self.result_label.grid(row=0, pady=10, column=0, columnspan=3)

        self.result_treeview = ttk.Treeview(self, selectmode="browse", columns=("emotion_status","support","recall",), show="headings", height=20)
        self.result_treeview.grid(row=1, padx=10, pady=10, column=0, columnspan=3)

        # Thiết lập các tiêu đề cột kết quả
        self.result_treeview.heading("emotion_status", text="Trạng thái cảm xúc")
        self.result_treeview.heading("support", text="Số cảm xúc")
        self.result_treeview.heading("recall", text="Độ chính xác")
    
        # Chèn dữ liệu vào Treeview
        self.result_treeview.delete(*self.result_treeview.get_children())

        # Tạo khung chứa các nút bên dưới bảng
        buttons_tab_train_1 = tk.Frame(self)
        buttons_tab_train_1.grid(row=2, column=1, columnspan=3, pady=10)

        # Nút bắt đầu huấn luyện
        self.start_train_button = tk.Button(buttons_tab_train_1, text="Bắt đầu huấn luyện", command=self.train_model)
        self.start_train_button.pack(side=tk.LEFT, padx=5)

        # Nút mở file khác
        self.open_file_train_evaluate_button = tk.Button(buttons_tab_train_1, text="Mở file lịch sử đánh giá mô hình huấn luyện", command=self.open_evaluation_train_file)
        self.open_file_train_evaluate_button.pack(side=tk.LEFT, padx=5)
                  
    def open_evaluation_train_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file JSON", 
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    history_dict = json.load(f)
                
                # Lấy thông tin từ file JSON
                report = history_dict.get("classification_report", {})
                total_images = history_dict.get("total_images", 0)
                total_train_images = history_dict.get("train_images", 0)
                total_test_images = history_dict.get("test_images", 0)
                accuracy_percentage = history_dict.get("accuracy_percentage", 0.0)
                
                # Cập nhật dữ liệu vào Treeview
                self.insert_evaluation_train_data(report)
                
                # Thêm các thông tin tổng quan
                self.result_treeview.insert("", "end", values=("Tổng số ảnh:", total_images, ""))
                self.result_treeview.insert("", "end", values=("Tổng số ảnh Huấn luyện:", total_train_images, ""))
                self.result_treeview.insert("", "end", values=("Tổng số ảnh kiểm tra:", total_test_images, ""))
                self.result_treeview.insert("", "end", values=("Độ chính xác mô hình:", "{:.2f}%".format(accuracy_percentage), ""))
                
                # Hiển thị biểu đồ nếu có dữ liệu
                if "accuracy" in history_dict or "loss" in history_dict:
                    self.history = tf.keras.callbacks.History()
                    self.history.history = {k: v for k, v in history_dict.items() if isinstance(v, list)}
                    self.plot_training_history()

            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể mở file: {e}")




    def insert_evaluation_train_data(self, report):
        # Danh sách các nhãn cảm xúc
        target_names = ["binh_thuong", "buon", "cuoi", "ngac_nhien", "so_hai", "tuc_gian"]

        # Xóa các hàng hiện có trong bảng
        self.result_treeview.delete(*self.result_treeview.get_children())

        # Thêm các dòng cho từng cảm xúc từ báo cáo phân loại
        for emotion in target_names:
            if emotion in report:
                metrics = report[emotion]
                support = metrics.get('support', 0)
                recall = metrics.get('recall', 0)
                precision = metrics.get('precision', 0)
                f1_score = metrics.get('f1-score', 0)
                # Định dạng các số liệu với 2 chữ số sau dấu phẩy
                self.result_treeview.insert('', 'end', values=(
                    emotion,
                    f"{support:.0f}",
                    f"{recall * 100:.2f}%",
                    f"{precision:.2f}",
                    f"{f1_score:.2f}"
                ))
            else:
                # Nếu không có lớp nào trong báo cáo, hãy hiển thị giá trị mặc định
                self.result_treeview.insert('', 'end', values=(
                    emotion,
                    "00.00",
                    "00.00",
                    "00.00",
                    "00.0"
                ))

        # Thêm một dòng ô trống để cách các thông tin tổng quan
        self.result_treeview.insert("", "end", values=("", "", "", ""))
        
        # Thêm dòng đánh giá kết quả
        self.result_treeview.insert("", "end", values=("Đánh giá kết quả", "", "", ""))
        
    def load_data(self):
        print("ĐANG TẢI HÌNH ẢNH...")
        self.du_lieu = []
        self.nhan = []
        
        for loai in self.LOAI:
            duong_dan = os.path.join(self.THUMUC, loai)
            tap_tin = glob.glob(os.path.join(duong_dan, "*.jpg")) + glob.glob(os.path.join(duong_dan, "*.png"))
            
            for tap_tin_path in tap_tin:
                anh = cv2.imread(tap_tin_path)
                anh = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
                anh = cv2.resize(anh, (48, 48))
                self.du_lieu.append(anh)
                self.nhan.append(loai)

        lb = LabelBinarizer()
        self.nhan = lb.fit_transform(self.nhan)
        self.du_lieu = np.array(self.du_lieu).reshape(len(self.du_lieu), 48, 48, 1)
        self.LOAI = lb.classes_

        # Trả về dữ liệu và nhãn đã nạp
        return self.du_lieu, self.nhan

    def update_treeview(self, classification_rep, total_images, total_train_images, total_test_images, accuracy_percentage):
        # Xóa tất cả các mục hiện có
        self.result_treeview.delete(*self.result_treeview.get_children())

        # Thêm các dòng cho từng cảm xúc từ báo cáo phân loại
        for emotion, metrics in classification_rep.items():
            if emotion in self.LOAI:
                self.result_treeview.insert("", "end", values=(emotion, "{:.0f}".format(metrics["support"]), "{:.2f}%".format(metrics["recall"] * 100)))

        # Thêm một dòng ô trống để cách các thông tin tổng quan
        self.result_treeview.insert("", "end", values=("", "", ""))
        
        # Thêm dòng đánh giá kết quả
        self.result_treeview.insert("", "end", values=("Đánh giá kết quả", "", ""))
        
        # Thêm các dòng cho tổng số ảnh kiểm tra và độ chính xác
        self.result_treeview.insert("", "end", values=("Tổng số ảnh:", total_images, ""))
        self.result_treeview.insert("", "end", values=("Tổng số ảnh huấn luyện:", total_train_images, ""))
        self.result_treeview.insert("", "end", values=("Tổng số ảnh kiểm tra:", total_test_images, ""))
        self.result_treeview.insert("", "end", values=("Độ chính xác mô hình:", "{:.2f}%".format(accuracy_percentage), ""))       

    def show_progress_window(self):
        progress_window = Toplevel(self)
        progress_window.title("Tiến độ đào tạo")
        progress_window.geometry("300x100")

        progress_label = ttk.Label(progress_window, text="Tiến độ đào tạo: 0%")
        progress_label.pack(pady=20)
        # Tạo thanh tiến trình
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        progress_bar.pack(pady=10)

        return progress_window, progress_label, progress_bar

    def update_progress(self, progress_window, progress_label, progress_bar, epoch, total_epochs):
        progress = (epoch / total_epochs) * 100
        progress_label.config(text=f"Tiến độ đào tạo: {int(progress)}%")
        progress_bar['value'] = progress
        progress_window.update_idletasks()

    def train_model(self):
        # Bắt đầu tiến trình mới để đào tạo
        training_thread = threading.Thread(target=self._train_model)
        training_thread.start()

    def _train_model(self):
        # Create a progress window for the training
        progress_window, progress_label, progress_bar = self.show_progress_window()

        # Load your data here
        self.du_lieu, self.nhan = self.load_data()
        tap_huan_luyen_X, tap_kiem_tra_X, tap_huan_luyen_Y, tap_kiem_tra_Y = train_test_split(
            self.du_lieu, self.nhan, test_size=0.20, stratify=self.nhan, random_state=math.floor(time.time())
        )
        
        # Create a CNN model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.LOAI), activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Image augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        epochs = 50
        
        # Get current timestamp for the model filename
        timestamp = time.strftime('%Y%m%d_%H%M%S')  # Format: 'YYYYMMDD_HHMMSS'
        file_name = f'train/demo_{timestamp}'  # Include timestamp in the filename

        # Checkpoint to save the best model
        # checkpoint = ModelCheckpoint(f'{file_name}.h5', save_best_only=True)
        checkpoint = ModelCheckpoint(f'{file_name}.h5', save_best_only=True, save_format="h5")

        # Create a custom callback for progress update
        class ProgressCallback(Callback):
            def __init__(self, app, progress_window, progress_label, progress_bar, total_epochs):
                self.app = app
                self.progress_window = progress_window
                self.progress_label = progress_label
                self.progress_bar = progress_bar
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                self.app.after(0, self.app.update_progress, self.progress_window, self.progress_label, self.progress_bar, epoch + 1, self.total_epochs)

        progress_callback = ProgressCallback(self, progress_window, progress_label, progress_bar, epochs)

        # Train the model
        self.history = self.model.fit(datagen.flow(tap_huan_luyen_X, tap_huan_luyen_Y, batch_size=32), 
                                      epochs=epochs, 
                                      validation_data=(tap_kiem_tra_X, tap_kiem_tra_Y), 
                                      callbacks=[checkpoint, progress_callback],
                                      verbose=0  # Disable verbose output
                                  )

        # Close the progress window when training is done
        self.after(0, progress_window.destroy)

        # Display a completion message
        # self.after(0, messagebox.showinfo, "Model Saved", f"Model saved to: {file_name}.h5")
        
        # Evaluate the model
        self.evaluate_model(tap_huan_luyen_X, tap_kiem_tra_X, tap_huan_luyen_Y, tap_kiem_tra_Y, file_name)
        
    # Huấn luyện mô hình và in ra Treeview
    def evaluate_model(self, tap_huan_luyen_X, tap_kiem_tra_X, tap_huan_luyen_Y, tap_kiem_tra_Y, file_name):
        print("Đánh giá mô hình...")
        du_doan = self.model.predict(tap_kiem_tra_X)
        y_true = np.argmax(tap_kiem_tra_Y, axis=1)
        y_pred = np.argmax(du_doan, axis=1)

        do_chinh_xac = accuracy_score(y_true, y_pred)
        phan_tram_du_doan_dung = do_chinh_xac * 100
        total_images = len(tap_huan_luyen_X) + len(tap_kiem_tra_X) # Tổng số ảnh
        total_train_images = len(tap_huan_luyen_X)  # Tổng số ảnh huấn luyện
        total_test_images = len(tap_kiem_tra_X)  # Tổng số ảnh kiểm tra

        # In ra kết quả đánh giá độ chính xác của mô hình
        print("Tổng số ảnh trong dữ liệu:", len(tap_huan_luyen_X) + len(tap_kiem_tra_X), "ảnh")
        print("Tổng số ảnh huấn luyện:", len(tap_huan_luyen_X), "ảnh")
        print("Tổng số ảnh kiểm tra:", len(tap_kiem_tra_X), "ảnh")
        print("Phần trăm dự đoán đúng trên tập kiểm tra: {:.2f}%".format(phan_tram_du_doan_dung))

        # Tạo báo cáo phân loại
        classification_rep = classification_report(
            y_true,
            y_pred,
            target_names=self.LOAI,
            output_dict=True,
            zero_division=0
        )

        # Cập nhật dữ liệu cho bảng sau khi huấn luyện xong
        self.insert_evaluation_train_data(classification_rep)  # Pass only the classification report
        # Cập nhật Treeview với kết quả sau khi huấn luyện xong
        self.update_treeview(classification_rep, total_images, total_train_images, total_test_images, phan_tram_du_doan_dung)

        # sau khi huấn luyện xong in chi tiết báo cáo phân loại
        print("Chi tiết báo cáo phân loại:")
        classification_rep1 = classification_report(np.argmax(tap_kiem_tra_Y, axis=1), np.argmax(du_doan, axis=1), target_names=self.LOAI)
        print(classification_rep1)

        history_dict = self.history.history
        additional_info = {
            "total_images": total_images,
            "train_images": total_train_images,
            "test_images": total_test_images,
            "accuracy_percentage": do_chinh_xac * 100,
            "classification_report": classification_rep
        }
        history_dict.update(additional_info)
        json_file_name = f'{file_name}.json'
        with open(json_file_name, 'w') as f:
            json.dump(history_dict, f)

        # Thông báo khi lưu file JSON
        print(f"Lịch sử huấn luyện đã được lưu vào file: {json_file_name}")
        # messagebox.showinfo("Lưu lịch sử huấn luyện", f"Lịch sử huấn luyện đã được lưu vào file: {json_file_name}")
        self.plot_training_history()
        
    def plot_training_history(self):
        # Lên lịch cho chức năng vẽ đồ thị chạy trên main thread
        self.after(0, self._plot_training_history)
    def _plot_training_history(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 2])
        plt.legend(loc='upper right')
        plt.show()