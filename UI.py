import tkinter as tk
from tkinter import ttk
import subprocess

def run_train():
    subprocess.Popen(['python', 'Train.py'])

def run_face_detect():
    subprocess.Popen(['python', 'FaceDetect.py'])

def run_recognize():
    subprocess.Popen(['python', 'Recognize.py'])

# Tạo giao diện đồ họa
root = tk.Tk()
root.title("Face Recognition System")

# Sử dụng ttk.Style để tùy chỉnh giao diện
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=10)

# Nút để chạy Train.py
train_button = ttk.Button(root, text="Train Faces", command=run_train)
train_button.pack(pady=12)

# Nút để chạy FaceDetect.py
face_detect_button = ttk.Button(root, text="Capture Faces", command=run_face_detect)
face_detect_button.pack(pady=12)

# Nút để chạy Recognize.py
recognize_button = ttk.Button(root, text="Recognize Faces", command=run_recognize)
recognize_button.pack(pady=12)

root.mainloop()
