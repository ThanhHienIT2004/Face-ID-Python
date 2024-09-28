import cv2
import numpy as np
from PIL import Image
import os
import tkinter as tk
from tkinter import messagebox

# Đường dẫn tới tập dữ liệu
path = 'dataset'

# Khởi tạo bộ nhận diện khuôn mặt LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Sử dụng bộ nhận diện khuôn mặt Haar Cascade
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Hàm lấy hình ảnh và nhãn từ thư mục
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # Mở hình ảnh và chuyển đổi sang thang độ xám
        PIL_img = Image.open(imagePath).convert('L')  # Chuyển sang ảnh grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        # Lấy ID từ tên tệp hình ảnh
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Nhận diện khuôn mặt trong hình ảnh
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

def train_faces():
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    messagebox.showinfo("Thông báo", f"{len(np.unique(ids))} khuôn mặt đã được train. Thoát.")

# Tạo giao diện đồ họa
root = tk.Tk()
root.title("Train Face Recognition")

train_button = tk.Button(root, text="Train Faces", command=train_faces)
train_button.pack(pady=20)

root.mainloop()
