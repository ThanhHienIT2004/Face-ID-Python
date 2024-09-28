import cv2
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

# Khởi tạo giao diện đồ họa
root = tk.Tk()
root.withdraw()  # Ẩn cửa sổ chính

# Nhập ID khuôn mặt
face_id = simpledialog.askstring("Lỗi","Nhập ID Khuôn Mặt:")

if not face_id:
    messagebox.showerror("Lỗi", "ID Khuôn Mặt không được để trống")
    exit()

# Khởi tạo camera
cam = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở được không
if not cam.isOpened():
    messagebox.showerror("Lỗi", "Không thể mở camera")
    exit()

cam.set(3, 640)  # Chiều rộng khung hình
cam.set(4, 480)  # Chiều cao khung hình

# Sử dụng bộ nhận diện khuôn mặt Haar Cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

messagebox.showinfo("Thông tin", "Chấp nhận bật camera")

count = 0

while True:
    ret, img = cam.read()
    if not ret:
        messagebox.showerror("Lỗi", "Không thể nhận diện khung hình từ camera")
        break

    img = cv2.flip(img, 1)  # Lật hình ảnh video theo chiều ngang (flip mã -1 lật cả ngang và dọc)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang màu xám
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Lưu lại khuôn mặt đã nhận diện
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

    # Hiển thị hình ảnh trong cửa sổ OpenCV
    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Nhấn ESC để thoát
    if k == 27:
        break
    elif count >= 30:  # Dừng lại sau khi đã lấy được 30 ảnh
        break

messagebox.showinfo("Thông tin", "Thoát")
cam.release()
cv2.destroyAllWindows()
