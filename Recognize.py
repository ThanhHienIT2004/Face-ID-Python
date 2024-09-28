import cv2
import numpy as np
import os

# Kiểm tra sự tồn tại của tệp mô hình
if not os.path.exists('trainer/trainer.yml'):
    print("Tệp mô hình không tồn tại. Thoát.")
    exit()

# Khởi tạo bộ nhận diện khuôn mặt LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Kiểm tra sự tồn tại của tệp cascade
if not os.path.exists("haarcascade_frontalface_default.xml"):
    print("Tệp cascade không tồn tại. Thoát.")
    exit()

# Sử dụng bộ nhận diện khuôn mặt Haar Cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Định dạng phông chữ hiển thị
font = cv2.FONT_HERSHEY_SIMPLEX

# Tên tương ứng với các ID khuôn mặt
names = ['0', 'Hien', 'My Tam','Son Tung', 'Jack', '5','6','7']

# Khởi tạo camera
cam = cv2.VideoCapture(0)  # Camera mặc định
cam.set(3, 640)  # Chiều rộng khung hình
cam.set(4, 480)  # Chiều cao khung hình

# Kích thước tối thiểu để nhận diện
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("Không thể đọc từ camera. Thoát.")
        break

    img = cv2.flip(img, 1)  # Lật ảnh theo chiều dọc

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dự đoán ID khuôn mặt và độ tin cậy
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Kiểm tra độ tin cậy (confidence)
        if confidence < 60:
            # Nếu độ tin cậy nhỏ hơn 60, coi là nhận diện khuôn mặt
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            # Nếu độ tin cậy lớn hơn hoặc bằng 100, coi là không nhận diện được
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Hiển thị ID và độ tin cậy trên hình ảnh
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h + 5), font, 1, (255, 255, 0), 1)

    # Hiển thị khung hình
    cv2.imshow('Nhận diện khuôn mặt', img)

    # Đọc phím nhấn
    k = cv2.waitKey(1) & 0xFF  # Sử dụng tham số 1 để kiểm tra phím nhấn liên tục
    def on_closing():

# Giải phóng bộ nhớ và đóng cửa sổ
        cam.release()
        cv2.destroyAllWindows()
print("\n [INFO] Thoát")

