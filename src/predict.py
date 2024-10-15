import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Đường dẫn đến thư mục test
test_data_dir = '../small_data/small_test'  # Thay đổi thành đường dẫn thực tế đến thư mục

# Tải mô hình đã được lưu
model = load_model('model_best.keras')

# In cấu trúc mô hình
print(model.summary())

# Khởi tạo danh sách để chứa các dự đoán
predictions = []

# Lặp qua từng ảnh trong thư mục
for filename in os.listdir(test_data_dir):
    if filename.endswith('.JPG') or filename.endswith('.jpeg') or filename.endswith('.png'):
        img_path = os.path.join(test_data_dir, filename)

        # Tải và tiền xử lý ảnh
        img = load_img(img_path, target_size=(150, 150))  # Kích thước giống như đã sử dụng trong huấn luyện
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Chuẩn hóa dữ liệu
        img_array = np.expand_dims(img_array, axis=0)  # Thêm một chiều để mô hình có thể nhận diện

        # Dự đoán
        prediction = model.predict(img_array)
        predictions.append((filename, prediction[0][0]))  # Lưu tên file và dự đoán

# In kết quả
for filename, pred in predictions:
    label = 'Class 1' if pred > 0.5 else 'Class 0'  # Thay đổi theo lớp thực tế
    print(f'File: {filename}, Predicted: {label}, Confidence: {pred:.4f}')