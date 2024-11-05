import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model_path = "best_model.keras"  # Đường dẫn đến mô hình đã huấn luyện
best_model = tf.keras.models.load_model(model_path)

# Định nghĩa đường dẫn đến thư mục dữ liệu test và các thuộc tính hình ảnh
test_data_dir = "../small_data/small_test"  # Thư mục chứa dữ liệu test
train_data_dir = "../small_data/small_train"  # Thư mục chứa dữ liệu huấn luyện
image_size = (64, 64)  # Kích thước đầu vào của mô hình
batch_size = 32  # Kích thước batch

# Kiểm tra thư mục dữ liệu test có tồn tại và chứa hình ảnh
if not os.path.exists(test_data_dir) or len(os.listdir(test_data_dir)) == 0:
    raise ValueError("Thư mục dữ liệu test rỗng hoặc không tồn tại. Vui lòng kiểm tra đường dẫn và cấu trúc.")

# Tải hình ảnh từ thư mục test
test_images = []
file_names = []
for file_name in os.listdir(test_data_dir):
    file_path = os.path.join(test_data_dir, file_name)
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Kiểm tra định dạng hình ảnh
        img = load_img(file_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Chuẩn hóa dữ liệu về [0, 1]
        test_images.append(img_array)
        file_names.append(file_name)  # Lưu tên file để in kết quả sau này

# Chuyển đổi danh sách thành numpy array
if not test_images:
    raise ValueError("Không tìm thấy hình ảnh nào trong thư mục dữ liệu test.")

test_images = np.array(test_images)

# Dự đoán
predictions = best_model.predict(test_images, batch_size=batch_size)

# Lấy tên lớp từ thư mục huấn luyện
class_labels = sorted(os.listdir(train_data_dir))  # Lấy tên các thư mục con (bệnh)

# Xử lý dự đoán và in kết quả
for i, prediction in enumerate(predictions):
    predicted_class_index = np.argmax(prediction)  # Lấy chỉ số của xác suất cao nhất
    predicted_class_label = class_labels[predicted_class_index]  # Lấy tên lớp từ danh sách
    predicted_probability = prediction[predicted_class_index] * 100  # Chuyển đổi sang phần trăm

    # In kết quả với tên file
    print(f"Hình ảnh: {file_names[i]}, Dự đoán Bệnh: {predicted_class_label}, Xác suất: {predicted_probability:.2f}%")