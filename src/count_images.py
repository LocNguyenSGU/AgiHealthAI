import os

# Đường dẫn tới thư mục chứa dữ liệu (train, valid, test)
data_dir = '../small_data'  # Thay thế bằng đường dẫn của bạn

# Hàm đếm số lượng ảnh trong mỗi lớp bệnh cho train, valid
def count_images_in_folder(folder_path):
    class_counts = {}  # Tạo một dictionary để lưu số lượng ảnh cho mỗi bệnh
    for disease_class in os.listdir(folder_path):
        disease_folder = os.path.join(folder_path, disease_class)
        if os.path.isdir(disease_folder):
            # Đếm số lượng ảnh trong từng thư mục con (tương ứng với mỗi bệnh)
            image_count = len(os.listdir(disease_folder))
            class_counts[disease_class] = image_count
    return class_counts

# Hàm đếm tổng số lượng ảnh trong thư mục test (không phân loại)
def count_images_in_test_folder(folder_path):
    # Đếm tổng số ảnh trong thư mục test
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Đếm số lượng ảnh trong từng thư mục train, valid
train_counts = count_images_in_folder(os.path.join(data_dir, 'small_train'))
valid_counts = count_images_in_folder(os.path.join(data_dir, 'small_valid'))

# Đếm tổng số lượng ảnh trong thư mục test
test_count = count_images_in_test_folder(os.path.join(data_dir, 'small_test'))

# In ra kết quả
print("Số lượng ảnh trong thư mục train:")
for disease, count in train_counts.items():
    print(f"{disease}: {count} ảnh")

print("\nSố lượng ảnh trong thư mục valid:")
for disease, count in valid_counts.items():
    print(f"{disease}: {count} ảnh")

print(f"\nTổng số lượng ảnh trong thư mục test: {test_count} ảnh")