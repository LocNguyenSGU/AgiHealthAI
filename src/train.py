import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import matplotlib.pyplot as plt

img_height, img_width = 64, 64
batch_size = 32

# Chuẩn bị dữ liệu train và validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    "../small_data/small_train",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_data = train_datagen.flow_from_directory(
    "../small_data/small_valid",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Định nghĩa mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # Output layer với softmax cho phân loại đa lớp
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Lưu mô hình tốt nhất ở định dạng .keras
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_accuracy', mode='max')
epochs = 30 # Số lần huấn luyện mô hình

# Huấn luyện mô hình
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[checkpoint]
)


# Lưu lịch sử vào file JSON
history_dict = history.history  # Lấy lịch sử huấn luyện
with open('history.json', 'w') as f:
    json.dump(history_dict, f)

# # Chuẩn bị dữ liệu test
# test_data = train_datagen.flow_from_directory(
#     "../small_data/small_test",
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# # Đánh giá mô hình trên tập test
# best_model = tf.keras.models.load_model("best_model.keras")
# test_loss, test_accuracy = best_model.evaluate(test_data)
# print(f"Test Accuracy: {test_accuracy:.2f}")

# Hàm vẽ biểu đồ từ file JSON
def plot_history(history_file):
    with open(history_file, 'r') as f:
        history_dict = json.load(f)

    plt.figure(figsize=(12, 4))

    # Độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Độ chính xác huấn luyện')
    plt.plot(history_dict['val_accuracy'], label='Độ chính xác kiểm tra')
    plt.title('Độ chính xác qua các epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Độ chính xác')
    plt.legend()

    # Mất mát
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Mất mát huấn luyện')
    plt.plot(history_dict['val_loss'], label='Mất mát kiểm tra')
    plt.title('Mất mát qua các epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mất mát')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Gọi hàm để vẽ biểu đồ từ file JSON
plot_history('history.json')