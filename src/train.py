import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Tắt cảnh báo
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Bước 1: Xây dựng mô hình
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile mô hình
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Dùng binary_crossentropy nếu có 2 lớp

# In ra tóm tắt mô hình
model.summary()

# Bước 2: Chuẩn bị dữ liệu
data_dir = '../small_data'  # Đường dẫn tới thư mục chứa dữ liệu

# Khởi tạo ImageDataGenerator cho dữ liệu huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Khởi tạo ImageDataGenerator cho dữ liệu kiểm tra
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Tạo generator cho dữ liệu huấn luyện
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'small_train'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # Sử dụng binary nếu chỉ có 2 lớp
)

# Tạo generator cho dữ liệu kiểm tra
validation_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'small_valid'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # Sử dụng binary nếu chỉ có 2 lớp
)

# Lưu mô hình tốt nhất
checkpoint = ModelCheckpoint('model_best.keras', monitor='val_accuracy', save_best_only=True)

# Thêm EarlyStopping nếu muốn dừng sớm
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Bước 3: Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping]  # Thêm callback vào đây
)

# Bước 4: Đánh giá mô hình
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Model accuracy: {test_accuracy * 100:.2f}%')



# Bước 5: Vẽ biểu đồ độ chính xác và tổn thất
import matplotlib.pyplot as plt

# Vẽ biểu đồ độ chính xác
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Vẽ biểu đồ tổn thất
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()