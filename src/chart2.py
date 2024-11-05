import matplotlib.pyplot as plt
import json

# Đọc dữ liệu từ file history.json
with open('history.json', 'r') as file:
    history = json.load(file)

# Vẽ biểu đồ độ chính xác và mất mát
plt.figure(figsize=(16, 6))

# Độ chính xác
plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Độ chính xác huấn luyện')
plt.plot(history['val_accuracy'], label='Độ chính xác kiểm tra')
plt.title('Độ chính xác qua các epochs')
plt.xlabel('Epochs')
plt.ylabel('Độ chính xác')
plt.legend()

# Mất mát
plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Mất mát huấn luyện')
plt.plot(history['val_loss'], label='Mất mát kiểm tra')
plt.title('Mất mát qua các epochs')
plt.xlabel('Epochs')
plt.ylabel('Mất mát')
plt.legend()

# Biểu đồ thêm (ví dụ: độ chính xác từng lớp)
if 'class_accuracy' in history:  # Giả định bạn có dữ liệu này
    plt.subplot(1, 3, 3)
    for class_name, acc in history['class_accuracy'].items():
        plt.plot(acc, label=class_name)
    plt.title('Độ chính xác từng lớp')
    plt.xlabel('Epochs')
    plt.ylabel('Độ chính xác')
    plt.legend()

plt.tight_layout()
plt.show()