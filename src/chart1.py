import matplotlib.pyplot as plt
import json

# Đọc dữ liệu từ file history.json
with open('history.json', 'r') as file:
    history = json.load(file)

# Vẽ biểu đồ độ chính xác
plt.figure(figsize=(12, 4))

# Độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Độ chính xác huấn luyện')
plt.plot(history['val_accuracy'], label='Độ chính xác kiểm tra')
plt.title('Độ chính xác qua các epochs')
plt.xlabel('Epochs')
plt.ylabel('Độ chính xác')
plt.legend()

# Mất mát
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Mất mát huấn luyện')
plt.plot(history['val_loss'], label='Mất mát kiểm tra')
plt.title('Mất mát qua các epochs')
plt.xlabel('Epochs')
plt.ylabel('Mất mát')
plt.legend()

plt.tight_layout()
plt.show()