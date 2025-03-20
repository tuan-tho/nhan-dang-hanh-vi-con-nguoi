import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import các module từ tf.keras thay vì keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split

# Kiểm tra GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"🔹 Using GPU: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("⚠️ GPU not found, using CPU instead.")

# Danh sách các lớp hành động
classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
num_timesteps = 7
num_classes = len(classes)

X, y = [], []
label = 0

# Load dataset từ thư mục
for cl in classes:
    folder_path = f'./dataset/{cl}'
    if not os.path.exists(folder_path):  # Kiểm tra nếu thư mục tồn tại
        print(f"⚠️ Warning: Folder '{folder_path}' not found!")
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        print(f'📂 Reading: {file_path}')

        try:
            data = pd.read_csv(file_path).iloc[:, 1:].values  # Bỏ cột đầu tiên nếu không cần
            if data.shape[0] < num_timesteps:
                print(f"⚠️ Skipping {file}: Not enough data points.")
                continue

            n_sample = len(data)
            for i in range(num_timesteps, n_sample):
                X.append(data[i - num_timesteps: i, :])
                y.append(label)

        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

    label += 1

print("✅ Dataset loaded successfully.")

X, y = np.array(X), np.array(y)
print("📊 Data Shape:", X.shape, y.shape)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Xây dựng mô hình LSTM
model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(units=128, return_sequences=True),
    Dropout(0.2),
    LSTM(units=128),  # Không có return_sequences ở layer cuối cùng
    Dropout(0.2),
    Dense(units=num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

# Lưu mô hình dưới dạng hình ảnh
try:
    plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
    print("📸 Model architecture saved as 'model_plot.png'")
except Exception as e:
    print(f"⚠️ Error saving model plot: {e}. Try installing 'graphviz' and 'pydot'.")

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình ở định dạng `.keras` (tốt hơn `.h5`)
os.makedirs("model", exist_ok=True)
model.save(f"model/model_{num_timesteps}.keras")
print("✅ Model saved successfully.")


# Hàm vẽ biểu đồ loss và accuracy
def visualize_history(history, title="Training Performance"):
    epochs = range(len(history.history["loss"]))
    plt.figure(figsize=(12, 5))

    # Vẽ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], "b", label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "r", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # Vẽ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], "b", label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "r", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")

    plt.suptitle(title)
    plt.show()


visualize_history(history)
