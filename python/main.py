import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
import tf2onnx
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Đường dẫn đến thư mục dữ liệu Fruits 360
train_dir = "fruits_dataset"
val_dir = "fruits_dataset"

# Tạo Data Generator cho dữ liệu huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Tải dữ liệu huấn luyện và kiểm tra từ thư mục Fruits 360
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Tải MobileNetV2 làm backbone và thêm các lớp cho SSD
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Xây dựng mô hình MobileNet-SSD đơn giản
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(train_data.num_classes, activation="softmax")
])

# Biên dịch mô hình
model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(train_data, epochs=5, validation_data=val_data)

# Xuất mô hình sang định dạng ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "fruit_detector.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print("Saved ONNX model to:", output_path)
