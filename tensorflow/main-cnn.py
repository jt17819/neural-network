import numpy as np
import tensorflow as tf
from tensorflow import keras

train_X = np.load("Data Set/Digits/Processed Training Images.npy")
train_y = np.load("Data Set/Digits/Processed Training Labels.npy")
test_X  = np.load("Data Set/Digits/Processed Test Images.npy")
test_y  = np.load("Data Set/Digits/Processed Test Labels.npy")

# Reshape to (N, 28, 28, 1) — TensorFlow uses channels-last
train_X = train_X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_X  = test_X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same", input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    # keras.layers.Conv2D(20, kernel_size=5, activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Dropout(0.5),

    
    keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Dropout(0.25),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_X, train_y, epochs=10, batch_size=64, validation_split=0.1)

loss, acc = model.evaluate(test_X, test_y)
print(f"Test accuracy: {acc * 100:.2f}%")

# Save the model
model.save("tensorflow/digits_model.keras")