import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Load data ---
train_X = np.load("Data Set/Digits/Processed Training Images.npy")
train_y = np.load("Data Set/Digits/Processed Training Labels.npy")
test_X  = np.load("Data Set/Digits/Processed Test Images.npy")
test_y  = np.load("Data Set/Digits/Processed Test Labels.npy")

# --- Preprocess ---
train_X = train_X.reshape(-1, 784).astype("float32") / 255.0
test_X  = test_X.reshape(-1, 784).astype("float32") / 255.0

# --- Build model ---
model = keras.Sequential([
    keras.layers.Dense(200, activation="tanh", input_shape=(784,)),
    keras.layers.Dense(100, activation="tanh"),
    keras.layers.Dense(10,  activation="softmax"),
])

# --- Compile ---
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.05),
    loss="sparse_categorical_crossentropy",   # labels are integers, not one-hot
    metrics=["accuracy"],
)

# --- Train ---
model.fit(train_X, train_y, epochs=20, batch_size=100, validation_split=0.1)

# --- Evaluate ---
loss, acc = model.evaluate(test_X, test_y)
print(f"Test accuracy: {acc * 100:.2f}%")