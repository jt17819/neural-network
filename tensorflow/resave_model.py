import tensorflow as tf

model = tf.keras.models.load_model("digits_model-99.keras")
model.save("digits_model-99.h5")
print("Saved as H5 successfully.")
