import tensorflow as tf

model_path = r"C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\output\bilstm_model_150(3).h5"

model = tf.keras.models.load_model(model_path)
model.summary()