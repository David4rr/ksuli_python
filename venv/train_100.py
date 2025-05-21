import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from make_folder import actions

# Load data yang sudah di preprocess
data = np.load(r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\preprocessed_1000_data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Callback EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Model LSTM
# model = Sequential([
#     Input(shape=(30, 1662)),
#     LSTM(64, return_sequences=True, activation='relu'),
#     Dropout(0.2), 
#     LSTM(128, return_sequences=True, activation='relu'),
#     Dropout(0.3),
#     LSTM(64, return_sequences=False, activation='relu'),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dense(len(actions), activation='softmax')
# ])

model = tf.keras.models.Sequential([
    # tf.keras.layers.Input(shape=(30, 1662)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 1662))),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(actions.shape[0], activation='softmax')
])

# Compile Model
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Training Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, lr_reduction],
    verbose=1
)

# Menampilkan Ringkasan Model dan menyimpanya
model.summary()

# Path penyimpanan model
model_path = r"C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\output\bilstm_model_100(2).h5"

# Simpan model setelah training
model.save(model_path)
print(f"Model berhasil disimpan di {model_path}")

# Plot hasil training
train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Akurasi
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

history_df = pd.DataFrame(history.history)
history_df['epoch'] = history_df.index + 1
history_csv_path = r"C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\model\training_history_100(2).csv"
history_df.to_csv(history_csv_path, index=False)
print(f"📊 History training disimpan di {history_csv_path}")

# # === Model Evaluation ===
# eval_results = model.evaluate(X_test, y_test, verbose=1)
# eval_path = r"C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\model\model_evaluation.txt"
# with open(eval_path, 'w') as f:
#     f.write(f"Evaluation Results on Test Data:\n")
#     for name, value in zip(model.metrics_names, eval_results):
#         f.write(f"{name}: {value:.4f}\n")
# print(f"📄 Hasil evaluasi model disimpan di {eval_path}")

# === Info Data Save ===
info_path = r"C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\model\data_info_150(2).txt"
with open(info_path, 'w') as f:
    f.write(f"Jumlah data training : {X_train.shape[0]}\n")
    f.write(f"Jumlah data testing  : {X_test.shape[0]}\n")
    f.write(f"Shape X_train: {X_train.shape}\n")
    f.write(f"Shape X_test : {X_test.shape}\n")
    f.write(f"Shape y_train: {y_train.shape}\n")
    f.write(f"Shape y_test : {y_test.shape}\n")
print(f"📄 Info data disimpan di {info_path}")


