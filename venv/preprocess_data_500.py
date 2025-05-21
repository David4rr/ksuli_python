from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from make_folder import DATA_PATH, actions, no_sequences, sequence_length

# Mapping label ke angka
label_map = {label: num for num, label in enumerate(actions)}
print(f"Label Mapping: {label_map}")

# --- Step 1: Mengumpulkan data hanya 50 per kategori ---
sequences, labels = [], []
print("\n📥 Mengumpulkan 50 data per kategori...")

for action in actions:
    for sequence in range(50):  # Ambil hanya 50 sequence per kategori
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print("\n✅ Data terkumpul!")
print(f"Total sequences: {len(sequences)}")  # Harusnya 500 total
print(f"Total labels: {len(labels)}")

# --- Step 2: Konversi ke array numpy ---
X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels).astype(int)


print("\n📊 Bentuk data sebelum split:")
print(f"X shape: {X.shape}")  # (jumlah_data, sequence_length, fitur)
print(f"y shape: {y.shape}")  # (jumlah_data, jumlah_kelas)
print(f"Contoh X[0] mean: {np.mean(X[0]):.4f}, std: {np.std(X[0]):.4f}")
print(f"Contoh y[0]: {y[0]}")

# --- Step 3: Split manual tanpa shuffle 40:10 per kategori ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n📂 Data setelah split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n🔍 Cek distribusi label di training set:")
print(pd.Series(np.argmax(y_train, axis=1)).value_counts().sort_index())

print("\n🔍 Cek distribusi label di testing set:")
print(pd.Series(np.argmax(y_test, axis=1)).value_counts().sort_index())

print("\n📑 Contoh X_train[0] DataFrame:")
sample_df = pd.DataFrame(X_train[0])
print(sample_df.head())

# --- Step 5: Save hasil preprocess ---
np.savez_compressed(
    r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\preprocessed_500_data.npz',
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

print("\n✅ Preprocessing selesai & disimpan ke NPZ (500 data, 400 train, 100 test)!")
