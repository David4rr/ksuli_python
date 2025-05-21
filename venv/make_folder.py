import os
import numpy as np

DATA_PATH = os.path.join(r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo', 'dataset')
actions = np.array(['halo', 'selamatDatang', 'mau', 'pesan', 'bayar', 'kembalian', 'lagi', 'terimakasih', 'apa', 'berapa'])
no_sequences = 150
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

print("Folder untuk aksi baru telah dibuat!")
