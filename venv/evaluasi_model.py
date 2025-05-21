import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Load the preprocessed test data
# data = np.load(r'C:\Users\david\Documents\my private document\Skripsi Project\testData\preprocessed_test_data(1).npz')
data = np.load(r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\preprocessed_1000_data.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load the trained model
model_path = r"C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\output\bilstm_model_100(2).h5"
model = tf.keras.models.load_model(model_path)

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Create labels for the plot
num_classes = y_test.shape[1]
labels = [str(i) for i in range(num_classes)]

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels,
            yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\output\confusion_matrix_new_test_bilstm100(2).png')
plt.close()

# Generate and save classification report
report = classification_report(y_test_classes, y_pred_classes, 
                            target_names=labels,
                            output_dict=True)

# Convert to DataFrame for better formatting
df_report = pd.DataFrame(report).transpose()

# Save classification report
report_path = r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\output\classification_report_new_test_bilstm100(2).csv'
df_report.to_csv(report_path)

# Print the results
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=labels))

# Calculate and print overall accuracy
accuracy = (y_pred_classes == y_test_classes).mean()
print(f"\nOverall Accuracy: {accuracy:.4f}")  