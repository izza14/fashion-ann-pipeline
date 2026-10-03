import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load processed test arrays
x_test = np.load(os.path.join('data', 'processed', 'x_test.npy'))
y_test = np.load(os.path.join('data', 'processed', 'y_test.npy'))

# Load the trained model
model = tf.keras.models.load_model(os.path.join('models', 'model.h5'))

# Compute test loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# Generate predictions for the confusion matrix
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create and save the confusion matrix image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.close()

# Write metrics to metrics.json at the project root
metrics = {
    'loss': float(loss),
    'accuracy': float(accuracy)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
print("Saved metrics.json and confusion_matrix.png")