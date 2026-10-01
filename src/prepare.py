import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Ensure the target directory exists
os.makedirs(os.path.join('data', 'raw'), exist_ok=True)

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Save the raw arrays to data/raw/
np.save(os.path.join('data', 'raw', 'x_train.npy'), x_train)
np.save(os.path.join('data', 'raw', 'y_train.npy'), y_train)
np.save(os.path.join('data', 'raw', 'x_test.npy'), x_test)
np.save(os.path.join('data', 'raw', 'y_test.npy'), y_test)

print("Fashion-MNIST raw data successfully saved to data/raw/")