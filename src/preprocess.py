import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# Load hyperparameters from params.yaml
with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

test_size = params['preprocess']['test_size']
seed = params['preprocess']['seed']

# Load raw arrays
x_train = np.load(os.path.join('data', 'raw', 'x_train.npy'))
y_train = np.load(os.path.join('data', 'raw', 'y_train.npy'))
x_test = np.load(os.path.join('data', 'raw', 'x_test.npy'))
y_test = np.load(os.path.join('data', 'raw', 'y_test.npy'))

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Split validation set
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=test_size, random_state=seed
)

# Ensure output directory exists
os.makedirs(os.path.join('data', 'processed'), exist_ok=True)

# Save processed arrays
np.save(os.path.join('data', 'processed', 'x_train.npy'), x_train)
np.save(os.path.join('data', 'processed', 'y_train.npy'), y_train)
np.save(os.path.join('data', 'processed', 'x_val.npy'), x_val)
np.save(os.path.join('data', 'processed', 'y_val.npy'), y_val)
np.save(os.path.join('data', 'processed', 'x_test.npy'), x_test)
np.save(os.path.join('data', 'processed', 'y_test.npy'), y_test)

print("Data successfully preprocessed and saved to data/processed/")