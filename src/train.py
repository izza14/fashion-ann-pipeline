import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load hyperparameters from params.yaml
with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

train_params = params['train']

# Load processed training and validation arrays
x_train = np.load(os.path.join('data', 'processed', 'x_train.npy'))
y_train = np.load(os.path.join('data', 'processed', 'y_train.npy'))
x_val = np.load(os.path.join('data', 'processed', 'x_val.npy'))
y_val = np.load(os.path.join('data', 'processed', 'y_val.npy'))

# Build the Sequential ANN
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(train_params['dense_units'], activation='relu'),
    Dropout(train_params['dropout_rate']),
    Dense(10, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=train_params['learning_rate'])
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=train_params['epochs'],
    batch_size=train_params['batch_size']
)

# Ensure output directory exists
os.makedirs('models', exist_ok=True)

# Save the trained model and training history
model.save(os.path.join('models', 'model.h5'))

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join('models', 'history.csv'), index=False)

print("Model and history successfully saved to models/")