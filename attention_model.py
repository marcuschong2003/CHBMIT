import numpy as np
import time
import tensorflow as tf
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPooling1D, Dense, Dropout, ReLU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras
from keras_self_attention import SeqSelfAttention
learning_rate = 0.0001
filter_size = 8
kernel_size = 5
FC_size = 128
Dropout_rate = 0.5

start = time.time()
ictal = np.loadtxt("Ictal.csv", delimiter=",")
nonictal = np.loadtxt("Non-Ictal.csv", delimiter=",")
print(ictal.shape)
print(nonictal.shape)
end = time.time()
print(f"Time used to load data: {end - start}")

X_df = np.vstack((nonictal, ictal))
Y_df = np.array([0]*nonictal.shape[0] + [1]*ictal.shape[0])
start = time.time()
print(f"Time to stack the matrix : {start - end}")

smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X_df, Y_df)
print('Resampled dataset shape: %s' % Counter(Y_res))
end = time.time()
print(f"Time used to oversample:{end - start}")

X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=42)
print(f"The dimension of X_train: {X_train.shape}")
print(f"The dimension of X_test: {X_test.shape}")
print(f"The dimension of Y_train: {Y_train.shape}")
print(f"The dimension of Y_test: {Y_test.shape}")

X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_test_tf = tf.convert_to_tensor(Y_test, dtype=tf.float32)

X_train_3d = tf.reshape(X_train_tf, (X_train_tf.shape[0], X_train_tf.shape[1], 1))
X_test_3d = tf.reshape(X_test_tf, (X_test_tf.shape[0], X_test_tf.shape[1], 1))

model = Sequential()
model.add(BatchNormalization())
model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(ReLU())
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Conv1D(filters=filter_size*2, kernel_size=kernel_size, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(ReLU())
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Conv1D(filters=filter_size*2, kernel_size=kernel_size, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(ReLU())
model.add(Dropout(0.4))
model.add(SeqSelfAttention(attention_activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics = ['accuracy'])
ES = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = model.fit(X_train_3d, Y_train, epochs=20, batch_size=16, validation_split=0.2, callbacks=[ES], verbose=1)
test_loss, test_accuracy = model.evaluate(X_test_3d, Y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
