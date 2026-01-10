import glob
import tensorflow as tf
import numpy as np
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPooling1D, ReLU, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention 

learning_rate = 0.0001
filter_size = 8
kernel_size = 5
FC_size = 128
Dropout_rate = 0.5
patient_count = 24
file_paths = glob.glob("F:/CHB-MIT/*.csv")
datasets_path = tf.data.Dataset.list_files(file_paths, shuffle=False)

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
model.compile(optimizer=Adam(learning_rate=learning_rate),\
              loss="binary_crossentropy", metrics=['accuracy'])
ES = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

print("Files in the dataset:")
for filename_t in datasets_path:
    filename = filename_t.numpy().decode("utf-8")


def doubledigit(n):
    if n < 10:
        return f'0{n}'
    else:
        return str(n)


for x in range(patient_count):
    print(f'Ictal_{doubledigit(x)}.csv')
    print(f'Non-Ictal_{doubledigit(x)}.csv')

for i in range(patient_count):
    timestamp = time.time()
    normal = np.loadtxt(f"F:/CHB-MIT/Non-Ictal_{doubledigit(i)}.csv", delimiter=",")
    ictal = np.loadtxt(f"F:/CHB-MIT/Ictal_{doubledigit(i)}.csv", delimiter=",")
    print(f"Time used to read Non-Ictal and Ictal data for patient {i+1}: {time.time()-timestamp} seconds")
    print(normal.shape)
    print(ictal.shape)
    timestamp = time.time()
    whole = np.vstack([normal, ictal])
    label = np.array([0]*normal.shape[0]+[1]*ictal.shape[0])
    print(f"Time used to concat the dataframe:{time.time()-timestamp} seconds")
    timestamp = time.time()
    X_train, X_test, Y_train, Y_test = train_test_split(whole, label, test_size=0.2, random_state=42)
    print(f"Time used to split the data :{time.time()-timestamp}")
    timestamp = time.time()
    smote = SMOTE(random_state=42)
    X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
    print(f"Time used to oversample: {time.time()-timestamp} seconds")

    print(f"The Dimension of X_train_res: {X_train_res.shape}")
    print(f"The Dimension of Y_train_res: {Y_train_res.shape}")
    print(f"The Dimension of X_test: {X_test.shape}")
    print(f"The Dimension of Y_test: {Y_test.shape}")

    X_train_tf = tf.convert_to_tensor(X_train_res, dtype=tf.float32)
    Y_train_tf = tf.convert_to_tensor(Y_train_res, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    Y_test_tf = tf.convert_to_tensor(Y_test, dtype=tf.float32)

    X_train_3d = tf.reshape(X_train_tf, (X_train_tf.shape[0], X_train_tf.shape[1], 1))
    X_test_3d = tf.reshape(X_test_tf, (X_test_tf.shape[0], X_test_tf.shape[1], 1))

    timestamp = time.time()
    history = model.fit(X_train_3d, Y_train_tf, epochs=100, batch_size=16, \
                        validation_split=0.2, callbacks=[ES], verbose=1)

    test_loss, test_accuracy = model.evaluate(X_test_3d, Y_test_tf, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Time used to train and infer:{time.time() - timestamp} seconds")
    del normal, ictal, whole, label
    del X_train, Y_train, X_test, Y_test, X_train_res, Y_train_res
    del X_train_tf, Y_train_tf, X_test_tf, Y_test_tf
    del X_train_3d, X_test_3d

model.save("attention_model.keras")
