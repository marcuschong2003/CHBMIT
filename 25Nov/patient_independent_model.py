import glob
import tensorflow as tf
import numpy as np
import time
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPooling1D, ReLU, Dropout, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

learning_rate = 0.0001
filter_size = 32
kernel_size = 5
FC_size = 128
Dropout_rate = 0.5
patient_count = 24
file_paths = glob.glob("F:/CHB-MIT/*.csv")
datasets_path = tf.data.Dataset.list_files(file_paths, shuffle=False)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

model = Sequential()
model.add(Conv1D(filters=filter_size, kernel_size=3, activation="relu"))
model.add(BatchNormalization(axis=2, momentum=0.9))
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2, strides=2))
#model.add(Dropout(0.4))
model.add(Conv1D(filters=filter_size*2, kernel_size=5, activation="relu"))
model.add(BatchNormalization(axis=2, momentum=0.9))
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2, strides=4))
#model.add(Dropout(0.4))
model.add(Conv1D(filters=filter_size*4, kernel_size=13, activation="relu"))
model.add(BatchNormalization(axis=2, momentum=0.9))
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2, strides=4))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.4))
#model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=Adam(learning_rate=learning_rate),\
              loss="binary_crossentropy", metrics=['accuracy',tf.keras.metrics.Recall(name="Recall"), tf.keras.metrics.SpecificityAtSensitivity(0.9, name='Spec_at_90_recall'), tf.keras.metrics.AUC(name="AUC")])
ES = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

print("Files in the dataset:")
for filename_t in datasets_path:
    filename = filename_t.numpy().decode("utf-8")


def doubledigit(n):
    if n < 10:
        return f'0{n}'
    else:
        return str(n)

rehearsal_X = np.empty((0, 2560))
rehearsal_Y = np.empty((0))

for x in range(patient_count):
    print(f'Ictal_{doubledigit(x)}.csv')
    print(f'Non-Ictal_{doubledigit(x)}.csv')

for i in range(patient_count):
    if i!= 0:
        Baseline = tf.keras.models.load_model(f"F:/Models/Incremental_{doubledigit(i)}")
        Rehearsal = tf.kears.models.load_model(f"F:/Models/Rehearsal_{doubledigit(i)}")
    else:
        Baseline = model
        Rehearsal = Baseline
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
    rehearsal_index = np.random.randint(len(Y_train), size=10000)
    rehearsal_data = X_train[rehearsal_index, :]
    rehearsal_label = Y_train[rehearsal_index]
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
    #Y_train_tf = tf.cast(Y_train_tf, tf.int64)
    #Y_test_tf = tf.cast(Y_test_tf, tf.int64)
    #Y_train_tf = tf.one_hot(Y_train_tf, depth=2, dtype=tf.int64)
    #Y_test_tf = tf.one_hot(Y_test_tf, depth=2, dtype=tf.int64)

    print(X_train_tf.dtype)
    print(Y_train_tf.dtype)
    print(X_test_tf.dtype)
    print(Y_test_tf.dtype)
    X_train_3d = tf.reshape(X_train_tf, (X_train_tf.shape[0], X_train_tf.shape[1], 1))
    X_test_3d = tf.reshape(X_test_tf, (X_test_tf.shape[0], X_test_tf.shape[1], 1))
    print(X_train_3d.dtype)
    print(X_test_3d.dtype)

    timestamp = time.time()
    history = Baseline.fit(X_train_3d, Y_train_tf, epochs=100, batch_size=16, \
                        validation_split=0.2, callbacks=[ES], verbose=1, class_weight="balanced")

    Metrics = Baseline.evaluate(X_test_3d, Y_test_tf, verbose=0)
    print("==========Incremental Performance==========")
    print(f"Test Loss: {Metrics[0]:.4f}")
    print(f"Test Accuracy: {Metrics[1]:.4f}")
    print(f"Test Recall: {Metrics[2]:.4f}")
    print(f"Test Spec_at_90_recall: {Metrics[3]:.4f}")
    print(f"Test AUC: {Metrics[4]:.4f}")
    print(f"Time used to train and infer:{time.time() - timestamp} seconds")
    Baseline.save(f"F:/Models/Incremental_{doubledigit(i)}")
    if i == 0:
        Baseline.save(f"F:/Models/Rehearsal_{doubledigit(i)}")
    if i != 0:
        print(f'Shape of rehearal X: {rehearsal_X.shape}')
        print(f'Shape of rehearal Y: {rehearsal_Y.shape}')
        Rehearsal.fit(tf.reshape(rehearsal_X, (rehearsal_X.shape[0], rehearsal_X.shape[1], 1)), rehearsal_Y, class_weight="balanced")
        Metrics = Rehearsal.evaluate(X_test_3d, Y_test_tf, verbose=0)
        print("==========Rehearsal Performance==========")
        print(f"Test Loss: {Metrics[0]:.4f}")
        print(f"Test Accuracy: {Metrics[1]:.4f}")
        print(f"Test Recall: {Metrics[2]:.4f}")
        print(f"Test Spec_at_90_recall: {Metrics[3]:.4f}")
        print(f"Test AUC: {Metrics[4]:.4f}")
        print(f"Time used to train and infer:{time.time() - timestamp} seconds")
        Rehearsal.save(f"F:/Models/Rehearsal_{doubledigit(i)}")
    rehearsal_X = np.concatenate((rehearsal_X, rehearsal_data), axis=0)
    rehearsal_Y = np.concatenate((rehearsal_Y, rehearsal_label), axis=0)
    del normal, ictal, whole, label
    del X_train, Y_train, X_test, Y_test, X_train_res, Y_train_res
    del X_train_tf, Y_train_tf, X_test_tf, Y_test_tf
    del X_train_3d, X_test_3d