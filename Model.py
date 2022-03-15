import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizer_v1 import Adam


# Configuration options
n_samples = 10000
n_features = 6
n_classes = 3
n_labels = 2
n_epochs = 50
random_state = 42
batch_size = 250
verbosity = 1
validation_split = 0.2

class Model:
    def __init__(self):
        print("Hellow")

    def train(self, json_file):
        dataset = pd.read_json(json_file)
        dataset = dataset.dropna()
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = [train_features.pop("d_r"), train_features.pop("d_b"), train_features.pop("d_g")]
        test_labels = [test_features.pop("d_r"), test_features.pop("d_b"), test_features.pop("d_g")]

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        print(normalizer.mean.numpy())

        # # Create the model
        # model = Sequential()
        # model.add(Dense(32, activation='relu', input_dim=n_features))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(8, activation='relu'))
        # model.add(Dense(n_classes, activation='sigmoid'))
        #
        # # Compile the model
        # model.compile(loss=binary_crossentropy,
        #               optimizer=Adam(),
        #               metrics=['accuracy'])
        #
        # # Fit data to model
        # model.fit(X_train, y_train,
        #           batch_size=batch_size,
        #           epochs=n_epochs,
        #           verbose=verbosity,
        #           validation_split=validation_split)
        #
        # # Generate generalization metrics
        # score = model.evaluate(X_test, y_test, verbose=0)
        # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
        df = dataframe.copy()
        labels = [df.pop("d_r"), df.pop("d_g"), df.pop("d_b")]
        df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    def get_normalization_layer(self, name, dataset):
        # Create a Normalization layer for the feature.
        normalizer = tf.keras.layers.Normalization(axis=None)

        # Prepare a Dataset that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer
