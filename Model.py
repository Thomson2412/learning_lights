import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import models
from keras import backend
from matplotlib import pyplot as plt


def rgb_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    loss = backend.abs(y_true - y_pred)
    return loss


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, max(history.history["val_loss"])])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


class Model:
    def __init__(self, json_file, validation_split):
        self.validation_split = validation_split

        dataset = pd.read_json(json_file)
        dataset = dataset.dropna()

        train_split = dataset.sample(frac=1 - validation_split, random_state=1)
        test_split = dataset.drop(train_split.index)

        self.train_features = train_split.copy()
        self.test_features = test_split.copy()

        self.train_labels = tf.transpose(
            tf.constant([
                self.train_features.pop("d_r"),
                self.train_features.pop("d_b"),
                self.train_features.pop("d_g")
            ], dtype=tf.float32)
        )
        self.test_labels = tf.transpose(
            tf.constant([
                self.test_features.pop("d_r"),
                self.test_features.pop("d_b"),
                self.test_features.pop("d_g")
            ], dtype=tf.float32)
        )

        # self.train_labels = tf.constant(self.train_features.pop("d_rgb"), dtype=tf.float32)
        # self.test_labels = tf.constant(self.test_features.pop("d_rgb"), dtype=tf.float32)

        # self.train_dataset_features = tf.data.Dataset.from_tensor_slices(train_features).batch(batch)
        # self.train_dataset_labels = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch)
        # self.test_dataset_features = tf.data.Dataset.from_tensor_slices(test_features).batch(batch)
        # self.test_dataset_labels = tf.data.Dataset.from_tensor_slices(test_labels).batch(batch)

        normalizer = layers.Normalization(axis=-1)
        normalizer.adapt(self.train_features)

        self.model = models.Sequential([
            normalizer,
            layers.InputLayer(self.train_features.shape[1]),
            layers.Dense(256, activation='relu',),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(3)
        ])
        self.model.summary()

        # self.model.compile(loss=rgb_loss,
        #                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        # self.model.compile(loss='MeanSquaredLogarithmicError',
        #                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
        # self.model.compile(loss="MeanSquaredError",
        #                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        self.model.compile(loss='MeanSquaredError',
                           optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
        # self.model.compile(loss='MeanAbsoluteError',
        #                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        # self.model.compile(loss='MeanAbsoluteError',
        #                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001))

    def train_model(self, epochs, steps_per_epoch):
        history = self.model.fit(
            self.train_features,
            self.train_labels,
            validation_split=self.validation_split,
            verbose=1,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )

        plot_loss(history)
        print(max(history.history["loss"]))
        print(min(history.history["loss"]))
        print(max(history.history["val_loss"]))
        print(min(history.history["val_loss"]))

    def predict(self):
        test_predictions = self.model.predict(self.train_features).flatten()
        a = plt.axes(aspect='equal')
        plt.scatter(self.train_labels, tf.reshape(test_predictions, self.train_labels.shape))
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [0, 50]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()

        error = tf.reshape(test_predictions, self.train_labels.shape) - self.test_labels
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error')
        _ = plt.ylabel('Count')
        plt.show()
