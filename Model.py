import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import models
from matplotlib import pyplot as plt


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
        dataset = pd.read_json(json_file)
        dataset = dataset.dropna()
        self.train_dataset = dataset.sample(frac=1-validation_split, random_state=1)
        self.test_dataset = dataset.drop(self.train_dataset.index)

        self.train_features = self.train_dataset.copy()
        self.test_features = self.test_dataset.copy()

        self.train_labels = np.array([
            self.train_features.pop("d_r"),
            self.train_features.pop("d_b"),
            self.train_features.pop("d_g")
        ]).T
        self.test_labels = np.array([
            self.test_features.pop("d_r"),
            self.test_features.pop("d_b"),
            self.test_features.pop("d_g")
        ]).T

        self.validation_split = validation_split

        normalizer = layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_features))

        self.model = models.Sequential([
            # normalizer,
            layers.InputLayer(self.train_features.shape[1]),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(3)
        ])
        self.model.summary()
        # self.model.compile(loss='MeanSquaredLogarithmicError',
        #                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
        self.model.compile(loss='MeanSquaredError',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        # self.model.compile(loss='MeanSquaredError',
        #                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001))
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
        test_predictions = self.model.predict(self.test_features).flatten()
        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels, test_predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [0, 50]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()

        error = test_predictions - self.test_labels.flatten()
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error')
        _ = plt.ylabel('Count')
        plt.show()
