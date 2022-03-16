import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import models
from matplotlib import pyplot as plt

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
    def __init__(self, json_file):
        dataset = pd.read_json(json_file)
        dataset = dataset.dropna()
        self.train_dataset = dataset.sample(frac=0.8, random_state=0)
        self.test_dataset = dataset.drop(self.train_dataset.index)

        self.train_features = self.train_dataset.copy()
        self.test_features = self.test_dataset.copy()

        self.train_labels = [
            self.train_features.pop("d_r"),
            self.train_features.pop("d_b"),
            self.train_features.pop("d_g")
        ]
        self.test_labels = [
            self.test_features.pop("d_r"),
            self.test_features.pop("d_b"),
            self.test_features.pop("d_g")
        ]

        self.test_results = {}
        self.dnn_model_multiple = None


    def train_dnn_multiple(self):
        normalizer = layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_features))
        self.dnn_model_multiple = self.build_and_compile_model(normalizer)
        self.dnn_model_multiple.summary()

        history = self.dnn_model_multiple.fit(
            self.train_features,
            self.train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        self.plot_loss(history)

        self.test_results['dnn_model_multiple'] = self.dnn_model_multiple.evaluate(
            self.test_features, self.test_labels,
            verbose=0)

    def train_dnn_single(self):
        normalizer = layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_features))
        dnn_model = self.build_and_compile_model(normalizer)
        dnn_model.summary()

        history = dnn_model.fit(
            self.train_features['325.0'],
            self.train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        self.plot_loss(history)

        self.test_results['dnn_model_single'] = dnn_model.evaluate(
            self.test_features['325.0'], self.test_labels,
            verbose=0)

    def build_and_compile_model(self, norm):
        model = models.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def train_multiple_input(self):
        normalizer = layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_features))

        linear_model = models.Sequential([
            normalizer,
            layers.Dense(units=1)
        ])

        print(linear_model.predict(self.train_features[:10]))
        print(linear_model.layers[1].kernel)

        linear_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        history = linear_model.fit(
            self.train_features,
            self.train_labels,
            epochs=100,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2)

        self.plot_loss(history)

        self.test_results['multiple_input'] = linear_model.evaluate(
            self.test_features, self.test_labels, verbose=0)

    def train_single_input(self):
        low_freq = np.array(self.train_features["325.0"])

        low_freq_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
        low_freq_normalizer.adapt(low_freq)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_features))

        low_freq_model = tf.keras.Sequential([
            low_freq_normalizer,
            layers.Dense(units=1)
        ])

        low_freq_model.summary()

        print(low_freq_model.predict(low_freq[:10]))

        low_freq_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss="mean_absolute_error")

        history = low_freq_model.fit(
            self.train_features["325.0"],
            self.train_labels,
            epochs=100,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2)

        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch
        print(hist.tail())

        self.plot_loss(history)

        self.test_results["single_input"] = low_freq_model.evaluate(
            self.test_features["325.0"],
            self.test_labels, verbose=0)

        x = tf.linspace(0.0, 250, 251)
        y = low_freq_model.predict(x).reshape(-1)

        self.plot_scatter(x, y, self.train_features, self.train_labels)

    def plot_loss(self, history):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.ylim([0, 100])
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_scatter(self, x, y, train_features, train_labels):
        plt.scatter(train_features['325.0'], train_labels[0], label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('325.0')
        plt.ylabel('RGB')
        plt.legend()
        plt.show()

    def print_performance(self):
        print(pd.DataFrame(self.test_results, index=['Mean absolute error [MPG]']).T)

    def predict(self):
        test_predictions = self.dnn_model_multiple.predict(self.test_features).flatten()

        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels[0], test_predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [0, 50]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()

        error = test_predictions - self.test_labels
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error')
        _ = plt.ylabel('Count')
