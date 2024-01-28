from abc import ABC, abstractmethod
import pandas as pd
import os

from keras.src.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report

import sklearn


class AbstractBaseModelFactory(ABC):

    @abstractmethod
    def create_model(self):
        pass


class AbstractModel(ABC):
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.path = None

    def load_train_test_data(self, train, test):
        self.path = os.getcwd()
        self.train_x = pd.read_csv(
            os.path.abspath(os.path.join(self.path, os.pardir, os.pardir)) + "\\data\\processed\\" + train[
                0]).to_numpy()
        self.train_y = pd.read_csv(
            os.path.abspath(os.path.join(self.path, os.pardir, os.pardir)) + "\\data\\processed\\" + train[
                1]).to_numpy()
        self.test_x = pd.read_csv(
            os.path.abspath(os.path.join(self.path, os.pardir, os.pardir)) + "\\data\\processed\\" + test[0]).to_numpy()
        self.test_y = pd.read_csv(
            os.path.abspath(os.path.join(self.path, os.pardir, os.pardir)) + "\\data\\processed\\" + test[1]).to_numpy()
        print('X Training Shape::', self.train_x.shape)
        print('X Testing  Shape::', self.test_x.shape)
        print('Y Training Shape::', self.train_y.shape)
        print('Y Testing  Shape::', self.test_y.shape)
        pass

    @abstractmethod
    def get_model(self):
        pass


class LogisticModel(AbstractModel):
    import numpy as np
    twits = np.array([0, 64, 9, 5, 131, 60, 3, 13, 487, 34, 130]) # Need to text to arraya
    twits = twits.reshape(1, -1)

    def get_model(self, *args):
        super().load_train_test_data(args[0], args[1])
        model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
        model.fit(self.train_x, self.train_y)
        print(classification_report(self.test_y, model.predict(self.test_x)))
        pred = model.predict(self.twits)
        print(pred)
        pass
    def get_token_from_input(self):
        reviews = ['very good quality though']
        tokenizer = Tokenizer(num_words=10)
        tokenizer.fit_on_texts(clean_df[CLEANED_TEXT].astype(str))
        sequences = tokenizer.texts_to_sequences(clean_df[CLEANED_TEXT].astype(str))

class SVMModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Support Vector Machines Model creation for :", args[0], ":")
        pass


class DTRFModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Decision Trees and Random Forests Model creationfor :", args[0], ":")
        pass
