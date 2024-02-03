from abc import ABC, abstractmethod
import pandas as pd
import os
import pickle
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.metrics import classification_report
import numpy as np
import sklearn

from src.utils.globals import MAX_SEQUENCE_LENGTH


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
        self.tokenizer = Tokenizer()

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
    twits = ['Who killed Ravana']  # Need to text to arraya
    def get_model(self, *args):
        print("The arfs are --" , args)
        super().load_train_test_data(args[1], args[2])
        model = sklearn.linear_model.LogisticRegression(max_iter=10000)
        model.fit(self.train_x, np.ravel(self.train_y))

        filename = str(args[0])+'.pkl'
        path = os.getcwd()
        savpath = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\models\\"
        with open(savpath+filename, 'wb') as file:
            pickle.dump(model, file)
        print(classification_report(self.test_y, model.predict(self.test_x)))
        tokenizer = self.get_token_from_input(args[0])
        sequences = tokenizer.texts_to_sequences(self.twits)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(data)
       # print("Model score is " , model.score(self.test_y,pred)* 100)

        print("THe PRedicted class is " , pred)
        if int(pred) == "1":
            print("Happy")
        else:
            print("UnHappy")
        pass

    def get_token_from_input(self,fileName):
        with open('../../models/'+fileName+'_tokenizer.pkl', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return self.tokenizer


class SVMModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Support Vector Machines Model creation for :", args[0], ":")
        pass


class DTRFModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Decision Trees and Random Forests Model creationfor :", args[0], ":")
        pass
