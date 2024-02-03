import os
import pickle

from keras.src.utils import pad_sequences

from src.utils.globals import MAX_SEQUENCE_LENGTH


class IntentClassifier:
    def __init__(self, filename):
        self.tokenizer = None
        self.sequences = None
        self._filename = filename

        print('IntentClassifier::init')

    def classify_intent(self, input_msg):
        input_msg = [input_msg]
        print("input_msg", input_msg)
        path = os.getcwd()
        savpath = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\models\\"
        with open(savpath+self._filename+'.pkl', 'rb') as file:
            model = pickle.load(file)
        with open(savpath+self._filename+'_tokenizer.pkl', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.sequences = self.tokenizer.texts_to_sequences(input_msg)
        data = pad_sequences(self.sequences, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(data)
        if int(pred) == "1":
            return "Happy"
        else:
            return "UnHappy"

    def get_filename(self):
        return self._filename

    def set_filename(self, filename):
        self._filename = filename
