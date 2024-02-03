import pickle

from keras.src.preprocessing.text import Tokenizer

from src.framework.AbstractBaseHandler import AbstractHandler
from src.utils.globals import MAX_NUM_WORDS, CLEANED_TEXT, MAX_SEQUENCE_LENGTH
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras import utils
import os
from sklearn import preprocessing
class DataPreProcess(AbstractHandler):

    def handle(self, item, clean_df) -> str:
        print("==> Data Tokenzitation is Started:" , item)
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(clean_df[CLEANED_TEXT].astype(str))
        sequences = tokenizer.texts_to_sequences(clean_df[CLEANED_TEXT].astype(str))
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        label_encoder = preprocessing.LabelEncoder()

        class_types = label_encoder.fit_transform(clean_df.iloc[:, 1])
        integer_mapping = {l: i for i, l in enumerate(label_encoder.classes_)}
        print("THe Integer maapping" , integer_mapping)
        print("==> Data Tokenzitation is Done with Length:", len(sequences))
        path = os.getcwd()
        savpath = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\models\\"
        with open(savpath+item['type']+'_tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return super().handle(item,sequences, class_types)


class DataPadTokenizer(AbstractHandler):
    def handle(self, item,sequences, class_types) -> str:
        print("==> Data Padding ia Started:")
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # encoder = LabelEncoder()
        # encoder.fit(class_types)
        # encoded_classes = encoder.transform(class_types)
        # labels = utils.to_categorical(encoded_classes)
        print('Shape of X ::', data.shape)
        print('Shape of Y ::', class_types.shape)
        print("==> Data Padding ia Done:")
        msg = super().handle(item,data, class_types)
        return msg
