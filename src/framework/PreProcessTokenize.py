from keras.src.preprocessing.text import Tokenizer

from src.framework.AbstractBaseHandler import AbstractHandler
from src.utils.globals import MAX_NUM_WORDS, CLEANED_TEXT, MAX_SEQUENCE_LENGTH
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras import utils



class DataPreProcess(AbstractHandler):

    def handle(self,item, clean_df) -> str:
        print("==> Data Tokenzitation is Started:")
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(clean_df[CLEANED_TEXT].astype(str))
        sequences = tokenizer.texts_to_sequences(clean_df[CLEANED_TEXT].astype(str))
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        class_types = clean_df.iloc[:, 1].to_numpy()
        print("==> Data Tokenzitation is Done with Length:" , len(sequences))
        return super().handle(sequences,class_types)


class DataPadTokenizer(AbstractHandler):
    def handle(self,sequences,class_types) -> str:
        print("==> Data Padding ia Started:")
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        encoder = LabelEncoder()
        encoder.fit(class_types)
        encoded_classes = encoder.transform(class_types)
        labels = utils.to_categorical(encoded_classes)
        print('Shape of X ::', data.shape)
        print('Shape of Y ::', labels.shape)
        print("==> Data Padding ia Done:")
        x_train, y_train, x_val, y_val = super().handle(data,labels)
        print("Traing and Test Data Created with the Following shape:")
        print('X Training Shape::', x_train.shape)
        print('X Testing  Shape::', x_val.shape)
        print('Y Training Shape::', y_train.shape)
        print('Y Testing  Shape::', y_val.shape)
        return x_train, y_train, x_val, y_val
