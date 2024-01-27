import string

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from tqdm.dask import TqdmCallback
from dask.diagnostics import ProgressBar

from src.framework.AbstractBaseHandler import AbstractHandler
from typing import Any
from pathlib import Path
import pandas as pd
import os
import spacy
import re
import json
import time
from progressbar import progressbar

from src.utils.CommonUtils import CommonUtils
from src.utils.globals import RAW_FILE, CLEAN_FILE, CONFIG_FILE

path = os.getcwd()
clean_df = pd.DataFrame()
stopwords_list = set(stopwords.words('english'))


class DataLoader(AbstractHandler):
    cu = CommonUtils()

    def __init__(self):
        self.isFileProcessed = False

    def __int__(self):
        print("::DataLoader::")

    def load_csv(self, fileName):
        df = pd.read_csv(fileName, encoding="latin1")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df

    def handle(self, configs) -> []:
        clean_files = []
        for items in configs:
            self.isFileProcessed = items['processed']
            if not self.isFileProcessed:
                fileName = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\data\\raw\\" + items[
                    'filename']
                print("*******************Traning Pipeline started for :",str(items['type']).upper()+ ": Classification *******************************")
                print("Processing Raw File :", fileName,":")
                df = self.load_csv(fileName)
                print("Total Records :%2d" % len(df), 'for:', items['type'], ':')
                print('Unique_Classes :%2d' % len(list(set(df.iloc[:, 1]))), 'for:', items['type'], ':')
                super().handle(items, df)
                clean_files.append(items['type'])
            else:
                clean_files.append(items['type'])

        return clean_files


class DataCleaner(AbstractHandler):
    isCleaningDone: bool = False
    def __int__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def handle(self, item, dataFrame) -> str:
        print("==> Data Cleaning Started")
        self.__int__()
        if not bool(item['processed'] == 'true'):
            df = self.clean_text(dataFrame)
            rawfileName = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\data\\raw\\" + item[
                'filename']

            outfileName = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\data\\processed\\" + item[
                'outputfile']
            df.to_csv(outfileName)
            print("Clean Processed File is created ->", outfileName)
            super().handle(DataLoader().load_csv(outfileName))
            del df
        return "I am good"

    def clean_text(self, orgnial_df):

        clean_df['clean_text'] = orgnial_df.iloc[:, 0].apply(self.convert_to_ascii) \
            .apply(self.remove_punctuation) \
            .apply(self.removing_unnecessary) \
            .apply(self.lemmatize_text) \
            .apply(self.text_standardization)
        clean_df['class_types'] = orgnial_df.iloc[:, 1]
        return clean_df

    def lemmatize_text(self, text):
        lemmatizer = WordNetLemmatizer()
        word_list = word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(w, 'v') for w in word_list])
        word_list = word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(w, 'n') for w in word_list])
        word_list = word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(w, pos="a") for w in word_list])
        return text

    def removing_unnecessary(self, text):
        '''Clean text by removing unnecessary characters and altering the format of words.'''
        TAG_RE = re.compile(r'<[^>]+>')
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"plz", "please", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"lol", "lot of laugh", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"info", "information", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
        text = TAG_RE.sub('', text)
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[+-]?([0-9]*[.])?[0-9]+', ' ', text)
        text = text.replace('?', ' ').replace('!', ' ').replace('\t', ' ')
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        text = pattern.sub('', text)

        # print(text)
        return text

    def text_standardization(self, text):
        words = text.split()
        new_text = ''
        new_words = []
        for word in words:
            if word.lower() in look_up_dict:
                word = look_up_dict[word.lower()]
            new_words.append(word)
            new_text = " ".join(new_words)
        return new_text

    def remove_punctuation(self, sen):
        text = "".join([c for c in sen if c not in string.punctuation])
        return text

    def convert_to_ascii(self, statement):
        import unicodedata
        text = unicodedata.normalize('NFKD', statement)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text


look_up_dict = {
    'rt': 'Retweet',
    "dm": "direct message",
    "awsm": "awesome",
    "luv": "love",
    'lv': 'love',
    'bbye': 'bye',
    '2moro': 'tomorrow',
    '2mrrw': 'tomorrow',
    '2mor': 'tomorrow',
    'tomor': 'tomorrow',
    'tomrw': 'tomorrow',
    ':)': 'smile',
    '2morow': 'tomorrow',
    'who r': 'who are',
    'w r': 'who are',
    'wh r': 'who are',
    'wh re': 'who are',
    'wh r': 'who are',
    'w ar': 'who are',
    'w ae': 'who are',
    'wh ae': 'who are',
    'helo': 'hello',
    'hlo': 'hello',
    'hllo': 'hello',
    'hloo': 'hello',
    'hloo': 'hello',
    'helo': 'hello',
    'heooo': 'hello',
    'hlooo': 'hello'
}
