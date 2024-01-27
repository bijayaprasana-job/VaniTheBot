'''
Created on Apr 28, 2020

@author: Bijay
'''
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
#from vani.sentence_util.SentencePreprocessor import SentencePreprocessor
class LoadRawData(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.resource_path =  Path(__file__).parent / "../../data/raw"
        
    def load_csv(self,type):
        if type == 'intent':
            df = pd.read_csv(self.resource_path.joinpath('Dataset_Final.csv'), encoding = "windows-1250")
            return df
        
        elif type == 'sentiment':
            df = pd.read_csv(self.resource_path.joinpath('sentiment_train_Final.csv'), encoding = "latin1")
            sentiment_type=df["Is_Response"]
            unique_sentiment_type = list(set(sentiment_type))
            print('No of Unique_Classes Length-->' , len(unique_sentiment_type))
            sentences = list(df["Description"])
            return sentiment_type, unique_sentiment_type, sentences
            
        elif type == 'sentence':
            df = pd.read_csv(self.resource_path.joinpath('Sen_Type_Class_ds.csv'), encoding = "latin1")
            sentype = df.iloc[:,-1]
            unique_sentype = list(set(sentype))
            sentences = list(df.iloc[:,0])
            print('No of Unique_Classes Length-->' , len(unique_sentype))
            return sentype, unique_sentype, sentences
        
        elif type == 'AboutVani':
            df = pd.read_csv(self.resource_path.joinpath('D:\TF\Chat-Bot\R&D\chat_dataset_new_one.csv'), encoding = "unicode_escape")
           # sp=SentencePreprocessor()
            df['answers'] = df['answers'].apply(str)
            df['questions'] = df['questions'].apply(str)
            df['clean_text_x'] = df['questions'].apply(sp.clean_text_new)
            df['clean_text_y'] = df['answers'].apply(sp.clean_text_new)
            # Filter out the questions that are too short/long
            short_questions = list(df['clean_text_x'])
            short_answers = list(df['clean_text_y'])
            return short_questions , short_answers
        
        
  
#   
# ld = LoadTrainingData()
# ld.load_csv('sentence')      
