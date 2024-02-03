from typing import Tuple, Any

from src.framework.AbstractBaseHandler import AbstractHandler
import numpy as np
import pandas as pd
import os
from src.utils.globals import VALIDATION_SPLIT


class GenerateTrainTestData(AbstractHandler):

    def handle(self, item, data, labels) -> tuple[Any, Any, Any, Any]:
        path = os.getcwd()
        outfileName = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + "\\data\\processed\\"
        print("The item in Train/Test is", item)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
        x_train = data[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]
        print("Traing and Test Data Created with the Following shape:")
        print('X Training Shape::', x_train.shape)
        print('X Testing  Shape::', x_val.shape)
        print('Y Training Shape::', y_train.shape)
        print('Y Testing  Shape::', y_val.shape)
        x_train_df = pd.DataFrame(x_train)
        x_train_df.to_csv(outfileName + item['train'][0] ,index=False)
        y_train_df = pd.DataFrame(y_train)
        #y_train_df = y_train_df.drop(y_train_df.columns[-1],axis=1)
        y_train_df.to_csv(outfileName + item['train'][1],index=False)
        x_test_df = pd.DataFrame(x_val)
        x_test_df.to_csv(outfileName + item['test'][0],index=False)
        y_test_df = pd.DataFrame(y_val)
       # y_test_df = y_test_df.drop(y_test_df.columns[-1],axis=1)
        y_test_df.to_csv(outfileName + item['test'][1],index=False)
        return True
