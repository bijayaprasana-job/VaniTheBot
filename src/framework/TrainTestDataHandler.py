from typing import Tuple, Any

from src.framework.AbstractBaseHandler import AbstractHandler
import numpy as np

from src.utils.globals import VALIDATION_SPLIT


class GenerateTrainTestData(AbstractHandler):

    def handle(self, data, labels) -> tuple[Any, Any, Any, Any]:
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
        x_train = data[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]
        return x_train, y_train, x_val, y_val  # super().handle()
