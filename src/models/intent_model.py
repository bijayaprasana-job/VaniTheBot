from abc import ABC, abstractmethod

from src.models.abs_base_model import AbstractBaseModelFactory, SVMModel, DTRFModel, LogisticModel


# class AbstractIntentModel(ABC):
#
#     @abstractmethod
#     def get_model(self):
#         pass


class IntentModelFactory(AbstractBaseModelFactory):
    model_type: str = ''

    def __int__(self, model_type):
        self.model_type = model_type

    def create_model(self, model_type, train, test):
        self.__int__(model_type)
        if self.model_type == 'logistic':
            LogisticModel().get_model(train, test)
        elif self.model_type == 'svm':
            SVMModel().get_model(train, test)
        elif self.model_type == 'dtrf':
            DTRFModel().get_model(train, test)
        else:
            print("Model Not Found")
        pass
