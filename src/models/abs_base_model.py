from abc import ABC, abstractmethod


class AbstractBaseModelFactory(ABC):

    @abstractmethod
    def create_model(self):
        pass

    # @abstractmethod
    # def create_sentiment_model(self):
    #     pass


class AbstractModel(ABC):

    @abstractmethod
    def get_model(self):
        pass


class LogisticModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Logistic Model creation for :", args[0], ":")
        pass


class SVMModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Support Vector Machines Model creation for :", args[0], ":")
        pass


class DTRFModel(AbstractModel):
    def get_model(self, *args):
        print("Inside Decision Trees and Random Forests Model creationfor :", args[0], ":")
        pass
