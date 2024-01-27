import json

from src.models.intent_model import IntentModelFactory
from src.models.sentiment_model import SentimentModelFactory
import os

from src.utils.globals import CONFIG_FILE


class CreateModel():

    def __get_factory(self, model_type):
        factory = None
        if model_type == "intent":
            factory = IntentModelFactory()
        elif model_type == "sentiment":
            factory = SentimentModelFactory()
        else:
            print("ERROR: unknown model type.")
        return factory

    def __create_model(self, factory=None):
        if not factory:
            print("factory object not passed")
        factory.create_model()

    def get_model_pipeline(self):

        path = os.getcwd()
        with open(os.path.abspath(os.path.join(path, os.pardir)) + CONFIG_FILE) as user_file:
            configs = json.load(user_file)
            print("Started Creating models")
        for items in configs:
            factory_object = self.__get_factory(items['type'])
            for model in items['models']:
                factory_object.create_model(model, items['type'])


# if __name__ == '__main__':
#     app_object = CreateModel()
#     path = os.getcwd()
#     with open(os.path.abspath(os.path.join(path, os.pardir)) + CONFIG_FILE) as user_file:
#         configs = json.load(user_file)
#     print("Started Creating models")
#     for items in configs:
#         factory_object = app_object.get_factory(items['type'])
#         for model in items['models']:
#             factory_object.create_model(model, items['type'])