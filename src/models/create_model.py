import json

from src.models.intent_model import IntentModelFactory
from src.models.sentiment_model import SentimentModelFactory
import os

from src.utils.globals import CONFIG_FILE


class CreateModel():
    config = []

    def __int__(self, config):
        self.config = config

    def __get_factory(self, model_type):
        factory = None
        model_type = model_type.strip()
        if str(model_type).strip() == 'intent':
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

    def get_model_pipeline(self, items):
        print("Started Creating models" , items['type'])
        factory_object = self.__get_factory(items['type'])
        for model in items['models']:
            factory_object.create_model(items['type'],model, items['train'],items['test'])


# if __name__ == '__main__':
#     app_object = CreateModel()
#     path = os.getcwd()
#     with open(os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + CONFIG_FILE) as user_file:
#         configs = json.load(user_file)
#     print("Started Creating models", configs)
#     for items in configs['globalconfig']['training']:
#         factory_object = app_object.get_factory(str(items['type']))
#         for model in items['models']:
#             factory_object.create_model(model, items['type'])
