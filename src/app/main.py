from flask import Flask, render_template, request

from src.framework.AbstractBaseHandler import AbstractHandler
from src.framework.DataLoadClean import DataLoader, DataCleaner
from src.framework.PreProcessTokenize import DataPreProcess, DataPadTokenizer
from src.framework.TrainTestDataHandler import GenerateTrainTestData
from src.models.create_model import CreateModel
import os
import json
from src.models.nlp.IntentClassifier import IntentClassifier
from src.utils.CommonUtils import CommonUtils
from src.utils.LoadRawData import LoadRawData
from src.utils.globals import CONFIG_FILE

app = Flask(__name__)
app.static_folder = 'static'
path = os.getcwd()
config_file_path = os.path.abspath(os.path.join(path, os.pardir,os.pardir)) + CONFIG_FILE;


def get_me_response(msg):
    return msg


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    dataset = LoadRawData()  # if mode is traning then only
    # Clean Data
    # Tokenize and pad data
    # Create specifc model
    # tranidn the model wth the data & save it
    # load the specific model for and predict , intent , stemnt , maina
    # if PRD MODE then take user UserInput as a Object [ stmt , intent , sentiment , main ]
    # send UserObject to model to get Repsone based on intent,sentment and mainchar
    # Create Response Object to send back to actul response
    raw_intent_df = dataset.load_csv('intent')
    intentCls = IntentClassifier()
    usertext = request.args.get('msg')
    return intentCls.classify_intent(usertext)


def tranin_mode(*args):
    commonUtil = CommonUtils()
    if not bool(args[0]['globalconfig']['modelcreated'] == 'true'):
        begineHandler = DataLoader()
        begineHandler.set_next(DataCleaner()).set_next(DataPreProcess()).set_next(DataPadTokenizer()).set_next(GenerateTrainTestData())
        result = begineHandler.handle(args[0]['globalconfig']['training'])
        # model_obj = CreateModel()
        # model_obj.get_model_pipeline();
        # args[0]['globalconfig']['modelcreated'] = 'true'
        # with open(config_file_path, 'w') as config_file:
        #     json.dump(args[0],config_file)
        print("Data Cleaning is done for ", (str(result).upper()))

    return True


if __name__ == "__main__":
    with open(config_file_path) as con_file:
        configs = json.load(con_file)

    # mode = input("Please enter Mode: ")
    # isTextCleaned = input("Text Cleaning is Done: ")
    if not tranin_mode(configs):
        app.run(host="localhost", port=8080)
