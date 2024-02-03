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
from src.utils import globals as gv

app = Flask(__name__)
app.static_folder = 'static'
path = os.getcwd()
config_file_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + gv.CONFIG_FILE;
intentCls = IntentClassifier('test')


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
    # Create Traingin/Test data
    # Create specifc model
    # tranidn the model wth the data & save it
    # load the specific model for and predict , intent , stemnt , maina
    # if PRD MODE then take user UserInput as a Object [ stmt , intent , sentiment , main ]
    # send UserObject to model to get Repsone based on intent,sentment and mainchar
    # Create Response Object to send back to actul response
    #raw_intent_df = dataset.load_csv('intent')
    print("FileName",intentCls.get_filename());
    usertext = request.args.get('msg')
    return intentCls.classify_intent(usertext)


def traning_mode(configs):
    commonUtil = CommonUtils()
    temp_item = []
    itemsary = configs['globalconfig']['training']
    print(type(configs['globalconfig']['training']))
    for index, items in enumerate(configs['globalconfig']['training']):
        if not items['processed']:
            print(index, items)
            begineHandler = DataLoader()
            begineHandler.set_next(DataCleaner()).set_next(DataPreProcess()).set_next(DataPadTokenizer()).set_next(GenerateTrainTestData())
            msg = begineHandler.handle(items)
            #items['processed'] = True
            print("------------------------Creating Modeleng for ----------------------" ,items['type'] , msg)
            itemsary[index] = items
            model_obj = CreateModel()
            print("items -->" , items)
            model_obj.get_model_pipeline(items);
            #items['modelcreated'] = True
            temp_item = items

    configs['globalconfig']['training'] = itemsary
    with open(config_file_path, 'w') as config_file:
        json.dump(configs, config_file)
    return temp_item


if __name__ == "__main__":
    configs = []
    path = os.getcwd()
    with open(os.path.abspath(os.path.join(path, os.pardir, os.pardir)) + gv.CONFIG_FILE) as con_file:
        configs = json.load(con_file)

    if configs['globalconfig']['mode'] == 'train':
        myitems = traning_mode(configs)
        print("MYits" , myitems)
        intentCls.set_filename(myitems['type'])
        app.run(host="localhost", port=8080)
