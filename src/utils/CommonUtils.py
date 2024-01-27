from os import path
from time import sleep
from progressbar import progressbar

class CommonUtils():

    def isModelExit(self):
        print("Chekcing is model exit or not")
        if path.exists("mymodel.h5"):
            return True
        return False;

    def bar(self,l):
        for i in progressbar(range(int(l/50))):
            sleep(0.02)

# cu = CommonUtils()
# cu.bar()
