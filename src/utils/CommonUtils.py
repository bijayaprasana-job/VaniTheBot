from os import path
from time import sleep
from tqdm import tqdm
from progress.spinner import MoonSpinner
from progressbar import progressbar
from progress.bar import Bar
from alive_progress import alive_bar
import time
class CommonUtils():

    def isModelExit(self):
        print("Chekcing is model exit or not")
        if path.exists("mymodel.h5"):
            return True
        return False;

    def bar(self, l):
        for x in 1000, 1500, 700, 0:
            with alive_bar(x) as bar:
                for i in range(1000):
                    time.sleep(.005)
                    bar()


# cu = CommonUtils()
# l =10000
# cu.bar(l)
