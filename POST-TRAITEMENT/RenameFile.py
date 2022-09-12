import os
import glob
from numpy import genfromtxt
import pandas as pd
import numpy

if __name__ == '__main__':

    #files = os.path.join("../CSV_FILE/GLOBAL/", "*.csv")
    files = os.path.join("../CSV_FILE/STOCHASTICITY/CHART8/GLOBAL/", "*.csv")
    files = glob.glob(files)
    print(files)


    for i in range(0,len(files)):
        #os.rename(files[i], "../CSV_FILE/GLOBAL/Result_" + str(i) + ".csv")
        os.rename(files[i], "../CSV_FILE/STOCHASTICITY/CHART8/GLOBAL/Results_"+str(i)+".csv")