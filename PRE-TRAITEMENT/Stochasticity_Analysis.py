from math import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
import warnings
'''
This script perform a Stochasticity Analysis to know how many replicat to perform to neutralize the stochasticity

'''
def readProblem(file):
    file = open(file, 'r')
    texte = file.read()
    texte = re.sub("\n", '', texte)
    texte = re.sub("\t", '', texte)
    texte_split = re.split(";", texte)
    names = texte_split[1]
    names = re.sub("\[", '', names)
    names = re.sub("]", '', names)
    names = re.sub("'", '', names)
    names_list = re.split(",", names)
    bounds_value = np.zeros((int(texte_split[0]), 2))
    bounds = texte_split[2]
    bounds = re.sub("\[\[", "[", bounds)
    bounds = re.sub("]]", "]", bounds)
    bounds = re.sub("] \[", "][", bounds)
    bounds = re.split("]\[", bounds)
    i = 0
    for val in bounds:
        tmp = re.sub("\[", '', val)
        tmp = re.sub("]", '', tmp)
        tmp = re.split(" ", tmp)

        bounds_value[i][0] = float(tmp[0])
        bounds_value[i][1] = float(tmp[1])
        i = i + 1

    problem = {
        'num_vars': int(texte_split[0]),
        'names': names_list,
        'bounds': bounds_value
    }
    file.close()
    return problem


def find_threshold(size,Cv_total,threshold):
    threshold_ok=False
    threshold_sample=0
    for i in range(0,size-1):
        for y in range(i+1,size):
            tmp_val=abs(Cv_total[i]-Cv_total[y])
            if tmp_val<=threshold and (not threshold_ok):
                threshold_ok=True
                threshold_sample=i+1
                break
    return threshold_sample

def calcul_Cv(size,Std_total,Mean_total):
    Cv_total = np.zeros(size)
    for i in range(0,size):
        if(i==0):
            Cv_total[i]=1.0
        else:
            Cv_total[i]=Std_total[i]/Mean_total[i]
    return Cv_total


def calcul_STD(size,data,Mean_total):
    Std_total = np.zeros(size)
    for i in range(0,size):
        sum_X = 0
        for y in range(0,i):
            sum_X = sum_X + pow((data[y] - Mean_total[i]), 2)
        Std_total[i]=np.sqrt(sum_X/(i+1))
    return Std_total


def calcul_mean(size,data):
    Mean_total = np.zeros(size)
    for i in range(0,size):
        for y in range(0,i):
            Mean_total[i]=Mean_total[i]+data[y]
        Mean_total[i]=Mean_total[i]/(i+1)

    return Mean_total

def calcul_Standard_error(size,STD):
    Se_total = np.zeros(size)
    for i in range(0,size):
        if(i==0):
            Se_total[i]=1.0
        else:
            Se_total[i]=STD[i]/sqrt(i+1)
    return Se_total

def nb_replicat_one_point(size,data,threshold):
    #Calcul all mean for each number of replicate
    mean = calcul_mean(size, data)

    #Calcul all STD for each number of replicate
    STD = calcul_STD(size, data, mean)

    #Calcul all Cv for each number of replicate
    Cv = calcul_Cv(size, STD, mean)

    #Cacul Standard error
    Se = calcul_Standard_error(size, STD)

    #Find the threshold value
    id_sample = find_threshold(size, Cv, threshold)
    #Plot a curve with the Number of samples and the coefficient of variation
    samples = list(range(1, size+1))
    plt.plot(samples, Cv,color="red",label="Coefficient of variation")
    #plt.plot(id_sample, Cv[id_sample], marker="o", color="red")
   #plt.text(id_sample, Cv[id_sample], id_sample)
    return id_sample

def calcul_variance(size,STD):
    Var_total = np.zeros(size-1)
    for i in range(0,size-1):
        Var_total[i]=pow(STD[i+1],2)
    return Var_total

def plot_se(size,data,threshold):
    #Calcul all mean for each number of replicate
    mean = calcul_mean(size, data)

    #Calcul all STD for each number of replicate
    STD = calcul_STD(size, data, mean)

    #Calcul all Cv for each number of replicate
    #Cv = calcul_Cv(size, STD, mean)

    #Cacul Standard error
    Se = calcul_Standard_error(size, STD)

    #id_sample = find_threshold(size, Se, threshold)
    first=True
    id_sample=None
    for i in range(0,len(Se)):
        if first and Se[i]<threshold:
            id_sample=i+1
            first=False

    if(id_sample==None):
        id_sample=len(Se)


    #Plot a curve with the standard error

    samples = list(range(1, size + 1))
    plt.plot(samples, Se,color="green",label="Standard Error")

    return id_sample


def mean_dif(mean,data):
    sum_X = 0
    for i in range(0, size):
        sum_X = sum_X + (data[i] - mean)
    mean_dif = sum_X / size
    return mean_dif


def student(data,size):
    mean = calcul_mean(size, data)
    STD = calcul_STD(size, data, mean)
    t_alph=0.678
    t_beta=0.678
    mean = calcul_mean(size, data)

    s= STD[len(STD)-1]
    sigma= mean_dif(mean,data)

    n_min= 2* ((s**2)/sigma)*((t_alph+t_beta)**2)

def student2(size,data):
    mean=calcul_mean(size,data)
    s=0
    #calcul s =>standardviation => écart type
    sum=0
    for i in range(0,size):
        sum=sum+(pow(data[i],2)-pow(mean[size-1],2))
        #sum=sum+pow(data[i]-mean[size-1],2)
    s=math.sqrt(sum/size)

    #calcul delta => lower bound on the absolute difference in means
    sum=0
    for i in range(0,size):
        sum=sum+ abs(data[i]-mean[size-1])
    delta=sum/size
    #Pour 80
    t1=1.664
    t2=1.292
    #Pour 100
    t1=1.660
    t2=1.290

    n_min= 2* (pow(s,2)/delta)*pow((t1+t2),2)

    return math.ceil(n_min)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s [option] -r INT -s INT -problem /path/to/problem.txt -data /path/to/data -output /path/to/output.png')

    parser.add_argument('-problem', metavar="/path/to/problem.txt", type=str, help='Path to the problem', required=True)
    parser.add_argument('-output', metavar="/path/to/output.png", type=str ,help='Output argument', required=True)
    parser.add_argument('-r',"--replicat",metavar="INT",type=int,help="The number of replicat",required=True)
    parser.add_argument('-s',"--sample",metavar="INT",type=int,help="The number of sample",required=True)
    parser.add_argument('-t', "--threshold", type=float, help="Threshold value for the number of min replicat (Default: Perform 0.001,0.05,0.01)", default=-1)
    parser.add_argument('-data',metavar="/path/to/data", type=str, help="Path to data folder", required=True)

    args = parser.parse_args()
    path_problem=args.problem
    path_output=args.output
    path_data=args.data
    replicat=args.replicat
    size=args.sample
    threshold=args.threshold

    #DATA

    problem = readProblem(path_problem)
    df = pd.read_csv(path_data+"/Results_0.csv")
    problem["names"] = df.columns
    output = problem["num_vars"]
    warnings.filterwarnings("ignore")


    print("\n== STOCHASTICITY ANALYSIS..\n")
    percent=[]
    if(threshold==-1):
        threshold=[0.05,0.01,0.005,0.001]
        percent=[0.10,0.075,0.05,0.025]
    else:
        threshold=[threshold]
        percent=[0.05]

    minimum=0
    for i in range(0, size):

        data = np.zeros(replicat)
        for z in range(0, replicat):
            df = pd.read_csv(path_data + "/Results_" + str(z) + ".csv")
            data[z] = df[problem["names"][output]][i]

        minimum = student2(replicat, data) + minimum
    minimum = round(minimum / size)
    print("\n")
    print("Min for Student")
    print(minimum)
    print("\n")
    for y in range(0,len(threshold)):
        minimum=0
        for i in range(0,size):

            data = np.zeros(replicat)
            for z in range(0,replicat):
                df = pd.read_csv(path_data+"/Results_"+str(z)+".csv")
                data[z]=df[problem["names"][output]][i]
            minimum=nb_replicat_one_point(replicat,data,threshold[y])+minimum
        minimum=round(minimum/size)
        print("Global minimum is for CV with threshold: "+str(threshold[y]))
        print(minimum)
        print("\n")
        plt.title("Minimum replicat required - Threshold: "+str(threshold[y])+" - Min : "+str(minimum))
        plt.ylabel("Coefficient of variation")
        plt.xlabel("Nb samples")
        path_output_tmp = re.sub(".png", "", path_output)
        plt.savefig(path_output_tmp+"_"+str(threshold[y])+".png")
        plt.clf()

        minimum=0
        for i in range(0,size):

            data = np.zeros(replicat)
            for z in range(0,replicat):
                df = pd.read_csv(path_data+"/Results_"+str(z)+".csv")
                data[z]=df[problem["names"][output]][i]
            minimum=plot_se(replicat,data,percent[y])+minimum
        minimum=round(minimum/size)
        print("Global minimum is for SE with percent: "+str(percent[y]*100)+"% ")
        print(minimum)
        print("\n")
        plt.title("Minimum replicat required - Threshold: "+str(percent[y]*100)+"% - Min : "+str(minimum))
        plt.ylabel("Standard_Error")
        plt.xlabel("Nb samples")
        path_output_tmp=re.sub(".png","",path_output)
        plt.savefig(path_output_tmp+"_"+str(percent[y]*100)+"_percent.png")
        plt.clf()

    print("== END")
