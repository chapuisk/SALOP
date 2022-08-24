from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
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

def nb_replicat_one_point(size,data,threshold,path_save):
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
    print("The n min for Coefficient of variation value is:")
    print(id_sample)

    #Plot a curve with the Number of samples and the coefficient of variation
    samples = list(range(1, size+1))
    plt.plot(samples, Cv,color="red",label="Coefficient of variation")
    plt.plot(id_sample, Cv[id_sample], marker="o", color="red")
    plt.text(id_sample, Cv[id_sample], id_sample)
    return id_sample

"""
    id_sample = find_threshold(size, Se, threshold)
    print("The n min for Standard Error value is:")
    print(id_sample)

    #Plot a curve with the standard error

    plt.plot(samples, Se,color="green",label="Standard Error")
    plt.xlabel("Number of samples")
    plt.ylabel("Coefficient")
    #plt.legend(loc="best")
    #plt.show()


    samples = list(range(1, size))
    Variance=calcul_variance(size,STD)
    plt.plot(samples,Variance,color="blue",label="Variance")
    #plt.xlabel("Number of samples")
    #plt.ylabel("Variance")
    plt.legend(loc="best")
    plt.xlim(0,size+1)
    plt.show()
"""
def calcul_variance(size,STD):
    Var_total = np.zeros(size-1)
    for i in range(0,size-1):
        Var_total[i]=pow(STD[i+1],2)
    return Var_total



if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s [option] -r INT -s INT -problem /path/to/problem.txt -data /path/to/data -output /path/to/output.csv')

    parser.add_argument('-problem', metavar="/path/to/problem.txt", type=str, help='Path to the problem', required=True)
    parser.add_argument('-output', metavar="/path/to/output.csv", type=str ,help='Output argument', required=True)
    parser.add_argument('-r',"--replicat",metavar="INT",type=int,help="The number of replicat",required=True)
    parser.add_argument('-s',"--sample",metavar="INT",type=int,help="The number of sample",required=True)
    parser.add_argument('-h', "--threshold", type=int, help="Threshold value for the number of min replicat (Default:0.01)", default=0.01)
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

    print("== STOCHASTICITY ANALYSIS..\n")
    minimum=0
    for i in range(0,size):

        data = np.zeros(replicat)
        for z in range(0,replicat):
            df = pd.read_csv(path_data+"/STOCHASTICITY/Results_"+str(z)+".csv")
            data[z]=df[problem["names"][output]][i]

        minimum=nb_replicat_one_point(replicat,data,threshold,path_output)+minimum
    minimum=round(minimum/size)	
    plt.savefig(path_save)
    print("Global minimum is:")
    print(minimum)
    print("== END")
