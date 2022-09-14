import argparse
import re
from SALib.analyze import sobol
from SALib.analyze import morris
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys

'''
This script performs a sensitivity analysis according on a selected method. In this script, you can find two methods:
-Sobol
-Morris

This script save in a given folder a .txt file with result and a .png file of the result.

This script in a part of a project folder and use externals files.
This script use some relative path that match with the project folder. If you want to use this script outside of this folder, you have to change some path.
This is a list of some relative path to change with explanation:

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

def Sobol(problem,Y,path_to_output):
    out = open(path_to_output, 'w')
    sys.stdout = out
    sys.stderr = out
    with open(path_to_output, 'w') as f:
        Si = sobol.analyze(problem, Y, print_to_console=True)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    out.close()
    Si.plot()
    output_png = re.sub("\.txt", ".png", path_to_output)
    plt.savefig(output_png)
    print("=== Done and Save")
    print("=== END")

def Morris(problem,X,Y,path_to_output,nb_parameter):
    out = open(path_to_output, 'w')
    sys.stdout = out
    sys.stderr = out
    with open(path_to_output, 'w') as f:
        Si = morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=True, num_levels=4)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    out.close()
    names = problem["names"]
    val_mu=[]
    val_sigma=[]
    for i in range(nb_parameter):
        val_mu.append(Si["mu"][i])
        val_sigma.append(Si["sigma"][i])

    data = {'mu': val_mu,
            'sigma': val_sigma
            }

    plt.scatter('mu', "sigma", data=data)
    for y in range(nb_parameter):
        plt.text(Si["mu"][y], Si["sigma"][y], names[y])
    plt.xlabel('mu')
    plt.ylabel('sigma')
    output_png = re.sub("\.txt", ".png", path_to_output)
    plt.savefig(output_png)
    print("=== Done and Save")
    print("=== END")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s [options] -analysis analysis /path/to/problem.txt /path/to/data.csv -output /path/to/output.txt')

    parser.add_argument('-analysis', metavar=("analysis","/path/to/problem.txt", "/path/to/data.csv"), nargs=3,help='Arguments for different analysis (sobol,morris)', required=True)
    parser.add_argument('-output', metavar="/path/to/output.txt", type=str ,help='Output argument', required=True)
    parser.add_argument('-i',"--id_output", metavar="INT",type=int, help='The id of the output to analyse (Default:0)', default=0)

    args = parser.parse_args()

    type_analysis,path_to_problem,path_to_data = args.analysis
    path_to_output =args.output
    id_output=args.id_output

    #Read Problem
    problem=readProblem(path_to_problem)

    print("=== Reading the data...\n")
    my_data = genfromtxt(path_to_data, delimiter=',')
    Y=my_data[1:]
    new_Y=np.array([])

    X=Y[:,:problem["num_vars"]]
    for cycle in Y:
        try:
            new_Y= np.append(new_Y,cycle[problem["num_vars"]+id_output])
        except:
            print("== ID OUTPUT NOT VALID")
            exit(0)

    if type_analysis=="sobol":
        print("=== Starting Sobol Analysis...\n")
        Sobol(problem,new_Y,path_to_output)
    else:
        if type_analysis=="morris":
            print("=== Starting Morris Analysis...\n")
            Morris(problem,X,new_Y,path_to_output,problem["num_vars"])
        else:
            print("==This Analysis doesn't exist... ( "+type_analysis+" )")
            exit(0)


