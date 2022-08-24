import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from numpy import genfromtxt
from math import *

def createDF(X,Res,Time,name,output):
    df=pd.DataFrame(columns=[name,"time",output])
    for i in range(0,len(X)-1):
        for y in range(0,len(Time)-1):
            df_tmp=pd.DataFrame.from_records([{name :X[i],"time":Time[y], output:Res[i][y]}])
            df=pd.concat([df,df_tmp],ignore_index= True)

    return df



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


def courbe_point():
    print("Courbe2")
    for i in range(0, output):
        df = pd.read_csv("./CHART4/GLOBAL/Results_TESTCHART_" + str(y) + ".csv")
        DFUN = df.drop_duplicates(subset=problem["names"][i])
        DFUN = DFUN.sort_values(by=problem["names"][i])
        moy = np.zeros(len(DFUN))
        STV = np.zeros(len(DFUN))
        STVmax = np.zeros(len(DFUN))
        STVmin = np.zeros(len(DFUN))

        for y in range(0, nb_replicat):
            df = pd.read_csv("./CHART4/GLOBAL/Results_TESTCHART_" + str(y) + ".csv")
            DFUN = df.drop_duplicates(subset=problem["names"][i])
            DFUN = DFUN.sort_values(by=problem["names"][i])
            moy = moy + DFUN[problem["names"][output]]
            plt.plot(DFUN[problem["names"][i]].to_numpy(), DFUN[problem["names"][output]].to_numpy(), 'o', color='grey',
                     alpha=0.3)
        moy = moy / nb_replicat
        for y in range(0, nb_replicat):
            df = pd.read_csv("./CHART4/GLOBAL/Results_TESTCHART_" + str(y) + ".csv")
            DFUN = df.drop_duplicates(subset=problem["names"][i])
            DFUN = DFUN.sort_values(by=problem["names"][i])
            STV = STV + ((DFUN[problem["names"][output]].to_numpy() - moy) * (
                        DFUN[problem["names"][output]].to_numpy() - moy))

        STV = np.sqrt(STV / (nb_replicat - 1))
        plt.plot(DFUN[problem["names"][i]].to_numpy(), moy, label="moy", color="red")
        STVmax = moy + STV
        STVmin = moy - STV
        plt.fill_between(DFUN[problem["names"][i]].to_numpy(), STVmin, STVmax, color="plum", linewidth=0.1, label="STD",
                         alpha=0.7)
        plt.xlabel(problem["names"][i])
        plt.ylabel(problem["names"][output])
        plt.legend(loc="best")
        plt.savefig("./CHART4/GLOBAL/Graph_Input_new_" + problem["names"][i] + ".png")
        plt.clf()

def courbe_simu():
    print("COURBE3")
    for i in range(0, sample):
        df = pd.read_csv("./CHART4/SAMPLE/SAMPLE_0/replicat_0.csv")
        moy = np.zeros(len(df))
        for y in range(0, nb_replicat):
            df = pd.read_csv("./CHART4/SAMPLE/SAMPLE_" + str(i) + "/replicat_" + str(y) + ".csv")
            moy = moy + df[problem["names"][output]]
            if y == 0:
                plt.plot(df["cycle"].to_numpy(), df[problem["names"][output]].to_numpy(), c="grey", lw=0.5, ls="--",
                         label="replicat")
            else:
                plt.plot(df["cycle"].to_numpy(), df[problem["names"][output]].to_numpy(), c="grey", lw=0.5, ls="--")
        moy = moy / nb_replicat
        plt.plot(df["cycle"].to_numpy(), moy, c="red", lw=1.5, ls="--", label="MOY")
        plt.xlabel("time")
        plt.ylabel(problem["names"][output])
        plt.title("SAMPLE_" + str(i))
        plt.legend(loc="best")
        plt.savefig("./CHART4/SAMPLE/SAMPLE_" + str(i) + "/GRAPH.png")
        plt.clf()

def heatmap_time():
    print("HEATMAP1")
    for z in range(0,output):
        df = pd.read_csv("./CHART4/GLOBAL/Results_TESTCHART_0.csv")
        df=df.drop_duplicates(subset=problem["names"][z])
        res = np.zeros(shape= (len(df),time))
        df = df.sort_values(by=problem["names"][z])
        l=df.index
        moy=np.zeros(time)
        tmp=0
        for h in l:

            for y in range(0,nb_replicat):
                df = pd.read_csv("./CHART4/SAMPLE/SAMPLE_" + str(h) + "/replicat_"+str(y)+".csv")
                moy=moy+df[problem["names"][output]]
            moy=moy/nb_replicat
            res[tmp]=moy
            tmp=tmp+1
        times=list(range(0,time))
        df = pd.read_csv("./CHART4/GLOBAL/Results_TESTCHART_0.csv")
        df = df.sort_values(by=problem["names"][z])
        fig, ax = plt.subplots()
        c = ax.pcolormesh(times, df[problem["names"][z]].unique(), res, cmap='RdBu_r', vmin=0, vmax=1)
        ymin=np.amin(df[problem["names"][z]].to_numpy())
        ymax=np.amax(df[problem["names"][z]].to_numpy())
        ax.axis([0, time, ymin, ymax])
        fig.colorbar(c, ax=ax,label=problem["names"][output])
        plt.xlabel("time")
        plt.ylabel(problem["names"][z])
        plt.savefig("./CHART4/GLOBAL/" +problem["names"][z] + "_HEAT_MAP.png")
        plt.clf()

def heatmap_time_2_factor():
    print("HeatMap2")
    for w in range(0,output-1):

        for n in range(w+1,output):
            if w!=n:
                df = pd.read_csv("./CHART5/GLOBAL/Results_TESTCHART_0.csv")
                df.sort_values(by=problem["names"][w])
                ValUn = df[problem["names"][w]].unique()
                NbVal = floor(len(ValUn) / cluster)
                val = []
                z = 0
                while (z < len(ValUn)):
                    tmp = []
                    for h in range(0, NbVal):
                        if ((len(ValUn) - z) != 0):
                            tmp.append(ValUn[z])
                            z = z + 1
                    val.append(tmp)

                df = pd.read_csv("./CHART5/GLOBAL/Results_TESTCHART_0.csv")
                df.sort_values(by=problem["names"][n])
                ValUn_2 = df[problem["names"][n]].unique()
                NbVal_2 = floor(len(ValUn_2) / cluster)
                val_2 = []
                z = 0
                while (z < len(ValUn_2)):
                    tmp = []
                    for h in range(0, NbVal_2):
                        if ((len(ValUn_2) - z) != 0):
                            tmp.append(ValUn_2[z])
                            z = z + 1
                    val_2.append(tmp)

                fig, axs = plt.subplots(cluster, cluster)
                for i in range(0, cluster):
                    df = pd.read_csv("./CHART5/GLOBAL/Results_TESTCHART_0.csv")
                    df = df.drop_duplicates(subset=problem["names"][w])
                    res = np.zeros(shape=(cluster, time))
                    tmp_moy = np.zeros(shape=(cluster, time))

                    for u in val[i]:
                        for y in range(0, cluster):
                            moy = np.zeros(time)
                            for v in val_2[y]:
                                df = pd.read_csv("./CHART5/GLOBAL/Results_TESTCHART_0.csv")
                                df = df.loc[df[problem["names"][w]] == u]
                                df = df.loc[df[problem["names"][n]] == v]
                                df = df[df[problem["names"][w]] != np.nan]
                                if (len(df) == 1):
                                    l = df.index
                                    for h in l:

                                        for z in range(0, nb_replicat):
                                            df = pd.read_csv(
                                                "./CHART5/SAMPLE/SAMPLE_" + str(h) + "/replicat_" + str(z) + ".csv")
                                            moy = moy + df[problem["names"][output]]
                                            axs[y, i].plot(df["cycle"].to_numpy(),
                                                           df[problem["names"][output]].to_numpy(), c="grey", lw=0.5,
                                                           ls="--")
                            moy = moy / (len(val_2[y]) * nb_replicat)
                            tmp_moy[y] = tmp_moy[y] + moy

                    for m in range(0,cluster):
                        res[m] = tmp_moy[m] / (len(val[i]))

                    df = pd.read_csv("./CHART5/GLOBAL/Results_TESTCHART_0.csv")
                    ymin = np.amin(df[problem["names"][w]].to_numpy())
                    ymax = np.amax(df[problem["names"][w]].to_numpy())

                    if (i == 0):
                        for g in range(0, cluster):
                            border_min = np.amin(val_2[g])
                            border_max = np.amax(val_2[g])
                            if (border_min == border_max):
                                text = round(border_min, 3)
                                axs[g, 0].set_ylabel(text)
                            else:
                                text = "[" + str(round(border_min, 3)) + "," + str(round(border_max, 3)) + "]"
                                axs[g, 0].set_ylabel(text)

                    times = list(range(0, 502))

                    border_min = np.amin(val[i])
                    border_max = np.amax(val[i])
                    if (border_max == border_min):
                        border_min = round(border_min, 3)
                        str_t = str(border_min)
                        axs[0, i].set_title(str_t)


                    else:
                        text = "[" + str(round(border_min, 3)) + "," + str(round(border_max, 3)) + "]"
                        axs[0, i].set_title(text)

                    for m in range(0,cluster):
                        axs[m, i].plot(times, res[m], c="red", lw=1.5, ls="--")
                        axs[m, i].axis([0, time, 0, 1])


                for ax in fig.get_axes():
                    ax.label_outer()
                fig.suptitle("x: " + problem["names"][w] + " / y: " + problem["names"][n])
                fig.tight_layout()
                plt.savefig("./CHART5/GLOBAL/" + problem["names"][w] +"_"+problem["names"][n] +"_HEAT_MAP_2.png")
                plt.clf()


def do_ranking_global(morris,sobol):
    tmp=np.zeros(len(morris))
    for i in range(0,len(morris)):
        tmp[i]=sobol[i]+morris[i]
    tmp_2=sorted(tmp)

    res=np.zeros(len(morris))
    for i in range(0,len(tmp)):
        for y in range(0,len(tmp)):
            if tmp_2[i]==tmp[y]:
                res[i]=y

    return res



def build_chart_prio_facteur(problem,morris,sobol,nb_var):
    rank=do_ranking_global(morris,sobol)
    print(rank)
    data=[]
    for i in rank :
        i=int(i)
        data.append([problem["names"][i],morris[i],sobol[i]])

    df = pd.DataFrame(data, columns=["Inputs", "Morris", "Sobol"])
    df.plot.barh(x="Inputs", y=["Morris", "Sobol"], color={"darkorange","cornflowerblue"},title="Inputs Importance Chart")
    plt.xlim(0,1)
    plt.show()

def morris_fit_data(morris):
    momo=morris / np.sqrt(np.sum(morris**2))
    return momo

if __name__ == '__main__':

    problem = readProblem("./model_problem_analysis.txt")


    df=pd.read_csv("./CHART4/GLOBAL/Results_TESTCHART_0.csv")

    problem["names"]=df.columns

    output=problem["num_vars"]
    nb_replicat=4
    sample=204
    time=502
    cluster=6

    morris=[0.497743,0.482880,0.02996,0.465042]
    morris=np.array(morris)
    morris=morris_fit_data(morris)
    sobol=[0.398204,0.236124,0.000781,0.52928]
    build_chart_prio_facteur(problem,morris,sobol,output)
