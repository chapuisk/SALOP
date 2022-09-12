import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from math import *
from matplotlib import cm
import os
import glob

'''
In this script you will find many methods allowing to draw graphics and to plot data on charts.
This script use relative path for the access to data. Please follow this structure.

|_ BUILD_CHARTS
               |_ BuildCharts.py
|_ CSV_FILE
           |_ GLOBAL
                    |_ Result_0.csv
                    |_ Result_1.csv
                    |_ ...
           |_ SAMPLING
                      |_ SAMPLE_0
                                 |_ replicat_0.csv
                                 |_ replicat_1.csv
                                 |_ ...
                      |_ SAMPLE_1
                      |_ ...
'''

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

'''
#######################################################################################
#######################################################################################
##############################      FOR OFAT        ###################################
#######################################################################################
#######################################################################################
'''
def courbe_point(id_output):
    print("=== Creating chart...\n")
    for i in range(0, output):
        df = pd.read_csv("../CSV_FILE/GLOBAL/Results_" + str(y) + ".csv")
        DFUN = df.drop_duplicates(subset=problem["names"][i])
        DFUN = DFUN.sort_values(by=problem["names"][i])
        moy = np.zeros(len(DFUN))
        STV = np.zeros(len(DFUN))
        STVmax = np.zeros(len(DFUN))
        STVmin = np.zeros(len(DFUN))

        for y in range(0, nb_replicat):
            df = pd.read_csv("../CSV_FILE/GLOBAL/Results_" + str(y) + ".csv")
            DFUN = df.drop_duplicates(subset=problem["names"][i])
            DFUN = DFUN.sort_values(by=problem["names"][i])
            moy = moy + DFUN[problem["names"][id_output]]
            plt.plot(DFUN[problem["names"][i]].to_numpy(), DFUN[problem["names"][id_output]].to_numpy(), 'o', color='grey',
                     alpha=0.3)
        moy = moy / nb_replicat
        for y in range(0, nb_replicat):
            df = pd.read_csv("../CSV_FILE/GLOBAL/Results_" + str(y) + ".csv")
            DFUN = df.drop_duplicates(subset=problem["names"][i])
            DFUN = DFUN.sort_values(by=problem["names"][i])
            STV = STV + ((DFUN[problem["names"][id_output]].to_numpy() - moy) * (
                        DFUN[problem["names"][id_output]].to_numpy() - moy))

        STV = np.sqrt(STV / (nb_replicat - 1))
        plt.plot(DFUN[problem["names"][i]].to_numpy(), moy, label="moy", color="red")
        STVmax = moy + STV
        STVmin = moy - STV
        plt.fill_between(DFUN[problem["names"][i]].to_numpy(), STVmin, STVmax, color="plum", linewidth=0.1, label="STD",
                         alpha=0.7)
        plt.xlabel(problem["names"][i])
        plt.ylabel(problem["names"][id_output])
        plt.legend(loc="best")

        if not os.path.exists("../CSV_FILE/GLOBAL/Analysis_" + problem["names"][id_output]):
            os.makedirs("../CSV_FILE/GLOBAL/Analysis_" + problem["names"][id_output])

        plt.savefig("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]+"/Graph_Input_All_Simu_" + problem["names"][i] + ".png")
        plt.clf()
    print("===  Chart Save...\n")

#Fonctionne, a supprimer si l'autre marche
'''
def courbe_simu(id_output):
    print("=== Creating chart...\n")
    for i in range(0, sample):
        #df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_0/replicat_0.csv")
        moy = np.zeros(len(time))
        for y in range(0, nb_replicat):
            df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_" + str(i) + "/replicat_" + str(y) + ".csv")
            moy = moy + df[problem["names"][id_output]]
            if y == 0:
                plt.plot(df["cycle"].to_numpy(), df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5, ls="--",
                         label="replicat")
            else:
                plt.plot(df["cycle"].to_numpy(), df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5, ls="--")
        moy = moy / nb_replicat
        plt.plot(df["cycle"].to_numpy(), moy, c="red", lw=1.5, ls="--", label="MOY")
        plt.xlabel("time")
        plt.ylabel(problem["names"][id_output])
        plt.title("SAMPLE_" + str(i))
        plt.legend(loc="best")
        plt.savefig("../CSV_FILE/SAMPLE/SAMPLE_" + str(i) + "/"+problem["names"][id_output]+"/GRAPH.png")
        plt.clf()
    print("===  Chart Save...\n")
'''

#A tester
def courbe_simu(id_output):
    print("=== Creating chart...\n")
    for i in range(0, sample):
        moy = np.zeros(len(time))
        file_names=[]
        for name in os.listdir("../CSV_FILE/SAMPLE/"):
            if os.path.isdir("../CSV_FILE/SAMPLE/"+name):
                file_names.append(name)

        for file in file_names:
            list_replicat = os.path.join("../CSV_FILE/SAMPLE/"+file+"/", "*.csv")
            list_replicat = glob.glob(list_replicat)

            first_one=True
            for replicat in list_replicat:
                df = pd.read_csv("../CSV_FILE/SAMPLE/"+file+"/replicat_"+replicat)
                moy = moy + df[problem["names"][id_output]]
                if first_one:
                    plt.plot(df["cycle"].to_numpy(), df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                             ls="--",
                             label="replicat")
                    first_one=False
                else:
                    plt.plot(df["cycle"].to_numpy(), df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                             ls="--")
            moy = moy / nb_replicat
            plt.plot(df["cycle"].to_numpy(), moy, c="red", lw=1.5, ls="--", label="MOY")
            plt.xlabel("time")
            plt.ylabel(problem["names"][id_output])
            plt.title("SAMPLE_" + str(i))
            plt.legend(loc="best")

            plt.savefig("../CSV_FILE/SAMPLE/"+file+"/"+problem["names"][id_output]+"/GRAPH.png")
            plt.clf()
    print("===  Chart Save...\n")

#Fonctionne, a supprimer si l'autre marche
'''
def heatmap_time(id_output):
    print("=== Creating chart...\n")
    for z in range(0,output):
        df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
        df=df.drop_duplicates(subset=problem["names"][z])
        res = np.zeros(shape= (len(df),time))
        df = df.sort_values(by=problem["names"][z])
        l=df.index
        moy=np.zeros(time)
        tmp=0
        for h in l:

            for y in range(0,nb_replicat):
                df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_" + str(h) + "/replicat_"+str(y)+".csv")
                moy=moy+df[problem["names"][id_output]]
            moy=moy/nb_replicat
            res[tmp]=moy
            tmp=tmp+1
        times=list(range(0,time))
        df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
        df = df.sort_values(by=problem["names"][z])
        fig, ax = plt.subplots()
        c = ax.pcolormesh(times, df[problem["names"][z]].unique(), res, cmap='RdBu_r', vmin=0, vmax=1)
        ymin=np.amin(df[problem["names"][z]].to_numpy())
        ymax=np.amax(df[problem["names"][z]].to_numpy())
        ax.axis([0, time, ymin, ymax])
        fig.colorbar(c, ax=ax,label=problem["names"][id_output])
        plt.xlabel("time")
        plt.ylabel(problem["names"][z])
        plt.savefig("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]+"/" +problem["names"][z] + "_HEAT_MAP.png")
        plt.clf()
    print("===  Chart Save...\n")
  '''

#A tester
def heatmap_time(id_output):
    print("=== Creating chart...\n")
    for z in range(0,output):
        df_ini = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
        df_ini=df_ini.drop_duplicates(subset=problem["names"][z])
        res = np.zeros(shape= (len(df_ini),time))
        df_ini = df_ini.sort_values(by=problem["names"][z])
        l=df_ini.index
        moy=np.zeros(time)
        tmp=0
        for h in l:
            txt=''
            for o in range(0,output-1):
                txt=txt+str(df_ini.get_value(h,problem["names"][o]))+"_"
            txt=txt+str(df_ini.get_value(h,problem["names"][output-1]))

            list_replicat = os.path.join("../CSV_FILE/SAMPLE/SAMPLE_" + txt + "/", "*.csv")
            list_replicat = glob.glob(list_replicat)

            for rep in list_replicat:
                df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_" +txt+"/"+rep)
                moy=moy+df[problem["names"][id_output]]
            moy=moy/nb_replicat
            res[tmp]=moy
            tmp=tmp+1
        times=list(range(0,time))
        df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
        df = df.sort_values(by=problem["names"][z])
        fig, ax = plt.subplots()
        c = ax.pcolormesh(times, df[problem["names"][z]].unique(), res, cmap='RdBu_r', vmin=0, vmax=1)
        ymin=np.amin(df[problem["names"][z]].to_numpy())
        ymax=np.amax(df[problem["names"][z]].to_numpy())
        ax.axis([0, time, ymin, ymax])
        fig.colorbar(c, ax=ax,label=problem["names"][id_output])
        plt.xlabel("time")
        plt.ylabel(problem["names"][z])

        if not os.path.exists("../CSV_FILE/GLOBAL/Analysis_" + problem["names"][id_output]):
            os.makedirs("../CSV_FILE/GLOBAL/Analysis_" + problem["names"][id_output])

        plt.savefig("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]+"/" +problem["names"][z] + "_HEAT_MAP.png")
        plt.clf()
    print("===  Chart Save...\n")
'''
#######################################################################################
#######################################################################################
##############################      FOR OFAT x2       #################################
#######################################################################################
#######################################################################################
'''

#Fonctionne, a supprimer si l'autre marche
'''
def heatmap_time_2_factor_Not_Defined_Cluster(cluster,id_output):
    print("=== Creating chart...\n")

    for w in range(0,output-1):

        for n in range(w+1,output):
            if w!=n:
                df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
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

                df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
                df.sort_values(by=problem["names"][n])
                ValUn_2 = df[problem["names"][n]].unique()
                NbVal_2 = floor(len(ValUn_2) / cluster)
                val_2 = []u
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
                    df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
                    df = df.drop_duplicates(subset=problem["names"][w])
                    res = np.zeros(shape=(cluster, time))
                    tmp_moy = np.zeros(shape=(cluster, time))

                    for u in val[i]:
                        for y in range(0, cluster):
                            moy = np.zeros(time)
                            for v in val_2[y]:
                                df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
                                df = df.loc[df[problem["names"][w]] == u]
                                df = df.loc[df[problem["names"][n]] == v]
                                df = df[df[problem["names"][w]] != np.nan]
                                if (len(df) == 1):
                                    l = df.index
                                    for h in l:

                                        for z in range(0, nb_replicat):
                                            df = pd.read_csv(
                                                "../CSV_FILE/SAMPLE/SAMPLE_" + str(h) + "/replicat_" + str(z) + ".csv")
                                            moy = moy + df[problem["names"][id_output]]
                                            axs[y, i].plot(df["cycle"].to_numpy(),
                                                           df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                                                           ls="--")
                            moy = moy / (len(val_2[y]) * nb_replicat)
                            tmp_moy[y] = tmp_moy[y] + moy

                    for m in range(0,cluster):
                        res[m] = tmp_moy[m] / (len(val[i]))

                    df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
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
                plt.savefig("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]+"/" + problem["names"][w] +"_"+problem["names"][n] +"_HEAT_MAP_2.png")
                plt.clf()
    print("===  Chart Save...\n")
'''
#A tester
def heatmap_time_2_factor_Not_Defined_Cluster(cluster,id_output):
    print("=== Creating chart...\n")

    for w in range(0,output-1):

        for n in range(w+1,output):
            if w!=n:
                df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
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

                df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
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
                    df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
                    df = df.drop_duplicates(subset=problem["names"][w])
                    res = np.zeros(shape=(cluster, time))
                    tmp_moy = np.zeros(shape=(cluster, time))

                    for u in val[i]:
                        for y in range(0, cluster):
                            moy = np.zeros(time)
                            for v in val_2[y]:
                                df_ini = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
                                df_ini = df_ini.loc[df_ini[problem["names"][w]] == u]
                                df_ini = df_ini.loc[df_ini[problem["names"][n]] == v]
                                df_ini = df_ini[df_ini[problem["names"][w]] != np.nan]
                                if (len(df) == 1):
                                    l = df_ini.index
                                    for h in l:
                                        txt = ''
                                        for o in range(0, output - 1):
                                            txt = txt + str(df_ini.get_value(h, problem["names"][o])) + "_"
                                        txt = txt + str(df_ini.get_value(h, problem["names"][output - 1]))

                                        list_replicat = os.path.join("../CSV_FILE/SAMPLE/SAMPLE_" + txt + "/",
                                                                         "*.csv")
                                        list_replicat = glob.glob(list_replicat)

                                        for rep in list_replicat:
                                            df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_" + txt + "/" + rep)
                                            moy = moy + df[problem["names"][id_output]]
                                            axs[y, i].plot(df["cycle"].to_numpy(),
                                                           df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                                                           ls="--")
                            moy = moy / (len(val_2[y]) * nb_replicat)
                            tmp_moy[y] = tmp_moy[y] + moy

                    for m in range(0,cluster):
                        res[m] = tmp_moy[m] / (len(val[i]))

                    df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
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

                if not os.path.exists("../CSV_FILE/GLOBAL/Analysis_" + problem["names"][id_output]):
                    os.makedirs("../CSV_FILE/GLOBAL/Analysis_" + problem["names"][id_output])

                plt.savefig("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]+"/" + problem["names"][w] +"_"+problem["names"][n] +"_HEAT_MAP_2.png")
                plt.clf()
    print("===  Chart Save...\n")


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



def build_chart_prio_facteur(problem,morris,sobolT,sobol,id_output):
    print("=== Creating chart...\n")
    rank=do_ranking_global(morris,sobolT)
    print(rank)
    data=[]
    for i in rank :
        i=int(i)
        data.append([problem["names"][i],morris[i],sobol[i],sobolT[i]])

    df = pd.DataFrame(data, columns=["Inputs", "Morris", "Sobol 1","Sobol T"])
    df.plot.barh(x="Inputs", y=["Morris", "Sobol 1","Sobol T"], color={"darkorange","cornflowerblue","darkred"},title="Inputs Importance Chart")
    plt.xlim(0,1)

    if not os.path.exists("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]):
        os.makedirs("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output])

    plt.savefig("../CSV_FILE/GLOBAL/Analysis_"+problem["names"][id_output]+"/Graph_Prio_Factor.png")
    plt.clf()
    print("===  Chart Save...\n")

def morris_fit_data(morris):
    momo=morris / np.sqrt(np.sum(morris**2))
    return momo

def chart3D(Innames,Outnames,result,AnalyseName):
    print("=== Creating chart...\n")

    result = np.array(result)


    colors=['r', 'b', 'g', 'y', 'b', 'p']
    fig = plt.figure(figsize=(8, 8), dpi=250)
    ax1 = fig.add_subplot(111, projection='3d')
    #ax1.set_xlabel('Parameters', labelpad=10)
    ax1.set_ylabel('Outputs', labelpad=10)
    ax1.set_zlabel(AnalyseName)

    xlabels=np.array(Innames)
    xpos = np.arange(xlabels.shape[0])
    ylabels= np.array(Outnames)
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = result
    zpos = zpos.ravel()


    dx = 0.5
    dy = 0.4
    dz = zpos


    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)


    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
    ax1.set_zlim(0,1)

    plt.savefig("../CSV_FILE/GLOBAL/3DGraph_Analysis.png")
    plt.clf()
    print("===  Chart Save...\n")

def analyseResult3D(morris,sobol):
    parameterNames=[]
    OutputNames=[]
    for i in range(0,len(problem["names"])):
        if i<output:
            parameterNames.append(problem["names"][i])
        else:
            OutputNames.append((problem["names"][i]))

    chart3D(parameterNames,
            output, [morris], "Analyse")

#A tester
def Cluster_time(id_output):
    print("=== Creating chart...\n")
    df = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
    nb_cluster = df["cluster"].unique()
    fig, axs = plt.subplots(1, nb_cluster)

    for i in range(0,nb_cluster):
        df_ini = pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
        df_ini = df_ini.loc[df_ini["cluster"] == i]
        df_ini = df_ini[df_ini["cluster"] != np.nan]
        l = df_ini.index
        moy = np.zero(time)
        for y in l:

            txt = ''
            for o in range(0, output - 1):
                txt = txt + str(df_ini.get_value(y, problem["names"][o])) + "_"
            txt = txt + str(df_ini.get_value(y, problem["names"][output - 1]))

            list_replicat = os.path.join("../CSV_FILE/SAMPLE/SAMPLE_" + txt + "/", "*.csv")
            list_replicat = glob.glob(list_replicat)

            for rep in list_replicat:
                df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_" + txt + "/" + rep)
                moy = moy + df[problem["names"][id_output]]
                axs[0, i].plot(df["cycle"].to_numpy(),
                               df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                               ls="--")

        moy = moy / (len(l) * nb_replicat)

        times = list(range(0, 502))
        axs[0, i].plot(times, moy, c="red", lw=1.5, ls="--")
        axs[0, i].axis([0, time, 0, 1])
    plt.show()
    print("===  Chart Save...\n")


if __name__ == '__main__':




    problem = readProblem("../PRE-TRAITEMENT/model.txt")
    df=pd.read_csv("../CSV_FILE/GLOBAL/Results_0.csv")
    problem["names"]=df.columns
    output=problem["num_vars"]

    #Parameters to change

    nb_replicat=4
    sample=204
    time=502
    cluster=6

    morris=[0.497743,0.482880,0.02996,0.465042]
    morris=np.array(morris)
    morris=morris_fit_data(morris)


    sobol=[0.187035,0.106427,0.003549,0.372382]
    sobolT = [0.398204, 0.236124, 0.000781, 0.52928]

    morris2=[0.497743,0.482880,0.02996,0.465042]
    morris2=np.array(morris2)
    morris2=morris_fit_data(morris2)


    sobol2=[0.187035,0.106427,0.003549,0.372382]
    sobolT2 = [0.398204, 0.236124, 0.000781, 0.52928]
    build_chart_prio_facteur(problem,morris,sobolT,sobol,output)



    #chart3D(["Infection Probability","Probability Dodge Disease","Nb initial infected","Probability to cure"],["Infection Rate","Test"],[sobolT,morris],"Analysis")
