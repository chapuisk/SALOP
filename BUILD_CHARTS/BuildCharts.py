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
                      |_ SAMPLE_PARAM_X_...
                                 |_ replicat_0.csv
                                 |_ replicat_1.csv
                                 |_ ...
                      |_ SAMPLE_PARAM_Y_...
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
        df = pd.read_csv(path_df+"GLOBAL/Results_" + str(i) + ".csv")
        DFUN = df.drop_duplicates(subset=problem["names"][i])
        DFUN = DFUN.sort_values(by=problem["names"][i])

        list = DFUN[DFUN[problem["names"][i]] == default_value[i]].index.tolist()
        DFUN=DFUN.drop(list)

        moy = np.zeros(len(DFUN))
        STV = np.zeros(len(DFUN))
        STVmax = np.zeros(len(DFUN))
        STVmin = np.zeros(len(DFUN))
        for y in range(0, nb_replicat):
            df = pd.read_csv(path_df+"GLOBAL/Results_" + str(y) + ".csv")
            DFUN = df.drop_duplicates(subset=problem["names"][i])
            DFUN = DFUN.sort_values(by=problem["names"][i])

            list = DFUN[DFUN[problem["names"][i]] == default_value[i]].index.tolist()
            DFUN = DFUN.drop(list)


            DFUN = DFUN.reset_index(drop=True)
            moy = moy + DFUN[problem["names"][id_output]]


            plt.plot(DFUN[problem["names"][i]].to_numpy(), DFUN[problem["names"][id_output]].to_numpy(), 'o', color='grey',
                     alpha=0.3)
        moy = moy.replace(np.nan,0)
        moy = moy / nb_replicat


        for y in range(0, nb_replicat):
            df = pd.read_csv(path_df+"GLOBAL/Results_" + str(y) + ".csv")
            DFUN = df.drop_duplicates(subset=problem["names"][i])
            DFUN = DFUN.sort_values(by=problem["names"][i])

            list = DFUN[DFUN[problem["names"][i]] == default_value[i]].index.tolist()
            DFUN = DFUN.drop(list)

            STV = STV + (DFUN[problem["names"][id_output]].to_numpy() - moy)**2
        STV = np.sqrt(STV / (nb_replicat - 1))
        plt.plot(DFUN[problem["names"][i]].to_numpy(), moy, label="moy", color="red")
        STVmax = moy + STV
        STVmin = moy - STV
        plt.fill_between(DFUN[problem["names"][i]].to_numpy(), STVmin, STVmax, color="plum", linewidth=0.1, label="STD",
                         alpha=0.7)
        plt.xlabel(problem["names"][i])
        plt.ylabel(problem["names"][id_output])
        plt.legend(loc="best")
        if not os.path.exists(path_df+"GLOBAL/Analysis_" + problem["names"][id_output]):
            os.makedirs(path_df+"GLOBAL/Analysis_" + problem["names"][id_output])
        plt.savefig(path_df+"GLOBAL/Analysis_"+problem["names"][id_output]+"/Graph_Input_All_Simu_" + problem["names"][i] + ".png")
        plt.clf()
    print("===  Chart Save...\n")

def courbe_simu(id_output):
    print("=== Creating chart...\n")
    file_names=[]
    for name in os.listdir(path_df+"SAMPLE/"):
        if os.path.isdir(path_df+"SAMPLE/"+name):
            file_names.append(name)
    for file in file_names:
        moy = np.zeros(time)
        list_replicat = os.path.join(path_df+"SAMPLE/"+file+"/", "*.csv")
        list_replicat = glob.glob(list_replicat)
        first_one=True
        for replicat in list_replicat:
            df = pd.read_csv(replicat)
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
        plt.title(file)
        plt.legend(loc="best")
        if not os.path.exists(path_df + "SAMPLE/"+file+"/" + problem["names"][id_output]):
            os.makedirs(path_df + "SAMPLE/"+file+"/" + problem["names"][id_output])
        plt.savefig(path_df+"SAMPLE/"+file+"/"+problem["names"][id_output]+"/GRAPH.png")
        plt.clf()
    print("===  Chart Save...\n")

def heatmap_time(id_output):
    print("=== Creating chart...\n")
    for z in range(0,output):
        df_ini = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
        df_ini=df_ini.drop_duplicates(subset=problem["names"][z])
        res = np.zeros(shape= (len(df_ini),time))
        df_ini = df_ini.sort_values(by=problem["names"][z])

        lists = df_ini[df_ini[problem["names"][z]] == default_value[z]].index.tolist()
        df_ini = df_ini.drop(lists)

        l=df_ini.index
        moy=np.zeros(time)
        tmp=0
        for h in l:
            txt=''
            for o in range(0,output-1):
                txt=txt+str(df_ini._get_value(h,problem["names"][o]))+separator
            txt=txt+str(df_ini._get_value(h,problem["names"][output-1]))
            list_replicat = os.path.join(path_df+"SAMPLE/SAMPLE_" + txt + "/", "*.csv")
            list_replicat = glob.glob(list_replicat)
            for rep in list_replicat:
                #df = pd.read_csv("../CSV_FILE/SAMPLE/SAMPLE_" +txt+"/"+rep)
                df = pd.read_csv(rep)
                moy=moy+df[problem["names"][id_output]]
            moy=moy/nb_replicat
            res[tmp]=moy
            tmp=tmp+1
        times=list(range(0,time))
        df = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
        df = df.sort_values(by=problem["names"][z])
        fig, ax = plt.subplots()
        c = ax.pcolormesh(times, df[problem["names"][z]].unique(), res, cmap='RdBu_r', vmin=0, vmax=1)
        ymin=np.amin(df[problem["names"][z]].to_numpy())
        ymax=np.amax(df[problem["names"][z]].to_numpy())
        ax.axis([0, time, ymin, ymax])
        fig.colorbar(c, ax=ax,label=problem["names"][id_output])
        plt.xlabel("time")
        plt.ylabel(problem["names"][z])
        if not os.path.exists(path_df+"GLOBAL/Analysis_" + problem["names"][id_output]):
            os.makedirs(path_df+"GLOBAL/Analysis_" + problem["names"][id_output])

        plt.savefig(path_df+"GLOBAL/Analysis_"+problem["names"][id_output]+"/" +problem["names"][z] + "_HEAT_MAP.png")
        plt.clf()
    print("===  Chart Save...\n")
'''
#######################################################################################
#######################################################################################
##############################      FOR OFAT x2       #################################
#######################################################################################
#######################################################################################
'''

#A tester
def heatmap_time_2_factor_Not_Defined_Cluster(cluster,id_output):
    print("=== Creating chart...\n")
    for w in range(0,output-1):
        for n in range(w+1,output):
            if w!=n:
                df = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
                df.sort_values(by=problem["names"][w])

                lists = df[df[problem["names"][w]] == default_value[w]].index.tolist()
                df = df.drop(lists)

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
                df = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
                df.sort_values(by=problem["names"][n])

                lists = df[df[problem["names"][n]] == default_value[n]].index.tolist()
                df = df.drop(lists)

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
                    df = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
                    df = df.drop_duplicates(subset=problem["names"][w])
                    res = np.zeros(shape=(cluster, time))
                    tmp_moy = np.zeros(shape=(cluster, time))
                    for u in val[i]:
                        for y in range(0, cluster):
                            moy = np.zeros(time)
                            for v in val_2[y]:
                                txt = ''
                                for o in range(0, output ):

                                    if(o==w):

                                        txt = txt +str(u)
                                    else:
                                        if(o==n):

                                            txt = txt +str(v)
                                        else:

                                            txt = txt + str(default_value[o]) +""
                                    if(o!=output-1):
                                        txt= txt +separator

                                list_replicat = os.path.join(path_df+"SAMPLE/SAMPLE_" + txt + "/",
                                                                         "*.csv")
                                list_replicat = glob.glob(list_replicat)

                                for rep in list_replicat:
                                    df = pd.read_csv( rep)
                                    moy = moy + df[problem["names"][id_output]]
                                    axs[y, i].plot(df["cycle"].to_numpy(),
                                                           df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                                                           ls="--")
                            moy = moy / (len(val_2[y]) * nb_replicat)
                            tmp_moy[y] = tmp_moy[y] + moy
                    for m in range(0,cluster):
                        res[m] = tmp_moy[m] / (len(val[i]))
                    df = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
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
                    times = list(range(0, time))
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
                if not os.path.exists(path_df+"GLOBAL/Analysis_" + problem["names"][id_output]):
                    os.makedirs(path_df+"GLOBAL/Analysis_" + problem["names"][id_output])
                plt.savefig(path_df+"GLOBAL/Analysis_"+problem["names"][id_output]+"/" + problem["names"][w] +"_"+problem["names"][n] +"_HEAT_MAP_2.png")
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
    momo=abs(morris) / np.sqrt(np.sum(morris**2))
    return momo


#METHOD NOT FINISH
def chart3D(Innames,Outnames,result,AnalyseName):
    print("=== Creating chart...\n")
    result = np.array(result)
    colors=['r', 'b', 'g', 'y', 'b', 'p']
    fig = plt.figure(figsize=(8, 8), dpi=250)
    ax1 = fig.add_subplot(111, projection='3d')
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

#Need to be tested
def Cluster_time(id_output):
    print("=== Creating chart...\n")
    df = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
    nb_cluster = df["cluster"].unique()
    fig, axs = plt.subplots(1, nb_cluster)
    for i in range(0,nb_cluster):
        df_ini = pd.read_csv(path_df+"GLOBAL/Results_0.csv")
        df_ini = df_ini.loc[df_ini["cluster"] == i]
        df_ini = df_ini[df_ini["cluster"] != np.nan]
        l = df_ini.index
        moy = np.zero(time)
        for y in l:
            txt = ''
            for o in range(0, output - 1):
                txt = txt + str(df_ini.get_value(y, problem["names"][o])) + separator
            txt = txt + str(df_ini.get_value(y, problem["names"][output - 1]))
            list_replicat = os.path.join(path_df+"SAMPLE/SAMPLE_" + txt + "/", "*.csv")
            list_replicat = glob.glob(list_replicat)
            for rep in list_replicat:
                df = pd.read_csv(path_df+"SAMPLE/SAMPLE_" + txt + "/" + rep)
                moy = moy + df[problem["names"][id_output]]
                axs[0, i].plot(df["cycle"].to_numpy(),
                               df[problem["names"][id_output]].to_numpy(), c="grey", lw=0.5,
                               ls="--")
        moy = moy / (len(l) * nb_replicat)
        times = list(range(0, time))
        axs[0, i].plot(times, moy, c="red", lw=1.5, ls="--")
        axs[0, i].axis([0, time, 0, 1])
    plt.show()
    print("===  Chart Save...\n")


if __name__ == '__main__':

    #Path to data folder
    path_df="../CSV_FILE/Results_OFATx2/"

    separator=""

    #Path to problem file
    problem = readProblem("../PRE-PROCESSING/model.txt")


    df=pd.read_csv(path_df+"GLOBAL/Results_0.csv")
    problem["names"]=df.columns
    output=problem["num_vars"]

    #Parameters to change
    nb_replicat=15
    sample=2646
    time=502
    cluster=6
    id_output=4
    default_value = [0.05, 0.025, 5, 0.001]

    morris=[]
    morris=np.array(morris)
    morris=morris_fit_data(morris)

    sobol=[]
    sobol=np.array(sobol)
    sobol=morris_fit_data(sobol)

    sobolT=[]
    sobolT=np.array(sobolT)
    sobolT=morris_fit_data(sobolT)

    #Function to call

    #courbe_point(4)
    #courbe_simu(4)
    #heatmap_time(4)
    #heatmap_time_2_factor_Not_Defined_Cluster(cluster,id_output)
    #build_chart_prio_facteur(problem,morris,sobolT,sobol,output)

    #Not working yet
    #chart3D(["Infection Probability","Probability Dodge Disease","Nb initial infected","Probability to cure"],["Infection Rate","Test"],[sobolT,morris],"Analysis")