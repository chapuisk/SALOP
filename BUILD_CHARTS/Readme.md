# Readme for BUILD_CHARTS
## Context
In this Readme, you will find all information you need for the use of "BUILD_CHARTS.py"

In this script, you will find all the necessary method to build charts using different sampling.
There is charts for OFAT, OFATx2, heatmap and pareto diagram. 
To build this charts, you just need to have your data in the folder CSV_FILE.

In this script you will find many methods allowing to draw graphics and to plot data on charts.
This script use relative path for the access to data. Please follow this structure.

|_ BUILD_CHARTS  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ BuildCharts.py  
|_ CSV_FILE  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ GLOBAL  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ Result_0.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ Result_1.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ SAMPLING  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ SAMPLE_PARAM_X_...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ replicat_0.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ replicat_1.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ SAMPLE_PARAM_Y_...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ ...

## Prerequisite
- pip install numpy
- pip install pandas
- pip install matplotlib

## Use

You have to code a main and call functions that you want to use.
  
