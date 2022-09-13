# SALOP
This git repository purpose is to perform Sensitivity Analysis with LOw a Priori of an agent based model using [Gama platform](gama-platform.org/). It also provides tools to run your model with HPC (High Performance Computing).  

## Prerequisite
- Last version of **Gama Dev**
`git clone https://github.com/gama-platform/gama.git`

***
# Use
All you need to use py and gaml scripts is described in a Readme.md file in each sub folder of the project. You can find below the classic pipeline to perform sensitivity analysis on a model.

## Sobol & Morris
The first step of sensitivity analysis should be Sobol and Morris since they provide information about the impact of an input variable on the variance of the output of your model. Because of performance and execution time issues, you should run your model headless.  

Find below the steps needed to compute Sobol and Morris indices :

- **experiment -** First you should write a GUI experiment in your model with each input parameters minimum and maximum values information. To retrieve your run in headless mode don't forget to save the input and output of your experiment in a .csv file as column at the end of your experiment.
- **sampling -** Then you need to generate XML files describing the parameters of your model, so you can run your experiment headless later on using the headless script provided by Gama. To do so, use [GenerateXML_Sensitivity.py](HPC/GenerateXML_Sensitivity.py) script with Sobol or Morris sampling.  
**Warning :** Choose wisely the number of sample for your sampling depending on the execution and ressources needed to run your experiment. The higher the sample value the higher the accuracy of your analysis but the lower the performances.  
$n_S = N \times (2 \times P + 2)$ and $n_M = N \times P$ with  
  -- $n_S$ and $n_M$ The number of experiments ran by Sobol and Morris  
  -- $N$ the sampling value (should be a power of 2 for Sobol and an even number for Morris)  
  -- $P$ the number of input parameters of the model  

- **headless -** Run headless the experiment created in step 1 thanks to the XML files of step 2 using the _gama_headless.sh_ bash script provided with Gama.
- **analysis -** Use Gama operators **morrisAnalysis** and **sobolAnalysis** to compute Sobol and Morris indices thanks to the output .csv file of your experiments.

## Stochasticity Analysis
The stochasticity analysis is required to lower the influence of randomness of your model on the analysis. The goal is to find the minimum number of replicates to get a nice representation of your model behaviour. To perform Stochasticity Analysis use [Stochasticity_Analysis.py](PRE-PROCESSING/Stochasticity_Analysis.py) script

## Finding subspaces of interest
The goal of sensitivity analysis is to map the output of your model against your inputs. That's why you need to find subspaces of interest of same behaviour in your inputs space. This git repository provides some tools such as Surrogate models, Clustering and multi-objective optimisation methods to find those sub spaces. You can output **exploration boxes** in a report to read directly suspaces of interest.

### Surrogate models
Two surrogate models are available in this repository. First a **multi linear regression model** for simple models and then a **CART model** (based on Regression tree) for more complex models. You can find implementation of thoses regression model in [SURROGATE-MODELS](SURROGATE-MODELS) folder. Note that these files are .gaml files that you should use directly in Gama.  
**Warning :** To use CART algorithm you need weka plugin. To add weka plugin to Gama developer version you need to go to [Gama experimental github](https://github.com/gama-platform/gama.experimental) and download the two folders **idees.gama.features.weka** and **idees.gama.weka**. Then in eclipse you should open those two folders as project. Finally, open run configuration menu and add weka to plugins section.

### Clustering and multi-objective optimisation
Two clustering method [Kmeans](CLUSTERING/kmeans.py) and [som](CLUSTERING/som.py) and one multi-objective optimisation method [pareto.py](CLUSTERING/pareto.py) are available in this repository. We also provide some tools to visualize clusters and information on the inputs of each clusters. 

## OFAT
## Visualization
