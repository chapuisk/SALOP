# Readme for Clustering
## Context
In this Readme, you will find all information you need for to use surrogate models.  
The surrogate models files are .gaml file and should be loaded with Gama. The different surrogate models available are
multi linear model and CART. They are a few other models that can be used because they are available in the gama plugin
**weka** but haven't been tested (smo, rbf, gaussian...)

## Prerequisite

- At least GAMA 1.8.2 and Gama plugin **weka**
(see official Gama [website](https://gama-platform.org/) to download Gama and load the plugin **weka**)


## Use
Open _weka_regression.gaml_ or _multi_linear_regression.gaml_ in gama. 
To use those model you just need to update the init block with the values corresponding to your model and then run 
the main methode. All information you need is recall in the header of the files.
