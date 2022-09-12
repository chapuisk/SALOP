# Readme for Clustering
## Context
In this Readme, you will find all information you need for to use clustering methods.
In the case of sensitivity analysis, clustering method should be used to find subspaces of interest to explore in which
the simulation behavior is similar.

## Prerequisite
- pip install pandas
- pip install numpy
- pip install scikit-learn
- pip install sklearn-som

## Use
To use clustering methods you need an input .csv file which contains inputs and outputs of your model as columns.
In general, you'll need to use the _load\_data_ method to load your .csv file. Then you can choose which algorithm 
you'll use to cluster your data. Once your data is labeled with the corresponding clusters you can use different 
visualization methods

- **data** folder contains an example dataset.
- **results** folder contains the output graphics of the algorithms below.
- **kmeans.py**, **som.py** and **pareto.py** contain example of use of available algorithm to cluster data.  
- **utils.py** contains common data loading and visualization methods.
