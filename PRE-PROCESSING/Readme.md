# Readme for PRE-PRETRAITEMENT
## Context
In this Readme, you will find all information you need for the use of "Stochasticity_Analysis.py" , "BuildModelFile.py" and "GenerateXML_CSV.py"

***
#Stochasticity Analysis
This python script perform a Stochasticity Analysis to found the minimum number of replicat needed to neutralize the stochasticity.

This script use three different methods.
- Coefficient Variation method use a threshold
- Standard Error method use a threshold (with a percent)
- Student law method

This script save for each method a chart with the results and all curves

## Prerequisite
- pip install matplotlib
- pip install pandas
- pip install numpy

## Use
***python3 Stochasticity_Analysis.py -r INT -s INT -problem /path/to/problem.txt -data /path/to/data -output /path/to/output.csv***

### Option:
- -h,--threshold : Threshold value for the number of min replicat (Default:Perform 0.001,0.05,0.01)
## Example for Sobol analysis
***python3 Stochasticity_Analysis.py -r 8 -s 100 -problem /path/to/problem.txt -data /path/to/data -output /path/to/output.csv***

***
# Model Builder file
This python script allow to build the model.txt file that contains all information necessary for the other scripts.
This script save the model at the given path

## Prerequisite
- pip install numpy

## Use
***python3 BuildModelFile.py -xml <experiment name> /path/to/file.gaml -o /path/to/model.txt***
## Example:
***python3 BuildModelFile.py -xml Sobol /path/to/file.gaml -o /path/to/model.txt***
***
# GenerateXML CSV
This python script generate XML files from CSV data.

## Prerequisite
- pip install pandas
- pip install numpy

## Use
***python3 GenerateXML_CSV.py [options] -f INT -c /path/to/CSV.csv -xml <experiment name> /path/to/file.gaml /path/to/file.xml***
### Option:
- -r, --replication : Number of replication for each parameter space (default: 1)
- -s, --split : Number of machines (or nodes for HPC) (default: 1)
- -S, --seed : Starting value for seeding simulation (default: 0)
