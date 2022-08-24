# Readme for Stochasticity Analysis
## Context
In this Readme, you will find all information you need for the use of "Stochasticity_Analysis.py"

This python script perform a Stochasticity Analysis to found the minimum number of replicat needed to neutralize the stochasticity.


## Prerequisite
- pip install matplotlib
- pip install pandas
- pip install numpy

## Use
***python3 Stochasticity_Analysis.py -r INT -s INT -problem /path/to/problem.txt -data /path/to/data -output /path/to/output.csv***

### Option:
- -h,--threshold : Threshold value for the number of min replicat (Default:0.01)
## Example for Sobol analysis
***python3 Stochasticity_Analysis.py -r 8 -s 100 -problem /path/to/problem.txt -data /path/to/data -output /path/to/output.csv***
