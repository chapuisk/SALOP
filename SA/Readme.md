# Readme for Sensitivity Analysis
## Context
In this Readme, you will find all information you need for the use of "sensitivity_analysis.py"

This python script perform a sensitivity analysis according to a chosen method.
This script has two different methods:
- Morris Analysis
- Sobol Analysis


## Prerequisite
- pip install numpy
- pip install scipy
- pip install matplotlib
- pip install SALib

## Use

***python3 sensitivity_analysis.py [option] -analysis analysis /path/to/problem.txt /path/to/data.csv -output /path/to/output.txt***

### Option:
- -i, --id_output : The id of the output to analyse (Default:0)

## Example for Sobol analysis

***python3 sensitivity_analysis.py -i 1 -analysis sobol ../HPC/model_problem_analysis.txt ../CSV_FILE/GLOBAL/my_data_0.csv -output ./RESULTS/Sobol_Analysis.txt***

