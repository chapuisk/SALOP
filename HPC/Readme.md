# Readme for HPC 
## Context
In this Readme, you will find all information you need for the use of "GenerateXML_Sensitivity.py"

This python script generate XML file that can be run with gama-headless for simulation.
This script build a Morris or Saltelli sampling for a sensitivity analysis.
This script was think to be use in a HPC (Hight Performance Computing).

This folder is made up of two others folder.
- The folder INPUTS where you can generate the XML file 
- The folder MODEL where you can put your GAMA model (.gaml)

## Prerequisite

- pip install numpy
- pip install scipy
- pip install matplotlib
- pip install SALib


## Use
This is the command to use with "GenerateXML_Sensitivity.py"

### Standard command:

***python3 GenerateXML_Sensitivity.py [option] -f [number of max cycles] -xml [name of the experiment] [path to the .gaml] [path to the xml]***

### Option:
- -r, --replication : Number of replication for each parameter space (default: 1)
- -s, --split : Number of machines (or nodes for HPC) (default: 1)
- -S, --seed : Starting value for seeding simulation (default: 0)
- -sample : Number of samples (default:128)
- -analysis : Method analysis (default:sobol), can be: sobol/morris

### Example for Morris Analysis:

***python3 GenerateXML_Sensitivity.py -s 5 -sample 400 -f 3000 -analysis morris -xml experiment ./MODEL/my/model/to/analyse.gaml  ./INPUTS/inputs.xml***

