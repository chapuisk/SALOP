# Readme for Post-Traitement
## Context
In this Readme, you will find all information you need for the use of "merge_results.py"

This python script merge and sort all CSV files in a folder.

This script is used to merge each CSV file generate by each machine/nodes.


## Prerequisite
- pip install pandas

## Use
***python3 merge_results.py -d /path/to/data/folder -o /path/to/output.csv***

### Option:
- -s, --sort : Name of the column to sort by (Default: number)
## Example 
***python3 merge_results.py -d ../CSV_FILE/TO_MERGE -o ../CSV_FILE/GLOBAL/RESULT_0.csv***
