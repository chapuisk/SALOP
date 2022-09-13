# Readme for Post-Processing
## Context
In this Readme, you will find all information you need for the use of "merge_results.py" and "RenameFile.py"
***
# Merge Results
This python script merge and sort all CSV files in a folder.

This script is used to merge each CSV file generate by each machine/nodes.


## Prerequisite
- pip install pandas

## Use
***python3 [option] merge_results.py -d /path/to/data/folder -o /path/to/output.csv***

### Option:
- -s, --sort : Name of the column to sort by (Default: number)
## Example 
***python3 merge_results.py -d ../CSV_FILE/TO_MERGE -o ../CSV_FILE/GLOBAL/RESULT_0.csv***

***
# Rename File
This python script rename all files in a folder with the name "Results_i" with i 0 to the number of file in the selected folder

##Use
***python3 RenaFile.py -data ../CSV_FILE/GLOBAL***

