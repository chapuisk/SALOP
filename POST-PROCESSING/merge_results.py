import glob
import os
import pandas as pd
import argparse

'''
This script merge and sort all CSV files in a folder.

This script is used to merge each CSV file generate by each machine/nodes.

/!\ don't use this script for simulation with replicat.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s [option] -d /path/to/data/folder -o /path/to/output.csv')

    parser.add_argument('-d',"--data_folder", metavar="/path/to/data/folder", type=str,help='Path to the data folder to merge', required=True)
    parser.add_argument('-o',"--output", metavar="/path/to/output.csv", type=str ,help='Output Argument', required=True)
    parser.add_argument('-s', "--sort",type=str,help='Name of the column to sort by (Default: number)', default="number")

    args = parser.parse_args()
    path_output=args.output
    path_input=args.data_folder
    colum_sort=args.sort

    path_input=path_input+"/"

    # setting the path for joining multiple files
    files = os.path.join(path_input, "*.csv")

    # list of merged files returned
    files = glob.glob(files)

    print("Resultant CSV after joining all CSV files at a particular location...\n")

    # joining files with concat and read_csv
    try:
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df=df.sort_values(by=colum_sort)
        del df[colum_sort]
        df.to_csv(path_output,index=False)
        print("== All CSV files merged \n")
        print("== END")
    except:
        print("== ERROR WRONG COLUMN")
