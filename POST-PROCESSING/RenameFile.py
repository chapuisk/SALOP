import os
import glob
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s -data path/to/data')
    parser.add_argument('-data', metavar="path/to/data",
                        help='path to the folder with data', required=True,type=str)
    args = parser.parse_args()
    path=args.data
    files = os.path.join(path+"/", "*.csv")
    files = glob.glob(files)
    for i in range(0,len(files)):
        os.rename(files[i], path++"/Results_"+str(i)+".csv")