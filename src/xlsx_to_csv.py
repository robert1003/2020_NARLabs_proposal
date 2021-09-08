import pandas as pd
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--src', required=True)
    parser.add_argument('--prefix')
    parser.add_argument('--dst', required=True)

    args = parser.parse_args()

    # Argument validation
    assert args.src.endswith('.xlsx')
    assert os.path.isdir(args.dst)
    if args.prefix is None:
        args.prefix = args.src.split('/')[-1][:-5]

    return args

def main():
    global args
    args = parse_args()
    xl = pd.ExcelFile(args.src, engine='openpyxl')
    for name in xl.sheet_names:
        print(name)
        xl.parse(name).to_csv(os.path.join(args.dst, ('' if args.prefix == '' else args.prefix+'-')+name+'.csv'))
#    df = pd.read_excel(args.path, engine='openpyxl', sheet_name=args.sheet_name)

if __name__ == '__main__':
    main()
