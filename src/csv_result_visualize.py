import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--src', required=True)
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--group')
    args = parser.parse_args()

    # Argument validation
    assert args.src.endswith('.csv')
    if args.group is not None:
        args.group = args.group.split(',')

    return args

def main():
    global args
    args = parse_args()
    df = pd.read_csv(args.src)
    cols = sorted(df['group'].unique())

    if args.group is not None:
        for col in args.group:
            assert col in cols
        cols = args.group

    for col in cols:
        sub_df = df.loc[df['group'] == col, :]
        plt.title(col)
        plt.bar(x=range(args.topk), height=sub_df['centrality'][:args.topk])
        plt.xticks(range(args.topk), sub_df['name'][:args.topk], rotation=90)
        plt.xlabel('name')
        plt.ylabel('centrality')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
