import matplotlib.pyplot as plt
import hypernetx as hnx
import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--src', required=True)
    parser.add_argument('--year', default=None)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # Argument validation
    assert args.src.endswith('.csv')
    if args.year:
        args.year = list(map(int, args.year.strip().split(',')))

    return args

def main():
    global args
    args = parse_args()
    df = pd.read_csv(args.src, index_col=None)

    if args.year:
        df = df.loc[df['DBYear'].isin(args.year), :]
    
    # convert name to id
    colName = ['Fullname', 'Country', 'FullOrgName']
    clusterCenter = [(args.size*30, -args.size*30), (0, args.size*30), (-args.size*30, -args.size*30)]
    dirc = [(args.size*5, args.size*5), (0, -args.size*5), (args.size*5, -args.size*5)]
    nickName = ['Researcher', 'Country', 'Organization']

    # add edges
    df_sub = df.sample(n=args.size, random_state=args.seed)
    hyperedges = dict(zip(df_sub.index, list(zip(*[df_sub[col] for col in colName]))))
    #for col in colName:
    #    hyperedges[col] = list(df_sub[col].unique())

    pos = {}
    for col, center, d in zip(colName, clusterCenter, dirc):
        for i, name in enumerate(df_sub[col].unique()):
            # USA is so fat that it will affect other edges
            # thus we isolate its position to bottom right
            if name == 'USA':
                pos[name] = (args.size*50, -args.size*50)
            else:
                pos[name] = (center[0]+d[0]*i, center[1]+d[1]*i)

    H = hnx.Hypergraph(hyperedges)
    hnx.draw(H, pos=pos, edge_labels=df_sub['ArticleNumber'])
    plt.show()

if __name__ == '__main__':
    main()
