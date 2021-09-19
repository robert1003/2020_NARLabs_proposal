import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--src', required=True)
    parser.add_argument('--entry_topk', default=6, type=int)
    parser.add_argument('--year', default=None)
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
    
    # get unique researchers, articles, countries, make it a mapping
    # convert each paper to (researcher_id, article_id, country_id)
    researchers, articles, countries, nationality, name = assignId(df)

    # create incidence (sparce) matrix
    E = getSparseE(df, len(name))

    # transpose and multiple it, then calculate the topk eigenvalues and vectors
    vals, vecs = getSparseEigenTopk(E.T @ E, args.entry_topk * 2)
    print(vals.shape)
    print(vecs.shape)

    # print the largest contributers
    for val, vec in zip(vals[:1], vecs.T[:1]):
        print('eigenvalue =', val.real)
        for idx, v in zip(vec.argsort()[-args.entry_topk:][::-1], np.sort(vec)[-args.entry_topk:][::-1]):
            print(name[idx], v.real)
        print('-'*50)
    
    # plot
    k = 20
    sorted_value = list(zip(vecs.T[0].argsort()[::-1], np.sort(vecs.T[0])[::-1]))
    plt.bar(x=range(k), height=[y for x, y in sorted_value if name[x][1] == 'researcher'][:k])
    plt.xticks(range(k), [name[x][0] for x, y in sorted_value if name[x][1] == 'researcher'][:k], rotation=90)
    plt.tight_layout()
    plt.show()

    k = 10
    sorted_value = list(zip(vecs.T[0].argsort()[::-1], np.sort(vecs.T[0])[::-1]))
    plt.bar(x=range(k), height=[y for x, y in sorted_value if name[x][1] == 'country'][:k])
    plt.xticks(range(k), [name[x][0] for x, y in sorted_value if name[x][1] == 'country'][:k], rotation=90)
    plt.tight_layout()
    plt.show()

    k = 20
    sorted_value = list(zip(vecs.T[0].argsort()[::-1], np.sort(vecs.T[0])[::-1]))
    plt.bar(x=range(k), height=[y for x, y in sorted_value if name[x][1] == 'article'][:k])
    plt.xticks(range(k), [name[x][0] for x, y in sorted_value if name[x][1] == 'article'][:k], rotation=60)
    plt.tight_layout()
    plt.show()

    # transpose and multiple it, then calculate the topk eigenvalues and vectors
    vals, vecs = eigs(E @ E.T, args.entry_topk * 2)
    print(vals.shape)
    print(vecs.shape)

    # print the largest contributers
    for val, vec in zip(vals[:1], vecs.T[:1]):
        print('eigenvalue =', val.real)
        for idx, v in zip(vec.argsort()[-args.entry_topk:][::-1], np.sort(vec)[-args.entry_topk:][::-1]):
            print(idx, v.real, df.iloc[idx, :].PaperTitle)
        print('-'*50)

if __name__ == '__main__':
    main()
