import os
import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from utils import *
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--year', default=None)
    parser.add_argument('--factor', default=None)
    args = parser.parse_args()

    # Argument validation
    assert args.src.endswith('.csv')
    assert not os.path.isfile(args.out)
    if args.year:
        args.year = list(map(int, args.year.strip().split(',')))

    return args

def main():
    global args
    args = parse_args()
    df = pd.read_csv(args.src, index_col=None)

    ######### calculate author centrality #########
    if args.year:
        df = df.loc[df['DBYear'].isin(args.year), :]

    # remove NBER and CEPR from data
    # (meeting in 2022/03/21)
    df = df.loc[~df['FullOrgName'].isin(['NBER', 'CEPR']), :]
    
    # get unique researchers, articles, countries, make it a mapping
    # convert each paper to (researcher_id, article_id, country_id)
    colName = ['Fullname']
    nickName = ['Researcher']
    colPrefix='_id_'
    colNameId = list(map(lambda x: colPrefix+x, colName))

    name2id, id2name, id2type = assignId(df, colName, colPrefix=colPrefix)

    # get co-authorship (hyper)edges
    group = defaultdict(lambda: set())
    for x, y in zip(df['PaperTitle'], df[colPrefix+'Fullname']):
        group[x].add(y)
    for x, y in group.items():
        print(x, [id2name[i] for i in y])

    E = getSparseIncidenceMatrixFromId(df, [], 0, author_group=group,)

    # transpose and multiple it, then calculate the topk eigenvalues and vectors
    vals, vecs = eigs(E.T@E, 1, which='LM')
    #vals, vecs = eig((E.T@E).todense())

    # get largest contributer in each colName
    maxEigenvalue = vals[0]
    maxEigenvector = np.abs(vecs.T[0])
    authorCentrality = {id2name[i]:maxEigenvector[i] for i in id2name}
    print(authorCentrality)
    print(*getTopkItemsFromLists(maxEigenvector, len(maxEigenvector), id2name, id2type), sep='\n')

if __name__ == '__main__':
    main()
