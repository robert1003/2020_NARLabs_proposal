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
    parser.add_argument('--topk', default=6, type=int)
    parser.add_argument('--year', default=None)
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

    if args.year:
        df = df.loc[df['DBYear'].isin(args.year), :]
    
    # get unique researchers, articles, countries, make it a mapping
    # convert each paper to (researcher_id, article_id, country_id)
    colName = ['Fullname', 'Country', 'FullOrgName']
    nickName = ['Researcher', 'Country', 'Organization']
    colPrefix='_id_'
    colNameId = list(map(lambda x: colPrefix+x, colName))

    name2id, id2name, id2type = assignId(df, colName, colPrefix=colPrefix)
    researcherId2CountryId = getTwoColIdMapping(df, colPrefix+'Fullname', colPrefix+'Country')

    # create incidence (sparce) matrix
    E = getSparseIncidenceMatrixFromId(df, colNameId, len(id2name))

    # transpose and multiple it, then calculate the topk eigenvalues and vectors
    vals, vecs = eigs(E.T@E, 1, which='LM')
    #vals, vecs = eig((E.T@E).todense())

    # get largest contributer in each colName
    maxEigenvalue = vals[0]
    maxEigenvector = np.abs(vecs.T[0])
    result = getTopkItemsFromLists(maxEigenvector, len(maxEigenvector), id2name, id2type)
    _data = [[x[0]] + x[1] for x in result]
    print(*result[:5], sep='\n')
    pd.DataFrame(_data, columns=['centrality', 'name', 'group']).to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
