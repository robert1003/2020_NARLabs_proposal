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
    colName = ['journaltitle', 'Fullname', 'Country', 'FullOrgName']
    nickName = ['Journal', 'Researcher', 'Country', 'Organization']
    colPrefix='_id_'
    colNameId = list(map(lambda x: colPrefix+x, colName))

    name2id, id2name, id2type = assignId(df, colName, colPrefix=colPrefix)
    researcherId2CountryId = getTwoColIdMapping(df, colPrefix+'Fullname', colPrefix+'Country')
    researcherId2OrgId = getTwoColIdMapping(df, colPrefix+'Fullname', colPrefix+'FullOrgName')
    orgId2CountryId = getTwoColIdMapping(df, colPrefix+'FullOrgName', colPrefix+'Country')

    # get country centrality ranking, then update researcherId2CountryId and orgId2CountryId
    # (meeting in 2022/01/17)
    E = getSparseIncidenceMatrixFromId(df, colNameId, len(id2name), None)
    vals, vecs = eigs(E.T@E, 1, which='LM')
    maxEigenvector = np.abs(vecs.T[0]) # idx of this vector correspond to idx of nodes
    researcherId2CountryId = getTwoColIdMapping(df, colPrefix+'Fullname', colPrefix+'Country', maxEigenvector)
    researcherId2OrgId = getTwoColIdMapping(df, colPrefix+'Fullname', colPrefix+'FullOrgName', maxEigenvector)
    orgId2CountryId = getTwoColIdMapping(df, colPrefix+'FullOrgName', colPrefix+'Country', maxEigenvector)

    # get co-authorship (hyper)edges
    group = defaultdict(lambda: set())
    for x, y in zip(df['PaperTitle'], df[colPrefix+'Fullname']):
        group[x].add(y)

    # create incidence (sparce) matrix
    def filter_func(x):
        return (researcherId2CountryId[x[1]] != x[2]) or \
                (researcherId2OrgId[x[1]] != x[3]) or \
                (orgId2CountryId[x[3]] != x[2])  

    E = getSparseIncidenceMatrixFromId(df, colNameId, len(id2name), 
        filter_func=filter_func, author_group=group)

    # transpose and multiple it, then calculate the topk eigenvalues and vectors
    vals, vecs = eigs(E.T@E, 1, which='LM')
    #vals, vecs = eig((E.T@E).todense())

    # get largest contributer in each colName
    maxEigenvalue = vals[0]
    maxEigenvector = np.abs(vecs.T[0])
    print(np.linalg.norm(E.T@E@maxEigenvector - maxEigenvalue*maxEigenvector))
    result = getTopkItemsFromLists(maxEigenvector, len(maxEigenvector), id2name, id2type)
    _data = [[x[0]] + x[1] for x in result]
    print(*result[:5], sep='\n')
    df = pd.DataFrame(_data, columns=['centrality', 'name', 'group'])
    
    def f(x):
        if x in name2id['Fullname']:
            return id2name[researcherId2CountryId[name2id['Fullname'][x]]]
        if x in name2id['FullOrgName']:
            return id2name[orgId2CountryId[name2id['FullOrgName'][x]]]
        return None
    
    def f2(x):
        if x in name2id['Fullname']:
            return id2name[researcherId2OrgId[name2id['Fullname'][x]]]

    df['country'] = df['name'].apply(f)
    df['organization'] = df['name'].apply(f2)
    df.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
