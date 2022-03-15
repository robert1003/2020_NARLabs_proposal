from scipy.sparse import csr_matrix
import itertools
import numpy as np
from collections import defaultdict

def assignId(df, cols, addIdCol=True, colPrefix='_id_'):
    # assign unique id to elements in columns specified in cols
    # then if addIdCol is on, convert element to id and store at df[colPrefix+colname]
    colnum = 0
    name = {}
    convert = lambda x, st: dict(zip(x, range(st, st + len(x))))
    inv_convert = lambda x, name: dict(zip(x.values(), [(t, name) for t in x.keys()]))

    name2id = {}
    id2name = {}
    id2type = {}
    for col in cols:
        name2id[col] = convert(df[col].unique().tolist(), colnum)
        id2name.update({y:x for x, y in name2id[col].items()})
        id2type.update({y:col for y in name2id[col].values()})
        colnum += len(name2id[col])

    if addIdCol:
        for col in cols:
            df['_id_{}'.format(col)] = df[col].apply(lambda x: name2id[col][x])

    return name2id, id2name, id2type

def getTwoColIdMapping(df, col1, col2, score_of_col2_elem=None):
    # create id mapping of two column
    col1_to_col2 = {}
    for i, j in zip(df[col1], df[col2]):
        if score_of_col2_elem is not None and i in col1_to_col2:
            if score_of_col2_elem[col1_to_col2[i]] < score_of_col2_elem[j]:
                col1_to_col2[i] = j
        else:
            col1_to_col2[i] = j

    return col1_to_col2

def getSparseIncidenceMatrixFromId(df, cols, colnum, filter_func=None, author_group=None): 
    # cols: column name
    # colnum: number of ids
    # rId: researcherId2CountryId
    #
    # convert our data to csr format
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix

    # paper hyperedge
    data, colIndex, rowPtr = [], [], [0]
    for x in zip(*[df[col] for col in cols]):
        # connect researcher only to one country (2021/12/21 meeting)
        if filter_func is not None and filter_func(x):
            continue
        data.append([1] * len(x))
        colIndex.append(x)
        rowPtr.append(rowPtr[-1] + len(x))

    # group hyperedge
    if author_group is not None:
        for paper_name, g in author_group.items():
            data.append([1] * len(g))
            colIndex.append(list(g))
            rowPtr.append(rowPtr[-1] + len(g))

    data = np.hstack(data).astype(np.float)
    colIndex = np.hstack(colIndex)

    # create csr sparse matrix
    return csr_matrix((data, colIndex, rowPtr))#shape=(len(df), colnum))

def getTopkItemsFromLists(val, k, *lists):
    indices = np.array(val).argsort()[::-1][:k]
    res = []
    for idx in indices:
        res.append((val[idx], [l[idx] for l in lists]))

    return res

def getAdjMatrix(df, edge_col_name, col):
    # get all the edges
    group = defaultdict(lambda: set())
    for x, y in zip(df[edge_col_name], df[col]):
        group[x].add(y)

    # construct graph
    cnt = defaultdict(lambda: defaultdict(lambda: 0))
    for _, li in group.items():
        # li are indices of each group connected by edge_col_name
        li = list(li)
        for i in range(len(li)):
            for j in range(i, len(li)):
                cnt[min(li[i], li[j])][max(li[i], li[j])] += 1

    return cnt