from scipy.sparse import csr_matrix
import itertools
import numpy as np

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

def getTwoColIdMapping(df, col1, col2):
    # create id mapping of two column
    col1_to_col2 = {}
    for i, j in zip(df[col1], df[col2]):
        col1_to_col2[i] = j

    return col1_to_col2

def getSparseIncidenceMatrixFromId(df, cols, colnum):
    # convert our data to csr format
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    data, colIndex, rowPtr = [], [], [0]
    for x in zip(*[df[col] for col in cols]):
        data.append([1] * len(x))
        colIndex.append(x)
        rowPtr.append(rowPtr[-1] + len(x))
    data = np.hstack(data).astype(np.float)
    colIndex = np.hstack(colIndex)

    # create csr sparse matrix
    return csr_matrix((data, colIndex, rowPtr), shape=(len(df), colnum))

def getTopkItemsFromLists(val, k, *lists):
    indices = np.array(val).argsort()[::-1][:k]
    res = []
    for idx in indices:
        res.append((val[idx], [l[idx] for l in lists]))

    return res
