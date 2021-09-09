from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import itertools

def assignId(df, addIdCol=True):
    colnum = 0
    name = {}
    convert = lambda x, st: dict(zip(x, range(st, st + len(x))))
    inv_convert = lambda x, name: dict(zip(x.values(), [(t, name) for t in x.keys()]))

    researchers = convert(df['Fullname'].unique().tolist(), colnum)
    name.update(inv_convert(researchers, 'researcher'))
    colnum += len(researchers)

    articles = convert(df['PaperTitle'].unique().tolist(), colnum)
    name.update(inv_convert(articles, 'article'))
    colnum += len(articles)

    countries = convert(df['Country'].unique().tolist(), colnum)
    name.update(inv_convert(countries, 'country'))
    colnum += len(countries)

    nationality = {}
    for i, j in zip(df['Fullname'], df['Country']):
        nationality[researchers[i]] = countries[j]
    
    if addIdCol:
        df['Fullname_id'] = df['Fullname'].apply(lambda x: researchers[x])
        df['PaperTitle_id'] = df['PaperTitle'].apply(lambda x: articles[x])
        df['Country_id'] = df['Country'].apply(lambda x: countries[x])

    return researchers, articles, countries, nationality, name

def getSparseE(df):
    indices = np.array(list(itertools.chain(*zip(df['Fullname_id'], df['PaperTitle_id'], df['Country_id']))))
    indptr = np.arange(0, len(indices)+1, 3)
    data = np.ones(indices.shape)
    E = csr_matrix((data, indices, indptr), shape=(len(df), colnum))

    return E

def getSparseEigenTopk(E, k):
    return eigs(E, k=k)
