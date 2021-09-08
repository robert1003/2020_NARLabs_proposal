import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import itertools
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--path', required=True)
    parser.add_argument('--sheet_name', required=True)
    parser.add_argument('--entry_topk', default=6, type=int)
    parser.add_argument('--year', default=None, type=int)

    args = parser.parse_args()

    # Argument validation
    assert args.path.endswith('.xlsx')

    return args

def main():
    global args
    args = parse_args()
    df = pd.read_excel(args.path, engine='openpyxl', sheet_name=args.sheet_name)
    if args.year:
        df = df.loc['DBYear', args.year]
    
    # get unique researchers, articles, countries, make it a mapping
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

    # convert each paper to (researcher_id, article_id, country_id)
    df['Fullname_id'] = df['Fullname'].apply(lambda x: researchers[x])
    df['PaperTitle_id'] = df['PaperTitle'].apply(lambda x: articles[x])
    df['Country_id'] = df['Country'].apply(lambda x: countries[x])

    # create incidence (sparce) matrix
    indices = np.array(list(itertools.chain(*zip(df['Fullname_id'], df['PaperTitle_id'], df['Country_id']))))
    indptr = np.arange(0, len(indices)+1, 3)
    data = np.ones(indices.shape)
    E = csr_matrix((data, indices, indptr), shape=(len(df), colnum))
    print('size of E =', E.shape)

    # transpose and multiple it, then calculate the topk eigenvalues and vectors
    vals, vecs = eigs(E.T @ E, k=10)
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
    vals, vecs = eigs(E @ E.T, k=10)
    print(vals.shape)
    print(vecs.shape)

    # print the largest contributers
    for val, vec in zip(vals[:1], vecs.T[:1]):
        print('eigenvalue =', val.real)
        for idx, v in zip(vec.argsort()[-args.entry_topk:][::-1], np.sort(vec)[-args.entry_topk:][::-1]):
            print(idx, v.real, df.iloc[idx, :])
        print('-'*50)

if __name__ == '__main__':
    main()
