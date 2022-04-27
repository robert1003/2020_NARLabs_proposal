from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig
import pandas as pd
import numpy as np
from utils import *

df = pd.read_csv('../data/2021-12-16/csv/2015.csv', index_col=None)

# remove NBER and CEPR from data
# (meeting in 2022/03/21)
df = df.loc[~df['FullOrgName'].isin(['NBER', 'CEPR']), :]

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

E = getSparseIncidenceMatrixFromId(df, colNameId, len(id2name), \
	node_weight=dict(zip(range(len(name2id)),np.random.uniform(size=(len(name2id),)))))
full_E = E.T@E

valerr = []
vecerr = []
for _ in range(100):
	idx = np.random.permutation(full_E.shape[0])[:1000]
	sub_E = full_E[idx][:, idx]

	topk = 1
	sparse_vals, sparse_vecs = eigs(sub_E, topk, which='LM')
	dense_vals, dense_vecs = eig(sub_E.todense())
	index = np.argsort(dense_vals)[::-1]
	dense_vals = dense_vals[index]
	dense_vecs = dense_vecs[:, index]

	valerr.append(np.sum((sparse_vals[:topk] - dense_vals[:topk])**2, axis=0))
	vecerr.append(np.sum((sparse_vecs[:,:topk] - dense_vecs[:,:topk])**2))

print('valerr')
print(pd.Series(valerr).describe())
print('vecerr')
print(pd.Series(vecerr).describe())

import matplotlib.pyplot as plt
fig, axs=plt.subplots(1,2)
axs[0].hist(valerr)
axs[1].hist(vecerr)
plt.show()