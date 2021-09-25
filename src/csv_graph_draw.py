import argparse
import random
import os
import pandas as pd
import numpy as np
from pyvis.network import Network
from collections import defaultdict
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--src', required=True)
    parser.add_argument('--year', default=None)

    args = parser.parse_args()

    # Argument validation
    assert args.src.endswith('.csv')
    if args.year:
        args.year = list(map(int, args.year.strip().split(',')))

    return args

def getAdjMatrix(df, edge_col, col):
    # get all the edges
    group = defaultdict(lambda: set())
    for x, y in zip(df[edge_col], df[col]):
        group[x].add(y)

    # construct graph
    cnt = defaultdict(lambda: defaultdict(lambda: 0))
    for _, li in group.items():
        # li are indices of each group connected by edge_col
        li = list(li)
        for i in range(len(li)):
            for j in range(i + 1, len(li)):
                cnt[min(li[i], li[j])][max(li[i], li[j])] += 1

    return cnt


def draw_country(net, df, countries):
    net.add_nodes(list(countries.values()), label=list(countries.keys()), size=[5]*len(countries))
    G_country = getAdjMatrix(df, 'country')

    # add edges to graph
    for i in G_country:
        for j in G_country[i]:
            net.add_edge(i, j, value=G_country[i][j])

def main():
    global args
    args = parse_args()
    df = pd.read_csv(args.src)
    if args.year:
        df = df.loc[df['DBYear'].isin(args.year), :]

    # convert name to id
    colName = ['Fullname', 'Country', 'FullOrgName']
    nickName = ['Researcher', 'Country', 'Organization']
    colPrefix='_id_'
    colNameId = list(map(lambda x: colPrefix+x, colName))

    name2id, id2name, id2type = assignId(df, colName, colPrefix=colPrefix)
    researcherId2CountryId = getTwoColIdMapping(df, colPrefix+'Fullname', colPrefix+'Country')


    # draw and export
    for col, nick in zip(colName, nickName):
        colors = defaultdict(lambda: "#%06x" % random.randint(0, 0xFFFFFF))
        net = Network(height='100%', width='100%')
        color = [colors[researcherId2CountryId[i]] for i in name2id[col].values()] if col == 'Fullname' else [colors[0]]*len(name2id[col])

        net.add_nodes(
            list(name2id[col].values()),
            label=list(name2id[col].keys()),
            size=[5]*len(name2id[col]),
            color=color
        )

        G = getAdjMatrix(df, 'PaperTitle', colPrefix+col)
        for i in G:
            for j in G[i]:
                net.add_edge(i, j, value=G[i][j])

        net.show_buttons(filter_=True)
        net.show('nodes_{}.html'.format(nick))

if __name__ == '__main__':
    main()
