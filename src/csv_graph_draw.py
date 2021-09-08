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
    parser.add_argument('--year', default=None, type=int)

    args = parser.parse_args()

    # Argument validation
    assert args.src.endswith('.csv')

    return args

def getAdjMatrix(df, Type):
    group = defaultdict(lambda: set())
    for a, b, c in zip(df['Fullname_id'], df['PaperTitle_id'], df['Country_id']):
        if Type == 'researcher':
            group[b].add(a)
        elif Type == 'country':
            group[b].add(c)
        else:
            raise NotImplementedError()
    cnt = defaultdict(lambda: defaultdict(lambda: 0))
    for _, li in group.items():
        li = list(li)
        for i in range(len(li)):
            for j in range(i + 1, len(li)):
                cnt[min(li[i], li[j])][max(li[i], li[j])] += 1

    return cnt

def main():
    global args
    args = parse_args()
    df = pd.read_csv(args.src)
    if args.year:
        df = df.loc[df['DBYear'] == args.year, :]

    # Network
    net = {
            'country': Network(height='600px', width='1300px'),
            'researcher': Network(height='600px', width='1300px')
        }
    
    # get unique researchers, articles, countries, make it a mapping
    # convert each paper to (researcher_id, article_id, country_id)
    researchers, articles, countries, nationality, name = assignId(df)

    # add node to Network
    net['country'].add_nodes(list(countries.values()), label=list(countries.keys()))
    colors = defaultdict(lambda: "#%06x" % random.randint(0, 0xFFFFFF))
    net['researcher'].add_nodes(list(researchers.values()), label=list(researchers.keys()), color=[colors[nationality[i]] for i in researchers.values()])

    # get adjacency matrix
    G_country = getAdjMatrix(df, 'country')
    G_author = getAdjMatrix(df, 'researcher')

    # add edges to graph
    for i in G_country:
        for j in G_country[i]:
            net['country'].add_edge(i, j, value=G_country[i][j])

    for i in G_author:
        for j in G_author[i]:
            net['researcher'].add_edge(i, j, value=G_author[i][j])

    # export graph
    net['country'].show_buttons(filter_=True)
    net['country'].show('nodes_country.html')
    net['researcher'].show_buttons(filter_=True)
    net['researcher'].show('nodes_researcher.html')

if __name__ == '__main__':
    main()
