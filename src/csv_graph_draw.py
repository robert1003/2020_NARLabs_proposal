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

def draw_country(net, df, countries):
    net.add_nodes(list(countries.values()), label=list(countries.keys()))
    G_country = getAdjMatrix(df, 'country')

    # add edges to graph
    for i in G_country:
        for j in G_country[i]:
            net.add_edge(i, j, value=G_country[i][j])

def draw_researcher(net, df, researchers, nationality):
    # add node to Network
    colors = defaultdict(lambda: "#%06x" % random.randint(0, 0xFFFFFF))
    net.add_nodes(list(researchers.values()), label=list(researchers.keys()), color=[colors[nationality[i]] for i in researchers.values()])

    # get adjacency matrix
    G_author = getAdjMatrix(df, 'researcher')

    for i in G_author:
        for j in G_author[i]:
            net.add_edge(i, j, value=G_author[i][j])

def main():
    global args
    args = parse_args()
    df = pd.read_csv(args.src)
    if args.year:
        df = df.loc[df['DBYear'] == args.year, :]

    # get unique researchers, articles, countries, make it a mapping
    # convert each paper to (researcher_id, article_id, country_id)
    researchers, articles, countries, nationality, name = assignId(df)

    # draw and export
    net = Network(height='100%', width='100%')
    draw_country(net, df, countries)
    net.show_buttons(filter_=True)
    net.show('nodes_country.html')

    net = Network(height='100%', width='100%')
    draw_researcher(net, df, researchers, nationality)
    net.show_buttons(filter_=True)
    net.show('nodes_researcher.html')

if __name__ == '__main__':
    main()
