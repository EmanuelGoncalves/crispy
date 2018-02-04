#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import igraph
import pandas as pd


STRING_FILE = 'data/resources/string/9606.protein.links.full.v10.5.txt'
STRING_ALIAS_FILE = 'data/resources/string/9606.protein.aliases.v10.5.txt'
STRING_PICKLE = 'data/igraph_string.pickle'

BIOGRID_FILE = 'data/resources/biogrid/BIOGRID-ALL-3.4.145.tab2.txt'
BIOGRID_PICKLE = 'data/igraph_biogrid.pickle'


def import_ppi(ppi_pickle):
    return igraph.Graph.Read_Pickle(ppi_pickle)


def build_biogrid_ppi(exp_type=None, missing_interactions=None):
    # 'Affinity Capture-MS', 'Affinity Capture-Western', 'Co-crystal Structure', 'Co-purification', 'Reconstituted Complex', 'PCA', 'Two-hybrid'
    exp_type = {'Affinity Capture-MS', 'Affinity Capture-Western'} if exp_type is None else exp_type

    missing_interactions = [('TP53', 'PPM1D')] if missing_interactions is None else missing_interactions

    # Import
    biogrid = pd.read_csv(BIOGRID_FILE, sep='\t')

    # Human only
    biogrid = biogrid[
        (biogrid['Organism Interactor A'] == 9606) & (biogrid['Organism Interactor B'] == 9606)
    ]

    # Physical interactions only
    biogrid = biogrid[biogrid['Experimental System Type'] == 'physical']

    # Filter by experimental type
    biogrid = biogrid[[i in exp_type for i in biogrid['Experimental System']]]

    # Interaction source map
    biogrid['interaction'] = biogrid['Official Symbol Interactor A'] + '<->' + biogrid['Official Symbol Interactor B']

    # Unfold associations
    biogrid = {
        (s, t) for p1, p2 in biogrid[['Official Symbol Interactor A', 'Official Symbol Interactor B']].values
        for s, t in [(p1, p2), (p2, p1)] if s != t
    }

    # Append missing interactions
    biogrid.update(missing_interactions)

    # Build igraph network
    # igraph network
    net_i = igraph.Graph(directed=False)

    # Initialise network lists
    edges = [(px, py) for px, py in biogrid]
    vertices = list({p for p1, p2 in biogrid for p in [p1, p2]})

    # Add nodes
    net_i.add_vertices(vertices)

    # Add edges
    net_i.add_edges(edges)

    # Simplify
    net_i = net_i.simplify()
    print(net_i.summary())

    # Export
    net_i.write_pickle(BIOGRID_PICKLE)


def build_string_ppi(score_thres=900):
    # ENSP map to gene symbol
    gmap = pd.read_csv(STRING_ALIAS_FILE, sep='\t')
    gmap = gmap[['BioMart_HUGO' in i.split(' ') for i in gmap['source']]]
    gmap = gmap.groupby('string_protein_id')['alias'].agg(lambda x: set(x)).to_dict()
    gmap = {k: list(gmap[k])[0] for k in gmap if len(gmap[k]) == 1}
    print('ENSP gene map: ', len(gmap))

    # Load String network
    net = pd.read_csv(STRING_FILE, sep=' ')

    # Filter by moderate confidence
    net = net[net['combined_score'] > score_thres]

    # Filter and map to gene symbol
    net = net[[p1 in gmap and p2 in gmap for p1, p2 in net[['protein1', 'protein2']].values]]
    net['protein1'] = [gmap[p1] for p1 in net['protein1']]
    net['protein2'] = [gmap[p2] for p2 in net['protein2']]
    print('String: ', len(net))

    #  String network
    net_i = igraph.Graph(directed=False)

    # Initialise network lists
    edges = [(px, py) for px, py in net[['protein1', 'protein2']].values]
    vertices = list(set(net['protein1']).union(net['protein2']))

    # Add nodes
    net_i.add_vertices(vertices)

    # Add edges
    net_i.add_edges(edges)

    # Add edge attribute score
    net_i.es['score'] = list(net['combined_score'])

    # Simplify
    net_i = net_i.simplify(combine_edges='max')
    print(net_i.summary())

    # Export
    net_i.write_pickle(STRING_PICKLE)


if __name__ == '__main__':
    build_biogrid_ppi()
    build_string_ppi()
    print('[INFO] PPIs pickle files created')
