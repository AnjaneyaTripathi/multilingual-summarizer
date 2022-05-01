import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from operator import add

import en_core_web_lg
nlp = en_core_web_lg.load()

def create_graph(final_nodes):
    source = []
    target = []
    edge = []
    indexes = []

    for i in (range(len(final_nodes))):
        ent1 = final_nodes[i][0]
        ent2 = final_nodes[i][2]
        rel = final_nodes[i][1] 
        source.append(ent1.lower().strip())
        target.append(ent2.lower().strip())
        edge.append("".join(rel).strip())
        indexes.append(i)
    if(len(edge) == 0 or len(final_nodes) == 0):
        return None
    else:
        G = nx.DiGraph(directed=True)
        for i in (range(len(edge))):
            G.add_weighted_edges_from([(source[i], target[i], i)])
        size=20
        if len(edge)/2 > 20:
            size = len(edge)/2
        plt.figure(figsize = (size, size))
        edge_labels = dict([((u, v, ), edge[d['weight']]) for u, v, d in G.edges(data = True)])
        pos = nx.spring_layout(G, k = 0.8)
        nx.draw(G, with_labels = True, node_color = 'lightblue', node_size=5000, edge_color='r', edge_cmap = plt.cm.Blues, pos=pos, font_size=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 15)
        return G, edge_labels

# def join_tuple_string(strings_tuple) -> str:
#     return ' '.join(strings_tuple)

# def generate_kg(text, filename):
#     f = sent_tokenize(text)

#     nodes = []
#     for sentence in f: 
#         tokens = nlp(sentence)
#         svos = findSVOs(tokens)
#         nodes.append(svos)

#     final_nodes = []

#     for node in nodes:
#         for j in node:
#             if(len(j) == 3):
#                 final_nodes.append(j)

#     print(final_nodes)

#     # joining all the tuples
#     result = map(join_tuple_string, final_nodes)
#     result = ". ".join(result)

#     # converting and printing the result
#     print(result)

#     create_graph(final_nodes)
#     plt.savefig('./data/kg_' + filename + '.png')
    

# def test():
#     filename = 'summary'

#     with open('./data/' + filename + '.txt', 'r') as file:
#         text = file.read()

#     f = sent_tokenize(text)

#     nodes = []
#     for sentence in f: 
#         tokens = nlp(sentence)
#         svos = findSVOs(tokens)
#         nodes.append(svos)

#     final_nodes = []

#     for node in nodes:
#         for j in node:
#             if(len(j) == 3):
#                 final_nodes.append(j)

#     print(final_nodes)

#     def join_tuple_string(strings_tuple) -> str:
#         return ' '.join(strings_tuple)

#     # joining all the tuples
#     result = map(join_tuple_string, final_nodes)
#     result = ". ".join(result)

#     # converting and printing the result
#     print(result)

#     create_graph(final_nodes)
#     plt.savefig('./images/kg_' + filename + '.png')
