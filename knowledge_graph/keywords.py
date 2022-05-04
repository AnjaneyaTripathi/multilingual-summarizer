import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from operator import add
from extractor.extractor import findSVOs
from keytotext import pipeline

import en_core_web_lg
nlp = en_core_web_lg.load()

filename = 'summary'

with open('./data/' + filename + '.txt', 'r') as file:
    text = file.read()

f = sent_tokenize(text)

nodes = []
for sentence in f: 
    tokens = nlp(sentence)
    svos = findSVOs(tokens)
    nodes.append(svos)

final_nodes = []

for node in nodes:
    for j in node:
        if(len(j) == 3):
            final_nodes.append(j)
            print(type(j))

nlp = pipeline("k2t-base")  #loading the pre-trained model
params = {"do_sample":True, "num_beams":4, "no_repeat_ngram_size":3, "early_stopping":True}    #decoding params

res = ""

for node in final_nodes:
    temp = nlp(np.asarray(node), **params)
    res = res + temp
    print(np.asarray(node))
    print(temp)
    print("\n\n")

print(res)