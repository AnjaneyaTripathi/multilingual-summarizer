import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from operator import add
from extractor.extractor import findSVOs
from transformers import pipeline
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

nlp = pipeline("k2t-base")  #loading the pre-trained model
params = {"do_sample":True, "num_beams":10, "no_repeat_ngram_size":3, "early_stopping":True}    #decoding params

res = ""

for node in final_nodes:
    temp = nlp(node, **params)
    res = res + temp
    print(node)
    print(temp)
    print("\n\n")


def bart_large_cnn(text):
    print("Original Text:\n", text)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False, truncation=True)
    print("Summary:\n", summary)
    return summary

bart_large_cnn(res)
# bart_large_cnn(result2)