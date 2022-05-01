import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from operator import add
from transformers import pipeline

import en_core_web_lg
nlp = en_core_web_lg.load()

# filename = 'imrankhan'

# with open('../data/generated_summaries/extractive/' + filename + '.txt', 'r') as file:
#     text = file.read()

# f = sent_tokenize(text)

# nodes = []
# for sentence in f: 
#     tokens = nlp(sentence)
#     svos = findSVOs(tokens)
#     nodes.append(svos)

# final_nodes = []

# for node in nodes:
#     for j in node:
#         if(len(j) == 3):
#             final_nodes.append(j)

# print(final_nodes)

# def join_tuple_string(strings_tuple) -> str:
#    return ' '.join(strings_tuple)

# result = map(join_tuple_string, final_nodes)
# result = ". ".join(result)

def bart_large_cnn(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False, truncation=True)
    return summary

# result1 = text + result
# result2 = result + text

# bart_large_cnn(result1)
# bart_large_cnn(result2)