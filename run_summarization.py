# under construction - DO NOT DELETE

import sys
import pandas as pd
from nltk.tokenize import sent_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from operator import add

import en_core_web_lg
nlp = en_core_web_lg.load()

from ExtractiveSummarization.tfidfSummarizer.enhanced_tfidf import get_summary
from translation.translation import translate
from knowledgeGraph.kg_constructor import create_graph
from knowledgeGraph.extractor.extractor import findSVOs
from knowledgeGraph.kg_summarizer import bart_large_cnn

def join_tuple_string(strings_tuple) -> str:
    return ' '.join(strings_tuple)

if __name__ == "__main__":

    topic = sys.argv[1]
    titles = ['Russia Ukraine war', 'रूस यूक्रेन युद्ध', 'रशिया युक्रेन युद्ध']
    languages = ['english', 'hindi', 'marathi']
    text = []
    int_summary = []

    with open('./data/original_docs/english/' + topic + '.txt', 'r', encoding="UTF-8") as file:
        text.append(file.read())
    with open('./data/original_docs/hindi/' + topic + '.txt', 'r', encoding="UTF-8") as file:
        text.append(file.read())
    with open('./data/original_docs/marathi/' + topic + '.txt', 'r', encoding="UTF-8") as file:
        text.append(file.read())
        
    for i in range(0, 3):
        res = get_summary(topic, titles[i], languages[i])
        if(languages[i] != 'english'):
            res = translate(res, languages[i])
            res = res.text
        int_summary.append(res)

    int_summary = ' '.join(int_summary)

    f = sent_tokenize(int_summary)

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

    result = map(join_tuple_string, final_nodes)
    result = ". ".join(result)

    create_graph(final_nodes)
    plt.savefig('./data/knowledge_graphs/kg_' + topic + '.png')

    final_text = int_summary + result

    final_summary = bart_large_cnn(final_text)
    final_summary = final_summary[0]['summary_text']

    print('---summariztion done:: ', final_summary)