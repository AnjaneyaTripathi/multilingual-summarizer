import os
from extractor import *

def get_resolved_entities(input_file, output_file):

    try:
        with open('./stanford-corenlp-4.4.0/' + output_file + '.txt', 'r') as f:
            processed_text = f.read()
    except:
        print("processed file doesn't exist, processing...")
        os.system('cd stanford-corenlp-4.4.0 && java -Xmx5g -cp "*" edu.stanford.nlp.naturalli.OpenIE ' + input_file + '.txt -resolve_coref true -output ' + output_file + '.txt')
        with open('./stanford-corenlp-4.4.0/' + output_file + '.txt', 'r') as f:
            processed_text = f.read()

    with open('./stanford-corenlp-4.4.0/' + input_file + '.txt', 'r') as f:
        text = f.read()

    sentences = processed_text.split('\n')
    tokens_list = []
    for sent in sentences:
        tokens_list.append(sent.split('\t')[1:4])
    return text, tokens_list

text, tokens_list = get_resolved_entities('news', 'news_output')

for t in tokens_list:
    print(t)

print(text)