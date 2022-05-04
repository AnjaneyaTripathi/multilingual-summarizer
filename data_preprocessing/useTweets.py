import pandas as pd
import numpy as np

def generate_text():
    data = pd.read_csv('./dataPreprocessing/clean_data.csv')
    sentences = data['text']

    final = []

    for idx, sentence in enumerate(sentences):
        if(data.iloc[idx]['language'] == 'en'):
            final.append(sentence)
    
    result = ' '.join(final)

    return result