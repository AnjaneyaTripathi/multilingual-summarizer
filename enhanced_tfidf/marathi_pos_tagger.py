# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:35:13 2022

@author: Isha
"""

import nltk
import pandas as pd
import pickle as pkl

from nltk import word_tokenize
from nltk.tag import untag
from nltk import UnigramTagger

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

from nltk.corpus import indian
nltk.download('indian')

tagged_set = 'marathi.pos'
articles = indian.sents(tagged_set)
count=len(articles)

train_perc = .9
train_rows = int(train_perc*count)
test_rows = train_rows + 1


data = indian.tagged_sents(tagged_set)
train_data = data[:train_rows] 
test_data = data[test_rows:]


unigram_tagger = UnigramTagger(train_data,backoff=nltk.DefaultTagger('NN'))


print(unigram_tagger.evaluate(test_data)) 

list='डॉ. मानसी शिरीष कणेकर  या मराठी लेखिका, कवयित्री व गायिका होत्या.'
words = word_tokenize(list)
# print(unigram_tagger.tag(words))

# filenm = 'marathi_pos.pickle'
# pickle = pkl.dump(unigram_tagger, open(filenm, 'wb'))



