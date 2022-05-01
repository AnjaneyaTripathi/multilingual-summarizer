# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:35:45 2022

@author: Isha
"""
import glob
import os
from translation.translation import translate
from ExtractiveSummarization.tfidfSummarizer.enhanced_tfidf import get_summary

def summarize_translate(filename, title, language):
    with open(filename, 'r', encoding="UTF-8") as file:
            text = file.read()
            if(len(text)>1000):
                text = get_summary(filename, title, language)
            if(language != 'english'):
                text = translate(text, language)
                text = text.text
    return text

folder = 'war'
titles = ['Russia Ukraine war', 'रूस यूक्रेन युद्ध', 'रशिया युक्रेन युद्ध']
languages = ['english', 'hindi', 'marathi']

path = 'samples/inputs/' + folder
i=0
doc=''
for filename in glob.glob(os.path.join(path, '*.txt')):
    text = summarize_translate(filename, titles[i], languages[i])
    i=i+1
    doc += text + '\n'

filename = 'samples/outputs/summary' + folder + '.txt'

with open(filename, 'w') as file:
    file.write(doc)
