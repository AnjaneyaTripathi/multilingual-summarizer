# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:35:45 2022

@author: Isha
"""

import sys
sys.path.insert(0, 'ExtractiveSummarization/tfidf-summarizer')

import enhanced_tfidf as summarizer

def summarize_translate(filename, title, language):
    with open(filename, 'r', encoding="UTF-8") as file:
            text = file.read()
            print(len(text))
            if(len(text)>1000):
                text = summarizer.get_summary(filename, title, language)

    sys.path.insert(0, 'Translation')

    import translator

    if(language != 'english'):
        text = translator.translate(text, language)

    return text.text

filenames = ['war_hindi.txt', 'war_marathi.txt']
titles = ['रूस यूक्रेन युद्ध', 'रशिया युक्रेन युद्ध']
languages = ['hindi', 'marathi']

for i in range(len(filenames)):
    doc = ''
    text = summarize_translate(filenames[i], titles[i], languages[i])
    doc += text + '\n'

print(doc)

