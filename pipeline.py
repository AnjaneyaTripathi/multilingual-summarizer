# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:35:45 2022

@author: Isha
"""

import sys
sys.path.insert(0, 'ExtractiveSummarization/tfidf-summarizer')

import enhanced_tfidf as summarizer

filename = 'blockchain_english.txt'
title = 'blockchain'
language = 'english'

with open(filename, 'r', encoding="UTF-8") as file:
        text = file.read()
        print(len(text))
        if(len(text)>1000):
            text = summarizer.get_summary(filename, title, language)
        

sys.path.insert(0, 'Translation')

import translator

if(language != 'english'):
    text = translator.translate(text, language)

print(text)
