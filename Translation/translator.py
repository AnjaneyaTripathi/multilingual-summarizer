

import nltk
nltk.download('punkt')

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translation = pipeline('translation', model=model, tokenizer=tokenizer)

def translate(textfile,n):
    result = []

    for i  in range(n):
      translated_text = translation(textfile, max_length=400)[i]['translation_text']
      result.append(translated_text)
    return result

from nltk.tokenize import sent_tokenize
with open('article.txt') as f:
    lines = f.readlines()
text = lines[0]
lines = sent_tokenize(text)
texttran = translate(lines,len(lines))

print(texttran)
with open('translatedarticle.txt', 'w') as f:
    f.writelines(texttran)