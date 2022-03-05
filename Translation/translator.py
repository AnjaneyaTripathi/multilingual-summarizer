from googletrans import Translator
from nltk.tokenize import sent_tokenize
translator = Translator()

def translate(text, language):
  if language=='hindi':
    result = translator.translate(text, src='hi', dest='en')
  else:
    result = translator.translate(text, src='mr', dest='en')
  return result

# import nltk
# nltk.download('punkt')

# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from nltk.tokenize import sent_tokenize

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
# translation = pipeline('translation', model=model, tokenizer=tokenizer)

# def translate(text):
#     result = []
#     lines = sent_tokenize(text)
#     n = len(lines)
#     for i  in range(n):
#       translated_text = translation(lines, max_length=400)[i]['translation_text']
#       result.append(translated_text)
#     return result