from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translation = pipeline('translation', model=model, tokenizer=tokenizer)

def translate_dialogue():
    translated_text = translation('how many colors are there in the rainbow?', max_length=40)[0]['translation_text']
    print(translated_text)

translate_dialogue()