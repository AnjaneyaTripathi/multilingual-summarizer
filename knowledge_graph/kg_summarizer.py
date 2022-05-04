from transformers import pipeline

import en_core_web_lg
nlp = en_core_web_lg.load()


def bart_large_cnn(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30,
                         do_sample=False, truncation=True)
    return summary
