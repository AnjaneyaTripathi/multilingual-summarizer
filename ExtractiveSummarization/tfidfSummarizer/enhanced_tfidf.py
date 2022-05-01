import re
import nltk
import math
import pickle as pkl
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import tnt
from nltk.corpus import indian

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('indian')

# Get sentences from the file
def clean_text(file_name, language):
    with open('./data/original_docs/english/' + file_name + '.txt', 'r', encoding="UTF-8") as file:
        text = file.read()
        text = removeBrackets(text)
    if language=='hindi':    
        article = text.split('.')
    else:
        article = text.split('.')
    sentences = []
    for sentence in article:
        sentences.append(sentence)
    sentences.pop() 
    
    return sentences

# counting the number of words in the document (sentence)
def cnt_words(sent):
    cnt = 0
    words = word_tokenize(sent)
    for word in words:
        cnt = cnt + 1
    return cnt
   
# getting data about each sentence (frequency of words) 
def cnt_in_sent(sentences):
    txt_data = []
    i = 0
    maxi = 0
    for sent in sentences:
        i = i + 1
        cnt = cnt_words(sent)
        maxi = max(maxi, cnt)
        temp = {'id' : i, 'word_cnt' : cnt}
        txt_data.append(temp)
    return txt_data, maxi

# Remove content in bracket throughout text
def removeBrackets(txt):
    x = re.sub("[(].*[)] ", "", txt)
    return x

# Find number of urls and emails in each sentence
def cnt_url_email(sent):
    url = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    email = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    res1 = re.findall(url,sent) 
    res2 = re.findall(email, sent)
    return len(res1) + len(res2)

# Find number of special characters in each sentence
def cnt_special_chars(sent):
    chars = ['#', '%', '&']
    cnt = 0
    for char in chars:
        cnt = cnt + sent.count(char)
    return cnt

# Find number of numeric words in each sentence. Later extend to differentiate monetary words, dates, measurements and normal numbers
def cnt_numbers(sent):
    regex = r'[+|-]?[0-9]+'
    res = re.findall(regex, sent)
    return len(res)

# Find number of words in inverted commas in each sentence
def cnt_quotes(sent):
    regex = r"(\'.*?\')|(\".*?\")"
    res = re.findall(regex, sent)
    cnt = 0
    for quote in res:
        cnt = cnt + cnt_words(quote[0])
    return cnt

# Find number of words from the title that occur in each sentence
def cnt_title_words(sent, title):
    t = word_tokenize(title)
    words = word_tokenize(sent)
    cnt = 0
    for word in words:
        if word in t:
          cnt = cnt + 1
    return cnt

def pos_tagging_marathi(sentence):
  wordsList = nltk.word_tokenize(sentence)
  tagged_words = marathi_pos.tag(wordsList)
  print(tagged_words)
  count=0
  for word in tagged_words:
    if 'NN' in word[1]:
      count+=1
  return count

def pos_tagging_hindi(sentence):
  train_data = indian.tagged_sents('hindi.pos')
  tnt_pos_tagger = tnt.TnT()
  tnt_pos_tagger.train(train_data)
  tagged_words = (tnt_pos_tagger.tag(nltk.word_tokenize(sentence)))
  count=0
  for word in tagged_words:
    if 'NN' in word[1]:
      count+=1
  return count

def pos_tagging_english(sentence):
  wordsList = nltk.word_tokenize(sentence)
  tagged_words = nltk.pos_tag(wordsList)
  count=0
  for word in tagged_words:
    if 'NN' in word[1]:
      count+=1
  return count

def sentence_position(i, count):
  return (count-i)/count

# creating a dictionary of words for each document (sentence)
def freq_dict(sentences):
    i = 0
    freq_list = []
    for sent in sentences:
        i = i + 1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word] = freq_dict[word] + 1
            else:
                freq_dict[word] = 1
            temp = {'id' : i, 'freq_dict' : freq_dict}
        freq_list.append(temp)
    return freq_list
   
# calculating the term frequency 
def calc_TF(text_data, freq_list):
    tf_scores = []
    for item in freq_list:
        ID = item['id']
        for k in item['freq_dict']:
            temp = {
                'id': item['id'],
                'tf_score': item['freq_dict'][k]/text_data[ID-1]['word_cnt'],
                'key': k
                }
            tf_scores.append(temp)
    return tf_scores
    
#calculating the inverse document frequency
def calc_IDF(text_data, freq_list):
    idf_scores =[]
    cnt = 0
    for item in freq_list:
        cnt = cnt + 1
        for k in item['freq_dict']:
            val = sum([k in it['freq_dict'] for it in freq_list])
            temp = {
                'id': cnt, 
                'idf_score': math.log(len(text_data)/(val+1)), 
                'key': k}
            idf_scores.append(temp)
    return idf_scores

# calculating TFIDF value
def calc_TFIDF(tf_scores, idf_scores):
    tfidf_scores = []
    for j in idf_scores:
        for i in tf_scores:
            if j['key'] == i['key'] and j['id'] == i['id']:
                temp = {
                    'id': j['id'],
                    'tfidf_score': j['idf_score'] * i['tf_score'],
                    'key': j['key']
                    }
                tfidf_scores.append(temp)
    return tfidf_scores

# giving each sentence a score
def sent_scores(tfidf_scores, sentences, text_data):
    sent_data = []
    for txt in text_data:
        score = 0
        for i in range(0, len(tfidf_scores)):
            t_dict = tfidf_scores[i]
            if txt['id'] == t_dict['id']:
                score = score + t_dict['tfidf_score']
        temp = {
            'id': txt['id'],
            'score': score-1,
            'sentence': sentences[txt['id']-1]}
        sent_data.append(temp)
    return sent_data

def get_summary(filename, title, language):
    sentences = clean_text(filename, language)
    length = len(sentences)
    # Higher preference to longer sentences
    txt_data, maxi = cnt_in_sent(sentences)
    for sentence in txt_data:
        sentence['num_words'] = sentence['word_cnt']/maxi
    for i in range(length):
        sentence = sentences[i]
        # Lower preference to sentences with urls and emails
        url_email = cnt_url_email(sentence)
        txt_data[i]['url_email'] = url_email/txt_data[i]['word_cnt']
        # Higher preference to sentences with special characters
        special_chars = cnt_special_chars(sentence)
        txt_data[i]['special_chars'] = special_chars/txt_data[i]['word_cnt']
        # Higher preference to sentences with numbers
        numbers = cnt_numbers(sentence)
        txt_data[i]['numbers'] = numbers/txt_data[i]['word_cnt']
        # Higher preference to sentences with quotes
        quote_chars = cnt_quotes(sentence)
        txt_data[i]['quote_chars'] = quote_chars/txt_data[i]['word_cnt']
        # Higher preference to sentences that contain title words
        title_words = cnt_title_words(sentence, title)
        txt_data[i]['title_words'] = title_words/txt_data[i]['word_cnt']
        # Higher preference to sentences with more nouns
        if language=='hindi':
            nouns = pos_tagging_hindi(sentence)
        else:
            nouns = pos_tagging_english(sentence)
        txt_data[i]['nouns'] = nouns/txt_data[i]['word_cnt']
        txt_data[i]['position'] = sentence_position(i, len(sentences))
    freq_list = freq_dict(sentences)
    text_data, num = cnt_in_sent(sentences)
    # TF-IDF Scores
    tf_scores = calc_TF(text_data, freq_list)
    idf_scores = calc_IDF(text_data, freq_list)
    
    tfidf_scores = calc_TFIDF(tf_scores, idf_scores)
    
    sent_data = sent_scores(tfidf_scores, sentences, text_data)
    num = 0
    for sent in sent_data:
      txt_data[num]['tf-idf'] = sent['score']
      num+=1
    
    # Generate final sentence score
    final_scores = []
    all_scores = 0
    for i in range(length):
      d = {}
      d['sentence'] = sentences[i]
      score = txt_data[i]['num_words'] - txt_data[i]['url_email'] + txt_data[i]['special_chars'] + txt_data[i]['numbers'] + txt_data[i]['quote_chars'] + txt_data[i]['title_words'] + txt_data[i]['nouns'] + txt_data[i]['position'] + txt_data[i]['tf-idf']
      d['score'] = score
      all_scores+=score
      final_scores.append(d)
      
    cut_off = all_scores/length    
    
    # Generate summary
    summary = ''
    for sent in final_scores:
      if sent['score']>cut_off:
        summary = summary + sent['sentence'] + '. '
    
    return summary

filenm = 'extractiveSummarization/tfidfSummarizer/marathi_pos.pickle'
marathi_pos = pkl.load(open(filenm, 'rb'))
    
# print(get_summary('cricket_hindi.txt', "भारत और साउथ अफ्रीका की टेस्ट",  'hindi'))
# print(get_summary('blockchain_english.txt', 'Blockchain', 'english'))
# print(get_summary('mansi_marathi.txt', 'मानसी शिरीष कणेकर', 'marathi'))