from keybert import KeyBERT
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import math
import re
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

ROUGE = Rouge()
WORD = re.compile(r"\w+")
kw_model = KeyBERT()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
nltk.download('punkt')

def getFrequencyVector(text):
    words = WORD.findall(text)
    return Counter(words)

def getBLEUScore(candidate, reference):
    candidate_sentences = sent_tokenize(candidate)
    reference_sentences = sent_tokenize(reference)

    candidate_tokens = []
    reference_tokens = []

    for sentence in candidate_sentences:
        tokens = word_tokenize(sentence)
        candidate_tokens.append(tokens)

    for sentence in reference_sentences:
        tokens = word_tokenize(sentence)
        reference_tokens.append(tokens)

    result = corpus_bleu(reference_tokens, candidate_tokens)

    return result

def getBERTScore(candidate, reference):
    P, R, F1 = score(candidate, reference, lang="en", verbose=True)
    result = [P.mean(), R.mean(), F1.mean()]

    return result

def getROUGEScore(candidate, reference):
    result = ROUGE.get_scores(candidate, reference)

    return result

def getEmbeddedCosineScore(candidate, reference):
    candidate_sentences = sent_tokenize(candidate)
    reference_sentences = sent_tokenize(reference)

    candidate_embeddings = []
    reference_embeddings = []

    for sentence in candidate_sentences:
        candidate_embeddings.append(embed([sentence]))

    for sentence in reference_sentences:
        reference_embeddings.append(embed([sentence]))

    result = 0

    for vec1 in candidate_embeddings:
        temp = 0
        for vec2 in reference_embeddings:
            temp += cosine_similarity(vec1, vec2)[0][0]
        temp /= len(reference_embeddings)
        result += temp
    
    result /= len(candidate_embeddings)

    return result

def getFrequencyCosineScore(candidate, reference):
    candidate_vector = getFrequencyVector(candidate)
    reference_vector = getFrequencyVector(reference)

    intersection = set(candidate_vector.keys()) & set(reference_vector.keys())
    numerator = sum([candidate_vector[x] * reference_vector[x] for x in intersection])

    sum1 = sum([candidate_vector[x] ** 2 for x in list(candidate_vector.keys())])
    sum2 = sum([reference_vector[x] ** 2 for x in list(reference_vector.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def getKeyBERTScore(candidate, reference):
    reference_keywords = kw_model.extract_keywords(reference)
    candidate_keywords = kw_model.extract_keywords(candidate)

    list1 = []
    list2 = []

    for i in reference_keywords:
        if(len(i[0])>4):
            list1.append(i[0])
    for i in candidate_keywords:
        if(len(i[0])>4):
            list2.append(i[0])

    common = set(list1) & set(list2)
    result = len(common)/(len(list(set(list1)|set(list2))))

    return result

def evaluate(candidate, reference):
    bleu_score = getBLEUScore(candidate, reference)
    print('evaluated bleu score...', bleu_score)

    bert_score = getBERTScore(candidate, reference)
    print('evaluated bert score...', bert_score)

    rouge_score = getROUGEScore(candidate, reference)
    print('evaluated rouge score...', rouge_score)

    # fix a bug in this - getting [], not a vector
    embeddedCosineScore = getEmbeddedCosineScore(candidate, reference)
    print('evaluated embedded cosine score...', embeddedCosineScore)

    frequencyCosineScore = getFrequencyCosineScore(candidate, reference)
    print('evaluated frequency cosine score...', frequencyCosineScore)

    keybert_score = getKeyBERTScore(candidate, reference)
    print('evaluated keybert score...', keybert_score)

    return bleu_score, bert_score, rouge_score, embeddedCosineScore, frequencyCosineScore, keybert_score

filename = 'bc'

with open('./data/kg_input/' + filename + '.txt', 'r') as file:
    reference = file.read()

with open('./data/kg_input/' + filename + '.txt', 'r') as file:
    candidate = file.read()

# evaluate(candidate, reference)

print(getEmbeddedCosineScore(candidate, reference))