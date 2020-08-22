"""
This file contains temporary utils that are quickly copied from the original code and must be refactored later.
"""
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


def nltk_stopword_tokenize(text, n, skip_unigrams=False, skip_n_grams=None):
    # text_unigrams = nltk.word_tokenize(text)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    tknzr = TweetTokenizer()
    if n == 1:
        # tokens = [t.lower() for t in text_unigrams if t.lower() not in stopwords.words('english')]
        # tokens = list(tokenizer(text))
        tokens = tknzr.tokenize(text)
        tokens = [t.lower() for t in tokens]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in string.punctuation]
    elif n >= 2:
        tokens = tknzr.tokenize(text)
        tokens = [t.lower() for t in tokens]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in string.punctuation]
        unigrams = tokens.copy()
        if skip_unigrams:
            tokens = []
        if skip_n_grams is None:
            skip_n_grams = []
        for i in range(2, n + 1):
            if i in skip_n_grams:
                continue
            ngrams = nltk.ngrams(unigrams, i)
            ngrams = [' '.join(tup) for tup in list(ngrams)]
            tokens = tokens + ngrams
    else:
        raise ValueError('unsupported n', n)
    return tokens


def rouge_precision(summary_tokens, text_tokens):
    overlap_count = 0
    if len(summary_tokens) == 0:
        # print(0)
        return 0, 0
    for elem in summary_tokens:
        if elem in text_tokens:
            overlap_count += 1
            # print(elem)
    precision = overlap_count / len(summary_tokens)
    # print('precision rouge ', overlap_count, precision)
    return precision, overlap_count


def rouge_recall(summary_tokens, text_tokens):
    overlap_count = 0
    if len(text_tokens) == 0:
        # print(0)
        return 0, 0
    for elem in summary_tokens:
        if elem in text_tokens:
            overlap_count += 1
            # print(elem)
    recall = overlap_count / len(text_tokens)
    # print('recall rouge ', overlap_count, recall)
    return recall, overlap_count


def calculate_scaled_f1_turn_distribution(chunk_tokens, turn_tokens) -> float:
    precision, overlap_count = rouge_precision(chunk_tokens, turn_tokens)
    recall, overlap_count = rouge_recall(chunk_tokens, turn_tokens)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    scaled_f1 = f1 * overlap_count
    return scaled_f1
