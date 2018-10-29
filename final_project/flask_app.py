import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from math import log
import nltk
nltk.download('stopwords')
nltk.download('punkt')

avito_ads = os.listdir('Avito')
avito_ads = ['Avito' + os.sep + a for a in avito_ads]

avito = []
for a in avito_ads:
    if a.startswith('Avito/www'):
        avito.append(a)

ads = dict(enumerate(avito))

mystem = Mystem()

def preprocessing(input_text, del_stopwords=True, del_digit=True):
    """
    :input: raw text
        1. lowercase, del punctuation, tokenize
        2. normal form
        3. del stopwords
        4. del digits
    :return: lemmas
    """
    russian_stopwords = set(stopwords.words('russian'))
    words = [x.lower().strip(string.punctuation + '»«–…') for x in word_tokenize(input_text)]
    lemmas = [mystem.lemmatize(x)[0] for x in words if x]

    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)

    return lemmas_arr

def prepare_data(avito):
    inv_idx_result = []
    lengths = {}
    ads_texts = []

    for a in avito:
        with open(a, 'r', encoding='utf-8') as f:
            f = f.read()
            ads_texts.append(f)

        avito_text = preprocessing(f)
        lengths[a] = len(avito_text)
        inv_idx_result.append(' '.join(avito_text))

    ads_texts = [i.replace('\xa0', '') for i in ads_texts]
    avito_texts = []
    for index, text in enumerate(ads_texts):
        avito_texts.append({'avito_text': text, 'index': index})

    ll = dict(enumerate(lengths.values()))
    avgdl = sum(lengths.values()) / len(lengths)

    return inv_idx_result, lengths, avito_texts, ll, avgdl

def get_inv_index(ii_data, ads):
    count = CountVectorizer()
    X = count.fit_transform(ii_data)
    key = [k for k in ads.keys()]
    term_doc_matrix = pd.DataFrame(X.toarray(), index=key, columns=count.get_feature_names())
    term_doc_matrix = [term_doc_matrix.index.tolist(), term_doc_matrix.columns.tolist(), term_doc_matrix.values.tolist()]

    return term_doc_matrix

def get_inv_idx(term_doc_matrix) -> dict:
    """
    Create inverted index by input doc collection
    :return: inverted index
    """
    count_idf = {}
    ads = term_doc_matrix[0]
    words = term_doc_matrix[1]
    freq = term_doc_matrix[2]
    N = len(ads)

    for i, j in enumerate(words):
        n = 0
        for f in freq:
            count = f[i]
            if count != 0:
                n += 1

        idf = log((N - n + 0.5) / (n + 0.5))
        count_idf[j] = idf

    return count_idf

k1 = 2.0
b = 0.75

def score_BM25(idf, qf, dl, avgdl, k1, b) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    return idf * (k1 + 1) * qf / (qf + k1 * (1 - b + b * dl / avgdl))

def get_search_result(query, c_idf, term_doc_matrix, ll, avgdl, avito_texts) -> list:
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """
    inv_idx_result = []
    res_text = []

    for q in query:
        if q in term_doc_matrix[1]:
            idx = term_doc_matrix[1].index(q)
            for i, a in enumerate(term_doc_matrix[0]):
                qf = term_doc_matrix[2][i][idx]
                dl = ll[a]
                idf = c_idf[q]
                okapi = score_BM25(idf, qf, dl, avgdl, k1, b)
                inv_idx_result.append({'index': a, 'okapi_score': okapi})

    res_inv_index = sorted(inv_idx_result, key=lambda k: k['okapi_score'], reverse=True)[:5]
    for i in res_inv_index:
        for j in avito_texts:
            if i['index'] == j['index']:
                res_text.append(j['avito_text'].replace('\n', ' *** '))

    return res_text

def search(query, c_idf, term_doc_matrix, ll, avgdl, avito_texts):
    query = preprocessing(query)
    result = get_search_result(query, c_idf, term_doc_matrix, ll, avgdl, avito_texts)

    return result

inv_idx_result, lengths, avito_texts, ll, avgdl = prepare_data(avito)
term_doc_matrix = get_inv_index(inv_idx_result, ads)
c_idf = get_inv_idx(term_doc_matrix)


from flask import Flask
from flask import url_for, render_template, request, redirect

app = Flask(__name__)

@app.route('/')
def index():
    if request.args:
        print(request.args)
        query = request.args['query']
        results = search(query, c_idf, term_doc_matrix, ll, avgdl, avito_texts)
        print(results)

        return render_template('results.html', results=results,
                                               query=query)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
