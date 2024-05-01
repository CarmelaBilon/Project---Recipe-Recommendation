from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
import time
import numpy as np
from ast import literal_eval
import string
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords  # Import stopwords library
from nltk.tokenize import word_tokenize  # Import word tokenizer
import mean_embed_vectorizer as mev
import tfidf_vectorizer as tv
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.NER_clean.values:
        # Decode since the value is in bytes
        doc = eval(doc.decode()) # eval so the string is read as a list
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

# def get_window(corpus):
#     lengths = [len(doc) for doc in corpus]
#     avg_len = float(sum(lengths)) / len(lengths)
#     return round(avg_len)

def clean_ingredients(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char.isalnum() or char in " "])
    # Remove stopwords (optional)
    stop_words = stopwords.words('english')
    text = " ".join([word for word in word_tokenize(text) if word not in stop_words])
    # Lemmatize
    lemm_words = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    text = " ".join(lemm_words)
    text_list = text.split(' ')
    # text_list = text_list.sort()
    return text_list

def get_recommendations(data, N, scores):
    """
    Rank scores and output a pandas data frame containing all the details of the top N recipes.
    :param scores: list of cosine similarities
    """
    print('Inside getting recommendations...')
    # load in recipe dataset
    df_recipes = data
    print(df_recipes.head())
    # df_recipes = pd.read_csv('test.csv')
    # order the scores with and filter to get the highest N scores
    start_sort = time.time()
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    total_sort = time.time() - start_sort   
    print(f'Total sort time: {total_sort}')
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["title", "NER_clean", "score", "directions"])
    count = 0
    for i in top:
        print(f'Building top N...{i}')
        recommendation.at[count, "title"] = df_recipes["title"][i]
        recommendation.at[count, "NER_clean"] = df_recipes["NER_clean"][i]
        recommendation.at[count, "directions"] = df_recipes["directions"][i]
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation

def convert_bytes_to_str(in_bytes):
    out_str = in_bytes.decode('utf-8')
    return out_str

def get_recs(ingredients, N=5, mean=False):
    """
    Get the top N recipe recomendations.
    :param ingredients: comma seperated string listing ingredients
    :param N: number of recommendations
    :param mean: False if using tfidf weighted embeddings, True if using simple mean
    """
    # load in word2vec model
    model = Word2Vec.load("local_model_cbow.bin")
    # normalize embeddings
    # model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")
    # load in data
    print("Loading recipe list...")
    data = pd.read_parquet("recipe_cleaned.parquet")
    # data = pd.read_csv('test.csv')
    # parse ingredients
    # data["NER"] = data.NER.apply(ingredient_parser)
    # create corpus
    data = data[['title','ingredients','NER_clean','directions']]
    print(data.head())
    corpus = get_and_sort_corpus(data)
    # Converting NER_clean here to string for sorting later since get_and_sort_corpus above
    # have bytes to string conversion inside. This will make the corpus and the dataset the same
    data['NER_clean'] = data.apply(lambda x: convert_bytes_to_str(x['NER_clean']),axis=1)

    # create embeddings for input text
    # clean ingredients from AWS
    input = clean_ingredients(ingredients)
    print(f'ingredients are {input}')

    if mean:
        print('getting average embedding using MEAN.')
        # get average embdeddings for each document
        mean_vec_tr = mev.MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        print(type(doc_vec))
        assert len(doc_vec) == len(corpus)
        
    else:
        # use TF-IDF as weights for each word embedding
        start_time = time.time()
        print('getting TF IDF weights for word embedding')
        tfidf_vec_tr = tv.TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        print(type(doc_vec))
        tf_vec_time = time.time() - start_time
        print(f'Total time for TF IDF Vectorizer: {tf_vec_time}')
        assert len(doc_vec) == len(corpus)

    # get embeddings for ingredient doc
    if mean:
        print('In mean...')
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        print('In TF IDF...')
        start_time = time.time()
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)
        tf_vec_txnfrm_time = time.time() - start_time
        print(f'Total time for TF vec txnfrm: {tf_vec_txnfrm_time}')

    # get cosine similarity between input embedding and all the document embeddings
    print('Running cosine similarity...')
    start_time = time.time()
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    cos_sim_time = time.time() - start_time
    print(f'Total time for cos sim: {cos_sim_time}')
    start_time = time.time()
    scores = list(cos_sim)
    print(f'Total time for converting to list: {time.time() - start_time}')
    # Filter top N recommendations
    print('Getting recommendations...')
    recommendations = get_recommendations(data, N, scores)
    print(recommendations)
    return recommendations  

