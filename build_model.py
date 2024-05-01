# This module will build the model "local_model_cbow.bin"

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords  # Import stopwords library
from nltk.tokenize import word_tokenize  # Import word tokenizer
import gensim
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import time

# Downloading the necessary components for nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

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
  return text_list

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.NER_clean.values:
        # Decode since the value is in bytes
        # doc = eval(doc.decode()) # eval so the string is read as a list
        # doc = eval(doc.decode())
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

# calculate average length of each document
def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_len = float(sum(lengths)) / len(lengths)
    return round(avg_len)

recipe_file = 'recipes.csv'

#Read the csv as a pandas df
df = pd.read_csv(recipe_file)

# Remove nulls
df.dropna(inplace=True)

print(df.info())

# Drop duplicates
duplicate_rows = df.duplicated()
print("Number of duplicate rows: ", duplicate_rows.sum())

print(df.head())

start_time = time.time()
# Apply cleaning function to ingredients column
print('Starting cleaning function...')
df['NER_clean'] = df['NER'].apply(clean_ingredients)
clean_time = time.time() - start_time
print(f'Input data cleaned in :{clean_time}')

# to generate the recipe_cleaned.parquet file
df.to_parquet('recipe_cleaned.parquet',compression='snappy')

# Create a new df for building the model with title, ingredients, NER_clean
df_for_model = df[['title','ingredients','NER_clean']]

# get corpus
print('Starting corpus sort...')
start_time = time.time()
corpus = get_and_sort_corpus(df_for_model)
print(f"Length of corpus: {len(corpus)}")
corpus_sort_time = time.time() - start_time
print(f'Corpus sort took: {corpus_sort_time}')

start_time = time.time()
print('Starting model build...')
model_cbow = Word2Vec(
corpus, sg=0, workers=8, window=get_window(corpus), min_count=1, vector_size=100
)
model_build_time = time.time() - start_time
print(f'Model build took: {model_build_time}')

#saving model to my directory
model_cbow.save('local_model_cbow_v2.bin')
print("Word2Vec model successfully trained")