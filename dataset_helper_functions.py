import pandas as pd
import nltk
import os
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
import string, re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sentence_transformers import SentenceTransformer
import pickle

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path}' is empty.")

def get_word_count(df, label):
    spam_word = []
    for msg in df[df['Label'] == label]['CleanMessage'].tolist():
        for word in msg.split():
            spam_word.append(word)
    return pd.DataFrame(Counter(spam_word).most_common(30))

def encode_labels(df):
    encoder = LabelEncoder()
    df['Classes'] = encoder.fit_transform(df['Label'])
    return df

def get_training_data(df, model_path):
    tf = TfidfVectorizer(max_features = 3000)
    X = tf.fit_transform(df['CleanMessage']).toarray()
    y = df['Classes'].values
    with open(os.path.join(model_path, 'tf-idf_model.pkl'), 'wb') as f:
        pickle.dump(tf, f)
    return X, y

def get_prediction_data(df, model_path):
    with open(os.path.join(model_path, 'tf-idf_model.pkl'), 'rb') as f:
        tf = pickle.load(f)
    X = tf.transform(df['CleanMessage']).toarray()
    return X

def get_clean_text(text):
    port_stemmer = PorterStemmer()

    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text= " ".join(text)
    text = [char for char in text if char not in string.punctuation]
    text = ''.join(text)
    text = [char for char in text if char not in re.findall(r"[0-9]", text)]
    text = ''.join(text)
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)

def get_clean_df(df):
    df['CleanMessage'] = df['Message'].apply(get_clean_text)
    return df
def check_missing_values(df):
    return df.isnull().sum()

def check_duplicates(df):
    return df.duplicated().sum()

def remove_duplicates(df):
    return df.drop_duplicates(keep = 'first').reset_index(drop = True)

def check_percentage_of_labels(df):
    return df['Label'].value_counts(normalize = True) * 100

def get_number_of_chars(df):
    df['NumOfChars'] = df['Message'].apply(len)
    return df

def get_number_of_words(df):
    df['NumOfWords'] = df['Message'].apply(lambda x : len(nltk.word_tokenize(x)))
    return df

def get_number_of_sents(df):
    df['NumOfSents'] = df['Message'].apply(lambda x : len(nltk.sent_tokenize(x)))
    return df

def data_cleaning(df):
    print("Null data count for each column:\n", check_missing_values(df))
    print("\nDublicate Count: ", check_duplicates(df))
    df = remove_duplicates(df)
    print("Dublicate Count: ", check_duplicates(df))
    print("DataFrame Shape: ", df.shape)
    print("\nPercentage of Labels: \n", check_percentage_of_labels(df))
    return df

def data_preprocessing(df):
    df = get_number_of_chars(df)
    df = get_number_of_words(df)
    df = get_number_of_sents(df)
    print("All Data Describe:\n", df.describe())
    print("\nHam Data Describe:\n", df[df['Label'] == 'ham'].describe())
    print("\nSpam Data Describe:\n", df[df['Label'] == 'spam'].describe())
    print("\nData Correlation:\n", df[['NumOfChars', 'NumOfWords', 'NumOfSents']].corr())
    df = get_clean_df(df)
    print("Cleaned Data: \n", df.head())
    print("Most common 'Ham' words: \n", get_word_count(df, "ham"))
    print("\nMost common 'Spam' words: \n", get_word_count(df, "spam"))
    df = encode_labels(df)
    return df

def data_preprocessing_for_predict(df):
    df = get_number_of_chars(df)
    df = get_number_of_words(df)
    df = get_number_of_sents(df)
    df = get_clean_df(df)
    return df

def prepare_data_for_method_ml(file_path, model_path):
    df = read_csv_file(file_path)
    print("Initial Data: \n", df.head())
    df = data_cleaning(df)
    df = data_preprocessing(df)
    X, y = get_training_data(df, model_path)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    return X, y

def prepare_data_for_method_ml_predict(text, model_path):
    df = pd.DataFrame()
    df['Message'] = [text]
    df = data_preprocessing_for_predict(df)
    X = get_prediction_data(df, model_path)
    print("X shape: ", X.shape)
    return X

def get_embeddings(df):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["Message"].tolist(), convert_to_numpy=True)
    return embeddings

def prepare_data_for_transformer(file_path):
    df = read_csv_file(file_path)
    print("Initial Data: \n", df.head())
    embeddings = get_embeddings(df)
    embeddings_df = pd.DataFrame(embeddings)
    df = encode_labels(df)
    y = df["Classes"].values
    X = embeddings_df.values
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    return X, y

def prepare_data_for_transformer_predict(text):
    df = pd.DataFrame()
    df['Message'] = [text]
    embeddings = get_embeddings(df)
    embeddings_df = pd.DataFrame(embeddings)
    X = embeddings_df.values
    print("X shape: ", X.shape)
    return X