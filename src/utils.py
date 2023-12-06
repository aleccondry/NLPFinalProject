# Utility functions for the project

import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import re

def load_article_data(path: str) -> dict[str, pd.DataFrame]:
    """
    Loads data from csv files at the given path and returns a dict of dataframes
    in the format: {filename: dataframe}
    """
    article_data = {}
    for filename in os.listdir(path):
        if "." in filename and filename.split('.')[1] == 'csv':
            df = pd.read_csv(os.path.join(path, filename), encoding='utf-8')
            article_data[filename] = df
    return article_data

def clean_data():
    """
    To be implemented later
    Will clean data and return dataframes with only valid, important data & consistent column names
    """
    articles = load_article_data(path='../data/')
    columns_kept = ["title", "content", "Headline", "Article text", "Article", "Heading"]
    clean_data_path = '../data/cleaned_data/'
    if not os.path.exists(clean_data_path):
        os.makedirs(clean_data_path)
    for filename, df in zip(articles.keys(), articles.values()):
        for col in df.columns:
            if not any(col == ck for ck in columns_kept):
                df.drop(columns=[col], inplace=True)
            elif col == "title" or col == "Headline" or col == "Heading":
                df.rename(columns={col: "headline"}, inplace=True)
            elif col == "content" or col == "Article text" or col == "Article":
                df.rename(columns={col: "article"}, inplace=True)
        if df.columns.tolist()[0] == 'article':
            df = df[df.columns.tolist()[-1::-1]]
        print(df.columns)
        df.to_csv(clean_data_path + "clean_" + filename, index=False)
        print()
    return


def build_tokenizer(data_tr, data_val, max_len):
    # Prepare a tokenizer on training data
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(list(data_tr))

    thresh = 2
    cnt = 0
    tot_cnt = 0
    for key, value in tokenizer.word_counts.items():
        tot_cnt = tot_cnt + 1
        if value < thresh:
            cnt = cnt + 1
        
    print("% of rare words in vocabulary: ", (cnt / tot_cnt) * 100)

    # Prepare a tokenizer, again -- by not considering the rare words
    tokenizer = Tokenizer(num_words = tot_cnt - cnt) 
    tokenizer.fit_on_texts(list(data_tr))

    # Convert text sequences to integer sequences 
    tr_seq = tokenizer.texts_to_sequences(data_tr) 
    val_seq = tokenizer.texts_to_sequences(data_val)

    # Pad zero upto maximum length
    tr = pad_sequences(tr_seq,  maxlen=max_len, padding='post')
    val = pad_sequences(val_seq, maxlen=max_len, padding='post')

    return tokenizer, tr, val

def text_strip(column):
    for row in column:
        #ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!
        row=re.sub("(\\t)", ' ', str(row)).lower() #remove escape charecters
        row=re.sub("(\\r)", ' ', str(row)).lower() 
        row=re.sub("(\\n)", ' ', str(row)).lower()
        
        row=re.sub("(__+)", ' ', str(row)).lower()   #remove _ if it occors more than one time consecutively
        row=re.sub("(--+)", ' ', str(row)).lower()   #remove - if it occors more than one time consecutively
        row=re.sub("(~~+)", ' ', str(row)).lower()   #remove ~ if it occors more than one time consecutively
        row=re.sub("(\+\++)", ' ', str(row)).lower()   #remove + if it occors more than one time consecutively
        row=re.sub("(\.\.+)", ' ', str(row)).lower()   #remove . if it occors more than one time consecutively
        
        row=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower() #remove <>()|&©ø"',;?~*!
        
        row=re.sub("(mailto:)", ' ', str(row)).lower() #remove mailto:
        row=re.sub(r"(\\x9\d)", ' ', str(row)).lower() #remove \x9* in text
        row=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower() #replace INC nums to INC_NUM
        row=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower() #replace CM# and CHG# to CM_NUM
        
        
        row=re.sub("(\.\s+)", ' ', str(row)).lower() #remove full stop at end of words(not between)
        row=re.sub("(\-\s+)", ' ', str(row)).lower() #remove - at end of words(not between)
        row=re.sub("(\:\s+)", ' ', str(row)).lower() #remove : at end of words(not between)
        
        row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
        
        #Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(row))
        except:
            pass #there might be emails with no url in them
        
        row = re.sub("(\s+)",' ',str(row)).lower() #remove multiple spaces
    
        #Should always be last
        row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
        yield row