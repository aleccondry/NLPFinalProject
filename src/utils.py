# Utility functions for the project

import os
import pandas as pd


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
