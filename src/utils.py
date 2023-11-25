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
        df = pd.read_csv(os.path.join(path, filename), encoding='utf-8')
        article_data[filename] = df
    return article_data

def clean_data():
    """
    To be implemented later
    Will clean data and return dataframes with only valid, important data & consistent column names
    """
    return