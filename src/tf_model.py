from transformers import AutoTokenizer, AdamWeightDecay, TFAutoModelForCausalLM
import tensorflow as tf
from tensorflow.python.client import device_lib
import utils
import os
import pandas as pd
from datasets import load_dataset
import random
import argparse

# Constants
TLDR = ' TL;DR '
MAX_LEN = 512
NUM_ELEMENTS = 50000
BATCHES = 2
NEW_DATA = True
SAVE_MODEL_PATH = '../trained_models/gpt2-summarization-gpu'

def get_data():
    datapath = "../data/cleaned_data/"
    if not os.path.exists(datapath):
        utils.clean_data()
    all_articles_dict = utils.load_article_data(path=datapath)
    del all_articles_dict['clean_Articles.csv']
    del all_articles_dict['clean_CNN_Articels_clean.csv']
    all_articles_df = pd.concat([df for df in all_articles_dict.values()])
    return all_articles_df

def clean_data(df: pd.DataFrame):
    # Format data by: article TL;DR headline
    all_articles = df.values.tolist()
    all_articles = [x[1].strip() + " TL;DR " + x[0].strip().replace(' - The New York Times', '') 
                    for x in all_articles 
                    if isinstance(x[0], str) and isinstance(x[1], str)][0:NUM_ELEMENTS]
    all_articles = pad_and_truncate_data(all_articles)
    return all_articles

def pad_and_truncate_data(dataset):
    """
    Format data to always contain the TL;DR and the entire headline. Truncate the article such that
    the whole string becomes MAX_LEN long.
    """
    ARTICLE_LEN = MAX_LEN - len(TLDR)
    result = []
    for d in dataset:
        try:
            article, headline = d.split(' TL;DR ')
            result.append(article[0:ARTICLE_LEN - len(headline)] + TLDR + headline)
        except ValueError:
            continue
    return result


def create_training_split_files(data):
    random.seed(11)
    random.shuffle(data)
    TRAIN_SPLIT = 0.9
    END_IDX = int(len(data) * TRAIN_SPLIT)
    with open("../data/train_data.txt", "w", encoding='utf-8') as txt_file:
        for line in data[0:END_IDX]:
            txt_file.write(line + "\n") # works with any number of elements in a line
    with open("../data/test_data.txt", "w", encoding='utf-8') as txt_file:
        for line in data[END_IDX:]:
            txt_file.write(line + "\n") # works with any number of elements in a line


def ld_dataset():
    datasets = load_dataset("text", data_files={"train": '../data/train_data.txt', "validation": '../data/test_data.txt'})
    print(datasets["train"][10])
    print(len(datasets['train']))
    print(len(datasets['validation']))
    return datasets


class TokenizerWrapper:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"],
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_len // 4)


def tokenize_data(datasets):
    # Add labels to tokenized data
    def add_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples
    
    tokenizer_wrapper = TokenizerWrapper(tokenizer, MAX_LEN)
    tokenized_datasets = datasets.map(
        tokenizer_wrapper.tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    print(tokenized_datasets["train"][1])
    print(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))
    print(len(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"])))

    lm_datasets = tokenized_datasets.map(
        add_labels,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    # Prepare training and validation datasets
    train_set = model.prepare_tf_dataset(
        lm_datasets["train"],
        shuffle=True,
        batch_size=BATCHES,
    )

    validation_set = model.prepare_tf_dataset(
        lm_datasets["validation"],
        shuffle=False,
        batch_size=BATCHES,
    )

    return train_set, validation_set


def train_model(model_to_train):
    # Compile and train model
    with tf.device('/GPU:0'):
        optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        model_to_train.compile(optimizer=optimizer)
        model_to_train.fit(train_set, 
                            validation_data=validation_set, 
                            epochs=2,  
                            verbose=True)
    print(f"Saving model at {SAVE_MODEL_PATH}")
    # loss: 2.4180 - val_loss: 2.4102 for 2 epochs 50,000
    model.save_pretrained('../trained_models/gpt2-summarization-gpu')

if __name__ == "__main__":
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(f'GPUs: {tf.config.list_physical_devices("GPU")}')
    tf.debugging.set_log_device_placement(False)

    # Load pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = TFAutoModelForCausalLM.from_pretrained('gpt2')
    
    if ("test_data.txt" in os.listdir('../data/') and "train_data.txt" in os.listdir('../data/')) and not NEW_DATA:
        print("Skipping data formation...")
    else:
        print("Getting data")
        all_articles_df = get_data()

        print("Cleaning data")
        all_articles = clean_data(all_articles_df)

        print("Creating training split")
        create_training_split_files(all_articles)

    print("Loading dataset")
    article_datasets = ld_dataset()

    print("Tokenizing data")
    train_set, validation_set = tokenize_data(article_datasets)

    print("Training...")
    train_model(model)





