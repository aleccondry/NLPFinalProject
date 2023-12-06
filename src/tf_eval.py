import utils
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from rouge_score import rouge_scorer, scoring
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
TLDR = ' TL;DR '
MAX_LEN = 512

def summarize_article(article):
    tokenized = tokenizer(article, return_tensors="np")
    outputs = model.generate(**tokenized, max_new_tokens=32, pad_token_id=50256)
    return tokenizer.decode(outputs[0])

def summarize_all_test_articles(ah, ph):
    with open('../data/test_data.txt', encoding='utf-8') as f:
        lines = f.readlines()
        preds = {'predicted headline': [], 'actual headline': []}
        for idx, line in enumerate(lines):
            article, actual_headline = line.strip().split(TLDR)
            article = article + TLDR
            ah[idx] = actual_headline
            predicted_headline = summarize_article(article).split(TLDR)[1].replace('<|endoftext|>', '.').strip()
            ph[idx] = predicted_headline
            preds['predicted headline'].append(predicted_headline)
            preds['actual headline'].append(actual_headline)
            print(f'{idx}:')
            print(f'\tactual: {actual_headline}')
            print(f'\tpredic: {predicted_headline}')
            if idx % 50 == 0 and idx != 0:
                preds_df = pd.DataFrame.from_dict(preds, orient='columns')
                preds_df.to_csv('../data/tf_results.csv')
        return ah, ph
    
def get_average_scores(rs):
    avg_rouge_scores = {}
    for dataset in rs:
        precision = 0
        recall = 0
        fmeasure = 0
        for score in rs[dataset]:
            precision += score['rouge1'].precision
            recall += score['rouge1'].recall
            fmeasure += score['rouge1'].fmeasure
        precision /= len(rs[dataset])
        recall /= len(rs[dataset])
        fmeasure /= len(rs[dataset])
        avg_rouge_scores[dataset] = scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)
    return avg_rouge_scores

def calculate_rouge_scores(ah, ph):    
    # calculate ROUGE score(s) for each dataset
    rs = {"tf": []}
    for idx in actual_headlines.keys():
        curr_score = scorer.score(ah[idx], ph[idx])
        if len(ph[idx]) > 0:
            rs['tf'].append(curr_score)
    return rs

def get_metrics(avg_scores, scores):
    avg_metrics = {'Precision': (avg_scores.precision), 'Recall': (avg_scores.recall), 'Fmeasure': (avg_scores.fmeasure)}
    metrics = {
        'Precision': [x['rouge1'].precision for x in scores],
        'Recall': [x['rouge1'].recall for x in scores],
        'fmeasure': [x['rouge1'].fmeasure for x in scores]
    }
    return avg_metrics, metrics

def create_plots(avg_met, met):
    # Plot average scores across all tested samples
    fig_avg, ax_avg = plt.subplots(layout='constrained')
    ax_avg.bar(avg_met.keys(), avg_met.values(), color=['blue', 'red', 'green'])
    ax_avg.set_ylabel('Metric Percentage')
    ax_avg.set_title('Average Metrics')
    plt.show()

    # Plot frequency of specific scores across all tested samples
    colors = ['blue', 'red', 'green']
    fig_hist, ax_hist = plt.subplots(1, 3, layout='constrained')
    for idx, key in enumerate(met.keys()):
        count, bin = np.histogram(met[key])
        ax_hist[idx].hist(bin[:-1], bin, weights=count, label=key, color=colors[idx])
        ax_hist[idx].set_xlabel(list(met.keys())[idx])
        ax_hist[idx].set_ylim(0, 15)
    ax_hist[0].set_ylabel('Frequency')
    fig_hist.suptitle('Frequency of Metrics')
    plt.show()

def load_results():
    df = pd.read_csv('../data/tf_results.csv')
    print(df)
    ah = {idx: x for idx, x in enumerate(df['actual headline'])}
    ph = {idx: x for idx, x in enumerate(df['predicted headline'])}
    print(ah)
    return ah, ph

if __name__ == "__main__":
    # can edit the different rouge scores we want; research which will be best
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = TFAutoModelForCausalLM.from_pretrained('../trained_models/gpt2-summarization-gpu/')

    NEW_DATA = True
    actual_headlines = {}
    predicted_headlines = {}
    if NEW_DATA: 
        actual_headlines, predicted_headlines = summarize_all_test_articles(actual_headlines, predicted_headlines)
    else:
        actual_headlines, predicted_headlines = load_results()
    rouge_scores = calculate_rouge_scores(actual_headlines, predicted_headlines)
    avg_rouge_scores = get_average_scores(rouge_scores)
    avg_metrics, metrics = get_metrics(avg_rouge_scores['tf'], rouge_scores['tf'])
    create_plots(avg_metrics, metrics)
