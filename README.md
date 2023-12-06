# NLPFinalProject

How to run:
1. Make sure all needed packages have been installed
2. Unzip articles1.zip, articles2.zip, articles3.zip into the /data/ folder 
3. Before making the encoder-decoder model run the following command in terminal:
    python -m spacy download en_core_web_sm
4. Run the clean_datasets.ipynb from top to bottom
5. Run tf_model.ipynb to create and save the transformer model.
6. Run ed_model.ipynb to create and save the encoder-decoder model.
7. Run ed_predict.ipynb to generate headlines for 1000 articles using the encoder-decoder model.
    Note: if results_ed.csv exists with 1000 generated headlines, delete the file to generate new headlines
8. Run tf_predict.ipynb to generate headlines for 1000 articles using the transformer model.
    Note: if results_tf.csv exists with 1000 generated headlines, delete the file to generate new headlines
9. Run the models_evaluation.ipynb to get metrics for the generated headlines.

File Structure:
- /data/:
    - /cleaned_data/:
        - clean_articles1.csv
        - clean_articles2.csv
        - clean_articles3.csv
    - articles1.zip
    - articles2.zip
    - articles3.zip
    - ed_cleaned_data.csv       -> data used for training and testing of encoder-decoder model
    - results_ed.csv            -> resulting headline generation for the encoder-decoder model
    - results_tf.csv            -> resulting headline generation for the transformer model
    - test_data.txt             -> validation data for the transformer model
    - train_data.txt            -> training data for the transformer model
- /src/
    - clean_datasets.ipynb      -> Creates /cleaned_data/ files, test_data.txt, and train_data.txt
    - ed_model.ipynb            -> Generates and trains the encoder-decoder model
    - ed_predict.ipynb          -> Generates headlines for articles using the encoder-decoder model
    - models_evaluation.ipynb   -> Evaluates the encoder-decoder and transformer models using rouge scores
    - tf_model.ipynb            -> Generates and trains the transformer model
    - tf_predict.ipynb          -> Generates headlines for articles using the transformer model
- /trained_models/
    - /gpt2-summmarization/     -> Fine-tuned GPT-2 model
    - decoder_model.h5
    - encoder_model.h5