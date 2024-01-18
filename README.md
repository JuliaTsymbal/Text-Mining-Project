# Text Classification and Text Summarization of BBC News

## Table of contents
* [Abstract](#abstract)
* [Paper](https://github.com/JuliaTsymbal/Text-Mining-Project/blob/main/ReportTM%26S.pdf) and [Slides](https://github.com/JuliaTsymbal/Text-Mining-Project/blob/main/SlideTM%26S.pdf)
* [Requirements](#requirements)
* [BBC News: Data and Text Pre-processing](#BBC-News-data-and-text-pre-processing)
* [Text Classification](#text-classification)
* [Extractive Text Summarization](#extractive-text-summarization)
* [Abstractive Text Summarization](#abstractive-text-summarization)
* [Status](#status)
* [Contributors](#contributors)

<a id="abstract"></a>
## Abstract


In our project, we utilized a dataset is available on the Kaggle platform, comprising [BBC news articles](https://www.kaggle.com/datasets/pariza/bbc-news-summary) categorized into different topics. The goal is to perform two key text-mining tasks: text classification and text summarization (extractive and abstraction).

Text classification is done using two textual representations: TF-IDF and Word2Vec. For modeling were chosen Logistic Regression, Decision tree, Random forest, SVM, and XGBoost. In the notebook 'TextClassification' you will find all the code and results.

Extractive summarization is done using spaCy and PyTextRank. In the notebook 'TextRank' you will find all the code and results.

Abstractive summarization is done using the model BART. In the notebook 'BART' you will find all the code and results.


<a id="requirements"></a>
## Requirements


- python 3.10.12
- matplotlib==3.7.1
- nltk==3.8.1
- pandas==1.5.3
- numpy==1.23.5
- spacy==3.6.1
- pytextrank==3.2.5
- rouge-score==0.1.2
- gensim==4.3.2
- xgboost==2.0.3
- scikit-learn==1.2.2
- torch==2.1.0+cu121
- transformers==4.35.2


<a id="BBC-News-data-and-text-pre-processing"></a>
## BBC News: Data and Text Pre-processing

### Step 0. Prepare Files

All code is running in Colab. Before usage make sure you have uploaded all files to your Google Drive and changed the path to your folder.  

### Step 1. Download and extract the dataset

A dataset is available on the Kaggle platform, comprising [BBC news articles](https://www.kaggle.com/datasets/pariza/bbc-news-summary) or you can use the `archive.zip` file with all data. For convenience, we have also added an already generated csv file `raw_data.csv` for direct use for preprocessing. Run `Preprocessing.ipynb` to get clear data for text classification. Arter run you obtain `clean_text.csv` file for using it in the next steps.


<a id="text-classification"></a>
## Text Classification 

Run `TextClassification.ipynb` script which will perform classification based on tf-idf and word2vec text representation. All these representations are there. Models used for classification are:
- LogisticRegression
- Decision tree
- Random forest
- SVM
- XGBoost

Then, evaluate the resulting best model on the test set concerning:
- Recall
- Precision
- Accuracy
- F1-score


<a id="extractive-text-summarization"></a>
## Extractive Text Summarization  

Run `TextRank.ipynb` script which will perform text summarization using spacy and pytextrank. 

Then, evaluate summaries quality using 
- Rouge1 
- RougeL
- Bleu


<a id="abstractive-text-summarization"></a>
## Abstractive Text Summarization
Run `BART.ipynb` script which will perform abstractive text summarization using the pre-trained Bart model. 

Then, evaluate summaries quality using 
- Rouge1
- Rouge2 
- RougeL
- Bleu


<a id="status"></a>
## Status

Project is done


<a id="contributors"></a>
## Contributors

* [Julia Tsymbal](https://github.com/JuliaTsymbal)
* [Paola Cavana](https://github.com/pcavana)
