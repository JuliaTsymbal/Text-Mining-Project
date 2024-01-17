# Text-Mining-Project
This repository contains text classification and text summarization solutions

In our project, we utilized a dataset available on the Kaggle platform, comprising BBC news articles (https://www.kaggle.com/datasets/pariza/bbc-news-summary) categorized into different topics. The goal is to perform two key text-mining tasks: text classification and text summarization (extractive and abstraction).

The dataset can be found in the file 'archive.zip' or using directly composed by us csv file 'raw_data.csv'. First of all, it's necessary to perform a 'Preprocessing' notebook to get 'raw_data.csv' and all preprocessing steps for classification mainly. 

Text classification is done using two textual representations: TF-IDF and Word2Vec. For modeling were chosen Logistic Regression, Decision tree, Random forest, SVM, and XGBoost. In the notebook 'TextClassification' you will find all the code and results.

Extractive summarization is done using spaCy and PyTextRank. In the notebook 'TextRank' you will find all the code and results.

Abstractive summarization is done using the model BART. In the notebook 'BART' you will find all the code and results.
