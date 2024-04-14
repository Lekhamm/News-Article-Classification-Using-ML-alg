# News Article Classification Using Machine Learning and NLP:
This repository contains Python scripts for text classification of BBC News articles using various machine learning algorithms, including Multinomial Naive Bayes and Gaussian Naive Bayes.

# Dataset
The dataset used for this project is the BBC News dataset, which contains news articles categorized into five categories: business, tech, politics, sport, and entertainment. The dataset consists of text data and corresponding category labels.

# Requirements
Make sure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- wordcloud
- scikit-learn

# Explanation
- naive_bayes_classification.py: This script performs the following tasks:
. Import necessary libraries and dataset.
. Preprocess the text data by removing tags, special characters, stopwords, and lemmatizing words.
. Convert text data into numerical features using CountVectorizer.
. Split the dataset into training and testing sets.
. Train Multinomial Naive Bayes and Gaussian Naive Bayes models.
. Evaluate the models' accuracy and visualize the confusion matrix.

-naive_bayes_classifier.py: This script implements a custom Naive Bayes classifier using a bag-of-words model. It includes the following steps:
. Define the NaiveBayesClassifier class.
. Train the classifier on the training data.
. Predict the labels for the test data.
. Evaluate the classifier's accuracy and visualize the confusion matrix.

# Results
After running the scripts, you will see the accuracy scores of the models and visualizations of the confusion matrices.
