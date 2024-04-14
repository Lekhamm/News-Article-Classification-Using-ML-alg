# News Article Classification Using Machine Learning and NLP:
This repository contains Python scripts for text classification of BBC News articles using various machine learning algorithms, including Multinomial Naive Bayes, Gaussian Naive Bayes, and a custom Naive Bayes classifier.

# Dataset
The BBC News dataset consists of news articles published by the BBC in 2004-2005, categorized into five categories: business, tech, politics, sport, and entertainment. Each article is labeled with one of these categories. The dataset is balanced, meaning it contains roughly the same number of articles for each category.

Features:
- Text Data: The main feature of the dataset is the textual content of each news article. This includes headlines, summaries, and main body text.
- Labels: Each article is labeled with one of the five categories mentioned above.

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
naive_bayes_classification.py: This script performs the following tasks:
- Import necessary libraries and dataset.
- Preprocess the text data by removing tags, special characters, stopwords, and lemmatizing words.
- Convert text data into numerical features using CountVectorizer.
- Split the dataset into training and testing sets.
- Train Multinomial Naive Bayes and Gaussian Naive Bayes models.
- Evaluate the models' accuracy and visualize the confusion matrix.

naive_bayes_classifier.py: This script implements a custom Naive Bayes classifier using a bag-of-words model. It includes the following steps:
- Define the NaiveBayesClassifier class.
- Train the classifier on the training data.
- Predict the labels for the test data.
- Evaluate the classifier's accuracy and visualize the confusion matrix.

# Results
After running the scripts, you will see the accuracy scores of the models and visualizations of the confusion matrices.
