---
layout: post
title: "Crop recommendation system by supervised machine learning"
subtitle: "Use Logistic Regression, Gaussian Naive Bayes, Random Forest and XGBoost to help farmers make informed decision about cultivation of crops and achieved almost perfect accuracy"
background: '/img/bg-about.jpg'
---


# crop_recommendation_system_by_supervised_machine_learning
Use Logistic Regression, Gaussian Naive Bayes, Random Forest and XGBoost to help farmers make informed decision about cultivation of crops. <br>
The best model(random forest) achieves almost perfect accuracy to recommend the correct crop based on 7 features (N, P, K, temperature, humidity, ph, rainfall) <br>

[View Notebook and code](https://github.com/lizhiyidaniel/crop_recommendation_system_by_supervised_machine_learning/blob/main/crop_recommendation_system_by_machine_learning.ipynb)

# Overview and Background
Precision Agriculture is a management technique that is based on observing, measuring and responding to inter and intra-field variability in crops. <br>
With the avent of techniques such as GPS and GNSS, farmers and researchers could measure many variables such as crop yield, terrain features, organic mantter content, moisture levels, nitrogen levels, K and others important variables. These data could also be collected by sensor arrays and these real-time sensors could measure chlorophyll levels to plant water status and etc. <br>
All these could be used to optimize crop inputs such as water, fertilizer or chemicals. It could suggest farmers to grow the most optimal crop for maximum yeild and profit by these features. It could help farmers too reduce crop failure and take informed decision about farming strategy. <br>

# About the Dataset
The dataset is obtained from kaggle and it has these data fields: <br>

- N - ratio of Nitrogen content in soil
- P - ratio of Phosphorous content in soil
- K - ratio of Potassium content in soil
- temperature - temperature in degree Celsius
- humidity - relative humidity in %
- ph - ph value of the soil
- rainfall - rainfall in mm
- The label is the type of recommended crop

# Techniques used

This is a supervised learning task that tries to identify the category that the object belongs to. So, I'll be trying commonly used classification algorithms to build the model.

## Logistic Regression

Logistic regresson is commonly used for binary classification problem and it uses sigmoid functin to return the probability of a label. The probability output from thee sigmoid function is compared wit a pre-defined threshold to generate a label. <br>
An alternative and modified version of logistic regression is called multinomial logistic regression that could predict a multinomial probability. <br>
common hyperparameters: penalty, max_iter, C, solver <br>

## Random Forest

Random forest is a commonly used ensemble methods that aggreagte results from multiple predictors (a collection of decisin trees). It utilizes bagging method that trains each tree on random sampling of the original dataset and take majority votes from trees. <br>
The advantage of using random forest is that it has better generalization comparing to a single decision tree. <br>
common hyperparameters: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, boostrap <br>

## Naive Bayes

Naive Bayes is an algorithm based on Bayes' Theorem. The naive assumption is that each feature is independent to each other and to calculate the conditional probability is based on prior knowledge. <br>
The advantage of naive bayes is that is does not require a huge set of dataset. Gaussian Naive Bayes is a common type that follows the normal distribution. <br>

## XGBoost

XGBoost is an ensemble tecnique but takes a iterative approach. Each tree is not isolation of each other but is trained in sequence and each one is trained to correct the errors made by the previous one. <br>
The advantage of it is that each model added is focused on correcting the mistakes made by the previous ones rather than learning the same mistakes.<br>

# References
- https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset?datasetId=1046158&sortBy=voteCount 
- https://en.wikipedia.org/wiki/Precision_agriculture 
- https://machinelearningmastery.com/multinomial-logistic-regression-with-python/ 
- https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501 
- https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7 
- https://www.kaggle.com/code/atharvaingle/what-crop-to-grow/notebook#Guassian-Naive-Bayes 
- https://www.kaggle.com/code/ysthehurricane/crop-recommendation-system-using-lightgbm 
