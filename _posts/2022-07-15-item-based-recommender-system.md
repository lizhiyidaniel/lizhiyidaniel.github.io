---
layout: post
title: "Item Based Recommender System"
subtitle: "It is a form of collaborative filtering for recommender systems based on the similarity between items calculated using people's ratings of those items. It helps to discover products for customers."
background: '/img/bg-about.jpg'
---

# Item_based_recommender_system
Recommender systems are algorithms that could help customers to discover products such as movies or songs by predicting their ratings of the item and display similar items they like. Item based recommender system is a form of collaborative filtering for recommender systems based on the similarity between items calculated using people's ratings of those items. 

[View Notebook and code](https://github.com/lizhiyidaniel/Item_based_recommender_system/blob/main/Movie_Recommender_System.ipynb)

# Theory behind Recommender systems:

Recommender systems are algorithms that could help customers to discover products such as movies or songs by predicting their ratings of the item and display similar items they like

## What is User-based colloborative filter?

- User-based collaborative filtering works by building a matrix of every piece of content that users bought or viewed
- Similarity scores are calculated between users to find similar users to each others
- For similar users, content not viewed or bought are recommended to users that haven't seen them before

### Limitations of user-based collaborative filter:

- There are more users (8 billion by theory) than products so it would be more complex
- The taste of customers could change over time

## What is item-based collaborative filter?

It is a form of collaborative filtering for recommender systems based on the similarity between items calculated using people's ratings of those items. Item-item collaborative filtering was invented and used by Amazon.com in 1998.

- It works by recommending elements based on items rather than poeple
- It could reduce the complexity of the problem comparing to user-based collaborative filteringg
- Products features are stable comparing to users

![recommender](/img/marketing/similarity.jpeg)

Reference:

https://levelup.gitconnected.com/the-mathematics-of-recommendation-systems-e8922a50bdea
https://en.wikipedia.org/wiki/Item-item_collaborative_filtering

# About the dataset:
Dataset MovieLens: https://grouplens.org/datasets/movielens/100k/

MovieLens 100K movie ratings. Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.
