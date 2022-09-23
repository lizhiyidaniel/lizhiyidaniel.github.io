---
layout: post
title: "Predict future sales with time series store sales data"
subtitle: "forecast future sales based on historical ones while taking into account of seasonality effects, demand, holidays, promotions, and competition by using Facebook Prophet model"
background: '/img/bg-about.jpg'
---


# Sales_prediction_time_series

[View Notebook](https://github.com/lizhiyidaniel/Sales_prediction_time_series/blob/main/future_time_series_sales_prediction.ipynb)

# Background
Companies need to leverage AI/ML models to develop predictive models to forecast sales in the future to become competitive and skyrocket growth.

Predictive models could forecast future sales based on historical ones while taking into account of seasonality effects, demand, holidays, promotions, and competition


![forecast](/dashboards/forecast.jpg)


Reference: https://www.databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html

# What is Facebook Prophet?
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

At its core, the Prophet procedure is an additive regression model with four main components:

- A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
- A yearly seasonal component modeled using Fourier series.
- A weekly seasonal component using dummy variables.
- A user-provided list of important holidays.
## Additive Regression model takes the form as:
![additive](/dashboards/additive.png)


## Using Facebook Prophet has these advantages:

- Accurate and fast
  - Facebook uses Prophet across applications to produce reliable forecasts for planning and goal setting
  - Forecasts could be fit with model in seconds
- Fually automatic
  - No need for data preprocessing
  - robust to outliers, missing data and dramatic changes
- Tunable forecasts
  - Human-interpretable parameters could be used to improve forecast by adding domain knowledge

Reference:
- https://facebook.github.io/prophet/
- https://research.facebook.com/blog/2017/02/prophet-forecasting-at-scale/
- https://towardsdatascience.com/predicting-values-using-linear-additive-regression-prophet-and-bsts-models-in-r-26485d629fae
# About the dataset
Data Source: https://www.kaggle.com/c/rossmann-store-sales/data

## Data files:

train.csv - historical data including Sales
store.csv - supplemental information about the stores
## Data Fields:

- Id - an Id that represents a (Store, Date) duple within the test set
- Store - a unique Id for each store
- Sales - the turnover for any given day (this is what you are predicting)
- Customers - the number of customers on a given day
- Open - an indicator for whether the store was open: 0 = closed, 1 = open
- StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are - - closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
- SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
- StoreType - differentiates between 4 different store models: a, b, c, d Assortment - describes an assortment level: a = basic, b = extra, c = extended
- CompetitionDistance - distance in meters to the nearest competitor store
- CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
- Promo - indicates whether a store is running a promo on that day
- Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
- Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
- PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

 forecast future sales based on historical ones while taking into account of seasonality effects, demand, holidays, promotions, and competition
