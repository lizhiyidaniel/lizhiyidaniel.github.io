---
layout: post
title: "Marketing Sementation by Unsupervised Learning"
subtitle: "Perform marketing segmentation for advertising by unsupervised learning techniques: clustering (k-means), principal component analysis and auto encoders"
background: '/img/bg-about.jpg'
---

View Original Github Repo: [click here](https://github.com/lizhiyidaniel/Marketing_segmentation_unsupervised_learning) <br>
View Original Notebook: [click here](https://github.com/lizhiyidaniel/Marketing_segmentation_unsupervised_learning/blob/main/Marketing_Segmentation_Unsupervised_Learning.ipynb)

# Understand Marketing Segmentation

Why Marketing is essential?

- Marketing is crucial for the growth and sustainability of businesses
- Marketing could help to build the company's brand, engage customers, grow revenue and increase sales

1. Growth: empowering business growth by reaching new customers
2. Education: educating and communicating value porposition to customers
3. Drive sales: driving sales and traffic to products/services
4. Engagement: engaging customers and understand their needs

Why Market Segmentation is important?

- One of the key pain points for marketers is to know customers and identify their needs
- By understanding customers, marketers could launch a targeted marketingg campaign to tailor for specific needs
- Data Sciene could be used to perform market segmentation

Task: Launch targeted ad marketing compaign by dividing customers into distinctive groups
data: banking data about customers for past 6 months

Data Source: <https://www.kaggle.com/arjunbhasin2013/ccdata>

Data Information:
This case requires to develop a customer segmentation to define marketing strategy. The
sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.

Following is the Data Dictionary for Credit Card dataset :-

- CUSTID : Identification of Credit Card holder (Categorical)
- BALANCE : Balance amount left in their account to make purchases
- BALANCEFREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
- PURCHASES : Amount of purchases made from account
ONEOFFPURCHASES : Maximum purchase amount done in one-go
- INSTALLMENTSPURCHASES : Amount of purchase done in installment
- CASHADVANCE : Cash in advance given by the user
- PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
- ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
- PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
- CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
- CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
- PURCHASESTRX : Numbe of purchase transactions made
- CREDITLIMIT : Limit of Credit Card for user
PAYMENTS : Amount of Payment done by user
MINIMUM_PAYMENTS : Minimum amount of payments made by user
- PRCFULLPAYMENT : Percent of full payment paid by user
- TENURE : Tenure of credit card service for user
