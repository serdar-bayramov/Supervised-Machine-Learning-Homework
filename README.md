# Supervised-Machine-Learning-Homework

In this assignment, we are building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

Using this data, we are creating machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier.


### Step 1 - Retrieve the data

There are two csv files; 2019 loans which is used for training the model and predict the credit risk of loans from the first quarter of the next year (2020).

* `2019loans.csv`
* `2020Q1loans.csv`

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Using pd.get_dummies() function we have converted categorical data to numeric columns. There were discrepancies in the number of categories between training and testing data which was equalized by removing one unnessary category. 

## The models

We are creating and comparing two models on this data: a logistic regression, and a random forests classifier. 

## Fit a LogisticRegression model and RandomForestClassifier model

Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier. 

## Revisit the Preprocessing: Scale the data

We used `StandardScaler` to scale the training and testing sets. Then, re-fit the LogisticRegression and RandomForestClassifier models on the scaled data and re-run the models to check how it affected the accuracy of the models. 

It is important to perform feature scaling when we are dealing with Gradient Descent Based algorithms (Linear and Logistic Regression) and distance-based algorithms such as KNN and K-means as these are very sensitive to the range of the data points. The scaling helps to balance the importance of each feature of the dataset. This step is not mandatory when dealing with Tree-based algorithms such as Random Forest Classifier. For that reason, we observe improvement in data score after scaling the logistic regression model, while the RFC scaled data score has worsened compared to non-scaled score.
