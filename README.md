# TitanicDataset

🛳️ Titanic Survival Prediction
This project is based on the classic Titanic dataset from Kaggle. The goal is to predict whether a passenger survived or not based on features such as age, gender, class, fare, and more.

Overview
Using a combination of data preprocessing, feature engineering, and model tuning, this project builds a classification model to predict passenger survival. The model is trained and evaluated using accuracy and a confusion matrix.

🔧 What This Project Does
*Loads and explores the Titanic dataset

*Handles missing values (e.g., Age, Embarked)

*Drops irrelevant features (Name, Ticket, etc.)

*Encodes categorical variables like Sex and Embarked

*Creates new features:

*FamilySize (SibSp + Parch)

*IsAlone (whether the passenger was alone)

*AgeBin and FareBin (binned versions of continuous features)

*Splits the dataset into training and testing sets

*Scales numerical features using MinMaxScaler

*Uses GridSearchCV to tune a K-Nearest Neighbors (KNN) classifier

*Evaluates the model with accuracy and a confusion matrix heatmap

📁 Files
Titanic-Dataset.csv – The dataset used in this project

TitanicDataset.py – The main Python script for data processing, training, and evaluation

README.md – This file

✅ Dependencies
pandas

numpy

scikit-learn

matplotlib

seaborn

📈 Example Output
Accuracy: 81-86% (varies slightly depending on model parameters)

Confusion Matrix: Visual output showing correct vs. incorrect predictions
