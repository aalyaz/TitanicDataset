# import additional packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
data = pd.read_csv("Titanic-Dataset.csv")
data.head(10)

# check data types
data.dtypes

# print the number of missing values
print(data.isnull().sum())

# drop irrelevant columns
data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
data.head(10)

# impute missing Age values with median based on Pclass
data["Age"] = data["Age"].fillna(data.groupby("Pclass")["Age"].transform("median"))
data.head(10)

# check for remaining missing values
print(data.isnull().sum())

# handle missing Embarked values
print(data["Embarked"].value_counts())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
print(data["Embarked"].value_counts())

# one-hot encode Embarked
embarked_dummies = pd.get_dummies(data["Embarked"], prefix="Embarked")
embarked_dummies = embarked_dummies.astype(int)
data = pd.concat([data, embarked_dummies], axis=1)
data.drop(columns=["Embarked"], inplace=True)
data.head(10)

# encode Sex column
data["Sex"] = data["Sex"].str.lower().str.strip()
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data.head(10)

# create FamilySize feature
data["FamilySize"] = data["SibSp"] + data["Parch"]
data.head(10)

# create IsAlone feature
data["IsAlone"] = (data["FamilySize"] == 0).astype(int)
data.head(10)

# create Fare bins
data["FareBin"] = pd.qcut(data["Fare"], 4, labels=False)
data.head(10)

# create Age bins
data["AgeBin"] = pd.cut(data["Age"], bins=5, labels=False)
data.head(10)

# define input features and target
X = data.drop(columns=["Survived"])
y = data["Survived"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# tune and return best KNN model
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)
print(best_model)

# make predictions and evaluate
prediction = best_model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f'accuracy = {accuracy*100:.2f}%')

matrix = confusion_matrix(y_test, prediction)
print('\nConfusion Matrix:')
print(matrix)

# plot confusion matrix
def plot_model(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d",
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()

plot_model(matrix)
