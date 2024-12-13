"""
LinkedIn Usage Predictor
Author: Michael Colina
Date: 12/06/2024

This script analyzes social media usage data to predict LinkedIn usage based on various demographic features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Q1: Read in the data and check dimensions
def load_data():
    file_path = r"C:\Users\mcoli\Downloads\social_media_usage.csv"
    s = pd.read_csv(file_path)
    print(f"The dimensions of the dataframe are: {s.shape}")
    return s

# Q2: Define clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Test clean_sm function
def test_clean_sm():
    toy_data = pd.DataFrame({
        'col1': [1, 0, 3],
        'col2': [2, 1, 0]
    })
    cleaned_data = toy_data.apply(lambda col: col.map(clean_sm))
    print("Original data:")
    print(toy_data)
    print("\nCleaned data:")
    print(cleaned_data)

# Q3: Create and prepare the feature dataset
def prepare_features(s):
    ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
    
    # Create target variable
    ss['sm_li'] = clean_sm(ss['web1h'])
    ss.drop('web1h', axis=1, inplace=True)
    
    # Clean and transform features
    ss.loc[ss['income'] > 9, 'income'] = np.nan
    ss.loc[ss['educ2'] > 8, 'educ2'] = np.nan
    ss.rename(columns={'educ2': 'education'}, inplace=True)
    
    ss['parent'] = np.where(ss['par'] == 1, 1, 0)
    ss.drop('par', axis=1, inplace=True)
    
    ss['married'] = np.where(ss['marital'] == 1, 1, 0)
    ss.drop('marital', axis=1, inplace=True)
    
    ss['female'] = np.where(ss['gender'] == 2, 1, 0)
    ss.drop('gender', axis=1, inplace=True)
    
    ss.loc[ss['age'] > 98, 'age'] = np.nan
    
    ss.dropna(inplace=True)
    return ss

# Q4: Create target vector and feature set
def create_feature_target_sets(ss):
    y = ss['sm_li']
    X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]
    return X, y

# Q5: Split data into training and test sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1008)

# Q6: Create and train logistic regression model
def train_model(X_train, y_train):
    lr = LogisticRegression(class_weight='balanced')
    return lr.fit(X_train, y_train)

# Q7 & Q8: Evaluate model and create confusion matrix
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(conf_matrix,
                      columns=["Predicted negative", "Predicted positive"],
                      index=["Actual negative", "Actual positive"]))
    
    return y_pred

# Q9: Calculate and display classification metrics
def display_classification_metrics(y_test, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Q10: Make predictions for specific cases
def predict_examples(model):
    # Example cases
    person_1 = pd.DataFrame([[8, 7, 0, 1, 1, 42]], 
                           columns=['income', 'education', 'parent', 'married', 'female', 'age'])
    person_2 = pd.DataFrame([[8, 7, 0, 1, 1, 82]], 
                           columns=['income', 'education', 'parent', 'married', 'female', 'age'])
    
    prob_1 = model.predict_proba(person_1)[0][1]
    prob_2 = model.predict_proba(person_2)[0][1]
    
    print(f"\nProbability predictions:")
    print(f"Person 1 (age 42): {prob_1:.4f}")
    print(f"Person 2 (age 82): {prob_2:.4f}")

def main():
    # Load and prepare data
    s = load_data()
    ss = prepare_features(s)
    
    # Create feature and target sets
    X, y = create_feature_target_sets(ss)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Display detailed metrics
    display_classification_metrics(y_test, y_pred)
    
    # Make example predictions
    predict_examples(model)

if __name__ == "__main__":
    main()
