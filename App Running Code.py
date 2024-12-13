import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Helper functions
def clean_sm(x):
    return np.where(x == 1, 1, 0)

def prepare_features(data):
    ss = data[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

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

def train_model(X_train, y_train):
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_train, y_train)
    return lr

# Load data and prepare model
@st.cache_data
def load_and_train_model():
    # Load data
    try:
        file_path = "social_media_usage.csv"  # Update path if needed
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'social_media_usage.csv' is in the app directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None, None

    # Prepare features
    try:
        prepared_data = prepare_features(data)
    except Exception as e:
        st.error(f"An error occurred during data preparation: {e}")
        return None, None

    # Create feature and target sets
    X = prepared_data[['income', 'education', 'parent', 'married', 'female', 'age']]
    y = prepared_data['sm_li']

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1008)
    except ValueError as e:
        st.error(f"An error occurred while splitting the data: {e}")
        return None, None

    # Train model
    try:
        model = train_model(X_train, y_train)
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None

    # Evaluate model
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        st.error(f"An error occurred during model evaluation: {e}")
        return None, None

    return model, accuracy

# Streamlit app
st.set_page_config(page_title="LinkedIn Usage Predictor", layout="centered")
st.title("🔮 LinkedIn Usage Predictor")

st.sidebar.header("About the App")
st.sidebar.write("""
This app predicts the probability of an individual being a LinkedIn user based on their demographic information.
Fill in the details below to get a prediction instantly!
""")

# Load model
model, accuracy = load_and_train_model()

if model is not None and accuracy is not None:
    # Display accuracy in the sidebar
    st.sidebar.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    # User input form
    st.subheader("Enter Your Details")
    with st.form("user_input_form"):
        income = st.slider("Income Level (1 = Lowest, 9 = Highest)", 1, 9, 5)
        education = st.slider("Education Level (1 = Less than High School, 8 = Advanced Degree)", 1, 8, 4)
        parent = st.radio("Are you a parent?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        married = st.radio("Are you married?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        female = st.radio("Are you female?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        age = st.number_input("Your Age (18-98)", min_value=18, max_value=98, value=30)

        submitted = st.form_submit_button("Predict")

    # Prediction results
    if submitted:
        input_data = pd.DataFrame({
            "income": [income],
            "education": [education],
            "parent": [parent],
            "married": [married],
            "female": [female],
            "age": [age]
        })

        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("Prediction Results")
            result_text = "You are likely to use LinkedIn." if prediction[0] == 1 else "You are unlikely to use LinkedIn."
            st.write(result_text)
            st.write(f"Probability of LinkedIn Usage: **{probability * 100:.2f}%**")

            # Optional: Include suggestions or insights based on probability
            if probability > 0.8:
                st.info("You are highly likely to use LinkedIn. Consider leveraging the platform to expand your professional network!")
            elif probability < 0.2:
                st.warning("You are less likely to use LinkedIn. Perhaps explore its benefits for career growth.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.stop()
