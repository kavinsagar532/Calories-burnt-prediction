import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Function to load and preprocess data
@st.cache
def load_and_preprocess_data():
    calories = pd.read_csv('calories.csv')
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

    # Encode Gender
    label_encoder = preprocessing.LabelEncoder()
    calories_data['Gender'] = label_encoder.fit_transform(calories_data['Gender'])
    
    return calories_data

# Function to train the model
@st.cache(allow_output_mutation=True)
def train_model(data):
    X = data.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = data['Calories']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    return model

# Load and preprocess data
calories_data = load_and_preprocess_data()

# Train the model
model = train_model(calories_data)

# Streamlit app
st.title('Calories Burned Prediction')

st.write('Please input the following details to predict the calories burned:')

# Inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
height = st.number_input('Height (in cm)', min_value=0, max_value=300, value=170)
weight = st.number_input('Weight (in kg)', min_value=0, max_value=500, value=70)
duration = st.number_input('Duration of exercise (in mins)', min_value=0, max_value=300, value=30)
heart_rate = st.number_input('Heart Rate', min_value=0, max_value=200, value=100)
body_temp = st.number_input('Body Temperature (in °C)', min_value=35.0, max_value=42.0, value=37.0)

# Convert gender to numeric
gender_numeric = 0 if gender == 'Male' else 1

# Predict button
if st.button('Predict Calories Burned'):
    input_data = [[gender_numeric, age, height, weight, duration, heart_rate, body_temp]]
    prediction = model.predict(input_data)
    st.write(f'Predicted Calories Burned: {prediction[0]:.2f}')
