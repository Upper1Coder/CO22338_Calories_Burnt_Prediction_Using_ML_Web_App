# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the models
models = {
    'Linear Regression': pickle.load(open('lr.pkl', 'rb')),
    'Ridge': pickle.load(open('rd.pkl', 'rb')),
    'Lasso': pickle.load(open('ls.pkl', 'rb')),
    'Decision Tree': pickle.load(open('dtr.pkl', 'rb')),
    'Random Forest': pickle.load(open('rfr.pkl', 'rb'))
}

# Load training data to populate dropdowns
x_train = pd.read_csv('X_train.csv')

# Model comparison data (based on your provided results)
comparison_data = {
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest'],
    'MSE': [131.99574, 131.99625, 143.82689, 27.15266, 6.97433],
    'R2_Score': [0.96729, 0.96729, 0.96436, 0.99327, 0.99827]
}
comparison_df = pd.DataFrame(comparison_data)

# Function to predict calories burnt
def pred(model, Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    Gender = 1 if Gender.lower() == 'male' else 0
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
    prediction = model.predict(features).reshape(1, -1)
    return prediction[0]

# Web app UI
st.title("Calories Burnt Predictor")

# Button to compare models
if st.button('Compare Models'):
    st.subheader("Model Comparison (MSE and R² Score)")
    st.dataframe(
        comparison_df.style.highlight_min(subset=['MSE'], color='lightgreen')
                        .highlight_max(subset=['R2_Score'], color='lightblue')
    )
    best_model = comparison_df.sort_values(by=['R2_Score', 'MSE'], ascending=[False, True]).iloc[0]
    st.success(f"🏆 Best Model: {best_model['Model']} (MSE: {best_model['MSE']:.4f}, R² Score: {best_model['R2_Score']:.5f})")

# Allow the user to choose a model
model_choice = st.selectbox('Choose Model', list(models.keys()))

# Select input features 
Gender = st.selectbox('Gender', ['Male', 'Female'])  
Age = st.selectbox('Age', sorted(x_train['Age'].unique()))
Height = st.selectbox('Height', sorted(x_train['Height'].unique()))
Weight = st.selectbox('Weight', sorted(x_train['Weight'].unique()))
Duration = st.selectbox('Duration (minutes)', sorted(x_train['Duration'].unique()))
Heart_rate = st.selectbox('Heart Rate (bpm)', sorted(x_train['Heart_Rate'].unique()))
Body_temp = st.selectbox('Body Temperature', sorted(x_train['Body_Temp'].unique()))

# Predict using the selected model
model = models[model_choice]
result = pred(model, Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)

# Button to display the prediction
if st.button('Predict'):
    st.write("Calories Burnt:", result)