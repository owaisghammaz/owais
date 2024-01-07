import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('trained_model_new.joblib')

# Function to predict based on input features
# Function to predict based on input features
def predict_SFS(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]  # Assuming model.predict returns an array
    prediction_text = "Successful surgery" if prediction == 0 else "Unsuccessful surgery"
    return prediction_text

# Create a Streamlit web app
def main():
    # Set app title and description
    st.title("Stone Free Status Prediction Web App")
    st.write("Enter the required information to predict the likelihood of successful PCNL surgery.")

    # Create input fields for user to enter information
    Age = st.number_input("Age", min_value=1, max_value=100, value=30)
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Stone_burden = st.number_input("Stone burden (mm2)", min_value=1, max_value=5000, value=30)
    History_of_Diabetes = st.selectbox("History of Diabetes", ("Yes", "No"))
    History_of_Hypertension = st.selectbox("History of Hypertension", ("Yes", "No"))
    History_of_Hyperlipidemia = st.selectbox("History of Hyperlipidemia", ("Yes", "No"))
    Laterality = st.selectbox("Unilateral kidney", ("Unilateral", "Bilateral"))
    Preoperative_Renal_Insufficiency = st.selectbox("Preoperative Renal Insufficiency", ("Yes", "No"))
    Preoperative_Anemia = st.selectbox("Preoperative Anemia", ("Yes", "No"))
    Preoperative_UTI = st.selectbox("Preoperative UTI", ("Yes", "No"))
    Previous_surgery_of_target_kidney = st.selectbox("Previous surgery of target kidney", ("Yes", "No"))
    SMWL = st.selectbox("SMWL", ("Yes", "No"))
    URSL = st.selectbox("URSL", ("Yes", "No"))
    PCNL = st.selectbox("PCNL", ("Yes", "No"))
    Hydronephrosis = st.selectbox("Hydronephrosis", ("Yes", "No"))
    Upper_calyx = st.selectbox("Upper_calyx", ("Yes", "No"))
    Middle_calyx = st.selectbox("Middle calyx", ("Yes", "No"))
    Lower_calyx = st.selectbox("Lower calyx", ("Yes", "No"))
    Pelvis = st.selectbox("Pelvis", ("Yes", "No"))
    Ureter = st.selectbox("Ureter", ("Yes", "No"))
    

    if st.button('Predict'):
        # Convert categorical inputs to numerical values
        Gender = 1 if Gender == "Male" else 0
        History_of_Diabetes = 1 if History_of_Diabetes == "Yes" else 0
        History_of_Hypertension = 1 if History_of_Hypertension == "Yes" else 0
        History_of_Hyperlipidemia = 1 if History_of_Hyperlipidemia == "Yes" else 0
        Laterality = 1 if Laterality == "Unilateral" else 0
        Preoperative_Renal_Insufficiency = 1 if Preoperative_Renal_Insufficiency == "Yes" else 0
        Preoperative_Anemia = 1 if Preoperative_Anemia == "Yes" else 0
        Preoperative_UTI = 1 if Preoperative_UTI == "Yes" else 0
        Previous_surgery_of_target_kidney = 1 if Previous_surgery_of_target_kidney == "Yes" else 0
        SMWL = 1 if SMWL == "Yes" else 0
        URSL = 1 if URSL == "Yes" else 0
        PCNL = 1 if PCNL == "Yes" else 0
        Hydronephrosis = 1 if Hydronephrosis == "Yes" else 0
        Upper_calyx = 1 if Upper_calyx == "Yes" else 0
        Middle_calyx = 1 if Middle_calyx == "Yes" else 0
        Lower_calyx = 1 if Lower_calyx == "Yes" else 0
        Pelvis = 1 if Pelvis == "Yes" else 0
        Ureter = 1 if Ureter == "Yes" else 0
        

        # Collect all inputs into a list or array
        input_data = [
            Age, 
            Gender, 
            Stone_burden, 
            History_of_Diabetes, 
            History_of_Hypertension, 
            History_of_Hyperlipidemia, 
            Laterality, 
            Preoperative_Renal_Insufficiency, 
            Preoperative_Anemia, 
            Preoperative_UTI, 
            Previous_surgery_of_target_kidney, 
            SMWL, 
            URSL, 
            PCNL, 
            Hydronephrosis, 
            Upper_calyx, 
            Middle_calyx, 
            Lower_calyx, 
            Pelvis, 
            Ureter, 
        ]

        # Make prediction
        prediction_text = predict_SFS(input_data)

        # Display the prediction
        st.write(f'Prediction: {prediction_text}')

if __name__ == '__main__':
    main()
