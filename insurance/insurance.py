import streamlit as st
import joblib
import pandas as pd
import json

# Load the saved model
with open('./deployment/proc_model/model_rf.pkl', 'rb') as file_1:
  model_rf= joblib.load(file_1)
  
with open('./deployment/proc_model/model_scaler.pkl', 'rb') as file_2:
  model_scaler=joblib.load(file_2)

with open('./deployment/proc_model/model_encoder1.pkl', 'rb') as file_3:
  model_encoder1= joblib.load(file_3)

with open('./deployment/proc_model/model_encoder2.pkl', 'rb') as file_4:
  model_encoder2= joblib.load(file_4)

#load list features
with open("./deployment/features/list_cat_cols_ohe.txt", "r") as cek1:
  cat_cols_ohe= json.load(cek1)

with open('./deployment/features/list_num_cols.txt', 'r') as cek2:
  num_cols= json.load(cek2)

with open('./deployment/features/list_cat_cols_ord.txt', 'r') as cek3:
  cat_cols_ord= json.load(cek3)


# function to preprocessing
def preproc(data):
    # encoding ohe
    data_ohe = model_encoder2.transform(data[cat_cols_ohe])
    feature_names = model_encoder2.get_feature_names_out(input_features=cat_cols_ohe)
    data_ohe = pd.DataFrame(data_ohe, columns=feature_names)
    #encoding ord
    data[cat_cols_ord]= model_encoder1.transform(data[cat_cols_ord])
    #scaling
    data[num_cols]= model_scaler.transform(data[num_cols])    
    
    #concat data encoding & scaling
    data_final = pd.concat([data,data_ohe], axis=1)
    data_final.drop(columns=cat_cols_ohe,inplace=True)
    return data_final
    
# Function to predict charges
def predict_charges(data):
    charges = model_rf.predict(data)
    return charges[0]

# Streamlit app
def main():
    # Set app title & description
    st.title("Prediksi Charges")
    st.write("Aplikasi ini memprediksi charges berdasarkan fitur yang diberikan.")

    # Collect user input
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    
    # Prepare input data
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Perform prediction
    if st.button("Predict Charges"):
        #Call the function to predict charges
        result = predict_charges(preproc(data))
        st.write("Predicted Charges: $", round(result, 2))

# Run app
if __name__ == "__main__":
    main()
