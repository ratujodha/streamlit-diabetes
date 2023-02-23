import pickle
import streamlit as st

#baca model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

#judul
st.title("Data Mining Prediksi Diabetes")
Pregnancies	= st.text_input('Input Pregnancies : ')
Glucose	= st.text_input('Input Glucose : ')	
BloodPressure	= st.text_input('Input BloodPressure : ')	
SkinThickness	= st.text_input('Input SkinThickness : ')	
Insulin	= st.text_input('Input Insulin : ')	
BMI	= st.text_input('Input BMI : ')	
DiabetesPedigreeFunction	= st.text_input('Input DiabetesPedigreeFunction : ')	
Age	= st.text_input('Input Age : ')

#code Untuk Prediksi
diab_diagnosis=''

#membuat tombol untuk prediksi
if st.button('Testing'):
    diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if(diab_prediction[0] ==1):
        diab_diagnosis = 'DIABET'
    else:
        diab_diagnosis = 'NO'
    st.success(diab_diagnosis)
