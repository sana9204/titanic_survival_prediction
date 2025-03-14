import pandas as pd
import streamlit as st
import joblib 



encoder=joblib.load('encoder.pkl')
model=joblib.load('model.pkl')
scaler=joblib.load('scaler.pkl')


st.title('Survived prediction')
pclass=st.selectbox('Selct Pclass',[1,2,3])
sex=st.selectbox('Gender',['male','female'])
age=st.number_input("Age")
sibsp=st.number_input("Sibling")
parch=st.selectbox('Parch',[0, 1, 2, 5, 3, 4, 6])
fare=st.number_input("Fare",format='%.4f')
embarked=st.selectbox('Embarked',['S', 'C', 'Q'])
gender_encoded=1 if sex=='male' else 0
embarked_encoded=encoder.transform([[embarked]])[0]



data=pd.DataFrame({
    'Pclass':pclass,
    'Sex':gender_encoded,
    'Age':age,
    'SibSp':sibsp,
    'Parch':parch,
    'Fare':fare,
    'Embarked':embarked_encoded
},index=[0])


data_scaled=scaler.transform(data)
pred=model.predict(data_scaled)[0]

result = "Survived" if pred==1 else "he/she no more"



if st.button("Predict"):
    st.write(f"Prediction: {result}")