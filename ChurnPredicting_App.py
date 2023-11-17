import streamlit as sl
import pandas as pd
import numpy as np
from pickle import load
from scikeras.wrappers import KerasClassifier
from keras.models import load_model



sl.title("Customer Churning Prediction Calculator")
sl.subheader("Input the following indictors:")
sl.text("By ~ Kwadjo Wusu-Ansah")

column, column_ = sl.columns(2)

with column:
    TotalCharges = sl.number_input("What are their total charges? ",min_value = 0.0, placeholder="Type a number...")
    sl.write('The current number is ', TotalCharges)
    
    MonthlyCharges = sl.number_input("What is monthly charges? ",min_value = 0.0, placeholder="Type a number...")
    sl.write('The current number is ', MonthlyCharges)
    
    Contract = sl.selectbox("What is the type of contract plan? ", ('Month-to-month','One year','Two year'), placeholder="Select contract plan...")
    sl.write('You selected:', Contract)
    
    OnlineSecurity = sl.selectbox("Is there online security avaliable? ", ('No','Yes', 'No internet service'), placeholder="Select response...")
    sl.write('You selected:', OnlineSecurity)
    
    PaymentMethod = sl.selectbox("What is their Payment method? ", ('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'), index=None, placeholder="Select payment method...")
    sl.write('You selected:', PaymentMethod)
   
    
    
with column_:
    tenure = sl.number_input("How long has been since the first contract (in years)? ",min_value=0.0,max_value=100.0, placeholder="Type a number...")
    sl.write('The current number is ', tenure)
    
    TechSupport = sl.selectbox("Does the individual require tech support? ", ('No','Yes','No internet service'), index=None, placeholder="Select contact method...")
    sl.write('You selected:', TechSupport)
    
    gender = sl.selectbox("What is the individual's gender? ", ('Female','Male'), index=None, placeholder="Select contact method...")
    sl.write('You selected:', gender)
    
    InternetService = sl.selectbox("Is there internet service avaliable? ", ('No','Yes', 'No internet service'), index=None, placeholder="Select response...")
    sl.write('You selected:',  InternetService)
    
    OnlineBackup = sl.selectbox("Is the individual able to backup information online? ", ('Yes' 'No' 'No internet service'), index=None, placeholder="Select response...")
    sl.write('You selected:', OnlineBackup)

    
text_input = pd.DataFrame(np.array([[Contract,
       OnlineSecurity, PaymentMethod, TechSupport, gender,
       InternetService, OnlineBackup]]
                     ),columns=['Contract',
       'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'gender',
       'InternetService', 'OnlineBackup'])

num_input = np.array([MonthlyCharges, TotalCharges, tenure])

num_input = pd.DataFrame(np.array([MonthlyCharges, TotalCharges, tenure])
                     ,columns=['MonthlyCharges','TotalCharges', 'tenure'])


#scale

#Encode

encoder = load(open('c_encoder.pkl','rb'))

scaler = load(open('n_scaler.pkl','rb'))

model = load_model('best_model.h5')





column = num_input.columns
num_input = scaler(num_input)

num_input = pd.DataFrame(num_input, columns = column)

text_input = encoder.transform(text_input)


input = pd.concat([num_input, text_input], axis = 1)










predict = sl.button("Predict")
if predict:
    sl.balloons()
    y_pred=model.predict(np.array(input))
    sl.success(f"The overall rating of your player is *{y_pred}*")
    