import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.header('Ruppe Value Predictor(USD)')
data = pd.read_csv('ur_value.csv')
data=data.dropna()

x = data.iloc[:,[1,2,3]].values
y = data.iloc[:,4].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

lc =LinearRegression()
lc.fit(x_train,y_train)

O = (st.number_input('Enter Opening Price'))
H = (st.number_input('Enter High Price'))
L = (st.number_input('Enter Low price'))

if(st.button('Predict Price')):
    pc = lc.predict([[O,H,L]])[0]
    st.success(pc)