import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.title('BoomBike Prediction')

# Load the trained model
pickle_in = open('bike.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.title('Team: Group 5')

def prediction(season, year, month, weathersit, temp, windspeed,const):
    if season == 'Spring':
        spring = 1
        summer = 0
        winter = 0
    elif season == 'Summer':
        spring = 0
        summer = 1
        winter = 0
    elif season == 'Winter':
        spring = 0
        summer = 0
        winter = 1
    if year == '2018':
        year = 0
    else:
        year = 1
    if month == 'Jan':
        Jan = 1
        July = 0
        Sep = 0
    elif month == 'July':
        Jan = 0
        July = 1
        Sep = 0
    elif month == 'Sep':
        Jan = 0
        July = 0
        Sep = 1
    else:
        Jan = 0
        July = 0
        Sep = 0
        
    if weathersit == 'Light_Snow':
        Light_Snow = 1
        Mist = 0
    elif weathersit == 'Mist':
        Light_Snow = 0
        Mist = 1
    else:
        Light_Snow = 0
        Mist = 0

    # Create a 2D array of the input features
    features = np.array([[spring, summer, winter, year, Jan, July, Sep, Light_Snow, Mist, temp, windspeed,const]])
    
    # Scale the features using MinMaxScaler
    #scaler = MinMaxScaler()
    #scaled_features = scaler.fit_transform(features)
    
    # Predict the bike demand
    prediction = classifier.predict(features)
    prediction = prediction[0] * 100  # Get the first element and multiply by 100
    return prediction
    
def main():
    
    # Use selectbox instead of multiselect for single selection
    year = st.selectbox('Year', ("2018", "2019"))
    season = st.selectbox('Season', ("Spring", "Summer", "Fall", "Winter"))
    month = st.selectbox('Month', ("Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"))
    windspeed = st.number_input("windspeed")
    weathersit = st.selectbox('Weather', ("Clear", "Mist", "Light Snow", "Rainfall"))
    temp = st.number_input("temp")
    const = st.number_input('const')
    result = ""
    
    if st.button('Predict'):
        result = prediction(season, year, month, weathersit, temp, windspeed,const)

    st.success('Bike Rental Prediction is {}'.format(result))

if __name__ == '__main__':
    main()
