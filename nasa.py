import streamlit as st
import pandas as pd
import numpy as np

st.title('Test')

DATE_COLUMN = 'date/time'
DATA_URL = ('nasa_input.csv')

#@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data

data_load_state = st.text('Loading data...')
data = load_data(100)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data["SiO2"], bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
#filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
#st.map(filtered_data)