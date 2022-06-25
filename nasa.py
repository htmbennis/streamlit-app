import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Gale Crater", page_icon="📊", initial_sidebar_state="expanded"
)

st.title('📊 Gale Crater')
st.header('Geochemistry analysis')
#st.subheader('This is a subheader')

DATA_FILE = ('nasa_input.csv')

#@st.cache
#data_load_state = st.text('Loading data...')
data = pd.read_csv(DATA_FILE)
#data_load_state.text("Data Loaded!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

LIST_OF_ATOMS = st.multiselect(
    "Choose atoms", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']), ['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']
)

IS_STAKED = st.selectbox(
     'How would you like to display graphs?',
     ('Stacked', 'Unstacked'))
#st.write('You selected:', IS_STAKED)

RANK_MIN_MAX = st.slider(
     'Select a range of data points',
     1, 1085, (1, 1085))
#st.write('Values:', RANK_MIN_MAX[0])

if not LIST_OF_ATOMS:
    st.error("Please select at least one atom.")

if LIST_OF_ATOMS:
  RANK_MIN = RANK_MIN_MAX[0]
  RANK_MAX = RANK_MIN_MAX[1]

  #plt.figure(figsize=(20,5))
  fig, ax = plt.subplots()
  if IS_STAKED == 'Stacked' :
    ax.stackplot(data['Rank'],data[LIST_OF_ATOMS].T, alpha = 0.4)
  if IS_STAKED == 'Unstacked' :
    ax.plot(data['Rank'],data[LIST_OF_ATOMS], alpha = 0.4) 

  ax.legend(LIST_OF_ATOMS,
      loc=5, 
      bbox_to_anchor=(1.2, 0.5),
      labelspacing=-2.5, # reverse legend
      frameon=False,         
      #fontsize=9.0
      )
  ax.set_xlim(RANK_MIN,RANK_MAX)
  ax.set_xlabel('Rank')
  ax.set_ylabel('Percentage of Atomic composition (%)')
  ax.grid(axis='y')
  st.pyplot(fig)

st.header('Mineralogy analysis')