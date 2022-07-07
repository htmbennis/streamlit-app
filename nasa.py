import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Gale Crater", page_icon="📊", initial_sidebar_state="expanded")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.title('📊 Gale Crater')
st.image('./curiosity_image.PNG')


def estim_mineral_func (X) :
  A = np.array([
    [-2.11 , 0.44, -1.05 , 1.44, 0.077, -0.05476 , -0.1407],
    [-1.96 , 0.8507, -2.033,  2.141, 0.0406 ,-0.228 , 2.7615],
    [1.355 ,0.737, 0.3744 ,2.6085, 0.0663 , 1.291, 0.0485],
    [-2.851  ,1.2109, -3.2887, 2.1291, 0.006971, -0.90095 , 1.5591]
  ])
  B = np.array([150.04, -46.20 , 105.25 ,-142.33 , -4.43,  24.93 ,-47.16])
  #X = np.array([44,22,8,3,6,8.6])
  #X = input_vector
  AX = np.matmul(np.transpose(A),X)
  Y = np.add(np.matmul(np.transpose(A),X),B)
  Y = np.maximum(Y, 0)
  Y = np.append(Y, 100 - np.sum(Y))
  Y = np.maximum(Y, 0)
  return Y


#########################################################
#          FROM GEOCHEMISTRY TO MINERALOGY              #
#########################################################

st.subheader('From geochemistry data to mineralogy estimation')

DATA_FILE_1 = ('nasa_input_2_1.csv')
DATA_FILE_2 = ('nasa_input_2_2.csv')

#@st.cache
data = pd.read_csv(DATA_FILE_1, sep=";")
data_2 = pd.read_csv(DATA_FILE_2, sep=";")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(data)
    #st.write(data)  #This is an other way to display table

SOLS_MIN_MAX = st.slider(
     'Select your range of sols to visualize (to zoom in and zoom out in the charts):',
     46, 3192, (46, 3192),
     help="Adjust range so you can focus on some data points in the graph.",)

#Geochemistry analysis
LIST_OF_ATOMS = st.multiselect(
    "Choose geochimicals:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']), ['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']
)


if not LIST_OF_ATOMS:
    st.error("Please select at least one atom.")

if LIST_OF_ATOMS:
  SOLS_MIN = SOLS_MIN_MAX[0]
  SOLS_MAX = SOLS_MIN_MAX[1]

  fig, ax = plt.subplots()
  ax.stackplot(data['Sols'],data[LIST_OF_ATOMS].T, alpha = 0.4)

  ax.legend(LIST_OF_ATOMS,
      loc=5, 
      bbox_to_anchor=(1.2, 0.5),
      labelspacing=-2.5, # reverse legend
      frameon=False,         
      #fontsize=9.0
      )
  ax.set_xlim(SOLS_MIN,SOLS_MAX)
  ax.set_xlabel('Sols')
  ax.set_ylabel('Percentage of Atomic composition (%)')
  ax.grid(axis='y')
  st.pyplot(fig)

#Mineralogy analysis
LIST_OF_MINERALS = st.multiselect(
    "Choose minerals:", 
    list(['Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay']), 
    ['Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay']
)

if not LIST_OF_MINERALS:
    st.error("Please select at least one atom.")

if LIST_OF_MINERALS:
  SOLS_MIN = SOLS_MIN_MAX[0]
  SOLS_MAX = SOLS_MIN_MAX[1]

  fig, ax = plt.subplots()
  ax.stackplot(data['Sols'],data[LIST_OF_MINERALS].T, alpha = 0.4)

  ax.legend(LIST_OF_MINERALS,
      loc=5, 
      bbox_to_anchor=(1.27, 0.5),
      labelspacing=-2.5, # reverse legend
      frameon=False,         
      #fontsize=9.0
      )
  ax.set_xlim(SOLS_MIN,SOLS_MAX)
  ax.set_xlabel('Sols')
  ax.set_ylabel('Percentage of Mineral composition (%)')
  ax.grid(axis='y')
  st.pyplot(fig)


#########################################################
#                      XYZ ANALYSIS                     #
#########################################################

st.subheader('Geochemistry and mineralogy xyz analysis')

col1, col2, col3 = st.columns(3)
with col1 : X_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose X axis:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O',
      'Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay']),0)

with col2 : Y_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose Y axis:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O',
      'Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay']),1)

with col3 : Z_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose Z axis:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O',
      'Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay']),7)

fig, ax = plt.subplots()
ax.scatter(data[X_OF_XYZ_ANALYSIS],data[Y_OF_XYZ_ANALYSIS],c=data[Z_OF_XYZ_ANALYSIS], cmap='coolwarm',alpha = 0.4) 
im = ax.scatter(data[X_OF_XYZ_ANALYSIS],data[Y_OF_XYZ_ANALYSIS],c=data[Z_OF_XYZ_ANALYSIS], cmap='coolwarm',alpha = 0.4) 
fig.colorbar(im)
ax.set_xlabel(X_OF_XYZ_ANALYSIS)
ax.set_ylabel(Y_OF_XYZ_ANALYSIS)
ax.grid(axis='both')
st.pyplot(fig)


#########################################################
#                 ANALYSIS WITH YOUR DATA               #
#########################################################

st.subheader('Run analysis with your own data')

with st.form(key="mineralogy_estim_form"):
  st.write('Insert concentration of each chemical:')
  #'SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O'
  col4, col5, col6 = st.columns(3)
  with col4: SIO2_INPUT = st.number_input('SiO2 (%)',min_value=0, max_value=100,value=50, step=1)
  with col5: FEO_INPUT = st.number_input('FeO (%)',min_value=0, max_value=100,value=10, step=1)
  with col6: AL2O3_INPUT = st.number_input('Al2O3 (%)',min_value=0, max_value=100,value=16, step=1)
  col7, col8, col9 = st.columns(3)
  with col7: SO3_INPUT = st.number_input('SO3 (%)',min_value=0, max_value=100,value=5, step=1)
  with col8: OTHERS_INPUT = st.number_input('Others (%)',min_value=0, max_value=100,value=19, step=1)
  sum_of_inputs = SIO2_INPUT + FEO_INPUT + AL2O3_INPUT + SO3_INPUT + OTHERS_INPUT
  
  submit_button = st.form_submit_button(label="Estimate mineralogy")
input_vector = [SIO2_INPUT,FEO_INPUT,AL2O3_INPUT,SO3_INPUT]

if sum_of_inputs != 100:
    st.error("Total of inputs should be equal to 100%. Please change inputs.")
    st.write("Total of inputs is equal to",sum_of_inputs)

if sum_of_inputs == 100:
  Y = estim_mineral_func(input_vector)
  
  col10, col11= st.columns(2)
  with col10 :
    st.markdown("**Input : Geochemistry**")
    fig, ax = plt.subplots()
    ax.pie(np.append(input_vector,OTHERS_INPUT), labels = ['SiO2', 'FeO', 'Al2O3', 'SO3','Others'] ,
          autopct=lambda p: '{:.0f}%'.format(p),
          startangle = 90)
    st.pyplot(fig) 
  with col11 :
    st.markdown("**Output : Estimated Mineralogy**") 
    fig, ax = plt.subplots()
    ax.pie(Y, labels = ['Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay','Amorhous'] ,
          autopct=lambda p: '{:.0f}%'.format(p),
          startangle = 90)
    st.pyplot(fig)


#########################################################
#              TEST WITH SPECIFC MEASUREMENTS           #
#########################################################


st.subheader('Estimation with our method vs Measurements')
SPECIFIC_MEASUREMENT = st.selectbox(
    "Choose a measurement:", 
    list(['RN','JK','CB','WJ','CH','MJ','TP','BK','BS','GB','GH','LB','OK',
'OU','MB','QL','SB','OB','DU','ST','HF','RH','AL','KM',
'GE','GE2','HU','EB','GG','MA','MA3','GR','NT','BD','PT','MG','ZS']),0)

#st.dataframe(data_2)
#st.write(data_2.where(data_2['Abreviation']=='JK', inplace = True)) ##la close where ne fonctionne pas!!!!
col15, col16, col17 = st.columns(3)
with col15 :
    st.markdown("**Measured Geochemistry**")
    #fig, ax = plt.subplots()
    #ax.pie(np.append(input_vector,OTHERS_INPUT), labels = ['SiO2', 'FeO', 'Al2O3', 'SO3','Others'] ,
    #      autopct=lambda p: '{:.0f}%'.format(p),
    #      startangle = 90)
    #st.pyplot(fig) 
with col16 :
    st.markdown("**Measured Mineralogy**")
with col17 :
    st.markdown("**Estimated Geochemistry**")    




#########################################################
#                 GEOSPATIAL VISUALIZATION              #
#########################################################
st.subheader('Geospatial visualization')

col12, col13, col14 = st.columns(3)
with col13 : 
  Z_OF_CHEMICAL_ANALYSIS = st.selectbox(
    "Choose geochemical:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']),0)
with col14 : 
  Z_OF_MINERAL_ANALYSIS = st.selectbox(
    "Choose mineral:", list(['Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay']),1)  

col18, col19, col20 = st.columns(3)
image = Image.open('image_satellite.PNG')
with col18 : 
  st.image(image)
with col19:
  fig, ax = plt.subplots()
  ax.scatter(data['Longitude'],data['Latitude'],c=data[Z_OF_CHEMICAL_ANALYSIS], cmap='coolwarm', s=400) 
  im = ax.scatter(data['Longitude'],data['Latitude'],c=data[Z_OF_CHEMICAL_ANALYSIS], cmap='coolwarm')  
  fig.colorbar(im)
  fig.set_size_inches(20, 26)
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.grid(axis='both')
  ax.set_xlim(137.325,137.475)  
  st.pyplot(fig)
with col20:
  fig, ax = plt.subplots()
  ax.scatter(data['Longitude'],data['Latitude'],c=data[Z_OF_MINERAL_ANALYSIS], cmap='coolwarm', s=400) 
  im = ax.scatter(data['Longitude'],data['Latitude'],c=data[Z_OF_MINERAL_ANALYSIS], cmap='coolwarm')  
  fig.colorbar(im)
  fig.set_size_inches(20, 26)
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.grid(axis='both')
  ax.set_xlim(137.325,137.475)
  st.pyplot(fig)  

