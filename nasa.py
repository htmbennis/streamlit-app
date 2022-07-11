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
                [0.985420687,-3.066379724,0.424739715,-1.051622287,1.022481735,0.467960914,-0.410071317,-0.654388826],
                [-0.079052703,-3.807826331,0.849542732,-2.033170572,1.619821563,0.841339507,-0.971899764,1.732725411],
                [-0.36141167,-2.523011493,0.451884864,0.374461398,1.715807751,0.073766502,2.191364637,-0.407165512],
                [1.790211749,-5.260412458,1.090036377,-3.288686911,1.27437956,0.053005324,-0.250028796,2.243423759]
                ])
  B = np.array([-13.16627197,282.1501528,-42.56947313,105.2539243,-98.68748535,-37.29211563,40.53349503,0.054267971])
  #X = np.array([44,22,8,3])
  AX = np.matmul(np.transpose(A),X)
  Y = np.add(np.matmul(np.transpose(A),X),B)
  Y = np.maximum(Y, 0)
  Y = (Y / np.sum(Y))*100  
  return Y

DATA_FILE_1 = ('nasa_input_2_1.csv')
DATA_FILE_2 = ('nasa_input_2_2.csv')

#@st.cache
data = pd.read_csv(DATA_FILE_1, sep=";")
data_2 = pd.read_csv(DATA_FILE_2, sep=";")


#########################################################
#          FROM GEOCHEMISTRY TO MINERALOGY              #
#########################################################

st.subheader('From geochemistry data to mineralogy estimation')

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
  ax.stackplot(data['Sols'],data[LIST_OF_MINERALS].T, alpha = 0.6)

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
  st.write('Insert concentration of each chemical element:')
  #'SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O'
  col4, col5, col6, col7 = st.columns(4)
  with col4: SIO2_INPUT = st.number_input('SiO2 (%)',min_value=0, max_value=100,value=44, step=1)
  with col5: FEO_INPUT = st.number_input('FeO (%)',min_value=0, max_value=100,value=22, step=1)
  with col6: AL2O3_INPUT = st.number_input('Al2O3 (%)',min_value=0, max_value=100,value=8, step=1)
  with col7: SO3_INPUT = st.number_input('SO3 (%)',min_value=0, max_value=100,value=3, step=1)  
  #col7, col8, col9 = st.columns(3)
  #with col8: OTHERS_INPUT = st.number_input('Others (%)',min_value=0, max_value=100,value=23, step=1)
  sum_of_inputs = SIO2_INPUT + FEO_INPUT + AL2O3_INPUT + SO3_INPUT #+ OTHERS_INPUT
  OTHERS_INPUT = 100 - sum_of_inputs 
  submit_button = st.form_submit_button(label="Estimate mineralogy")
input_vector = [SIO2_INPUT,FEO_INPUT,AL2O3_INPUT,SO3_INPUT]

if sum_of_inputs > 100:
    st.error("Total of inputs should be equal to 100%. Please change inputs.")
    st.write("Total of inputs is equal to",sum_of_inputs)

if sum_of_inputs <= 100:
  Y = estim_mineral_func(input_vector)
  
  col10, col11= st.columns(2)
  with col10 :
    st.markdown("**Input : Geochemistry**")
    fig, ax = plt.subplots()
    ax.pie(np.append(input_vector,OTHERS_INPUT), labels = ['SiO2', 'FeO', 'Al2O3', 'SO3','Others'] ,
          autopct=lambda p: '{:.0f}%'.format(p),
          startangle = 90,
          colors=['#a5c8e1','#ffcb9e','#aad9aa','#eea8a9','#a5a5a5'])
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    st.pyplot(fig) 
  with col11 :
    st.markdown("**Output : Estimated Mineralogy**") 
    fig, ax = plt.subplots()
    ax.pie(Y, labels = ['Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay'] ,
          autopct=lambda p: '{:.0f}%'.format(p),
          startangle = 90)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)    
    st.pyplot(fig)


#########################################################
#              TEST WITH SPECIFC MEASUREMENTS           #
#########################################################


st.subheader('Estimation with our method vs Measurements')
SPECIFIC_MEASUREMENT = st.selectbox(
    "Choose a CheMin sample:", 
    list(['RN','JK','CB','WJ','CH','MJ','TP','BK','BS','GB','GH','LB','OK',
'OU','MB','QL','SB','OB','DU','ST','HF','RH','AL','KM',
'GE','GE2','HU','EB','GG','MA','MA3','GR','NT','BD','PT','MG','ZS']),1)

record = np.array(data_2[data_2.Abreviation==SPECIFIC_MEASUREMENT]).flatten()
sum_of_x=record[3]+record[4]+record[5]+record[6]+record[7]+record[8]+record[9]
record_X = np.array([record[3],record[4],record[5],record[6],record[7],record[8],record[9],100-sum_of_x])
record_Y_measure = np.array([record[10],record[11],record[12],record[13],record[14],record[15],record[16],record[17]])
record_Y_estim = estim_mineral_func (record_X[:4])

col15, col16, col17 = st.columns(3)
with col15 :
    st.markdown("**Measured Geochemistry**")
    fig, ax = plt.subplots()
    plt.pie(record_X, labels = ['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O','Others'] ,
        autopct=lambda p: '{:.0f}%'.format(p),
        startangle = 90,
        colors=['#a5c8e1','#ffcb9e','#aad9aa','#eea8a9','#d4c2e4','#d0bbb7','#f3c8e6','#a5a5a5'])
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)    
    st.pyplot(fig)
with col16 :
    st.markdown("**Measured Mineralogy**")
    fig, ax = plt.subplots()
    plt.pie(record_Y_measure, labels = ['Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay'] ,
        autopct=lambda p: '{:.0f}%'.format(p),
        startangle = 90)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)    
    st.pyplot(fig)
with col17 :
    fig, ax = plt.subplots()
    st.markdown("**Estimated Mineralogy***")   
    plt.pie(record_Y_estim, labels = ['Amorphous','Pyroxene','Sulfate','Magnetite','Hematite','Feldspar','Plagioclase','Clay'] ,
        autopct=lambda p: '{:.0f}%'.format(p),
        startangle = 90) 
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)    
    st.pyplot(fig)

st.write ('*Note: Here, the estimation function of mineralogy includes major minerals only')


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

