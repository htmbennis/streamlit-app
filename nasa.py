import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Gale Crater", page_icon="📊", initial_sidebar_state="expanded"
)

st.title('📊 Gale Crater')

st.image('./curiosity_image.PNG'
  #, caption='Curiosity in the Gale Crater'
  )

st.subheader('From geochemistry data to mineralogy estimation')
#st.markdown('Streamlit is **_really_ cool**.')

DATA_FILE = ('nasa_input_1.csv')

#@st.cache
#data_load_state = st.text('Loading data...')
data = pd.read_csv(DATA_FILE)
#data_load_state.text("Data Loaded!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(data)
    #st.write(data)  #This is an other way to display table


SOLS_MIN_MAX = st.slider(
     'Select your range of sols to visualize (to zoom in and zoom out in the charts):',
     46, 3192, (46, 3192),
     help="Adjust range so you can focus on some data points in the graph.",)
#st.write('Values:', Sols_MIN_MAX[0])

LIST_OF_ATOMS = st.multiselect(
    "Choose geochimicals:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']), ['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O']
)

#IS_STAKED = st.selectbox(
#     'How would you like to display graphs?',
#     ('Stacked', 'Unstacked'),
#     help="Stacked or Unstacked.",)
#st.write('You selected:', IS_STAKED)


if not LIST_OF_ATOMS:
    st.error("Please select at least one atom.")

if LIST_OF_ATOMS:
  SOLS_MIN = SOLS_MIN_MAX[0]
  SOLS_MAX = SOLS_MIN_MAX[1]

  fig, ax = plt.subplots()
  #if IS_STAKED == 'Stacked' :
  ax.stackplot(data['Sols'],data[LIST_OF_ATOMS].T, alpha = 0.4)
  #if IS_STAKED == 'Unstacked' :
  #  ax.plot(data['Sols'],data[LIST_OF_ATOMS], alpha = 0.4) 

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
  ax.set_title('TITLE TO BE COMPLETED')
  st.pyplot(fig)

##Mineralogy Analysis

LIST_OF_MINERALS = st.multiselect(
    "Choose minerals:", 
    list(['Amorhous', 'Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay']), 
    ['Amorhous', 'Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay']
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
  ax.set_title('TITLE TO BE COMPLETED')
  st.pyplot(fig)


st.subheader('Geochemistry and mineralogy xyz analysis')

colA, colB, colC = st.columns(3)
with colA : X_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose X axis:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O',
      'Amorhous', 'Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay']),0)

with colB : Y_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose Y axis:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O',
      'Amorhous', 'Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay']),1)

with colC : Z_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose Z axis:", list(['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O',
      'Amorhous', 'Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay']),7)

fig, ax = plt.subplots()
ax.scatter(data[X_OF_XYZ_ANALYSIS],data[Y_OF_XYZ_ANALYSIS],c=data[Z_OF_XYZ_ANALYSIS], cmap='coolwarm',alpha = 0.4) 
im = ax.scatter(data[X_OF_XYZ_ANALYSIS],data[Y_OF_XYZ_ANALYSIS],c=data[Z_OF_XYZ_ANALYSIS], cmap='coolwarm',alpha = 0.4) 
#ax.legend(Z_OF_XYZ_ANALYSIS,
#    loc=5, 
#    bbox_to_anchor=(1.27, 0.5),
#    labelspacing=-2.5, # reverse legend
#    frameon=False,         
#    #fontsize=9.0
#    )
fig.colorbar(im)
ax.set_xlabel(X_OF_XYZ_ANALYSIS)
ax.set_ylabel(Y_OF_XYZ_ANALYSIS)
ax.grid(axis='both')
st.pyplot(fig)


st.subheader('Run analysis with your own data')

with st.form(key="mineralogy_estim_form"):
  st.write('Insert cocentration of each chemical:')
  #'SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Na2O'
  col1, col2, col3, col4 = st.columns(4)
  with col1: SIO2_INPUT = st.number_input('SiO2 (%)',min_value=0, max_value=100,value=50, step=1)
  with col2: FEO_INPUT = st.number_input('FeO (%)',min_value=0, max_value=100,value=10, step=1)
  with col3: AL2O3_INPUT = st.number_input('Al2O3 (%)',min_value=0, max_value=100,value=16, step=1)
  with col4: SO3_INPUT = st.number_input('SO3 (%)',min_value=0, max_value=100,value=5, step=1)
  col5, col6, col7, col8 = st.columns(4)
  with col5: CAO_INPUT = st.number_input('CaO (%)',min_value=0, max_value=100,value=6, step=1)
  with col6: MGO_INPUT = st.number_input('MgO (%)',min_value=0, max_value=100,value=5, step=1)
  with col7: OTHERS_INPUT = st.number_input('Others (%)',min_value=0, max_value=100,value=2, step=1)
  sum_of_inputs = SIO2_INPUT + FEO_INPUT + AL2O3_INPUT + SO3_INPUT + CAO_INPUT + MGO_INPUT + OTHERS_INPUT
  st.write(sum_of_inputs)
  submit_button = st.form_submit_button(label="Estimate mineralogy")
input_vector = [SIO2_INPUT,FEO_INPUT,AL2O3_INPUT,SO3_INPUT,CAO_INPUT,MGO_INPUT]

st.markdown("To do : Mettre ici une alerte si jamais la somme ne fait pas 100") 
A = np.array([
    [-2.11 , 0.44, -1.05 , 1.44, 0.077, -0.05476 , -0.1407],
    [-1.96 , 0.8507, -2.033,  2.141, 0.0406 ,-0.228 , 2.7615],
    [1.355 ,0.737, 0.3744 ,2.6085, 0.0663 , 1.291, 0.0485],
    [-2.851  ,1.2109, -3.2887, 2.1291, 0.006971, -0.90095 , 1.5591],
    [0, 0, 0, 0, 0 ,0, 0],
    [0 ,0 ,0 ,0 ,0 ,0 ,0]
  ])
B = np.array([150.04, -46.20 , 105.25 ,-142.33 , -4.43,  24.93 ,-47.16])
#X = np.array([44,22,8,3,6,8.6])
X = input_vector
AX = np.matmul(np.transpose(A),X)
Y = np.add(np.matmul(np.transpose(A),X),B)
Y = np.maximum(Y, 0)
Y = np.append(Y, 100 - np.sum(Y))
Y = np.maximum(Y, 0)
col9, col10= st.columns(2)
with col9 :
  st.markdown("**Input : Geochemistry**")
  fig, ax = plt.subplots()
  ax.pie(np.append(X,OTHERS_INPUT), labels = ['SiO2', 'FeO', 'Al2O3', 'SO3', 'CaO', 'MgO','Others'] ,
        autopct=lambda p: '{:.0f}%'.format(p),
        startangle = 90)
  st.pyplot(fig) 
with col10 :
  st.markdown("**Output : Estimated Mineralogy**") 
  fig, ax = plt.subplots()
  ax.pie(Y, labels = ['Mafic', 'Sulfate', 'Magnetite', 'Hematite', 'Quartz','Plagioclase','Clay','Amorhous'] ,
        autopct=lambda p: '{:.0f}%'.format(p),
        startangle = 90)
  st.pyplot(fig)


st.subheader('Estimation with our method vs Measurements')
X_OF_XYZ_ANALYSIS = st.selectbox(
    "Choose a measurement:", 
    list(['John_Klein','Cumberland','Confidence_Hills','Mojave','Telegraph_Peak',  
      'Buckskin','Big_Sky','Greenhorn','Lubango','Okoruso', 
      'Oudam','Marimba','Quela','Sebina','Duluth','Stoer','Highfield',  
      'Rock_Hall','Aberlady','Kilmarie','Glen Etive','Glen Etive 2']),0)
st.markdown("To do : Ensuite afficher 3 colonnes : Input, Output, Measurement avec des camembers pour le moment") 

st.subheader('Geospatial approach')
