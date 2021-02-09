import streamlit as st
import pandas as pd 
import seaborn as sns
sns.set_style('whitegrid')
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

#personalise
#display thayer school logo
imagee = cv2.imread('images/new_logo.png')
cv2.imshow('Image', imagee)
st.image(imagee, caption='Thayer School of Engineering at Dartmouth')

# front end elements of the web page 
html_temp = """ 
<div style ="background-color:yellow;padding:13px"> 
<h1 style ="color:black;text-align:center;">PATIENT SURVIVAL PREDICTION APP</h1> 
</div> 
"""
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 



#create containers
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

#write in header section
with header:
    st.title("PCADS- CAPSTONE PROJECT")
    st.text('Developed by ....Subramanian Hariharan ')
    st.text('Date ...........February 2021')
    st.markdown("> “You can have data without information, but you cannot have information without data.” \n\n—Daniel Keys Moran")
    st.header('Problem Statement')
    st.markdown("""The dataset is sourced from open domain and is about data related to a hospital in Greenland. It is the responsibility of Hospital Administration to provide efficient and timely patient care to ensure that the individual concerned recovers from the illness completely. The Hospital admistration is seeking answers for improving the standard 
                of medical care by addressing related factors. The data is related to a particular department in the Hospital where mortality rate is beyond acceptable levels. Accordingly, it is understood that towards this, the data related to patient has been obtained from multiple SQL tables in the database and collated. The historical data collates information 
                regarding demographics of the patient as well as treatment and medical condition and includes a binary variable which indicates whether the patient survived at the end of 12 months of care. The attributes  which affect the patient survival in a significant way can also be flagged.
                The **Objective** of the Project is to analyse the factors given in the dataset and predict chance of survival of patient at the end of 12 months of treatment. 
                with dataset:""")
    st.header('PATIENT DATASET')
    st.text('This Dataset is available in open domain...https://dphi.tech/')
data = pd.read_csv('dataset/patient_survival.csv')#read the csv file into df
#function to show first five rows of trg dataset
def view_dataset(data):
    st.text('A glance at top 5 rows of dataset')
    s = data.head().style.background_gradient(cmap='viridis')
    st.dataframe(s) # output first 5 lines of dataset
    return None 
    


def show_features():
        
    st.header('Dataset Features')
    st.markdown('* **The features of this dataset are** ')
    
    st.markdown("""The "Survived_1_year" column is a target variable which has binary entries (0 or 1).

                Survived_1_year == 0, implies that the patient did not survive after 1 year of treatment Survived_1_year == 1, implies that the patient survived after 1 year of treatment

                The features in the dataset are as follows :-

                -ID_Patient_Care_Situation: Care situation of a patient during treatment

                -Diagnosed_Condition: The diagnosed condition of the patient

                -ID_Patient: Patient identifier number

                -Treatment_with_drugs: Class of drugs used during treatment

                -Survived_1_year: If the patient survived after one year (0 means did not survive; 1 means survived)

                -Patient_Age: Age of the patient

                -Patient_Body_Mass_Index: A calculated value based on the patient’s weight, height, etc.

                -Patient_Smoker: If the patient was a smoker or not

                -Patient_Rural_Urban: If the patient stayed in Rural or Urban part of the country

                -Previous_Condition: Condition of the patient before the start of the treatment ( This variable is splitted into 8 columns - A, B, C, D, E, F, Z and Number_of_prev_cond. A, B, C, D, E, F and Z are the previous conditions of the patient. Suppose for one patient, if the entry in column A is 1, it means that the previous condition of the patient was A. If the patient didn't have that condition, it is 0 and same for other conditions. If a patient has previous condition as A and C , columns A and C will have entries as 1 and 1 respectively while the other column B, D, E, F, Z will have entries 0, 0, 0, 0, 0 respectively. The column Number_of_prev_cond will have entry as 2 i.e. 1 + 0 + 1 + 0 + 0 + 0 + 0 + 0 = 2 in this case. )""")
    return None

    #display some plots
def plot_graphs(data):
    st.header('Some Visualisation of Training Data...')
    plt.figure(figsize=(4,1))
    st.subheader('Count plot of Survived Column in Dataset')
    sns.countplot(x=data['Survived_1_year'],color='g',palette='Set1')
    st.pyplot()
    
    numeric_columns=['Patient_Age','Patient_Body_Mass_Index','Diagnosed_Condition','Number_of_prev_cond']
    st.markdown("**Histogram of Numeric Attributes**")
    f=plt.figure(figsize=(20,10))
    i=1
    for col in numeric_columns:
        f.add_subplot(2,2,i)
        sns.histplot(data=data,x=col,hue='Survived_1_year',kde=True,palette='Set1')
        i+=1
    st.pyplot()

    # box plot of numeric variables
    st.markdown('**Box Plots of Numeric Attributes with Survived column as hue**')
    f=plt.figure(figsize=(20,10))
    i=1
    for col in numeric_columns:
        f.add_subplot(2,2,i)
        sns.boxplot(y=col,data=data,x='Survived_1_year',showmeans=True,palette='Set1')
        i+=1
    st.pyplot()
    #countplots of categorical attributes
    st.markdown('**Count plot of Smokers and Rural/Urban with Survived as hue**')
    f=plt.figure(figsize=(20,10))
    f.add_subplot(1,2,1)
    sns.countplot(data['Patient_Smoker'],hue=data['Survived_1_year'],palette='Set1')
    f.add_subplot(1,2,2)
    sns.countplot(data['Patient_Rural_Urban'],hue=data['Survived_1_year'],palette='Set1')
    st.pyplot(f)
    return None

with features:
#plot if check box clicked
    if st.checkbox('View first five rows of Training Dataset'):
        view_dataset(data)
    if st.checkbox('Description of Features'):
        show_features()
    if st.checkbox('Load Plots For Training Dataset'):
        plot_graphs(data)
    


#the model_training block displays input data and prediction
with model_training:
    st.header('Model Prediction')
    st.subheader('Model Predicts whether a patient will survive at the end of one year of treatment based on user input of features')
    sel_col,disp_col = st.beta_columns(2)
    st.write('Please Enter Patient Attributes for Model Prediction')

    
    #patient previous condition
    A = st.selectbox('What is Patient Previous Condiion "A" ?',(0.0,1.0))#drop down
    B = st.selectbox('What is Patient Previous Condiion "B" ?',(0.0,1.0))#drop down
    C = st.selectbox('What is Patient Previous Condiion "C" ?',(0.0,1.0))#drop down
    D = st.selectbox('What is Patient Previous Condiion "D" ?',(0.0,1.0))#drop down
    E = st.selectbox('What is Patient Previous Condiion "E" ?',(0.0,1.0))#drop down
    
    
    #patient treatment drug
    DX1 = st.selectbox('What is Patient Treated Drug "DX1" ?',(0.0,1.0))#drop down
    DX2 = st.selectbox('What is Patient Treated Drug "DX2" ?',(0.0,1.0))#drop down
    DX3 = st.selectbox('What is Patient Treated Drug "DX3" ?',(0.0,1.0))#drop down
    DX4 = st.selectbox('What is Patient Treated Drug "DX4" ?',(0.0,1.0))#drop down
    DX5 = st.selectbox('What is Patient Treated Drug "DX5" ?',(0.0,1.0))#drop down
    DX6 = st.selectbox('What is Patient Treated Drug "DX6" ?',(0.0,1.0))#drop down
    

    #patient smoker or no
    smoker = st.selectbox('What is Smoking Habit "Yes/No/Cannot Say" ?',('Yes','No','Cannot_Say'))#drop down
    
    if smoker == 'Yes':
        Patient_Smoker_YES =1.0
        Patient_Smoker_NO =0.0
    elif smoker == 'Cannot_Say':
        Patient_Smoker_YES =0.0
        Patient_Smoker_NO =1.0
    else:
        Patient_Smoker_YES =0.0
        Patient_Smoker_NO =1.0

    #patient rural or urban
    geog = st.selectbox('Where does patient stay "Rural/Urban" ?',('Rural','Urban'))#drop down
    if geog == 'Rural':
        Patient_Rural_Urban_RURAL =1.0
        Patient_Rural_Urban_URBAN =0.0
    else:
        Patient_Rural_Urban_RURAL =0.0
        Patient_Rural_Urban_URBAN =1.0

    #patient ID
    Patient_ID = st.number_input("Enter Patient ID")
    Patient_ID = float(Patient_ID)
    
    # Number_of_prev_cond
    Number_of_prev_cond= st.number_input("Enter Number of prev condition")
    Number_of_prev_cond = float(Number_of_prev_cond)

    # patient Age
    Patient_Age = st.number_input("Enter Patient Age")
    Patient_Age = float(Patient_Age)
    

    # Diagnosed_Condition
    Diagnosed_Condition = st.number_input("Enter Patient Diagnosed_Condition")
    Diagnosed_Condition = float(Diagnosed_Condition)
    

    #Patient_Body_Mass_Index
    Patient_Body_Mass_Index = st.number_input('Input BMI of Patient', value=1.0)
    Patient_Body_Mass_Index = float(Patient_Body_Mass_Index)
    

    # No_of_treatment_drugs
    No_of_treatment_drugs = DX1+DX2+DX3+DX4+DX5+DX6
    No_of_treatment_drugs = float(No_of_treatment_drugs)

        

#run the function
#Diagnosed_Condition, Patient_ID, Patient_Age,Patient_Body_Mass_Index,A, B, C,D,E,Number_of_prev_cond,DX1,DX2,DX3,DX4,DX5,DX6,Patient_Smoker_NO,Patient_Smoker_YES,Patient_Rural_Urban_RURAL,Patient_Rural_Urban_URBAN,No_of_treatment_drugs = get_user_data()

# Load the model from the file 
XGB_from_joblib = joblib.load('patient_survival.pkl')

st.write('A dataframe constructed with User Input Attributes is ...')   

X_test = pd.DataFrame({'Diagnosed_Condition':Diagnosed_Condition,
                        'Patient_ID':Patient_ID, 
                        'Patient_Age':Patient_Age,
                        'Patient_Body_Mass_Index':Patient_Body_Mass_Index,
                         'A':A, 'B':B, 'C':C, 'D':D, 'E':E,
                        'Number_of_prev_cond':Number_of_prev_cond, 
                        'DX1':DX1, 'DX2':DX2, 'DX3':DX3, 'DX4':DX4, 'DX5':DX5, 'DX6':DX6,
                        'Patient_Smoker_NO':Patient_Smoker_NO,
                        'Patient_Smoker_YES':Patient_Smoker_YES,
                        'Patient_Rural_Urban_RURAL':Patient_Rural_Urban_RURAL,
                        'Patient_Rural_Urban_URBAN':Patient_Rural_Urban_URBAN, 
                        'No_of_treatment_drugs':No_of_treatment_drugs},index=[1])
st.write(X_test)

               
# Use the loaded model to make predictions 
@st.cache
def authenticate(username, password):
    return username == "admin" and password == "admin"

username = st.text_input('username',value='admin')
password = st.text_input('password')

st.cache()
def load_authorize(username,password):
    import time
    time.sleep(5)
    return username,password

st.cache()
def predict_model(X):
    result = XGB_from_joblib.predict(X)
    if result == 0:
        answer = 'Not_Survived_1_year'
    else:
        answer = 'Survived_1_year'

    st.subheader('The Model prediction for the given patient....')
    st.success(answer)
    prob = pd.DataFrame(XGB_from_joblib.predict_proba(X_test),columns=['Not_Survived','Survived'])
    st.success('Probability for Survived_1_year')
    st.write(prob)

if st.button("Predict"):
    data = load_authorize(username,password)

if authenticate(username, password):
    st.success('You are authenticated!')
    predict_model(X_test)    
else:
    st.error('The username or password you have entered is invalid.')
