import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib



diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = pickle.load(open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\models\heart.sav', 'rb'))
parkinson_model = joblib.load("models/parkinsons1.sav")

lung_cancer_model= joblib.load("models/lung_cancer_model.sav")



breast_cancer_model = joblib.load('models/breast_cancer.sav')


chronic_disease_model = joblib.load('models/kidney_model1.pkl')



hepatitis_model = joblib.load('models/hepititis_model.sav')


liver_model = joblib.load('models/liver_model.sav')
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

logo = Image.open(r"C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\logo2.png")  
st.image(logo, width=150)  
st.title("Multiple Disease Prediction System")


# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Disease Prediction',
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Parkison Prediction',
        'Liver prediction',
        'Hepatitis prediction',
        'Lung Cancer Prediction',
        'Chronic Kidney prediction',
        'Breast Cancer Prediction',

    ],
        icons=['','activity', 'heart', 'person','person','person','person','bar-chart-fill'],
        default_index=0)




# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')


    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    
    if st.button('Predict'): 
        
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')




# Diabetes prediction page
if selected == 'Diabetes Prediction':  
    st.title("Diabetes disease prediction")
    image = Image.open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\diabetes image.jpg')
    st.image(image, caption='diabetes disease prediction')
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnencies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure  value")
    with col1:

        SkinThickness = st.number_input("Sckinthickness value")

    with col2:

        Insulin = st.number_input("Insulin value ")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.number_input(
            "Diabetespedigreefunction value")
    with col2:

        Age = st.number_input("AGE")

    # code for prediction
    diabetes_dig = ''

    
    if st.button("Diabetes test result"):
        diabetes_prediction=[[]]
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

       
        if diabetes_prediction[0] == 1:
            diabetes_dig = "we are really sorry to say but it seems like you are Diabetic."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            diabetes_dig = 'Congratulation,You are not diabetic'
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + diabetes_dig)
    
    if st.button("Consult a Doctor", key="consult_diabetes"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)

        
        



# Heart prediction page
if selected == 'Heart disease Prediction':
    st.title("Heart disease prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='heart failuire')
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1)

    with col2:
        sex_display = ("Female", "Male")
        sex_value = st.selectbox("Gender", range(len(sex_display)), format_func=lambda x: sex_display[x])
        sex = sex_value  

    with col3:
        cp_display = ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic")
        cp_value = st.selectbox("Chest Pain Type", range(len(cp_display)), format_func=lambda x: cp_display[x])
        cp = cp_value 

    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250)

    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)

    with col3:
        restecg_display = ("Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy")
        restecg_value = st.selectbox("Resting ECG", range(len(restecg_display)), format_func=lambda x: restecg_display[x])
        restecg = restecg_value  

    with col1:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)

    with col2:
        oldpeak = st.number_input("ST Depression (Oldpeak)")

    with col3:
        slope_display = ("Upsloping", "Flat", "Downsloping")
        slope_value = st.selectbox("Slope of Peak Exercise ST Segment", range(len(slope_display)), format_func=lambda x: slope_display[x])
        slope = slope_value  

    with col1:
        ca = st.number_input("Major Vessels Colored by Fluoroscopy (0-3)", min_value=0, max_value=3, step=1)

    with col2:
        thal_display = ("Normal", "Fixed Defect", "Reversible Defect")
        thal_value = st.selectbox("Thalassemia", range(len(thal_display)), format_func=lambda x: thal_display[x])
        thal = thal_value  

    with col3:
        exang = st.checkbox("Exercise Induced Angina")
        exang = 1 if exang else 0

    with col1:
        fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")
        fbs = 1 if fbs else 0

    # Prediction
    heart_dig = ''

    if st.button("Heart Test Result"):
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]]

        heart_prediction = heart_model.predict(input_data)

        if heart_prediction[0] == 1:
            heart_dig = 'We are really sorry to say but it seems like you have Heart Disease.'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            heart_dig = "Congratulations, You don't have Heart Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

    
        if name.strip():
            st.success(name + ', ' + heart_dig)
        else:
            st.success(heart_dig)

    if st.button("Consult a Doctor", key="consult_heart"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)
    

# Parkinson's Disease      
       
if selected == 'Parkison Prediction':
    st.title("Parkison prediction")
    image = Image.open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\parkinson.jpg')
    st.image(image, caption='Parkinson disease prediction')
  

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP = st.number_input("Fo", help="MDVP:Fo - Fundamental frequency (Hz)")
    with col2:
        MDVPFIZ = st.number_input("Fhi", help="MDVP:Fhi - Maximum vocal fundamental frequency (Hz)")
    with col3:
        MDVPFLO = st.number_input("Flo", help="MDVP:Flo - Minimum vocal fundamental frequency (Hz)")

    with col1:
        MDVPJITTER = st.number_input("Jitter(%)", help="MDVP:Jitter(%) - Variation in fundamental frequency")
    with col2:
        MDVPJitterAbs = st.number_input("Jitter(Abs)", help="MDVP:Jitter(Abs) - Absolute jitter value")
    with col3:
        MDVPRAP = st.number_input("RAP", help="MDVP:RAP - Relative amplitude perturbation")

    with col1:
        MDVPPPQ = st.number_input("PPQ", help="MDVP:PPQ - Pitch period perturbation quotient")
    with col2:
        JitterDDP = st.number_input("DDP", help="Jitter:DDP - Difference of differences of periods")
    with col3:
        MDVPShimmer = st.number_input("Shimmer", help="MDVP:Shimmer - Amplitude variation")

    with col1:
        MDVPShimmer_dB = st.number_input("Shimmer(dB)", help="Shimmer(dB) - Amplitude variation in dB")
    with col2:
        Shimmer_APQ3 = st.number_input("APQ3", help="Shimmer:APQ3 - 3-point amplitude perturbation quotient")
    with col3:
        ShimmerAPQ5 = st.number_input("APQ5", help="Shimmer:APQ5 - 5-point amplitude perturbation quotient")

    with col1:
        MDVP_APQ = st.number_input("APQ", help="MDVP:APQ - Amplitude perturbation quotient")
    with col2:
        ShimmerDDA = st.number_input("DDA", help="Shimmer:DDA - Average absolute difference of amplitudes")
    with col3:
        NHR = st.number_input("NHR", help="NHR - Noise-to-harmonics ratio")

    with col1:
        HNR = st.number_input("HNR", help="HNR - Harmonics-to-noise ratio")
    with col2:
        RPDE = st.number_input("RPDE", help="RPDE - Recurrence period density entropy")
    with col3:
        DFA = st.number_input("DFA", help="DFA - Detrended fluctuation analysis")

    with col1:
        spread1 = st.number_input("Spread1", help="Spread1 - Nonlinear measure of frequency variation")
    with col2:
        spread2 = st.number_input("Spread2", help="Spread2 - Nonlinear measure of frequency variation")
    with col3:
        D2 = st.number_input("D2", help="D2 - Correlation dimension")

    with col1:
        PPE = st.number_input("PPE", help="PPE - Pitch period entropy")

    
    input_data = [
        MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP,
        MDVPPPQ, JitterDDP, MDVPShimmer, MDVPShimmer_dB, Shimmer_APQ3,
        ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR, RPDE, DFA,
        spread1, spread2, D2, PPE
    ]

    parkinson_dig = ''

    if st.button("Parkinson test result"):
        if name.strip() == '' or all(val == 0.0 for val in input_data):
            st.warning("⚠️ Please enter your name and fill in all the parameters before predicting.")
        else:
            prediction = parkinson_model.predict([input_data])
            if prediction[0] == 1:
                parkinson_dig = 'We are really sorry to say but it seems like you have Parkinson disease'
                image = Image.open('positive.jpg')
                st.image(image)
            else:
                parkinson_dig = "Congratulations, You don't have Parkinson disease"
                image = Image.open('negative.jpg')
                st.image(image)
            st.success(f"{name}, {parkinson_dig}")
    if st.button("Consult a Doctor", key="consult_Parkinson"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)




lung_cancer_data = pd.read_csv('data/lung_cancer.csv')


lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})



if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    image = Image.open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\lung.png')
    st.image(image, caption='Lung Cancer Prediction')

    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("Age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    
    
    cancer_result = ''

    
    if st.button("Predict Lung Cancer"):
        
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        
        user_data.columns = user_data.columns.str.strip()

        
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        
        cancer_prediction = lung_cancer_model.predict(user_data)

        
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + cancer_result)
    if st.button("Consult a Doctor", key="consult_Lung"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)





if selected == 'Liver prediction':  
    st.title("Liver disease prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col2:
        age = st.number_input("Entre your age")
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") 
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") 
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") 
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") 
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")
    with col3:
        Albumin = st.number_input("Entre your Albumin") 
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio")  
    
    liver_dig = ''

    
    if st.button("Liver test result"):
        liver_prediction=[[]]
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

        
        if liver_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            liver_dig = "we are really sorry to say but it seems like you have liver disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            liver_dig = "Congratulation , You don't have liver disease."
        st.success(name+' , ' + liver_dig)
    if st.button("Consult a Doctor", key="consult_Liver"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)






# Hepatitis prediction page
if selected == 'Hepatitis prediction':
    st.title("Hepatitis Prediction")
    image = Image.open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\hepatitis.jpg')
    st.image(image, caption='Hepatitis Prediction')

    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Enter your age") 
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 2
    with col3:
        total_bilirubin = st.number_input("Enter your Total Bilirubin") 

    with col1:
        direct_bilirubin = st.number_input("Enter your Direct Bilirubin")  
    with col2:
        alkaline_phosphatase = st.number_input("Enter your Alkaline Phosphatase") 
    with col3:
        alamine_aminotransferase = st.number_input("Enter your Alamine Aminotransferase")  

    with col1:
        aspartate_aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")  
    with col2:
        total_proteins = st.number_input("Enter your Total Proteins") 
    with col3:
        albumin = st.number_input("Enter your Albumin") 

    with col1:
        albumin_and_globulin_ratio = st.number_input("Enter your Albumin and Globulin Ratio") 

    with col2:
        your_ggt_value = st.number_input("Enter your GGT value") 
    with col3:
        your_prot_value = st.number_input("Enter your PROT value")  

    
    hepatitis_result = ''

    
    if st.button("Predict Hepatitis"):
        
        user_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ALB': [total_bilirubin],  
            'ALP': [direct_bilirubin],  
            'ALT': [alkaline_phosphatase], 
            'AST': [alamine_aminotransferase],
            'BIL': [aspartate_aminotransferase],  
            'CHE': [total_proteins],  
            'CHOL': [albumin],  
            'CREA': [albumin_and_globulin_ratio],  
            'GGT': [your_ggt_value],  
            'PROT': [your_prot_value]  
        })

        # Perform prediction
        hepatitis_prediction = hepatitis_model.predict(user_data)
        
        if hepatitis_prediction[0] == 1:
            hepatitis_result = "We are really sorry to say but it seems like you have Hepatitis."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            hepatitis_result = 'Congratulations, you do not have Hepatitis.'
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + hepatitis_result)
    if st.button("Consult a Doctor", key="consult_Hepatitis"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)


from sklearn.preprocessing import LabelEncoder
import joblib


# Chronic Kidney Disease Prediction Page
if selected == 'Chronic Kidney prediction':
    st.title("Chronic Kidney Disease Prediction")
    image = Image.open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\kidney.jpeg')
    st.image(image, caption='Kidney Disease Prediction')
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Enter your age", 1, 100, 25)  
    with col2:
        bp = st.slider("Enter your Blood Pressure", 50, 200, 120)  
    with col3:
        sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02)  

    with col1:
        al = st.slider("Enter your Albumin", 0, 5, 0)  
    with col2:
        su = st.slider("Enter your Sugar", 0, 5, 0)  
    with col3:
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        rbc = 1 if rbc == "Normal" else 0

    with col1:
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        pc = 1 if pc == "Normal" else 0
    with col2:
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        pcc = 1 if pcc == "Present" else 0
    with col3:
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        ba = 1 if ba == "Present" else 0

    with col1:
        bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120)  
    with col2:
        bu = st.slider("Enter your Blood Urea", 10, 200, 60)  
    with col3:
        sc = st.slider("Enter your Serum Creatinine", 0, 10, 3)  

    with col1:
        sod = st.slider("Enter your Sodium", 100, 200, 140)  
    with col2:
        pot = st.slider("Enter your Potassium", 2, 7, 4)  
    with col3:
        hemo = st.slider("Enter your Hemoglobin", 3, 17, 12)  

    with col1:
        pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40)  
    with col2:
        wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000)  
    with col3:
        rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4)  

    with col1:
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        htn = 1 if htn == "Yes" else 0
    with col2:
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        dm = 1 if dm == "Yes" else 0
    with col3:
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        cad = 1 if cad == "Yes" else 0

    with col1:
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        appet = 1 if appet == "Good" else 0
    with col2:
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        pe = 1 if pe == "Yes" else 0
    with col3:
        ane = st.selectbox("Anemia", ["Yes", "No"])
        ane = 1 if ane == "Yes" else 0

    
    kidney_result = ''

    
    if st.button("Predict Chronic Kidney Disease"):
        
        user_input = pd.DataFrame({
            'age': [age],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'su': [su],
            'rbc': [rbc],
            'pc': [pc],
            'pcc': [pcc],
            'ba': [ba],
            'bgr': [bgr],
            'bu': [bu],
            'sc': [sc],
            'sod': [sod],
            'pot': [pot],
            'hemo': [hemo],
            'pcv': [pcv],
            'wc': [wc],
            'rc': [rc],
            'htn': [htn],
            'dm': [dm],
            'cad': [cad],
            'appet': [appet],
            'pe': [pe],
            'ane': [ane]
        })

        
        kidney_prediction = chronic_disease_model.predict(user_input)
        
        if kidney_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = "we are really sorry to say but it seems like you have kidney disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = "Congratulation , You don't have kidney disease."
        st.success(name+' , ' + kidney_prediction_dig)
    if st.button("Consult a Doctor", key="consult_Kidney"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)



# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction")
    image = Image.open(r'C:\Users\ISHITA\OneDrive\Documents\Desktop\home1\project1\Multiple-Disease-Prediction-Webapp\Frontend\breast.jpg')
    st.image(image, caption='Breast Cancer Prediction')
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0)
        texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0)
        perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0)

    with col2:
        area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0)
        smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1)
        compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15)

    with col3:
        concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2)
        concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1)
        symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5)

    with col1:
        fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05)
        radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0)
        texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0)

    with col2:
        perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0)
        area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0)
        smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01)

    with col3:
        compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1)
        concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02)
        concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01)

    with col1:
        symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5)
        fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05)

    with col2:
        radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0)
        texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0)
        perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0)

    with col3:
        area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0)
        smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15)
        compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3)

    with col1:
        concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4)
        concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1)
        symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5)

    with col2:
        fractal_dimension_worst = st.slider("Enter your Fractal Dimension Worst", 0.01, 0.2, 0.1)

        
    breast_cancer_result = ''

    
    if st.button("Predict Breast Cancer"):
        
        user_input = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],  
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave points_se': [concave_points_se],  
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],  
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst],
        })

        
        breast_cancer_prediction = breast_cancer_model.predict(user_input)
        
        if breast_cancer_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you have Breast Cancer."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."

        st.success(breast_cancer_result)
    if st.button("Consult a Doctor", key="consult_breast"):
        st.markdown("[Click here to consult a doctor on Practo](https://www.practo.com)", unsafe_allow_html=True)
