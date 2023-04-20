# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Development and validation of an interpretable machine learning scoring tool for estimating early death among patients with bone metastasis: a retrospective, population-based cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Age = st.sidebar.slider("Age", 30, 100)
Sex = st.sidebar.selectbox("Sex", ("Female", "Male"))
Race = st.sidebar.selectbox("Race", ("Black", "Others","Unknown", "White"))
Primarysite = st.sidebar.selectbox("Primary site", ("Slow", "Moderate", "Rapid"))
Maritalstatus = st.sidebar.selectbox("Marital status", ("Divorced", "Married", "Separated", "Never married", "Unknown", "Unmarried or Domestic Partner", "Widowed"))
Ruralurban = st.sidebar.selectbox("Rural urban", ("Metropolitan counties", "Nonmetropolitan counties", "Unknown"))
Tstage= st.sidebar.selectbox("T stage", ("T0", "T1", "T2", "T3", "T4", "Tx"))
Nstage= st.sidebar.selectbox("N stage", ("N0", "N1", "N2", "N3", "Nx"))
Brainm = st.sidebar.selectbox("Brain metastasis", ("No", "Unknown", "Yes"))
Liverm = st.sidebar.selectbox("Liver metastasis", ("No", "Unknown", "Yes"))
Lungm = st.sidebar.selectbox("Lung metastasis", ("No", "Unknown", "Yes"))
Cancerdirectedsurgery = st.sidebar.selectbox("Cancer directed surgery", ("No", "Unknown", "Yes"))
Radiation = st.sidebar.selectbox("Radiation", ("No/Unknown", "Yes"))
Chemotherapy = st.sidebar.selectbox("Chemotherapy", ("No/Unknown", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("gbm_clf_final_round-web.pkl")
    x = pd.DataFrame([[Age, Sex, Race, Primarysite, Maritalstatus, Ruralurban, Tstage, Nstage, Brainm, Liverm, Lungm, Cancerdirectedsurgery, Radiation, Chemotherapy]],
                     columns=["Age", "Sex", "Race", "Primarysite", "Maritalstatus", "Ruralurban", "Tstage", "Nstage", "Brainm", "Liverm", "Lungm", "Cancerdirectedsurgery", "Radiation", "Chemotherapy"])
    x = x.replace(["Female", "Male"], [1, 2])
    x = x.replace(["Black", "Others", "Unknown", "White"], [1, 2, 3, 4])
    x = x.replace(["Slow", "Moderate", "Rapid"], [1, 2, 3])
    x = x.replace(["Divorced", "Married", "Separated", "Never married", "Unknown", "Unmarried or Domestic Partner", "Widowed"], [1, 2, 3, 4, 5, 6, 7])
    x = x.replace(["Metropolitan counties", "Nonmetropolitan counties", "Unknown"], [1, 2, 3])
    x = x.replace(["T0", "T1","T2", "T3", "T4", "Tx"], [1, 2, 3, 4, 5, 6])
    x = x.replace(["N0", "N1","N2", "N3", "Nx"], [1, 2, 3, 4, 5])
    x = x.replace(["No", "Unknown", "Yes"], [1, 2, 3])
    x = x.replace(["No/Unknown", "Yes"], [1, 2])


    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of experiencing early death: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.37:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.37:
        st.markdown(f"Recommendations: For patients in the low-risk groups, invasive surgery, such as excisional surgery, and long-course radiotherapy were recommended, because those patients might suffer from poor quality of life for a very long time, if only palliative interventions were performed.")
    else:
        st.markdown(f"Recommendations: Patients in the high-risk groups were 4.5-fold chances to suffer from early death than patients in the low-risk groups (P<0.001). Open surgery was not recommended to those patients. They should better be treated with radiotherapy alone, best supportive care, or minimal invasive techniques such as cementoplasty to palliatively alleviate pain.")

st.subheader('About the model')
st.markdown('This online calculator is freely accessible, and it’s algorithm was based on the gradient boosting machine. Internal validation showed that the AUC of the model was 0.858 (95% CI: 0.851-0.865); External validation was also up to 0.847 (95% CI: 0.798-0.895).')