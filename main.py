# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Prediction of early death among bone metastasis patients: development and validation using machine learning based on 118,227 patients from the SEER database")
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
        st.markdown(f"Based on the findings, it is recommended that patients in the low-risk groups should be considered for invasive surgeries, such as excisional surgery, and long-course radiotherapy. This recommendation is based on the understanding that these patients may experience a prolonged period of poor quality of life if only palliative interventions are performed. By opting for more aggressive treatment approaches, there is a potential to improve the overall quality of life and potentially extend survival for these patients. However, individual patient factors, preferences, and the overall clinical context should be taken into account when making treatment decisions. Therefore, it is strongly advised to consult with healthcare professionals to assess the specific needs and circumstances of each patient before determining the most suitable treatment plan.")
    else:
        st.markdown(f"Based on the results, it is evident that patients in the high-risk groups face a significantly higher risk of early death compared to those in the low-risk groups (4.5-fold higher, P<0.001). In light of this, it is advised to avoid open surgery for these patients. Instead, treatment options such as radiotherapy alone, best supportive care, or minimally invasive techniques like cementoplasty should be considered to palliatively alleviate pain and improve quality of life. By opting for these alternative approaches, patients can receive effective pain management and supportive care without subjecting themselves to the potential risks and complications associated with open surgery. It is crucial to prioritize the well-being and comfort of the patients in the high-risk groups, and these treatment modalities offer a more suitable and safe approach. However, it is important to note that treatment decisions should always be individualized and based on a comprehensive assessment of each patient's specific condition, preferences, and overall clinical context. Consulting with healthcare professionals is strongly recommended to determine the most appropriate treatment plan for patients in the high-risk groups.")

st.subheader('About the model')
st.markdown('The algorithm used in this freely accessible online calculator is based on the gradient boosting machine. Internal validation of the model demonstrated an area under the curve (AUC) of 0.858 (95% CI: 0.851-0.865), indicating its strong predictive performance. Furthermore, external validation of the model yielded an AUC of up to 0.847 (95% CI: 0.798-0.895), confirming its robustness and generalizability beyond the initial dataset. While the model achieved good prediction performance, it is important to note that its use should be limited to research purposes only. This means that the model can be utilized to gain insights, explore relationships, and generate hypotheses in a research setting. However, it is not recommended for direct application in clinical or real-world decision-making without further validation and consideration of other factors. Additional research, external validation, and rigorous evaluation are necessary before considering the deployment of the model in practical settings.')
