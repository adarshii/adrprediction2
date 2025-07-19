# semaglutide_adr_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import gzip
import os
import traceback
pip install -r requirements.txt
streamlit run app.py
# Debugging: Show environment info
st.write("Python version:", sys.version)
st.write("Working directory:", os.getcwd())
st.write("Files in directory:", os.listdir('.'))

# Load model components with error handling
@st.cache_resource
def load_components():
    try:
        # Debug file sizes
        st.write("Model file size:", os.path.getsize('semaglutide_model.pkl'))
        st.write("Preprocessor file size:", os.path.getsize('preprocessor.pkl'))
        st.write("Top reactions file size:", os.path.getsize('top_reactions.pkl'))
        
        # Try different loading methods
        try:
            model = joblib.load('semaglutide_model.pkl')
        except:
            model = joblib.load('semaglutide_model.pkl', mmap_mode='r', encoding='latin1')
        
        preprocessor = joblib.load('preprocessor.pkl', mmap_mode='r')
        top_reactions = joblib.load('top_reactions.pkl')
        
        return model, preprocessor, top_reactions
    except Exception as e:
        st.error(f"LOAD ERROR: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()

try:
    model, preprocessor, TOP_REACTIONS = load_components()
    st.success("Model components loaded successfully!")
except:
    st.error("Failed to load model components")

# Title
st.title("Semaglutide ADR Prediction & Risk Assessment")

# PubChem Data Section (using your provided data)
st.subheader("Semaglutide Chemical Information")
if st.checkbox("Show PubChem Compound Data"):
    pubchem_data = {
        "Compound CID": [56843331, 162393099],
        "Name": ["Semaglutide", "Semaglutide Acetate"],
        "Molecular Formula": ["C187H291N45O59", "C189H295N45O61"],
        "Molecular Weight": [4114.0, 4174.0],
        "SMILES": ["CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C21)...", "CC(C)(C)OC(=O)[C@H](CCC(=O)NCCOCCOCC(=O)NCCOCCOCC(=O)O)NC(=O)CCCCCCCCCCCCCCCCC(=O)O"]
    }
    pubchem_df = pd.DataFrame(pubchem_data)
    st.dataframe(pubchem_df)

# File Upload Section
st.subheader("Upload Patient Data")
csv_file = st.file_uploader("Upload Patient Metadata (.csv)", type=['csv'])
tsv_file = st.file_uploader("Upload Gene Expression Data (.tsv or .tsv.gz)", type=['tsv', 'gz'])

if csv_file and tsv_file:
    try:
        patient_data = pd.read_csv(csv_file)
        if tsv_file.name.endswith('.gz'):
            gene_data = pd.read_csv(tsv_file, sep='\t', compression='gzip', index_col=0)
        else:
            gene_data = pd.read_csv(tsv_file, sep='\t', index_col=0)
            
        st.success("Files uploaded successfully!")
        st.write("### Patient Metadata Preview")
        st.dataframe(patient_data.head(3))
        
        st.write("### Gene Expression Preview")
        st.dataframe(gene_data.iloc[:3, :3])
        
    except Exception as e:
        st.error(f"Error reading files: {str(e)}")

# Manual Input Section
st.subheader("Manual Patient Assessment")
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 55)
        weight = st.number_input("Weight (kg)", 40, 200, 85)
        bmi = st.number_input("BMI", 15.0, 50.0, 28.5)
        sex = st.radio("Sex", ["Male", "Female", "Other"])
        
    with col2:
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension")
        kidney_disease = st.checkbox("Kidney Disease")
        reactions = st.multiselect("Observed Reactions", 
                                  ["Nausea", "Vomiting", "Diarrhea", "Constipation", "Abdominal Pain"],
                                  ["Nausea"])
        
    submitted = st.form_submit_button("Assess ADR Risk")
    
    if submitted:
        # Create feature dictionary
        features = {
            'age': age,
            'weight': weight,
            'bmi': bmi,
            'sex': sex.lower(),
            'diabetes': int(diabetes),
            'hypertension': int(hypertension),
            'kidney_disease': int(kidney_disease),
        }
        
        # Add reactions
        for r in TOP_REACTIONS:
            features[f'react_{r}'] = 1 if r in reactions else 0
        
        try:
            # Transform and predict
            features_df = pd.DataFrame([features])
            processed = preprocessor.transform(features_df)
            probability = model.predict_proba(processed)[0][1]
            
            # Display results
            st.subheader("Risk Assessment")
            risk_level = "High Risk 🔴" if probability > 0.7 else "Medium Risk 🟠" if probability > 0.4 else "Low Risk 🟢"
            st.metric("ADR Probability", f"{probability:.1%}", risk_level)
            st.progress(probability)
            
            # Explainability
            st.subheader("Key Risk Factors")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(processed)
            
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=7, show=False)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error(traceback.format_exc())

# Sidebar Resources
st.sidebar.header("Clinical Resources")
st.sidebar.download_button("Download Semaglutide Prescribing Info", 
                          data=open("semaglutide_info.pdf", "rb").read(),
                          file_name="semaglutide_prescribing_info.pdf")

st.sidebar.header("About This Tool")
st.sidebar.info("""
- **Model**: XGBoost classifier
- **Training Data**: 42,826 FAERS reports
- **Version**: 2.1.0
- **Last Updated**: 2023-11-15
""")
joblib.dump(model, 'semaglutide_model.pkl', protocol=4, compress=3)

# Add PubChem link
st.sidebar.markdown("[PubChem Semaglutide Data](https://pubchem.ncbi.nlm.nih.gov/compound/Semaglutide)")
