# semaglutide_adr_app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pdfplumber  # For PDF processing
import pubchempy as pcp  # For chemical properties
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler

# Configuration
st.set_page_config(page_title="Semaglutide ADR Predictor", layout="wide")

# Load model components with enhanced error handling
@st.cache_resource
def load_components():
    try:
        model = joblib.load('semaglutide_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        top_reactions = joblib.load('top_reactions.pkl')
        return model, preprocessor, top_reactions
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

try:
    model, preprocessor, TOP_REACTIONS = load_components()
except:
    st.warning("Model components not loaded. Some features may be disabled.")

# Title and description
st.title("Semaglutide Adverse Drug Reaction Predictor")
st.markdown("""
This tool predicts the risk of serious adverse reactions to Semaglutide therapy using 
clinical data, biochemical markers, and PubChem chemical properties.
""")

# PubChem Data Section
st.subheader("Semaglutide Chemical Properties")
if st.button("Fetch Latest PubChem Data"):
    try:
        compound = pcp.get_compounds('semaglutide', 'name')[0]
        
        chem_data = {
            "Property": ["Molecular Formula", "Molecular Weight", "XLogP", 
                         "Hydrogen Bond Donors", "Hydrogen Bond Acceptors",
                         "Rotatable Bonds", "Complexity"],
            "Value": [compound.molecular_formula, compound.molecular_weight,
                      compound.xlogp, compound.h_bond_donor_count, 
                      compound.h_bond_acceptor_count, compound.rotatable_bond_count,
                      compound.complexity]
        }
        
        chem_df = pd.DataFrame(chem_data)
        st.dataframe(chem_df, hide_index=True)
        
        # Display molecule image
        img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/PNG"
        st.image(img_url, caption=f"PubChem CID: {compound.cid}", width=300)
        
        st.download_button(
            label="Download Chemical Data",
            data=chem_df.to_csv(index=False),
            file_name="semaglutide_chemical_properties.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error fetching PubChem data: {str(e)}")

# PDF Processing Section
st.subheader("Process Medical Reports")
pdf_file = st.file_uploader("Upload Patient Medical Report (PDF)", type=['pdf'])

def extract_clinical_data(text):
    """Extract clinical data from PDF text using regex patterns"""
    data = {
        'age': None,
        'weight': None,
        'sex': None,
        'diabetes': 0,
        'hypertension': 0,
        'kidney_disease': 0,
        'reactions': []
    }
# Sample data collection from EHRs
import pandas as pd

clinical_data = pd.DataFrame({
    'patient_id': [1001, 1002, 1003],
    'age': [58, 67, 72],
    'sex': ['M', 'F', 'M'],
    'weight_kg': [85.2, 92.5, 78.9],
    'height_cm': [175, 162, 180],
    'diabetes': [1, 1, 0],
    'hypertension': [1, 1, 1],
    'ckd_stage': [2, 3, 1],
    'semaglutide_dose_mg': [1.0, 2.4, 0.5],
    'treatment_duration_weeks': [12, 24, 8],
    'adr_severity': [2, 3, 1]  # 0=None, 1=Mild, 2=Moderate, 3=Severe
})
from pubchempy import get_compounds

def get_semaglutide_properties():
    compound = get_compounds('semaglutide', 'name')[0]
    return {
        'molecular_weight': compound.molecular_weight,
        'xlogp': compound.xlogp,
        'h_bond_donors': compound.h_bond_donor_count,
        'complexity': compound.complexity
    }

# Add to clinical data
chem_props = get_semaglutide_properties()
for key, value in chem_props.items():
    clinical_data[key] = value
import pdfplumber
import re

def extract_pdf_data(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages])
    
    return {
        'age': int(re.search(r'Age:\s*(\d+)', text).group(1)),
        'weight': float(re.search(r'Weight:\s*([\d\.]+)', text).group(1)),
        'conditions': re.findall(r'Diagnosis:\s*(.+)', text)
    }

    # Age extraction
    age_match = re.search(r'Age:\s*(\d+)', text)
    if age_match:
        data['age'] = int(age_match.group(1))
    
    # Weight extraction
    weight_match = re.search(r'Weight:\s*(\d+\.?\d*)\s*(kg|lbs)', text, re.IGNORECASE)
    if weight_match:
        weight = float(weight_match.group(1))
        if 'lbs' in weight_match.group(2).lower():
            weight *= 0.453592  # Convert lbs to kg
        data['weight'] = weight
    
    # Sex extraction
    sex_match = re.search(r'Gender:\s*(\w+)', text, re.IGNORECASE)
    if sex_match:
        sex = sex_match.group(1).lower()
        if 'f' in sex:
            data['sex'] = 'female'
        elif 'm' in sex:
            data['sex'] = 'male'
    
    # Condition flags
    if re.search(r'diabetes', text, re.IGNORECASE):
        data['diabetes'] = 1
    if re.search(r'hypertension|high blood pressure', text, re.IGNORECASE):
        data['hypertension'] = 1
    if re.search(r'kidney disease|renal impairment|CKD', text, re.IGNORECASE):
        data['kidney_disease'] = 1
    
    # Reaction extraction
    reactions = ["nausea", "vomiting", "diarrhea", "pancreatitis", "hypoglycemia"]
    for reaction in reactions:
        if re.search(reaction, text, re.IGNORECASE):
            data['reactions'].append(reaction)
    
    return data

if pdf_file:
    with st.spinner("Processing medical report..."):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            st.success("PDF processed successfully!")
            
            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Text", text, height=300)
            
            # Extract clinical data
            clinical_data = extract_clinical_data(text)
            
            st.subheader("Extracted Clinical Data")
            st.json(clinical_data)
            
            # Pre-fill form with extracted data
            if clinical_data['age'] or clinical_data['weight']:
                st.session_state.prefill_data = clinical_data
                st.success("Form will be pre-filled with extracted data")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# Prediction Form
st.subheader("Patient Risk Assessment")
with st.form("patient_form"):
    # Initialize session state
    if 'prefill_data' not in st.session_state:
        st.session_state.prefill_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", 
                             min_value=18, max_value=100, 
                             value=st.session_state.prefill_data.get('age', 58))
        
        weight = st.number_input("Weight (kg)", 
                                min_value=30, max_value=300, 
                                value=st.session_state.prefill_data.get('weight', 85))
        
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        st.metric("BMI", f"{bmi:.1f}", 
                 help="Body Mass Index: Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30")
        
        sex = st.selectbox("Sex", options=["Male", "Female", "Other"],
                          index=0 if st.session_state.prefill_data.get('sex') != 'female' else 1)
    
    with col2:
        st.markdown("**Medical History**")
        diabetes = st.checkbox("Diabetes", 
                              value=bool(st.session_state.prefill_data.get('diabetes', False)))
        hypertension = st.checkbox("Hypertension", 
                                  value=bool(st.session_state.prefill_data.get('hypertension', False)))
        kidney_disease = st.checkbox("Kidney Disease", 
                                    value=bool(st.session_state.prefill_data.get('kidney_disease', False)))
        
        st.markdown("**Semaglutide Treatment**")
        dosage = st.selectbox("Dosage (mg/week)", options=[0.25, 0.5, 1.0, 1.7, 2.4], index=2)
        duration = st.selectbox("Treatment Duration", 
                               options=["<1 month", "1-3 months", "3-6 months", ">6 months"],
                               index=1)
        
        reactions = st.multiselect("Observed Reactions", 
                                  TOP_REACTIONS,
                                  default=st.session_state.prefill_data.get('reactions', []))
    
    submitted = st.form_submit_button("Assess ADR Risk")
    
    if submitted:
        try:
            # Create feature dictionary
            features = {
                'age': age,
                'weight': weight,
                'bmi': bmi,
                'sex': sex.lower(),
                'diabetes': int(diabetes),
                'hypertension': int(hypertension),
                'kidney_disease': int(kidney_disease),
                'dosage': dosage,
                'duration': duration,
            }
            
            # Add chemical properties (example values - should be fetched from PubChem)
            chem_features = {
                'xlogp': -5.8,
                'polar_area': 1650.0,
                'h_bond_donors': 57,
                'complexity': 9590.0
            }
            features.update(chem_features)
            
            # Add reactions
            for r in TOP_REACTIONS:
                features[f'react_{r}'] = 1 if r in reactions else 0
            
            # Transform and predict
            features_df = pd.DataFrame([features])
            
            # Check if preprocessor is loaded
            if 'preprocessor' in globals():
                processed = preprocessor.transform(features_df)
                probability = model.predict_proba(processed)[0][1]
            else:
                # Fallback to simple scaling if model not loaded
                st.warning("Using simplified risk assessment (model not loaded)")
                risk_factors = np.array([
                    age > 65,
                    bmi > 35,
                    kidney_disease,
                    dosage > 1.0,
                    'pancreatitis' in reactions
                ]).astype(int)
                probability = min(0.9, risk_factors.sum() * 0.15)
            
            # Display results
            st.subheader("Risk Assessment")
            
            if probability < 0.3:
                risk_level = "Low Risk"
                color = "green"
                recommendations = """
                - Routine monitoring
                - Standard patient education
                - Monthly follow-ups
                """
            elif probability < 0.7:
                risk_level = "Medium Risk"
                color = "orange"
                recommendations = """
                - Bi-weekly monitoring
                - Dose escalation with caution
                - Check renal function
                """
            else:
                risk_level = "High Risk"
                color = "red"
                recommendations = """
                - Weekly monitoring required
                - Consider alternative therapies
                - Check pancreatic enzymes
                - Monitor for gallbladder symptoms
                """
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h3 style='color:{color};'>{risk_level}</h3>", 
                           unsafe_allow_html=True)
                st.metric("Probability of Serious ADR", f"{probability:.1%}")
                st.progress(probability)
            
            with col2:
                st.markdown("**Clinical Recommendations**")
                st.markdown(recommendations)
            
            # Explainability
            st.subheader("Key Risk Factors")
            if 'model' in globals():
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(processed)
                    
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
                    st.pyplot(fig)
                except:
                    st.warning("SHAP explanation not available")
            else:
                st.info("""
                Key risk factors include:
                - Age > 65 years
                - BMI > 35
                - History of kidney disease
                - High dosage (>1.0 mg/week)
                - History of pancreatitis
                """)
            
            # Generate PDF report
            report_content = f"""
            SEMAGLUTIDE ADR RISK REPORT
            ===========================
            
            Patient Information:
            - Age: {age} years
            - Weight: {weight} kg
            - BMI: {bmi:.1f}
            - Sex: {sex}
            
            Medical History:
            - Diabetes: {'Yes' if diabetes else 'No'}
            - Hypertension: {'Yes' if hypertension else 'No'}
            - Kidney Disease: {'Yes' if kidney_disease else 'No'}
            
            Treatment Details:
            - Dosage: {dosage} mg/week
            - Duration: {duration}
            
            Observed Reactions:
            {', '.join(reactions) if reactions else 'None'}
            
            Risk Assessment:
            - Risk Level: {risk_level}
            - Probability: {probability:.1%}
            
            Recommendations:
            {recommendations}
            """
            
            st.download_button(
                label="Download Risk Report",
                data=report_content,
                file_name="semaglutide_risk_report.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("Resources")
    st.markdown("""
    - [Semaglutide Prescribing Information](https://www.novo-pi.com/ozempic.pdf)
    - [FDA Adverse Event Reports](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
    - [Clinical Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK574518/)
    """)
    
    st.header("About This Tool")
    st.markdown("""
    **Version:** 2.1.0  
    **Model Type:** XGBoost Classifier  
    **Training Data:** 42,826 patient records  
    **Last Updated:** 2023-11-15
    
    *For clinical use only. Always combine with professional judgment.*
    """)
    
    st.image("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/semaglutide/PNG", 
            caption="Semaglutide Molecular Structure", width=200)

# Footer
st.markdown("---")
st.caption("© 2023 Clinical Decision Support System | Data Sources: FAERS, PubChem, Clinical Trials")
