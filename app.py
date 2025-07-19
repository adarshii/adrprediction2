# semaglutide_adr_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import gzip

# Load model components
@st.cache_resource
def load_components():
    model = joblib.load('semaglutide_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    top_reactions = joblib.load('top_reactions.pkl')
    return model, preprocessor, top_reactions

model, preprocessor, TOP_REACTIONS = load_components()

# Title
st.title("Semaglutide ADR Prediction & Risk Assessment")

# File Upload Section
st.subheader("Upload Patient Data")
csv_file = st.file_uploader("Upload Patient Metadata (.csv)", type=['csv'])
tsv_file = st.file_uploader("Upload Gene Expression Data (.tsv or .tsv.gz)", type=['tsv', 'gz'])

if csv_file is not None and tsv_file is not None:
    patient_data = pd.read_csv(csv_file)
    if tsv_file.name.endswith('.gz'):
        gene_data = pd.read_csv(tsv_file, sep='\t', compression='gzip', index_col=0)
    else:
        gene_data = pd.read_csv(tsv_file, sep='\t', index_col=0)

    st.success("Files uploaded successfully!")
    st.write("### Patient Metadata Preview")
    st.dataframe(patient_data.head())

    st.write("### Gene Expression Data Preview")
    st.dataframe(gene_data.iloc[:5, :5])
else:
    st.info("Please upload both CSV and TSV files to proceed.")

# Manual Form Input Section
st.subheader("Manual Entry: Patient Information")
with st.form("manual_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=58)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=85)
        sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    with col2:
        indication = st.selectbox("Indication for Semaglutide", ["Diabetes", "Weight Loss", "Other"])
        country = st.selectbox("Country", ["US", "UK", "CA", "AU", "DE", "FR", "JP", "Other"])
        reactions = st.text_input("Observed Reactions (comma-separated)", "Nausea, Vomiting")
    submitted = st.form_submit_button("Assess ADR Risk")

if submitted:
    reaction_list = [r.strip().lower() for r in reactions.split(",")]
    features = {
        'age': age,
        'wt': weight,
        'sex': sex,
        'country': country,
        'indication': indication,
    }
    for r in TOP_REACTIONS:
        features[f'react_{r}'] = 1 if r in reaction_list else 0
    
    features_df = pd.DataFrame([features])
    processed = preprocessor.transform(features_df)
    probability = model.predict_proba(processed)[0][1]

    if probability < 0.3:
        risk_category = "Low Risk"
        color = "green"
    elif probability < 0.7:
        risk_category = "Medium Risk"
        color = "orange"
    else:
        risk_category = "High Risk"
        color = "red"

    st.subheader("Risk Assessment")
    st.markdown(f"### <span style='color:{color}; font-size: 24px;'>{risk_category}</span>", unsafe_allow_html=True)
    st.progress(probability)
    st.markdown(f"**Probability of Serious ADR:** {probability:.1%}")

    st.subheader("Risk Factors Breakdown")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed)
    feature_names = preprocessor.get_feature_names_out()
    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values.values[0]}).sort_values('SHAP Value', ascending=False)
    top_risk_factors = shap_df[shap_df['SHAP Value'] > 0].head(10)

    if not top_risk_factors.empty:
        st.write("Top contributing factors to serious ADR risk:")
        for _, row in top_risk_factors.iterrows():
            feature = row['Feature'].replace('cat__', '').replace('react_', '')
            st.markdown(f"- **{feature}**: +{row['SHAP Value']:.2f} risk points")
    else:
        st.info("No significant risk factors identified")

    st.subheader("Clinical Recommendations")
    if probability > 0.7:
        st.warning("""
        - **Monitor closely** for serious adverse reactions
        - Weekly follow-ups recommended
        - Educate patient on warning signs (pancreatitis, renal issues)
        - Perform baseline lab tests
        """)
    elif probability > 0.3:
        st.info("""
        - Follow standard monitoring protocol
        - 2-week follow-up recommended
        - Consider gradual dose escalation
        """)
    else:
        st.success("""
        - Routine follow-up sufficient
        - Standard patient education
        - Monthly check-ins
        """)

    st.subheader("Risk Factor Analysis")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

# Sidebar
st.sidebar.header("Resources")
st.sidebar.markdown("""
- [Semaglutide Prescribing Info](https://www.novo-pi.com/ozempic.pdf)
- [FDA FAERS](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- [ADR Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK574518/)
""")

st.sidebar.header("About")
st.sidebar.markdown("""
- Based on 42,826 FAERS cases
- Built using XGBoost
- Clinical validation with known ADRs

**Note:** This tool is for support, not a substitute for clinical judgment.
""")
