import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import os
# Configuration
#API_URL = "http://127.0.0.1:8000"  # Adjust if your FastAPI runs elsewhere
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="ML Submission Portal", layout="wide")

st.title("🛡️ Model & Data Quality Dashboard")
st.markdown("Upload your model artifacts and data to trigger automated quality checks.")




# --- Sidebar: Identity & Selection ---
with st.sidebar:
    st.header("Submission Management")
    
    # Fetch existing IDs from the API
    try:
        response = requests.get(f"{API_URL}/submissions")
        if response.status_code == 200:
            existing_ids = response.json()
        else:
            existing_ids = []
    except Exception:
        existing_ids = []
        st.error("Could not connect to backend to fetch IDs.")

    # UI for selecting or creating an ID
    mode = st.radio("Mode", ["Select Existing", "Create New"])
    
    if mode == "Select Existing" and existing_ids:
        submission_id = st.selectbox("Choose a Submission ID", options=existing_ids)
    else:
        submission_id = st.text_input("Enter New Submission ID", value="user_001")
        if mode == "Select Existing" and not existing_ids:
            st.warning("No existing submissions found. Please create one.")

    st.divider()
    st.info(f"Active ID: **{submission_id}**")

# --- Tabs for different steps ---
tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "📊 Data Profiling", "🔍 Model Scan"])

# --- TAB 1: UPLOAD ---
with tab1:
    st.header("Upload Artifacts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Files")
        model_file = st.file_uploader("Upload model.py", type=["py"])
        checkpoint_file = st.file_uploader("Upload checkpoint", type=None)
        
        if st.button("Submit Model"):
            if model_file and checkpoint_file and submission_id:
                files = {
                    "model_file": (model_file.name, model_file.getvalue()),
                    "checkpoint_file": (checkpoint_file.name, checkpoint_file.getvalue())
                }
                params = {"submission_id": submission_id}
                res = requests.post(f"{API_URL}/upload/model", params=params, files=files)
                if res.status_code == 200:
                    st.success("Model uploaded successfully!")
                else:
                    st.error(f"Upload failed: {res.json().get('detail')}")

    with col2:
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload data.csv", type=["csv"])
        
        if st.button("Submit Data"):
            if data_file and submission_id:
                files = {"file": (data_file.name, data_file.getvalue())}
                params = {"submission_id": submission_id}
                res = requests.post(f"{API_URL}/upload/data", params=params, files=files)
                if res.status_code == 200:
                    st.success("Data uploaded successfully!")
                else:
                    st.error(f"Upload failed: {res.json().get('detail')}")

# --- TAB 2: DATA PROFILING ---
with tab2:
    st.header("Data Profiling Report")
    if st.button("Generate Profiling Report"):
        with st.spinner("Running ydata-profiling..."):
            res = requests.get(f"{API_URL}/check_data", params={"submission_id": submission_id})
            if res.status_code == 200:
                st.components.v1.html(res.text, height=800, scrolling=True)
            else:
                st.warning(res.json().get("detail", "Error fetching report"))

# --- TAB 3: MODEL SCAN ---
with tab3:
    st.header("Giskard Model Scan")
    st.markdown("Scans for robustness, bias, and data leakage (EU AI Act Compliance).")
    
    if st.button("Run Comprehensive Scan"):
        with st.spinner("Scanning model (this may take a minute)..."):
            try:
                res = requests.get(f"{API_URL}/check_model", params={"submission_id": submission_id})
                if res.status_code == 200:
                    st.components.v1.html(res.text, height=1000, scrolling=True)
                else:
                    st.error(res.json().get("detail", "Error during scan"))
            except Exception as e:
                st.error(f"Connection Error: {e}")