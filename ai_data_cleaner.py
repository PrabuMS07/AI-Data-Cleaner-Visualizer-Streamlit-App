import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import re
import numpy as np
import os
import streamlit as st
import uuid
import shutil
from pathlib import Path


# Constants
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_DIR = "temp"

# Create temporary directory
os.makedirs(TEMP_DIR, exist_ok=True)

# Streamlit app layout
st.title("Dataset Preprocessing and Visualization")
st.write("Upload a CSV dataset to preprocess, summarize, visualize, and compare.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

# 🔸 Generate dataset summary prompt
def generate_summary_prompt(df):
    summary = {
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape,
    }

    prompt = (
        "You are a professional data analyst. Based on the summary of the dataset provided below, follow a step-by-step process to understand what the dataset is about and extract meaningful insights.\n\n"
        "First, identify the structure of the dataset from the number of rows, columns, and data types.\n"
        "Then, look at the missing values and identify where data quality issues might exist.\n"
        "Analyze column names and infer what kind of information the dataset contains.\n"
        "Think through each column, whether it's categorical or numeric, and what role it might play (e.g., target variable, identifier, feature).\n"
        "Reflect on any visible patterns or points of interest (e.g., imbalance, column diversity, data density).\n"
        "Use this thinking to write:\n\n"
        "*Description*: A short paragraph explaining what the dataset is likely about.\n"
        "*Insights*: At least 8 to 10 insightful observations based strictly on the provided summary (do not assume unlisted stats).\n\n"
        "Be factual. Do not generate synthetic examples or code. Avoid hallucinations.\n\n"
        f"Dataset Summary:\n"
        f"Columns: {summary['columns']}\n"
        f"Missing Values: {summary['missing_values']}\n"
        f"Data Types: {summary['dtypes']}\n"
        f"Shape: {summary['shape']}\n\n"
    )

    return prompt

# 🔸 Generate preprocessing prompt
def generate_preprocessing_prompt(df, dataset_path, cleaned_path):
    summary = {
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
    }

    prompt = (
        "You are a skilled Python data preprocessing expert.\n"
        "Your task is to generate robust and production-safe Python code that performs preprocessing on any given tabular dataset.\n\n"
        f"Dataset Path: {dataset_path}\nOutput Path: {cleaned_path}\n\n"
        "Dataset Overview:\n"
        f"Missing Values: {summary['missing_values']}\n"
        f"Data Types: {summary['dtypes']}\n"
        f"Categorical Columns: {summary['categorical_columns']}\n"
        f"Numeric Columns: {summary['numeric_columns']}\n\n"

        "Instructions:\n"
        "- Load the dataset from the given path into a DataFrame.\n"
        "- Dynamically identify numeric and categorical columns using select_dtypes.\n"
        "- Drop columns with more than 80% missing values or columns with >=90% identical values (i.e., low variance).\n"
        "- Impute missing numeric columns with median using SimpleImputer.\n"
        "- Impute missing categorical columns with mode using SimpleImputer.\n"
        "- For categorical columns, use a single instance of LabelEncoder, applied within a loop.\n"
        "- Handle outliers using IQR (set outliers to NaN, then re-impute).\n"
        "- Scale all numeric columns using StandardScaler from sklearn.\n"
        "- Ensure all column operations include checks for existence and avoid hardcoding column names.\n"
        "- Update column lists if columns are dropped or transformed.\n"
        "- Save the cleaned dataset to the provided output path.\n\n"

        "Constraints:\n"
        "- Do not hardcode column names.\n"
        "- Do not use non-standard libraries or deprecated methods.\n"
        "- Ensure the code is fully executable in a Python script.\n"
        "- Ensure all transformations are applied to the main DataFrame only.\n"
        "- Avoid hallucinations. Generate code strictly based on instructions.\n\n"

        "Your Output:\n"
        "- Return only clean and complete Python code. No markdown, comments, or explanations.\n"
        "- Handle every possible dataset shape and structure robustly.\n"
    )

    return prompt





# 🔸 Clean LLM response
def clean_llm_response(response):
    if not response:
        return None
    cleaned = re.sub(r'```python\n|```|\n\s*#[^\n]*', '', response).strip()
    return cleaned

# 🔸 LLaMA API Request
def get_llama_response(prompt):
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        )
        if response.status_code == 200:
            raw_response = response.json().get("response")
            return clean_llm_response(raw_response)
        else:
            st.error(f"⚠️ Ollama API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error connecting to Ollama: {e}")
        return None

# 🔸 Apply generated code
def apply_generated_code(df, code, code_type="preprocessing"):
    try:
        local_env = {
            "df": df.copy(), "pd": pd, "SimpleImputer": SimpleImputer,
            "LabelEncoder": LabelEncoder, "StandardScaler": StandardScaler,
            "sns": sns, "plt": plt, "np": np, "os": os
        }
        with open(os.path.join(TEMP_DIR, f"{code_type}_code.py"), "w") as f:
            f.write(code)
        exec(code, globals(), local_env)
        if code_type == "preprocessing":
            return local_env.get("df", df)
        return True
    except KeyError as e:
        st.error(f"⚠️ KeyError in {code_type} code: {e}. Check dataset columns.")
        return df if code_type == "preprocessing" else False
    except NameError as e:
        st.error(f"⚠️ NameError in {code_type} code: {e}")
        return df if code_type == "preprocessing" else False
    except Exception as e:
        st.error(f"⚠️ Error in {code_type} code: {e}")
        return df if code_type == "preprocessing" else False

# Process uploaded file
if uploaded_file is not None:
    try:
        # Save uploaded file to temp directory
        file_id = str(uuid.uuid4())
        dataset_path = os.path.join(TEMP_DIR, f"dataset_{file_id}.csv")
        cleaned_path = os.path.join(TEMP_DIR, f"cleaned_{file_id}.csv")
        
        # Read and save the uploaded CSV
        df = pd.read_csv(uploaded_file)
        df.to_csv(dataset_path, index=False)
        st.success("✅ Dataset uploaded successfully!")

        # Display dataset summary
        st.subheader("Dataset Summary")
        summary_prompt = generate_summary_prompt(df)
        summary_response = get_llama_response(summary_prompt)
        if summary_response:
            st.text_area("Summary and Insights", summary_response, height=200)
        else:
            st.error("⚠️ Failed to generate dataset summary.")

        # Preprocess the dataset
        st.subheader("Preprocessing")
        preprocessing_prompt = generate_preprocessing_prompt(df, dataset_path, cleaned_path)
        preprocessing_code = get_llama_response(preprocessing_prompt)
        if preprocessing_code:
            st.code(preprocessing_code, language="python")
            df_cleaned = apply_generated_code(df, preprocessing_code, "preprocessing")
            df_cleaned.to_csv(cleaned_path, index=False)
            st.success(f"✅ Cleaned dataset saved as {cleaned_path}")
        else:
            st.error("⚠️ Failed to generate preprocessing code.")
            df_cleaned = df.copy()  # Fallback to original dataset

        # Generate and display visualizations
        st.subheader("Visualizations")
        visualization_prompt = generate_visualization_prompt(df_cleaned, cleaned_path)
        visualization_code = get_llama_response(visualization_prompt)
        if visualization_code:
            st.code(visualization_code, language="python")
            if apply_generated_code(df_cleaned, visualization_code, "visualization"):
                # Display all PNG files in temp directory
                for file in Path(TEMP_DIR).glob("*.png"):
                    st.image(str(file), caption=file.name)
                st.success("📈 Visualizations generated successfully.")
            else:
                st.error("⚠️ Failed to generate visualizations.")
        else:
            st.error("⚠️ Failed to generate visualization code.")

        # Display original and cleaned datasets side by side
        st.subheader("Original vs Cleaned Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Dataset**")
            st.dataframe(df, height=300)
        with col2:
            st.write("**Cleaned Dataset**")
            st.dataframe(df_cleaned, height=300)

    except Exception as e:
        st.error(f"⚠️ Error processing dataset: {e}")

else:
    st.info("Please upload a CSV file to begin.")

# Cleanup (optional, uncomment to enable)
# if os.path.exists(TEMP_DIR):
#     shutil.rmtree(TEMP_DIR)      streamlit run AI_Data_Cleaner.py