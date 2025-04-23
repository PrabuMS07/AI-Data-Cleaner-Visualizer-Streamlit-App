import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import ast
from sklearn.preprocessing import LabelEncoder

# Define Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU

# Load Titanic dataset (modify the path accordingly)
df = pd.read_csv("C:/Users/PRABU/Downloads/mini pro/titanic.csv")

# Handle non-numeric columns (label encoding for categorical columns)
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col])

# Select only numeric columns for tensor conversion
df_numeric = df.select_dtypes(include=[float, int])

# Convert the DataFrame of numeric values into a PyTorch tensor
df_tensor = torch.tensor(df_numeric.values).to(device)

# Generate dataset-specific prompt
def generate_prompt(df):
    missing_values = df.isnull().sum().to_dict()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    prompt = (
        "Analyze the following dataset and suggest data cleaning steps in Python code.\n\n"
        f"Missing Values:\n{missing_values}\n\n"
        f"Categorical Columns:\n{categorical_columns}\n\n"
        "Return only executable Python code in JSON format."
    )
    return prompt

# Get response from LLaMA
def get_llama_response(prompt):
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
    )
    if response.status_code == 200:
        return response.json().get("response")
    else:
        return None

# Apply suggested cleaning steps
def apply_cleaning_steps(df, cleaning_code):
    try:
        cleaning_steps = ast.literal_eval(cleaning_code)  # Convert JSON to dict
        exec(cleaning_steps)  # Execute the cleaning steps
        return df
    except Exception as e:
        print("Error applying cleaning steps:", e)
        return df

# Generate the prompt and get response
dynamic_prompt = generate_prompt(df)
llama_response = get_llama_response(dynamic_prompt)

if llama_response:
    print("LLaMA Suggested Cleaning Steps:")
    print(llama_response)
    df = apply_cleaning_steps(df, llama_response)  # Apply cleaning

# Convert DataFrame back to tensor (if needed for further computation)
df_tensor = torch.tensor(df_numeric.values).to(device)

# Move DataFrame tensor back to CPU (if you need to convert for visualization)
df_tensor = df_tensor.to("cpu")  # If needed for visualization steps on CPU

# Visualization - Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)
print("Data preprocessing completed and saved as 'cleaned_titanic.csv'.")
