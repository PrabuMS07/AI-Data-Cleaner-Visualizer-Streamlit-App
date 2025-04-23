import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import ast
from sklearn.preprocessing import LabelEncoder


OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"


device = "cuda" if torch.cuda.is_available() else "cpu"  


df = pd.read_csv("C:/Users/PRABU/Downloads/mini pro/titanic.csv")

label_encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col])

df_numeric = df.select_dtypes(include=[float, int])


df_tensor = torch.tensor(df_numeric.values).to(device)


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

def get_llama_response(prompt):
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
    )
    if response.status_code == 200:
        return response.json().get("response")
    else:
        return None


def apply_cleaning_steps(df, cleaning_code):
    try:
        cleaning_steps = ast.literal_eval(cleaning_code) 
        exec(cleaning_steps)  
        return df
    except Exception as e:
        print("Error applying cleaning steps:", e)
        return df


dynamic_prompt = generate_prompt(df)
llama_response = get_llama_response(dynamic_prompt)

if llama_response:
    print("LLaMA Suggested Cleaning Steps:")
    print(llama_response)
    df = apply_cleaning_steps(df, llama_response)  


df_tensor = torch.tensor(df_numeric.values).to(device)


df_tensor = df_tensor.to("cpu")  


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


df.to_csv("cleaned_titanic.csv", index=False)
print("Data preprocessing completed and saved as 'cleaned_titanic.csv'.")







