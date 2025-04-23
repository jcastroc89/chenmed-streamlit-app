pip install faiss-cpu

pip install streamlit

# Packages Used
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import streamlit as st
import os

# API key
with open("API_KEY.txt", "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

# The RAG will work based on the ICD 10 data extracted from the governmets page
with open("icd10cm_codes_2025.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

icd_data = []
for line in lines:
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        code, description = parts
        icd_data.append({"code": code, "text": f"{code}: {description}"})

# generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [entry["text"] for entry in icd_data]
embeddings = model.encode(texts, convert_to_numpy=True)

# since it took a long time to generate the embeddings i want to save them
np.save("icd10_embeddings.npy", embeddings)
pd.DataFrame(icd_data).to_csv("icd10_code_map.csv", index=False)


embeddings = np.load("icd10_embeddings.npy")
icd_df = pd.read_csv("icd10_code_map.csv")

# building Faiss index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
index_map = {i: icd_data[i] for i in range(len(icd_data))}

# Streamlit
st.set_page_config(page_title="ICD-10 Rules Chatbot", page_icon="🧠")
st.title("🩺 ICD-10 Rules Generator Chatbot")
st.write("Ask about an ICD-10 code, like `E11.9` or a disease like `Ebola`. We'll generate diagnostic rules for you.")

user_input = st.text_input("Enter ICD-10 code or disease:")


# Retrieving and generating the rules based on the txt files
def retrieve_and_generate_rules(query: str, top_k: int = 1) -> list:
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = [index_map[idx]["text"] for idx in indices[0]]
    context = "\n".join(retrieved_chunks)

    prompt = f'''
[SYSTEM ROLE]
You are a certified expert in medical coding and billing, specializing in ICD-10 classification. You have deep knowledge of diagnostic criteria, coding guidelines, and best practices. You excel at analyzing patient summaries and identifying applicable ICD-10 codes, along with the clinical reasoning behind them.

[USER CONTEXT]
The human user is a medical coding student who is learning how to accurately apply ICD-10 codes. They want to understand the decision-making process and criteria used by experts to assign specific codes.

[TASK]
You will be provided with an ICD-10 code. Your task is to return **eight diagnostic rules or criteria** that must be met for this code to be assigned. These may include clinical symptoms, lab results, physician documentation, coding guidelines, or exclusion criteria.

[OUTPUT FORMAT]
Return a clear, concise, bulleted list of 8 rules or criteria.
- Each bullet should be phrased in a way that is informative and suitable for copy-pasting into a `.docx` file.
- Do not include introductory or concluding text.

ICD-10 Reference:
{context}
'''

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical coding assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip().split("\n")

st.subheader("📋 Diagnostic Rules")
st.markdown(rules)

st.download_button("⬇️ Download Rules as Text File", data=rules, file_name=f"{code}_rules.txt")


# example
rules = retrieve_and_generate_rules("A984 Ebola virus disease")
for rule in rules:
    print(rule)
