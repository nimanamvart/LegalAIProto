import streamlit as st
import json
import faiss
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables (OpenAI key)
load_dotenv()

# Load model (cloud-compatible)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # Small, fast, and works on Streamlit Cloud

# Load structured law content
with open("real_eu_laws.json", "r", encoding="utf-8") as f:
    real_laws = json.load(f)

# Prepare searchable index
def flatten_laws(laws):
    passages = []
    meta = []
    for law in laws:
        for art in law["articles"]:
            for para in art["paragraphs"]:
                ref = f"{law['title']}, Article {art['article']}, Paragraph {para['number']}"
                passages.append(para['text'])
                meta.append({"text": para['text'], "ref": ref, "url": law['url']})
    return passages, meta

texts, metadata = flatten_laws(real_laws)
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# UI
st.set_page_config(page_title="EU Law Q&A", layout="centered")
st.title("📘 EU Law Assistant")

query = st.text_input("Ask a question about EU digital laws:")

# Keyword filter (warn if not relevant)
allowed_keywords = [
    "data", "gdpr", "privacy", "platform", "processing", "controller", "ai", "artificial intelligence",
    "gatekeeper", "dma", "dsa", "services", "market", "regulation", "consent", "user", "personal data",
    "automated decision", "intermediary", "provider", "access", "sharing", "dataset", "open data", 
    "eprivacy", "security", "cyber", "incident", "compliance", "obligation", "law", "eu", "municipality"
]

if query:
    lower_query = query.lower()
    if not any(keyword in lower_query for keyword in allowed_keywords):
        st.warning("⚠️ This question may not be clearly related to EU digital laws. Please ensure your terms are specific.")

    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)

    relevant_contexts = [metadata[i]["text"] for i in I[0]]
    references = [metadata[i] for i in I[0]]

    system_prompt = "You are a legal assistant answering questions about EU digital laws. Answer clearly. Use YES or NO if appropriate and include references."
    user_prompt = f"Question: {query}\n\nRelevant legal texts:\n" + "\n\n".join(relevant_contexts)

    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("Missing API key")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
    except Exception:
        top = references[0]
        answer = f"Top match: {top['ref']}\n\n{top['text']}"

    st.subheader("Answer")
    st.write(answer)

    st.markdown("### References")
    for ref in references:
        st.write(f"**{ref['ref']}**")
        st.markdown(f"[View full law]({ref['url']})")
        st.markdown("---")
