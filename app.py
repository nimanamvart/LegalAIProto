
import streamlit as st
import json
import faiss
import os
from sentence_transformers import SentenceTransformer, models
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

# Load model from local path
word_embedding_model = models.Transformer("./", max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Load structured law content
with open("real_eu_laws.json", "r", encoding="utf-8") as f:
    real_laws = json.load(f)

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

st.set_page_config(page_title="EU Law Q&A", layout="centered")
st.title("📘 EU Law Assistant")

query = st.text_input("Ask a question about EU digital laws:")

if query:
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
