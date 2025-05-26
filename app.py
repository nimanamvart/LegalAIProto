import streamlit as st
import json
import faiss
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables (OpenAI API key)
load_dotenv()

# Initialize OpenAI client (new SDK >=1.0 format)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Load EU law database
with open("real_eu_laws.json", "r", encoding="utf-8") as f:
    real_laws = json.load(f)

# Flatten recitals and articles into searchable format
def flatten_laws(laws):
    passages = []
    meta = []
    for law in laws:
        for rec in law.get("recitals", []):
            ref = f"{law['title']}, Recital ({rec['number']})"
            passages.append(rec["text"])
            meta.append({"text": rec["text"], "ref": ref, "url": law["url"]})
        for art in law.get("articles", []):
            for para in art.get("paragraphs", []):
                ref = f"{law['title']}, Article {art['article']}, Paragraph {para['number']}"
                passages.append(para["text"])
                meta.append({"text": para["text"], "ref": ref, "url": law["url"]})
    return passages, meta

texts, metadata = flatten_laws(real_laws)
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Streamlit interface
st.set_page_config(page_title="EU Law Q&A", layout="centered")
st.title("📘 EU Law Assistant")

query = st.text_input("Ask a question about EU digital laws:")

# Optional: warn if unrelated
allowed_keywords = [
    "data", "gdpr", "privacy", "platform", "processing", "controller", "ai", "artificial intelligence",
    "gatekeeper", "dma", "dsa", "services", "market", "regulation", "consent", "user", "personal data",
    "automated decision", "intermediary", "provider", "access", "sharing", "dataset", "open data", 
    "eprivacy", "security", "cyber", "incident", "compliance", "obligation", "law", "eu", "municipality"
]

if query:
    lower_query = query.lower()
    if not any(keyword in lower_query for keyword in allowed_keywords):
        st.warning("⚠️ This question may not clearly relate to EU digital laws.")

    # Encode and search
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)

    relevant_contexts = [metadata[i]["text"] for i in I[0]]
    references = [metadata[i] for i in I[0]]

    # Compose GPT prompt
    system_prompt = "You are a legal assistant answering questions about EU digital laws. Start with YES or NO if appropriate, then justify your answer using the text provided."
    user_prompt = f"Question: {query}\n\nRelevant legal texts:\n" + "\n\n".join(relevant_contexts)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response.choices[0].message.content
        st.success("✅ GPT response received.")

    except Exception as e:
        st.error(f"⚠️ GPT response failed: {e}")
        answer = "⚠️ AI response failed. Showing top 2 closest legal matches:\n\n"
        for i, ref in enumerate(references[:2], start=1):
            answer += f"{i}. **{ref['ref']}**\n\"{ref['text']}\"\n\n"

    st.subheader("Answer")
    st.write(answer)

    # Only show top 2 citations
    st.markdown("### References")
    for ref in references[:2]:
        st.write(f"**{ref['ref']}**")
        st.markdown(f"> {ref['text']}")
        st.markdown(f"[View full law]({ref['url']})")
        st.markdown("---")
