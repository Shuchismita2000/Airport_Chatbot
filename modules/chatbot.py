import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json

# Load FAISS index and metadata
index_file = "vector_index.faiss"
metadata_file = "metadata.json"

index = faiss.read_index(index_file)
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app
st.title("Changi and Jewel Chatbot")
st.write("Ask me anything about Changi Airport or Jewel Changi Airport!")

# User query input
query = st.text_input("Enter your question:")

def query_faiss(question, top_k=5):
    """
    Search the FAISS index with the user's question and return top results.
    """
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# Display results
if st.button("Submit"):
    if query.strip():
        st.write(f"### Question: {query}")
        results = query_faiss(query)
        if results:
            st.write("### Top Results:")
            for i, result in enumerate(results):
                st.write(f"**{i+1}.** {result['text']} _(Category: {result['category']})_")
        else:
            st.write("Sorry, I couldn't find any relevant information.")
    else:
        st.write("Please enter a question.")

