import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json

# Load FAISS index and metadata
index_file = "app\vector_index.faiss"
metadata_file = "app\metadata.json"

index = faiss.read_index(index_file)
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app - Add a sidebar and header
st.set_page_config(
    page_title="Changi & Jewel Chatbot",
    page_icon="✈️",
    layout="centered",
)

# Sidebar with additional information
st.sidebar.title("About This Chatbot")
st.sidebar.write(
    """
    This chatbot is designed to help you find information about **Changi Airport** 
    and **Jewel Changi Airport**. It uses state-of-the-art AI to search a semantic 
    database and provide relevant answers to your questions.
    """
)
st.sidebar.write("### Features:")
st.sidebar.markdown(
    """
    - Discover attractions, dining, and shopping options.
    - Find information about transportation and facilities.
    - Get personalized recommendations based on your queries.
    """
)
st.sidebar.write("### How It Works:")
st.sidebar.markdown(
    """
    1. Enter your question in the text box below.
    2. Click **Submit** to get top results matching your query.
    3. Explore categories for additional insights.
    """
)

# Main Page Design
st.title("🛫 Changi & Jewel Chatbot 🛬")
st.markdown(
    """
    Welcome to the **Changi and Jewel Chatbot**! Start by selecting a question 
    from the list below or enter your own query in the input box.
    """
)


# Example questions
st.markdown("### 💡 **Need inspiration? Try these questions:**")
example_questions = [
    "What are the top attractions at Jewel Changi Airport?",
    "What are the best dining options at Jewel Changi Airport?",
    "Which luxury brands are available at Changi Airport?",
    "How do I get from Terminal 1 to Jewel?",
    "Are there any lounges available at Changi Airport?",
]

# Radio button for prefilled questions
selected_question = st.radio(
    "Select a question:", 
    options=[""] + example_questions,
    index=0
)

# Query input box (autofills when question is selected from the radio button)
query = st.text_input(
    "💬 **Ask me anything about Changi Airport or Jewel Changi Airport!**", 
    value=selected_question if selected_question else ""
)

# Query Function
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
        st.markdown(f"### 🔍 **Your Question:** {query}")
        results = query_faiss(query)
        if results:
            st.markdown("### ✅ **Top Results:**")
            for i, result in enumerate(results):
                st.write(f"**{i+1}.** {result['text']} _(Category: {result['category']})_")
        else:
            st.markdown("🚫 **Sorry, I couldn't find any relevant information.**")
    else:
        st.markdown("⚠️ **Please enter a question to proceed.**")

# Footer
st.markdown(
    """
    ---
    **Note:** This chatbot is powered by advanced AI embeddings and FAISS for semantic search. 
    Feel free to explore and discover the wonders of Changi and Jewel Changi Airport!
    """
)
