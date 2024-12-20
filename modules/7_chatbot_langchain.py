import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import json

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index and metadata
index_file = "./data/vector_index.faiss"
metadata_file = "./data/metadata.json"

# Load metadata
with open(metadata_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load FAISS vector store
vector_store = FAISS.load_local(index_file, embedding_model)

# Create LangChain Retriever
retriever = vector_store.as_retriever()

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a helpful assistant for Changi and Jewel Changi Airport. Use the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""",
)

# Initialize LangChain QA chain
qa_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
    llm=ChatOpenAI(temperature=0.7)  # Replace with HuggingFaceHub if no OpenAI key
)

# Streamlit UI
st.set_page_config(page_title="Changi & Jewel Chatbot", page_icon="✈️")

st.title("🛫 Changi & Jewel Chatbot 🛬")
st.markdown(
    """
    Welcome to the **Changi and Jewel Chatbot**! Ask any question about Changi Airport or Jewel, and I will provide the best answer based on available data.
    """
)

# User query input
query = st.text_input("💬 **Ask your question:**")

# Handle query submission
if st.button("Submit"):
    if query.strip():
        st.markdown(f"### 🔍 **Your Question:** {query}")
        
        # Retrieve and display results
        result = qa_chain.run(query)
        
        if result:
            st.markdown("### ✅ **Answer:**")
            st.write(result["result"])
            
            # Display source documents
            st.markdown("### 📚 **Sources:**")
            for doc in result["source_documents"]:
                st.write(f"- {doc.page_content} _(Category: {doc.metadata.get('category')})_")
        else:
            st.markdown("🚫 **No relevant information found.**")
    else:
        st.markdown("⚠️ **Please enter a question.**")

