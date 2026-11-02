import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# --- SETUP & CACHING ---
#DB_DIR = "./chroma_db"
DB_DIR = "./chroma_db_advanced"

# @st.cache_resource ensures the DB and models only load once, not on every chat message
@st.cache_resource
def load_rag_pipeline():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore, llm

vectorstore, llm = load_rag_pipeline()

# --- UI CONFIGURATION ---
st.title("Multimodal RAG Assistant")
st.markdown("Ask questions about your 5G network whitepapers and technical diagrams.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages and their attached images
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg:
            for img_path, caption in msg["images"]:
                st.image(img_path, caption=caption, use_container_width=True)

# --- CHAT LOGIC ---
if prompt := st.chat_input("Ask a question about the architecture..."):
    
    # 1. Show user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Assistant Response Block
    with st.chat_message("assistant"):
        status_text = st.empty()
        status_text.markdown("🔍 *Searching documents...*")
        
        # Retrieve context
        results = vectorstore.similarity_search(prompt, k=8)
        context = ""
        found_images = []
        print("\n--- RETRIEVED CHUNKS FOR THIS QUERY ---")
        for res in results:
            print(f"Type: {res.metadata.get('type')}, Page: {res.metadata.get('page')}")
        print("---------------------------------------\n")
        # Parse results and display images instantly before text generation finishes
        for i, res in enumerate(results):
            doc_type = res.metadata.get("type", "Unknown")
            source = res.metadata.get("source", "Unknown file")
            page = res.metadata.get("page", "Unknown page")
            
            if doc_type == "diagram":
                img_path = res.metadata.get("image_path")
                if img_path and os.path.exists(img_path):
                    caption = f"Source: {source} (Page {page})"
                    found_images.append((img_path, caption))
                    st.image(img_path, caption=caption, use_container_width=True)
                    
            context += f"\n--- Source {i+1} ({doc_type} from {source}, page {page}) ---\n{res.page_content}\n"
            
        status_text.markdown("🧠 *Reading context and generating answer...*")
        
        # Build prompt and generate answer
        llm_prompt = f"""You are a technical assistant. Use the following context to answer the user.
        Context: {context}
        
        Question: {prompt}
        Answer:"""
        
        response = llm.invoke(llm_prompt)
        final_answer = response.content
        
        # Replace status text with the final answer
        status_text.markdown(final_answer)
        
        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer,
            "images": found_images
        })