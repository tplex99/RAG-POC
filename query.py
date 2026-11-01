from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from PIL import Image  # Added this import
import os

# --- CONFIGURATION ---
DB_DIR = "./chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

llm = ChatOllama(
    model="llama3.1:8b", 
    temperature=0,      
    base_url="http://127.0.0.1:11434"
)

print("Loading Vector Database...")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
print("Database connected. Ready for queries.\n")

def ask_multimodal_rag(question):
    print(f"Searching for: '{question}'...")
    
    results = vectorstore.similarity_search(question, k=4)
    
    context = ""
    for i, res in enumerate(results):
        source = res.metadata.get("source", "Unknown file")
        page = res.metadata.get("page", "Unknown page")
        doc_type = res.metadata.get("type", "Unknown type")
        
        # --- NEW VISUAL DISPLAY LOGIC ---
        if doc_type == "diagram":
            image_path = res.metadata.get("image_path")
            if image_path and os.path.exists(image_path):
                print(f"\n[!] Opening reference diagram from page {page} of {source}...")
                try:
                    img = Image.open(image_path)
                    img.show() # Opens the image in Windows Photos/Default Viewer
                except Exception as e:
                    print(f"Could not open image: {e}")
        # --------------------------------
        
        context += f"\n--- Source {i+1} ({doc_type} from {source}, page {page}) ---\n"
        context += res.page_content + "\n"
        
    prompt = f"""You are a highly capable technical assistant. Use the following retrieved context (which includes extracted text and vision-model summaries of diagrams) to answer the user's question. 
    If the answer is not contained in the context, explicitly state that you do not know. Do not guess.

    Context: {context}

    Question: {question}
    
    Answer:"""
    
    print("Generating answer (this will hit the GPU)...\n")
    response = llm.invoke(prompt)
    
    return response.content, context

if __name__ == "__main__":
    test_question = "how is the transition from 4 to 5G described in the document?"
    
    answer, retrieved_context = ask_multimodal_rag(test_question)
    
    print("================ FINAL ANSWER ================\n")
    print(answer)
    print("\n==============================================")