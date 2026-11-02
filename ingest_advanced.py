import fitz  # PyMuPDF
import os
import base64
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

# --- CONFIGURATION ---
DOC_DIR = "reff-docs"
DB_DIR = "./chroma_db_advanced"
IMAGE_DIR = "./page_renders"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 1. Initialize Models
vision_model = ChatOllama(model="qwen2.5vl:7b", temperature=0, base_url="http://127.0.0.1:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_high_fidelity(directory_path):
    all_documents = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"\n--- Processing Document: {filename} ---")
            
            doc = fitz.open(pdf_path)
            
            # We want roughly 200 DPI for the vision model (default is 72)
            zoom = 200 / 72  
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # --- 1. Fast Text Extraction ---
                text = page.get_text()
                if text.strip():
                    all_documents.append(
                        Document(page_content=text, metadata={"source": filename, "page": page_num + 1, "type": "text"})
                    )
                
                # --- 2. Full Page Rasterization (No Poppler required) ---
                print(f"Rasterizing page {page_num + 1}...")
                
                # get_pixmap takes a picture of the whole page, flattening all text, vectors, and embedded images
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                img_filename = os.path.join(IMAGE_DIR, f"{filename}_page_{page_num + 1}.png")
                pix.save(img_filename)
                
                print(f"Vision Analysis on page {page_num + 1}...")
                img_b64 = image_to_base64(img_filename)
                
                prompt_text = """
                Does this page contain a technical architecture diagram, network topology, or flowchart?
                
                If NO (e.g., it is just text, a corporate logo, or a photo of a person): 
                Reply ONLY with the exact word "SKIP". Do not write anything else.
                
                If YES: 
                Start your response with the word "DIAGRAM_FOUND" followed by a colon, then provide a detailed technical breakdown:
                - Major components (RAN, Core, etc.)
                - Data flow and connections
                - Visible text labels and protocols
                """
                
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                )
                
                try:
                    response = vision_model.invoke([message])
                    vision_analysis = response.content.strip()
                    
                    # Force strict matching to ignore hallucinations
                    if vision_analysis.startswith("DIAGRAM_FOUND"):
                        print(f" -> Found and described diagram on page {page_num + 1}")
                        
                        # Remove the "DIAGRAM_FOUND:" prefix before saving to the DB
                        clean_description = vision_analysis.replace("DIAGRAM_FOUND:", "").strip()
                        
                        all_documents.append(
                            Document(
                                page_content=f"Diagram Description: {clean_description}", 
                                metadata={"source": filename, "page": page_num + 1, "type": "diagram", "image_path": img_filename}
                            )
                        )
                    else:
                        print(f" -> Skipped page {page_num + 1} (No valid technical diagram)")
                except Exception as e:
                    print(f"Error during vision analysis on page {page_num + 1}: {e}")
                    
    return all_documents

print("Starting Advanced Extraction...")
docs = process_high_fidelity(DOC_DIR)

if docs:
    print(f"\nProcessed {len(docs)} chunks. Building New Vector DB...")
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("Advanced Database populated successfully!")