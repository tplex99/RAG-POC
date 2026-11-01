import fitz  # PyMuPDF
import os
import base64
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

# --- CONFIGURATION ---
DOC_DIR = "reff-docs"  # Your folder with technical documents
DB_DIR = "./chroma_db"
IMAGE_DIR = "./extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize the Vision Model and Embedding Model
vision_model = ChatOllama(model="qwen2.5vl:7b", temperature=0, base_url="http://127.0.0.1:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_directory(directory_path):
    all_documents = []
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"\n--- Processing Document: {filename} ---")
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 1. Extract Text
                text = page.get_text()
                if text.strip():
                    all_documents.append(
                        Document(page_content=text, metadata={"source": filename, "page": page_num, "type": "text"})
                    )
                    
                # 2. Extract Images/Diagrams
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # --- NEW: Size Filter ---
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Skip images smaller than 200x200 (logos, icons, tiny junk)
                    if width < 200 or height < 200:
                        continue
                    # ------------------------
                    
                    image_bytes = base_image["image"]
                    
                    # Save image temporarily
                    img_filename = os.path.join(IMAGE_DIR, f"{filename}_page_{page_num}_img_{img_index}.{base_image['ext']}")
                    with open(img_filename, "wb") as f:
                        f.write(image_bytes)
                        
                    print(f"Summarizing diagram on page {page_num} of {filename}...")
                    
                    img_b64 = image_to_base64(img_filename)
                    
                    # Prompt the vision model
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this architecture or technical diagram in detail. List the components, text labels, and how they connect."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    )
                    
                    # Generate summary with vision model
                    try:
                        response = vision_model.invoke([message])
                        diagram_summary = response.content
                        
                        all_documents.append(
                            Document(
                                page_content=f"Diagram Description: {diagram_summary}", 
                                metadata={"source": filename, "page": page_num, "type": "diagram", "image_path": img_filename}
                            )
                        )
                    except Exception as e:
                        print(f"Error summarizing image in {filename} page {page_num}: {e}")
                        
    return all_documents

print("Extracting and processing all PDFs in folder...")
docs = process_directory(DOC_DIR)

if docs:
    print(f"\nProcessed {len(docs)} total chunks (text + diagrams). Building Vector DB...")
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("Vector Database populated successfully!")
else:
    print("No PDFs found or no content extracted. Please check the reff-doc folder.")