
# Local Multimodal RAG POC (RTX 4070 / 12GB VRAM)



This project is a Proof of Concept (POC) for a local Multimodal Retrieval-Augmented Generation (RAG) pipeline. It is specifically designed to run on consumer hardware with 12GB of VRAM (like an NVIDIA RTX 4070) without crashing due to Out-Of-Memory (OOM) errors.



It accomplishes this by using **Ollama** to dynamically load and unload highly optimized quantized models, ensuring only one model occupies the GPU at any given time.



## 🚀 Features



* **100% Local and Private:** No data leaves your machine. No API keys required.

* **Diagram Understanding:** Extracts diagrams from PDFs and uses a Vision Language Model (VLM) to generate searchable text summaries of the visual data.

* **Junk Image Filtering:** Automatically ignores small corporate logos and UI icons to save processing time.

* **Visual Retrieval:** When a diagram summary is used to answer a query, the original image pops up on your screen for visual reference.

* **Vector Storage:** Uses ChromaDB for fast, persistent local vector storage.



## 🛠️ Hardware & Software Requirements



* **GPU:** NVIDIA GPU with at least 12GB VRAM (Tested on RTX 4070)

* **OS:** Windows / Linux

* **Software:** [Ollama](https://ollama.com/) must be installed and running in the background.

* **Python:** 3.8 or higher.



## 📦 Installation & Setup



**1. Install Python Dependencies**

Open your terminal and run:

```bash

pip install -U langchain-ollama pymupdf chromadb langchain langchain-chroma pillow

```



**2. Pull the Required Local Models**

This pipeline uses three specific models to balance capability and memory usage. Run these commands in your terminal to download them via Ollama:

```bash

# 1. Vision Model (For reading and summarizing technical diagrams ~5GB VRAM)

ollama pull qwen2.5vl:7b



# 2. Embedding Model (For converting text/summaries to vectors <1GB VRAM)

ollama pull nomic-embed-text



# 3. Final Text Model (For answering the user's queries ~5.5GB VRAM)

ollama pull llama3.1:8b

```



## 📂 Project Structure



* `ingest_folder.py` - The ingestion script. Scans PDFs, extracts text and diagrams, generates summaries, and populates the database.

* `query.py` - The query script. Takes a user question, retrieves relevant text/diagrams, opens the reference images, and generates an answer.

* `reff-doc/` - **(Create this folder)** Place all your target PDF documents in here.

* `extracted_images/` - Auto-generated. Stores the raw `.png` files of diagrams extracted from the PDFs.

* `chroma_db/` - Auto-generated. The local vector database.



## 🏃 Usage



### Phase 1: Ingestion



1. Ensure the Ollama server is running (check your system tray or run `ollama serve`).

2. Place your technical PDFs into the `reff-doc` directory.

3. Run the ingestion script:

```bash

python ingest_folder.py

```

*Note: The GPU will run at 100% during this phase as it processes the images. Small images (<200x200 pixels) are automatically skipped.*



### Phase 2: Querying



1. Open `query.py` in a text editor.

2. Scroll to the bottom and modify the `test_question` variable to ask a specific question about your documents.

3. Run the query script:

```bash

python query.py

```

*If a diagram is relevant to the answer, your default image viewer will pop open to display it while the LLM generates the text response in your terminal.*



## ⚠️ Troubleshooting



* **`WinError 10049` or Connection Errors:** Ensure Ollama is running. The scripts are hardcoded to look for `http://127.0.0.1:11434` to bypass Windows localhost routing issues.

* **OOM / Crashing:** Ensure you are not running other heavy VRAM applications (like heavy video games) while running the ingestion or query scripts.