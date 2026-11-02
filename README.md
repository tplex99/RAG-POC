# Multimodal RAG POC (Local Environment)



A Proof of Concept (POC) for a fully local Multimodal Retrieval-Augmented Generation (RAG) pipeline. This system is designed to ingest technical whitepapers (PDFs), extract and understand both raw text and complex architectural diagrams, and answer user queries using a conversational interface.



This architecture is optimized to run entirely on a single consumer GPU with 12GB VRAM (e.g., RTX 4070) by dynamically loading and unloading quantized models via Ollama.



## 🧠 Model Architecture



To respect the 12GB VRAM limit without encountering Out-Of-Memory (OOM) errors, the workload is distributed across three specialized, 4-bit quantized models:



1.  **Vision / Ingestion (`qwen2.5vl:7b`)**: Scans rasterized PDF pages, identifies technical diagrams, and generates highly detailed text summaries (components, flows, protocols). *Uses ~5.5GB VRAM.*

2.  **Embedding (`nomic-embed-text`)**: Converts PDF text and vision summaries into vectors for ChromaDB. *Uses < 1GB VRAM.*

3.  **Reasoning / Generation (`llama3.1:8b`)**: The final chat model that synthesizes retrieved context to answer user queries. *Uses ~5.5GB VRAM.*



## ⚙️ Setup & Installation



### 1. Prerequisites

Ensure you have Python 3.10+ installed. Install the required Python packages in your virtual environment:

```bash

pip install -U langchain-ollama pymupdf chromadb langchain langchain-chroma pillow streamlit

```



### 2. Install & Start Ollama

Download Ollama from [ollama.com](https://ollama.com). Ensure the background server is running (check your system tray), then pull the required models:

```bash

ollama pull qwen2.5vl:7b

ollama pull nomic-embed-text

ollama pull llama3.1:8b

```



## 🚀 Execution Pipeline



### Phase 1: Ingestion & Database Building

Place your target PDFs inside a folder named `reff-doc/` in the root directory. 



We utilize an **Advanced Rasterization Strategy** (`ingest_advanced_pymupdf.py`). Instead of hunting for embedded image files (which fails on vector-based diagrams from Word/Visio), this script takes a high-res screenshot of *every* page using PyMuPDF and passes it to the Vision model to evaluate.



Run the ingestion script:

```bash

python ingest_advanced_pymupdf.py

```

*Note: This process is computationally heavy as the GPU analyzes every page. Extracted page renders will be saved to `./page_renders` and data will be vectorized into `./chroma_db_advanced`.*



### Phase 2: Interactive Web UI (Streamlit)

Once the database is populated, launch the chat interface:

```bash

streamlit run app.py

```

This will start a local web server (usually at `http://localhost:8501`). The UI will maintain chat history and render the actual PDF diagrams inline whenever the LLM references them.



## ⚠️ Current Limitations & Required Tuning



The Advanced Ingestion pipeline is currently a **Work in Progress** and requires further tuning for a production-ready state:



*   **Vision False Positives:** While strict prompting (`SKIP` vs `DIAGRAM_FOUND`) is implemented, Vision Language Models (VLMs) can still occasionally hallucinate, mistaking stock photos or complex text layouts for architectural diagrams. 

*   **Prompt Engineering:** The prompt used in the ingestion phase may need to be tailored to the specific style of your company's network diagrams to ensure it captures the right acronyms and data flows.

*   **Vector Ranking:** Because PyMuPDF extracts both raw text and the vision summaries, the embedding model sometimes ranks raw text higher than the diagram summary. To counter this, the retrieval `k` value in `app.py` has been increased, but threshold tuning is still required to guarantee the diagram is always pulled into context.