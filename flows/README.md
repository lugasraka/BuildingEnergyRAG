# Langflow Flows Directory

This directory contains exported Langflow flows for the EcoMindAI RAG system.

## Quick Start

### 1. Install Langflow
```bash
pip install langflow
```

### 2. Launch the Visual Editor
```bash
langflow run
```
This opens the Langflow UI at `http://localhost:7860` (or another port if busy).

### 3. Build Your Energy RAG Flow

Add these components and connect them:

```
[Chat Input]
     ↓
[File Loader] → path: knowledge_base/sustainability_manual.txt
     ↓
[Recursive Character Text Splitter] → chunk_size: 500, overlap: 50
     ↓
[HuggingFace Embeddings] → model: all-MiniLM-L6-v2
     ↓
[FAISS Vector Store]
     ↓
[Ollama LLM] → model: llama3.2, temperature: 0.7
     ↓
[RetrievalQA Chain] → chain_type: stuff
     ↓
[Chat Output]
```

### 4. Export Your Flow
1. Click the **Export** button (download icon)
2. Save the file as `energy_rag_flow.json` in this directory

### 5. Enable Langflow in the App
Edit `app.py` and change:
```python
USE_LANGFLOW = True  # Set to True to use Langflow
```

## File Structure
```
flows/
├── README.md                 # This file
└── energy_rag_flow.json      # Your exported Langflow flow (create this)
```

## Troubleshooting

### "Langflow not installed"
```bash
pip install langflow
```

### "Flow file not found"
Make sure you exported your flow to `flows/energy_rag_flow.json`

### "Ollama connection error"
Ensure Ollama is running:
```bash
ollama serve
```

### Port conflict with Gradio
If both apps try to use port 7860, run Langflow on a different port:
```bash
langflow run --port 7861
```
