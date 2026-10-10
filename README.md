# Medical Chatbot (fixed setup)

Flask + LangChain RAG chatbot using a Gemini model and a Pinecone vector index.

## 1. Requirements

Install Python 3.11 (recommended) and create a virtual environment from this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure API keys

Copy `.env.example` to `.env`, then fill in your own valid `PINECONE_API_KEY` and `GOOGLE_API_KEY`.
Do not commit or share `.env`. `GOOGLE_MODEL` defaults to `gemini-2.5-flash`;
change it only to a model available to your Google AI account.

## 3. Create/populate the Pinecone index

The app expects an index named `medical-chatbot` by default. The index must have the
embedding dimension used by `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
If you have not indexed the included `data/Medical_book.pdf`, run:

```bash
python store_index.py
```

This downloads the embedding model on first use and uploads the PDF chunks to Pinecone.
If an existing index has a different dimension, create a new empty index or change
`PINECONE_INDEX_NAME` before indexing.

## 4. Run the app

```bash
python app.py
```

Open http://127.0.0.1:8080. `http://127.0.0.1:8080/health` should return `{"status":"ok"}`.
The first request may take a little longer while the embedding model loads.

## Common problems

- **Missing API key**: make sure `.env` is in this project folder and has both keys.
- **Pinecone index not found / dimension mismatch**: run `python store_index.py` and verify
  `PINECONE_INDEX_NAME` matches the index name in Pinecone.
- **Gemini quota/model error**: check Google AI Studio API key access, billing/quota and
  `GOOGLE_MODEL`.
- **Install errors**: use Python 3.11 and run commands inside the activated virtual environment.

Medical safety: this app gives general information only; it is not a diagnosis or a substitute
for professional medical care. Seek emergency care for potentially life-threatening symptoms.
