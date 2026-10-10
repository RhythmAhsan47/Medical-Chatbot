# Medical Chatbot (fixed setup)

Flask + LangChain RAG chatbot using a Gemini model and a Pinecone vector index.

## 1. Requirements

This project is intended to run in its own virtual environment. If your terminal shows both `(base)` and `(.venv)`, do not trust bare `python`/`pip` commands: Conda may be leaking into imports. Use the project-local interpreter explicitly.

In Git Bash, from this project folder, run this once to rebuild a clean environment and install dependencies:

```bash
./setup_gitbash.sh
```

Then launch the app with:

```bash
./.venv/Scripts/python.exe app.py
```

Or use `./run_gitbash.sh` to install dependencies and launch in one command. On Windows, double-click `run_windows.bat`. The scripts explicitly call `.venv/Scripts/python.exe`, rather than whichever interpreter happens to be first on PATH.

To verify which interpreter is used, run:

```bash
./.venv/Scripts/python.exe -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)"
```

The executable path should be inside this project's `.venv` directory.

## 2. Configure API keys

Copy `.env.example` to `.env`, then fill in your own valid `PINECONE_API_KEY` and `GOOGLE_API_KEY`.
Do not commit or share `.env`. `GOOGLE_MODEL` defaults to `gemini-3.8-flash`;
change it only to a model available to your Google AI account.

## 3. Create/populate the Pinecone index

The app expects an index named `medical-chatbot` by default. The index must have the
embedding dimension used by `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
If you have not indexed the included `data/Medical_book.pdf`, run:

```bash
python store_index.py
```

This downloads the embedding model on first use and uploads the PDF chunks to Pinecone. This project uses the official `pinecone` SDK directly instead of `langchain-pinecone`, whose published releases currently exclude Python 3.14.
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
- **Install errors**: use the explicit `.venv/Scripts/python.exe` interpreter or the included setup script; do not run the app with a Conda/base interpreter. If a dependency still reports no compatible distribution, share the full pip error; Python 3.14 support depends on each dependency having a compatible release.

Medical safety: this app gives general information only; it is not a diagnosis or a substitute
for professional medical care. Seek emergency care for potentially life-threatening symptoms.
