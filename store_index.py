"""Create/populate the Pinecone index from PDFs in ./data.

Run once after adding API keys to .env: python store_index.py
"""
import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
if not api_key:
    raise SystemExit("PINECONE_API_KEY is missing. Copy .env.example to .env and add your key.")

documents = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader).load()
if not documents:
    raise SystemExit("No PDF files found in ./data. Add your medical PDF and try again.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
dimension = len(embeddings.embed_query("medical chatbot index check"))

pc = Pinecone(api_key=api_key)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
index = pc.Index(index_name)

# Store chunk text in the `text` metadata key, which the app retriever reads.
batch_size = 100
for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]
    vectors = []
    for chunk in batch:
        metadata = {"text": chunk.page_content}
        if chunk.metadata.get("source"):
            metadata["source"] = str(chunk.metadata["source"])
        if chunk.metadata.get("page") is not None:
            metadata["page"] = int(chunk.metadata["page"])
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embeddings.embed_query(chunk.page_content),
            "metadata": metadata,
        })
    index.upsert(vectors=vectors)

print(f"Uploaded {len(chunks)} chunks to Pinecone index '{index_name}'.")
