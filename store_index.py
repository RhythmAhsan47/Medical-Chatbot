"""Create/populate the Pinecone index from PDFs in ./data.

Run once after adding API keys to .env: python store_index.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
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
        spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                            region=os.getenv("PINECONE_REGION", "us-east-1")),
    )

PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
)
print(f"Uploaded {len(chunks)} chunks to Pinecone index '{index_name}'.")
