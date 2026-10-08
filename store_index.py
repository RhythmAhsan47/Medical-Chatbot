import os
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("Environment variables loaded successfully")


import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


extracted_data = load_pdf_files("data")

print("Number of documents:", len(extracted_data))


from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


minimal_docs = filter_to_minimal_docs(extracted_data)

print("Minimal documents:", len(minimal_docs))


from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    texts_chunk = text_splitter.split_documents(minimal_docs)

    return texts_chunk


texts_chunk = text_split(minimal_docs)

print("Number of chunks:", len(texts_chunk))


from langchain_community.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    return embeddings


embedding = download_embeddings()

vector = embedding.embed_query("Hello world")

print("Embedding dimension:", len(vector))


from pinecone import Pinecone, ServerlessSpec

# Make sure your API key is available
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not loaded")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=len(vector),
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

print("Pinecone index ready")


from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)

print("Documents uploaded to Pinecone successfully")


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("Retriever ready")


from langchain_google_genai import ChatGoogleGenerativeAI

chatModel = ChatGoogleGenerativeAI(
     model="gemini-3.5-flash-lite"
)

response = chatModel.invoke("Hello, are you working?")

print(response.content)


from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
You are a helpful medical assistant.

You will be provided with context from medical documents.

Answer the user's question based only on the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided medical documents."

Do not make up medical information.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

print("Prompt created successfully")


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

print("RAG chain created successfully")


response = rag_chain.invoke({
    "input": "What is the treatment for diabetes?"
})

print(response["answer"])