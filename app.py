from flask import Flask, render_template, request

from src.helper import download_hugging_face_embeddings

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

from src.prompt import prompt

import os


# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# -----------------------------
# Load Hugging Face Embeddings
# -----------------------------

embeddings = download_hugging_face_embeddings()


# -----------------------------
# Connect to Pinecone
# -----------------------------

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -----------------------------
# Gemini Model
# -----------------------------

chatModel = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# -----------------------------
# Create RAG Chain
# -----------------------------

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

print("RAG chain created successfully!")


# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def index():
    return render_template("chat.html")


# -----------------------------
# Chat Endpoint
# -----------------------------

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form.get("msg", "")

    if not msg:
        return "Please enter a question."

    print("Question:", msg)

    response = rag_chain.invoke({
        "input": msg
    })

    answer = response["answer"]

    print("Response:", answer)

    return str(answer)


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )