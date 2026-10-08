from flask import Flask, render_template, request

from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt

import os


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)


# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =========================================================
# Load Hugging Face Embeddings
# =========================================================

embeddings = download_hugging_face_embeddings()


# =========================================================
# Connect to Existing Pinecone Index
# =========================================================

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


# =========================================================
# Gemini Model
# =========================================================

chatModel = ChatGoogleGenerativeAI(
     model="gemini-3.5-flash-lite"
)


# =========================================================
# Create RAG Chain
# =========================================================

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# =========================================================
# Conversation Memory
# =========================================================

store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


print("RAG chain created successfully!")


# =========================================================
# Home Page
# =========================================================

@app.route("/")
def index():
    return render_template("chat.html")


# =========================================================
# Chat Endpoint
# =========================================================

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question."

    print("Question:", msg)

    try:

        response = rag_chain_with_history.invoke(
            {
                "input": msg
            },
            config={
                "configurable": {
                    "session_id": "default"
                }
            }
        )

        answer = response["answer"]

        print("Response:", answer)

        return str(answer)

    except Exception as e:

        print("ERROR:", e)

        error_message = str(e)

        # Gemini quota error
        if "429" in error_message or "ResourceExhausted" in error_message:

            return (
                "The AI service has reached its current Gemini API quota. "
                "Please try again later or use another available Gemini model."
            )

        # Other errors
        return (
            "Sorry, something went wrong while generating the answer. "
            "Please try again."
        )


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )