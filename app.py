import os
from functools import lru_cache

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt
from src.pinecone_retriever import DirectPineconeRetriever

load_dotenv()
app = Flask(__name__)
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


@lru_cache(maxsize=1)
def build_chain():
    pinecone_key = os.getenv("PINECONE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    model_name = os.getenv("GOOGLE_MODEL", "gemini-3.8-flash")

    missing = [
        name for name, value in (
            ("PINECONE_API_KEY", pinecone_key),
            ("GOOGLE_API_KEY", google_key),
        ) if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing environment variable(s): " + ", ".join(missing)
            + ". Copy .env.example to .env and add your API keys."
        )

    os.environ["PINECONE_API_KEY"] = pinecone_key
    os.environ["GOOGLE_API_KEY"] = google_key

    embeddings = download_hugging_face_embeddings()
    pinecone = Pinecone(api_key=pinecone_key)
    index = pinecone.Index(index_name)
    retriever = DirectPineconeRetriever(index=index, embeddings=embeddings, k=3)
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_key,
        temperature=0.2,
    )
    answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, answer_chain)
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/get", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    msg = (request.form.get("msg") or payload.get("msg") or "").strip()
    if not msg:
        return "Please enter a question.", 400

    try:
        chain = build_chain()
        response = chain.invoke(
            {"input": msg},
            config={"configurable": {"session_id": request.form.get("session_id", "default")}},
        )
        return str(response.get("answer", "Sorry, I couldn't generate an answer."))
    except Exception:
        app.logger.exception("Chat request failed")
        return (
            "The chatbot could not answer right now. Check that your API keys are valid, "
            "the Pinecone index exists and contains medical documents, and your API quota is available."
        ), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
