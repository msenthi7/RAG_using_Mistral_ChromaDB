import os
from flask import Flask, render_template, request

# LangChain + Vector DB
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat memory (new API)
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ----------------- Config -----------------
PERSIST_DIR = "./chroma_db"           # Path where your Chroma DB lives
COLLECTION  = "medical-chatbot"       # Chroma collection name
LLM_MODEL   = "mistral"               # Ollama model (e.g., mistral, llama3.1:8b)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval score threshold (lower = closer; tune 0.25–0.40)
RELEVANCE_THRESHOLD = 0.30

app = Flask(__name__)

# ----------------- Intent routing helpers -----------------
GREETINGS = {"hi","hello","hey","yo","hola","namaste","sup","hi!","hello!","hey!"}
SMALLTALK_STARTS = ("how are you", "what's up", "how’s it going", "good morning", "good evening")
RESET_TRIGGERS = {"reset","end","stop","clear","goodbye","bye","quit"}
NEGATIVE_FEEDBACK = {
    "not what i expected","this is not what i expected",
    "that’s wrong","that is wrong","bad answer","not helpful"
}

def is_smalltalk(text: str) -> bool:
    t = text.strip().lower()
    return (t in GREETINGS) or any(t.startswith(p) for p in SMALLTALK_STARTS) or len(t) <= 3

def is_reset(text: str) -> bool:
    return text.strip().lower() in RESET_TRIGGERS

def is_negative(text: str) -> bool:
    return text.strip().lower() in NEGATIVE_FEEDBACK

# ----------------- Vector store -----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
docsearch = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# Use MMR to reduce redundant/noisy chunks
retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

# ----------------- LLM -----------------
chatModel = ChatOllama(model=LLM_MODEL, temperature=0.2)

# ----------------- Conversation history store -----------------
# session_id -> ChatMessageHistory
store = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ----------------- Prompts & Chains -----------------
# Stronger prompt: HTML fragments + structured output + “elaborate” support
system_prompt = (
    "You are a Medical assistant for question-answering tasks.\n"
    "- Use the provided context ONLY when the user's question is medical or document-related.\n"
    "- If the answer is not present in the context, say you don't know.\n"
    "- Return an HTML fragment (no <html> or <body>). Use <p>, <ul><li>, <ol><li>, <strong>.\n"
    "- Default to a short <p>. When listing items, also add concise bullet points.\n"
    "- If the user asks to 'elaborate' or 'explain more', provide a longer answer with clear bullets and brief sub-points.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("history"),   # memory goes here
    ("human", "{input}")
])

# RAG chain: retrieved docs -> stuffed into prompt -> LLM
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Wrap the RAG chain with message history (stateful RAG)
rag_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# (Optional) free-chat chain without retrieval, also stateful, returns HTML too
free_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Reply in an HTML fragment using <p> and, when helpful, <ul><li>. Be brief unless asked to elaborate."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
free_chain = free_prompt | chatModel
chat_with_history = RunnableWithMessageHistory(
    free_chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ----------------- Routes -----------------
@app.route("/")
def index():
    # Make sure templates/chat.html exists
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    session_id = request.form.get("session_id", "default")
    t = msg.strip()

    # 1) Reset / end conversation
    if is_reset(t):
        store.pop(session_id, None)
        return "Conversation ended. Start a new question anytime."

    # 2) Greetings / small talk (no RAG)
    if is_smalltalk(t):
        result = chat_with_history.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}},
        )
        return str(result.content)

    # 3) Acknowledge negative feedback
    if is_negative(t):
        return "Thanks for the feedback. Tell me what you expected and I’ll try again, or say <strong>reset</strong> to start fresh."

    # 4) Relevance gate: only use RAG if we actually found relevant docs
    # (For Chroma cosine distance: LOWER score = more similar)
    docs_scores = docsearch.similarity_search_with_score(msg, k=6)
    relevant_docs = [d for (d, s) in docs_scores if s is not None and s < RELEVANCE_THRESHOLD]

    if not relevant_docs:
        # Out-of-domain → conversational fallback (still HTML)
        fallback = chat_with_history.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}},
        )
        return str(fallback.content)

    # 5) On-topic: stateful RAG
    result = rag_with_history.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}},
    )
    # retrieval chain usually returns {"answer": "..."}
    return result["answer"] if isinstance(result, dict) and "answer" in result else str(result)

# Optional free-chat endpoint (if you want to call it directly)
@app.route("/chat", methods=["POST"])
def free_chat():
    msg = request.form["msg"]
    session_id = request.form.get("session_id", "default")
    result = chat_with_history.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}},
    )
    return str(result.content)

# ----------------- Main -----------------
if __name__ == "__main__":
    # Ensure Ollama is running: `ollama serve`
    # Ensure your Chroma DB exists in ./chroma_db
    app.run(host="0.0.0.0", port=8080, debug=True)
