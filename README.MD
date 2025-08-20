# Medical Chatbot (Local RAG + Ollama + Chroma)

A lightweight, fully local medical Q\&A chatbot.

It uses **Ollama** to run an LLM on your machine, **Chroma** as a local vector DB, and **HuggingFace** sentence embeddings for fast semantic search over your PDFs.

### Why this setup?

  * **Local-first**: No cloud keys needed. Your PDFs never leave your machine.
  * **Chroma**: Simple, embedded vector DB with disk persistence—ideal for local projects.
  * **HuggingFace embeddings**: Fast CPU model (**all-MiniLM-L6-v2**, 384-dim) keeps latency low.
  * **Ollama**: Runs models like **mistral** or **llama3** locally with one command.
  * **Stateful RAG**: We use LangChain’s `RunnableWithMessageHistory` to remember conversation context (follow-ups work).
  * **Intent routing**: Greets naturally, handles small-talk, and gracefully falls back when the query is off-topic.
  * **Readable output**: The LLM is prompted to return **HTML fragments** using paragraphs and bullet points for clear UI rendering.

-----

### Project Structure

```bash
medical_chatbot/
├─ app.py                 # Flask app (stateful RAG + intent router)
├─ store_index.py         # Build/update local Chroma index from PDFs
├─ requirements.txt
├─ Dockerfile
├─ src/
│  ├─ __init__.py
│  └─ helper.py           # PDF loader, splitter, embeddings helper (if used)
├─ data/                  # Put your PDFs here
├─ chroma_db/             # Created by store_index.py (persisted index)
├─ templates/
│  └─ chat.html           # Your chat UI (uses your CSS)
└─ static/
   └─ style.css           # Dark chat theme
```

-----

### Prerequisites

  * **Python** 3.11 (recommended for compatibility)
  * `pip` (or conda)
  * **Ollama** installed: `https://ollama.com/`

After install, pull a model (e.g., `mistral`):

```bash
ollama pull mistral
```

-----

### 1\) Install dependencies (local)

```bash
# From project root
pip install -r requirements.txt
```

(If you use conda, activate your env first.)

-----

### 2\) Build the semantic index (once per PDF update)

Put your PDFs in `./data/`, then run:

```bash
python store_index.py
```

You should see logs like:

```
📁 Loading PDFs from: data/
✂️ Prepared <N> chunks.
🗄️ Writing to Chroma at: ./chroma_db (collection: medical-chatbot)
✅ Chroma index created/updated and persisted.
```

This creates a **local** Chroma collection at `./chroma_db` that `app.py` connects to.

-----

### 3\) Start Ollama (LLM backend)

Open a terminal and run:

```bash
ollama serve
```

Make sure the model is available:

```bash
ollama pull mistral
```

You can switch models in `app.py` via `LLM_MODEL`.

-----

### 4\) Run the web app (local)

In another terminal:

```bash
python app.py
```

Open `http://127.0.0.1:8080/` in your browser.

#### How the UI behaves

  * **Greetings / small‑talk** (“hi”, “how are you?”) → friendly reply (no medical context).
  * **On-topic medical queries** → **stateful RAG** over your PDFs (follow‑ups like “elaborate” work).
  * **Out‑of‑domain queries** → conversational fallback (won’t hallucinate from random medical chunks).
  * **End/reset** → send `reset`, `end`, `stop`, or `clear`. This clears just the current session’s memory.

The frontend can pass a `session_id` with each request so multiple tabs/users have independent memory.

**Example jQuery AJAX (already in your HTML):**

```javascript
$.ajax({
  data: {
    msg: rawText,
    session_id: 'default' // or generate a UUID per tab
  },
  type: "POST",
  url: "/get"
});
```

-----

### 5\) Run with Docker (optional)

**Build:**

```bash
docker build -t medical-chatbot .
```

**Run:**

```bash
docker run -p 8080:8080 medical-chatbot
```

Open `http://127.0.0.1:8080/`

**Note**: This container runs only the Flask app. **Ollama must be running on the host** (`ollama serve`) so the app can contact it.

-----

### Config knobs (quick)

  * **Model**: `LLM_MODEL` in `app.py` (e.g., `"mistral"`, `"llama3.1:8b"`).
  * **Retriever**: MMR used to reduce redundancy:
    ```python
    retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )
    ```
  * **Relevance gate**: `RELEVANCE_THRESHOLD = 0.30` in `app.py`. Lower = stricter; typical range `0.25–0.40`. Prevents pulling random context.
  * **HTML output**: System prompt tells the model to return **HTML fragments** (`<p>`, `<ul><li>`), improving readability in your chat UI.
  * **Session memory**: `RunnableWithMessageHistory` + `MessagesPlaceholder("history")`. Each request should include a `session_id`.

-----

### Troubleshooting

  * **No answers / random text for “hi”**
    Make sure the intent router is present (we bypass RAG for small‑talk).
  * **No documents retrieved**
    Verify `./data/` has PDFs, `store_index.py` prepared chunks, and both `app.py` and `store_index.py` point to the same:
    ```python
    PERSIST_DIR = "./chroma_db"
    COLLECTION = "medical-chatbot"
    ```
  * **Model/LLM errors**
    Ensure `ollama serve` is running and the model is pulled: `ollama pull mistral`.
  * **Install errors (Docker or local)**
    Stick to **Python 3.11** for best compatibility with `chromadb`, `langchain-*`, and `sentence-transformers`.

-----

### Reasoning: why we made these choices

  * **Local privacy**: PDFs never leave your machine (Ollama + Chroma are local).
  * **Performance & simplicity**: MiniLM embeddings are fast on CPU; Chroma persists to disk automatically.
  * **Good UX**: Intent routing avoids awkward “Hi → random medical summary”. Relevance gating reduces off-topic hallucinations.
  * **Maintainability**: Uses the latest LangChain interfaces (`langchain_chroma`, `langchain_huggingface`, `RunnableWithMessageHistory`) to avoid deprecations and ease upgrades.
  * **Readable answers**: HTML fragments (paragraphs & bullets) render nicely in your chat UI and expand automatically on “elaborate”.

-----

### One-liners

**Rebuild index then run app:**

```bash
python store_index.py && python app.py
```

**Quick curl test:**

```bash
curl -X POST -d "msg=What does the context say about acromegaly?" -d "session_id=default" http://127.0.0.1:8080/get
```

**Reset conversation:**

```bash
curl -X POST -d "msg=reset" -d "session_id=default" http://127.0.0.1:8080/get
```