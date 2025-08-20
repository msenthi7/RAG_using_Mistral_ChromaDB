# store_index.py
# Build (or update) a local Chroma DB from PDFs in ./data

import os
from pathlib import Path

from src.helper import (
    load_pdf_files,              # loads PDFs → Documents
    filter_to_minimal_docs,     # keeps {page_content, source}
    text_split,                 # splits into chunks
    download_embeddings,  # returns HF embeddings
)

from langchain_community.vectorstores import Chroma

# --------- Config ---------
DATA_DIR     = "data/"                 # your PDFs go here
PERSIST_DIR  = "./chroma_db"           # Chroma storage folder
COLLECTION   = "medical-chatbot"       # Collection name

def main():
    # 1) Load → minimalize → split
    print(f"📁 Loading PDFs from: {DATA_DIR}")
    extracted = load_pdf_files(data=DATA_DIR)
    minimal   = filter_to_minimal_docs(extracted)
    chunks    = text_split(minimal)
    print(f"✂️ Prepared {len(chunks)} chunks.")

    # 2) Embeddings
    embeddings = download_embeddings()

    # 3) Create (or append) local Chroma collection and persist
    print(f"🗄️ Writing to Chroma at: {PERSIST_DIR} (collection: {COLLECTION})")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )
    vectordb.persist()
    print("✅ Chroma index created/updated and persisted.")

if __name__ == "__main__":
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    main()
