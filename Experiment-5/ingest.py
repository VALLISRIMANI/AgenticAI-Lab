from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCUMENT_DIR = "documents"
VECTORSTORE_DIR = "vectorstore"

def load_documents():
    documents = []

    for file in Path(DOCUMENT_DIR).glob("*.txt"):
        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file.name

        documents.extend(docs)

    return documents

def main():
    documents = load_documents()

    print("Documents loaded:", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )

    print("Vector store created successfully.")

if __name__ == "__main__":
    main()