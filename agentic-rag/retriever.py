from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTORSTORE_DIR = "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

query = "What is the minimum attendance required?"

docs = retriever.invoke(query)

print("\n" + "=" * 60)
print("RETRIEVAL RESULTS")
print("=" * 60)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print("Source:", doc.metadata.get("source"))
    print("Content:")
    print(doc.page_content)