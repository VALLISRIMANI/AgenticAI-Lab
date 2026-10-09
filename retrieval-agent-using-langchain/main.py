# Retrieval Agent using LangChain

import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate

# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# =========================================================
# 2. Load PDF
# =========================================================

PDF_PATH = r"C:\Users\manig\OneDrive\Desktop\AgenticAI\retrieval-agent-using-langchain\VII SEMESTER COURSE STRUCTURE AND SYLLABUS_CSE & CST_V23 Regulation.pdf"

print("\nLoading document...")

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# =========================================================
# 3. Split document into chunks
# =========================================================

print("\nSplitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# =========================================================
# 4. Create embeddings
# =========================================================

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# 5. Store embeddings in Chroma
# =========================================================

print("\nCreating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="campus_syllabus"
)

print("Vector database created.")

# =========================================================
# 6. Create retriever
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# =========================================================
# 7. Create Groq LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# =========================================================
# 8. Create PromptTemplate
# =========================================================

prompt = PromptTemplate.from_template("""
You are a campus student assistant.

Answer the student's question using ONLY the
information provided in the context.

If the answer is not present in the context,
say:

"I could not find this information in the provided document."

Do not make up information.

Context:
{context}

Student Question:
{question}

Answer:
""")

# =========================================================
# 9. Ask questions
# =========================================================

def ask_question(question):

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    print("\nRETRIEVED CHUNKS:")
    
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:500])

    # Combine retrieved text
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Create prompt
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Generate answer
    response = llm.invoke(final_prompt)

    print("\nANSWER:")
    print(response.content)

# =========================================================
# 10. Interactive loop
# =========================================================

print("\n" + "=" * 60)
print("       CAMPUS RETRIEVAL AGENT")
print("=" * 60)

print("Ask questions about your syllabus.")
print("Type 'exit' to stop.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question.strip():
        continue

    try:
        ask_question(question)

    except Exception as e:
        print("\nError:", e)