import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in the .env file")

PDF_PATH = r"C:\Users\manig\OneDrive\Desktop\23A81A0522-AgenticAI\Experiment-3\VII SEMESTER COURSE STRUCTURE AND SYLLABUS_CSE & CST_V23 Regulation.pdf"

def create_retrieval_agent():
    print("\nLoading document...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print("\nSplitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\nCreating Chroma vector database...")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="campus_syllabus")
    print("Vector database created.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    prompt = PromptTemplate.from_template("""
You are a campus student assistant.
Answer the student's question using ONLY the information provided in the context.
If the answer is not present in the context, say:
"I could not find this information in the provided document."
Do not make up information.

Context:
{context}

Student Question:
{question}

Answer:
""")
    return retriever, llm, prompt

def ask_question(question, retriever, llm, prompt):
    print("\n" + "-" * 50)
    print("QUESTION:")
    print("-" * 50)
    print(question)

    retrieved_docs = retriever.invoke(question)

    print("\nRETRIEVED CHUNKS:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print("\n" + "-" * 50)
        print(f"Chunk {i}")
        print("-" * 50)
        print(doc.page_content[:500])

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    print("\n" + "-" * 50)
    print("ANSWER:")
    print("-" * 50)
    print(response.content)

def main():
    print("\n" + "-" * 50)
    print("       CAMPUS RETRIEVAL AGENT")
    print("-" * 50)
    print("Ask questions about your syllabus.")
    print("Type 'exit' to stop.")

    retriever, llm, prompt = create_retrieval_agent()

    while True:
        question = input("\nAsk your question: ").strip()

        if question.lower() == "exit":
            print("Agent stopped.")
            break

        if not question:
            continue

        try:
            ask_question(question, retriever, llm, prompt)
        except Exception as e:
            print("\nError:", e)

if __name__ == "__main__":
    main()