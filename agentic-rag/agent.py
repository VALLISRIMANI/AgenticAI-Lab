from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import retriever

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a Campus Information Assistant.

Answer the student's question ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context, say:
   "The provided campus documents do not contain this information."
4. Give a concise answer.
5. Give a confidence level: High, Medium, or Low.

Context: {context}

Student Question: {question}

Return your response in this format:

Answer: <answer>

Confidence: <High/Medium/Low> """)

def ask_question(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    response = llm.invoke(
        prompt.format(
            question=question,
            context=context
        )
    )

    return response.content, docs

if __name__ == "__main__":
    print("\n=" * 60)
    print("       CAMPUS INFORMATION RAG AGENT")
    print("=" * 60)

    print("\nExample questions:")
    print("1. What is the minimum attendance required?")
    print("2. Are smart watches allowed in examinations?")
    print("\nType 'exit' to stop.")

    while True:
        question = input("\nStudent: ").strip()

        if question.lower() in ("exit", "quit"):
            print("\nCampus Information Agent stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer, docs = ask_question(question)

            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(answer)

            print("\n" + "=" * 60)
            print("REFERENCE SNIPPETS")
            print("=" * 60)

            seen = set()

            for doc in docs:
                source = doc.metadata.get("source")
                content = doc.page_content
                key = (source, content)

                if key not in seen:
                    print(f"\n[{source}]")
                    print(content)
                    seen.add(key)

        except Exception as e:
            print(f"Error processing question: {e}")