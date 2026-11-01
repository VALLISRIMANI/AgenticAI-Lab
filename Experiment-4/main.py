import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

class AgentState(TypedDict):
    question: str
    response: str
    has_error: bool
    attempts: int

def generate_code(state: AgentState):
    print("\n[Agent] Generating code...")
    prompt = f"""
You are a Python coding assistant.
Answer the following programming question:
{state["question"]}

Provide a simple Python solution.
Include:
1. A short explanation
2. Python code
Do not intentionally introduce errors.
"""
    result = llm.invoke(prompt)
    return {"response": result.content, "attempts": state["attempts"] + 1}

def check_errors(state: AgentState):
    print("[Agent] Checking generated response...")
    error_words = ["Traceback", "Error:", "SyntaxError", "NameError", "TypeError", "IndentationError"]
    has_error = any(word.lower() in state["response"].lower() for word in error_words)
    print("[Agent] Error detected." if has_error else "[Agent] No basic error detected.")
    return {"has_error": has_error}

def correct_code(state: AgentState):
    print("[Agent] Attempting self-correction...")
    prompt = f"""
You are a Python code reviewer.
Original question: {state["question"]}
Previous answer: {state["response"]}

Review the answer and correct any errors.
Return:
1. A short explanation
2. Correct Python code
Make the solution simple and beginner-friendly.
"""
    result = llm.invoke(prompt)
    return {"response": result.content, "attempts": state["attempts"] + 1}

def decide_next_step(state: AgentState):
    return "correct" if state["has_error"] else "finish"

workflow = StateGraph(AgentState)
workflow.add_node("generate", generate_code)
workflow.add_node("check", check_errors)
workflow.add_node("correct", correct_code)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check")
workflow.add_conditional_edges("check", decide_next_step, {"correct": "correct", "finish": END})
workflow.add_edge("correct", END)
app = workflow.compile()

def main():
    print("\n" + "-" * 50)
    print("        SELF-CORRECTION CODING AGENT")
    print("-" * 50)
    print("\nExamples:")
    print("- Reverse a Python list")
    print("- Calculate factorial of a number")
    print("- Read a text file in Python")
    print("- Find the largest number in a list")
    print("\nType 'exit' to stop.")

    while True:
        question = input("\nEnter your programming question: ").strip()

        if question.lower() == "exit":
            print("Agent stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            initial_state = {"question": question, "response": "", "has_error": False, "attempts": 0}
            final_state = app.invoke(initial_state)

            print("\n" + "-" * 50)
            print("                  FINAL ANSWER")
            print("-" * 50)
            print(final_state["response"])
            print("\nAttempts:", final_state["attempts"])

        except Exception as e:
            print("\nError:", e)

if __name__ == "__main__":
    main()