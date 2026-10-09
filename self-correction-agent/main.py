# Self Correction Agent

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# --------------------------------------------------
# Initialize Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# --------------------------------------------------
# Define Agent State
# --------------------------------------------------

class AgentState(TypedDict):
    question: str
    response: str
    has_error: bool
    attempts: int

# --------------------------------------------------
# Node 1: Generate Code
# --------------------------------------------------

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

    return {
        "response": result.content,
        "attempts": state["attempts"] + 1
    }

# --------------------------------------------------
# Node 2: Check for Basic Errors
# --------------------------------------------------

def check_errors(state: AgentState):

    print("[Agent] Checking generated response...")

    response = state["response"]

    error_words = [
        "Traceback",
        "Error:",
        "SyntaxError",
        "NameError",
        "TypeError",
        "IndentationError"
    ]

    has_error = any(
        word.lower() in response.lower()
        for word in error_words
    )

    if has_error:
        print("[Agent] Error detected.")
    else:
        print("[Agent] No basic error detected.")

    return {
        "has_error": has_error
    }

# --------------------------------------------------
# Node 3: Correct Response
# --------------------------------------------------

def correct_code(state: AgentState):

    print("[Agent] Attempting self-correction...")

    prompt = f"""
You are a Python code reviewer.

The original programming question was:

{state["question"]}

The previous answer was:

{state["response"]}

Review the answer carefully.

If there is an error, correct it.

Return the improved answer with:
1. A short explanation
2. Correct Python code

Make the solution simple and beginner-friendly.
"""

    result = llm.invoke(prompt)

    return {
        "response": result.content,
        "attempts": state["attempts"] + 1
    }

# --------------------------------------------------
# Conditional Routing
# --------------------------------------------------

def decide_next_step(state: AgentState):

    if state["has_error"]:
        return "correct"

    return "finish"

# --------------------------------------------------
# Create LangGraph
# --------------------------------------------------

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate", generate_code)
workflow.add_node("check", check_errors)
workflow.add_node("correct", correct_code)

# Start → Generate
workflow.add_edge(START, "generate")

# Generate → Check
workflow.add_edge("generate", "check")

# Check → Correct OR Finish
workflow.add_conditional_edges(
    "check",
    decide_next_step,
    {
        "correct": "correct",
        "finish": END
    }
)

# Correct → End
workflow.add_edge("correct", END)

# Compile graph
app = workflow.compile()

# --------------------------------------------------
# Main Program
# --------------------------------------------------

print("\n")
print("=" * 55)
print("        SELF-CORRECTION CODING AGENT")
print("=" * 55)

print("\nExamples:")
print("- Reverse a Python list")
print("- Calculate factorial of a number")
print("- Read a text file in Python")
print("- Find the largest number in a list")

print("\nType 'exit' to stop.")

while True:
    question = input("\nEnter your programming question: ")

    if question.lower() == "exit":
            print("Goodbye!")
            break
    
    initial_state = {
        "question": question,
        "response": "",
        "has_error": False,
        "attempts": 0
    }

    # Run LangGraph
    final_state = app.invoke(initial_state)

    # --------------------------------------------------
    # Display Final Answer
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("                  FINAL ANSWER")
    print("=" * 55)

    print(final_state["response"])

    print("\nAttempts:", final_state["attempts"])