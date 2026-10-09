# Student Helper Agent

import os
import json
from groq import Groq
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

# --------------------------------------------------
# 2. Define tools
# --------------------------------------------------

def math_tool(query):
    """
    Handles mathematical questions.
    """

    prompt = f"""
You are a helpful mathematics tutor.

Solve or explain the following mathematical question
step by step.

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def python_help(query):
    """
    Handles Python syntax and programming questions.
    """

    prompt = f"""
You are a Python programming tutor.

Explain the following Python question clearly.
If code is required, provide a simple example.

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def english_grammar(query):
    """
    Handles English grammar questions.
    """

    prompt = f"""
You are an English grammar tutor.

Correct or explain the following English sentence or
grammar question.

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# --------------------------------------------------
# 3. Tool registry
# --------------------------------------------------

tools = {
    "math": math_tool,
    "python_help": python_help,
    "english_grammar": english_grammar
}

# --------------------------------------------------
# 4. LLM-based router
# --------------------------------------------------

def classify_query(query):

    system_prompt = """
You are a Student Helper Agent.

Your task is to determine which tool should handle
the user's query.

Available tools:

1. math
   - Mathematics questions
   - Arithmetic
   - Algebra
   - Calculus
   - Mathematical explanations

2. python_help
   - Python programming
   - Python syntax
   - Python errors
   - Python concepts

3. english_grammar
   - English grammar
   - Sentence correction
   - Grammar explanations
   - English usage

Return ONLY valid JSON.

The JSON must have exactly this format:

{
    "tool": "math"
}

or

{
    "tool": "python_help"
}

or

{
    "tool": "english_grammar"
}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    return json.loads(content)

# --------------------------------------------------
# 5. Agent
# --------------------------------------------------

def student_helper_agent(query):

    print("\n[Agent] Understanding your request...")

    try:

        # Step 1: Ask LLM which tool to use
        decision = classify_query(query)

        selected_tool = decision.get("tool")

        print(f"[Agent] Selected tool: {selected_tool}")

        # Step 2: Validate tool
        if selected_tool not in tools:
            return "Error: Invalid tool selected by the LLM."

        # Step 3: Call selected tool
        tool_function = tools[selected_tool]

        print("[Agent] Calling tool...")

        result = tool_function(query)

        return result

    except json.JSONDecodeError:
        return "Error: The LLM returned an invalid routing response."

    except Exception as e:
        return f"Error while executing the agent: {e}"

# --------------------------------------------------
# 6. Command-line interface
# --------------------------------------------------

def main():

    print("\n")
    print("=" * 50)
    print("       STUDENT HELPER AGENT")
    print("=" * 50)

    print("\nAvailable areas:")
    print("- Mathematics")
    print("- Python")
    print("- English Grammar")

    print("\nType 'exit' to stop.")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            print("Agent: Goodbye!")
            break

        if not query:
            print("Agent: Please enter a question.")
            continue

        result = student_helper_agent(query)

        print("\nAgent:")
        print(result)

if __name__ == "__main__":
    main()