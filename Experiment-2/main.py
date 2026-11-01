import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

def math_tool(query):
    prompt = f"""
You are a helpful mathematics tutor.
Solve or explain the following mathematical question step by step.
Question: {query}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def python_help(query):
    prompt = f"""
You are a Python programming tutor.
Explain the following Python question clearly.
If code is required, provide a simple example.
Question: {query}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def english_grammar(query):
    prompt = f"""
You are an English grammar tutor.
Correct or explain the following English sentence or grammar question.
Question: {query}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

tools = {
    "math": math_tool,
    "python_help": python_help,
    "english_grammar": english_grammar
}

def classify_query(query):
    system_prompt = """
You are a Student Helper Agent.
Determine which tool should handle the user's query.

Available tools:
1. math - Mathematics, arithmetic, algebra, calculus.
2. python_help - Python programming, syntax, errors, concepts.
3. english_grammar - English grammar, sentence correction, English usage.

Return ONLY valid JSON in one of these formats:
{"tool": "math"}
{"tool": "python_help"}
{"tool": "english_grammar"}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

def student_helper_agent(query):
    print("\n[Agent] Understanding your request...")
    try:
        decision = classify_query(query)
        selected_tool = decision.get("tool")
        print(f"[Agent] Selected tool: {selected_tool}")

        if selected_tool not in tools:
            return "Error: Invalid tool selected by the LLM."

        print("[Agent] Calling tool...")
        return tools[selected_tool](query)

    except json.JSONDecodeError:
        return "Error: The LLM returned an invalid routing response."
    except Exception as e:
        return f"Error while executing the agent: {e}"

def main():
    print("\n" + "-" * 50)
    print("       STUDENT HELPER AGENT")
    print("-" * 50)
    print("\nAvailable areas:")
    print("- Mathematics\n- Python\n- English Grammar")
    print("\nType 'exit' to stop.")

    while True:
        query = input("\nAsk your question: ").strip()

        if query.lower() == "exit":
            print("Agent stopped.")
            break

        if not query:
            print("Agent: Please enter a question.")
            continue

        result = student_helper_agent(query)
        print("\nAgent:")
        print(result)

if __name__ == "__main__":
    main()