import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

client = Groq(api_key=api_key)

def academic_function():
    return "Academic Tool: Here is your academic information."

def exam_function():
    return "Exam Tool: Here is your examination information."

def general_function():
    return "General Tool: Here is your general campus information."

def classify_intent(query):
    system_prompt = """
You are a campus query classification agent.
Classify the user's query into exactly ONE category: Academic, Exam, or General.

Academic: Questions about subjects, courses, syllabus, classes, assignments, attendance, faculty, or academic learning.
Exam: Questions about examinations, exam timetable, marks, results, hall tickets, internal exams, or semester exams.
General: Questions about library, transport, hostel, campus facilities, timings, events, or other general campus information.

Return ONLY ONE category name: Academic, Exam, or General.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def route_query(query):
    intent = classify_intent(query)
    print(f"\nIntent: {intent}")

    if intent == "Academic":
        return academic_function()
    elif intent == "Exam":
        return exam_function()
    elif intent == "General":
        return general_function()
    else:
        return "Router Error: The LLM returned an invalid intent."

def main():
    print("\n" + "-" * 50)
    print("       Campus Query Router Agent")
    print("-" * 50)
    print("\nAvailable intents:\n1. Academic\n2. Exam\n3. General")
    print("\nType 'exit' to stop the agent.")

    while True:
        query = input("\nAsk your question: ")

        if query.lower() == "exit":
            print("Agent stopped.")
            break

        if not query.strip():
            print("Please enter a question.")
            continue

        try:
            print(f"Tool Response: {route_query(query)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()