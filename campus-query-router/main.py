# Campus Query Router Agent

import os

from dotenv import load_dotenv
from groq import Groq

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# --------------------------------------------------
# 2. Create Groq client
# --------------------------------------------------

client = Groq(api_key=api_key)

# --------------------------------------------------
# 3. Stub tools / functions
# --------------------------------------------------

def academic_function():
    return "Academic Tool: Here is your academic information."

def exam_function():
    return "Exam Tool: Here is your examination information."

def general_function():
    return "General Tool: Here is your general campus information."

# --------------------------------------------------
# 4. Intent classification
# --------------------------------------------------

def classify_intent(query):

    system_prompt = """
You are a campus query classification agent.

Classify the user's query into exactly ONE of these categories:

Academic
Exam
General

Definitions:

Academic:
Questions about subjects, courses, syllabus, classes,
assignments, attendance, faculty, or academic learning.

Exam:
Questions about examinations, exam timetable, marks,
results, hall tickets, internal exams, or semester exams.

General:
Questions about general campus information such as
library, transport, hostel, campus facilities, timings,
events, or anything that does not belong to Academic or Exam.

IMPORTANT:
Return ONLY ONE category name:
Academic
Exam
General
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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

    intent = response.choices[0].message.content.strip()

    return intent

# --------------------------------------------------
# 5. Route query to appropriate function
# --------------------------------------------------

def route_query(query):

    intent = classify_intent(query)

    print(f"\nIntent: {intent}")

    if intent == "Academic":

        response = academic_function()

    elif intent == "Exam":

        response = exam_function()

    elif intent == "General":

        response = general_function()

    else:

        response = (
            "Router Error: The LLM returned an invalid intent."
        )

    return response

# --------------------------------------------------
# 6. Console interaction
# --------------------------------------------------

def main():

    print("\n")
    print("=" * 50)
    print("       Campus Query Router Agent")
    print("=" * 50)

    print("\nAvailable intents:")
    print("1. Academic")
    print("2. Exam")
    print("3. General")

    print("\nType 'exit' to stop the agent.")

    while True:

        query = input("\nAsk your question:\n> ")

        if query.lower() == "exit":
            print("\nAgent stopped.")
            break

        if not query.strip():
            print("Please enter a question.")
            continue

        try:

            tool_response = route_query(query)

            print(f"Tool Response: {tool_response}")

        except Exception as e:

            print(f"Error: {e}")


# --------------------------------------------------
# 7. Program entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()