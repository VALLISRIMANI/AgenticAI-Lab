from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_agent(question):

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Student Assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content