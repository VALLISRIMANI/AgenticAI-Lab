import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

research_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Research Agent.
Analyze the user's topic and generate 2-5 relevant, concise bullet points.
Focus on important facts and key ideas.
Do not write a long explanation."""),
    ("human", "Research this topic: {topic}")
])

research_agent = research_prompt | llm

refiner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Refiner Agent.
Convert the research bullet points into a clear, well-structured short explanation.
Use simple language, logical flow, and a suitable heading.
Do not introduce unsupported facts."""),
    ("human", """Topic: {topic}

Research Agent's bullet points:
{research_points}

Write the final explanation.""")
])

refiner_agent = refiner_prompt | llm

def run_pipeline(topic):
    research_result = research_agent.invoke({"topic": topic})
    research_points = research_result.content

    print("\n" + "-" * 50)
    print("RESEARCH AGENT OUTPUT")
    print("-" * 50)
    print(research_points)

    refined_result = refiner_agent.invoke({"topic": topic, "research_points": research_points})

    print("\n" + "-" * 50)
    print("FINAL REFINED ANSWER")
    print("-" * 50)
    print(refined_result.content)

def main():
    print("\n" + "-" * 50)
    print("       RESEARCH AND REFINEMENT AGENT")
    print("-" * 50)
    print("\nType 'exit' to stop.")

    while True:
        topic = input("\nEnter a topic: ").strip()

        if topic.lower() == "exit":
            print("Agent stopped.")
            break

        if not topic:
            print("Please enter a valid topic.")
            continue

        try:
            run_pipeline(topic)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()