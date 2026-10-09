import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)

# Agent 1: Research Agent
research_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Research Agent.
        Analyze the user's topic and generate 5-7
        relevant, concise bullet points.
        Focus on important facts and key ideas.
        Do not write a long explanation."""
    ),
    ("human", "Research this topic: {topic}")
])

research_agent = research_prompt | llm

# Agent 2: Refiner Agent
refiner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Refiner Agent.
        Convert the research bullet points into a
        clear, well-structured short explanation.
        Use simple language, logical flow, and a
        suitable heading.
        Do not introduce unsupported facts."""
    ),
    (
        "human",
        """Topic: {topic}

        Research Agent's bullet points:
        {research_points}

        Write the final explanation."""
    )
])

refiner_agent = refiner_prompt | llm


def run_pipeline(topic):
    # Step 1: Research
    research_result = research_agent.invoke({
        "topic": topic
    })

    research_points = research_result.content

    print("\n--- Research Agent Output ---")
    print(research_points)

    # Step 2: Refine the research
    refined_result = refiner_agent.invoke({
        "topic": topic,
        "research_points": research_points
    })

    print("\n--- Final Refined Answer ---")
    print(refined_result.content)


if __name__ == "__main__":
    topic = input("Enter a topic: ").strip()

    if topic:
        run_pipeline(topic)
    else:
        print("Please enter a valid topic.")