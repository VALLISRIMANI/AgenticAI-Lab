import os
import requests

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Load configuration from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = os.getenv("GROQ_MODEL")
WEATHER_API = os.getenv("WEATHER_API_URL")
GEOCODING_API = os.getenv("GEOCODING_API_URL")

# Extract city using Groq
def extract_city(query):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                Extract the city from the user's
                weather query.
                Return only the city name.
                """
            },
            {"role": "user", "content": query}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# Get city coordinates
def get_coordinates(city):
    response = requests.get(
        GEOCODING_API,
        params={"name": city, "count": 1},
        timeout=15
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return None

    return results[0]

# Fetch weather
def get_weather(latitude, longitude):
    response = requests.get(
        WEATHER_API,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "auto"
        },
        timeout=15
    )

    response.raise_for_status()

    return response.json()["current"]

# Generate simple explanation
def explain_weather(city, weather):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a weather assistant.
                Explain the weather in 2-3 simple sentences.
                Mention temperature, humidity and wind speed.
                Do not invent information.
                """
            },
            {
                "role": "user",
                "content": f"""
                City: {city}
                Temperature: {weather['temperature_2m']} °C
                Humidity: {weather['relative_humidity_2m']} %
                Wind Speed: {weather['wind_speed_10m']} km/h
                """
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# Main Agent
def weather_agent():
    print("\n===== WEATHER LOOKUP AGENT =====")
    print("Ask about the weather in any city.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Agent stopped.")
            break

        try:
            print("\n[Agent] Understanding query...")

            city = extract_city(query)

            print("[Agent] Fetching weather...")

            location = get_coordinates(city)

            if not location:
                print("City not found. Try another city.")
                continue

            weather = get_weather(
                location["latitude"],
                location["longitude"]
            )

            print("[Agent] Generating response...\n")

            answer = explain_weather(
                location["name"],
                weather
            )

            print("Weather Report")
            print("-------------------------")
            print("City:", location["name"])
            print(answer)
            print("-------------------------\n")

        except requests.exceptions.RequestException as e:
            print("API connection error:", e)

        except Exception as e:
            print("Agent error:", e)

if __name__ == "__main__":
    weather_agent()