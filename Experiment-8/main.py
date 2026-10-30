from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent
import time

app = FastAPI(title="Student Helper Agent API")

class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_agent(request: QueryRequest):

    start_time = time.time()

    print("\nNew Request Received")
    print("Question:", request.question)

    try:
        answer = run_agent(request.question)

        latency = round(time.time() - start_time, 3)

        print("Response:", answer)
        print("Latency:", latency, "seconds")

        return {
            "status": "success",
            "question": request.question,
            "answer": answer,
            "latency": latency
        }

    except Exception as e:

        print("Error:", str(e))

        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/")
def home():
    return {"message": "Student Helper Agent API is running"}