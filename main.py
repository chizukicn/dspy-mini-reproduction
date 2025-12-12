import dspy
import dspy.streaming 
import asyncio
import mlflow
import mlflow.dspy
import random
import dotenv
import os
dotenv.load_dotenv()
mlflow.set_tracking_uri("http://localhost:5000")  # Use local MLflow server
mlflow.set_experiment("dspy-mini-reproduction")
mlflow.dspy.autolog()

LLM_MODEL = os.getenv("LLM_OPENAI_MODEL")  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


lm = dspy.LM(model=f"openai/{LLM_MODEL}", api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
dspy.configure(lm=lm)


first_time = True

def search_weather(city: str):
    global first_time
    if first_time:
        raise RuntimeError("First time")
    first_time = False
    
    return {
        "city": city,
        "weather": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
        "temperature": random.randint(0, 40),
        "humidity": random.randint(0, 100),
        "pressure": random.randint(900, 1100),
        "wind_speed": random.randint(0, 100),
        "wind_direction": random.choice(["N", "S", "E", "W"]),
        "wind_gust": random.randint(0, 100),
        "wind_gust_direction": random.choice(["N", "S", "E", "W"]),
        "wind_gust_speed": random.randint(0, 100),
    }

react = dspy.ReAct("question -> answer", tools=[search_weather])


stream_react = dspy.streaming.streamify(react,stream_listeners=[
    dspy.streaming.StreamListener("answer")
])


async def main():
    city = random.choice(["Tokyo", "London", "Paris", "Berlin", "Rome", "Madrid", "Berlin", "Rome", "Madrid", "Berlin", "Rome", "Madrid"])
    pred = stream_react(question=f"What is the weather in {city}?")
    async for chunk in pred:
        print(chunk)

asyncio.run(main())

