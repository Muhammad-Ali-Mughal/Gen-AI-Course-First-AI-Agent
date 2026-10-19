import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from kawnix_provider_auth import MemoryNonceStore
from kawnix_provider_auth.flask import require_kawnix_signature

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import requests

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ.get("GROQ_API_KEY"))

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    api_key = os.environ.get("WEATHERSTACK_API_KEY")

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": city
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "error" in data:
        return f"Could not get weather: {data['error']['info']}"

    location = data["location"]
    current = data["current"]

    return f"""
    City: {location['name']}
    Country: {location['country']}
    Temperature: {current['temperature']}°C
    Weather: {current['weather_descriptions'][0]}
    Humidity: {current['humidity']}%
    Wind Speed: {current['wind_speed']} km/h
    """

tools = [search_tool, get_weather]

agent = create_agent(
    model=llm,
    tools=tools
)


def pretty_print(response):

    for message in response["messages"]:
        msg_type = type(message).__name__

        if msg_type == "HumanMessage":
            print(f"\n🧑 USER: {message.content}")

        elif msg_type == "AIMessage" and message.tool_calls:
            for call in message.tool_calls:
                query = call["args"].get("query", "")
                print(f"🔍 SEARCHING: {query}")

        elif msg_type == "ToolMessage":
            print(f"📄 SEARCH RESULTS: {message.content[:200]}...")

        elif msg_type == "AIMessage" and message.content:
            print(f"\n🤖 ANSWER: {message.content}")


app = Flask(__name__)
nonces = MemoryNonceStore()


def lookup_kawnix_secret(key_id):
    expected_key_id = os.environ.get("KAWNIX_GATEWAY_KEY_ID")
    signing_secret = os.environ.get("KAWNIX_GATEWAY_SIGNING_SECRET")

    if key_id == expected_key_id:
        return signing_secret

    return None


@app.post("/chat")
@require_kawnix_signature(
    key_lookup=lookup_kawnix_secret,
    nonce_store=nonces,
)
def chat():
    body = request.get_json(force=True, silent=True) or {}

    if body.get("operation") == "health":
        return jsonify({
            "status": "ok",
            "ready": True,
        }), 200

    message = body.get("message")

    if not message:
        return jsonify({"error": "Request body must contain a 'message' field."}), 400

    response = agent.invoke({"messages": [{"role": "user", "content": message}]})
    pretty_print(response)

    return jsonify({"response": response["messages"][-1].content}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)
    # pretty_print(agent.invoke({"messages": [{"role": "user", "content": "what is the capital of Pakistan and what is the weather there?"}]}))
