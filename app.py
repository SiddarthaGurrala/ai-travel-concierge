import os
import requests
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Page configuration
st.set_page_config(page_title="AI Travel Concierge", page_icon="✈️", layout="centered")
st.title("✈️ AI Travel Concierge")
st.caption("Track A: Streamlit + LangChain AI Agent with Tool Calling")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "")
    st.markdown("---")
    st.markdown("**Tools Enabled:**\n- 🔍 DuckDuckGo Web Search\n- ☀️ Live Weather Lookup")

# Define Custom Tools
@tool
def get_current_weather(city: str) -> str:
    """Fetch real-time weather and temperature for any city using Open-Meteo API."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()
        if not geo_res.get("results"):
            return f"Could not find coordinates for '{city}'."
        
        loc = geo_res["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_res = requests.get(weather_url, timeout=5).json()
        curr = w_res.get("current_weather", {})
        temp = curr.get("temperature")
        wind = curr.get("windspeed")
        return f"Current weather in {loc['name']}, {loc.get('country', '')}: {temp}°C, Wind Speed: {wind} km/h."
    except Exception as e:
        return f"Failed to retrieve weather data: {str(e)}"

# Setup Tools and Search
search_tool = DuckDuckGoSearchRun()
tools = [search_tool, get_current_weather]

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Travel Concierge. Ask me about flight tips, weather forecasts, or destinations to explore!"}
    ]

# Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Processing
if user_prompt := st.chat_input("Where do you want to travel?"):
    if not api_key:
        st.warning("Please provide a Gemini API Key in the sidebar to proceed.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Initialize LLM & Agent
    # Initialize LLM & Agent
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        agent_executor = create_react_agent(llm, tools)

        # Build context messages
        agent_inputs = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Researching destinations and live travel details..."):
                response = agent_executor.invoke({"messages": agent_inputs})
                final_answer = response["messages"][-1].content
                st.markdown(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

    except Exception as err:
        st.error(f"Error processing your request: {err}")